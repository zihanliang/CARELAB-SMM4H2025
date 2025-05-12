import os
import sys
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModel
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from dataclasses import field

# Monkey patch: change __init__ and __post_init__ in TrainingArguments.
try:
    _orig_trainargs_init = TrainingArguments.__init__
    def _patched_trainargs_init(self, *args, **kwargs):
        self._custom_evaluation_strategy = kwargs.pop("evaluation_strategy", "epoch")
        self._custom_save_strategy = kwargs.pop("save_strategy", "epoch")
        if "deepspeed_plugin" in kwargs:
            kwargs.pop("deepspeed_plugin")
        return _orig_trainargs_init(self, *args, **kwargs)
    TrainingArguments.__init__ = _patched_trainargs_init

    def force_trainargs_post_init(self):
        self.evaluation_strategy = IntervalStrategy.EPOCH
        self.save_strategy = SaveStrategy.EPOCH
        if not hasattr(self, "distributed_state"):
            self.distributed_state = None
        if not hasattr(self, "deepspeed_plugin"):
            self.deepspeed_plugin = None
        if not hasattr(self, "fsdp_config") or self.fsdp_config is None:
            self.fsdp_config = {"xla": False}
    TrainingArguments.__post_init__ = force_trainargs_post_init

    print("TrainingArguments monkey patch applied successfully.")
except Exception as e:
    print("Error while monkey patching TrainingArguments:", e)

# Monkey patch: Trainer's accelerator_config
try:
    _orig_trainer_create_accel = Trainer.create_accelerator_and_postprocess

    class DummyAccelConfig:
        def __init__(self):
            self.split_batches = False
            self.dispatch_batches = None
            self.even_batches = True
            self.use_seedable_sampler = True
            self.gradient_accumulation_kwargs = {}
            self.non_blocking = False
        def to_dict(self):
            return {
                "split_batches": self.split_batches,
                "dispatch_batches": self.dispatch_batches,
                "even_batches": self.even_batches,
                "use_seedable_sampler": self.use_seedable_sampler,
                "gradient_accumulation_kwargs": self.gradient_accumulation_kwargs,
                "non_blocking": self.non_blocking,
            }

    def _patched_trainer_create_accel(self):
        if self.args.accelerator_config is None:
            self.args.accelerator_config = DummyAccelConfig()
        return _orig_trainer_create_accel(self)
    Trainer.create_accelerator_and_postprocess = _patched_trainer_create_accel

    print("Trainer.create_accelerator_and_postprocess monkey patch applied successfully.")
except Exception as e:
    print("Error while monkey patching Trainer:", e)

# Monkey patch: Accelerator.__init__
try:
    _orig_accelerator_init = Accelerator.__init__
    def _patched_accelerator_init(self, *args, **kwargs):
        for key in ["dispatch_batches", "even_batches", "use_seedable_sampler"]:
            kwargs.pop(key, None)
        _orig_accelerator_init(self, *args, **kwargs)
        if not hasattr(self.state, "distributed_type"):
            from accelerate.state import DistributedType
            self.state.distributed_type = DistributedType.NO
    Accelerator.__init__ = _patched_accelerator_init
    print("Accelerator monkey patch applied successfully.")
except Exception as e:
    print("Error while monkey patching Accelerator:", e)

# Monkey patch: AcceleratorState._reset_state
try:
    from accelerate.state import AcceleratorState, DistributedType
    @classmethod
    def _patched_reset_state(cls, *args, **kwargs):
        cls.distributed_type = DistributedType.NO
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AcceleratorState._reset_state = _patched_reset_state
    print("AcceleratorState._reset_state patch applied successfully.")
except Exception as e:
    print("Error while patching AcceleratorState._reset_state:", e)

import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from torchcrf import CRF

labels = [
    "O", 
    "B-TARGET_ORG", "I-TARGET_ORG", 
    "B-PRODUCT", "I-PRODUCT", 
    "B-INFECTION", "I-INFECTION", 
    "B-SAFETY_INCIDENT", "I-SAFETY_INCIDENT", 
    "B-AFFECTED_NUM", "I-AFFECTED_NUM",
    "B-LOCATION", "I-LOCATION"
]
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

column_to_entity = {
    "organization": "TARGET_ORG",
    "product": "PRODUCT",
    "disease": "INFECTION",
    "cause": "SAFETY_INCIDENT",
    "number_of_people_affected": "AFFECTED_NUM",
    "location": "LOCATION"
}

def load_data(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df

def preprocess_dataframe(df):
    return df

def find_all_occurrences(text, substring):
    """Returns a list of the starting positions of all substrings in the text."""
    starts = []
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx == -1:
            break
        starts.append(idx)
        start = idx + 1
    return starts

def get_entities_from_example(example):
    """
    Returns a list of entities like {“start”: start position, “end”: end position,
    “label”: entity type}, based on the contents of each (non-empty) column in the
    CSV and a predefined column-to-entity mapping relationship.
    """
    entities = []
    text = example["text"]
    for col, ent_type in column_to_entity.items():
        if col in example and pd.notna(example[col]):
            ent_text = str(example[col]).strip()
            if ent_text:
                start_positions = find_all_occurrences(text, ent_text)
                for st in start_positions:
                    entities.append({
                        "start": st,
                        "end": st + len(ent_text),
                        "label": ent_type
                    })
    return entities

###################################
# Segmentation and Tag Alignment
###################################
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, return_offsets_mapping=True, max_length=512)
    all_labels = []
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        text = examples["text"][i]
        example = { key: examples[key][i] for key in examples.keys() }
        entities = get_entities_from_example(example)
        char_labels = ["O"] * len(text)
        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            ent_label = ent["label"]
            if 0 <= start < len(text):
                char_labels[start] = "B-" + ent_label
                for j in range(start + 1, end):
                    if j < len(text):
                        char_labels[j] = "I-" + ent_label
        label_ids = []
        for offset in offsets:
            if offset is None or offset[0] == offset[1]:
                label_ids.append(-100)
            else:
                label = char_labels[offset[0]]
                label_ids.append(label2id.get(label, label2id["O"]))
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

###################################
# Sequence labeling helper functions
###################################
def extract_event(predicted_labels, tokens):
    """
    Stitch the token sequence into entities according to the BIO rules
    and return a dictionary like {entity type: [entity text, ...]} Dictionary
    """
    event = defaultdict(list)
    current_entity = None
    current_tokens = []
    for label, token in zip(predicted_labels, tokens):
        if label.startswith("B-"):
            if current_entity is not None:
                event[current_entity].append(" ".join(current_tokens))
            current_entity = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_entity is not None:
            current_tokens.append(token)
        else:
            if current_entity is not None:
                event[current_entity].append(" ".join(current_tokens))
            current_entity = None
            current_tokens = []
    if current_entity is not None:
        event[current_entity].append(" ".join(current_tokens))
    return dict(event)

def convert_gold_to_event(example):
    """
    Convert real annotations to {entity type: [entity text, ...]} Format
    """
    text = example["text"]
    entities = get_entities_from_example(example)
    event = defaultdict(list)
    for ent in entities:
        label = ent["label"]
        start, end = ent["start"], ent["end"]
        ent_text = text[start:end]
        event[label].append(ent_text)
    return dict(event)

def events_exact_match(pred_event, gold_event):
    """
    If the predicted event is identical to the real event in terms of entity
    type and set of entity texts, then return True
    """
    if set(pred_event.keys()) != set(gold_event.keys()):
        return False
    for key in pred_event:
        if set(pred_event[key]) != set(gold_event[key]):
            return False
    return True

###################################
# Customizing Advanced TokenClassification Models
###################################
from transformers.modeling_outputs import TokenClassifierOutput

class AdvancedTokenClassificationModel(AutoModelForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.class_weights = torch.ones(config.num_labels)
        self.loss_type = "crf"
        self.gamma = 2.0
        self.label_smoothing = 0.0
        if self.loss_type == "crf":
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        else:
            self.crf = None
        self.config.gradient_checkpointing = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        class_weights = kwargs.pop("class_weights", None)
        loss_type = kwargs.pop("loss_type", "crf")
        gamma = kwargs.pop("gamma", 2.0)
        label_smoothing = kwargs.pop("label_smoothing", 0.0)
        model = super(AdvancedTokenClassificationModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.class_weights = class_weights if class_weights is not None else torch.ones(model.config.num_labels)
        model.loss_type = loss_type
        model.gamma = gamma
        model.label_smoothing = label_smoothing
        if loss_type == "crf":
            model.crf = CRF(num_tags=model.config.num_labels, batch_first=True)
        else:
            model.crf = None
        return model

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = outputs.logits
        if labels is not None:
            if self.loss_type == "crf":
                mask = labels.ne(-100)
                tags = labels.clone()
                tags[~mask] = 0
                crf_loss = - self.crf(logits, tags, mask=mask, reduction='mean')
                return TokenClassifierOutput(loss=crf_loss, logits=logits)
            else:
                loss = self.compute_loss(TokenClassifierOutput(logits=logits), labels)
                return TokenClassifierOutput(loss=loss, logits=logits)
        else:
            if self.loss_type == "crf":
                mask = attention_mask.bool()
                decoded_tags = self.crf.decode(logits, mask=mask)
                return TokenClassifierOutput(logits=logits, hidden_states=decoded_tags)
            else:
                return TokenClassifierOutput(logits=logits)

    def compute_loss(self, model_outputs, labels):
        logits = model_outputs.logits  # (batch_size, seq_len, num_labels)
        logits = logits.view(-1, self.num_labels)
        labels = labels.view(-1)
        valid_mask = labels.ne(-100)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        logits = logits[valid_mask]
        labels = labels[valid_mask]
        if self.loss_type == "focal":
            ce_loss = F.cross_entropy(logits, labels, reduction='none', weight=self.class_weights.to(logits.device))
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()
        elif self.loss_type == "ce":
            if self.label_smoothing > 0:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device),
                                                       ignore_index=-100,
                                                       label_smoothing=self.label_smoothing)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device),
                                                       ignore_index=-100)
            return loss_fct(logits, labels)
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device),
                                                   ignore_index=-100)
            return loss_fct(logits, labels)

###################################
# Customized Adversarial Trainer (based on FGM adversarial training)
###################################
class AdversarialTrainer(Trainer):
    def training_step(self, model, inputs, num_items):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()
        epsilon = 0.5
        if hasattr(model, "roberta"):
            embed_layer = model.roberta.embeddings.word_embeddings
        else:
            embed_layer = model.base_model.embeddings.word_embeddings
        delta = epsilon * torch.sign(embed_layer.weight.grad)
        embed_layer.weight.data.add_(delta)
        adv_loss = self.compute_loss(model, inputs)
        adv_loss.backward()
        embed_layer.weight.data.sub_(delta)
        total_loss = loss + adv_loss
        return total_loss.detach()

###################################
# Auxiliary function: calculating category weights (based on token level statistics)
###################################
def compute_class_weights(dataset, num_labels):
    counter = Counter()
    for example in dataset:
        for label in example["labels"]:
            if label != -100:
                counter[label] += 1
    weights = []
    for i in range(num_labels):
        count = counter[i]
        weights.append(1.0 / count if count > 0 else 1.0)
    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum()
    return weights

###################################
# Modify compute_metrics: support output after CRF decoding
###################################
def compute_metrics(p):
    predictions, labels_true = p
    if isinstance(predictions, list):
        true_predictions = predictions
    else:
        predictions = np.argmax(predictions, axis=2)
        true_predictions = []
        for pred, label in zip(predictions, labels_true):
            pred_labels = []
            label_list = []
            for p_idx, l_idx in zip(pred, label):
                if l_idx != -100:
                    pred_labels.append(id2label[p_idx])
                    label_list.append(id2label[l_idx])
            true_predictions.append(pred_labels)
    true_labels = []
    for label in labels_true:
        label_list = []
        for l in label:
            if l != -100:
                label_list.append(id2label[l])
        true_labels.append(label_list)
    seqeval_metric = evaluate.load("seqeval")
    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
         "precision": results["overall_precision"],
         "recall": results["overall_recall"],
         "f1": results["overall_f1"],
         "accuracy": results["overall_accuracy"],
    }

###################################
# Mainstream processes: training, assessment and advanced analysis (confusion matrices, classification reports, etc.)
###################################
if __name__ == "__main__":
    train_path = "SMM4H-2025-Task5-Train_subtask2.csv"
    val_path = "SMM4H-2025-Task5-Validation_subtask2.csv"
    
    train_df, val_df = load_data(train_path, val_path)
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
###################################
# 9. Initialize Tokenizer and Model (using roberta-large)
###################################
    model_name = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True, batch_size=8, remove_columns=list(train_df.columns))
    val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True, batch_size=8, remove_columns=list(val_df.columns))
    
    class_weights = compute_class_weights(train_dataset, num_labels=len(labels))
    
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        gradient_checkpointing=True
    )
    model = AdvancedTokenClassificationModel.from_pretrained(
        model_name,
        config=config,
        class_weights=class_weights,
        loss_type="crf",
        gamma=2.0,
        label_smoothing=0.0
    )
    
###################################
# Defining Training Parameters and Customizing AdversarialTrainer
###################################
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    torch.cuda.empty_cache()
    
    trainer = AdversarialTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
###################################
# Start training and evaluation (Token-Level and Event-Level)
###################################
    trainer.train()
    
    metrics = trainer.evaluate()
    print("Token-Level Evaluation Metrics:")
    print(metrics)
    
    exact_match_count = 0
    total_examples = len(val_df)
    device = model.device
    for i in tqdm(range(total_examples), desc="Evaluating event extraction"):
        example = val_df.iloc[i]
        text = example["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: t.to(device) for k, t in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        if model.loss_type == "crf":
            pred_ids = outputs.hidden_states[0]
        else:
            pred_ids = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy().tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        filtered_tokens = []
        filtered_preds = []
        for token, p in zip(tokens, pred_ids):
            if token in tokenizer.all_special_tokens:
                continue
            if isinstance(p, int):
                filtered_preds.append(id2label[p])
            else:
                filtered_preds.append(id2label[p])
            filtered_tokens.append(token)
        pred_event = extract_event(filtered_preds, filtered_tokens)
        gold_event = convert_gold_to_event(example)
        if events_exact_match(pred_event, gold_event):
            exact_match_count += 1
    event_exact_match_rate = exact_match_count / total_examples
    print(f"Event-Level Exact Match Rate: {event_exact_match_rate:.4f}")
    
###################################
# Advanced Analytics: Confusion Matrix vs. Categorical Reporting (Filter out “O” tags)
###################################
    flat_preds = []
    flat_labels = []
    for batch in trainer.get_eval_dataloader():
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        if model.loss_type == "crf":
            batch_preds = outputs.hidden_states[0]
        else:
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        labels_batch = batch["labels"].cpu().numpy()
        if isinstance(batch_preds, list):
            for pred_seq, l_seq in zip(batch_preds, labels_batch):
                for p, l in zip(pred_seq, l_seq):
                    if l != -100:
                        flat_preds.append(p)
                        flat_labels.append(l)
        else:
            for p_seq, l_seq in zip(batch_preds, labels_batch):
                for p, l in zip(p_seq, l_seq):
                    if l != -100:
                        flat_preds.append(p)
                        flat_labels.append(l)
    filtered_preds, filtered_labels = [], []
    o_id = label2id["O"]
    for p, l in zip(flat_preds, flat_labels):
        if p != o_id and l != o_id:
            filtered_preds.append(p)
            filtered_labels.append(l)
    ALL_LABELS = [
        "O", 
        "B-TARGET_ORG", "I-TARGET_ORG", 
        "B-PRODUCT", "I-PRODUCT", 
        "B-INFECTION", "I-INFECTION", 
        "B-SAFETY_INCIDENT", "I-SAFETY_INCIDENT", 
        "B-AFFECTED_NUM", "I-AFFECTED_NUM",
        "B-LOCATION", "I-LOCATION"
    ]
    entity_only = [lab for lab in ALL_LABELS if lab != "O"]
    entity_ids = [label2id[lab] for lab in entity_only]
    
    cm_entities = confusion_matrix(filtered_labels, filtered_preds, labels=entity_ids)
    print("\nConfusion Matrix (Gold, Pred ≠ O):")
    print(cm_entities)
    
    print("\nClassification Report (Gold, Pred ≠ O):")
    print(classification_report(
        filtered_labels,
        filtered_preds,
        labels=entity_ids,
        target_names=entity_only,
        zero_division=0
    ))
    
###################################
# Entity-Level (Chunk-Level) F1 Evaluation (based on seqeval)
###################################
    preds = []
    labels_true = []
    for batch in trainer.get_eval_dataloader():
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        if model.loss_type == "crf":
            batch_preds = outputs.hidden_states[0]
        else:
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        labels_batch = batch["labels"].cpu().numpy()
        preds.extend(batch_preds)
        labels_true.extend(labels_batch)
    
    label_seqeval_preds = []
    label_seqeval_trues = []
    if isinstance(preds, list):
        for pred_seq, label_seq in zip(preds, labels_true):
            pred_labels_str = []
            gold_labels_str = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                pred_labels_str.append(id2label[p])
                gold_labels_str.append(id2label[l])
            label_seqeval_preds.append(pred_labels_str)
            label_seqeval_trues.append(gold_labels_str)
    else:
        for pred_seq, label_seq in zip(preds, labels_true):
            pred_labels_str = []
            gold_labels_str = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                pred_labels_str.append(id2label[p])
                gold_labels_str.append(id2label[l])
            label_seqeval_preds.append(pred_labels_str)
            label_seqeval_trues.append(gold_labels_str)
    seqeval_metric = evaluate.load("seqeval")
    chunk_results = seqeval_metric.compute(
        predictions=label_seqeval_preds,
        references=label_seqeval_trues
    )
    print("\nChunk-Level (Entity-Level) Seqeval Results:")
    print(chunk_results)