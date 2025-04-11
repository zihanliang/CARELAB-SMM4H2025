# -------------------------------------------------------------------
# Monkey patch code: Solve issues related to TrainingArguments, Trainer, and Accelerator in the environment
# -------------------------------------------------------------------
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

# Import required modules for transformers, accelerate, and sklearn
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from dataclasses import field

# Monkey patch: Modify __init__ and __post_init__ of TrainingArguments
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

# Monkey patch: accelerator_config of Trainer
try:
    _orig_trainer_create_accel = Trainer.create_accelerator_and_postprocess

    class DummyAccelConfig:
        def __init__(self):
            self.split_batches = False
            self.dispatch_batches = None
            self.even_batches = True
            self.use_seedable_sampler = True
            self.gradient_accumulation_kwargs = {}  # Must exist
            self.non_blocking = False               # New attribute
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

# Monkey patch: AcceleratorState._reset_state (using classmethod)
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

# -------------------------------------------------------------------
# Define labels, data loading, and utility functions
# -------------------------------------------------------------------
from collections import defaultdict, Counter
from tqdm import tqdm
import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Requires torchcrf: pip install pytorch-crf
from TorchCRF import CRF

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
    # Additional data cleaning if necessary; here we simply return the dataframe
    return df

def find_all_occurrences(text, substring):
    """Return a list of starting positions for all occurrences of substring in text"""
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
    For each non-empty column in the CSV and based on the predefined column-to-entity mapping,
    return a list of entities as {"start": start position, "end": end position, "label": entity type}
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
# Tokenization and label alignment
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
    Concatenate token sequences into entities according to BIO rules,
    returning a dictionary {entity type: [entity text, ...]}.
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
    Convert the gold-standard annotations into {entity type: [entity text, ...]} format.
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
    Return True if the predicted event and gold event are exactly matched in both entity types and text sets.
    """
    if set(pred_event.keys()) != set(gold_event.keys()):
        return False
    for key in pred_event:
        if set(pred_event[key]) != set(gold_event[key]):
            return False
    return True

###################################
# Custom Advanced TokenClassification model
# Supports CRF layer and multiple loss types ("crf", "focal", "ce")
###################################
from transformers.modeling_outputs import TokenClassifierOutput

class AdvancedTokenClassificationModel(AutoModelForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        # Default parameters; can be overridden during from_pretrained
        self.class_weights = torch.ones(config.num_labels)
        self.loss_type = "crf"  # Default using CRF; can be set to "focal" or "ce"
        self.gamma = 2.0
        self.label_smoothing = 0.0
        if self.loss_type == "crf":
            self.crf = CRF(config.num_labels)
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
            model.crf = CRF(model.config.num_labels)
        else:
            model.crf = None
        return model

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = outputs.logits  # Shape: (batch_size, seq_len, num_labels)
        if labels is not None:
            if self.loss_type == "crf":
                mask = labels.ne(-100)
                tags = labels.clone()
                tags[~mask] = 0  # Set ignore_index portions to 0
                crf_loss = - self.crf(logits, tags, mask=mask).mean()
                return TokenClassifierOutput(loss=crf_loss, logits=logits)
            else:
                loss = self.compute_loss(TokenClassifierOutput(logits=logits), labels)
                return TokenClassifierOutput(loss=loss, logits=logits)
        else:
            if self.loss_type == "crf":
                mask = attention_mask.bool()
                decoded_tags = self.crf.viterbi_decode(logits, mask)
                return TokenClassifierOutput(logits=logits, hidden_states=decoded_tags)
            else:
                return TokenClassifierOutput(logits=logits)

    def compute_loss(self, model_outputs, labels):
        logits = model_outputs.logits  # (batch_size, seq_len, num_labels)
        logits = logits.view(-1, self.config.num_labels)
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
# Custom Adversarial Trainer (based on FGM adversarial training)
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
# Helper: Compute class weights based on token-level distribution
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
    weights = weights / weights.sum()  # Normalize
    return weights

###################################
# Metrics calculation for evaluation (using seqeval)
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
            for p_idx, l_idx in zip(pred, label):
                if l_idx != -100:
                    pred_labels.append(id2label[p_idx])
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
# Main training process
###################################
from datasets import Dataset
from transformers import DataCollatorForTokenClassification

if __name__ == "__main__":
    # File paths for training and validation CSVs
    train_path = "SMM4H-2025-Task5-Train_subtask2.csv"
    val_path = "SMM4H-2025-Task5-Validation_subtask2.csv"
    
    # Load and preprocess data
    train_df, val_df = load_data(train_path, val_path)
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    ###################################
    # Initialize tokenizer and model (using roberta-large)
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
    # Training parameters and trainer setup
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
    
    # Clear GPU cache before training
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
    
    # Begin training
    trainer.train()
    
    # Optionally conduct evaluation: Print token-level evaluation metrics
    metrics = trainer.evaluate()
    print("Token-Level Evaluation Metrics:")
    print(metrics)
