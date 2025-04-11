import os
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm
import evaluate
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoModelForTokenClassification
from datasets import Dataset

# Define labels and mappings
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

# The following remains consistent with training
column_to_entity = {
    "organization": "TARGET_ORG",
    "product": "PRODUCT",
    "disease": "INFECTION",
    "cause": "SAFETY_INCIDENT",
    "number_of_people_affected": "AFFECTED_NUM",
    "location": "LOCATION"
}

def load_data(val_path):
    val_df = pd.read_csv(val_path)
    return val_df

def preprocess_dataframe(df):
    return df

def find_all_occurrences(text, substring):
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
    if set(pred_event.keys()) != set(gold_event.keys()):
        return False
    for key in pred_event:
        if set(pred_event[key]) != set(gold_event[key]):
            return False
    return True

# Load the tokenizer and model
model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the saved best model
from TorchCRF import CRF  # Ensure TorchCRF is installed correctly

class AdvancedTokenClassificationModelWrapper(torch.nn.Module):
    """
    A wrapper for AdvancedTokenClassificationModel to facilitate loading and calling the CRF layer.
    Remains consistent with the training process.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, **kwargs):
        return self.model(**kwargs)
    @property
    def loss_type(self):
        return self.model.loss_type
    @property
    def crf(self):
        return self.model.crf
    @property
    def device(self):
        return next(self.model.parameters()).device

# Assume the model was saved in the "./results" directory during training
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)
# Load the model using from_pretrained (consistent with the AdvancedTokenClassificationModel used during training)
model = AutoModelForTokenClassification.from_pretrained("./results", config=config)
# If the internal model type is AdvancedTokenClassificationModel, then loss_type and crf are already set.
model_wrapper = AdvancedTokenClassificationModelWrapper(model)
model_wrapper.to("cuda" if torch.cuda.is_available() else "cpu")

# Load validation data, and perform tokenization and label alignment
val_path = "SMM4H-2025-Task5-Validation_subtask2.csv"
val_df = load_data(val_path)
val_df = preprocess_dataframe(val_df)
val_dataset = Dataset.from_pandas(val_df)

def tokenize_and_align_labels_eval(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, return_offsets_mapping=True, max_length=512)
    # No need to align labels since this is only for prediction
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

val_dataset = val_dataset.map(tokenize_and_align_labels_eval, batched=True, batch_size=8, remove_columns=list(val_df.columns))
data_collator = DataCollatorForTokenClassification(tokenizer)

# Begin Event-Level evaluation
exact_match_count = 0
total_examples = len(val_df)
device = model_wrapper.device

for i in tqdm(range(total_examples), desc="Evaluating event extraction"):
    example = val_df.iloc[i]
    text = example["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_wrapper(**inputs)
    # For the CRF branch, check whether hidden_states is None
    if model_wrapper.loss_type == "crf":
        if outputs.hidden_states is not None:
            pred_ids = outputs.hidden_states[0]
        else:
            mask = inputs['attention_mask'].bool()
            pred_ids = model_wrapper.crf.viterbi_decode(outputs.logits, mask)
            # pred_ids may be a list-of-lists
            if isinstance(pred_ids, list):
                pred_ids = pred_ids[0]
            elif torch.is_tensor(pred_ids):
                pred_ids = pred_ids[0].cpu().numpy().tolist()
            else:
                pred_ids = list(pred_ids)[0]
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

# Token-Level evaluation (using Trainer to evaluate the entire validation set)
training_args = TrainingArguments(output_dir="./results_eval")
trainer = Trainer(
    model=model_wrapper.model,  # Use the internal model for evaluation
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

metrics = trainer.evaluate(val_dataset)
print("Token-Level Evaluation Metrics:")
print(metrics)

# Calculate the confusion matrix and classification report (filtering out the "O" label)
flat_preds = []
flat_labels = []

for batch in trainer.get_eval_dataloader():
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model_wrapper(**batch)
    if model_wrapper.loss_type == "crf":
        if outputs.hidden_states is not None:
            batch_preds = outputs.hidden_states[0]
        else:
            mask = batch['attention_mask'].bool()
            batch_preds = model_wrapper.crf.viterbi_decode(outputs.logits, mask)
    else:
        batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    labels_batch = batch["labels"].cpu().numpy() if "labels" in batch else None
