import json
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
import evaluate

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


###################################
# 1. Define labels, data loading, and utility functions #
###################################

# All entity labels (BIO) + O
# Added LOCATION entity type that was missing
labels = [
    "O", 
    "B-TARGET_ORG", "I-TARGET_ORG", 
    "B-PRODUCT", "I-PRODUCT", 
    "B-INFECTION", "I-INFECTION", 
    "B-SAFETY_INCIDENT", "I-SAFETY_INCIDENT", 
    "B-AFFECTED_NUM", "I-AFFECTED_NUM",
    "B-LOCATION", "I-LOCATION"  # Added LOCATION entity
]
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Map CSV columns to entity types
# Added location to the column mapping
column_to_entity = {
    "organization": "TARGET_ORG",
    "product": "PRODUCT",
    "disease": "INFECTION",
    "cause": "SAFETY_INCIDENT",
    "number_of_people_affected": "AFFECTED_NUM",
    "location": "LOCATION"  # Added missing location mapping
}


def load_data(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df

def preprocess_dataframe(df):
    # Perform additional cleaning if needed
    return df

def find_all_occurrences(text, substring):
    """
    Return a list of starting positions of all occurrences of the substring in text.
    If not found, return an empty list.
    """
    starts = []
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx == -1:
            break
        starts.append(idx)
        start = idx + 1  # Continue searching from the next character
    return starts


def get_entities_from_example(example):
    """
    Iterate over the columns in the mapping; if a column has a value, search for all occurrences of the substring in text,
    and generate annotations in the form {"start": start, "end": end, "label": ent_type}.
    """
    entities = []
    text = example["text"]
    for col, ent_type in column_to_entity.items():
        if col in example and pd.notna(example[col]):
            ent_text = str(example[col]).strip()
            if ent_text:
                # Find all occurrences
                start_positions = find_all_occurrences(text, ent_text)
                for st in start_positions:
                    entities.append({
                        "start": st,
                        "end": st + len(ent_text),
                        "label": ent_type
                    })
    return entities


###########################
# 2. Tokenization & Label Alignment  #
###########################

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, return_offsets_mapping=True)
    all_labels = []
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        text = examples["text"][i]
        # Get all entities for the sample from the CSV
        example = { key: examples[key][i] for key in examples.keys() }
        entities = get_entities_from_example(example)
        # Initialize all character labels as "O"
        char_labels = ["O"] * len(text)

        # Based on entity positions, mark corresponding characters with B- or I-
        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            ent_label = ent["label"]  # e.g., "PRODUCT", "INFECTION", etc.
            if 0 <= start < len(text):
                char_labels[start] = "B-" + ent_label
                for j in range(start + 1, end):
                    if j < len(text):
                        char_labels[j] = "I-" + ent_label

        # Align char-level labels to token-level
        label_ids = []
        for offset in offsets:
            # For special tokens like [CLS] and [SEP], the offset may be empty or equal on both ends
            if offset is None or offset[0] == offset[1]:
                label_ids.append(-100)
            else:
                label = char_labels[offset[0]]
                label_ids.append(label2id.get(label, label2id["O"]))
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs


###########################
# 3. Helper Functions for Sequence Labeling   #
###########################

def extract_event(predicted_labels, tokens):
    """
    Concatenate tokens into entity text and categorize based on BIO labels.
    predicted_labels: e.g., ["B-PRODUCT", "I-PRODUCT", "O", ...]
    tokens: corresponding to the length of predicted_labels
    """
    event = defaultdict(list)
    current_entity = None
    current_tokens = []
    for label, token in zip(predicted_labels, tokens):
        if label.startswith("B-"):
            # If there is an ongoing accumulated entity, collect it first
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
    # If there is an entity not collected at the end
    if current_entity is not None:
        event[current_entity].append(" ".join(current_tokens))

    return dict(event)


def convert_gold_to_event(example):
    """
    Convert gold information into a dict (entity type -> list of entity text).
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
    Determine if the predicted event exactly matches the gold event:
      - The set of entity types is identical
      - The set of texts under each entity type is identical
    """
    if set(pred_event.keys()) != set(gold_event.keys()):
        return False
    for key in pred_event:
        if set(pred_event[key]) != set(gold_event[key]):
            return False
    return True


def compute_metrics(p):
    """
    Token-level (BIO sequence labeling) evaluation, calling seqeval to compute P/R/F1/Accuracy.
    Note: The metrics here are token-level, and while seqeval can also perform chunk-level evaluation,
    we only take the overall P/R/F1.
    """
    predictions, labels_true = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = []
    true_predictions = []
    for pred, label in zip(predictions, labels_true):
        pred_labels = []
        label_list = []
        for p_idx, l_idx in zip(pred, label):
            if l_idx != -100:
                pred_labels.append(id2label[p_idx])
                label_list.append(id2label[l_idx])
        true_labels.append(label_list)
        true_predictions.append(pred_labels)

    seqeval_metric = evaluate.load("seqeval")
    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


########################################
# 4. Main Process: Training, Evaluation, and Advanced Analysis (Confusion Matrix, Entity-level F1, etc.)
########################################

if __name__ == "__main__":
    train_path = "SMM4H-2025-Task5-Train_subtask2.csv"
    val_path = "SMM4H-2025-Task5-Validation_subtask2.csv"

    # Load and preprocess data
    train_df, val_df = load_data(train_path, val_path)
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Initialize tokenizer and model
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # Tokenize data & align labels
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

    # Define training parameters
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=15,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    # Dynamic padding
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Construct Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Token-level evaluation
    metrics = trainer.evaluate()
    print("Token-level Evaluation Metrics:")
    print(metrics)

    # Event-level exact match
    exact_match_count = 0
    total_examples = len(val_df)
    device = model.device
    for i in tqdm(range(total_examples), desc="Evaluating event extraction"):
        example = val_df.iloc[i]
        text = example["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: t.to(device) for k, t in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs).logits
        pred_ids = torch.argmax(outputs, dim=2)[0].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        filtered_tokens = []
        filtered_preds = []
        for token, p_id in zip(tokens, pred_ids):
            if token in tokenizer.all_special_tokens:
                continue
            filtered_tokens.append(token)
            filtered_preds.append(id2label[p_id])
        pred_event = extract_event(filtered_preds, filtered_tokens)
        gold_event = convert_gold_to_event(example)
        if events_exact_match(pred_event, gold_event):
            exact_match_count += 1
    event_exact_match_rate = exact_match_count / total_examples
    print(f"Event-level Exact Match Rate: {event_exact_match_rate:.4f}")


    ##########################################
    # 5. Advanced Analysis: Confusion Matrix & Report Excluding O→O #
    ##########################################

    # Collect all predictions and true labels for confusion matrix
    device = model.device
    flat_preds = []
    flat_labels = []
    
    for batch in trainer.get_eval_dataloader():
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        
        for p, l in zip(predictions, labels):
            for pred_id, label_id in zip(p, l):
                if label_id != -100:  # Ignore padding tokens
                    flat_preds.append(pred_id)
                    flat_labels.append(label_id)

    filtered_preds, filtered_labels = [], []
    o_id = label2id["O"]

    for p, l in zip(flat_preds, flat_labels):
        # Keep only where both p and l are not O
        if p != o_id and l != o_id:
            filtered_preds.append(p)
            filtered_labels.append(l)

    # This way, "O" will not appear, allowing for all entity labels
    # Use the original labels list, not the numpy array
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
    print("\nConfusion Matrix (Gold,Pred ≠ O):")
    print(cm_entities)

    print("\nClassification Report (Gold,Pred ≠ O):")
    print(classification_report(
        filtered_labels,
        filtered_preds,
        labels=entity_ids,
        target_names=entity_only,
        zero_division=0
    ))


    ####################################
    # 6. Entity-level (Chunk-Level) F1 Evaluation   #
    ####################################
    
    # Get predictions for validation set
    preds = []
    labels_true = []
    
    for batch in trainer.get_eval_dataloader():
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        
        preds.extend(predictions)
        labels_true.extend(labels)
    
    # Convert to string labels after filtering out -100
    label_seqeval_preds = []
    label_seqeval_trues = []
    for pred_seq, label_seq in zip(preds, labels_true):
        pred_labels_str = []
        gold_labels_str = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            # Ensure that p and l are valid indices for id2label
            if p in id2label:
                pred_labels_str.append(id2label[p])
            else:
                # Handle unexpected prediction ID
                pred_labels_str.append("O")
                
            if l in id2label:
                gold_labels_str.append(id2label[l])
            else:
                # Handle unexpected gold label ID
                gold_labels_str.append("O")
                
        label_seqeval_preds.append(pred_labels_str)
        label_seqeval_trues.append(gold_labels_str)

    seqeval_metric = evaluate.load("seqeval")
    chunk_results = seqeval_metric.compute(
        predictions=label_seqeval_preds,
        references=label_seqeval_trues
    )
    print("\nChunk-Level (entity-level) Seqeval Results:")
    print(chunk_results)