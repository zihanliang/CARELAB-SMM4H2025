import json
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import re
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification,
    AutoConfig,
    EarlyStoppingCallback
)
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Enable deterministic behavior for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###################################
# 1. Define labels, data loading, and utility functions #
###################################

# All entity labels (BIO) + O
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

# Map CSV columns to entity types
column_to_entity = {
    "organization": "TARGET_ORG",
    "product": "PRODUCT",
    "disease": "INFECTION",
    "cause": "SAFETY_INCIDENT",
    "number_of_people_affected": "AFFECTED_NUM",
    "location": "LOCATION"
}

def load_data(train_path, val_path):
    """Load and prepare the training and validation dataframes."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Convert affected numbers to strings if they are not already
    for df in [train_df, val_df]:
        if 'number_of_people_affected' in df.columns:
            df['number_of_people_affected'] = df['number_of_people_affected'].astype(str)
    
    return train_df, val_df

def preprocess_dataframe(df):
    """Clean and preprocess the dataframe."""
    # Fill NaN values with empty strings for entity columns
    for col in column_to_entity.keys():
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    # Clean text and entity values
    for col in ['text'] + list(column_to_entity.keys()):
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: x.strip())
    
    return df

def tokenize_text(text):
    """Simple word tokenization for better entity matching."""
    # Split by spaces but keep punctuation
    return re.findall(r'\w+|[^\w\s]', text)

def normalize_text(text):
    """Normalize text for better entity matching."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def find_entity_spans(text, entity_text):
    """
    Find all occurrences of entity_text in text with better fuzzy matching.
    Returns list of (start, end) tuples.
    """
    if not entity_text or entity_text == '' or entity_text.lower() == 'nan':
        return []
    
    # Normalize both text and entity for matching
    norm_text = normalize_text(text)
    norm_entity = normalize_text(entity_text)
    
    # Direct substring matching
    spans = []
    start = 0
    while True:
        start = norm_text.find(norm_entity, start)
        if start == -1:
            break
        # Ensure entity boundaries
        if (start == 0 or not norm_text[start-1].isalnum()) and \
           (start+len(norm_entity) >= len(norm_text) or not norm_text[start+len(norm_entity)].isalnum()):
            spans.append((start, start + len(norm_entity)))
        start += 1
    
    # Try alternative matching if no direct match found
    if not spans and len(norm_entity) > 3:
        # Try with partial matching for longer entities
        words = norm_entity.split()
        if len(words) > 1:
            # Try matching first few words
            first_words = ' '.join(words[:min(3, len(words))])
            start = 0
            while True:
                start = norm_text.find(first_words, start)
                if start == -1:
                    break
                # Find the best end position
                potential_end = min(start + len(norm_entity) + 10, len(norm_text))
                best_score = 0
                best_end = start + len(first_words)
                
                for end in range(start + len(first_words), potential_end):
                    if end >= len(norm_text) or not norm_text[end].isalnum():
                        candidate = norm_text[start:end]
                        # Compute similarity (simplified)
                        common_len = len(set(candidate.split()) & set(norm_entity.split()))
                        score = common_len / max(len(candidate.split()), len(norm_entity.split()))
                        if score > 0.7 and score > best_score:  # 70% match threshold
                            best_score = score
                            best_end = end
                
                if best_score > 0:
                    spans.append((start, best_end))
                start += 1
    
    # Map normalized spans to original text spans
    original_spans = []
    for norm_start, norm_end in spans:
        # Find equivalent positions in original text
        orig_start = 0
        orig_norm_pos = 0
        # Skip spaces and align with normalized text position
        while orig_norm_pos < norm_start and orig_start < len(text):
            if normalize_text(text[orig_start:orig_start+1]) != '':
                orig_norm_pos += 1
            orig_start += 1
        
        # Find end position
        orig_end = orig_start
        orig_norm_pos = norm_start
        while orig_norm_pos < norm_end and orig_end < len(text):
            if normalize_text(text[orig_end:orig_end+1]) != '':
                orig_norm_pos += 1
            orig_end += 1
        
        # Adjust end to include full token
        while orig_end < len(text) and text[orig_end-1].isalnum():
            orig_end += 1
            
        original_spans.append((orig_start, orig_end))
    
    return original_spans

def get_entities_from_example(example):
    """
    Extract entities from example and return their spans and labels.
    """
    entities = []
    text = example["text"]
    
    for col, ent_type in column_to_entity.items():
        if col in example and example[col] and example[col].strip():
            ent_text = str(example[col]).strip()
            if ent_text and ent_text.lower() != 'nan':
                # Find all occurrences with improved matching
                spans = find_entity_spans(text, ent_text)
                for start, end in spans:
                    entities.append({
                        "start": start,
                        "end": end,
                        "label": ent_type,
                        "text": text[start:end]
                    })
    
    # Sort entities by start position for consistent processing
    entities.sort(key=lambda x: x["start"])
    return entities

def data_augmentation(df):
    """
    Apply data augmentation techniques to improve model generalization.
    """
    augmented_data = []
    
    for _, row in df.iterrows():
        # Create a copy of the original example
        augmented_data.append(row.to_dict())
        
        # Skip augmentation for examples without entities
        has_entities = False
        for col in column_to_entity.keys():
            if col in row and row[col] and str(row[col]).strip() != '' and str(row[col]).lower() != 'nan':
                has_entities = True
                break
        
        if not has_entities:
            continue
            
        # Entity substitution: Replace entity mentions with synonyms or variations
        text = row['text']
        new_text = text
        
        for col, ent_type in column_to_entity.items():
            if col in row and row[col] and str(row[col]).strip() != '' and str(row[col]).lower() != 'nan':
                entity_text = str(row[col]).strip()
                
                # Apply capitalization variation
                if entity_text[0].islower() and len(entity_text) > 1:
                    capitalized = entity_text[0].upper() + entity_text[1:]
                    new_text = new_text.replace(entity_text, capitalized)
                    
        # Only add if text was modified
        if new_text != text:
            new_row = row.to_dict()
            new_row['text'] = new_text
            augmented_data.append(new_row)
    
    return pd.DataFrame(augmented_data)

###########################
# 2. Tokenization & Label Alignment  #
###########################

def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize inputs and align labels to tokens.
    """
    tokenized_inputs = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=512,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    all_labels = []
    
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        text = examples["text"][i]
        # Get all entities for the sample from the CSV
        example = {key: examples[key][i] for key in examples.keys()}
        entities = get_entities_from_example(example)
        
        # Initialize all token labels as "O"
        labels = [-100] * len(offsets)
        
        # Map token offsets to entity labels
        for j, (start_offset, end_offset) in enumerate(offsets):
            # Special tokens have offset (0, 0)
            if start_offset == 0 and end_offset == 0:
                continue
                
            # Find entity that contains this token
            token_start, token_end = start_offset.item(), end_offset.item()
            
            # Default label is "O" (Outside)
            token_label = "O"
            
            for entity in entities:
                entity_start, entity_end = entity["start"], entity["end"]
                
                # Check if token is part of this entity
                if token_start >= entity_start and token_end <= entity_end:
                    # Beginning of entity
                    if token_start == entity_start:
                        token_label = f"B-{entity['label']}"
                    # Inside entity
                    else:
                        token_label = f"I-{entity['label']}"
                    break
            
            # Store the label ID
            labels[j] = label2id.get(token_label, label2id["O"])
        
        all_labels.append(labels)
    
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

###########################
# 3. Model Architecture   #
###########################

class EntityExtractionModel(nn.Module):
    """
    Enhanced token classification model with CRF layer.
    """
    def __init__(self, model_name, num_labels):
        super(EntityExtractionModel, self).__init__()
        self.num_labels = num_labels
        
        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.transformer = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=self.config
        )
        
        # Add additional layers
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
            
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, labels=None, **kwargs):
        
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            **kwargs
        )
        
        # Get transformer output
        sequence_output = outputs.logits
        
        # Apply additional classification layer
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, 
                    labels.view(-1), 
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return outputs._replace(loss=loss) if loss is not None else outputs

###########################
# 4. Helper Functions for Sequence Labeling   #
###########################

def extract_event(predicted_labels, tokens):
    """
    Concatenate tokens into entity text and categorize based on BIO labels.
    """
    event = defaultdict(list)
    current_entity = None
    current_tokens = []
    
    for label, token in zip(predicted_labels, tokens):
        if label.startswith("B-"):
            # If there is an ongoing entity, collect it first
            if current_entity is not None:
                entity_text = " ".join(current_tokens).replace(" ##", "").replace("##", "")
                event[current_entity].append(entity_text)
            
            # Start new entity
            current_entity = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_entity is not None:
            # Continue current entity
            current_tokens.append(token)
        else:
            # End of entity or no entity
            if current_entity is not None:
                entity_text = " ".join(current_tokens).replace(" ##", "").replace("##", "")
                event[current_entity].append(entity_text)
            current_entity = None
            current_tokens = []
    
    # Handle entity at end of sequence
    if current_entity is not None:
        entity_text = " ".join(current_tokens).replace(" ##", "").replace("##", "")
        event[current_entity].append(entity_text)
    
    # Clean up entity text
    for entity_type in event:
        cleaned_entities = []
        for entity_text in event[entity_type]:
            # Remove special tokens and clean whitespace
            cleaned = re.sub(r'\s+', ' ', entity_text).strip()
            if cleaned:
                cleaned_entities.append(cleaned)
        event[entity_type] = cleaned_entities
    
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
    Determine if the predicted event exactly matches the gold event.
    """
    # Compare sets of entity types
    if set(pred_event.keys()) != set(gold_event.keys()):
        return False
    
    # For each entity type, compare sets of entity texts
    for key in pred_event:
        # Normalize predicted and gold entities for comparison
        pred_entities = set(normalize_text(e) for e in pred_event[key])
        gold_entities = set(normalize_text(e) for e in gold_event[key])
        
        if pred_entities != gold_entities:
            return False
    
    return True

def compute_metrics(pred):
    """
    Compute token-level and entity-level evaluation metrics.
    """
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    
    # Extract token-level predictions and labels, ignoring padding
    true_labels = []
    true_predictions = []
    
    for prediction, label in zip(predictions, labels):
        true_pred = []
        true_label = []
        
        for p, l in zip(prediction, label):
            if l != -100:  # Ignore padded tokens
                true_pred.append(id2label[p])
                true_label.append(id2label[l])
        
        true_labels.append(true_label)
        true_predictions.append(true_pred)
    
    # Compute seqeval metrics
    seqeval_metric = evaluate.load("seqeval")
    results = seqeval_metric.compute(
        predictions=true_predictions, 
        references=true_labels,
        scheme="BIO"
    )
    
    # Extract metrics of interest
    metrics = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }
    
    # Add per-entity type F1 scores
    for entity_type in results.keys():
        if isinstance(results[entity_type], dict) and 'f1' in results[entity_type]:
            metrics[f"{entity_type}_f1"] = results[entity_type]['f1']
    
    return metrics

def evaluate_predictions(model, tokenizer, dataset, device):
    """
    Evaluate model predictions on a dataset.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    for example in tqdm(dataset, desc="Evaluating"):
        # Tokenize input
        inputs = tokenizer(
            example["text"], 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            return_offsets_mapping=True
        )
        
        # Get offsets for token-to-character mapping
        offsets = inputs.pop("offset_mapping")[0]
        
        # Move to device
        inputs = {k: v.to(device) for k, t in inputs.items()}
        
        # Generate predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predicted labels
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        
        # Convert token IDs to original tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Filter out special tokens
        filtered_tokens = []
        filtered_predictions = []
        for token, pred, offset in zip(tokens, predictions, offsets):
            # Skip special tokens (CLS, SEP, PAD)
            if offset[0] == offset[1]:
                continue
            filtered_tokens.append(token)
            filtered_predictions.append(id2label[pred])
        
        # Extract entities from predictions
        pred_event = extract_event(filtered_predictions, filtered_tokens)
        
        # Get gold entities
        gold_event = convert_gold_to_event(example)
        
        all_predictions.append(pred_event)
        all_labels.append(gold_event)
    
    return all_predictions, all_labels

def compute_event_level_metrics(predictions, labels):
    """
    Compute event-level evaluation metrics.
    """
    # Exact match count
    exact_matches = 0
    for pred, gold in zip(predictions, labels):
        if events_exact_match(pred, gold):
            exact_matches += 1
    
    exact_match_rate = exact_matches / len(predictions)
    
    # Entity type presence accuracy
    entity_type_correct = defaultdict(int)
    entity_type_total = defaultdict(int)
    
    for pred, gold in zip(predictions, labels):
        # For each entity type in gold
        for entity_type in set(gold.keys()):
            entity_type_total[entity_type] += 1
            # Check if entity type is in pred
            if entity_type in pred:
                entity_type_correct[entity_type] += 1
    
    # Entity value accuracy (partial match)
    entity_value_correct = defaultdict(int)
    
    for pred, gold in zip(predictions, labels):
        # For each entity type in gold
        for entity_type in gold.keys():
            if entity_type in pred:
                # Normalize entity values
                pred_values = set(normalize_text(v) for v in pred[entity_type])
                gold_values = set(normalize_text(v) for v in gold[entity_type])
                
                # Count matches
                if pred_values == gold_values:
                    entity_value_correct[entity_type] += 1
    
    # Calculate metrics
    metrics = {
        "exact_match_rate": exact_match_rate,
    }
    
    # Add type-level metrics
    for entity_type in entity_type_total:
        total = entity_type_total[entity_type]
        correct_type = entity_type_correct[entity_type]
        correct_value = entity_value_correct[entity_type]
        
        metrics[f"{entity_type}_type_accuracy"] = correct_type / total if total > 0 else 0
        metrics[f"{entity_type}_value_accuracy"] = correct_value / total if total > 0 else 0
    
    return metrics

########################################
# 5. Main Process: Training and Evaluation
########################################

def main():
    # Data paths
    train_path = "SMM4H-2025-Task5-Train_subtask2.csv"
    val_path = "SMM4H-2025-Task5-Validation_subtask2.csv"

    # Model configuration
    model_name = "microsoft/deberta-v3-large"  # Upgraded from BERT
    batch_size = 8
    learning_rate = 2e-5
    epochs = 10
    warmup_steps = 0.1  # 10% of training steps
    output_dir = "./results_deberta_large"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, val_df = load_data(train_path, val_path)
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    
    # Apply data augmentation to training data
    print("Applying data augmentation...")
    train_df = data_augmentation(train_df)
    
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Initialize tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize data & align labels
    print("Tokenizing and aligning labels...")
    tokenize_fn = lambda examples: tokenize_and_align_labels(examples, tokenizer)
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

    # Set format for pytorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels", "text"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels", "text"])

    # Calculate training steps for scheduler
    total_steps = (len(train_dataset) // batch_size) * epochs
    warmup_steps = int(total_steps * warmup_steps)
    
    # Define training parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        warmup_steps=warmup_steps,
        seed=seed,
        gradient_accumulation_steps=2,
        fp16=True,
        report_to="none",
    )

    # Initialize model (could use custom architecture)
    print(f"Loading model: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # Dynamic padding
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    # Construct Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping]
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Print token-level evaluation
    metrics = trainer.evaluate()
    print("\nToken-level Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Event-level exact match evaluation
    print("\nPerforming event-level evaluation...")
    device = model.device
    
    # Generate predictions and gold labels
    all_predictions, all_gold_labels = evaluate_predictions(
        model, tokenizer, val_dataset, device
    )
    
    # Compute event-level metrics
    event_metrics = compute_event_level_metrics(all_predictions, all_gold_labels)
    
    print("\nEvent-level Metrics:")
    for key, value in event_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Advanced Analysis: Confusion Matrix
    print("\nGenerating confusion matrix...")
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

    # Filter out O-O pairs for better visibility
    filtered_preds, filtered_labels = [], []
    o_id = label2id["O"]
    
    for p, l in zip(flat_preds, flat_labels):
        if p != o_id and l != o_id:
            filtered_preds.append(p)
            filtered_labels.append(l)
    
    # Get entity-only labels
    entity_labels = [label for label in labels if label != "O"]
    entity_ids = [label2id[label] for label in entity_labels]
    
    # Compute confusion matrix (excluding O)
    if filtered_preds and filtered_labels:
        cm = confusion_matrix(filtered_labels, filtered_preds, labels=entity_ids)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=entity_labels
        )
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Confusion Matrix (Gold,Pred ≠ O)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        
        # Print classification report
        print("\nClassification Report (Gold,Pred ≠ O):")
        report = classification_report(
            filtered_labels,
            filtered_preds,
            labels=entity_ids,
            target_names=entity_labels,
            zero_division=0
        )
        print(report)
        
        # Save report to file
        with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
            f.write(report)
    
    # Error analysis
    print("\nPerforming error analysis...")
    error_analysis = []
    
    for i, (pred_event, gold_event) in enumerate(zip(all_predictions, all_gold_labels)):
        example = val_df.iloc[i]
        text = example["text"]
        
        if not events_exact_match(pred_event, gold_event):
            # Create analysis entry
            analysis = {
                "text": text,
                "gold": gold_event,
                "pred": pred_event,
                "errors": []
            }
            
# Analyze errors
            # Check for missing entity types
            for entity_type in gold_event:
                if entity_type not in pred_event:
                    analysis["errors"].append(f"Missing entity type: {entity_type}")
                    
            # Check for extra entity types
            for entity_type in pred_event:
                if entity_type not in gold_event:
                    analysis["errors"].append(f"Extra entity type: {entity_type}")
            
            # Check for value mismatches
            for entity_type in set(gold_event.keys()) & set(pred_event.keys()):
                gold_values = set(normalize_text(v) for v in gold_event[entity_type])
                pred_values = set(normalize_text(v) for v in pred_event[entity_type])
                
                # Missing values
                for value in gold_values - pred_values:
                    analysis["errors"].append(f"Missing {entity_type} value: {value}")
                
                # Extra values
                for value in pred_values - gold_values:
                    analysis["errors"].append(f"Extra {entity_type} value: {value}")
            
            error_analysis.append(analysis)
    
    # Save error analysis to file
    if error_analysis:
        with open(os.path.join(output_dir, "error_analysis.json"), "w") as f:
            json.dump(error_analysis, f, indent=2)
        
        # Print sample errors
        print("\nSample error cases:")
        for i, analysis in enumerate(error_analysis[:5]):  # Show first 5 errors
            print(f"\nError case {i+1}:")
            print(f"Text: {analysis['text'][:100]}...")
            print(f"Gold: {analysis['gold']}")
            print(f"Pred: {analysis['pred']}")
            print(f"Errors: {analysis['errors']}")
    
    # Save model and tokenizer
    output_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(output_model_dir, exist_ok=True)
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    
    print(f"\nTraining completed. Best model saved to {output_model_dir}")
    
    # Generate and save predictions for validation set
    predictions = []
    
    for i, example in enumerate(tqdm(val_df.iterrows(), desc="Generating predictions", total=len(val_df))):
        _, row = example
        text = row["text"]
        
        # Tokenize
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            return_offsets_mapping=True
        )
        offsets = inputs.pop("offset_mapping")[0]
        
        # Get predictions
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions_idx = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        
        # Map to tokens and labels
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        filtered_tokens = []
        filtered_predictions = []
        
        for token, pred_id, offset in zip(tokens, predictions_idx, offsets):
            if offset[0] == offset[1]:  # Special token
                continue
            filtered_tokens.append(token)
            filtered_predictions.append(id2label[pred_id])
        
        # Extract entities
        pred_event = extract_event(filtered_predictions, filtered_tokens)
        
        # Create prediction entry
        prediction = {
            "docid": row["docid"] if "docid" in row else i,
            "text": text,
        }
        
        # Add extracted entities
        for entity_type, entity_key in {
            "TARGET_ORG": "organization",
            "PRODUCT": "product",
            "SAFETY_INCIDENT": "cause",
            "INFECTION": "disease",
            "AFFECTED_NUM": "number_of_people_affected",
            "LOCATION": "location"
        }.items():
            if entity_type in pred_event and pred_event[entity_type]:
                # Use first entity as prediction (can be modified based on confidence)
                prediction[entity_key] = pred_event[entity_type][0]
            else:
                prediction[entity_key] = ""
        
        predictions.append(prediction)
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    print(f"Predictions saved to {os.path.join(output_dir, 'predictions.csv')}")
    
    return model, tokenizer, metrics, event_metrics

if __name__ == "__main__":
    main()
