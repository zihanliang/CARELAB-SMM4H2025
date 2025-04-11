"""
Advanced Food Safety Article Classification Model with Enhanced Neither Training Data:
- Significantly improved "Neither" class detection using specialized techniques and additional augmented data
- Combines RoBERTa-large base model with custom classification head
- Implements aggressive data balancing and specialized loss functions
- Uses a targeted post-processing pipeline for better "Neither" classification
- Adds ensemble techniques to improve overall robustness
- Incorporates additional FDA Neither augmented samples from fda_neither_augmented_100.json

Training file: preprocessed_SMM4H-2025-Task5-Train_subtask1.csv
Validation file: preprocessed_SMM4H-2025-Task5-Validation_subtask1.csv
"""

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

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModel, EarlyStoppingCallback
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from dataclasses import field

try:
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

# -------------------------------------------------------------------
# Monkey patch: Trainer 的 accelerator_config
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Monkey patch: Accelerator.__init__
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Monkey patch: AcceleratorState._reset_state（采用 classmethod 形式）
# -------------------------------------------------------------------
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

# -------------------------------
# Seed Setting
# -------------------------------
def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------------
# Label Mapping
# -------------------------------
label2id = {"Food Recall": 0, "Foodborne Disease Outbreak": 1, "Neither": 2}
id2label = {v: k for k, v in label2id.items()}

# -------------------------------
# Class-Balanced Focal Loss Implementation
# -------------------------------
class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss:
    Combines focal loss with class-balanced weighting based on effective number of samples
    """
    def __init__(self, alpha=None, gamma=2.0, beta=0.9999, samples_per_class=None, reduction='mean'):
        super(CBFocalLoss, self).__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma  # focusing parameter
        self.reduction = reduction
        self.beta = beta
        self.samples_per_class = samples_per_class
        
        # Calculate class-balanced weights if samples_per_class is provided
        if samples_per_class is not None:
            effective_num = 1.0 - torch.pow(self.beta, torch.tensor(samples_per_class).float())
            self.cb_weights = (1.0 - self.beta) / effective_num
            self.cb_weights = self.cb_weights / torch.sum(self.cb_weights) * len(samples_per_class)
            
            # Combine with provided alpha if it exists
            if self.alpha is not None:
                self.weights = self.alpha * self.cb_weights
            else:
                self.weights = self.cb_weights
        else:
            self.weights = self.alpha
        
    def forward(self, inputs, targets):
        """Forward pass with dynamic weighting"""
        # Apply class weights if available
        if self.weights is not None:
            weights = self.weights.to(inputs.device)
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=weights)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            
        pt = torch.exp(-ce_loss)  # predicted probability
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -------------------------------
# Enhanced Dataset with Text Analysis Features
# -------------------------------
class EnhancedDataset(Dataset):
    """
    Enhanced Dataset: 
    - Supports original text and augmented text
    - Extracts text features as additional signals for "Neither" detection
    - Can use multiple augmentation strategies
    """
    def __init__(self, dataframe, tokenizer, max_length=512, label_column="Subtask1_Label", 
                 label2id=label2id, use_augmentation=False, use_features=True):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_column = label_column
        self.label2id = label2id
        self.use_augmentation = use_augmentation
        self.use_features = use_features
        
        # Print label distribution for monitoring
        self.label_counts = dataframe[label_column].value_counts().to_dict()
        print(f"Label distribution: {self.label_counts}")
        
        # Add text features if enabled
        if self.use_features:
            self._add_text_features()
    
    def _add_text_features(self):
        """Add engineered text features that help identify 'Neither' class"""
        # Check for food recall and outbreak keywords in texts
        food_recall_keywords = ["recall", "recalled", "recalling", "fda", "withdraw", "alert"]
        outbreak_keywords = ["outbreak", "infection", "infected", "sick", "illness", "case", "symptom"]
        
        # Extract features
        self.df['text_length'] = self.df['cleaned_text'].apply(lambda x: len(str(x)))
        self.df['has_recall_terms'] = self.df['cleaned_text'].apply(
            lambda x: any(term in str(x).lower() for term in food_recall_keywords))
        self.df['has_outbreak_terms'] = self.df['cleaned_text'].apply(
            lambda x: any(term in str(x).lower() for term in outbreak_keywords))
        self.df['neither_indicator'] = ~(self.df['has_recall_terms'] | self.df['has_outbreak_terms'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Select text based on augmentation settings
        if self.use_augmentation and random.random() < 0.3 and "augmented_text" in row and row["augmented_text"]:
            text = str(row["augmented_text"])
        else:
            text = str(row["cleaned_text"])
        
        # Get label
        label = row[self.label_column]
        if pd.isna(label):
            label = "Neither"
        if self.label2id:
            label = self.label2id[label]
        # Tokenize text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Remove batch dimension
        for key in tokenized:
            tokenized[key] = tokenized[key].squeeze(0)
        
        # Create basic return dictionary
        item_dict = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label
        }
        
        # Add text features if enabled
        if self.use_features:
            # Text length as a normalized feature (0-1)
            text_len = min(1.0, row.get('text_length', len(text)) / 5000.0)
            has_recall = float(row.get('has_recall_terms', False))
            has_outbreak = float(row.get('has_outbreak_terms', False))
            neither_ind = float(row.get('neither_indicator', False))
            
            item_dict["text_features"] = torch.tensor(
                [text_len, has_recall, has_outbreak, neither_ind], 
                dtype=torch.float
            )
        
        # Add segments for hierarchical processing if available
        if "segments" in row:
            try:
                segments = ast.literal_eval(str(row["segments"]))
                if isinstance(segments, list) and len(segments) > 0:
                    item_dict["segments"] = segments
            except:
                pass
        
        return item_dict

def enhanced_collate_fn(batch):
    """
    Enhanced collate function:
    Handles both standard inputs and additional features
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    
    # Create basic batch dictionary
    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    # Add text features if present
    if "text_features" in batch[0]:
        text_features = torch.stack([item["text_features"] for item in batch])
        batch_dict["text_features"] = text_features
    
    # Add segments if present
    if "segments" in batch[0]:
        batch_dict["segments"] = [item.get("segments", []) for item in batch]
    
    return batch_dict

# -------------------------------
# Enhanced Classifier with Text Features
# -------------------------------
class EnhancedClassifier(nn.Module):
    """
    Enhanced Classifier:
    - Uses RoBERTa-large as encoder
    - Incorporates text features for better "Neither" detection
    - Implements dual classification heads with different weights
    - Uses temperature scaling for calibration
    """
    def __init__(self, model_name="roberta-large", num_labels=3, class_weight=None, 
                 alpha=0.7, gamma=2.0, temperature=1.0, use_features=True):
        super(EnhancedClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Feature usage flag
        self.use_features = use_features
        feature_size = 4 if use_features else 0
        
        # Main classifier with feature integration
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
        
        # Specialized classifier for the "Neither" class
        self.neither_classifier = nn.Sequential(
            nn.Linear(hidden_size + feature_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)  # Binary: is it "Neither" or not
        )
        
        # Auxiliary classifier for regularization
        self.aux_classifier = nn.Linear(hidden_size, num_labels)
        
        # Binary classifier specifically trained to distinguish "Neither" vs others
        self.binary_classifier = nn.Linear(hidden_size + feature_size, 2)
        
        # Loss and parameters
        self.class_weight = class_weight
        self.focal_loss = CBFocalLoss(alpha=class_weight, gamma=gamma)
        self.alpha = alpha  # weight for main classifier
        self.temperature = temperature  # temperature for calibration
        
        # Layer to process text features
        if use_features:
            self.feature_projection = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU()
            )
            
            # Feature fusion layer
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_size + 32, hidden_size + feature_size),
                nn.LayerNorm(hidden_size + feature_size)
            )

    def forward(self, input_ids, attention_mask, labels=None, text_features=None):
        # Encode the text
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Process and integrate text features if available
        if self.use_features and text_features is not None:
            # Project text features
            projected_features = self.feature_projection(text_features)
            
            # Concatenate with text embedding
            combined_repr = torch.cat([pooled_output, projected_features], dim=1)
            
            # Apply fusion layer
            fused_repr = self.fusion_layer(combined_repr)
            
            # Use fused representation for classification
            logits = self.classifier(fused_repr)
            
            # Apply neither-specific classifier
            neither_logits = self.neither_classifier(fused_repr).squeeze(-1)
            
            # Binary classification: Neither vs Others
            binary_logits = self.binary_classifier(fused_repr)
        else:
            # Use only text representation
            logits = self.classifier(pooled_output)
            neither_logits = torch.zeros(pooled_output.size(0), device=pooled_output.device)
            binary_logits = None
        
        # Auxiliary classifier
        aux_logits = self.aux_classifier(pooled_output)
        
        # Modify logits for "Neither" class based on neither_classifier output
        if neither_logits is not None:
            # Scale logits for "Neither" class (index 2)
            neither_weight = 1.2  # Boost Neither class prediction
            neither_boost = F.sigmoid(neither_logits).unsqueeze(1) * neither_weight
            
            # Apply selective boosting to the "Neither" class logits only
            boost_mask = torch.zeros_like(logits)
            boost_mask[:, 2] = 1.0  # Apply only to "Neither" class (index 2)
            logits = logits + boost_mask * neither_boost
        
        # Apply temperature scaling for better calibration
        scaled_logits = logits / self.temperature
        
        # Combine with auxiliary logits
        combined_logits = self.alpha * scaled_logits + (1 - self.alpha) * aux_logits
        
        # If labels are provided, compute loss
        loss = None
        if labels is not None:
            # Main loss: focal loss on combined logits
            main_loss = self.focal_loss(combined_logits, labels)
            
            # Auxiliary loss: cross entropy on aux_logits
            if self.class_weight is not None:
                weight = self.class_weight.to(aux_logits.device)
                aux_loss = F.cross_entropy(aux_logits, labels, weight=weight)
            else:
                aux_loss = F.cross_entropy(aux_logits, labels)
            
            # Binary loss for Neither detection if binary_logits available
            binary_loss = 0
            if binary_logits is not None:
                # Create binary targets: 1 for Neither, 0 for others
                binary_targets = (labels == 2).long()
                binary_loss = F.cross_entropy(binary_logits, binary_targets)
            
            # Neither-specific loss
            neither_loss = 0
            if neither_logits is not None:
                neither_targets = (labels == 2).float()
                neither_loss = F.binary_cross_entropy_with_logits(
                    neither_logits, 
                    neither_targets,
                    pos_weight=torch.tensor([5.0]).to(neither_logits.device)  # Higher weight for Neither
                )
            
            # Total loss
            loss = main_loss + 0.3 * aux_loss + 0.2 * binary_loss + 0.5 * neither_loss
            
        return {"loss": loss, "logits": combined_logits, "neither_score": neither_logits}

    def predict(self, input_ids, attention_mask, text_features=None):
        """
        Prediction function that returns calibrated logits
        """
        outputs = self.forward(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            text_features=text_features
        )
        return outputs["logits"]

# -------------------------------
# Evaluation Metrics
# -------------------------------
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics with special handling for the 'Neither' class
    """
    # Extract logits and labels
    logits, labels = eval_pred
    
    # Handle potential issues with dimensions
    if isinstance(logits, tuple):
        logits = logits[0]  # Take only the main logits if multiple outputs
    
    if isinstance(labels, list) and len(labels) == 1:
        labels = labels[0]
    
    # Ensure labels are properly shaped
    labels = np.array(labels)
    if labels.ndim > 1:
        labels = labels.squeeze()
    
    # Get class predictions
    preds = np.argmax(logits, axis=-1)
    
    # Safety check for dimension mismatch
    if len(preds) != len(labels):
        min_len = min(len(preds), len(labels))
        preds = preds[:min_len]
        labels = labels[:min_len]
    
    # Compute metrics with special handling for potential missing classes
    try:
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, labels=[0, 1, 2], zero_division=0
        )
        
        precision_weighted = np.average(precision, weights=support)
        recall_weighted = np.average(recall, weights=support)
        f1_weighted = np.average(f1, weights=support)
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        
        acc = accuracy_score(labels, preds)
        
        # Initialize metrics dictionary
        metrics = {
            "accuracy": acc,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
        }
        
        # Add per-class metrics
        for i, label in enumerate(["food_recall", "foodborne_disease", "neither"]):
            if i < len(precision):
                metrics[f"{label}_precision"] = float(precision[i])
                metrics[f"{label}_recall"] = float(recall[i])
                metrics[f"{label}_f1"] = float(f1[i])
                metrics[f"{label}_support"] = int(support[i])
        
    except Exception as e:
        print(f"Error in computing metrics: {e}")
        # Fallback metrics
        metrics = {
            "accuracy": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "food_recall_f1": 0.0,
            "foodborne_disease_f1": 0.0,
            "neither_f1": 0.0
        }
    
    # Print detailed classification report
    try:
        print("\n" + classification_report(
            labels, preds, 
            target_names=['Food Recall', 'Foodborne Disease Outbreak', 'Neither'],
            digits=4
        ))
    except Exception as e:
        print(f"Error generating classification report: {e}")
    
    return metrics

# -------------------------------
# Data Preparation and Training
# -------------------------------

# Read preprocessed CSV data
train_data_file = "preprocessed_SMM4H-2025-Task5-Train_subtask1_full.csv"
val_data_file = "preprocessed_SMM4H-Task5-Validation_subtask1_full.csv"

df_train = pd.read_csv(train_data_file)
df_val = pd.read_csv(val_data_file)

print(f"Training set size: {len(df_train)}")
print(f"Validation set size: {len(df_val)}")

# Check label distribution
train_label_dist = df_train["Subtask1_Label"].value_counts()
val_label_dist = df_val["Subtask1_Label"].value_counts()

print(f"Training label distribution:\n{train_label_dist}")
print(f"Validation label distribution:\n{val_label_dist}")

# Advanced data balancing through extreme oversampling of "Neither" class
def balanced_sampling(df, minority_class="Neither", oversample_ratio=10.0):
    """
    Aggressively oversample the minority class with additional augmentation
    """
    minority_samples = df[df["Subtask1_Label"] == minority_class]
    
    if len(minority_samples) == 0:
        print("Warning: No samples for the minority class!")
        return df
    
    # Exact copying for consistent oversampling
    oversampled = pd.concat([df] + [minority_samples] * int(oversample_ratio), ignore_index=True)
    
    print(f"Dataset size after oversampling: {len(oversampled)}")
    print(f"Label distribution after oversampling:\n{oversampled['Subtask1_Label'].value_counts()}")
    
    return oversampled

# Create balanced training set (extremely aggressive on "Neither" class)
df_train_balanced = balanced_sampling(df_train, "Neither", oversample_ratio=10.0)

# Use RoBERTa-large as the base model
model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create enhanced datasets
train_dataset = EnhancedDataset(df_train_balanced, tokenizer, max_length=512, use_augmentation=True, use_features=True)
val_dataset = EnhancedDataset(df_val, tokenizer, max_length=512, use_augmentation=False, use_features=True)

# Calculate special class weights for "Neither" detection
train_labels = df_train_balanced["Subtask1_Label"].map(label2id).values
samples_per_class = np.bincount(np.clip(train_labels.astype(int), 0, None))

# Very high weight for "Neither" class (index 2)
class_weights = torch.tensor([1.0, 1.0, 5.0], dtype=torch.float)
print("Class weights:", class_weights.numpy())

# Initialize the enhanced model
model = EnhancedClassifier(
    model_name=model_name,
    num_labels=3,
    class_weight=class_weights,
    alpha=0.7,  # weight for main classifier
    gamma=2.0,  # focal loss parameter
    temperature=0.8,  # sharper logits (< 1.0)
    use_features=True
)

# Configure advanced training parameters
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./st1_model_neither_enhanced_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="neither_f1",
    greater_is_better=True,
    fp16=True,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    seed=42,
    report_to="none",
)

# Create custom trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=enhanced_collate_fn,
    compute_metrics=compute_metrics,
)

print("Starting model training with specialized Neither-class enhancements...")
trainer.train()

print("Evaluating final model...")
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Save the final model
output_dir = "./enhanced_neither_model"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# -------------------------------
# Advanced Prediction with Uncertainty Estimation
# -------------------------------
def predict_with_uncertainty(df_test, model_path="./enhanced_neither_model", monte_carlo_samples=10):
    """
    Prediction with Monte Carlo dropout for uncertainty estimation
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = EnhancedClassifier(
        model_name="roberta-large", 
        num_labels=3,
        use_features=True
    )
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create dataset with features
    test_dataset = EnhancedDataset(df_test, tokenizer, max_length=512, use_features=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=enhanced_collate_fn)
    
    all_preds = []
    all_uncertainties = []
    
    # Enable dropout for MC sampling
    model.train()  # Set to train mode to enable dropout
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Multiple forward passes for uncertainty estimation
            batch_logits = []
            for _ in range(monte_carlo_samples):
                logits = model.predict(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    text_features=batch.get("text_features")
                )
                batch_logits.append(logits)
            
            # Stack and compute mean and variance
            stacked_logits = torch.stack(batch_logits)  # [samples, batch, classes]
            mean_logits = stacked_logits.mean(dim=0)  # [batch, classes]
            
            # Compute predictive entropy as uncertainty
            probs = F.softmax(mean_logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            # Get predictions from mean logits
            preds = torch.argmax(mean_logits, dim=1).cpu().numpy()
            uncertainties = entropy.cpu().numpy()
            
            all_preds.extend(preds)
            all_uncertainties.extend(uncertainties)
    
    # Map predictions to labels
    pred_labels = [id2label[pred] for pred in all_preds]
    
    # Create results DataFrame
    df_result = df_test.copy()
    df_result["predicted_label"] = pred_labels
    df_result["prediction_uncertainty"] = all_uncertainties
    
    return df_result, all_preds, all_uncertainties

# -------------------------------
# Advanced Neither Detection Post-Processing
# -------------------------------
def advanced_neither_detection(df_result, text_column="cleaned_text", uncertainty_threshold=0.7):
    """
    Advanced post-processing with linguistic rules and uncertainty thresholding
    """
    # Keywords and patterns for different classes
    neither_indicators = [
        "not food", "unrelated", "non-food", "not related", "irrelevant",
        "not about food", "nothing to do with food", "different topic"
    ]
    
    food_recall_indicators = [
        "recall", "recalled", "recalling", "withdrawing", "fda recall", 
        "voluntary recall", "remove from shelves", "pull products"
    ]
    
    foodborne_indicators = [
        "outbreak", "infection", "infected", "illness", "sick", "symptoms",
        "disease", "foodborne", "food poisoning", "contamination"
    ]
    
    # Check each document with linguistic rules and uncertainty
    for idx, row in df_result.iterrows():
        if text_column in row and row[text_column]:
            text = str(row[text_column]).lower()
            current_pred = row['predicted_label']
            uncertainty = row.get('prediction_uncertainty', 0)
            
            # Rule 1: Strong "Neither" indicators
            if any(indicator in text for indicator in neither_indicators):
                df_result.at[idx, 'predicted_label'] = 'Neither'
                continue
                
            # Rule 2: High uncertainty cases with no strong food-related indicators
            if (uncertainty > uncertainty_threshold and
                not any(indicator in text for indicator in food_recall_indicators) and
                not any(indicator in text for indicator in foodborne_indicators)):
                df_result.at[idx, 'predicted_label'] = 'Neither'
                continue
                
            # Rule 3: Short texts with minimal food-related content
            if (len(text) < 300 and 
                not any(food_term in text for food_term in [
                    "food", "fda", "recall", "outbreak", "listeria", "salmonella", 
                    "e. coli", "product", "contamination"
                ])):
                df_result.at[idx, 'predicted_label'] = 'Neither'
                continue
                
            # Rule 4: Length-based heuristic - very long texts are unlikely to be "Neither"
            if len(text) > 3000 and current_pred == 'Neither':
                # Check for strong recall or outbreak indicators before changing
                if any(indicator in text for indicator in food_recall_indicators):
                    df_result.at[idx, 'predicted_label'] = 'Food Recall'
                elif any(indicator in text for indicator in foodborne_indicators):
                    df_result.at[idx, 'predicted_label'] = 'Foodborne Disease Outbreak'
    
    return df_result

# -------------------------------
# Multi-Model Ensemble with Weighting
# -------------------------------
def train_multiple_models(n_models=3):
    """
    Train multiple models with different configurations for ensemble
    """
    model_paths = []
    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}...")
        
        # Vary seed for diversity
        set_seed(42 + i * 100)
        
        # Different oversampling ratios for diversity
        oversample_ratio = 8.0 + i * 2.0
        df_train_variant = balanced_sampling(df_train, "Neither", oversample_ratio=oversample_ratio)
        
# Create datasets with varying augmentation
        train_dataset_variant = EnhancedDataset(
            df_train_variant, tokenizer, max_length=512, 
            use_augmentation=True, 
            use_features=True
        )
        
        # Vary model parameters slightly
        model_variant = EnhancedClassifier(
            model_name=model_name,
            num_labels=3,
            class_weight=torch.tensor([1.0, 1.0, 5.0 + i]),  # Increasing Neither weight
            alpha=0.7 - (i * 0.05),  # Vary blend factors
            gamma=2.0 + (i * 0.5),   # Vary focal loss gamma
            temperature=0.8 - (i * 0.1),  # Vary temperature
            use_features=True
        )
        
        # Vary training arguments
        variant_output_dir = f"./neither_model_variant_{i+1}"
        training_args_variant = TrainingArguments(
            output_dir=variant_output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-6 * (1.0 + (i * 0.2)),  # Vary learning rate
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            num_train_epochs=8 + i,  # Slightly different training lengths
            weight_decay=0.01 + (i * 0.002),
            logging_dir=f"./logs_variant_{i+1}",
            logging_steps=20,
            load_best_model_at_end=True,
            metric_for_best_model="neither_f1",
            greater_is_better=True,
            fp16=True,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1 + (i * 0.02),
            seed=42 + (i * 100),
            report_to="none",
        )
        
        # Create custom trainer
        trainer_variant = Trainer(
            model=model_variant,
            args=training_args_variant,
            train_dataset=train_dataset_variant,
            eval_dataset=val_dataset,
            data_collator=enhanced_collate_fn,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
        
        # Train the model
        trainer_variant.train()
        
        # Evaluate the model
        eval_results = trainer_variant.evaluate()
        neither_f1 = eval_results.get("neither_f1", 0)
        print(f"Model {i+1} Neither F1: {neither_f1:.4f}")
        
        # Save the model
        trainer_variant.save_model(variant_output_dir)
        tokenizer.save_pretrained(variant_output_dir)
        model_paths.append(variant_output_dir)
    
    print(f"Successfully trained {n_models} different model variants")
    return model_paths

# -------------------------------
# Weighted Ensemble Prediction
# -------------------------------
def ensemble_predict(df_test, model_paths, weights=None):
    """
    Ensemble prediction with weighted voting and advanced calibration
    """
    if not model_paths:
        raise ValueError("No model paths provided for ensemble prediction")
    
    # Load tokenizer from the first model
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
    
    # Default to equal weights if not provided
    if weights is None:
        weights = [1.0] * len(model_paths)
    else:
        # Normalize weights
        weights = [w / sum(weights) for w in weights]
    
    # Initialize arrays for model outputs
    all_logits = []
    all_probs = []
    all_neither_scores = []
    
    # Process each model
    for i, model_path in enumerate(model_paths):
        print(f"Running prediction with model {i+1}/{len(model_paths)}...")
        
        # Load the model
        model = EnhancedClassifier(
            model_name="roberta-large", 
            num_labels=3,
            use_features=True
        )
        try:
            model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            continue
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Create dataset with features
        test_dataset = EnhancedDataset(df_test, tokenizer, max_length=512, use_features=True)
        test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=enhanced_collate_fn)
        
        model_logits = []
        model_neither_scores = []
        
        # Process batches
        with torch.no_grad():
            for batch in test_dataloader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    text_features=batch.get("text_features")
                )
                
                logits = outputs["logits"]
                neither_score = outputs.get("neither_score", None)
                
                # Store batch outputs
                model_logits.append(logits.cpu())
                if neither_score is not None:
                    model_neither_scores.append(neither_score.cpu())
        
        # Concatenate batch outputs
        if model_logits:
            full_logits = torch.cat(model_logits, dim=0).numpy()
            all_logits.append(full_logits)
            
            # Convert to probabilities
            probs = torch.softmax(torch.tensor(full_logits), dim=1).numpy()
            all_probs.append(probs)
        
        if model_neither_scores and model_neither_scores[0] is not None:
            full_neither_scores = torch.cat(model_neither_scores, dim=0).numpy()
            all_neither_scores.append(full_neither_scores)
    
    # Apply ensemble method
    if not all_logits:
        raise ValueError("No valid predictions from any model")
    
    # Compute weighted average probabilities
    weighted_probs = np.zeros_like(all_probs[0])
    for i, probs in enumerate(all_probs):
        weighted_probs += probs * weights[i]
    
    # Get final predictions
    final_preds = np.argmax(weighted_probs, axis=1)
    
    # Extra boost for Neither class based on specialized scores
    if all_neither_scores:
        neither_boost = np.zeros(len(final_preds))
        for i, scores in enumerate(all_neither_scores):
            neither_boost += sigmoid(scores) * weights[i]
        
        # Apply boost to borderline cases
        for i, (pred, boost) in enumerate(zip(final_preds, neither_boost)):
            # If Neither score is very high but prediction is not Neither
            if boost > 0.7 and pred != 2:
                final_preds[i] = 2  # Change to Neither
            # If prediction is Neither but score is very low
            elif boost < 0.2 and pred == 2:
                # Find next highest probability class
                probs_without_neither = weighted_probs[i].copy()
                probs_without_neither[2] = 0  # Zero out Neither class
                final_preds[i] = np.argmax(probs_without_neither)
    
    # Map predictions to labels
    pred_labels = [id2label[pred] for pred in final_preds]
    
    # Create results DataFrame
    df_result = df_test.copy()
    df_result["predicted_label"] = pred_labels
    df_result["neither_confidence"] = weighted_probs[:, 2]  # Store confidence for Neither class
    
    return df_result, final_preds, weighted_probs

# Helper function for sigmoid
def sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

# -------------------------------
# Complete Pipeline for Neither Detection
# -------------------------------
def complete_neither_detection_pipeline(df_test, ensemble_size=3, use_post_processing=True):
    """
    Complete pipeline combining ensemble models and rule-based post-processing
    """
    print("Starting complete Neither detection pipeline...")
    
    # 1. Train multiple models for ensemble
    print(f"Training {ensemble_size} models for ensemble...")
    model_paths = train_multiple_models(n_models=ensemble_size)
    
    # 2. Use ensemble prediction
    print("Running ensemble prediction...")
    df_result, _, probs = ensemble_predict(df_test, model_paths)
    
    # 3. Apply advanced post-processing
    if use_post_processing:
        print("Applying advanced post-processing rules...")
        df_result = advanced_neither_detection(df_result)
    
    print("Pipeline completed successfully")
    return df_result

# -------------------------------
# Main Execution Flow
# -------------------------------
if __name__ == "__main__":
    print("Food Safety Article Classification with Enhanced Neither Detection")
    print("Available functions for prediction:")
    print("1. predict_with_uncertainty() - Single model prediction with uncertainty estimation")
    print("2. ensemble_predict() - Multi-model ensemble prediction")
    print("3. advanced_neither_detection() - Apply advanced rule-based post-processing")
    print("4. complete_neither_detection_pipeline() - Full pipeline with ensemble and post-processing")
    print("\nExample usage:")
    print("result_df = complete_neither_detection_pipeline(test_data, ensemble_size=3)")
