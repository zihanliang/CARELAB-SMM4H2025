# Task 4: Insomnia Detection from Clinical Notes

## Code and Model Structure

This task focuses on detecting insomnia-related indicators in clinical notes using a combination of rule-based and machine learning approaches. The key script is:

```
Task 4.py        # Main script for training, inference, and evaluation
```

The code supports three subtasks:

* **Subtask 1**: Binary classification (Insomnia: yes/no)
* **Subtask 2a**: Multi-label classification (Definition 1, Definition 2, Rule A, Rule B, Rule C)
* **Subtask 2b**: Evidence extraction for each rule

Models and outputs are saved to the following folders:

```
models/                      # Stores trained models and vectorizers
predictions/                 # Stores prediction results in raw or competition format
```

Supported models:

* Traditional ML: Logistic Regression, SVM, Random Forest, XGBoost
* Deep Learning: BiLSTM
* Transformer-based: ClinicalBERT

---

## How to Use

Below are example commands to run the pipeline. Replace paths with your actual file locations.

### 1. Train Basic Models

```bash
python insomnia_detection.py \
  --noteevents path/to/NOTEEVENTS.csv \
  --subtask1_labels path/to/subtask1_labels.json \
  --subtask2a_labels path/to/subtask2a_labels.json \
  --subtask2b_labels path/to/subtask2b_labels.json \
  --model_dir models \
  --train
```

### 2. Train with BERT and Data Augmentation

```bash
python insomnia_detection.py \
  --noteevents path/to/NOTEEVENTS.csv \
  --subtask1_labels path/to/subtask1_labels.json \
  --subtask2a_labels path/to/subtask2a_labels.json \
  --subtask2b_labels path/to/subtask2b_labels.json \
  --model_dir models \
  --train --use_bert --data_augmentation
```

### 3. Predict and Evaluate on Test Set

```bash
python insomnia_detection.py \
  --noteevents path/to/NOTEEVENTS.csv \
  --test_ids path/to/test_ids.txt \
  --subtask1_labels path/to/subtask1_labels.json \
  --subtask2a_labels path/to/subtask2a_labels.json \
  --subtask2b_labels path/to/subtask2b_labels.json \
  --model_dir models \
  --predict --predict_subtask all
```

### 4. Predict with BERT and Export in Competition Format

```bash
python insomnia_detection.py \
  --noteevents path/to/NOTEEVENTS.csv \
  --test_ids path/to/test_ids.txt \
  --model_dir models \
  --predict --use_bert --competition_format \
  --output_dir predictions
```

### 5. Full Pipeline (Train + Predict)

```bash
python insomnia_detection.py \
  --noteevents path/to/NOTEEVENTS.csv \
  --test_ids path/to/test_ids.txt \
  --subtask1_labels path/to/subtask1_labels.json \
  --subtask2a_labels path/to/subtask2a_labels.json \
  --subtask2b_labels path/to/subtask2b_labels.json \
  --model_dir models \
  --train --predict --predict_subtask all
```
