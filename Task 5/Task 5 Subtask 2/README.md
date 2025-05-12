# Task 5: Subtask 2 – Information Extraction from Food Safety Articles

## Code and File Structure

This subtask performs **sequence labeling** on food safety-related articles to extract key information such as product, organization, infection type, and location. The approach integrates:

* `RoBERTa-large` for token classification
* CRF decoding for better entity boundary modeling
* Adversarial training (FGM) for robustness
* Custom event-level evaluation (exact match)
* Token-level, chunk-level, and confusion matrix evaluations

```
Task 5 Subtask 2.py                    # Main training & evaluation script (CRF, adversarial trainer)
SMM4H-2025-Task5-Train_subtask2.csv    # Raw training data with gold annotations
SMM4H-2025-Task5-Validation_subtask2.csv # Raw validation data
prediction_task5_subtask2.xlsx         # Sample submission format
```

Entities include:

* `TARGET_ORG`, `PRODUCT`, `INFECTION`, `SAFETY_INCIDENT`, `AFFECTED_NUM`, `LOCATION`

---

## How to Use

### Step 1: Prepare Your Data

Ensure the CSV files:

* `SMM4H-2025-Task5-Train_subtask2.csv`
* `SMM4H-2025-Task5-Validation_subtask2.csv`
  are placed in the working directory.

### Step 2: Run the Model Training and Evaluation

```bash
python "Task 5 Subtask 2.py"
```

This script will:

* Load and preprocess the data
* Tokenize and align BIO tags
* Train a `roberta-large + CRF` model with adversarial training
* Perform evaluation on token-level, chunk-level, and event-level
* Output classification metrics and confusion matrices to the console

No additional preprocessing notebook is required for this subtask.
