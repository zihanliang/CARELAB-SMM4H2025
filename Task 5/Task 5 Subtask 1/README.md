# Task 5: Subtask 1 – Food Safety Article Classification with Enhanced "Neither" Detection

## Code and File Structure

This subtask classifies food safety-related news articles into three categories: `Food Recall`, `Foodborne Disease Outbreak`, and `Neither`. The code integrates advanced techniques for improving performance on the underrepresented "Neither" class.

```
Task 5 Subtask 1.py                     # Main training & evaluation script with advanced modeling
task5_subtask1_dataprocessing.ipynb    # Notebook for preprocessing and feature engineering
SMM4H-2025-Task5-Train_subtask1.csv    # Raw training data
SMM4H-2025-Task5-Validation_subtask1.csv # Raw validation data
fda_neither_augmented_100.json         # Augmented "Neither" samples (for oversampling)
prediction_task5_subtask1.xlsx         # Sample output format for predictions
```

Model techniques include:

* RoBERTa-large with enhanced classification heads
* Class-balanced focal loss
* Aggressive oversampling of "Neither"
* Rule-based post-processing
* Monte Carlo dropout for uncertainty estimation
* Multi-model ensemble prediction

---

## How to Use

### Step 1: Preprocess the Data

Open and run the full notebook:

```bash
task5_subtask1_dataprocessing.ipynb
```

This generates cleaned and augmented files (e.g., `preprocessed_SMM4H-2025-Task5-Train_subtask1_full.csv`) needed for model training.

### Step 2: Run the Main Script

Execute the enhanced training pipeline (including training, evaluation, and saving models):

```bash
python "Task 5 Subtask 1.py"
```

This will:

* Train the enhanced RoBERTa-large classifier
* Evaluate on validation set
* Save the model to `./enhanced_neither_model/`
* Print performance metrics with special emphasis on "Neither" class

### Optional: Run Full Ensemble Pipeline

If you want to train multiple models and apply post-processing:

```python
from Task_5_Subtask_1 import complete_neither_detection_pipeline
result_df = complete_neither_detection_pipeline(test_data, ensemble_size=3)
```

Make sure `test_data` is preprocessed in the same format as training data.
