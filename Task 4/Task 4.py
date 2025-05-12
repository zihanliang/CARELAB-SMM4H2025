import re
import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import warnings
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import nltk
import pandas as pd
from nltk.corpus import wordnet
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')

# Download the necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def load_test_ids(test_id_file):
    """Load a list of test ids from a file"""
    with open(test_id_file, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]
    return test_ids


def predict_insomnia_for_test(notes_df, test_ids, model_dir='models', subtask='all', use_bert=False, ground_truth=None):
    """
    The prediction function specially prepared for the test set only processes the notes with the specified ID.
    """
    # Filter out the notes corresponding to the test ID from notes_df
    test_df = notes_df[notes_df['ROW_ID'].isin([int(nid) for nid in test_ids])]

    predictions = predict_insomnia(test_df, model_dir, subtask, use_bert, ground_truth)
    return predictions


def save_competition_format(predictions, subtask, output_file):
    """Save the prediction results in the format specified by the competition."""
    result = {}

    if subtask == '1':
        # Subtask 1: -> {"Insomnia": "yes"/"no"}
        for note_id, pred in predictions.items():
            if "Insomnia" in pred:
                result[note_id] = {"Insomnia": pred["Insomnia"]}
            else:
                print(f"Warning: Note ID {note_id} missing 'Insomnia' prediction")
            # result[note_id] = {"Insomnia": pred["Insomnia"]}

    elif subtask == '2a':
        # Subtask 2A: Each note ID-> contains a dictionary of 5 rule judgments.
        for note_id, pred in predictions.items():
            result[note_id] = {
                "Definition 1": pred.get("Definition 1", "no"),
                "Definition 2": pred.get("Definition 2", "no"),
                "Rule A": pred.get("Rule A", "no"),
                "Rule B": pred.get("Rule B", "no"),
                "Rule C": pred.get("Rule C", "no")
            }

    elif subtask == '2b':
        # Subtask 2B: Each note ID-> contains a dictionary of 5 rules, judgments and evidences.
        for note_id, pred in predictions.items():
            result[note_id] = {}
            for rule in ["Definition 1", "Definition 2", "Rule A", "Rule B", "Rule C"]:
                if rule in pred:
                    label = "yes" if pred[rule] == "yes" else "no"
                    evidence = []
                    if label == "yes" and f"{rule}_evidence" in pred:
                        evidence_data = pred[f"{rule}_evidence"]
                        if isinstance(evidence_data, dict) and "text" in evidence_data:
                            evidence = evidence_data["text"]
                        else:
                            evidence = evidence_data

                    result[note_id][rule] = {
                        "label": label,
                        "text": evidence if label == "yes" else []
                    }

    # Save the results to a file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"The result has been saved to {output_file}")

# Define keywords and drugs in insomnia rules
def define_keywords():
    # Definition 1: Keywords for sleep difficulties
    difficulty_sleeping_keywords = [
        'insomnia', 'trouble sleeping', 'difficulty sleeping', 'can\'t sleep', 'unable to sleep',
        'difficulty falling asleep', 'wakes up', 'early waking', 'early awakening',
        'sleep disturbance', 'sleep disorder', 'sleep problem', 'poor sleep', 'restless sleep',
        'night awakening', 'awake at night', 'sleep initiation', 'sleep maintenance'
    ]

    # Definition 2: Keywords for daytime dysfunction
    daytime_impairment_keywords = [
        'fatigue', 'tired', 'malaise', 'exhausted', 'lethargic', 'drowsy', 'sleepy',
        'attention', 'concentration', 'memory problem', 'forgetful',
        'performance', 'irritable', 'irritability', 'mood disturbance', 'agitated',
        'decreased motivation', 'lack of energy', 'low energy', 'no initiative',
        'accident', 'error', 'mistake', 'dissatisfaction with sleep', 'sleep concern',
        'daytime sleepiness', 'hyperactive', 'impulsive', 'aggressive'
    ]

    # Rule B: List of Main Insomnia Drugs
    primary_insomnia_meds = [
        'estazolam', 'eszopiclone', 'flurazepam', 'lemborexant', 'quazepam',
        'ramelteon', 'suvorexant', 'temazepam', 'triazolam', 'zaleplon', 'zolpidem',
        'lunesta', 'ambien', 'dalmane', 'dayvigo', 'doral', 'rozerem', 'belsomra',
        'restoril', 'halcion', 'sonata', 'edluar', 'intermezzo'
    ]

    # Rule C: List of Secondary Insomnia Drugs
    secondary_insomnia_meds = [
        'acamprosate', 'alprazolam', 'clonazepam', 'clonidine', 'diazepam',
        'diphenhydramine', 'doxepin', 'gabapentin', 'hydroxyzine', 'lorazepam',
        'melatonin', 'mirtazapine', 'olanzapine', 'quetiapine', 'trazodone',
        'xanax', 'klonopin', 'valium', 'benadryl', 'silenor', 'neurontin',
        'atarax', 'vistaril', 'ativan', 'remeron', 'zyprexa', 'seroquel'
    ]

    return (difficulty_sleeping_keywords, daytime_impairment_keywords,
            primary_insomnia_meds, secondary_insomnia_meds)


# Text preprocessing function
def preprocess_text(text):
    text = text.lower()

    # Use spaCy for processing
    doc = nlp(text[:100000])  # Limit the processing length to prevent memory overflow

    # Remove stop words and non-alphabetic characters, and perform word shape reduction.
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    return ' '.join(tokens)


# Feature Extraction Function
def extract_features(text):
    """
        Extracting features about insomnia from texts
    """
    features = {}
    difficulty_sleeping_kw, daytime_impairment_kw, primary_meds, secondary_meds = define_keywords()

    # Check whether the text contains all kinds of keywords.
    features['has_difficulty_sleeping'] = int(contains_keywords(text, difficulty_sleeping_kw))
    features['has_daytime_impairment'] = int(contains_keywords(text, daytime_impairment_kw))
    features['has_primary_meds'] = int(contains_keywords(text, primary_meds))
    features['has_secondary_meds'] = int(contains_keywords(text, secondary_meds))

    # Generate features according to rules
    features['meets_rule_a'] = int(features['has_difficulty_sleeping'] and features['has_daytime_impairment'])
    features['meets_rule_c'] = int((features['has_difficulty_sleeping'] or features['has_daytime_impairment'])
                                   and features['has_secondary_meds'])

    # Add keyword frequency characteristics
    features['difficulty_sleeping_count'] = count_keywords(text, difficulty_sleeping_kw)
    features['daytime_impairment_count'] = count_keywords(text, daytime_impairment_kw)
    features['primary_meds_count'] = count_keywords(text, primary_meds)
    features['secondary_meds_count'] = count_keywords(text, secondary_meds)

    return features


# Calculate the number of keyword occurrences
def count_keywords(text, keywords):
    """
    Calculate the total number of occurrences of keywords in the text.
    """
    text_lower = text.lower()
    count = 0
    for keyword in keywords:
        count += len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
    return count


# Check whether the text contains specific keywords.
def contains_keywords(text, keywords):
    """
    Check whether the text contains any keywords in the keyword list.

    Parameters:
    Text (str): the text to check.
    Keywords (list): keyword list

    Return:
    Bool: True if any keywords are found.
    """
    text_lower = text.lower()
    for keyword in keywords:
        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower):
            return True
    return False


# Apply the insomnia rule
def apply_insomnia_rules(text):
    """
    Applying Insomnia Rules to Classify Clinical Notes

    Parameters:
    Text (str): clinical note text.

    Return:
    Dict: a dictionary containing the application results of each rule.
    """
    # Get a list of keywords and drugs
    difficulty_sleeping_kw, daytime_impairment_kw, primary_meds, secondary_meds = define_keywords()

    # Check whether the text contains keywords of definition 1 and definition 2.
    has_difficulty_sleeping = contains_keywords(text, difficulty_sleeping_kw)
    has_daytime_impairment = contains_keywords(text, daytime_impairment_kw)

    # Application Rule A: Both Definition 1 and Definition 2 are required.
    rule_a = has_difficulty_sleeping and has_daytime_impairment

    # Application Rule B: Prescription contains main insomnia drugs.
    rule_b = contains_keywords(text, primary_meds)

    # Application Rule C: The prescription contains minor insomnia drugs and has symptoms defined as 1 or 2.
    rule_c = (has_difficulty_sleeping or has_daytime_impairment) and contains_keywords(text, secondary_meds)

    # Returns the result of each rule
    result = {
        "Definition 1": 1 if has_difficulty_sleeping else 0,
        "Definition 2": 1 if has_daytime_impairment else 0,
        "Rule A": 1 if rule_a else 0,
        "Rule B": 1 if rule_b else 0,
        "Rule C": 1 if rule_c else 0,
        "Insomnia": 1 if (rule_a or rule_b or rule_c) else 0
    }

    return result


def load_labels(annotation_files):
    """
    Load tags from annotation files

    Parameters:
    Annotation_files (dict): a dictionary containing subtasks and corresponding file paths.

    Return:
    Dict: a dictionary containing all labels.
    """
    labels = {}


    if 'subtask1' in annotation_files and annotation_files['subtask1'] and os.path.exists(annotation_files['subtask1']):
        with open(annotation_files['subtask1'], 'r') as f:
            task1_data = json.load(f)
            for note_id, data in task1_data.items():
                if note_id not in labels:
                    labels[note_id] = {}
                labels[note_id]['Insomnia'] = 1 if data['Insomnia'].lower() == 'yes' else 0


    if 'subtask2a' in annotation_files and annotation_files['subtask2a'] and os.path.exists(
            annotation_files['subtask2a']):
        with open(annotation_files['subtask2a'], 'r') as f:
            task2a_data = json.load(f)
            for note_id, data in task2a_data.items():
                if note_id not in labels:
                    labels[note_id] = {}
                for key in ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']:
                    labels[note_id][key] = 1 if data[key].lower() == 'yes' else 0
                if 'Insomnia' not in labels[note_id]:
                    labels[note_id]['Insomnia'] = 1 if (labels[note_id]['Rule A'] == 1 or
                                                        labels[note_id]['Rule B'] == 1 or
                                                        labels[note_id]['Rule C'] == 1) else 0

    if 'subtask2b' in annotation_files and annotation_files['subtask2b'] and os.path.exists(
            annotation_files['subtask2b']):
        with open(annotation_files['subtask2b'], 'r') as f:
            task2b_data = json.load(f)

            for note_id, data in task2b_data.items():
                if note_id not in labels:
                    labels[note_id] = {}

                for key in ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']:
                    if key in data:
                        if 'text' in data[key]:
                            if key not in labels[note_id]:
                                labels[note_id][key] = 1 if data[key]['label'].lower() == 'yes' else 0

                            labels[note_id][f"{key}_evidence"] = data[key]['text']

    if not labels and 'subtask2a' in annotation_files and annotation_files['subtask2a']:
        print("Warning: No label file was found. Rules will be used to generate pseudo labels. ")
        return labels

    return labels


def prepare_training_data(notes_df, labels, subtask='1'):
    """
    Prepare training data

    Parameters:
    Notes_df (DataFrame): a DataFrame containing notes.
    Labels (dict): the mapping from note ID to label.
    Subtask (str): subtask ID ('1', '2a', '2b')

    Return:
    Tuple: (feature, label)
    """
    X = []
    y = []

    for idx, row in notes_df.iterrows():
        note_id = str(row['ROW_ID'])

        if note_id in labels:
            text = row['TEXT']

            features = extract_features(text)

            X.append({
                'text': text,
                'features': features
            })

            if subtask == '1':
                if 'Insomnia' in labels[note_id]:
                    y.append(labels[note_id]['Insomnia'])
                else:
                    has_rule_a = 'Rule A' in labels[note_id] and labels[note_id]['Rule A'] == 1
                    has_rule_b = 'Rule B' in labels[note_id] and labels[note_id]['Rule B'] == 1
                    has_rule_c = 'Rule C' in labels[note_id] and labels[note_id]['Rule C'] == 1

                    y.append(1 if (has_rule_a or has_rule_b or has_rule_c) else 0)

            elif subtask == '2a':
                label_vector = []
                for key in ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']:
                    if key in labels[note_id]:
                        label_vector.append(labels[note_id][key])
                    else:
                        if key == 'Rule A':
                            has_def1 = 'Definition 1' in labels[note_id] and labels[note_id]['Definition 1'] == 1
                            has_def2 = 'Definition 2' in labels[note_id] and labels[note_id]['Definition 2'] == 1
                            label_vector.append(1 if (has_def1 and has_def2) else 0)
                        elif key == 'Rule B':
                            label_vector.append(1 if features['has_primary_meds'] else 0)
                        elif key == 'Rule C':
                            has_any_def = (
                                        ('Definition 1' in labels[note_id] and labels[note_id]['Definition 1'] == 1) or
                                        ('Definition 2' in labels[note_id] and labels[note_id]['Definition 2'] == 1))
                            label_vector.append(1 if (has_any_def and features['has_secondary_meds']) else 0)
                        else:
                            # Definition 1 and Definition 2
                            if key == 'Definition 1':
                                label_vector.append(1 if features['has_difficulty_sleeping'] else 0)
                            else:  # Definition 2
                                label_vector.append(1 if features['has_daytime_impairment'] else 0)

                if len(label_vector) == 5:
                    y.append(label_vector)

            elif subtask == '2b':

                evidence_dict = {}
                for key in ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']:
                    evidence_key = f"{key}_evidence"
                    if evidence_key in labels[note_id]:
                        evidence_dict[key] = {
                            "label": "yes" if labels[note_id][key] == 1 else "no",
                            "text": labels[note_id][evidence_key]
                        }
                    else:
                        if key == 'Definition 1':
                            keywords = define_keywords()[0]  # Sleep difficulty keywords
                        elif key == 'Definition 2':
                            keywords = define_keywords()[1]  # daytime dysfunction keywords
                        elif key == 'Rule B':
                            keywords = define_keywords()[2]  # Main insomnia drugs
                        elif key == 'Rule C':
                            keywords = define_keywords()[3]  # Secondary insomnia drugs
                        else:  # Rule A
                            keywords = define_keywords()[0] + define_keywords()[1]

                        evidence = extract_evidence(text, key, keywords)
                        evidence_dict[key] = {
                            "label": "yes" if (key in labels[note_id] and labels[note_id][key] == 1) else "no",
                            "text": evidence
                        }

                y.append(evidence_dict)

    if subtask in ['1', '2a']:
        return X, np.array(y)
    else:  # subtask == '2b'
        return X, y


# Analyze the drug information in the notes
def analyze_drugs(text):
    """
    Analyze the drug information in the notes
    """
    _, _, primary_meds, secondary_meds = define_keywords()
    all_meds = primary_meds + secondary_meds

    found_meds = []
    for med in all_meds:
        if re.search(r'\b' + re.escape(med) + r'\b', text.lower()):
            found_meds.append(med)

    return found_meds


def transform_features(X):
    """
    Convert original features into model input format.
    """
    texts = [item['text'] for item in X]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=3
    )
    text_features = vectorizer.fit_transform(texts)

    manual_features = np.array([[
        item['features']['has_difficulty_sleeping'],
        item['features']['has_daytime_impairment'],
        item['features']['has_primary_meds'],
        item['features']['has_secondary_meds'],
        item['features']['meets_rule_a'],
        item['features']['meets_rule_c'],
        item['features']['difficulty_sleeping_count'],
        item['features']['daytime_impairment_count'],
        item['features']['primary_meds_count'],
        item['features']['secondary_meds_count']
    ] for item in X])

    return text_features, manual_features, vectorizer


def extract_evidence(text, label, keywords, context_window=100):
    """
    Extracting evidence supporting labels from text

    Parameters:
    Text (str): source text.
    Label (str): label name.
    Keywords (list): keywords related to this tag.
    Context_window (int): how many characters around the keyword are extracted as the context.

    Return:
    List: a list of textual evidence
    """
    evidence = []
    text_lower = text.lower()

    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.finditer(pattern, text_lower)

        for match in matches:
            start = max(0, match.start() - context_window)
            end = min(len(text), match.end() + context_window)

            sentence_start = text.rfind('.', 0, match.start())
            sentence_start = max(start, sentence_start + 1 if sentence_start != -1 else start)

            sentence_end = text.find('.', match.end())
            sentence_end = min(end, sentence_end + 1 if sentence_end != -1 else end)

            evidence_text = text[sentence_start:sentence_end].strip()
            evidence.append(evidence_text)

    evidence = list(set(evidence))
    evidence = [e for e in evidence if len(e) > 10]

    return evidence


# Text augmentation for medical notes
def augment_text(text, n=2):
    """
    Create n augmented versions of the input text using synonym replacement
    """
    augmented_texts = [text]
    words = text.split()

    # Only try to augment if we have enough words
    if len(words) < 10:
        return augmented_texts

    # Try to generate n augmented versions
    for _ in range(n):
        aug_words = words.copy()

        # Replace random words with synonyms (up to 20% of words)
        num_to_replace = max(1, int(len(words) * 0.2))
        indices_to_replace = np.random.choice(range(len(words)), num_to_replace, replace=False)

        for idx in indices_to_replace:
            word = words[idx]
            # Try to find synonyms
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.append(lemma.name().replace('_', ' '))

            # If synonyms found, replace the word
            if synonyms:
                aug_words[idx] = np.random.choice(synonyms)

        augmented_texts.append(' '.join(aug_words))

    return augmented_texts


# Modified prepare_training_data with augmentation
def prepare_training_data_with_augmentation(notes_df, labels, subtask='1', augment=True):
    """
    Prepare training data with optional augmentation
    """
    X = []
    y = []

    for idx, row in notes_df.iterrows():
        note_id = str(row['ROW_ID'])

        if note_id in labels:
            # Extract text
            text = row['TEXT']

            # Extract features
            features = extract_features(text)

            # Store original data
            X.append({
                'text': text,
                'features': features
            })

            if subtask == '1':
                if 'Insomnia' in labels[note_id]:
                    y.append(labels[note_id]['Insomnia'])
                else:
                    # If there is no Insomnia tag, use rules to infer.
                    has_rule_a = 'Rule A' in labels[note_id] and labels[note_id]['Rule A'] == 1
                    has_rule_b = 'Rule B' in labels[note_id] and labels[note_id]['Rule B'] == 1
                    has_rule_c = 'Rule C' in labels[note_id] and labels[note_id]['Rule C'] == 1

                    y.append(1 if (has_rule_a or has_rule_b or has_rule_c) else 0)

            elif subtask == '2a':
                # For each rule label, set it to 0 if it does not exist.
                label_vector = []
                for key in ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']:
                    if key in labels[note_id]:
                        label_vector.append(labels[note_id][key])
                    else:
                        if key == 'Rule A':
                            has_def1 = 'Definition 1' in labels[note_id] and labels[note_id]['Definition 1'] == 1
                            has_def2 = 'Definition 2' in labels[note_id] and labels[note_id]['Definition 2'] == 1
                            label_vector.append(1 if (has_def1 and has_def2) else 0)
                        elif key == 'Rule B':
                            label_vector.append(1 if features['has_primary_meds'] else 0)
                        elif key == 'Rule C':
                            has_any_def = (
                                        ('Definition 1' in labels[note_id] and labels[note_id]['Definition 1'] == 1) or
                                        ('Definition 2' in labels[note_id] and labels[note_id]['Definition 2'] == 1))
                            label_vector.append(1 if (has_any_def and features['has_secondary_meds']) else 0)
                        else:
                            # Definition 1 和 Definition 2
                            if key == 'Definition 1':
                                label_vector.append(1 if features['has_difficulty_sleeping'] else 0)
                            else:  # Definition 2
                                label_vector.append(1 if features['has_daytime_impairment'] else 0)

                if len(label_vector) == 5:
                    y.append(label_vector)

            # Add augmented samples if needed
            if augment:
                augmented_texts = augment_text(text, n=2)  # Generate 2 augmented versions

                # Skip first one as it's the original text
                for aug_text in augmented_texts[1:]:
                    aug_features = extract_features(aug_text)

                    # Add augmented sample
                    X.append({
                        'text': aug_text,
                        'features': aug_features
                    })

                    if subtask == '1':
                        if 'Insomnia' in labels[note_id]:
                            y.append(labels[note_id]['Insomnia'])
                        else:
                            has_rule_a = 'Rule A' in labels[note_id] and labels[note_id]['Rule A'] == 1
                            has_rule_b = 'Rule B' in labels[note_id] and labels[note_id]['Rule B'] == 1
                            has_rule_c = 'Rule C' in labels[note_id] and labels[note_id]['Rule C'] == 1

                            y.append(1 if (has_rule_a or has_rule_b or has_rule_c) else 0)

                    elif subtask == '2a':
                        if len(label_vector) == 5:
                            y.append(label_vector)

    return X, np.array(y)


def predict_insomnia(notes_df, model_dir='models', subtask='all', use_bert=False, ground_truth=None):
    """
    Predicting insomnia by using the saved model

    Parameters:
    Notes_df (DataFrame): a DataFrame containing notes.
    Model_dir (str): model directory
    Subtask (str): subtask ID ('1', '2a', '2b', 'all')
    Use_bert (bool): whether to use BERT model to predict subtask 1.

    Return:
    Dict: mapping of id to forecast
    """
    try:
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
    except:
        print(f"Warning: No vectorizer found in {model_dir}. A new vector machine will be created. " )
        # If there is no pre-trained vectorizer, create a new one.
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=3
        )
        # Here we use all the notes to fit the vector machine.
        vectorizer.fit([row['TEXT'] for _, row in notes_df.iterrows()])

    predictions = {}
    subtasks_to_run = []

    # Identify subtasks to run
    if subtask == 'all':
        subtasks_to_run = ['1', '2a', '2b']
    else:
        subtasks_to_run = [subtask]

    X = []
    note_ids = []

    print(f"processing {len(notes_df)} notes ...")
    for idx, row in notes_df.iterrows():
        note_id = str(row['ROW_ID'])
        text = row['TEXT']
        features = extract_features(text)

        X.append({
            'text': text,
            'features': features
        })
        note_ids.append(note_id)

    texts = [item['text'] for item in X]
    text_features = vectorizer.transform(texts)

    manual_features = np.array([[
        item['features']['has_difficulty_sleeping'],
        item['features']['has_daytime_impairment'],
        item['features']['has_primary_meds'],
        item['features']['has_secondary_meds'],
        item['features']['meets_rule_a'],
        item['features']['meets_rule_c'],
        item['features']['difficulty_sleeping_count'],
        item['features']['daytime_impairment_count'],
        item['features']['primary_meds_count'],
        item['features']['secondary_meds_count']
    ] for item in X])

    X_combined = np.hstack((text_features.toarray(), manual_features))

    # subtask 1: Binary Classification
    if '1' in subtasks_to_run:
        print("Execute subtask 1 forecast ...")
        # Try to use the BERT model (if available and specified)
        if use_bert and os.path.exists(os.path.join(model_dir, "bert_classifier")):
            try:
                from transformers import pipeline
                classifier = pipeline("text-classification", model=os.path.join(model_dir, "bert_classifier"))

                # Use BERT prediction
                bert_predictions = []
                for text in texts:
                    truncated_text = text[:512 * 3]
                    result = classifier(truncated_text)[0]
                    bert_predictions.append(1 if result['label'] == 'LABEL_1' else 0)

                # Convert BERT prediction results into output format
                for i, note_id in enumerate(note_ids):
                    if note_id not in predictions:
                        predictions[note_id] = {}
                    predictions[note_id]["Insomnia"] = "yes" if bert_predictions[i] == 1 else "no"
                print("use BERT model to predict the completion of subtask 1")
            except Exception as e:
                print(f"Error in using BERT prediction: {e}")
                print("Back to the traditional machine learning model ...")
                use_bert = False


        # If BERT is not used or BERT fails, the traditional machine learning model is used.
        if not use_bert:
            try:
                with open(os.path.join(model_dir, 'subtask1_model.pkl'), 'rb') as f:
                    model = pickle.load(f)

                y_pred = model.predict(X_combined)

                for i, note_id in enumerate(note_ids):
                    if note_id not in predictions:
                        predictions[note_id] = {}
                    predictions[note_id]["Insomnia"] = "yes" if y_pred[i] == 1 else "no"

                print("Predicting the completion of subtask 1 with traditional model")
            except Exception as e:
                print(f"Error in predicting subtask 1: {e}")
                print ("Use the rule method to predict ...")
                for i, note_id in enumerate(note_ids):
                    if note_id not in predictions:
                        predictions[note_id] = {}
                    rules = apply_insomnia_rules(texts[i])
                    predictions[note_id]["Insomnia"] = "yes" if rules["Insomnia"] == 1 else "no"

    # subtask 2A: Multi-label classification
    if '2a' in subtasks_to_run:
        print("Execute subtask 2A prediction ...")
        try:
            with open(os.path.join(model_dir, 'subtask2a_models.pkl'), 'rb') as f:
                models = pickle.load(f)

            # Try to load the label name, if it doesn't exist, use the default value.
            try:
                with open(os.path.join(model_dir, 'label_names.pkl'), 'rb') as f:
                    labels = pickle.load(f)
            except:
                labels = ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']

            for i, note_id in enumerate(note_ids):
                if note_id not in predictions:
                    predictions[note_id] = {}

                for j, label in enumerate(labels):
                    if j < len(models):  # Make sure we don't go beyond the model list.
                        pred = models[j].predict(X_combined[i:i + 1])[0]
                        predictions[note_id][label] = "yes" if pred == 1 else "no"

            print("subtask 2A is predicted to be completed")
        except Exception as e:
            print(f"Error in predicting subtask 2A: {e}")
            print("Use the rule method to predict ...")
            # If the model is not available, use the rule method.
            for i, note_id in enumerate(note_ids):
                if note_id not in predictions:
                    predictions[note_id] = {}
                rules = apply_insomnia_rules(texts[i])
                for key in ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']:
                    predictions[note_id][key] = "yes" if rules[key] == 1 else "no"

    # subtask 2B: Evidence-based classification
    if '2b' in subtasks_to_run:
        print("Execute subtask 2B forecast ...")

        if all(note_id in predictions and 'Definition 1' in predictions[note_id] for note_id in note_ids):
            print("Extract evidence based on the results of subtask 2A ...")
        else:
            print("Generate labels for subtask 2B using rule method ...")
            # Apply rules to get labels
            for i, note_id in enumerate(note_ids):
                if note_id not in predictions:
                    predictions[note_id] = {}
                rules = apply_insomnia_rules(texts[i])
                for key in ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']:
                    predictions[note_id][key] = "yes" if rules[key] == 1 else "no"

        # Now extract evidence for each tag
        evidence_labels = ['Definition 1', 'Definition 2', 'Rule B', 'Rule C']

        difficulty_sleeping_kw, daytime_impairment_kw, primary_meds, secondary_meds = define_keywords()
        keyword_maps = {
            'Definition 1': difficulty_sleeping_kw,
            'Definition 2': daytime_impairment_kw,
            'Rule B': primary_meds,
            'Rule C': secondary_meds
        }

        # Store the prediction result of 2B
        predictions_2b = {}

        for i, note_id in enumerate(note_ids):
            predictions_2b[note_id] = {}

            for label in evidence_labels:
                if note_id in predictions and label in predictions[note_id]:
                    is_positive = predictions[note_id][label] == "yes"

                    if is_positive:

                        evidence = extract_evidence(texts[i], label, keyword_maps[label])

                        predictions_2b[note_id][label] = {
                            "label": "yes",
                            "text": evidence[:5]
                        }
                    else:
                        predictions_2b[note_id][label] = {
                            "label": "no",
                            "text": []
                        }

            if (note_id in predictions and
                    'Rule A' in predictions[note_id] and
                    predictions[note_id]['Rule A'] == "yes"):

                combined_kw = difficulty_sleeping_kw + daytime_impairment_kw
                evidence = extract_evidence(texts[i], 'Rule A', combined_kw)

                predictions_2b[note_id]['Rule A'] = {
                    "label": "yes",
                    "text": evidence[:5]
                }
            else:
                predictions_2b[note_id]['Rule A'] = {
                    "label": "no",
                    "text": []
                }

        for note_id, preds in predictions_2b.items():
            if subtask == '2b':
                predictions[note_id] = preds
            else:
                for key, value in preds.items():
                    predictions[note_id][key + "_evidence"] = value

        print("subtask 2B is predicted to be completed")
    if ground_truth is not None:
        print("\nEvaluation against ground truth:")
        print("--------------------------------")

        # Evaluate subtask 1 if predicted
        if '1' in subtasks_to_run:
            y_true = []
            y_pred = []

            for note_id in note_ids:
                if note_id in ground_truth and 'Insomnia' in ground_truth[note_id]:
                    y_true.append(1 if ground_truth[note_id]['Insomnia'] == 1 else 0)
                    y_pred.append(1 if predictions[note_id]['Insomnia'] == 'yes' else 0)

            if y_true and y_pred:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )
                print(f"\nSubtask 1 results:")
                print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
                print(f"Classification Report:\n{classification_report(y_true, y_pred, zero_division=0)}")

        # Evaluate subtask 2a if predicted
        if '2a' in subtasks_to_run:
            print("\nSubtask 2a results:")
            label_names = ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']

            label_metrics = {}
            all_f1 = []

            for label in label_names:
                y_true = []
                y_pred = []

                for note_id in note_ids:
                    if note_id in ground_truth and label in ground_truth[note_id]:
                        y_true.append(1 if ground_truth[note_id][label] == 1 else 0)
                        y_pred.append(1 if predictions[note_id][label] == 'yes' else 0)

                if y_true and y_pred:
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='binary', zero_division=0
                    )
                    label_metrics[label] = {'precision': precision, 'recall': recall, 'f1': f1}
                    all_f1.append(f1)
                    print(f"{label}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            if all_f1:
                print(f"Average F1 Score across all labels: {sum(all_f1) / len(all_f1):.4f}")

        # Evaluate subtask 2b if predicted (for labels only, evidence is harder to evaluate automatically)
        if '2b' in subtasks_to_run:
            print("\nSubtask 2b results (label prediction only):")
            label_names = ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']

            label_metrics = {}
            all_f1 = []

            for label in label_names:
                y_true = []
                y_pred = []

                for note_id in note_ids:
                    if (note_id in ground_truth and
                            label in ground_truth[note_id] and
                            note_id in predictions_2b and
                            label in predictions_2b[note_id]):
                        y_true.append(1 if ground_truth[note_id][label] == 1 else 0)
                        y_pred.append(1 if predictions_2b[note_id][label]['label'] == 'yes' else 0)

                if y_true and y_pred:
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='binary', zero_division=0
                    )
                    label_metrics[label] = {'precision': precision, 'recall': recall, 'f1': f1}
                    all_f1.append(f1)
                    print(f"{label}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            if all_f1:
                print(f"Average F1 Score across all labels: {sum(all_f1) / len(all_f1):.4f}")

            print("\nNote: Evidence extraction quality requires manual evaluation")

    return predictions


# Cross-validation for small datasets
def train_models_with_cv(X, y, model_dir='models', n_splits=5):
    """
    Train models using cross-validation for better evaluation on small datasets
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    text_features, manual_features, vectorizer = X

    # Training for subtask 1 (binary classification)
    if '1' in y:
        # Combine features
        X_combined = np.hstack((text_features.toarray(), manual_features))
        y_labels = y['1']

        # Use SMOTE to handle class imbalance
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_combined, y_labels)
            print(f"Applied SMOTE: Original shape {X_combined.shape}, Resampled shape {X_resampled.shape}")
            X_combined = X_resampled
            y_labels = y_resampled
        except ValueError as e:
            print(f"Could not apply SMOTE due to: {e}")
            print("Continuing with original data...")

        # Set up cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Models to evaluate
        models = {
            'logistic': LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear'),
            'svm': CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=2000)),
            'rf': RandomForestClassifier(n_estimators=200, class_weight='balanced', min_samples_leaf=1),
            'xgb': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                scale_pos_weight=sum(y_labels == 0) / max(sum(y_labels == 1), 1),
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }

        best_model = None
        best_score = 0
        best_model_name = ""

        print("\nCross-validation results for subtask 1:")
        print("----------------------------------------")

        for name, model in models.items():
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_combined, y_labels,
                cv=cv,
                scoring=make_scorer(f1_score)
            )

            mean_score = cv_scores.mean()
            std_score = cv_scores.std()

            print(f"Model: {name}, F1 Score: {mean_score:.4f} (±{std_score:.4f})")

            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name

        # Train final model on all data
        print(f"\nTraining final model ({best_model_name}) on all data...")
        final_model = models[best_model_name].fit(X_combined, y_labels)

        # Save best model and vectorizer
        with open(os.path.join(model_dir, 'subtask1_model.pkl'), 'wb') as f:
            pickle.dump(final_model, f)

        print(f"Subtask 1 - Best model: {best_model_name}, F1 Score: {best_score:.4f}")

    # Training for subtask 2A (multi-label classification)
    if '2a' in y:
        # Combine features
        X_combined = np.hstack((text_features.toarray(), manual_features))
        y_multilabel = y['2a']

        # Set up cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Train a model for each label
        models = []
        label_names = ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']
        label_f1_scores = []  # Store F1 scores for each label

        print("\nCross-validation results for subtask 2a:")
        print("----------------------------------------")

        for i in range(5):  # 5 different labels
            print(f"\nTraining classifier for {label_names[i]}:")

            # Get label column
            y_col = y_multilabel[:, i]
            unique_classes = np.unique(y_col)
            if len(unique_classes) == 1:
                print(f"Warning: All samples for {label_names[i]} have the same label ({unique_classes[0]})")
                print(f"  > Creating a DummyClassifier that always predicts {unique_classes[0]}")

                from sklearn.dummy import DummyClassifier
                best_model = DummyClassifier(strategy="constant", constant=unique_classes[0])
                best_model.fit(X_combined[:1], y_col[:1])
                models.append(best_model)
                # Add F1 score
                if unique_classes[0] == 1:
                    # If all are positive, F1 is 1.0 (perfect)
                    label_f1_scores.append(1.0)
                else:
                    # If all are negative, F1 can't be calculated (we'll use 0)
                    label_f1_scores.append(0.0)
                continue

            # Try SMOTE for this label if imbalanced
            try:
                if sum(y_col == 1) >= 5 and sum(y_col == 0) >= 5:  # Need at least 5 samples per class
                    smote = SMOTE(random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X_combined, y_col)
                    print(
                        f"Applied SMOTE: {sum(y_col == 1)}/{len(y_col)} positive samples → {sum(y_resampled == 1)}/{len(y_resampled)}")
                    X_label = X_resampled
                    y_label = y_resampled
                else:
                    X_label = X_combined
                    y_label = y_col
            except Exception as e:
                print(f"Could not apply SMOTE due to: {e}")
                X_label = X_combined
                y_label = y_col

            # Calculate class weights
            pos_weight = sum(y_label == 0) / max(sum(y_label == 1), 1)

            model_candidates = {
                'logistic': LogisticRegression(class_weight='balanced', max_iter=2000, solver='liblinear'),
                'svm': CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=2000)),
                'rf': RandomForestClassifier(n_estimators=200, class_weight='balanced', min_samples_leaf=1),
                'xgb': xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    scale_pos_weight=pos_weight,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            }

            best_model = None
            best_score = 0
            best_model_name = ""

            for name, model in model_candidates.items():
                try:
                    # Handle case where all labels are the same
                    if len(np.unique(y_label)) == 1:
                        print(f"Warning: All samples for {label_names[i]} have the same label ({y_label[0]})")
                        dummy_score = 1.0 if y_label[
                                                 0] == 1 else 0.0  # Perfect score for all positives, 0 for all negatives
                        print(f"  - {name} F1 score: {dummy_score:.4f} (dummy score - all samples same class)")

                        # If all labels are positive, the classifier should always predict positive
                        if y_label[0] == 1 and best_score < dummy_score:
                            from sklearn.dummy import DummyClassifier
                            best_model = DummyClassifier(strategy="constant", constant=1)
                            best_model.fit(X_label, y_label)
                            best_score = dummy_score
                            best_model_name = f"{name} (dummy - predict all 1)"
                        continue

                    # Cross-validation for this model
                    cv_scores = cross_val_score(
                        model, X_label, y_label,
                        cv=min(n_splits, sum(y_label == 1)),  # Can't have more splits than positive samples
                        scoring=make_scorer(f1_score)
                    )

                    mean_score = cv_scores.mean()
                    std_score = cv_scores.std()

                    print(f"  - {name} F1 score: {mean_score:.4f} (±{std_score:.4f})")

                    if mean_score > best_score:
                        best_score = mean_score
                        best_model_name = name
                except Exception as e:
                    print(f"  - {name} failed: {e}")

            # Train final model on all data
            if best_model is None:
                print(f"  > No suitable model found for {label_names[i]}, using Logistic Regression")
                best_model = LogisticRegression(class_weight='balanced', max_iter=2000, solver='liblinear')
                best_model_name = "logistic (fallback)"
                # Fit model to calculate performance
                best_model.fit(X_label, y_label)
                try:
                    # Cross-validation to get performance estimate
                    cv_scores = cross_val_score(
                        best_model, X_label, y_label,
                        cv=min(3, sum(y_label == 1)),  # Use at most 3 folds for fallback
                        scoring=make_scorer(f1_score)
                    )
                    best_score = cv_scores.mean()
                except:
                    best_score = 0.0  # If we can't calculate, assume 0
            else:
                print(f"  > Selected {best_model_name} as best model for {label_names[i]} (F1: {best_score:.4f})")
                if "dummy" not in best_model_name:
                    best_model = model_candidates[best_model_name]

            # Fit on all data
            best_model.fit(X_label, y_label)
            models.append(best_model)
            label_f1_scores.append(best_score)

        # Print overall performance summary for subtask 2a
        print("\nSubtask 2a - Performance Summary:")
        print("----------------------------------")
        for i, label in enumerate(label_names):
            print(f"{label}: F1 Score = {label_f1_scores[i]:.4f}")

        # Calculate and print the average F1 score across all labels
        average_f1 = sum(label_f1_scores) / len(label_f1_scores)
        print(f"Average F1 Score: {average_f1:.4f}")

        # Save models
        with open(os.path.join(model_dir, 'subtask2a_models.pkl'), 'wb') as f:
            pickle.dump(models, f)

        # Save label names
        with open(os.path.join(model_dir, 'label_names.pkl'), 'wb') as f:
            pickle.dump(label_names, f)

    # Process subtask 2B (if necessary)
    if '2b' in y:
        label_names = ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']

        print("\nEvaluation for subtask 2b:")
        print("--------------------------")

        # For subtask 2b, we'll evaluate both the label prediction (using 2a models)
        # and evidence extraction quality

        # First, test if 2a models exist and can be used for 2b
        if '2a' in y and len(models) == 5:
            print("Using models from subtask 2a for label prediction in subtask 2b")

            # Copy performance metrics from subtask 2a for the label prediction part
            print("\nLabel prediction performance (from subtask 2a models):")
            for i, label in enumerate(label_names):
                print(f"{label}: F1 Score = {label_f1_scores[i]:.4f}")
            print(f"Average Label Prediction F1 Score: {average_f1:.4f}")

            # Add notes about evidence extraction evaluation
            print("\nEvidence extraction quality:")
            print("Evidence extraction is evaluated based on keyword matching in clinical notes.")
            print("This rule-based approach has high precision but may miss contextual evidence.")

            # Save the model
            with open(os.path.join(model_dir, 'subtask2b_models.pkl'), 'wb') as f:
                if os.path.exists(os.path.join(model_dir, 'subtask2a_models.pkl')):
                    with open(os.path.join(model_dir, 'subtask2a_models.pkl'), 'rb') as f2a:
                        models_2a = pickle.load(f2a)
                    pickle.dump(models_2a, f)
                    print("Models from subtask 2a saved for use in subtask 2b")
                else:
                    print(
                        "The model of subtask 2A does not exist, and a rule-based model will be created for subtask 2B")
                    pickle.dump([], f)
        else:
            print("No models from subtask 2a available, using rule-based approach for subtask 2b")
            print("Rule-based approach performance cannot be directly quantified through cross-validation")

            # Save empty list as placeholder
            with open(os.path.join(model_dir, 'subtask2b_models.pkl'), 'wb') as f:
                pickle.dump([], f)

    # Save vectorizer
    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    print("\nModel training and saving complete.")


# training BiLSTM model
def train_bilstm_model(X, y, model_dir='models'):
    """
    Train a BiLSTM model for text classification
    """
    # Extract text from X
    texts = [item['text'][:10000] for item in X]  # Limit text length

    # Tokenize text
    max_words = 10000  # Maximum vocabulary size
    max_len = 500  # Maximum sequence length

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences
    X_padded = pad_sequences(sequences, maxlen=max_len)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_padded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build model
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(max_len, 1)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'bilstm_model.h5'),
        save_best_only=True,
        monitor='val_loss'
    )

    # Reshape input for LSTM (samples, time steps, features)
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # Train model
    try:
        history = model.fit(
            X_train_reshaped, y_train,
            epochs=20,
            batch_size=16,
            validation_data=(X_val_reshaped, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        # Save tokenizer
        with open(os.path.join(model_dir, 'bilstm_tokenizer.pkl'), 'wb') as f:
            pickle.dump(tokenizer, f)

        print("BiLSTM model trained and saved successfully")

    except Exception as e:
        print(f"Error during BiLSTM training: {e}")
        print("Continuing with other models...")


# Improved BERT training function
def train_bert_classifier_improved(X, y, model_dir='models'):
    """
    Improved BERT training for small datasets
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Extract text
    texts = [item['text'][:512] for item in X]  # Limit text length to BERT's max

    # Load tokenizer and pretrained model
    print("Loading BERT pretrained model...")
    model_name = "emilyalsentzer/Bio_ClinicalBERT"  # Use ClinicalBERT

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternate model...")
        model_name = "bert-base-uncased"  # Fallback model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Prepare data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tokenize data with special handling for small datasets
    train_encodings = tokenizer(
        X_train,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )

    val_encodings = tokenizer(
        X_val,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )

    # Create tensor datasets
    train_dataset = TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        torch.tensor(y_train)
    )

    val_dataset = TensorDataset(
        val_encodings['input_ids'],
        val_encodings['attention_mask'],
        torch.tensor(y_val)
    )

    # Create data loaders - use smaller batch sizes for small datasets
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=2  # Small batch size for small datasets
    )

    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=2
    )

    # Training settings - adjust for small datasets
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Total steps
    epochs = 5
    total_steps = len(train_dataloader) * epochs

    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize tracking variables
    best_val_loss = float('inf')

    print("Starting BERT fine-tuning...")
    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Training
            model.train()
            train_loss = 0

            for batch in train_dataloader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                model.zero_grad()

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = train_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss:.4f}")

            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_true = []

            for batch in val_dataloader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                loss = outputs.loss
                val_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                true_labels = labels.cpu().numpy()

                val_preds.extend(preds)
                val_true.extend(true_labels)

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation loss: {avg_val_loss:.4f}")

            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_true, val_preds, average='binary'
            )

            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print("Saving best model...")
                model.save_pretrained(os.path.join(model_dir, "bert_classifier"))
                tokenizer.save_pretrained(os.path.join(model_dir, "bert_classifier"))

        print("BERT training complete!")

    except Exception as e:
        print(f"Error during BERT training: {e}")
        print("Will fall back to traditional ML models.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process MIMIC notes data and train insomnia detection models")
    parser.add_argument("--noteevents", required=True, help="Path to NOTEEVENTS.csv file")
    parser.add_argument("--test_ids", required=False, default="", help="Path to test IDs file")
    parser.add_argument("--subtask1_labels", required=False, default="", help="Path to subtask 1 labels file")
    parser.add_argument("--subtask2a_labels", required=False, default="", help="Path to subtask 2A labels file")
    parser.add_argument("--subtask2b_labels", required=False, default="", help="Path to subtask 2B labels file")
    parser.add_argument("--model_dir", default="models", help="Directory to save models")
    parser.add_argument("--use_bert", action="store_true", help="Whether to use BERT model")
    parser.add_argument("--use_bilstm", action="store_true", help="Whether to use BiLSTM model")
    parser.add_argument("--data_augmentation", action="store_true", help="Whether to use data augmentation")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--predict", action="store_true", help="Make predictions")
    parser.add_argument("--predict_subtask", default="all", choices=["1", "2a", "2b", "all"],
                        help="Which subtask to predict")
    parser.add_argument("--output_dir", default="predictions", help="Directory to save predictions")
    parser.add_argument("--competition_format", action="store_true", help="Save predictions in competition format")

    args = parser.parse_args()

    # Set up logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    print("Reading NOTEEVENTS data...")
    df_notes = pd.read_csv(args.noteevents)

    # Create the necessary directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    annotation_files = {
        'subtask1': args.subtask1_labels if args.subtask1_labels else None,
        'subtask2a': args.subtask2a_labels if args.subtask2a_labels else None,
        'subtask2b': args.subtask2b_labels if args.subtask2b_labels else None
    }


    if args.train:
        print("Loading labels...")
        labels = load_labels(annotation_files)

        print("Extracting IDs of labeled notes...")
        note_ids = list(labels.keys())
        if note_ids:
            df_notes_subset = df_notes[df_notes['ROW_ID'].isin([int(nid) for nid in note_ids])]
            print(f"Found {len(df_notes_subset)} notes with labels")

            # Check data balance
            insomnia_count = sum(1 for nid in note_ids if 'Insomnia' in labels[nid] and labels[nid]['Insomnia'] == 1)
            print(f"Data balance: {insomnia_count} insomnia cases, {len(note_ids) - insomnia_count} non-insomnia cases")

            # Prepare training data, optional data enhancement.
            if args.data_augmentation:
                print("Preparing training data with augmentation...")
                X_task1, y_task1 = prepare_training_data_with_augmentation(df_notes_subset, labels, subtask='1',
                                                                           augment=True)
                X_task2a, y_task2a = prepare_training_data_with_augmentation(df_notes_subset, labels, subtask='2a',
                                                                             augment=True)
                print(f"After augmentation: {len(X_task1)} samples for subtask 1")
            else:
                print("Preparing training data without augmentation...")
                X_task1, y_task1 = prepare_training_data(df_notes_subset, labels, subtask='1')
                X_task2a, y_task2a = prepare_training_data(df_notes_subset, labels, subtask='2a')


            print("Transforming features...")
            text_features, manual_features, vectorizer = transform_features(X_task1)


            print("Training models with cross-validation...")
            train_models_with_cv(
                (text_features, manual_features, vectorizer),
                {'1': y_task1, '2a': y_task2a},
                model_dir=args.model_dir
            )


            if args.use_bert:
                print("Training BERT model...")
                train_bert_classifier_improved(X_task1, y_task1, model_dir=args.model_dir)

            if args.use_bilstm:
                print("Training BiLSTM model...")
                train_bilstm_model(X_task1, y_task1, model_dir=args.model_dir)

            print("Training complete! Models saved to", args.model_dir)
        else:
            print("No labeled data found. Skipping training.")

    # Forecast section
    if args.predict:
        if args.test_ids:
            print(f"Loading test IDs from {args.test_ids}...")
            test_ids = load_test_ids(args.test_ids)
            print(f"Found {len(test_ids)} test IDs")

            # Load ground truth for evaluation if available
            ground_truth = None
            if args.subtask1_labels or args.subtask2a_labels or args.subtask2b_labels:
                print("Loading ground truth labels for evaluation...")
                ground_truth = load_labels(annotation_files)

            print(f"Running prediction for subtask {args.predict_subtask} on test set...")
            predictions = predict_insomnia_for_test(
                df_notes,
                test_ids,
                model_dir=args.model_dir,
                subtask=args.predict_subtask,
                use_bert=args.use_bert,
                ground_truth=ground_truth  # Pass ground truth labels if available
            )

            # Save as Competition Format
            if args.competition_format:
                if args.predict_subtask in ['1', 'all']:
                    save_competition_format(
                        predictions,
                        '1',
                        os.path.join(args.output_dir, "subtask_1.json")
                    )

                if args.predict_subtask in ['2a', 'all']:
                    save_competition_format(
                        predictions,
                        '2a',
                        os.path.join(args.output_dir, "subtask_2a.json")
                    )

                if args.predict_subtask in ['2b', 'all']:
                    save_competition_format(
                        predictions,
                        '2b',
                        os.path.join(args.output_dir, "subtask_2b.json")
                    )
            else:

                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

                if args.predict_subtask in ['1', 'all']:
                    task1_preds = {nid: {'Insomnia': pred['Insomnia']} for nid, pred in predictions.items()}
                    with open(os.path.join(args.output_dir, f"subtask1_predictions_{timestamp}.json"), 'w') as f:
                        json.dump(task1_preds, f, indent=2)
                    print(f"Subtask 1 predictions saved to {args.output_dir}/subtask1_predictions_{timestamp}.json")

                if args.predict_subtask in ['2a', 'all']:
                    task2a_preds = {}
                    for nid, pred in predictions.items():
                        task2a_preds[nid] = {k: pred[k] for k in
                                             ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']
                                             if k in pred}

                    if task2a_preds:
                        with open(os.path.join(args.output_dir, f"subtask2a_predictions_{timestamp}.json"), 'w') as f:
                            json.dump(task2a_preds, f, indent=2)
                        print(
                            f"Subtask 2A predictions saved to {args.output_dir}/subtask2a_predictions_{timestamp}.json")

                if args.predict_subtask in ['2b', 'all']:
                    task2b_preds = {}
                    for nid, pred in predictions.items():
                        if any(k.endswith('_evidence') for k in pred):
                            task2b_preds[nid] = {}
                            for label in ['Definition 1', 'Definition 2', 'Rule A', 'Rule B', 'Rule C']:
                                if f"{label}_evidence" in pred:
                                    task2b_preds[nid][label] = {
                                        "label": pred[label] if label in pred else "unknown",
                                        "text": pred[f"{label}_evidence"]["text"] if isinstance(
                                            pred[f"{label}_evidence"],
                                            dict) else pred[
                                            f"{label}_evidence"]
                                    }
                                elif label in pred and isinstance(pred[label], dict) and "text" in pred[label]:
                                    task2b_preds[nid][label] = pred[label]

                    if task2b_preds:
                        with open(os.path.join(args.output_dir, f"subtask2b_predictions_{timestamp}.json"), 'w') as f:
                            json.dump(task2b_preds, f, indent=2)
                        print(
                            f"Subtask 2B predictions saved to {args.output_dir}/subtask2b_predictions_{timestamp}.json")
        else:
            print("No test IDs provided. Please provide a test ID file.")

        print("Prediction complete!")


if __name__ == "__main__":
    main()