def load_test_ids(test_id_file):
    with open(test_id_file, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]
    return test_ids


def predict_insomnia_for_test(notes_df, test_ids, model_dir='models', subtask='all', use_bert=False):
    test_df = notes_df[notes_df['ROW_ID'].isin([int(nid) for nid in test_ids])]

    predictions = predict_insomnia(test_df, model_dir, subtask, use_bert)
    return predictions


def save_competition_format(predictions, subtask, output_file):
    result = {}

    if subtask == '1':
        for note_id, pred in predictions.items():
            result[note_id] = {"Insomnia": pred["Insomnia"]}

    elif subtask == '2a':
        for note_id, pred in predictions.items():
            result[note_id] = {
                "Definition 1": pred.get("Definition 1", "no"),
                "Definition 2": pred.get("Definition 2", "no"),
                "Rule A": pred.get("Rule A", "no"),
                "Rule B": pred.get("Rule B", "no"),
                "Rule C": pred.get("Rule C", "no")
            }

    elif subtask == '2b':
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

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"The results have been saved to {output_file}")
    