import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

INPUT_FILE = "testing/evaluation/experiment_results.csv"
LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]


def compute_metrics(model_config=None, run_id=None):

    df = pd.read_csv(INPUT_FILE)

    # -----------------------------
    # Filtering
    # -----------------------------
    if model_config is not None:
        df = df[df["model_config"] == model_config]

    if run_id is not None:
        df = df[df["run_id"] == run_id]

    if len(df) == 0:
        print("No rows found for given filter.")
        return

    y_true = df["ground_truth"]
    y_pred = df["predicted_label"]

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0
    )

    # Hallucination Rate
    hallucinations = df[
        (df["predicted_label"] == "SUPPORTED") &
        (df["ground_truth"] != "SUPPORTED")
    ]
    hallucination_rate = len(hallucinations) / len(df)

    print("\nRows evaluated:", len(df))
    print("\nPer-class metrics:")
    for label in LABELS:
        print(
            label,
            "Precision:", report[label]["precision"],
            "Recall:", report[label]["recall"],
            "F1:", report[label]["f1-score"]
        )

    print("\nConfusion Matrix:\n", cm)
    print("\nAccuracy:", acc)
    print("\nMacro F1:", report["macro avg"]["f1-score"])
    print("\nWeighted F1:", report["weighted avg"]["f1-score"])
    print("\nHallucination Rate:", hallucination_rate)