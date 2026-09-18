import json
import os
import pickle
from pathlib import Path

import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

ROOT_DIR = Path(__file__).resolve().parent.parent


def main():
    test_data = pd.read_csv(ROOT_DIR / "data/processed/test_processed.csv")

    X_test = test_data.drop(columns=["Potability"])
    y_test = test_data["Potability"].values

    with (ROOT_DIR / "model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)

    y_pred = model.predict(X_test)

    run_id = None
    run_id_path = ROOT_DIR / ".mlflow_run_id"
    if run_id_path.exists():
        with run_id_path.open() as run_file:
            run_id = run_file.read().strip()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT_DIR / 'mlruns'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "water-potability"))

    run_context = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run()
    with run_context:
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1score = f1_score(y_test, y_pred, zero_division=0)
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision": pre,
            "test_recall": recall,
            "test_f1_score": f1score,
        })

    metrics_dict = {
        "acc": acc,
        "precision": pre,
        "recall": recall,
        "f1_score": f1score,
    }
    with (ROOT_DIR / "metrics.json").open("w") as file:
        json.dump(metrics_dict, file, indent=4)


if __name__ == "__main__":
    main()