"""
model_evaluation.py — ChurnOps

Loads the best model (model.pkl) and the test set, computes full
classification metrics, logs them back to the corresponding MLflow run,
and writes metrics.json for the dashboard and DVC.
"""

import json
import os
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGET   = "Churn Value"


def main():
    test = pd.read_csv(ROOT_DIR / "data/processed/test_processed.csv")
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].values

    with (ROOT_DIR / "model.pkl").open("rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc      = accuracy_score(y_test, y_pred)
    prec     = precision_score(y_test, y_pred, zero_division=0)
    recall   = recall_score(y_test, y_pred, zero_division=0)
    f1       = f1_score(y_test, y_pred, zero_division=0)
    roc_auc  = roc_auc_score(y_test, y_prob)
    cm       = confusion_matrix(y_test, y_pred)

    # Resume the best run to append evaluation metrics
    run_id = None
    run_id_path = ROOT_DIR / ".mlflow_run_id"
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT_DIR / 'mlruns'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-churn"))

    run_ctx = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run()
    with run_ctx:
        mlflow.log_metrics({
            "eval_accuracy":  acc,
            "eval_precision": prec,
            "eval_recall":    recall,
            "eval_f1_score":  f1,
            "eval_roc_auc":   roc_auc,
            # Confusion matrix cells
            "eval_tn": int(cm[0, 0]),
            "eval_fp": int(cm[0, 1]),
            "eval_fn": int(cm[1, 0]),
            "eval_tp": int(cm[1, 1]),
        })

    metrics_dict = {
        "model": type(model).__name__,
        "accuracy":  acc,
        "precision": prec,
        "recall":    recall,
        "f1_score":  f1,
        "roc_auc":   roc_auc,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }
    with (ROOT_DIR / "metrics.json").open("w") as mf:
        json.dump(metrics_dict, mf, indent=4)

    print(f"Model: {metrics_dict['model']}")
    print(f"  ROC-AUC  = {roc_auc:.4f}")
    print(f"  Accuracy = {acc:.4f}")
    print(f"  F1       = {f1:.4f}")
    print(f"  Precision= {prec:.4f}")
    print(f"  Recall   = {recall:.4f}")
    print(f"  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    print("Evaluation completed. metrics.json written.")


if __name__ == "__main__":
    main()