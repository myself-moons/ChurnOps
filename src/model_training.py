"""
model_training.py — ChurnOps

Trains Logistic Regression, Random Forest, and/or XGBoost on the
processed churn dataset. Tracks every trial in the MLflow
'customer-churn' experiment. Selects the best model by ROC-AUC
and saves it as model.pkl.
"""

import argparse
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGET   = "Churn Value"


# --------------------------------------------------------------------------- #
# CLI / params                                                                 #
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train and track a churn classification model")
    parser.add_argument("--params-file",  type=Path, default=ROOT_DIR / "params.yaml")
    parser.add_argument("--model-type",   choices=["logistic_regression", "random_forest", "xgboost", "all"])
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--max-depth",    type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--learning-rate",    type=float)
    parser.add_argument("--subsample",        type=float)
    parser.add_argument("--colsample-bytree", type=float)
    parser.add_argument("--max-iter",     type=int)
    parser.add_argument("--C",            type=float, dest="C")
    return parser.parse_args()


def load_params(args):
    with args.params_file.open() as f:
        configured = yaml.safe_load(f).get("model", {})

    params = {
        "model_type":       configured.get("model_type",       "all"),
        "n_estimators":     configured.get("n_estimators",     200),
        "max_depth":        configured.get("max_depth",        None),
        "random_state":     configured.get("random_state",     42),
        "learning_rate":    configured.get("learning_rate",    0.1),
        "subsample":        configured.get("subsample",        1.0),
        "colsample_bytree": configured.get("colsample_bytree", 1.0),
        "max_iter":         configured.get("max_iter",         1000),
        "C":                configured.get("C",                1.0),
    }
    # CLI overrides
    for name in params:
        cli_val = getattr(args, name, None)
        if cli_val is not None:
            params[name] = cli_val
    return params


# --------------------------------------------------------------------------- #
# Model factory                                                                #
# --------------------------------------------------------------------------- #
def build_model(params: dict, model_type: str):
    rs = params["random_state"]

    if model_type == "logistic_regression":
        return LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            random_state=rs,
            class_weight="balanced",
            solver="lbfgs",
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=rs,
            class_weight="balanced",
            n_jobs=-1,
        )

    if model_type == "xgboost":
        # scale_pos_weight compensates for the class imbalance
        # ratio = number of negatives / number of positives
        # This is set dynamically based on training data inside main()
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"] or 6,
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=rs,
            eval_metric="logloss",
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1_score":  f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    args   = parse_args()
    params = load_params(args)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT_DIR / 'mlruns'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-churn"))

    train = pd.read_csv(ROOT_DIR / "data/processed/train_processed.csv")
    test  = pd.read_csv(ROOT_DIR / "data/processed/test_processed.csv")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET].values
    X_test  = test.drop(columns=[TARGET])
    y_test  = test[TARGET].values

    # Class imbalance ratio for XGBoost scale_pos_weight
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model_choices = (
        ["logistic_regression", "random_forest", "xgboost"]
        if params["model_type"] == "all"
        else [params["model_type"]]
    )

    best_model   = None
    best_roc_auc = -1.0
    best_run_id  = None

    for model_type in model_choices:
        trial_params = {**params, "model_type": model_type}

        with mlflow.start_run(run_name=model_type) as run:
            clf = build_model(trial_params, model_type)

            # Inject scale_pos_weight for XGBoost after construction
            if model_type == "xgboost":
                clf.set_params(scale_pos_weight=scale_pos_weight)

            clf.fit(X_train, y_train)

            # Training metrics
            train_prob = clf.predict_proba(X_train)[:, 1]
            train_pred = clf.predict(X_train)
            train_m    = compute_metrics(y_train, train_pred, train_prob)

            # Test metrics
            test_prob  = clf.predict_proba(X_test)[:, 1]
            test_pred  = clf.predict(X_test)
            test_m     = compute_metrics(y_test, test_pred, test_prob)

            # Log everything
            mlflow.log_params(trial_params)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
            mlflow.log_metrics({f"test_{k}":  v for k, v in test_m.items()})
            mlflow.sklearn.log_model(clf, "model")

            print(
                f"  [{model_type}]  "
                f"ROC-AUC={test_m['roc_auc']:.4f}  "
                f"F1={test_m['f1_score']:.4f}  "
                f"Acc={test_m['accuracy']:.4f}"
            )

            # Select best by ROC-AUC (primary metric for churn)
            if test_m["roc_auc"] > best_roc_auc:
                best_model   = clf
                best_roc_auc = test_m["roc_auc"]
                best_run_id  = run.info.run_id

    print(f"\nBest model: {type(best_model).__name__}  ROC-AUC={best_roc_auc:.4f}")

    (ROOT_DIR / ".mlflow_run_id").write_text(best_run_id)

    with (ROOT_DIR / "model.pkl").open("wb") as f:
        pickle.dump(best_model, f)

    print("model.pkl saved.")


if __name__ == "__main__":
    main()