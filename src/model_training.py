r"""
model_training.py -- ChurnOps

MODEL SELECTION METHODOLOGY (correct, post-fix):
-------------------------------------------------
Full dataset
    |
    |-- Training data (80%)  --> Stratified 5-fold CV
    |                                |
    |                                |-- Logistic Regression  -> mean CV ROC-AUC
    |                                |-- Random Forest        -> mean CV ROC-AUC
    |                                \-- XGBoost             -> mean CV ROC-AUC
    |
    |   1. Select model with highest mean CV ROC-AUC
    |   2. Retrain selected model on the COMPLETE training split
    |
    \-- Test data (20%) -- UNTOUCHED until step 3.
                3. Evaluate the retrained winner ONCE -> final metrics

The test set is NOT used for model selection.
"""

import argparse
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGET   = "Churn Value"

# -- Cross-validation configuration ------------------------------------------ #
CV_N_SPLITS   = 5
CV_SHUFFLE    = True
# Random state for StratifiedKFold comes from params (model.random_state)


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


def get_model_params(params: dict, model_type: str) -> dict:
    """Extract and return only hyperparameters relevant to the specified model."""
    rs = params.get("random_state", 42)
    if model_type == "logistic_regression":
        return {
            "model_type": "logistic_regression",
            "C": float(params.get("C", 1.0)),
            "max_iter": int(params.get("max_iter", 1000)),
            "random_state": int(rs),
            "class_weight": "balanced",
            "solver": "lbfgs",
        }
    elif model_type == "random_forest":
        return {
            "model_type": "random_forest",
            "n_estimators": int(params.get("n_estimators", 100)),
            "max_depth": int(params.get("max_depth", 6)),
            "random_state": int(rs),
            "class_weight": "balanced",
        }
    elif model_type == "xgboost":
        return {
            "model_type": "xgboost",
            "n_estimators": int(params.get("n_estimators", 100)),
            "max_depth": int(params.get("max_depth", 6)),
            "learning_rate": float(params.get("learning_rate", 0.1)),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "random_state": int(rs),
            "eval_metric": "logloss",
        }
    return {"model_type": model_type, "random_state": int(rs)}


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    args   = parse_args()
    params = load_params(args)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT_DIR / 'mlruns'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-churn"))

    # -- Load data ----------------------------------------------------------- #
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

    # -- Stratified K-Fold cross-validator ----------------------------------- #
    skf = StratifiedKFold(
        n_splits=CV_N_SPLITS,
        shuffle=CV_SHUFFLE,
        random_state=params["random_state"],
    )

    # -- Phase 1: Cross-validate all candidate models on TRAINING data only -- #
    # The test set is NOT touched during this phase.
    print(f"\n-- Phase 1: {CV_N_SPLITS}-fold Stratified CV on training data --")

    cv_results: dict[str, dict] = {}          # model_type -> {mean, std, fold_scores, clf, run_id}

    for model_type in model_choices:
        trial_params = {**params, "model_type": model_type}
        clf = build_model(trial_params, model_type)

        # Inject scale_pos_weight for XGBoost after construction
        if model_type == "xgboost":
            clf.set_params(scale_pos_weight=scale_pos_weight)

        # Run CV on training data -- ROC-AUC per fold
        fold_scores = cross_val_score(
            clf,
            X_train, y_train,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1,
        )

        mean_auc = float(np.mean(fold_scores))
        std_auc  = float(np.std(fold_scores))

        print(
            f"  [{model_type}]  CV ROC-AUC = {mean_auc:.4f} +/- {std_auc:.4f}  "
            f"(folds: {[round(s, 4) for s in fold_scores]})"
        )

        # Log CV results to MLflow (one run per candidate)
        model_specific_params = get_model_params(trial_params, model_type)
        with mlflow.start_run(run_name=f"{model_type}_cv") as run:
            mlflow.set_tags({
                "run_stage": "cv_candidate",
                "model_family": model_type,
            })
            mlflow.log_params(model_specific_params)
            mlflow.log_metrics({
                "cv_mean_roc_auc": mean_auc,
                "cv_std_roc_auc":  std_auc,
                **{f"cv_fold_{i+1}_roc_auc": float(s) for i, s in enumerate(fold_scores)},
            })

        cv_results[model_type] = {
            "mean_auc":   mean_auc,
            "std_auc":    std_auc,
            "fold_scores": fold_scores.tolist(),
            "params":      trial_params,
            "cv_run_id":   run.info.run_id,
        }

    # -- Phase 2: Select the winning model from CV results ------------------- #
    # Test set has NOT been used yet.
    best_model_type = max(cv_results, key=lambda m: cv_results[m]["mean_auc"])
    best_cv         = cv_results[best_model_type]

    print(f"\n-- Phase 2: Selected '{best_model_type}' (CV ROC-AUC = {best_cv['mean_auc']:.4f}) --")
    print("  Retraining selected model on COMPLETE training split...")

    # -- Phase 3: Retrain the winner on the FULL training split -------------- #
    final_clf = build_model(best_cv["params"], best_model_type)
    if best_model_type == "xgboost":
        final_clf.set_params(scale_pos_weight=scale_pos_weight)

    final_clf.fit(X_train, y_train)

    # -- Phase 4: Evaluate ONCE on the untouched test split ------------------ #
    # This is the only place the test set is used -- AFTER model selection.
    print("\n-- Phase 3: Final evaluation on UNTOUCHED test split (one shot) --")

    test_prob = final_clf.predict_proba(X_test)[:, 1]
    test_pred = final_clf.predict(X_test)
    test_m    = compute_metrics(y_test, test_pred, test_prob)

    print(
        f"  [{best_model_type}]  "
        f"Test ROC-AUC={test_m['roc_auc']:.4f}  "
        f"F1={test_m['f1_score']:.4f}  "
        f"Acc={test_m['accuracy']:.4f}"
    )

    # -- Phase 5: Log final model and test metrics to MLflow ----------------- #
    winner_params = get_model_params(best_cv["params"], best_model_type)
    with mlflow.start_run(run_name=f"{best_model_type}_final") as final_run:
        mlflow.set_tags({
            "run_stage": "final_champion",
            "model_family": best_model_type,
            "selected_as_champion": "true",
        })
        mlflow.log_params({
            **winner_params,
            "selection_method": "stratified_5fold_cv",
        })
        mlflow.log_metrics({
            # CV selection metrics
            "cv_mean_roc_auc": best_cv["mean_auc"],
            "cv_std_roc_auc":  best_cv["std_auc"],
            # Final test metrics (generated ONCE, after model selection)
            "test_roc_auc":  test_m["roc_auc"],
            "test_accuracy": test_m["accuracy"],
            "test_precision":test_m["precision"],
            "test_recall":   test_m["recall"],
            "test_f1_score": test_m["f1_score"],
        })
        mlflow.sklearn.log_model(final_clf, "model")
        final_run_id = final_run.info.run_id

    print(f"\nFinal model: {type(final_clf).__name__}")
    print(f"  CV mean ROC-AUC : {best_cv['mean_auc']:.4f} +/- {best_cv['std_auc']:.4f}")
    print(f"  Test ROC-AUC    : {test_m['roc_auc']:.4f}")
    print(f"  Test Accuracy   : {test_m['accuracy']:.4f}")
    print(f"  Test F1         : {test_m['f1_score']:.4f}")

    # -- Save artifacts ------------------------------------------------------ #
    (ROOT_DIR / ".mlflow_run_id").write_text(final_run_id)

    with (ROOT_DIR / "model.pkl").open("wb") as f:
        pickle.dump(final_clf, f)

    print("\nmodel.pkl saved.")


if __name__ == "__main__":
    main()