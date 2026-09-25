"""
main.py — ChurnOps FastAPI application

Routes:
  GET  /               Landing / overview page
  GET  /dashboard      MLflow experiment dashboard
  GET  /predict        Prediction UI
  GET  /docs           Swagger API docs (built-in)
  GET  /api/dashboard  JSON dashboard data
  GET  /api/runs       JSON MLflow run history
  GET  /api/monitor    JSON operational monitoring stats
  POST /predict        Churn prediction endpoint

Prediction flow:
  Request (Customer fields)
  → Pydantic validation
  → prepare_features() (null fills + manual encoding)
  → preprocessor.transform() (fitted sklearn ColumnTransformer)
  → model.predict() + predict_proba()
  → prediction_logger.log_prediction()
  → JSON response
"""

import json
import os
import pickle
import time
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

try:
    from src.data_model import Customer
    from src.data_preprocessing import prepare_features
    from src.monitor import get_operational_stats
    from src.prediction_logger import log_prediction
except ImportError:  # pragma: no cover — supports running the module directly
    from data_model import Customer
    from data_preprocessing import prepare_features
    from monitor import get_operational_stats
    from prediction_logger import log_prediction

app = FastAPI(
    title="ChurnOps — Customer Churn Prediction",
    description=(
        "Customer churn classification API. Predicts whether a customer "
        "is likely to churn based on their demographics, service usage, "
        "and billing information."
    ),
    version="1.0.0",
)

BASE_DIR          = Path(__file__).resolve().parent.parent
MODEL_PATH        = BASE_DIR / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.pkl"
METRICS_PATH      = BASE_DIR / "metrics.json"
DASHBOARD_PATH    = BASE_DIR / "src" / "dashboard.html"
LANDING_PATH      = BASE_DIR / "src" / "landing.html"
PREDICT_PATH      = BASE_DIR / "src" / "predict.html"
TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI", f"file:{BASE_DIR / 'mlruns'}")

# Load model and preprocessor at startup
with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

with PREPROCESSOR_PATH.open("rb") as f:
    preprocessor = pickle.load(f)

SELECTED_MODEL = type(model).__name__


# ============================================================================ #
# HTML page routes                                                              #
# ============================================================================ #
@app.get("/", response_class=HTMLResponse)
def index():
    return LANDING_PATH.read_text(encoding="utf-8")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_PATH.read_text(encoding="utf-8")


@app.get("/predict", response_class=HTMLResponse)
def prediction_page():
    return PREDICT_PATH.read_text(encoding="utf-8")


# ============================================================================ #
# Internal helpers                                                              #
# ============================================================================ #
def _dataset_summary(path: Path) -> dict:
    if not path.exists():
        return {"available": False, "rows": 0, "features": 0, "missing_values": 0}
    data = pd.read_csv(path)
    target = "Churn Value"
    dist = {}
    if target in data.columns:
        dist = {
            str(label): int(count)
            for label, count in data[target].value_counts().sort_index().items()
        }
    return {
        "available": True,
        "rows": len(data),
        "features": len(data.columns) - 1,
        "missing_values": int(data.isna().sum().sum()),
        "class_distribution": dist,
    }


def _mlflow_runs() -> list:
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name("customer-churn")
    if experiment is None:
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    run_history = []
    for _, run in runs.iterrows():
        model_type = run.get("params.model_type")
        if pd.isna(model_type):
            model_type = "unknown"

        def clean(value):
            return None if pd.isna(value) else value

        # Format model-specific parameter summary
        param_parts = []
        if model_type == "logistic_regression":
            if pd.notna(run.get("params.C")):
                param_parts.append(f"C={run.get('params.C')}")
            if pd.notna(run.get("params.max_iter")):
                param_parts.append(f"iter={run.get('params.max_iter')}")
        elif model_type in ("random_forest", "xgboost"):
            if pd.notna(run.get("params.n_estimators")):
                param_parts.append(f"trees={run.get('params.n_estimators')}")
            if pd.notna(run.get("params.max_depth")):
                param_parts.append(f"depth={run.get('params.max_depth')}")
            if pd.notna(run.get("params.learning_rate")):
                param_parts.append(f"lr={run.get('params.learning_rate')}")

        run_name = clean(run.get("tags.mlflow.runName"))
        run_stage = clean(run.get("tags.run_stage"))
        if not run_stage:
            run_stage = "Champion" if (run_name and "final" in run_name) else "5-Fold CV"

        run_history.append({
            "run_id":        clean(run.get("run_id")),
            "run_name":      run_name,
            "run_stage":     run_stage,
            "status":        clean(run.get("status")) or "UNKNOWN",
            "start_time":    run.get("start_time").isoformat() if pd.notna(run.get("start_time")) else None,
            "model_type":    model_type,
            "params_summary": ", ".join(param_parts) if param_parts else "-",
            "n_estimators":  clean(run.get("params.n_estimators")),
            "max_depth":     clean(run.get("params.max_depth")),
            "random_state":  clean(run.get("params.random_state")),
            "learning_rate": clean(run.get("params.learning_rate")),
            "C":             clean(run.get("params.C")),
            "max_iter":      clean(run.get("params.max_iter")),
            # CV selection metrics (candidate runs)
            "cv_mean_roc_auc": clean(run.get("metrics.cv_mean_roc_auc")),
            "cv_std_roc_auc":  clean(run.get("metrics.cv_std_roc_auc")),
            # Final test metrics (only present on the *_final run)
            "test_roc_auc":  clean(run.get("metrics.test_roc_auc")),
            "test_accuracy": clean(run.get("metrics.test_accuracy")),
            "test_f1_score": clean(run.get("metrics.test_f1_score")),
            "test_precision":clean(run.get("metrics.test_precision")),
            "test_recall":   clean(run.get("metrics.test_recall")),
        })
    return run_history


# ============================================================================ #
# JSON API routes                                                               #
# ============================================================================ #
@app.get("/api/dashboard")
def dashboard_data():
    runs       = _mlflow_runs()
    metrics    = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    latest_run = runs[0] if runs else None

    # Best run by ROC-AUC (primary metric for churn)
    best_run = max(
        (r for r in runs if isinstance(r.get("test_roc_auc"), (float, int))),
        key=lambda r: r["test_roc_auc"],
        default=None,
    )

    return {
        "project": {
            "name": "ChurnOps — Customer Churn Prediction",
            "description": (
                "A DVC-managed model comparison pipeline with MLflow experiment tracking "
                "and a FastAPI prediction service for customer churn classification."
            ),
            "pipeline": [
                "Data Collection",
                "Data Preprocessing",
                "Model Training",
                "Evaluation",
            ],
        },
        "datasets": {
            "train":           _dataset_summary(BASE_DIR / "data/raw/train.csv"),
            "test":            _dataset_summary(BASE_DIR / "data/raw/test.csv"),
            "processed_train": _dataset_summary(BASE_DIR / "data/processed/train_processed.csv"),
            "processed_test":  _dataset_summary(BASE_DIR / "data/processed/test_processed.csv"),
        },
        "results": metrics,
        "tracking": {
            "experiment":      "customer-churn",
            "selected_model":  SELECTED_MODEL,
            "run_count":       len(runs),
            "finished_count":  sum(r["status"] == "FINISHED" for r in runs),
            "latest_run":      latest_run,
            "best_run":        best_run,
        },
        "runs": runs,
    }


@app.get("/api/runs")
def runs():
    return {"experiment": "customer-churn", "runs": _mlflow_runs()}


@app.get("/api/monitor")
def monitor_status():
    """
    Operational monitoring statistics derived from the prediction log.

    Returns Layer A stats (count, distribution, latency, model version usage).
    Layer B (ROC-AUC, F1) requires ground-truth labels — see the monitoring
    documentation in src/monitor.py for the integration workflow.
    """
    from src.retrain_trigger import check_distribution_drift
    stats = get_operational_stats()
    drift = check_distribution_drift()
    return {
        "operational": stats,
        "drift_check": drift,
        "note": (
            "Model performance metrics (ROC-AUC, Precision, Recall, F1) "
            "require ground-truth churn outcomes. These become available "
            "30-90 days after predictions, once actual churn is observed."
        ),
    }


# ============================================================================ #
# Prediction endpoint                                                           #
# ============================================================================ #
@app.post("/predict")
def model_predict(payload: Customer):
    """
    Predict whether a customer is likely to churn.

    Returns the predicted class (0=retained, 1=churn), the churn probability,
    and the model type used.
    """
    start_ms = time.time() * 1000

    # Build a DataFrame with column names matching training data
    row = {
        "Gender":                             payload.Gender,
        "Age":                                payload.Age,
        "Married":                            payload.Married,
        "Number of Dependents":               payload.Number_of_Dependents,
        "Satisfaction Score":                 payload.Satisfaction_Score,
        "Referred a Friend":                  payload.Referred_a_Friend,
        "Number of Referrals":                payload.Number_of_Referrals,
        "Tenure in Months":                   payload.Tenure_in_Months,
        "Offer":                              payload.Offer,
        "Phone Service":                      payload.Phone_Service,
        "Multiple Lines":                     payload.Multiple_Lines,
        "Internet Service":                   payload.Internet_Service,
        "Internet Type":                      payload.Internet_Type,
        "Online Security":                    payload.Online_Security,
        "Online Backup":                      payload.Online_Backup,
        "Device Protection Plan":             payload.Device_Protection_Plan,
        "Premium Tech Support":               payload.Premium_Tech_Support,
        "Streaming TV":                       payload.Streaming_TV,
        "Streaming Movies":                   payload.Streaming_Movies,
        "Streaming Music":                    payload.Streaming_Music,
        "Unlimited Data":                     payload.Unlimited_Data,
        "Contract":                           payload.Contract,
        "Paperless Billing":                  payload.Paperless_Billing,
        "Payment Method":                     payload.Payment_Method,
        "Avg Monthly Long Distance Charges":  payload.Avg_Monthly_Long_Distance_Charges,
        "Avg Monthly GB Download":            payload.Avg_Monthly_GB_Download,
        "Monthly Charge":                     payload.Monthly_Charge,
        "Total Charges":                      payload.Total_Charges,
        "Total Refunds":                      payload.Total_Refunds,
        "Total Extra Data Charges":           payload.Total_Extra_Data_Charges,
        "Total Long Distance Charges":        payload.Total_Long_Distance_Charges,
        "Total Revenue":                      payload.Total_Revenue,
    }

    sample_df = pd.DataFrame([row])

    # Apply the exact same preprocessing used during training
    sample_prepared = prepare_features(sample_df)
    sample_encoded  = preprocessor.transform(sample_prepared)

    predicted_value = int(model.predict(sample_encoded)[0])
    churn_prob      = float(model.predict_proba(sample_encoded)[0][1])
    latency_ms      = time.time() * 1000 - start_ms

    # Log prediction for operational monitoring
    log_prediction(
        prediction=predicted_value,
        probability=churn_prob,
        model_version=SELECTED_MODEL,
        latency_ms=latency_ms,
    )

    return {
        "prediction":        predicted_value,
        "churn":             predicted_value == 1,
        "prediction_label":  "Likely to Churn" if predicted_value == 1 else "Likely to Stay",
        "churn_probability": round(churn_prob, 4),
        "model":             SELECTED_MODEL,
        "latency_ms":        round(latency_ms, 2),
    }
