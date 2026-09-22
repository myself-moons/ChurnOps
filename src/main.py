from pathlib import Path
import pickle
import json
import os

import mlflow
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

try:
    from src.data_model import Water
except ImportError:  # pragma: no cover - supports running the module directly
    from data_model import Water

app = FastAPI(
    title="Water Potability Prediction",
    description="Predicting Water Potability",
)
 # Setting path for model.pkl file and loading the model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"
DASHBOARD_PATH = BASE_DIR / "src" / "dashboard.html"
LANDING_PATH = BASE_DIR / "src" / "landing.html"
PREDICT_PATH = BASE_DIR / "src" / "predict.html"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{BASE_DIR / 'mlruns'}")

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)
SELECTED_MODEL = type(model).__name__


@app.get("/", response_class=HTMLResponse)
def index():
    return LANDING_PATH.read_text()


def _dataset_summary(path: Path):
    if not path.exists() and path.name == "train.csv":
        path = BASE_DIR / "water_potability (1).csv"
    if not path.exists():
        return {"available": False, "rows": 0, "features": 0, "missing_values": 0}

    data = pd.read_csv(path)
    return {
        "available": True,
        "rows": len(data),
        "features": len(data.columns) - 1,
        "missing_values": int(data.isna().sum().sum()),
        "class_distribution": {
            str(label): int(count)
            for label, count in data["Potability"].value_counts().sort_index().items()
        },
    }


def _mlflow_runs():
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name("water-potability")
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
            model_type = "random_forest"

        def clean(value):
            return None if pd.isna(value) else value

        run_history.append({
            "run_id": clean(run.get("run_id")),
            "status": clean(run.get("status")) or "UNKNOWN",
            "start_time": run.get("start_time").isoformat() if pd.notna(run.get("start_time")) else None,
            "model_type": model_type,
            "n_estimators": clean(run.get("params.n_estimators")),
            "max_depth": clean(run.get("params.max_depth")),
            "random_state": clean(run.get("params.random_state")),
            "learning_rate": clean(run.get("params.learning_rate")),
            "subsample": clean(run.get("params.subsample")),
            "colsample_bytree": clean(run.get("params.colsample_bytree")),
            "train_accuracy": clean(run.get("metrics.train_accuracy")),
            "test_accuracy": clean(run.get("metrics.test_accuracy")),
            "test_f1_score": clean(run.get("metrics.test_f1_score")),
        })
    return run_history


@app.get("/api/dashboard")
def dashboard_data():
    runs = _mlflow_runs()
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    latest_run = runs[0] if runs else None
    best_run = max(
        (run for run in runs if isinstance(run.get("test_accuracy"), (float, int))),
        key=lambda run: run["test_accuracy"],
        default=None,
    )
    return {
        "project": {
            "name": "Water Potability Prediction",
            "description": "A DVC-managed Random Forest pipeline with MLflow experiment tracking and a FastAPI prediction service.",
            "pipeline": ["Data Collection", "Data Preprocessing", "Model Training", "Evaluation"],
        },
        "datasets": {
            "train": _dataset_summary(BASE_DIR / "data/raw/train.csv"),
            "test": _dataset_summary(BASE_DIR / "data/raw/test.csv"),
            "processed_train": _dataset_summary(BASE_DIR / "data/processed/train_processed.csv"),
            "processed_test": _dataset_summary(BASE_DIR / "data/processed/test_processed.csv"),
        },
        "results": metrics,
        "tracking": {
            "experiment": "water-potability",
            "selected_model": SELECTED_MODEL,
            "run_count": len(runs),
            "finished_count": sum(run["status"] == "FINISHED" for run in runs),
            "latest_run": latest_run,
            "best_run": best_run,
        },
        "runs": runs,
    }


@app.get("/api/runs")
def runs():
    return {"experiment": "water-potability", "runs": _mlflow_runs()}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_PATH.read_text()


@app.get("/predict", response_class=HTMLResponse)
def prediction_page():
    return PREDICT_PATH.read_text()


@app.post("/predict")
def model_predict(payload: Water):
    sample = pd.DataFrame(
        [{
            "ph": payload.ph,
            "Hardness": payload.Hardness,
            "Solids": payload.Solids,
            "Chloramines": payload.Chloramines,
            "Sulfate": payload.Sulfate,
            "Conductivity": payload.Conductivity,
            "Organic_carbon": payload.Organic_carbon,
            "Trihalomethanes": payload.Trihalomethanes,
            "Turbidity": payload.Turbidity,
        }]
    )

    predicted_value = int(model.predict(sample)[0])
    prediction = "Water is Consumable" if predicted_value == 1 else "Water is not Consumable"
    return {"prediction": prediction, "model": SELECTED_MODEL}
