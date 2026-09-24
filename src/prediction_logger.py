"""
prediction_logger.py — ChurnOps

Appends one JSON record per prediction to predictions.jsonl.
This file accumulates operational prediction data that can later
be combined with ground-truth labels for model performance evaluation.

IMPORTANT DESIGN NOTE
---------------------
Prediction logs alone cannot provide true model performance metrics
(ROC-AUC, precision, recall, F1) because they contain predicted labels
and probabilities but NOT actual ground-truth outcomes.

True performance monitoring requires a separate feedback loop where
ground-truth churn labels (e.g., from CRM systems after 30-90 days)
are matched against these logged predictions.

This module handles Layer A — Operational monitoring only:
  - prediction count
  - prediction distribution (churn vs not-churn)
  - probability distribution
  - request latency
  - model version usage

Layer B (Model Performance) is enabled by the separate
retrain_trigger.py module, which accepts ground-truth data.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT_DIR / "predictions.jsonl"


def log_prediction(
    *,
    prediction: int,
    probability: float,
    model_version: str,
    latency_ms: float,
    input_features: dict | None = None,
) -> None:
    """
    Append a single prediction record to predictions.jsonl.

    Args:
        prediction:     0 (not churn) or 1 (churn)
        probability:    Model confidence score for the churn class (0.0–1.0)
        model_version:  Model class name, e.g. 'XGBClassifier'
        latency_ms:     End-to-end prediction latency in milliseconds
        input_features: Optional dict of raw input features (for auditability)
    """
    record = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "prediction":    int(prediction),
        "probability":   round(float(probability), 6),
        "model_version": model_version,
        "latency_ms":    round(float(latency_ms), 3),
    }
    # Store input features if provided (useful for drift detection later)
    if input_features is not None:
        record["input_features"] = input_features

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
