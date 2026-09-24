"""
monitor.py — ChurnOps

Reads predictions.jsonl and computes operational monitoring statistics.

TWO-LAYER MONITORING ARCHITECTURE
----------------------------------

Layer A — Operational monitoring (available immediately from prediction logs):
  - Total prediction count
  - Churn prediction rate (% of predictions that are churn)
  - Probability distribution (mean, min, max, std)
  - Average and P95 latency
  - Model version breakdown

Layer B — Model performance monitoring (requires ground-truth labels):
  - ROC-AUC, Precision, Recall, F1 against actual churn outcomes
  - This layer is exposed via get_performance_metrics() which accepts
    a list of (prediction_id, actual_label) tuples from a ground-truth feed.

IMPORTANT LIMITATION
--------------------
Without ground-truth labels, it is impossible to know whether the model
is performing well. Probability distribution shifts can be early warning
signals, but they are NOT a substitute for true performance evaluation.

The typical telco churn cycle is 30–90 days — a customer's actual churn
status is only known after that period. The recommended integration is:
  1. Log predictions with a stable identifier (customer_id, request_id)
  2. After 30-90 days, receive the actual outcomes from the CRM/billing system
  3. Match outcomes to logged predictions using the identifier
  4. Call evaluate_with_ground_truth() to compute real performance metrics
  5. Compare against params.yaml threshold to decide on retraining
"""

import json
import statistics
from pathlib import Path
from typing import Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT_DIR / "predictions.jsonl"


def _load_logs() -> list[dict]:
    """Load all prediction log records."""
    if not LOG_PATH.exists():
        return []
    records = []
    with LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def get_operational_stats(recent_n: int = 100) -> dict:
    """
    Layer A: Operational statistics computed directly from the prediction log.
    No ground-truth labels required.

    Args:
        recent_n: Number of most-recent predictions to analyse. Use -1 for all.
    """
    records = _load_logs()
    if not records:
        return {
            "status": "no_predictions",
            "message": "No predictions have been logged yet.",
            "total_predictions": 0,
        }

    window = records[-recent_n:] if recent_n > 0 else records
    total  = len(records)

    predictions  = [r["prediction"] for r in window]
    probs        = [r["probability"] for r in window]
    latencies    = [r["latency_ms"] for r in window if "latency_ms" in r]
    model_vers   = {}
    for r in window:
        mv = r.get("model_version", "unknown")
        model_vers[mv] = model_vers.get(mv, 0) + 1

    churn_rate = sum(predictions) / len(predictions) if predictions else 0.0

    # P95 latency
    sorted_lat = sorted(latencies)
    p95_latency = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else None

    return {
        "status": "ok",
        "total_predictions":  total,
        "window_size":        len(window),
        "churn_rate":         round(churn_rate, 4),
        "churn_count":        sum(predictions),
        "non_churn_count":    len(predictions) - sum(predictions),
        "probability_stats": {
            "mean": round(statistics.mean(probs), 4)  if probs else None,
            "min":  round(min(probs), 4)              if probs else None,
            "max":  round(max(probs), 4)              if probs else None,
            "std":  round(statistics.stdev(probs), 4) if len(probs) > 1 else 0.0,
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2) if latencies else None,
            "p95":  round(p95_latency, 2)                if p95_latency else None,
        },
        "model_versions": model_vers,
        "layer_a_note": (
            "Operational stats only. Model performance metrics (ROC-AUC, F1) "
            "require ground-truth labels. See /api/monitor for the retraining trigger."
        ),
    }


def evaluate_with_ground_truth(
    labeled_outcomes: list[dict],
) -> Optional[dict]:
    """
    Layer B: Model performance evaluation.

    Args:
        labeled_outcomes: List of dicts, each containing:
            {
                "timestamp": "<ISO timestamp matching a log record>",
                "actual_label": 0 or 1
            }

    Returns:
        Performance metrics dict, or None if insufficient data.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError:
        return {"error": "sklearn not available for performance evaluation"}

    logs    = _load_logs()
    log_map = {r["timestamp"]: r for r in logs}

    y_true, y_pred, y_prob = [], [], []
    for outcome in labeled_outcomes:
        ts = outcome.get("timestamp")
        if ts in log_map:
            y_true.append(outcome["actual_label"])
            y_pred.append(log_map[ts]["prediction"])
            y_prob.append(log_map[ts]["probability"])

    if len(y_true) < 10:
        return {
            "status":  "insufficient_data",
            "matched": len(y_true),
            "message": "Need at least 10 matched ground-truth labels for evaluation.",
        }

    return {
        "status":    "evaluated",
        "n_samples": len(y_true),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
