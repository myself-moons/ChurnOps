"""
retrain_trigger.py — ChurnOps

Evaluates whether model retraining should be recommended.

DESIGN INTENT
-------------
Retraining is triggered when measured model performance drops below
a configured threshold. However, measuring performance in production
requires ground-truth labels — which are only available after a delay
(typically 30-90 days in a telecom churn context).

This module implements two trigger modes:

1. evaluate_with_labels(labeled_outcomes)
   Computes actual ROC-AUC from provided ground-truth labels,
   compares against params.yaml threshold, and recommends retraining.

2. check_distribution_drift(recent_predictions)
   A weaker signal — checks whether the recent churn prediction rate
   has shifted significantly from the training distribution.
   Can be used as an early warning WITHOUT ground-truth labels.
   NOTE: distribution shift alone does not prove performance degradation.

The typical workflow is:
  Production predictions
  + Later ground-truth outcomes (from CRM/billing after 30-90 days)
  → evaluate_with_labels()
  → compare against threshold
  → retraining recommendation
"""

from pathlib import Path
from typing import Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_threshold() -> dict:
    """Load monitoring configuration from params.yaml."""
    params_path = ROOT_DIR / "params.yaml"
    if not params_path.exists():
        return {"roc_auc_threshold": 0.75, "min_predictions": 50}
    with params_path.open() as f:
        config = yaml.safe_load(f)
    return config.get("monitoring", {"roc_auc_threshold": 0.75, "min_predictions": 50})


def evaluate_with_labels(
    labeled_outcomes: list[dict],
) -> dict:
    """
    Layer B trigger: Compute ROC-AUC from ground-truth labels and decide
    whether retraining is needed.

    Args:
        labeled_outcomes: List of dicts:
            {"timestamp": "<matches a prediction log entry>", "actual_label": 0|1}

    Returns:
        dict with keys:
            retrain_recommended: bool
            reason: str
            roc_auc: float (if computable)
            threshold: float
    """
    from src.monitor import evaluate_with_ground_truth

    config    = _load_threshold()
    threshold = config.get("roc_auc_threshold", 0.75)
    min_preds = config.get("min_predictions", 50)

    result = evaluate_with_ground_truth(labeled_outcomes)
    if result is None or result.get("status") != "evaluated":
        return {
            "retrain_recommended": False,
            "reason": result.get("message", "Insufficient ground-truth data for evaluation."),
            "threshold": threshold,
        }

    roc_auc = result["roc_auc"]
    retrain  = roc_auc < threshold

    return {
        "retrain_recommended": retrain,
        "reason": (
            f"ROC-AUC {roc_auc:.4f} is below threshold {threshold:.4f}. "
            f"Retraining is recommended."
            if retrain else
            f"ROC-AUC {roc_auc:.4f} is above threshold {threshold:.4f}. "
            f"Model performance is acceptable."
        ),
        "roc_auc":   roc_auc,
        "threshold": threshold,
        "n_samples": result.get("n_samples"),
        "performance": result,
    }


def check_distribution_drift(
    training_churn_rate: float = 0.265,
    drift_tolerance: float = 0.10,
) -> dict:
    """
    Layer A weak signal: Compare recent prediction distribution against
    the training distribution to detect potential drift.

    This is an early warning only. It does NOT measure model performance.
    A shift in predicted churn rate may indicate concept drift, data
    pipeline issues, or genuine business changes — but cannot distinguish
    between them without ground-truth labels.

    Args:
        training_churn_rate: Churn rate in the training data (0.265 = 26.5%)
        drift_tolerance:     Alert if absolute deviation exceeds this value

    Returns:
        dict with drift assessment
    """
    from src.monitor import get_operational_stats

    stats = get_operational_stats()
    if stats.get("status") != "ok":
        return {
            "drift_detected": False,
            "reason": "No prediction data available.",
        }

    recent_rate = stats["churn_rate"]
    deviation   = abs(recent_rate - training_churn_rate)
    drift       = deviation > drift_tolerance

    return {
        "drift_detected":           drift,
        "training_churn_rate":      training_churn_rate,
        "recent_churn_rate":        recent_rate,
        "absolute_deviation":       round(deviation, 4),
        "tolerance":                drift_tolerance,
        "total_predictions":        stats["total_predictions"],
        "warning": (
            "Prediction distribution has shifted significantly. "
            "This is an early warning signal — ground-truth evaluation is needed "
            "to confirm whether model performance has degraded."
            if drift else None
        ),
        "note": (
            "Distribution drift is a weak signal only. "
            "True performance monitoring requires ground-truth labels."
        ),
    }
