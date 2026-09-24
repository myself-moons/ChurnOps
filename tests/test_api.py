"""
tests/test_api.py — ChurnOps

API test suite covering:
  - GET route status codes and content checks
  - Valid churn prediction (all required fields)
  - Invalid / missing field handling
  - Response structure verification
  - Probability range validation
  - Dashboard data structure
  - Monitoring endpoint availability
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

# --------------------------------------------------------------------------- #
# Sample payload — a high-churn-risk customer profile                         #
# --------------------------------------------------------------------------- #
CHURN_PAYLOAD = {
    "Gender": "Male",
    "Age": 55,
    "Married": "No",
    "Number of Dependents": 0,
    "Satisfaction Score": 1,
    "Referred a Friend": "No",
    "Number of Referrals": 0,
    "Tenure in Months": 2,
    "Offer": "No Offer",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Yes",
    "Internet Type": "Fiber Optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection Plan": "No",
    "Premium Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Streaming Music": "No",
    "Unlimited Data": "No",
    "Contract": "Month-to-Month",
    "Paperless Billing": "Yes",
    "Payment Method": "Bank Withdrawal",
    "Avg Monthly Long Distance Charges": 0.0,
    "Avg Monthly GB Download": 5,
    "Monthly Charge": 95.0,
    "Total Charges": 190.0,
    "Total Refunds": 0.0,
    "Total Extra Data Charges": 0.0,
    "Total Long Distance Charges": 0.0,
    "Total Revenue": 190.0,
}

# A customer profile with lower churn risk
RETAIN_PAYLOAD = {**CHURN_PAYLOAD,
    "Satisfaction Score": 5,
    "Tenure in Months": 60,
    "Contract": "Two Year",
    "Online Security": "Yes",
    "Premium Tech Support": "Yes",
    "Total Charges": 5700.0,
    "Total Revenue": 5700.0,
}


# --------------------------------------------------------------------------- #
# HTML page routes                                                             #
# --------------------------------------------------------------------------- #
class TestPageRoutes:
    def test_landing_page(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "ChurnOps" in response.text

    def test_dashboard_page(self):
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "customer-churn" in response.text

    def test_predict_page(self):
        response = client.get("/predict")
        assert response.status_code == 200
        assert "Predict Customer Churn" in response.text


# --------------------------------------------------------------------------- #
# Prediction endpoint                                                          #
# --------------------------------------------------------------------------- #
class TestPredictEndpoint:
    def test_valid_prediction_returns_200(self):
        response = client.post("/predict", json=CHURN_PAYLOAD)
        assert response.status_code == 200

    def test_prediction_response_structure(self):
        response = client.post("/predict", json=CHURN_PAYLOAD)
        data = response.json()
        assert "prediction" in data
        assert "churn" in data
        assert "prediction_label" in data
        assert "churn_probability" in data
        assert "model" in data
        assert "latency_ms" in data

    def test_prediction_is_binary(self):
        response = client.post("/predict", json=CHURN_PAYLOAD)
        data = response.json()
        assert data["prediction"] in (0, 1)

    def test_churn_matches_prediction(self):
        response = client.post("/predict", json=CHURN_PAYLOAD)
        data = response.json()
        assert data["churn"] == (data["prediction"] == 1)

    def test_probability_in_valid_range(self):
        response = client.post("/predict", json=CHURN_PAYLOAD)
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_prediction_label_values(self):
        response = client.post("/predict", json=CHURN_PAYLOAD)
        label = response.json()["prediction_label"]
        assert label in ("Likely to Churn", "Likely to Stay")

    def test_model_field_is_string(self):
        response = client.post("/predict", json=CHURN_PAYLOAD)
        assert isinstance(response.json()["model"], str)

    def test_retain_profile_probability(self):
        """A low-risk profile should have a lower churn probability than a high-risk one."""
        high_risk = client.post("/predict", json=CHURN_PAYLOAD).json()["churn_probability"]
        low_risk  = client.post("/predict", json=RETAIN_PAYLOAD).json()["churn_probability"]
        assert low_risk < high_risk, (
            f"Expected retain profile ({low_risk:.3f}) to have lower churn prob "
            f"than churn profile ({high_risk:.3f})"
        )


# --------------------------------------------------------------------------- #
# Input validation                                                             #
# --------------------------------------------------------------------------- #
class TestInputValidation:
    def test_missing_required_field_returns_422(self):
        """Omitting a required field should return HTTP 422."""
        incomplete = {k: v for k, v in CHURN_PAYLOAD.items() if k != "Tenure in Months"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_invalid_categorical_value_returns_422(self):
        """An invalid enum value for a categorical should return HTTP 422."""
        bad = {**CHURN_PAYLOAD, "Contract": "Weekly"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_invalid_satisfaction_score_returns_422(self):
        """Satisfaction Score must be 1-5."""
        bad = {**CHURN_PAYLOAD, "Satisfaction Score": 10}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_negative_age_returns_422(self):
        bad = {**CHURN_PAYLOAD, "Age": -5}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422


# --------------------------------------------------------------------------- #
# JSON API routes                                                              #
# --------------------------------------------------------------------------- #
class TestDashboardAPI:
    def test_dashboard_api_returns_200(self):
        response = client.get("/api/dashboard")
        assert response.status_code == 200

    def test_dashboard_has_required_keys(self):
        data = client.get("/api/dashboard").json()
        assert "project" in data
        assert "datasets" in data
        assert "results" in data
        assert "tracking" in data
        assert "runs" in data

    def test_dashboard_tracking_has_experiment(self):
        data = client.get("/api/dashboard").json()
        assert data["tracking"]["experiment"] == "customer-churn"

    def test_dashboard_results_has_roc_auc(self):
        data = client.get("/api/dashboard").json()
        assert "roc_auc" in data["results"]

    def test_runs_api(self):
        response = client.get("/api/runs")
        assert response.status_code == 200
        data = response.json()
        assert "experiment" in data
        assert "runs" in data
        assert data["experiment"] == "customer-churn"


class TestMonitorAPI:
    def test_monitor_endpoint_returns_200(self):
        response = client.get("/api/monitor")
        assert response.status_code == 200

    def test_monitor_has_operational_key(self):
        data = client.get("/api/monitor").json()
        assert "operational" in data

    def test_monitor_has_drift_check(self):
        data = client.get("/api/monitor").json()
        assert "drift_check" in data

    def test_monitor_has_note_about_ground_truth(self):
        data = client.get("/api/monitor").json()
        assert "note" in data
