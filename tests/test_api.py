from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "From raw water data to a tracked model." in response.text


def test_dashboard_page():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Run history &amp; progress" in response.text


def test_prediction_page():
    response = client.get("/predict")
    assert response.status_code == 200
    assert "Test a water sample." in response.text


def test_dashboard_data():
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["tracking"]["run_count"] >= 1
    assert data["datasets"]["train"]["rows"] > 0
    assert "f1_score" in data["results"]


def test_predict_endpoint():
    payload = {
        "ph": 7.0,
        "Hardness": 200.0,
        "Solids": 20000.0,
        "Chloramines": 7.0,
        "Sulfate": 300.0,
        "Conductivity": 450.0,
        "Organic_carbon": 15.0,
        "Trihalomethanes": 60.0,
        "Turbidity": 3.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in {"Water is Consumable", "Water is not Consumable"}
