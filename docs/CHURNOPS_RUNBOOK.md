# ChurnOps Operational Runbook

## Requirements

- Python 3.12
- Git
- A local checkout of this repository

## Setup

```bash
cd ChurnOps
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

Verify tools:

```bash
python --version
dvc --version
mlflow --version
```

## Run the Pipeline

```bash
dvc repro
```

Pipeline stages:
1. Data_Collection  -> data/raw/train.csv + test.csv
2. Data_Preprocessing -> data/processed/*.csv + preprocessor.pkl
3. Model_Training   -> model.pkl + .mlflow_run_id
4. Evaluation       -> metrics.json

Check status:

```bash
dvc status
dvc dag
```

## Run Tests

```bash
pytest -v
```

24 tests expected. All must pass.

## Start the API

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Pages:
- http://localhost:8000/           Project overview
- http://localhost:8000/dashboard  MLflow dashboard
- http://localhost:8000/predict    Prediction form
- http://localhost:8000/docs       Swagger API docs

## MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

Experiment: customer-churn

## Retrain Models

Edit params.yaml, then:

```bash
dvc repro
```

Each run creates new MLflow entries while preserving history.

One-off training without DVC:

```bash
python src/model_training.py --model-type all
python src/model_evaluation.py
```

## Docker

Build:

```bash
docker build -t churnops .
```

Run:

```bash
docker run -p 8000:8000 churnops
```

## Monitoring

Predictions are logged to predictions.jsonl automatically.

Check operational stats:

```
GET /api/monitor
```

For performance evaluation with ground truth:

```python
from src.retrain_trigger import evaluate_with_labels

labeled = [
    {"timestamp": "<ISO timestamp>", "actual_label": 1},
    ...
]
result = evaluate_with_labels(labeled)
print(result["retrain_recommended"], result["roc_auc"])
```

## Common Issues

### ModuleNotFoundError
Activate the venv and verify: python -m pip show mlflow xgboost

### DVC says outputs are tracked by Git
Generated data/raw/*.csv and data/processed/*.csv must be DVC outputs.
Check .gitignore and run dvc repro from the project root.

### Sklearn version warning on unpickling
The model and preprocessor were built with the venv Python.
Re-run the pipeline with the venv activated to regenerate consistent artifacts.
