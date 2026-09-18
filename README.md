# Water Potability ML Pipeline

This repository contains a reproducible water-potability classification system:

```text
Source CSV -> DVC data collection -> preprocessing -> model training -> evaluation
                                                   |                     |
                                                   +-- MLflow run --------+
FastAPI overview, prediction UI, and experiment dashboard
```

The project supports both `random_forest` and `xgboost` without replacing
historical runs. DVC controls the pipeline and parameters, MLflow records each
trial, Jenkins automates validation, and FastAPI exposes the results.

## Quick start

```bash
cd /workspaces/Jenkins_Test
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

dvc dag
dvc repro
pytest -q
```

Runbook: [Steps_for_Pipeline](Steps_for_Pipeline)

## Run the web application

```bash
source .venv/bin/activate
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Open these pages:

| URL | Purpose |
| --- | --- |
| `http://localhost:8000/` | Project overview and pipeline landing page |
| `http://localhost:8000/dashboard` | MLflow run history and comparison dashboard |
| `http://localhost:8000/predict` | Visual prediction form |
| `http://localhost:8000/docs` | Interactive API documentation |

The JSON endpoints are `/api/dashboard` and `/api/runs`. The prediction API
is `POST /predict`.

## Run and compare models

Edit [params.yaml](params.yaml), then run:

```bash
dvc repro
```

Select a model with `model.model_type`:

```yaml
model:
  model_type: xgboost
  n_estimators: 150
  max_depth: 6
  random_state: 42
  learning_rate: 0.1
  subsample: 0.9
  colsample_bytree: 0.9
```

Use `random_forest` to run the Random Forest implementation. Every training
run logs model type, parameters, training metrics, test metrics, and the model
artifact to the `water-potability` MLflow experiment.

View runs locally:

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000` and compare model families. A shared team setup
should set `MLFLOW_TRACKING_URI` to a persistent MLflow server before running
`dvc repro`.

## Jenkins

The committed [JenkinsFile](JenkinsFile) checks out the repository, creates a
Python 3.12 environment, installs dependencies, runs `dvc repro`, runs tests,
and archives `metrics.json` and `model.pkl`.

Configure a Jenkins Pipeline job with:

- Definition: `Pipeline script from SCM`
- Branch: `main`
- Script path: `JenkinsFile`
- Python 3.12 available on the agent

For setup details and common errors, use [Steps_for_Pipeline](Steps_for_Pipeline).

## Repository map

- `dvc.yaml`, `dvc.lock`: reproducible pipeline definition and lock state
- `params.yaml`: model family and hyperparameters
- `src/model_training.py`: model factory and MLflow training logging
- `src/model_evaluation.py`: test metrics and MLflow evaluation logging
- `src/main.py`: FastAPI routes and dashboard data APIs
- `src/landing.html`, `src/dashboard.html`, `src/predict.html`: web views
- `tests/test_api.py`: API and page regression tests
- `mlruns/`: local MLflow tracking store, ignored by Git

## License and data note

The project uses the local `water_potability (1).csv` source dataset. Generated
raw and processed CSVs are DVC outputs and are not committed directly to Git.
