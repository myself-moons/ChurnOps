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
trial in the versioned `mlruns/` directory, Jenkins automates validation, and
FastAPI exposes the results.

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

git add dvc.lock metrics.json mlruns model.pkl params.yaml
git commit -m "Train models and update tracked runs"
git push origin main
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

## Deploy on Render

This repository includes [render.yaml](render.yaml) for a Render Blueprint.
The web service uses Python 3.12, installs `requirements.txt`, loads the
committed `model.pkl` and `mlruns/` history, and starts FastAPI on Render's
`$PORT`.

### Blueprint deployment

1. Push the repository, including `render.yaml`, to GitHub or GitLab.
2. In Render, select **New > Blueprint** and connect the repository.
3. Select the branch containing `render.yaml` and apply the Blueprint.
4. Wait for the build to finish, then open the generated `onrender.com` URL.

The deployed pages are `/`, `/dashboard`, `/predict`, and `/docs`. The
prediction API is `POST /predict`.

The default configuration uses Render's free plan and Oregon region. Change
`plan` or `region` in `render.yaml` before deploying if needed. No secret
environment variables are required. `MLFLOW_TRACKING_URI` is intentionally
unset, so MLflow reads the committed `mlruns/` directory. Render's filesystem
is ephemeral, but the history is restored from Git on every deployment.

### Manual web-service settings

If you create the service from the Render dashboard instead of the Blueprint,
use:

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Python version | `3.12.3` |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn src.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/docs` |

Do not use `--reload` in the Render start command. Keep the service root at
the repository root so the relative DVC and data paths resolve correctly.

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
- `mlruns/`: versioned MLflow tracking store containing pushed run history

## License and data note

The project uses the local `water_potability (1).csv` source dataset. Generated
raw and processed CSVs are DVC outputs and are not committed directly to Git.
