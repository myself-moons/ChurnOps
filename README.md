# ChurnOps

## Customer Churn Prediction MLOps System

A reproducible, end-to-end MLOps pipeline that trains and serves a customer churn classifier.

You can access the live, deployed model here: **[ChurnOps Web App](https://churnops-ss5e.onrender.com/)**

```
Churn_Data Excel files
        ↓  DVC
Data Collection (merge + split)
        ↓
Data Preprocessing (encode + scale → preprocessor.pkl)
        ↓
Model Training (LR / RF / XGBoost → model.pkl)
        ↓  MLflow
Experiment Tracking (customer-churn experiment)
        ↓
Model Evaluation (ROC-AUC, F1, Precision, Recall, Confusion Matrix)
        ↓
Best Model (selected by ROC-AUC)
        ↓
FastAPI Serving (POST /predict)
        ↓
Prediction UI + MLflow Dashboard
        ↓
Prediction Logging (predictions.jsonl)
        ↓
Operational Monitoring (/api/monitor)
        ↓
Retraining Trigger (threshold-based, ground-truth-aware)
```

---

## Problem Statement

Customer churn — the loss of subscribers to a competitor or cancellation — is one of the most costly problems for subscription-based businesses. This project builds a complete MLOps pipeline that:

- Trains multiple classification models on real telecom customer data
- Selects the best model by ROC-AUC (not accuracy — see [Evaluation Metrics](#evaluation-metrics))
- Exposes a REST API for real-time churn predictions
- Logs predictions for operational monitoring
- Provides a clear architecture for performance-based retraining

---

## Dataset

**Source:** 5 Excel files in `Churn_Data/`

| File | Rows | Content |
|---|---|---|
| `Demographics.xlsx` | 7,043 | Age, gender, dependents |
| `Location.xlsx` | 7,043 | City, zip, lat/lon |
| `Population.xlsx` | 1,671 | Zip-code population lookup |
| `Services.xlsx` | 7,043 | Subscriptions, contract, charges, tenure |
| `Status.xlsx` | 7,043 | Churn label, satisfaction, CLTV |

**Join key:** `Customer ID` (all files). Population joined via `Zip Code`.

**Merged shape:** 7,043 rows × 47 columns before cleaning.

**Target:** `Churn Value` — binary (0 = retained, 1 = churned)

**Class distribution:** 73.5% retained / 26.5% churned (imbalance ratio ~2.77:1)

### Feature Decisions

See `docs/CHURNOPS_MIGRATION_PLAN.md` for the full decision rationale.

**Dropped columns (leakage / identifiers / geographic):**
- Identifiers: `Customer ID`, `Count`, `Location ID`, `Service ID`, `Status ID`
- Leakage: `Churn Label` (duplicate of target), `Churn Score` (post-hoc propensity), `CLTV` (computed after churn)
- Geographic: `City`, `Zip Code`, `Latitude`, `Longitude`, `Population`
- Redundant: `Under 30`, `Senior Citizen`, `Dependents`, `Quarter`

**Final feature set:** 12 numerical + 19 categorical = ~30 encoded features

---

## Architecture

```
src/
  data_collection.py      Merge Excel files → stratified train/test split
  data_preprocessing.py   sklearn ColumnTransformer → preprocessor.pkl
  model_training.py       LR + RF + XGBoost, ROC-AUC selection → model.pkl
  model_evaluation.py     Full metrics → metrics.json
  data_model.py           Customer Pydantic model (API validation)
  main.py                 FastAPI application
  prediction_logger.py    Append predictions to predictions.jsonl
  monitor.py              Operational + performance monitoring
  retrain_trigger.py      ROC-AUC threshold-based retraining decision
  landing.html            Project overview page
  dashboard.html          MLflow run comparison dashboard
  predict.html            Customer churn prediction form
tests/
  test_api.py             24 tests covering API, validation, structure
docs/
  CHURNOPS_MIGRATION_PLAN.md  Full migration analysis and decision log
  CHURNOPS_RUNBOOK.md         Operational setup and runbook
```

---

## ML Pipeline

### 1. Data Collection (`src/data_collection.py`)

- Reads 5 Excel files from `Churn_Data/`
- Merges on `Customer ID` (and `Zip Code` for population)
- Drops leakage, identifier, and geographic columns
- Clips `Number of Dependents` negative values to 0
- Stratified 80/20 train/test split (preserves churn class ratio)
- Outputs: `data/raw/train.csv`, `data/raw/test.csv`

### 2. Data Preprocessing (`src/data_preprocessing.py`)

Built as a fitted **sklearn `ColumnTransformer` pipeline**:

| Transformer | Applied to |
|---|---|
| `StandardScaler` | 12 numerical features |
| `OrdinalEncoder` | `Contract` (ordered: Month-to-Month < One Year < Two Year) |
| `OneHotEncoder` | `Offer`, `Internet Type`, `Payment Method` |
| Binary mapping | Yes/No, Male/Female columns |

**Fitted on training data only.** Saved as `preprocessor.pkl` and loaded by the API at inference time — no preprocessing logic is duplicated.

### 3. Model Training (`src/model_training.py`)

Three classifiers trained and tracked in parallel:

| Model | Class Balancing |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` (ratio of negatives/positives) |

The best model (by mean CV ROC-AUC across 5 stratified folds on training data) is retrained on the complete training split and saved as `model.pkl`.

---

## Models Evaluated

| Model | Test ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression ★ | **0.9914** | 0.9028 | 0.8652 | 0.9439 | 0.9461 |
| Random Forest | 0.9844 | 0.9040 | — | — | 0.9489 |
| XGBoost | 0.9912 | 0.9285 | — | — | 0.9624 |

**Selected model: Logistic Regression** (highest ROC-AUC)

---

## Evaluation Metrics

**Primary metric: ROC-AUC**

The class distribution (73.5% / 26.5%) makes accuracy a poor selection criterion — a classifier that predicts "not churned" for every customer achieves 73.5% accuracy but ROC-AUC of only ~0.5.

ROC-AUC measures the model's discriminative ability across all classification thresholds, independent of class distribution. It is the industry standard for churn and fraud classification.

**All tracked metrics:**
- ROC-AUC (primary / model selection)
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix (TN, FP, FN, TP)

---

## MLflow Usage

Experiment name: `customer-churn`

Every training run logs:
- Model type and hyperparameters
- Train and test metrics for all 5 metrics
- Model artifact (`mlflow.sklearn.log_model`)

View run history locally:

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

Or browse the built-in dashboard at `http://localhost:8000/dashboard` after starting the API.

---

## DVC Usage

The pipeline is fully reproducible through DVC:

```bash
dvc repro
```

Pipeline stages:

| Stage | Inputs | Outputs |
|---|---|---|
| `Data_Collection` | 5 Excel files | `data/raw/train.csv`, `data/raw/test.csv` |
| `Data_Preprocessing` | Raw CSVs | Processed CSVs, `preprocessor.pkl` |
| `Model_Training` | Processed CSVs | `model.pkl`, `.mlflow_run_id` |
| `Evaluation` | model.pkl, preprocessor.pkl | `metrics.json` |

Visualise the DAG:

```bash
dvc dag
```

---

## FastAPI

Start the server:

```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

| URL | Purpose |
|---|---|
| `http://localhost:8000/` | Project overview |
| `http://localhost:8000/dashboard` | MLflow run comparison |
| `http://localhost:8000/predict` | Customer churn prediction form |
| `http://localhost:8000/docs` | Interactive API documentation |
| `http://localhost:8000/api/dashboard` | Dashboard data as JSON |
| `http://localhost:8000/api/runs` | MLflow run history as JSON |
| `http://localhost:8000/api/monitor` | Operational monitoring stats |

### Prediction API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "Total Revenue": 190.0
  }'
```

**Response:**

```json
{
  "prediction": 1,
  "churn": true,
  "prediction_label": "Likely to Churn",
  "churn_probability": 0.9823,
  "model": "LogisticRegression",
  "latency_ms": 4.2
}
```

---

## UI

The prediction form at `/predict` accepts the full customer profile through grouped dropdown and numeric input fields. No manual preprocessing is required — the API applies `preprocessor.pkl` automatically.

The dashboard at `/dashboard` displays:
- ROC-AUC as the primary comparison metric
- All tracked metrics for every MLflow run
- Model type color coding (LR / RF / XGBoost)
- Confusion matrix from `metrics.json`

---

## Monitoring

Every prediction is logged to `predictions.jsonl`:

```json
{
  "timestamp": "2026-09-24T09:00:00+00:00",
  "prediction": 1,
  "probability": 0.9823,
  "model_version": "LogisticRegression",
  "latency_ms": 4.2
}
```

The `/api/monitor` endpoint returns:

**Layer A — Operational (immediate, no labels required):**
- Total prediction count
- Churn prediction rate
- Probability distribution (mean, min, max, std)
- Latency (mean and P95)
- Model version usage

**Layer B — Performance (requires ground-truth labels):**

> Real model performance metrics (ROC-AUC, Precision, Recall, F1) require actual churn outcomes. In a telecom context, these are only known 30–90 days after prediction. The `monitor.py:evaluate_with_ground_truth()` function accepts labeled outcomes and computes real metrics when available.

---

## Retraining Strategy

The retraining trigger compares measured ROC-AUC against the threshold in `params.yaml`:

```yaml
monitoring:
  roc_auc_threshold: 0.75
  min_predictions: 50
```

**Workflow:**

1. Predictions are logged to `predictions.jsonl` (timestamp + probability)
2. After 30–90 days, actual churn outcomes become available from CRM/billing
3. Match outcomes to prediction logs by timestamp
4. Call `retrain_trigger.evaluate_with_labels(labeled_outcomes)`
5. If ROC-AUC < threshold → `retrain_recommended: true`
6. Re-run `dvc repro` to retrain with updated data

An early-warning signal (without labels) is available via `check_distribution_drift()`, which flags significant shifts in the predicted churn rate distribution.

---

## Testing

```bash
pytest -v
```

**24 tests** covering:
- HTML page routes (ChurnOps content verification)
- Prediction response structure and types
- Binary prediction / churn flag consistency
- Probability range [0.0, 1.0]
- Prediction label enumeration
- High-risk vs low-risk profile ordering
- Missing field → HTTP 422
- Invalid categorical value → HTTP 422
- Out-of-range score → HTTP 422
- Dashboard API structure and experiment name
- ROC-AUC in results
- Monitoring endpoint structure and notes

---

## Docker

Build:

```bash
docker build -t churnops .
```

Run:

```bash
docker run -p 8000:8000 churnops
```

Open `http://localhost:8000`.

The image is self-contained — `model.pkl`, `preprocessor.pkl`, `metrics.json`, and `mlruns/` are bundled so the dashboard has full run history without re-training.

---

## How to Run Locally

```bash
# Clone and navigate
cd ChurnOps

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install --upgrade pip
pip install -r requirements.txt

# Run the full pipeline
dvc repro

# Run tests
pytest -v

# Start the API
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

---

## How to Reproduce the Pipeline

```bash
# Reproduce all stages
dvc repro

# View pipeline graph
dvc dag

# Check pipeline status
dvc status

# Run MLflow UI separately
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

Changing `params.yaml` and re-running `dvc repro` creates new MLflow runs while preserving previous runs for comparison.

---

## Project Structure

```
ChurnOps/
├── Churn_Data/             Raw Excel source files (unchanged)
├── data/
│   ├── raw/                DVC output — train/test splits
│   └── processed/          DVC output — encoded features
├── src/
│   ├── data_collection.py  Data merge and split
│   ├── data_preprocessing.py  sklearn pipeline + preprocessor.pkl
│   ├── model_training.py   LR + RF + XGBoost, MLflow tracking
│   ├── model_evaluation.py Full metrics + metrics.json
│   ├── data_model.py       Customer Pydantic model
│   ├── prediction_logger.py  Prediction logging
│   ├── monitor.py          Two-layer monitoring
│   ├── retrain_trigger.py  Retraining decision
│   ├── main.py             FastAPI application
│   ├── landing.html        Overview page
│   ├── dashboard.html      MLflow dashboard
│   └── predict.html        Prediction form
├── tests/
│   └── test_api.py         24-test suite
├── docs/
│   ├── CHURNOPS_MIGRATION_PLAN.md
│   └── CHURNOPS_RUNBOOK.md
├── dvc.yaml                Pipeline definition
├── dvc.lock                Reproducibility lock
├── params.yaml             Model + monitoring config
├── metrics.json            Latest evaluation results
├── model.pkl               Best trained model
├── preprocessor.pkl        Fitted sklearn pipeline
├── predictions.jsonl       Prediction log
├── conftest.py             Pytest path configuration
├── requirements.txt        Python dependencies
├── Dockerfile              Container build
├── JenkinsFile             CI pipeline
└── render.yaml             Render deployment
```

---

## CI / Jenkins

The `JenkinsFile` runs:
1. Checkout
2. Create Python 3.12 virtual environment
3. Install `requirements.txt`
4. `dvc repro` (full pipeline)
5. `pytest -q` (all tests)
6. Archive `metrics.json`, `model.pkl`, `preprocessor.pkl`

Configure a Pipeline job pointing to `JenkinsFile` with Python 3.12 available on the agent.
