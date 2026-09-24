# ChurnOps Migration Plan

**From:** WaterOps — Water Potability MLOps System
**To:** ChurnOps — Customer Churn Prediction MLOps System
**Created:** 2026-09-24
**Status:** PHASE 1 COMPLETE — Awaiting approval to proceed

---

## 1. Current WaterOps Architecture

### 1.1 Execution Flow

```
water_potability (1).csv
        |
src/data_collection.py          --> data/raw/train.csv, data/raw/test.csv
        |
src/data_preprocessing.py       --> data/processed/train_processed.csv
                                    data/processed/test_processed.csv
        |
src/model_training.py           --> model.pkl, .mlflow_run_id
  (RandomForest + XGBoost)          [MLflow experiment: water-potability]
        |
src/model_evaluation.py         --> metrics.json
        |
src/main.py (FastAPI)
  GET  /              --> src/landing.html
  GET  /dashboard     --> src/dashboard.html
  GET  /predict       --> src/predict.html
  POST /predict       --> Water model --> classification result
  GET  /api/dashboard --> JSON dashboard payload
  GET  /api/runs      --> JSON MLflow runs list
  GET  /docs          --> Swagger UI
```

### 1.2 DVC Pipeline

| Stage | Command | Inputs | Outputs |
|---|---|---|---|
| Data_Collection | python src/data_collection.py | source CSV | data/raw/*.csv |
| Data_Preprocessing | python src/data_preprocessing.py | data/raw/*.csv | data/processed/*.csv |
| Model_Training | python src/model_training.py --model-type both | processed CSVs, params.yaml | model.pkl, .mlflow_run_id |
| Evaluation | python src/model_evaluation.py | model.pkl, .mlflow_run_id | metrics.json |

**Structural bug found:** The Data_Collection stage incorrectly lists the generated train/test CSVs
as `deps` rather than `outs`. DVC does not cache them. This will be corrected in ChurnOps.

### 1.3 MLflow Integration

- **Experiment name:** water-potability
- **Tracking URI:** local file:./mlruns (env-var configurable)
- **Logged params:** model_type, n_estimators, max_depth, random_state, learning_rate, subsample, colsample_bytree
- **Logged metrics:** train_accuracy, train_f1_score, test_accuracy, test_f1_score, test_precision, test_recall
- **Logged artifacts:** model via mlflow.sklearn.log_model
- **Run history:** 9 runs (2 model families x multiple parameter sets)

### 1.4 Models Compared

| Model | Library | Selection Criterion |
|---|---|---|
| Random Forest | sklearn.ensemble.RandomForestClassifier | Higher test accuracy |
| XGBoost | xgboost.XGBClassifier | Higher test accuracy |

### 1.5 Current WaterOps Metrics

acc=0.6723, precision=0.7042, recall=0.2049, f1_score=0.3175

Low recall indicates class imbalance challenges in WaterOps — not carried forward.

### 1.6 Frontend

Three self-contained HTML files (inline CSS + vanilla JS):
- src/landing.html  --> /             Pipeline overview, dataset stats
- src/dashboard.html --> /dashboard   MLflow run table, metrics bars
- src/predict.html  --> /predict      9-slider form for water features

### 1.7 Tests

tests/test_api.py — 5 tests, all entirely WaterOps-specific (hardcoded water text, water payloads)

### 1.8 Deployment

- Render: render.yaml (Python 3.12, free plan)
- Jenkins: JenkinsFile (5 stages: checkout, install, dvc repro, pytest, archive)
- No Dockerfile exists

---

## 2. Churn Dataset Details

### 2.1 Source

Location: Churn_Data/ — 5 Excel (.xlsx) files
NOTE: Requires openpyxl — NOT in current requirements.txt (Risk #1)

| File | Rows | Cols | Content |
|---|---|---|---|
| Demographics.xlsx | 7043 | 9 | Age, gender, dependents |
| Location.xlsx | 7043 | 10 | City, state, zip, lat/lon |
| Population.xlsx | 1671 | 3 | Zip-code population |
| Services.xlsx | 7043 | 31 | Subscriptions, contract, charges, tenure |
| Status.xlsx | 7043 | 11 | Churn label, score, satisfaction, CLTV |

Join key: Customer ID across all files. Population joins on Zip Code.

### 2.2 Merged Dataset: 7,043 rows x 47 columns

### 2.3 Target Column

**Churn Value** — binary integer (0 = retained, 1 = churned)
Churn Label is the string equivalent (Yes/No) — perfectly correlated, dropped before training.

### 2.4 Class Distribution

| Class | Count | Pct |
|---|---|---|
| Not Churned (0) | 5174 | 73.5% |
| Churned (1) | 1869 | 26.5% |

Imbalance ratio ~2.77:1 — moderate. Not severe enough for SMOTE, but sufficient to make
accuracy a poor selection metric. ROC-AUC will be the primary evaluation metric.

### 2.5 Feature Decisions

#### DROP — Identifiers / Administrative
- Customer ID, Count (always 1), Location ID, Service ID, Status ID

#### DROP — Direct data leakage
- Churn Label     (string duplicate of target)
- Churn Score     (propensity score derived from churn outcome — critical leakage)
- CLTV            (Customer Lifetime Value computed after churn is known — leakage)

#### DROP — Geographic (high cardinality, low individual-level signal)
- City, Zip Code, Latitude, Longitude, Population

#### DROP — Redundant binary flags
- Under 30, Senior Citizen (redundant with Age)
- Dependents (redundant with Number of Dependents)

#### DROP — Temporal artifact
- Quarter (single constant value in export)

#### KEEP — Numerical features (12)
Age, Number of Dependents*, Number of Referrals, Tenure in Months,
Avg Monthly Long Distance Charges, Avg Monthly GB Download, Monthly Charge,
Total Charges, Total Refunds, Total Extra Data Charges,
Total Long Distance Charges, Total Revenue

*Number of Dependents has min=-2 (data error). Clip to 0 in preprocessing.

#### KEEP — Categorical features (encoded)
Gender, Married, Referred a Friend,
Offer (3877 nulls -> fill "No Offer" -> one-hot),
Phone Service, Multiple Lines, Internet Service,
Internet Type (1526 nulls -> fill "None" -> one-hot),
Online Security, Online Backup, Device Protection Plan, Premium Tech Support,
Streaming TV, Streaming Movies, Streaming Music, Unlimited Data,
Contract (ordinal: Month-to-Month=0, One Year=1, Two Year=2),
Paperless Billing, Payment Method (one-hot), Satisfaction Score (ordinal 1-5)

### 2.6 Preprocessing Architecture

WaterOps: fill numerical nulls with median. No saved preprocessor artifact.

ChurnOps: fitted sklearn Pipeline with ColumnTransformer:
1. Clip Number of Dependents to min=0
2. Fill Offer nulls -> "No Offer"; Internet Type nulls -> "None"
3. Binary-encode Yes/No and Male/Female columns
4. Ordinal-encode Contract
5. One-hot-encode Offer, Internet Type, Payment Method
6. StandardScaler on all numerical features

Fitted pipeline saved as preprocessor.pkl — loaded at API inference time.
This is a significant improvement over WaterOps.

---

## 3. Component Analysis

### 3.1 Reusable Unchanged (Pure infrastructure)

DVC 4-stage pattern, MLflow tracking/URI setup, MLflow run resumption,
best-model selection + pkl save, FastAPI app skeleton, parse_args/load_params,
build_model factory pattern, HTML CSS design system, dashboard JS fetch pattern,
Jenkins pipeline structure, Render config structure, requirements base, test file structure.

### 3.2 Components Requiring Modification

| Component | What Changes |
|---|---|
| src/data_collection.py | Read 5 Excel files, merge, drop leakage/ID cols, split, save CSV |
| src/data_preprocessing.py | Full ColumnTransformer pipeline; save preprocessor.pkl |
| src/model_training.py | Add LR; target=Churn Value; primary=ROC-AUC; balanced weights; experiment=customer-churn |
| src/model_evaluation.py | Add ROC-AUC, confusion matrix; churn target and experiment |
| src/data_model.py | Water -> Customer Pydantic model (~30 fields) |
| src/main.py | Load preprocessor.pkl; transform before predict; return probability; monitoring endpoint |
| src/landing.html | ChurnOps branding and pipeline description |
| src/dashboard.html | ROC-AUC stat labels; customer-churn experiment |
| src/predict.html | Customer churn form (dropdowns + numeric inputs) |
| tests/test_api.py | All fixtures and assertions replaced with churn equivalents |
| dvc.yaml | Fix Data_Collection outs; add preprocessor.pkl; metrics.json as outs |
| params.yaml | Add LR params; add monitoring threshold |
| requirements.txt | Add openpyxl |

### 3.3 New Components Required

| File | Purpose |
|---|---|
| src/prediction_logger.py | Append each prediction to predictions.jsonl |
| src/monitor.py | Rolling performance metrics from prediction log |
| src/retrain_trigger.py | Compare performance vs. threshold; return recommendation |
| Dockerfile | Container build |
| docker-compose.yml | App + optional MLflow service |
| docs/CHURNOPS_RUNBOOK.md | Updated operational runbook |

---

## 4. Proposed ChurnOps Architecture

### 4.1 Updated DVC Pipeline (dvc.yaml)

```yaml
stages:
  Data_Collection:
    cmd: python src/data_collection.py
    deps:
      - src/data_collection.py
      - Churn_Data/Demographics.xlsx
      - Churn_Data/Location.xlsx
      - Churn_Data/Population.xlsx
      - Churn_Data/Services.xlsx
      - Churn_Data/Status.xlsx
    outs:
      - data/raw/train.csv
      - data/raw/test.csv

  Data_Preprocessing:
    cmd: python src/data_preprocessing.py
    deps:
      - src/data_preprocessing.py
      - data/raw/train.csv
      - data/raw/test.csv
    outs:
      - data/processed/train_processed.csv
      - data/processed/test_processed.csv
      - preprocessor.pkl

  Model_Training:
    cmd: python src/model_training.py --model-type all
    deps:
      - data/processed/train_processed.csv
      - data/processed/test_processed.csv
      - src/model_training.py
    params:
      - model
    outs:
      - model.pkl
      - .mlflow_run_id

  Evaluation:
    cmd: python src/model_evaluation.py
    deps:
      - model.pkl
      - preprocessor.pkl
      - .mlflow_run_id
      - src/model_evaluation.py
    outs:
      - metrics.json
```

### 4.2 Models (Phase 3)

| Model | Rationale |
|---|---|
| Logistic Regression | Strong baseline; interpretable; already in sklearn |
| Random Forest | Handles mixed features well; already in requirements |
| XGBoost | Strong tabular performance; already a dependency |

All models use class_weight='balanced' or equivalent.

### 4.3 Primary Selection Metric: ROC-AUC

Chosen because:
- 73.5%/26.5% imbalance makes accuracy misleading
- ROC-AUC is threshold-independent and distribution-independent
- Industry standard for churn/fraud classification
- Naive majority-class classifier achieves 73.5% accuracy but only ~0.5 ROC-AUC

Secondary metrics tracked: accuracy, precision, recall, F1 (macro + churn-class), confusion matrix

### 4.4 Monitoring Design

```
POST /predict -> prediction_logger.py -> predictions.jsonl
                                             |
                                       monitor.py (rolling window)
                                             |
                                       retrain_trigger.py (vs. params.yaml threshold)
                                             |
                                       GET /api/monitor -> JSON status
```

Prediction log entry:
```json
{
  "timestamp": "ISO-8601",
  "prediction": 0,
  "probability": 0.23,
  "model_version": "XGBClassifier",
  "latency_ms": 4.2
}
```

params.yaml will include monitoring.roc_auc_threshold (default 0.75).

---

## 5. Risks and Issues

| # | Risk | Severity | Fix |
|---|---|---|---|
| 1 | openpyxl not in requirements.txt | High | Add openpyxl>=3.1.0 |
| 2 | DVC Data_Collection bug (outs listed as deps) | Medium | Fix dvc.yaml |
| 3 | No preprocessor.pkl in WaterOps | Medium | Save ColumnTransformer; load in API |
| 4 | Churn Score leakage (derived from churn outcome) | Critical | Drop before training |
| 5 | CLTV leakage (computed using churn dates) | High | Drop before training |
| 6 | Number of Dependents min=-2 (data error) | Low | Clip to 0 |
| 7 | No Dockerfile | Low | Add Dockerfile + docker-compose.yml |
| 8 | predict.html uses sliders; churn has categoricals | Medium | Redesign with dropdowns |
| 9 | All 5 tests are WaterOps-specific | Medium | Replace with churn equivalents |
| 10 | Dashboard shows accuracy as primary stat | Low | Update labels to ROC-AUC |

---

## 6. Phase Execution Order (After Approval)

| Phase | Scope | Key outputs |
|---|---|---|
| 2 | Dataset adaptation | data_collection.py, data_preprocessing.py, preprocessor.pkl |
| 3 | Modeling | LR+RF+XGB, ROC-AUC primary, balanced weights |
| 4 | DVC | Corrected dvc.yaml, regenerated dvc.lock |
| 5 | MLflow | customer-churn experiment, ROC-AUC metrics |
| 6 | Model selection | ROC-AUC best-model logic |
| 7 | FastAPI | Preprocessor load, probability output, monitoring endpoint |
| 8 | UI | ChurnOps HTML, customer churn form |
| 9 | Monitoring | prediction_logger.py, monitor.py, retrain_trigger.py |
| 10 | Retraining | params.yaml threshold, documented workflow |
| 11 | Testing | New test_api.py for churn |
| 12 | Docker | Dockerfile, docker-compose.yml |
| 13 | CI/Jenkins | Updated JenkinsFile |
| 14 | Docs | README.md, CHURNOPS_RUNBOOK.md |
