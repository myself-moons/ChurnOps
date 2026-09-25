# ChurnOps — Complete Technical & MLOps Audit
**Purpose:** Evaluate ChurnOps as the foundation for CreditOps  
**Date:** 2026-09-25  
**Auditor:** Antigravity  
**Scope:** Read-only inspection. No files modified.

---

## STEP 1 — Repository Inventory

### Structure

```
ChurnOps/
├── src/                       ← all Python source + HTML UI
├── tests/test_api.py          ← single test file, 219 lines
├── docs/
│   ├── CHURNOPS_MIGRATION_PLAN.md   ← detailed migration doc (good)
│   └── CHURNOPS_RUNBOOK.md          ← operational runbook (contains WaterOps residue)
├── Churn_Data/                ← NOT present (Excel files missing from repo)
├── Credit_Data/creditcard.csv ← 144 MB CSV — WRONG DOMAIN DATA (CreditOps data already here)
├── creditcard.csv.zip         ← 66 MB zip — duplicate of above
├── preprocess_backup/processed/
│   ├── train_processed.csv    ← stale backup, not part of DVC pipeline
│   └── test_processed.csv     ← stale backup, not part of DVC pipeline
├── mlruns/                    ← contains 3 experiment folders (0, 258.., 668..)
│   ├── 0/                     ← WaterOps default experiment (artifact_location points to /workspaces/WaterOps)
│   ├── 258.../                ← WaterOps runs (artifact URIs point to C:\Users\kes\.gemini\scratch\WaterOps)
│   └── 668.../                ← WaterOps runs (artifact URIs point to /workspaces/WaterOps)
├── dvc.yaml                   ← 4-stage pipeline (correct)
├── dvc.lock                   ← locked hashes (present)
├── params.yaml                ← model + monitoring config
├── metrics.json               ← evaluation output (LogisticRegression, ROC-AUC 0.9914)
├── model.pkl                  ← 1892 bytes — suspiciously small for any of the 3 models
├── preprocessor.pkl           ← 3875 bytes (present)
├── pipeline.ipynb             ← WaterOps notebook — loads water_potability.csv
├── Steps_for_Pipeline         ← references /workspaces/Jenkins_Test, WaterOps paths
├── render.yaml                ← deployment config (service name: churn-ops-api)
├── JenkinsFile                ← CI pipeline
├── Dockerfile                 ← present
├── requirements.txt           ← 13 pinned deps
├── runtime.txt                ← python-3.12.3
└── conftest.py                ← sys.path setup for pytest
```

### Inventory Assessment

| Item | Status | Note |
|---|---|---|
| `src/` | KEEP | All 12 files are ChurnOps-specific |
| `tests/test_api.py` | KEEP | 24 ChurnOps tests |
| `docs/CHURNOPS_MIGRATION_PLAN.md` | KEEP (archive) | Historical context |
| `docs/CHURNOPS_RUNBOOK.md` | FIX | Contains `cd WaterOps` — wrong directory name |
| `Churn_Data/` | MISSING | Excel source files not committed (expected) |
| `Credit_Data/creditcard.csv` | REMOVE | 144 MB wrong-domain file — belongs in CreditOps |
| `creditcard.csv.zip` | REMOVE | Duplicate of above |
| `preprocess_backup/` | REMOVE | Stale backup, not tracked by DVC, confusing |
| `mlruns/0/`, `mlruns/258../`, `mlruns/668../` | REMOVE | All three are WaterOps experiment runs with stale absolute paths |
| `pipeline.ipynb` | REMOVE | WaterOps notebook — loads `water_potability.csv`, no ChurnOps content |
| `Steps_for_Pipeline` | FIX | References `/workspaces/Jenkins_Test` (old env path) |
| `model.pkl` | INVESTIGATE | 1892 bytes is implausibly small; inspect before trusting |
| `preprocessor.pkl` | KEEP | Correct, 3875 bytes |
| `dvc.yaml`, `dvc.lock` | KEEP | Correct |
| `params.yaml` | KEEP | Correct |
| `metrics.json` | KEEP | Correct output |
| `Dockerfile`, `JenkinsFile`, `render.yaml` | KEEP | Correct |
| `requirements.txt`, `runtime.txt` | KEEP | Correct |

---

## STEP 2 — Data & Leakage Audit

**Files:** [`data_collection.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/src/data_collection.py)

### Target

`Churn Value` — binary integer (0 = retained, 1 = churned). ✅ Correct.

### Leakage Column Handling

| Column | Risk | Decision | Assessment |
|---|---|---|---|
| `Churn Label` | **Critical** — string duplicate of target | Dropped | ✅ Correct |
| `Churn Score` | **Critical** — propensity score derived from churn outcome, computed post-hoc | Dropped | ✅ Correct |
| `CLTV` | **High** — Customer Lifetime Value computed using known churn dates | Dropped | ✅ Correct |
| `Customer ID` | Identifier — no predictive value | Dropped | ✅ Correct |
| `Count`, `Location ID`, `Service ID`, `Status ID` | Surrogate keys | Dropped | ✅ Correct |
| `City`, `Zip Code`, `Latitude`, `Longitude`, `Population` | Geographic — high cardinality | Dropped | ✅ Correct |
| `Under 30`, `Senior Citizen` | Redundant with `Age` | Dropped | ✅ Correct |
| `Dependents` | Redundant with `Number of Dependents` | Dropped | ✅ Correct |
| `Quarter` | Temporal constant (single value in export) | Dropped | ✅ Correct |

**No leakage columns reach the model.** All three critical leakage fields are explicitly dropped before splitting.

### Questionable Features Remaining

**`Satisfaction Score`** — kept as a numeric feature. This warrants scrutiny:
- In a churn dataset, satisfaction is typically measured *at the time of contact*, often by a survey triggered by churn-related events.
- If satisfaction scores were collected *after* the churn decision was made (or as part of an exit survey), this would be post-hoc information and constitute soft leakage.
- However, satisfaction surveys are also routinely collected on a periodic basis regardless of churn, in which case it is a legitimate predictor.
- **Conclusion:** The migration plan keeps it and the code keeps it. Without the raw data dictionary confirming the survey cadence, this is an acceptable decision but should be documented as a known risk. It is not flagged as a hard error.

**`Total Revenue`, `Total Charges`** — these are cumulative financial metrics. For a churned customer, these naturally reflect the full contract period, while a retained customer's values are still accumulating. This creates a mild temporal bias: churned customers may systematically have lower totals simply because they left earlier, not because low spend predicts churn. However, since `Tenure in Months` is also a feature, the model can account for this relationship. This is an acceptable design choice but worth documenting.

### Train/Test Split

```python
train_test_split(df, test_size=0.20, random_state=42, stratify=df["Churn Value"])
```

- Stratification: ✅ Correct — preserves 73.5%/26.5% ratio in both splits.
- Random state: ✅ Fixed at 42 — reproducible.
- Split ratio: ✅ 80/20 — appropriate for 7,043 rows.

### Preprocessing Fit/Transform Logic

In [`data_preprocessing.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/src/data_preprocessing.py):

```python
preprocessor.fit_transform(X_train)   # fitted on train only
preprocessor.transform(X_test)        # transform-only on test
```

✅ **Correct.** The preprocessor is fitted exclusively on training data. Test data is never exposed to the fit step.

### API Preprocessing Consistency

In [`main.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/src/main.py) L284–286:
```python
sample_prepared = prepare_features(sample_df)   # same null-fill + binary encode
sample_encoded  = preprocessor.transform(sample_prepared)  # same fitted transformer
```

✅ **Correct.** The API imports and calls `prepare_features()` from `data_preprocessing.py` and then calls `.transform()` on the loaded `preprocessor.pkl`. There is no duplicated preprocessing logic.

---

## STEP 3 — Train/Validation/Test Methodology

**Files:** [`model_training.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/src/model_training.py), [`model_evaluation.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/src/model_evaluation.py)

### How Training Works

All three models (LR, RF, XGBoost) are trained on `train_processed.csv`.  
After training, **each model is evaluated on `test_processed.csv`**, and the model with the highest **test ROC-AUC** is selected as the winner and saved as `model.pkl`.

```python
# model_training.py L196-199
if test_m["roc_auc"] > best_roc_auc:
    best_model   = clf
    best_roc_auc = test_m["roc_auc"]
    best_run_id  = run.info.run_id
```

### ⚠️ METHODOLOGICAL ISSUE: Model Selection on the Test Set

**This is the most important ML methodology finding in this audit.**

The current pipeline uses the **test set for model selection**, not just for final reporting. This means:

1. Three models are compared using test-set ROC-AUC scores.
2. The best-performing model on the test set is selected.
3. That same model's test-set metrics are then reported as the final evaluation.

**Why this is a problem:** The test set has been used to make a decision (which model to pick). The reported test metrics are therefore optimistically biased — the winning model's performance is partially a result of it getting lucky on this specific test set. The true generalization performance is unknown.

**The correct structure would be:**

```
Training data (80%)
    ↓
Cross-validation (e.g., 5-fold stratified CV) or a held-out validation set
    ↓  ← Model selection happens here
Lock selected model type + hyperparameters
    ↓
Retrain selected model on full training data
    ↓
Evaluate ONCE on the untouched test set (20%)
    ↓
Report final metrics
```

The test set should be touched **exactly once** — for the final evaluation report.

**Practical impact for ChurnOps:** With ROC-AUC values of 0.9914 (LR), 0.9912 (XGBoost), and 0.9844 (RF) on a 7,000-row dataset, the bias from test-set selection is likely small but is nevertheless a methodological error that should be documented and fixed before CreditOps (especially since CreditOps will involve fraud detection, where rigorous methodology is expected).

### Stratification

- Training split: ✅ Stratified (`stratify=df["Churn Value"]` in `data_collection.py`)
- Cross-validation: N/A — CV is not currently used.
- Class balancing in models: ✅ LR uses `class_weight='balanced'`, RF uses `class_weight='balanced'`, XGBoost uses `scale_pos_weight = neg/pos` computed dynamically from training labels.

---

## STEP 4 — Evaluation Methodology

**Reported metrics (from `metrics.json`):**

| Metric | Value | Model |
|---|---|---|
| Model | LogisticRegression | — |
| ROC-AUC | 0.9914 | ← primary |
| Accuracy | 0.9461 | |
| Precision | 0.8652 | |
| Recall | 0.9439 | |
| F1 | 0.9028 | |
| TN / FP / FN / TP | 980 / 55 / 21 / 353 | |

### Metric-by-Metric Assessment for Churn Prediction

**ROC-AUC (primary — 0.9914)**  
ROC-AUC measures the probability that the model ranks a randomly chosen churned customer higher than a randomly chosen retained customer, across all possible decision thresholds. It is threshold-independent and class-imbalance-robust. For a 73.5%/26.5% imbalanced dataset, this is the correct primary metric. A naive majority-class classifier achieves ~0.5 ROC-AUC despite 73.5% accuracy. This metric choice is sound.  
**However:** A ROC-AUC of 0.9914 is extremely high for a real-world churn dataset and should be treated with caution until the train/test methodology issue above is resolved.

**Accuracy (0.9461)**  
Given the class imbalance, accuracy is misleading as a standalone metric. A model predicting "no churn" for all customers achieves 73.5% accuracy. However, it is tracked alongside other metrics, which is acceptable. Its use as a secondary metric is fine.

**Precision (0.8652)**  
Of all customers the model predicted would churn, 86.5% actually did. This is relevant to business cost: false churn predictions lead to unnecessary retention interventions. High precision reduces wasted retention spend.

**Recall (0.9439)**  
Of all customers who actually churned, the model correctly identified 94.4%. This is the most important operational metric for churn — missing a churning customer (false negative) is typically more costly than a false alarm. High recall here is the right property.

**F1 (0.9028)**  
Harmonic mean of precision and recall. Useful as a single-number summary for imbalanced problems. Good secondary metric.

**Confusion matrix (TN=980, FP=55, FN=21, TP=353)**  
At the default 0.5 threshold:
- 21 churners missed (FN) — low, good
- 55 non-churners incorrectly flagged (FP) — acceptable
- The model leans toward catching churners (high recall), which is the right business orientation.

### Are the reported metrics generated by actual code?

✅ Yes. `model_evaluation.py` directly loads `model.pkl` and `test_processed.csv`, computes all metrics using sklearn, and writes `metrics.json`. The metrics are not hard-coded. However they share the test-set selection bias noted in Step 3.

---

## STEP 5 — MLflow Audit

**Files:** [`model_training.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/src/model_training.py), [`model_evaluation.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/src/model_evaluation.py), `mlruns/`

### Configuration

| Setting | Value | Source | Assessment |
|---|---|---|---|
| Experiment name | `customer-churn` | `os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-churn")` | ✅ Correct, env-var configurable |
| Tracking URI | `file:{ROOT_DIR}/mlruns` | `os.getenv("MLFLOW_TRACKING_URI", ...)` | ✅ Correct, env-var configurable |
| Run creation | `mlflow.start_run(run_name=model_type)` | Per model type | ✅ One run per model |

### Logged Parameters

```python
mlflow.log_params(trial_params)
# Contains: model_type, n_estimators, max_depth, random_state, learning_rate,
#           subsample, colsample_bytree, max_iter, C
```

✅ All hyperparameters logged. Parameters match `params.yaml` — no hard-coding.

### Logged Metrics

```python
mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
mlflow.log_metrics({f"test_{k}":  v for k, v in test_m.items()})
```

Logged: `train_accuracy`, `train_precision`, `train_recall`, `train_f1_score`, `train_roc_auc`, and corresponding test_ metrics. ✅

`model_evaluation.py` resumes the best run and adds `eval_` prefixed metrics + confusion matrix cells. ✅ This is a good pattern — training metrics and evaluation metrics in the same run.

### Model / Artifact Logging

```python
mlflow.sklearn.log_model(clf, "model")
```

✅ Models are logged as MLflow artifacts per run.

### Preprocessing Artifact

❌ **The `preprocessor.pkl` is NOT logged to MLflow.** It is saved to disk and used by the API, but it is not tracked as an MLflow artifact. This means if a future run produces a different model and preprocessor, there is no guarantee the MLflow model artifact and the `preprocessor.pkl` on disk are in sync.

**Impact:** Low for current ChurnOps (single environment), but a problem if CreditOps involves multiple environments or remote deployment.

### WaterOps MLflow Contamination

The `mlruns/` directory contains **three experiment folders, all from WaterOps**:
- `mlruns/0/meta.yaml` → `artifact_location: file:///workspaces/WaterOps/mlruns/0`
- `mlruns/668.../meta.yaml` → `name: water-potability` (experiment name)
- `mlruns/258.../meta.yaml` → artifact URIs pointing to `C:\Users\kes\.gemini\antigravity\scratch\WaterOps`

There is **no `customer-churn` MLflow experiment** in the current `mlruns/` directory. This means the current `model.pkl` and `metrics.json` were either:
1. Generated after the mlruns were pushed (i.e., training was run locally but run history was not committed), or
2. Generated and the mlruns folder was not updated in the repository.

The dashboard will show WaterOps experiment runs, not ChurnOps runs, when `mlruns/` is served from the committed state.

---

## STEP 6 — DVC Audit

**Files:** [`dvc.yaml`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/dvc.yaml), [`dvc.lock`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/dvc.lock)

### Stage Definitions

| Stage | Command | Deps | Outs | Assessment |
|---|---|---|---|---|
| `Data_Collection` | `python src/data_collection.py` | `data_collection.py` + 5 Excel files | `data/raw/train.csv`, `data/raw/test.csv` | ✅ Correct |
| `Data_Preprocessing` | `python src/data_preprocessing.py` | `data_preprocessing.py` + raw CSVs | processed CSVs + `preprocessor.pkl` | ✅ Correct |
| `Model_Training` | `python src/model_training.py --model-type all` | `model_training.py` + processed CSVs | `model.pkl`, `.mlflow_run_id` | ✅ Correct |
| `Evaluation` | `python src/model_evaluation.py` | `model_evaluation.py` + `model.pkl` + `preprocessor.pkl` + `.mlflow_run_id` | `metrics.json` | ✅ Correct |

**The bug identified in the migration plan (Data_Collection listing outs as deps) has been fixed.** The current `dvc.yaml` has correct `deps` and `outs`.

### Parameters

`dvc.lock` correctly reflects:
```yaml
params:
  params.yaml:
    model:
      model_type: all
      n_estimators: 200
      max_depth: 8
      ...
```

✅ Parameters in `dvc.lock` match `params.yaml`. DVC will correctly detect changes.

### Dependency Chain

```
5 Excel files → Data_Collection → raw CSVs
raw CSVs → Data_Preprocessing → processed CSVs + preprocessor.pkl
processed CSVs → Model_Training → model.pkl + .mlflow_run_id
model.pkl + preprocessor.pkl + .mlflow_run_id → Evaluation → metrics.json
```

✅ Ordering is correct. No circular dependencies. No inverted deps/outs.

### DVC Lock Consistency

The `dvc.lock` hashes correspond to file sizes matching the actual files on disk (e.g., `preprocessor.pkl` is 3875 bytes in both lock and filesystem). ✅

### Reproducibility Verification Commands

To verify reproducibility **without destroying any data**, run these (read-only):

```bash
# Check if any stage is out of date (safe — does not run anything)
dvc status

# Show the pipeline graph
dvc dag

# Inspect what DVC tracks
dvc list .
```

> [!CAUTION]
> Do NOT run `dvc repro` without first verifying that `Churn_Data/*.xlsx` Excel files are present. The stage will fail if source data is missing and may corrupt the lock file.

---

## STEP 7 — FastAPI Audit

**Files:** [`main.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/src/main.py), [`data_model.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/src/data_model.py)

### Model & Preprocessor Loading

```python
with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)
with PREPROCESSOR_PATH.open("rb") as f:
    preprocessor = pickle.load(f)
```

✅ Both loaded at startup — correct. No per-request loading.

### Prediction Flow (L244–307)

```
Request (Customer JSON)
  → Pydantic Customer model (field validation + type enforcement)
  → Build DataFrame with exact training column names
  → prepare_features() [null-fill + binary encode — same as training]
  → preprocessor.transform() [fitted ColumnTransformer — same as training]
  → model.predict() + model.predict_proba()
  → log_prediction()
  → JSON response
```

✅ **The inference pipeline exactly mirrors the training pipeline.** No shortcuts, no reimplemented logic.

### Request Validation (Pydantic Model)

The `Customer` model uses `Literal` types for all categoricals:
```python
Gender: Literal["Male", "Female"]
Contract: Literal["Month-to-Month", "One Year", "Two Year"]
Internet_Type: Literal["None", "DSL", "Cable", "Fiber Optic"]
```

✅ Any invalid categorical value returns HTTP 422 before reaching the model — confirmed by tests.  
✅ Numeric fields have range bounds (e.g., `Age: int = Field(..., ge=0, le=120)`).

### Feature Order / Column Name Mismatch Risk

The `row` dict in `main.py` L247–280 manually maps Pydantic field names to training column names. This is explicit and correct, but it creates a maintenance risk: if a column is added, renamed, or removed in training, the dict must be updated manually in `main.py`. There is no automated consistency check.

**This is the most important FastAPI finding for CreditOps.**

### Error Handling

❌ There is no `try/except` around `model.predict()` and `preprocessor.transform()`. If an unexpected value reaches the sklearn pipeline at inference time (e.g., a NaN from missing optional field defaults), it will raise an unhandled exception and return HTTP 500 with a raw traceback rather than a clean error message.

### Routes

| Route | Type | Status |
|---|---|---|
| `GET /` | HTML | ✅ Landing page |
| `GET /dashboard` | HTML | ✅ Dashboard |
| `GET /predict` | HTML | ✅ Prediction form |
| `GET /api/dashboard` | JSON | ✅ Dashboard data |
| `GET /api/runs` | JSON | ✅ MLflow runs |
| `GET /api/monitor` | JSON | ✅ Monitoring stats |
| `POST /predict` | JSON | ✅ Prediction endpoint |
| `GET /docs` | Built-in | ✅ Swagger UI |

---

## STEP 8 — Monitoring Audit

**Files:** [`prediction_logger.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/src/prediction_logger.py), [`monitor.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/src/monitor.py)

### Layer A — Operational Monitoring ✅

Available immediately from `predictions.jsonl`. No ground-truth required.

- Total prediction count: ✅
- Churn prediction rate: ✅
- Probability distribution (mean, min, max, std): ✅
- Latency (mean + P95): ✅
- Model version breakdown: ✅

### Layer B — Model Performance Monitoring ✅ (architecture correct)

`monitor.py` exposes `evaluate_with_ground_truth(labeled_outcomes)` which accepts a list of `{timestamp, actual_label}` dicts, matches them against prediction logs, and computes real performance metrics.

✅ **The code correctly distinguishes between operational metrics (available immediately) and model performance metrics (requires ground truth).** This distinction is explicitly documented in both `prediction_logger.py` and `monitor.py`.

### Layer C — Ground-Truth-Dependent Monitoring

The code explicitly acknowledges that precision, recall, F1, and ROC-AUC require actual outcomes:
```python
# prediction_logger.py docstring:
# Prediction logs alone cannot provide true model performance metrics
# (ROC-AUC, precision, recall, F1) because they contain predicted labels
# and probabilities but NOT actual ground-truth outcomes.
```

✅ This is architecturally sound. The limitation is documented, not ignored.

### Missing: Request-Level Feature Logging

`log_prediction()` accepts an optional `input_features: dict | None = None` parameter, but `main.py` **never passes `input_features`**:

```python
# main.py L293–298
log_prediction(
    prediction=predicted_value,
    probability=churn_prob,
    model_version=SELECTED_MODEL,
    latency_ms=latency_ms,
    # input_features NOT passed
)
```

This means the prediction log contains no feature data — only probabilities and predictions. Feature-level drift detection (comparing input distributions over time) is therefore not possible from the logs alone.

---

## STEP 9 — Retraining Audit

**File:** [`retrain_trigger.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/src/retrain_trigger.py)

### Trigger Mechanisms

**1. `evaluate_with_labels(labeled_outcomes)` — Label-based trigger**

Computes ROC-AUC from matched ground-truth labels. Compares against `params.yaml → monitoring.roc_auc_threshold` (default 0.75). Returns `retrain_recommended: True/False`.

✅ Correct architecture. Requires external ground-truth input — this is honest and correct.

**2. `check_distribution_drift(training_churn_rate=0.265, drift_tolerance=0.10)` — Weak signal**

Compares recent prediction churn rate against the 26.5% training rate. Flags if deviation > 10%.

⚠️ **The `training_churn_rate=0.265` is hardcoded as a default parameter** rather than loaded from `params.yaml` or computed from the training data. If retraining produces a model from data with a different base rate, this parameter would silently become stale.

✅ However, the code correctly labels this as a "weak signal" and "early warning only" — it explicitly says it cannot distinguish between concept drift, data pipeline issues, or genuine business changes.

### Is Retraining Safe?

Currently: **Structural placeholder with correct design intent.** The retraining itself is not automated — it requires a human to call `evaluate_with_labels()` with actual outcomes, review the result, and then manually trigger `dvc repro`. This is the correct level of automation for a first iteration.

**The recommended architecture is already implemented:** the trigger separates distribution-based early warnings from label-based performance confirmation.

---

## STEP 10 — Testing Audit

**File:** [`tests/test_api.py`](file:///c:/Users/malha/OneDrive/Documents/GitHub/ChurnOps/tests/test_api.py)

### Test Count & Coverage

| Class | Count | What is tested |
|---|---|---|
| `TestPageRoutes` | 3 | GET / , /dashboard, /predict — status code + content string |
| `TestPredictEndpoint` | 8 | Status, response structure, binary prediction, churn flag, probability range, label values, model field type, high-risk > low-risk ordering |
| `TestInputValidation` | 4 | Missing field → 422, invalid categorical → 422, bad score range → 422, negative age → 422 |
| `TestDashboardAPI` | 5 | Status, required keys, experiment name, ROC-AUC presence, /api/runs structure |
| `TestMonitorAPI` | 4 | Status, operational key, drift_check key, ground-truth note |
| **Total** | **24** | |

### Assessment

✅ The `test_retain_profile_probability` test verifies model behavior, not just HTTP status — a high-risk profile produces higher churn probability than a low-risk profile. This is a genuine behavioral test.

✅ Pydantic validation tests cover the most important invalid inputs.

### Missing Tests

| Missing Test | Severity |
|---|---|
| No test for the `/api/monitor` when `predictions.jsonl` is non-empty | Medium |
| No test for `evaluate_with_ground_truth()` / Layer B performance evaluation | Medium |
| No test for `log_prediction()` side effect (file written to disk) | Low |
| No test for model loading failure (what if `model.pkl` is absent?) | Medium |
| No test for feature count mismatch (extra/missing column in request dict) | Low |
| No preprocessing isolation test (verify `prepare_features` output) | Low |
| No test for the drift trigger (`check_distribution_drift`) | Low |

The README claims 24 tests — this is accurate. The test count in `CHURNOPS_RUNBOOK.md` also says "24 tests expected" — consistent.

---

## STEP 11 — Docker / Jenkins / Deployment Audit

### Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY model.pkl preprocessor.pkl metrics.json ./
COPY mlruns/ ./mlruns/
COPY params.yaml .
RUN touch predictions.jsonl
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

✅ Dependency layer cached before source copy — correct.  
✅ Copies `model.pkl`, `preprocessor.pkl`, `metrics.json` — artifacts baked into image.  
✅ `predictions.jsonl` initialized as empty writable file.  
⚠️ Copies `mlruns/` which currently contains only WaterOps experiments — the bundled dashboard will show stale data until ChurnOps training runs are committed.  
❌ **No `.dockerignore` file.** The `COPY src/` step will copy `__pycache__/` into the image.  
❌ No `Churn_Data/` — correct (source data not bundled), but DVC pipeline cannot run inside the container.

### JenkinsFile

```groovy
stage('Install dependencies') → python3.12 -m venv .venv; pip install -r requirements.txt
stage('Run DVC pipeline')    → dvc repro
stage('Run tests')           → pytest -q
stage('Archive results')     → metrics.json, model.pkl, preprocessor.pkl
```

✅ Pipeline structure is correct and complete.  
⚠️ `stage('Run DVC pipeline')` will fail on a fresh Jenkins agent if `Churn_Data/*.xlsx` Excel files are not available (they are not committed to the repo). The pipeline assumes the agent has access to source data — this needs to be documented or the data made available via DVC remote.  
❌ No notification/alert stage (failure email, Slack, etc.) — acceptable for a first iteration.  
❌ No Docker build stage — the Jenkins pipeline does not build or push a Docker image.

### render.yaml

```yaml
name: churn-ops-api
runtime: python
buildCommand: pip install -r requirements.txt
startCommand: uvicorn src.main:app --host 0.0.0.0 --port $PORT
healthCheckPath: /docs
```

✅ Service name, build command, start command, and health check are all correct.  
⚠️ The Render deployment requires `model.pkl`, `preprocessor.pkl`, `metrics.json`, and `mlruns/` to be committed to the repository — they are, which is correct for this pattern.

### requirements.txt

```
pandas==2.3.2
numpy==1.26.4
scikit-learn==1.5.1
fastapi==0.110.0
uvicorn==0.29.0
pydantic==2.11.7
mlflow==2.17.2
PyYAML==6.0.2
pytest==8.3.3
httpx==0.27.2
dvc==3.67.1
xgboost==2.1.4
openpyxl==3.1.5
```

✅ All 13 dependencies are pinned to exact versions.  
✅ `openpyxl` is present (needed to read Excel files).  
⚠️ `pytest` and `httpx` are runtime dependencies in production Docker image — they should ideally be in a `requirements-dev.txt`. Acceptable for this scale.  
⚠️ `dvc` is a runtime dependency in the Docker image but is never used at inference time — adds ~50 MB to image size unnecessarily.  
✅ `pydantic==2.11.7` — Pydantic v2 is used consistently throughout.

---

## STEP 12 — WaterOps Residue

### Found in Source/Config Files

| File | Line | Content | Action |
|---|---|---|---|
| `pipeline.ipynb` | 42–48 | Loads `water_potability.csv`, `water_potability (1).csv` | **REMOVE** entire notebook |
| `README.md` | 408 | `cd WaterOps` | **FIX** → `cd ChurnOps` |
| `README.md` | 453 | `WaterOps/` in project tree | **FIX** → `ChurnOps/` |
| `docs/CHURNOPS_RUNBOOK.md` | 12 | `cd WaterOps` | **FIX** → `cd ChurnOps` |

### Found in MLflow Tracking Store

All runs in `mlruns/` are WaterOps experiments:

| Path | Residue | Action |
|---|---|---|
| `mlruns/0/meta.yaml` | `artifact_location: file:///workspaces/WaterOps/...` | **REMOVE** entire `mlruns/0/` |
| `mlruns/668../meta.yaml` | `name: water-potability` | **REMOVE** entire `mlruns/668../` |
| `mlruns/258../meta.yaml` | artifact URIs → `C:\Users\kes\.gemini\scratch\WaterOps` | **REMOVE** entire `mlruns/258../` |

**There is no `customer-churn` MLflow experiment in the committed `mlruns/`.** The entire committed `mlruns/` directory contains only WaterOps run history.

### Found in Wrong-Domain Data Files

| File | Size | Content | Action |
|---|---|---|---|
| `Credit_Data/creditcard.csv` | 144 MB | Credit card fraud dataset | **REMOVE** from ChurnOps |
| `creditcard.csv.zip` | 66 MB | Same data, compressed | **REMOVE** from ChurnOps |

### Found in Docs (Intentional — Archive)

The `docs/CHURNOPS_MIGRATION_PLAN.md` deliberately references WaterOps as historical context. This is **not residue** — it is the migration history document. Keep as-is.

---

## STEP 13 — Code Quality & Reusability

### Reusable MLOps Infrastructure (domain-agnostic)

| Component | File | Notes |
|---|---|---|
| DVC 4-stage pipeline pattern | `dvc.yaml` | Change data source deps only |
| MLflow experiment setup | `model_training.py` L134–136 | Change experiment name |
| MLflow run-per-model loop | `model_training.py` L160–199 | Change model types |
| MLflow run resumption | `model_evaluation.py` L48–57 | Unchanged |
| `compute_metrics()` function | `model_training.py` L117–124 | Unchanged |
| Model factory pattern | `model_training.py` L78–114 | Change model types |
| `parse_args() / load_params()` | `model_training.py` L37–72 | Unchanged |
| FastAPI app skeleton | `main.py` | Change domain content |
| FastAPI startup loading | `main.py` L66–72 | Unchanged |
| `_mlflow_runs()` helper | `main.py` L116–153 | Change experiment name |
| Prediction logger JSONL pattern | `prediction_logger.py` | Unchanged |
| Two-layer monitoring architecture | `monitor.py` | Unchanged |
| Ground-truth evaluation function | `monitor.py` L121–174 | Unchanged |
| Retraining trigger structure | `retrain_trigger.py` | Change default churn rate |
| Docker image structure | `Dockerfile` | Unchanged |
| Jenkins pipeline stages | `JenkinsFile` | Unchanged |
| Render deployment config | `render.yaml` | Change service name |
| `conftest.py` sys.path setup | `conftest.py` | Unchanged |
| pytest class-based structure | `tests/test_api.py` | Structure reusable; fixtures replaced |

### Domain-Specific Churn Logic (must be reworked)

| Component | File | What is churn-specific |
|---|---|---|
| Data source (Excel files, merge) | `data_collection.py` | All column names, DROP_COLS list, merge logic |
| Feature definitions | `data_preprocessing.py` | YES_NO_COLS, NUMERIC_COLS, OHE_COLS, FILL_NULLS |
| Pydantic schema | `data_model.py` | All 32 Customer fields |
| Row dict in predict endpoint | `main.py` L247–280 | All field mappings |
| Dashboard experiment name | `main.py` L118 | `"customer-churn"` string |
| `_dataset_summary` target col | `main.py` L100 | `"Churn Value"` string |
| Training base churn rate | `retrain_trigger.py` L103 | `training_churn_rate=0.265` |
| HTML UI — all three pages | `landing.html`, `dashboard.html`, `predict.html` | All text, form fields, column labels |
| Test fixtures | `tests/test_api.py` L23–67 | `CHURN_PAYLOAD`, `RETAIN_PAYLOAD` |

---

## STEP 14 — CreditOps Readiness

### Verdict: **READY AFTER FIXES**

---

### Status Table

| Area | Status | Severity | Finding | Required Action |
|---|---|---|---|---|
| Data Leakage | ✅ CLEAN | — | All three critical leakage columns (Churn Label, Churn Score, CLTV) correctly dropped. Preprocessing fitted on train only. | None |
| Train/Test Methodology | ⚠️ ISSUE | Medium | Model selection uses test-set ROC-AUC — test set is contaminated by selection | Introduce CV or validation split for model selection before CreditOps |
| Model Selection | ⚠️ ISSUE | Medium | Best model chosen by test-set performance — reported metrics are optimistically biased | Fix together with train/test methodology |
| Preprocessing | ✅ CLEAN | — | Fitted on train only; loaded correctly at inference; identical to training pipeline in API | None |
| MLflow | ⚠️ PARTIAL | Low | `preprocessor.pkl` not logged as MLflow artifact; committed `mlruns/` contains only WaterOps history | Log preprocessor; clear WaterOps mlruns |
| DVC | ✅ CLEAN | — | Pipeline structure correct; deps/outs not inverted; parameters tracked; lock consistent | None |
| FastAPI | ⚠️ PARTIAL | Low | No try/except around inference; `input_features` never passed to logger; no `.dockerignore` | Add error handling; log features |
| Monitoring | ✅ CLEAN | — | Two-layer architecture correctly implemented; ground-truth limitation explicitly documented | None |
| Retraining | ⚠️ PARTIAL | Low | `training_churn_rate=0.265` hardcoded as function default; no automated trigger | Move to params.yaml |
| Tests | ✅ GOOD | — | 24 tests, behavioral tests present, good validation coverage | Add model-loading and Layer-B tests |
| Docker | ⚠️ PARTIAL | Low | No `.dockerignore`; `mlruns/` contains WaterOps data; `dvc` and `pytest` in production image | Add `.dockerignore`; clear mlruns |
| Jenkins | ⚠️ PARTIAL | Low | Assumes Excel source data on agent; no Docker build stage | Document data requirement |
| Deployment | ✅ CLEAN | — | `render.yaml` correct; health check present; Python version consistent | None |
| Documentation | ⚠️ PARTIAL | Low | README + Runbook contain `cd WaterOps` residue on 4 lines | Fix 4 lines |
| WaterOps Residue | ❌ PRESENT | Medium | `pipeline.ipynb` (water CSV), `mlruns/` (water experiments), `Credit_Data/`, `creditcard.csv.zip`, README/Runbook text | Remove listed files; fix docs |
| Reusability | ✅ HIGH | — | Infrastructure is cleanly separated from domain logic; 18+ components reusable unchanged | None |

---

### MUST FIX BEFORE CREDITOPS

These are blocking issues. CreditOps should not be started until they are resolved in ChurnOps (or intentionally carried as known debt into CreditOps and fixed there):

1. **Train/Test Methodology — Model Selection on Test Set**  
   Introduce stratified cross-validation on the training set for model selection. Reserve the test set for a single final evaluation after the model type is locked. This affects `model_training.py` and is foundational for CreditOps (fraud detection requires rigorous methodology).

2. **WaterOps MLruns Purge**  
   Delete `mlruns/0/`, `mlruns/258../`, and `mlruns/668../`. They contain only WaterOps experiment data with stale absolute paths pointing to other machines. The dashboard will show incorrect data until this is done.

3. **Remove Wrong-Domain Data Files**  
   Delete `Credit_Data/` (144 MB), `creditcard.csv.zip` (66 MB), and `preprocess_backup/`. These should not be in the ChurnOps repository.

4. **Remove `pipeline.ipynb`**  
   This is the original WaterOps notebook. It loads `water_potability.csv` and has no ChurnOps content. It is confusing and should not exist in this repository.

---

### SHOULD FIX

Important but non-blocking:

5. **Add try/except around inference in `main.py`**  
   Wrap `preprocessor.transform()` and `model.predict()` in a try/except to return HTTP 500 with a clean JSON error body instead of an unhandled traceback.

6. **Log `input_features` in `log_prediction()`**  
   Pass `input_features=row` in `main.py` so feature data is available for drift detection. The function signature already supports it — just needs to be wired up.

7. **Move `training_churn_rate` to `params.yaml`**  
   The hardcoded `0.265` default in `retrain_trigger.py` should be read from `params.yaml → monitoring.training_churn_rate` to stay consistent with actual training data.

8. **Log `preprocessor.pkl` as an MLflow artifact**  
   Add `mlflow.log_artifact(str(ROOT_DIR / "preprocessor.pkl"))` after saving it in `data_preprocessing.py` or at the end of the training run. This ensures the model and preprocessor are always co-located in MLflow.

9. **Fix README.md + CHURNOPS_RUNBOOK.md WaterOps text**  
   Four lines: README L408, L453; Runbook L12. Two-minute fix.

10. **Add `.dockerignore`**  
    Exclude `__pycache__/`, `.git/`, `Churn_Data/`, `data/`, `mlruns/`, `*.ipynb`, `tests/` from the Docker build context.

---

### OPTIONAL

Nice-to-have, do not delay CreditOps:

11. Move `pytest` and `httpx` to `requirements-dev.txt`
12. Move `dvc` out of production Docker image (it is not needed at inference time)
13. Add a Docker build + push stage to `JenkinsFile`
14. Add missing tests for model-loading failure and Layer B evaluation
15. Add `docker-compose.yml` for local development

---

### SAFE TO REUSE UNCHANGED IN CREDITOPS

These files can be copied directly:

| File | What it provides |
|---|---|
| `src/prediction_logger.py` | JSONL prediction logging |
| `src/monitor.py` | Two-layer monitoring + ground-truth evaluation |
| `src/retrain_trigger.py` | Threshold-based retraining trigger (after `training_churn_rate` fix) |
| `conftest.py` | pytest sys.path config |
| `dvc.yaml` (structure) | 4-stage pipeline pattern |
| `JenkinsFile` | CI pipeline stages |
| `Dockerfile` (after `.dockerignore` + mlruns fix) | Container structure |
| `render.yaml` (change service name) | Render deployment |
| `params.yaml` (change values) | Config structure |

---

### MUST BE REWORKED FOR CREDITOPS

| Component | Why |
|---|---|
| `src/data_collection.py` | Credit card data is one CSV, not 5 Excel files; different columns, different target |
| `src/data_preprocessing.py` | Feature definitions, encoders, and NUMERIC_COLS are all churn-specific |
| `src/data_model.py` | All 32 Customer fields are churn-specific; replace with credit transaction schema |
| `src/main.py` L247–280 | Row dict maps churn-specific Pydantic fields; must be replaced |
| `src/main.py` L100, L118 | Hardcoded `"Churn Value"` target and `"customer-churn"` experiment name |
| `src/landing.html` | Churn branding, pipeline description, stats |
| `src/dashboard.html` | `customer-churn` experiment name label, churn-specific metric names |
| `src/predict.html` | Churn prediction form with churn-specific fields and dropdowns |
| `tests/test_api.py` | `CHURN_PAYLOAD`, `RETAIN_PAYLOAD`, "ChurnOps" content assertions |
| `dvc.yaml` deps | Data source paths point to `Churn_Data/*.xlsx`; change to `Credit_Data/creditcard.csv` |
| `README.md` | Full rewrite for CreditOps |

---

### FINAL RECOMMENDATION

**Fix ChurnOps first — 4 must-fixes, then proceed to CreditOps.**

The MLOps infrastructure is well-designed and genuinely reusable. The monitoring architecture, preprocessing pipeline, FastAPI pattern, DVC structure, and Docker/Jenkins/Render setup are all sound and can be carried forward.

The must-fixes are:
1. Model selection methodology (test-set contamination)
2. MLruns purge (WaterOps history)
3. Remove wrong-domain data files (`Credit_Data/`, `creditcard.csv.zip`, `preprocess_backup/`)
4. Remove `pipeline.ipynb`

Fixes 2, 3, and 4 are minutes of work each. Fix 1 requires modifying `model_training.py` to use cross-validation. Together, these four fixes take roughly 2–4 hours. After that, ChurnOps is a clean, correct, and reusable foundation.

Do **not** proceed directly to CreditOps without fix 1 — repeating the test-set selection error in a fraud detection context (where the stakes are higher) would undermine the entire project's credibility.

---

*Audit complete. No files were modified. Awaiting approval to proceed with fixes.*
