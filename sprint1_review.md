# Sprint 1 Review — Data + Reproducible ML Pipeline

## Sprint Goal
Build a deterministic, validated, reproducible ML training pipeline that supports model competition and automatically identifies a champion model, while preserving Skops serialization, existing API interfaces, and test compatibility.

---

## User Stories
- **US-1.1 (Reproducibility):** As an ML Engineer, I need all sources of randomness seeded via centralized configuration so that two consecutive training runs produce identical data splits and metrics.
- **US-1.2 (Data Validation):** As a Data Engineer, I need incoming clinical data validated against an explicit schema (bounds, types, categories) before training or inference to fail fast on corrupted data.
- **US-1.3 (Model Competition):** As an MLOps Engineer, I want multiple candidate models (Logistic Regression, Random Forest, ExtraTrees, XGBoost) trained and evaluated through a single fair pipeline with identical splits and preprocessing.
- **US-1.4 (Champion Selection):** As a Technical Lead, I need an automated, quality-gated selection policy to promote the best model by Macro F1 and fail safely if minimum thresholds are not met.
- **US-1.5 (Structured Metrics):** As a DevOps Engineer, I want versionable, structured `metrics.json` outputs for programmatic evaluation alongside backward-compatible `metrics.txt`.

---

## Completed Tasks
- [x] **S1-T1 (Deterministic Training):** Audited all randomness; centralized seeds in `configs/model_config.yaml` (`shuffle_random_state: 125`, `split_random_state: 125`, `cv_random_state: 125`, model seeds = 125, `n_jobs: 1`).
- [x] **S1-T2 (Data Validation):** Implemented strict Pandera validation in `src/data_validation.py` covering age, sex, BP, cholesterol, Na-to-K ratio, and drug targets.
- [x] **S1-T3 (Structured Metrics):** Added `Results/metrics.json` capturing dataset details, split seeds, all candidate metrics, and champion parameters. Retained `Results/metrics.txt` for CI/CML compatibility.
- [x] **S1-T4 (Model Competition):** Implemented `src/model_registry.py`, `src/preprocessing.py`, and `src/evaluate.py` to evaluate 4 candidate models under 5-fold Stratified CV and identical held-out test splits.
- [x] **S1-T5 (Champion / Challenger Selection):** Implemented `src/champion_selection.py` with minimum quality gates (`min_accuracy: 0.80`, `min_macro_f1: 0.75`, `min_recall: 0.70`), ranking by primary metric, and `NoValidChampionError` safe failure.
- [x] **S1-T6 (Model Artifact):** Preserved secure Skops serialization to `Model/drug_pipeline.skops` with expanded `security.trusted_types` and generated `Model/model_metadata.json`.
- [x] **S1-T7 (Testing):** Expanded test suite from 6 to 25 automated tests covering validation, determinism, candidate registry, champion selection, metrics JSON, and reproducibility.
- [x] **S1-T8 (Documentation):** Updated `docs/ml-pipeline.md`, `docs/architecture.md`, `README.md`, and reconciled historical metric discrepancies.

---

## Files Modified
- `configs/model_config.yaml` — Added data seeds, validation constraints, model candidate configurations, champion selection criteria, and trusted types.
- `requirements.txt` — Added `pandera==0.22.0`, `xgboost==2.1.3`, `pyyaml==6.0.2`.
- `train.py` — Refactored into an 8-step modular, deterministic pipeline with automated model competition and champion promotion.
- `api/main.py` — Updated trusted types loading to support any champion candidate model.
- `tests/test_pipeline.py` — Updated trusted types loading to maintain 100% baseline test compatibility.
- `docs/ml-pipeline.md` — Updated pipeline specifications, preprocessing contracts, and metric reconciliation.
- `docs/architecture.md` — Updated system architecture diagram and component inventory.
- `README.md` — Updated project description, competition results table, and usage instructions.

---

## Files Created
- `src/__init__.py` — Package initialization.
- `src/config.py` — Centralized configuration loader and trusted types helper.
- `src/data_validation.py` — Pandera schema validation functions (`build_schema`, `validate_dataset`, `load_and_validate`).
- `src/preprocessing.py` — Shared `ColumnTransformer` preprocessor factory.
- `src/model_registry.py` — Candidate model factory with `XGBStringClassifier` wrapper for label contract compliance.
- `src/evaluate.py` — 5-Fold Stratified Cross-Validation and test evaluation functions.
- `src/champion_selection.py` — Quality gate filter, primary metric ranker, and leaderboard formatter.
- `tests/test_sprint1.py` — 19 comprehensive unit tests for Sprint 1 capabilities.
- `Results/metrics.json` — Machine-readable evaluation output.
- `Results/competition_report.json` — Ranked leaderboard JSON report.
- `Model/model_metadata.json` — Champion model metadata and hyperparameters.
- `sprint1_review.md` — Formal Sprint 1 report.

---

## Tests
- **Total Tests:** 25 passing (0 failing, 0 skipped)
- **Baseline Tests (`tests/test_pipeline.py`):** 6/6 PASS
  - `test_model_file_exists` (PASS)
  - `test_model_loads` (PASS)
  - `test_prediction_output` (PASS)
  - `test_prediction_shape` (PASS)
  - `test_api_health_endpoint` (PASS)
  - `test_api_predict_endpoint` (PASS)
- **Sprint 1 Tests (`tests/test_sprint1.py`):** 19/19 PASS
  - Data Validation: 7 tests (valid passes, missing column, extra column, invalid datatype, invalid category, out of range, nan/infinite)
  - Deterministic Splitting: 2 tests (identical splits, seed sensitivity)
  - Model Registry & Label Contract: 3 tests (candidate instantiation, unknown class KeyError, string label output)
  - Champion Selection: 4 tests (primary metric rank, quality gate rejection, no valid champion error, leaderboard formatting)
  - Metrics Persistence: 2 tests (metrics.json schema, model_metadata.json existence)
  - Reproducibility: 1 test (pipeline reproducibility across runs)

---

## Reproducibility Results
Verified across three consecutive end-to-end runs:
- **Run 1:** Champion = `xgboost` | Accuracy = 1.0000 | Macro F1 = 1.0000 | CV Macro F1 = 0.9633
- **Run 2:** Champion = `xgboost` | Accuracy = 1.0000 | Macro F1 = 1.0000 | CV Macro F1 = 0.9633
- **Run 3:** Champion = `xgboost` | Accuracy = 1.0000 | Macro F1 = 1.0000 | CV Macro F1 = 0.9633
- **Result:** **100% Deterministic & Reproducible**. All candidate models produced identical accuracy, F1, and cross-validation scores across all 3 runs.

---

## Model Competition Results

| Rank | Model | Parameters | 5-Fold CV Macro F1 | Test Accuracy | Test Macro F1 | Test Macro Recall | Status |
|---|---|---|---|---|---|---|---|
| **1** | **XGBoost** | `n_est=100, seed=125, nthread=1` | **0.9633 ± 0.0330** | **1.0000** | **1.0000** | **1.0000** | **🏆 Champion** |
| **2** | **ExtraTrees** | `n_est=100, seed=125, n_jobs=1` | 0.9489 ± 0.0821 | 0.9833 | 0.9780 | 0.9926 | Challenger #1 |
| **3** | **Random Forest** | `n_est=100, seed=125, n_jobs=1` | 0.8452 ± 0.0665 | 0.9833 | 0.9717 | 0.9600 | Challenger #2 |
| **4** | **Logistic Regression** | `solver=lbfgs, max_iter=2000, seed=125` | 0.8407 ± 0.0598 | 0.9667 | 0.9425 | 0.9526 | Challenger #3 |

All candidates satisfied the quality gate (`accuracy >= 0.80`, `macro_f1 >= 0.75`, `macro_recall >= 0.70`).

---

## Champion Model
- **Algorithm:** XGBoost (`XGBStringClassifier`)
- **Pipeline Architecture:** `ColumnTransformer(OrdinalEncoder + SimpleImputer + StandardScaler) -> XGBStringClassifier`
- **Artifact Path:** `Model/drug_pipeline.skops`
- **Metadata Path:** `Model/model_metadata.json`
- **Output Contract:** Directly outputs native string drug labels (`DrugY`, `drugA`, `drugB`, `drugC`, `drugX`).

---

## Metric Reconciliation
- **Old Documentation Claim:** `97% Accuracy / 0.94 F1`
- **Old Artifact (`Results/metrics.txt`):** `0.95 Accuracy / 0.86 Macro F1` (caused by unseeded `drug_df.sample(frac=1)`)
- **New Verified Baseline (Random Forest):** `98.33% Accuracy / 0.9717 Macro F1`
- **New Champion Baseline (XGBoost):** `100.0% Accuracy / 1.0000 Macro F1` (CV Macro F1 = 0.9633 ± 0.0330)

---

## Technical Debt Added/Resolved
- **Resolved:** Fixed unseeded shuffling in `train.py`.
- **Resolved:** Unified model evaluation into a single fair competition pipeline.
- **Resolved:** Resolved XGBoost string label output contract using `XGBStringClassifier`.
- **Resolved:** Added comprehensive Pandera schema validation preventing silent corruptions.
- **Resolved:** Reconciled historical documentation metric discrepancies.
- **Added:** None. No hacks or temporary bypasses introduced.

---

## CI Status
- `pytest`: 25/25 PASS
- API endpoints: `/health` (HTTP 200) and `/predict` (HTTP 200) functional
- Format & lint compatible

---

## Git Commits
Structured into logical conventional commits:
1. `fix: make training deterministic and seed all pipeline stages`
2. `feat: add pandera data validation layer`
3. `feat: implement multi-model competition and shared preprocessing`
4. `feat: add quality gate and champion selection policy`
5. `feat: add structured metrics and model metadata persistence`
6. `test: add comprehensive sprint 1 test suite`
7. `docs: update ml pipeline, architecture, and reconcile metrics`

---

## Sprint Acceptance
Sprint 1 satisfies all acceptance criteria:
- [x] Training is deterministic
- [x] All randomness is controlled
- [x] Data validation is implemented & invalid data rejected
- [x] `metrics.json` is generated
- [x] Multiple models trained under fair evaluation protocol
- [x] Champion selection is automatic and configurable
- [x] No model promoted if minimum criteria fail
- [x] Existing Skops serialization remains secure
- [x] Existing API continues to function
- [x] Existing and new tests pass (25/25)
- [x] Repeated runs produce reproducible results
- [x] Documentation updated

---

## Recommendation for Sprint 2
1. Introduce MLflow for experiment tracking and metric versioning.
2. Introduce DVC for data version control and dataset provenance.
3. Add SHAP/explainability layer for model interpretability.
4. Prepare model registry integration.
