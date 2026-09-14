# Machine Learning Pipeline — MedMLOps-X

## 1. Problem Formulation

The machine learning use case is multi-class classification predicting the appropriate drug category for a patient based on clinical biomarkers.

- **Dataset:** `Data/drug.csv` (200 patient records)
- **Target Label:** `Drug` (`DrugY`, `drugA`, `drugB`, `drugC`, `drugX`)
- **Features:**
  - `Age` (Numeric, range 15–74)
  - `Sex` (Categorical: `M`, `F`)
  - `BP` (Categorical: `HIGH`, `LOW`, `NORMAL`)
  - `Cholesterol` (Categorical: `HIGH`, `NORMAL`)
  - `Na_to_K` (Numeric: Sodium-to-Potassium ratio in blood, range 6.2–38.2)

---

## 2. Data Validation Layer (Pandera)

Data validation is performed prior to any preprocessing or training using a strict Pandera schema defined in `src/data_validation.py` driven by constraints in `configs/model_config.yaml`:
- **Age:** Integer, bounds [15, 74], non-null
- **Sex:** String, category whitelist `['F', 'M']`
- **BP:** String, category whitelist `['HIGH', 'LOW', 'NORMAL']`
- **Cholesterol:** String, category whitelist `['HIGH', 'NORMAL']`
- **Na_to_K:** Float, bounds [6.0, 39.0], finite, non-null
- **Drug:** String, target category whitelist `['DrugY', 'drugA', 'drugB', 'drugC', 'drugX']`
- **Contract Enforcement:** Strict schema rejects uncontracted extra columns, invalid types, and out-of-range values with actionable error diagnostics.

---

## 3. Shared Preprocessing Contract

All candidate models share an identical preprocessing contract built by `src/preprocessing.py` using `sklearn.compose.ColumnTransformer`:
1. **Categorical Features** (`Sex`, `BP`, `Cholesterol`):
   - `OrdinalEncoder()` maps categories into numerical representations.
2. **Numeric Features** (`Age`, `Na_to_K`):
   - `SimpleImputer(strategy="median")` handles any missing values safely.
   - `StandardScaler()` standardizes features to zero mean and unit variance.

Each model pipeline encapsulates both preprocessing and estimator in a single atomic `sklearn.pipeline.Pipeline`, preventing training/serving skew.

---

## 4. Deterministic Training & Model Competition

All sources of randomness are centralized in `configs/model_config.yaml`:
- `shuffle_random_state: 125` (dataset shuffle `drug_df.sample`)
- `split_random_state: 125` (stratified 70/30 train/test split)
- `cv_random_state: 125` (5-fold Stratified K-Fold CV)
- Estimator seeds fixed (`random_state=125` / `seed=125`, `n_jobs=1` / `nthread=1`)

### Candidate Models (`src/model_registry.py`):
1. **Logistic Regression** (L-BFGS solver, `max_iter=2000`)
2. **Random Forest** (`n_estimators=100`)
3. **ExtraTrees** (`n_estimators=100`)
4. **XGBoost** (`XGBStringClassifier` wrapper ensuring string class label contract)

Each candidate is evaluated using:
- 5-Fold Stratified Cross-Validation on the training set
- Final evaluation on the held-out test split (30%, 60 records)

---

## 5. Champion Selection Policy (`src/champion_selection.py`)

1. **Quality Gate Filtering:**
   Candidates must pass configured minimum thresholds:
   - `min_accuracy: 0.80`
   - `min_macro_f1: 0.75`
   - `min_recall: 0.70`
2. **Primary Metric Ranking:**
   Passing models are ranked descending by `primary_metric: macro_f1`.
3. **Safe Failure:**
   If no candidate passes the quality gate, `NoValidChampionError` is raised and training halts safely. No poor model is ever silently promoted.

---

## 6. Artifacts & Metrics Structure

- **Model Artifact:** `Model/drug_pipeline.skops` (Champion model serialized with explicit trusted types whitelist)
- **Model Metadata:** `Model/model_metadata.json` (Champion metadata, hyperparameters, CV metrics, split seeds)
- **Structured Metrics:** `Results/metrics.json` (Machine-readable benchmark covering all candidates, runtime, and dataset split details)
- **Competition Report:** `Results/competition_report.json` (Full ranked leaderboard)
- **CML Compatibility:** `Results/metrics.txt` (Text summary for GitHub Actions pull request comments)
- **Visualizations:** `Results/model_results.png` (Confusion matrix of the champion model)

---

## 7. Metric Reconciliation & Baseline

| Model | CV Macro F1 | Test Accuracy | Test Macro F1 | Status |
|---|---|---|---|---|
| **XGBoost** | 0.9633 ± 0.0330 | **1.0000** | **1.0000** | **Champion** |
| **ExtraTrees** | 0.9489 ± 0.0821 | 0.9833 | 0.9780 | Challenger #1 |
| **Random Forest** | 0.8452 ± 0.0665 | 0.9833 | 0.9717 | Challenger #2 |
| **Logistic Regression** | 0.8407 ± 0.0598 | 0.9667 | 0.9425 | Challenger #3 |

### Reconciliation Note:
- **Historical Discrepancy:** The initial repository README claimed `97.0% Accuracy / 0.94 F1` while `Results/metrics.txt` reported `Accuracy = 0.95, F1 Score = 0.86` due to unseeded shuffling (`drug_df.sample(frac=1)`).
- **Verified Baseline:** Under the deterministic Sprint 1 pipeline (`seed=125`), the true baseline for Random Forest is **98.33% Accuracy / 0.9717 Macro F1**. Model competition identified **XGBoost** as Champion with **100.0% Accuracy / 1.0000 Macro F1**, fully reproduced across multiple independent runs.
