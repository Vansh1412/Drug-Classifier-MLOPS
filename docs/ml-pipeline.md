# Machine Learning Pipeline

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

## 2. Preprocessing & Feature Engineering

The pipeline uses `sklearn.compose.ColumnTransformer`:
1. **Categorical Features** (`Sex`, `BP`, `Cholesterol`):
   - `OrdinalEncoder()` maps categories to numeric indices.
2. **Numeric Features** (`Age`, `Na_to_K`):
   - `SimpleImputer(strategy="median")` handles any missing values.
   - `StandardScaler()` standardizes features to zero mean and unit variance.

## 3. Estimator Architecture

- **Classifier:** `RandomForestClassifier`
  - `n_estimators`: 10
  - `random_state`: 125 (configured in `configs/model_config.yaml`)
- **Pipeline Integration:**
  Both preprocessing and the classifier are bundled into a single atomic `sklearn.pipeline.Pipeline` object to eliminate train/serving skew.

## 4. Model Persistence via Skops

The model is serialized into `Model/drug_pipeline.skops` using the `skops` library. Unlike pickle, `skops` enforces explicit whitelisting of trusted classes during loading, mitigating arbitrary code execution risks. See `docs/adr/ADR-001-model-serialization.md`.

## 5. Evaluation & Metrics

- **Accuracy Score** on 30% held-out test split
- **Macro F1 Score** to account for multi-class distribution balance
- **Confusion Matrix** visualization saved to `Results/model_results.png`
- **Metrics Log** written to `Results/metrics.txt`
