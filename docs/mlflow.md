# MLflow Experiment Tracking & Model Lineage Guide

## Overview

**MedMLOps-X** integrates **MLflow** for full-lifecycle experiment tracking, candidate model competition logging, hyperparameter tracking, metric persistence, artifact storage, and model lineage.

Tracking is configured locally via a resilient SQLite database backend, eliminating external cloud dependencies while enabling rich visual analysis.

---

## Configuration

Tracking settings are defined in `configs/model_config.yaml`:

```yaml
mlflow:
  tracking_uri: "sqlite:///mlruns.db"
  experiment_name: "MedMLOps-X-Drug-Classifier"
```

The database `mlruns.db` and any local artifact folders (`mlruns/`, `mlartifacts/`) are excluded from Git via `.gitignore`.

---

## Tracking Architecture

During pipeline execution (`train.py`), two levels of runs are created under the `MedMLOps-X-Drug-Classifier` experiment:

```
Experiment: MedMLOps-X-Drug-Classifier
│
├── Candidate Run 1: logistic_regression
│     ├── Params: model_name, C, penalty, cv_folds, test_size, seed...
│     ├── Metrics: accuracy, macro_f1, macro_recall, cv_macro_f1_mean, train_time_seconds...
│     └── Tags: candidate_name, git_commit, dvc_dataset_hash, python_version, run_type=candidate
│
├── Candidate Run 2: random_forest
├── Candidate Run 3: extra_trees
├── Candidate Run 4: xgboost
│
└── Champion Run: champion_xgboost
      ├── Params: candidate parameters + quality gate thresholds (min_accuracy, min_macro_f1)
      ├── Metrics: champion accuracy, macro_f1, macro_recall, cv metrics, gate_passed=1.0
      ├── Tags: is_champion=true, quality_gate_status=PASSED, primary_metric=macro_f1, git_commit, dvc_dataset_hash...
      ├── Artifacts: metrics.json, competition_report.json, metrics.txt, model_metadata.json, model_results.png
      └── Model: MLflow Scikit-Learn Flavor (Pipeline with Skops serialization & trusted types)
```

---

## Launching the MLflow UI

To start the interactive web UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```

Open `http://localhost:5000` in any web browser to:
1. Inspect the leaderboard across all candidate runs.
2. Filter for champion runs with tag query: `tags.is_champion = 'true'`.
3. Compare hyperparameter sensitivities and cross-validation standard deviations.
4. Download or preview artifacts (confusion matrices, competition reports).

---

## Logged Entities

### 1. Parameters
- `model_name`: Classifier identifier (`logistic_regression`, `random_forest`, `extra_trees`, `xgboost`).
- `cv_folds`: Number of stratified cross-validation folds (`5`).
- `test_size`: Fraction reserved for test evaluation (`0.30`).
- `split_random_state` & `shuffle_random_state`: Global seed (`125`).
- Model-specific hyperparameters (e.g., `n_estimators`, `max_depth`, `learning_rate`, `C`).

### 2. Metrics
- `accuracy`: Test set overall classification accuracy.
- `macro_f1`: Balanced unweighted F1 across all 5 drug classes.
- `macro_recall`: Balanced unweighted recall across all 5 drug classes.
- `cv_macro_f1_mean`: Mean F1 across CV folds.
- `cv_macro_f1_std`: Stability/variance across CV folds.
- `train_time_seconds`: Training duration.

### 3. Lineage Tags
- `git_commit`: Active Git commit SHA (e.g. `9ebf4c8`).
- `dvc_dataset_hash`: Active DVC md5 hash of `Data/drug.csv` (e.g. `7a10a9eb01e3d5664125ea1dd179c8c9`).
- `python_version`: Active Python runtime environment version.
- `champion_model`: Name of the selected champion model.
- `quality_gate_status`: `PASSED` or `FAILED`.

### 4. Model Artifact
The champion model is logged using `mlflow.sklearn.log_model`:
- **Format**: Scikit-Learn `Pipeline` wrapping `ColumnTransformer` preprocessing + champion classifier.
- **Label Contract**: Preserves `XGBStringClassifier` so `.predict()` produces original drug names (`DrugY`, `drugA`, `drugB`, `drugC`, `drugX`).
- **Skops Security**: Deserialization trusted types are registered in the MLmodel configuration for secure loading:
  ```python
  import mlflow.sklearn
  model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
  predictions = model.predict([[30, "M", "HIGH", "NORMAL", 15.4]])
  ```
- **Dual Persistence**: The standalone `Model/drug_pipeline.skops` remains available for the existing FastAPI endpoint.
