# MedMLOps-X Architecture

## 1. Overview

**MedMLOps-X** is an enterprise-grade, explainable, quality-gated, observable, and self-monitoring MLOps platform. It demonstrates how machine learning models move safely and deterministically from experimentation to production serving using an educational drug classification problem.

```
                    MEDMLOPS-X ARCHITECTURE
                                │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
       Git                    DVC                   MLflow
        │                      │                      │
    Code (SHA)         Dataset Versioning         Experiments
        │               (Data/drug.csv.dvc)    (sqlite:///mlruns.db)
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ↓
                        Training Pipeline
                               ↓
                       Data Validation (Pandera)
                               ↓
                         Preprocessing
                               ↓
                      Model Competition
                     /    /       \       \
                    LR   RF        ET     XGB
                     \    \       /       /
                      └────┴─────┴───────┘
                               ↓
                           Evaluation
                               ↓
                          Quality Gate
                               ↓
                            Champion
                               ↓
                       Dual Model Artifacts
                      /                   \
        Model/drug_pipeline.skops     MLflow Model Flavor
                     │                         │
                     v                         v
        FastAPI Production Serving     Experiment Registry
                     │
        ┌────────────┴────────────┬────────────────────────┐
        ↓                         ↓                        ↓
  Core Inference           Explainability            Observability
 (/predict, /health)      (SHAP /explain)         (/metrics, /version)
```

## 2. Component Directory

| Component | Path | Responsibility |
|---|---|---|
| **Data Layer** | `Data/drug.csv` | Canonical patient tabular dataset (~200 records) |
| **Data Versioning** | `Data/drug.csv.dvc`, `dvc.yaml` | DVC dataset tracking and stage reproducibility (`dvc repro`) |
| **Experiment Tracking** | `src/tracking.py`, `mlruns.db` | MLflow run tracking, parameter/metric logging, and model logging |
| **Data Validation** | `src/data_validation.py` | Pandera schema enforcing clinical value ranges and datatypes |
| **Model Registry** | `src/model_registry.py` | Factory for multi-model competition (LR, RF, ET, XGBoost) |
| **Preprocessing** | `src/preprocessing.py` | Shared ColumnTransformer (OrdinalEncoder + StandardScaler) |
| **Evaluation Engine** | `src/evaluate.py` | 5-Fold Stratified CV and multi-class metrics computation |
| **Champion Selection** | `src/champion_selection.py` | Quality gate thresholding and primary metric leaderboard |
| **Explainability Service** | `src/explainability.py` | SHAP TreeExplainer service and domain feature attribution mapping |
| **API Metrics Collector** | `src/api_metrics.py` | In-memory thread-safe operational metrics and latency tracker |
| **Pipeline Runner** | `train.py` | Deterministic end-to-end training and competition execution |
| **Config Layer** | `configs/model_config.yaml` | Centralized hyperparams, seeds, thresholds, MLflow/SHAP config |
| **Model Artifact** | `Model/drug_pipeline.skops` | Serialized champion scikit-learn Pipeline using `skops` |
| **Model Metadata** | `Model/model_metadata.json` | Metadata, hyperparameters, and evaluation records |
| **Structured Metrics** | `Results/metrics.json` | Machine-readable metrics for all candidates |
| **API Layer** | `api/main.py` | FastAPI production REST server (/health, /predict, /explain, /version, /metrics) |
| **UI Layer** | `App/drug_app.py` | Gradio interface for Hugging Face Spaces |
| **Test Suite** | `tests/` | 62 automated pytest tests covering baseline, Sprint 1, Sprint 2, and Sprint 3 |
| **Container** | `Dockerfile`, `docker-compose.yml` | Multi-stage containerization with health checks |
| **CI Automation** | `.github/workflows/ci.yml` | Continuous integration and CML evaluation reports |
| **CD Automation** | `.github/workflows/cd.yml` | Branch-guarded continuous deployment to HF Spaces |
| **Host Deploy** | `deploy.sh` | Shell script for zero-downtime Linux container rollout |
| **Tool Config** | `pyproject.toml` | Unified configuration for pytest, black, ruff, coverage |

## 3. Medical Safety Boundary

⚠️ **Important:** MedMLOps-X is an educational machine-learning demonstration and is NOT a certified clinical diagnostic or prescribing tool. Predictions and SHAP feature attributions are generated for instructional MLOps demonstration only and must never be utilized as medical advice.
