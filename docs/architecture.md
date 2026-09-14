# MedMLOps-X Architecture

## 1. Overview

**MedMLOps-X** is an enterprise-grade, explainable, quality-gated, observable, and self-monitoring MLOps platform. It demonstrates how machine learning models move safely and deterministically from experimentation to production serving using an educational drug classification problem.

```
+-----------------------------------------------------------------------------------+
|                                  DATA & TRAINING                                  |
|                                                                                   |
|  [Data/drug.csv]  -->  [train.py]  -->  [ColumnTransformer + RandomForest]       |
|                                    -->  [Model/drug_pipeline.skops]               |
|                                    -->  [Results/metrics.txt + model_results.png] |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                              TESTING & VERIFICATION                               |
|                                                                                   |
|  [pyproject.toml]  -->  [tests/test_pipeline.py] (6 pytest automated tests)       |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                              SERVING & INTERFACES                                 |
|                                                                                   |
|  * FastAPI REST API (api/main.py) -- GET /health, POST /predict                   |
|  * Gradio Web UI (App/drug_app.py) -- Interactive HF Spaces frontend             |
|  * Multi-Stage Dockerfile -- Minimal python:3.10-slim runtime                     |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                                CI/CD AUTOMATION                                   |
|                                                                                   |
|  * GitHub Actions CI (.github/workflows/ci.yml) -- format, train, test, docker    |
|  * GitHub Actions CD (.github/workflows/cd.yml) -- gated deploy to HF Spaces      |
|  * deploy.sh -- Linux zero-downtime container replacement with health checks      |
+-----------------------------------------------------------------------------------+
```

## 2. Component Directory

| Component | Path | Responsibility |
|---|---|---|
| **Data Layer** | `Data/drug.csv` | Canonical patient tabular dataset (~200 records) |
| **Pipeline Layer** | `train.py` | Feature preprocessing and model training pipeline |
| **Artifact Store** | `Model/drug_pipeline.skops` | Serialized scikit-learn Pipeline using `skops` |
| **Config Layer** | `configs/model_config.yaml` | Centralized hyperparams, seeds, thresholds, trusted types |
| **API Layer** | `api/main.py` | FastAPI production REST server with Pydantic validation |
| **UI Layer** | `App/drug_app.py` | Gradio interface for Hugging Face Spaces |
| **Test Suite** | `tests/test_pipeline.py` | Pytest unit and integration test suite |
| **Container** | `Dockerfile`, `docker-compose.yml` | Multi-stage containerization with health checks |
| **CI Automation** | `.github/workflows/ci.yml` | Continuous integration and CML evaluation reports |
| **CD Automation** | `.github/workflows/cd.yml` | Branch-guarded continuous deployment to HF Spaces |
| **Host Deploy** | `deploy.sh` | Shell script for zero-downtime Linux container rollout |
| **Tool Config** | `pyproject.toml` | Unified configuration for pytest, black, ruff, coverage |

## 3. Medical Safety Boundary

⚠️ **Important:** MedMLOps-X is an educational machine-learning demonstration and is NOT a certified clinical diagnostic or prescribing tool. Predictions are generated for instructional MLOps demonstration only and must never be utilized as medical advice.
