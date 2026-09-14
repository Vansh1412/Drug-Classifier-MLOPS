# Drug Classifier MLOps

[![CI](https://github.com/Vansh1412/Drug-Classifier-MLOPS/actions/workflows/ci.yml/badge.svg)](https://github.com/Vansh1412/Drug-Classifier-MLOPS/actions/workflows/ci.yml)
[![CD](https://github.com/Vansh1412/Drug-Classifier-MLOPS/actions/workflows/cd.yml/badge.svg)](https://github.com/Vansh1412/Drug-Classifier-MLOPS/actions/workflows/cd.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vansh1412/Drug-Classifier-MLOPS/blob/main/notebook.ipynb)

An end-to-end MLOps platform demonstrating automated data validation, deterministic multi-model competition, champion selection, testing, containerization, and continuous deployment.

## Project Description

This project builds a **Drug Classification** platform predicting the appropriate medication class based on patient biomarkers (Age, Sex, Blood Pressure, Cholesterol, Na-to-K ratio). The training pipeline is fully reproducible and selects a champion model from multiple competing algorithms based on automated quality gates.

**Key Features:**
- 📦 **Data & Pipeline Versioning** — DVC tracks canonical dataset hashes and pipeline DAG (`dvc repro`)
- 🧪 **Experiment Tracking** — MLflow tracks parameters, metrics, cross-validation runs, artifacts, and champion lineage
- 🛡️ **Data Validation** — Pandera schema enforcing value ranges, biological limits, and data types
- 🎲 **Deterministic ML** — Centralized seed configuration ensuring 100% reproducible splits and metrics
- 🏆 **Model Competition** — Automated evaluation of Logistic Regression, Random Forest, ExtraTrees, and XGBoost
- 🥇 **Champion Selection** — Quality gate thresholding (Macro F1, Accuracy, Recall) with automated champion promotion
- 🌐 **REST API** — FastAPI server with `POST /predict` and `GET /health` endpoints
- 🔒 **Safe Serialization** — Skops model persistence with strict trusted type whitelisting
- ✅ **Automated Testing** — 39 pytest tests covering baseline, Sprint 1, and Sprint 2 functionality
- ⚙️ **CI/CD** — GitHub Actions pipeline: format → train → test → evaluate → deploy

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data & Pipeline Versioning | DVC (Data Version Control) |
| Experiment Tracking | MLflow |
| Data Validation | Pandera |
| ML & Competition | Scikit-learn, XGBoost, Pandas, NumPy, Skops |
| API | FastAPI, Uvicorn, Pydantic |
| Testing | pytest, httpx |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions, CML (Iterative) |
| Deployment | Hugging Face Spaces, Linux |

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict drug
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "sex": "M", "blood_pressure": "HIGH", "cholesterol": "NORMAL", "na_to_k_ratio": 15.4}'
```

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run deterministic training & competition directly
python train.py

# Or reproduce via DVC pipeline
dvc repro

# Start local MLflow experiment dashboard
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

# Run comprehensive test suite (39 tests)
pytest -v

# Start FastAPI server
uvicorn api.main:app --port 8000
```

## Model Competition Results

| Rank | Model | 5-Fold CV Macro F1 | Test Accuracy | Test Macro F1 | Status |
|---|---|---|---|---|---|
| 🏆 1 | **XGBoost** | 0.9633 ± 0.0330 | **100.0%** | **1.0000** | **Champion** |
| 🥈 2 | ExtraTrees | 0.9489 ± 0.0821 | 98.33% | 0.9780 | Challenger #1 |
| 🥉 3 | RandomForest | 0.8452 ± 0.0665 | 98.33% | 0.9717 | Challenger #2 |
| 4 | LogisticRegression | 0.8407 ± 0.0598 | 96.67% | 0.9425 | Challenger #3 |

> **Metric Reconciliation:** Initial documentation historically cited `97.0% Accuracy / 0.94 F1` while `metrics.txt` showed `0.95 Accuracy / 0.86 Macro F1` due to unseeded shuffling. Under Sprint 1's deterministic pipeline (`seed=125`), Random Forest achieves `98.33% / 0.9717`, while model competition promotes **XGBoost** as Champion with `100.0% / 1.0000` (100% reproducible across independent runs).

![Confusion Matrix](./Results/model_results.png)

## Documentation

- 🏛️ [Architecture](docs/architecture.md) — End-to-end system and component design
- 🔄 [ML Pipeline](docs/ml-pipeline.md) — Validation, preprocessing, competition, and evaluation details
- 📦 [DVC Versioning](docs/dvc.md) — Dataset tracking, hash verification, and pipeline execution
- 🧪 [MLflow Tracking](docs/mlflow.md) — Experiment tracking, parameter logging, UI guide, and lineage
- 🛡️ [Security](docs/security.md) — Serialization security and vulnerability scanning
- 🚢 [Deployment](docs/deployment.md) — Container and zero-downtime deployment guide
- 🧪 [Testing](docs/testing.md) — Test taxonomy and automated test execution
