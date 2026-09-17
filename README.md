# Drug Classifier MLOps (MedMLOps-X)

[![CI](https://github.com/Vansh1412/Drug-Classifier-MLOPS/actions/workflows/ci.yml/badge.svg)](https://github.com/Vansh1412/Drug-Classifier-MLOPS/actions/workflows/ci.yml)
[![CD](https://github.com/Vansh1412/Drug-Classifier-MLOPS/actions/workflows/cd.yml/badge.svg)](https://github.com/Vansh1412/Drug-Classifier-MLOPS/actions/workflows/cd.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vansh1412/Drug-Classifier-MLOPS/blob/main/notebook.ipynb)

An end-to-end MLOps platform demonstrating automated data validation, deterministic multi-model competition, champion selection, SHAP explainability, production FastAPI serving, live observability, containerization, and continuous deployment.

## Project Description

This project builds a **Drug Classification** platform predicting the appropriate medication class based on patient biomarkers (Age, Sex, Blood Pressure, Cholesterol, Na-to-K ratio). The training pipeline is fully reproducible and selects a champion model from multiple competing algorithms based on automated quality gates.

**Key Features:**
- 🔄 **Automated Retraining Engine** — Drift-certified, idempotent retraining pipeline isolating candidates in `Model/challengers/`
- ⚖️ **Champion/Challenger Quality Gate** — Dual-gated validation enforcing absolute quality floors and non-regression against the active champion
- 🔁 **Atomic Promotion & Rollback** — Atomic model deployment with instant archive and rollback at `/models/rollback`
- 🔍 **SHAP Explainability** — Local feature attribution (TreeExplainer) quantifying how each biometric contributed to the prediction
- 📊 **Operational Observability** — Live request counters, status code breakdown, latency percentiles (avg, p95), and uptime at `/metrics`
- 🌐 **Production FastAPI Server** — Endpoints for `/health`, `/predict`, `/explain`, `/version`, `/metrics`, `/monitoring`, `/retraining`, and `/models`
- 📦 **Data & Pipeline Versioning** — DVC tracks canonical dataset hashes and pipeline DAG (`dvc repro`)
- 🧪 **Experiment Tracking** — MLflow tracks parameters, metrics, cross-validation runs, artifacts, and champion lineage
- 🛡️ **Data Validation** — Pandera schema enforcing value ranges, biological limits, and data types
- 🎲 **Deterministic ML** — Centralized seed configuration ensuring 100% reproducible splits and metrics
- 🏆 **Model Competition** — Automated evaluation of Logistic Regression, Random Forest, ExtraTrees, and XGBoost
- 🥇 **Champion Selection** — Quality gate thresholding (Macro F1, Accuracy, Recall) with automated champion promotion
- 🔒 **Safe Serialization** — Skops model persistence with strict trusted type whitelisting
- 🐳 **Hardened Containerization** — Multi-stage Docker build, non-root user (appuser UID 10001), healthcheck endpoint polling
- 📈 **Continuous Model Monitoring** — Statistical drift detection (Kolmogorov-Smirnov, Wasserstein, PSI), data quality checks, and prediction tracking
- ✅ **Automated Testing** — 140 pytest tests covering baseline, Sprint 1, Sprint 2, Sprint 3, Sprint 4, Sprint 5, and Sprint 6
- ⚙️ **CI/CD** — GitHub Actions pipeline: format → lint → DVC integrity → 140 tests → Docker build & smoke test → deploy

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Monitoring & Drift | SciPy, NumPy, Pandera (KS-test, Wasserstein, PSI, Chi-Square) |
| Retraining & Registry | Skops, SciPy, FastAPI, Scikit-learn, XGBoost |
| Explainability | SHAP (TreeExplainer) |
| API & Observability | FastAPI, Uvicorn, Pydantic, In-Memory Metrics Collector |
| Data & Pipeline Versioning | DVC (Data Version Control) |
| Experiment Tracking | MLflow |
| Data Validation | Pandera |
| ML & Competition | Scikit-learn, XGBoost, Pandas, NumPy, Skops |
| Testing | pytest, httpx |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions, CML (Iterative) |
| Deployment | Hugging Face Spaces, Linux |

## API Usage

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Predict recommended drug (with confidence & probabilities)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "sex": "M", "blood_pressure": "HIGH", "cholesterol": "NORMAL", "na_to_k_ratio": 15.4}'

# 3. Explain prediction with SHAP local feature attributions
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "sex": "M", "blood_pressure": "HIGH", "cholesterol": "NORMAL", "na_to_k_ratio": 15.4}'

# 4. View system, model, and data lineage version metadata
curl http://localhost:8000/version

# 5. View live API operational metrics (requests, latencies, status codes)
curl http://localhost:8000/metrics

# 6. Evaluate batch of patient records for population data and prediction drift
curl -X POST http://localhost:8000/monitoring/drift \
  -H "Content-Type: application/json" \
  -d '{"records": [{"age": 30, "sex": "M", "blood_pressure": "HIGH", "cholesterol": "NORMAL", "na_to_k_ratio": 15.4}]}'

# 7. Retrieve the latest persisted monitoring and drift report summary
curl http://localhost:8000/monitoring/latest

# 8. Inspect active model champion metadata and promotion lineage
curl http://localhost:8000/models/champion

# 9. Query current retraining engine status and trigger history
curl http://localhost:8000/retraining/status

# 10. Securely trigger drift-certified retraining (requires Admin API Key)
curl -X POST http://localhost:8000/retraining/trigger \
  -H "Content-Type: application/json" \
  -H "X-API-Key: medmlops-admin-key-sprint6" \
  -d '{"report_id": "latest", "auto_promote": true}'

# 11. Securely roll back to previous champion model (requires Admin API Key)
curl -X POST http://localhost:8000/models/rollback \
  -H "X-API-Key: medmlops-admin-key-sprint6"
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

# Run comprehensive test suite (140 tests)
pytest -v

# Start FastAPI server directly
uvicorn api.main:app --port 8000
```

## Docker Containerization

```bash
# 1. Build hardened production multi-stage image
docker build -t medmlops-x:local .

# 2. Run container as non-root user with port 8000 mapped
docker run -d --name medmlops-api -p 8000:8000 medmlops-x:local

# 3. Check container health status
docker inspect --format='{{json .State.Health}}' medmlops-api

# 4. Confirm non-root execution (appuser: UID 10001)
docker exec medmlops-api whoami

# 5. Stop and clean up container
docker stop medmlops-api && docker rm medmlops-api

# Or launch via Docker Compose:
docker compose up -d
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
- 🔄 [Automated Retraining & Rollback](docs/retraining.md) — Retraining orchestrator, drift certification, quality gates, and registry promotion
- 🐳 [Docker Architecture](docs/docker.md) — Multi-stage build, non-root user, and container hardening
- ⚙️ [CI/CD Pipeline](docs/ci-cd.md) — GitHub Actions automated quality gates, smoke tests, and permissions
- 🔍 [Explainability](docs/explainability.md) — SHAP TreeExplainer methodology, mapping, and interpretation
- 🌐 [REST API Reference](docs/api.md) — Endpoints, schemas, validation rules, and error codes
- 🔄 [ML Pipeline](docs/ml-pipeline.md) — Validation, preprocessing, competition, and evaluation details
- 📦 [DVC Versioning](docs/dvc.md) — Dataset tracking, hash verification, and pipeline execution
- 🧪 [MLflow Tracking](docs/mlflow.md) — Experiment tracking, parameter logging, UI guide, and lineage
- 🛡️ [Security](docs/security.md) — Serialization security, input validation, and medical safety boundary
- 🚢 [Deployment](docs/deployment.md) — Container and zero-downtime deployment guide
- 📈 [Continuous Monitoring](docs/monitoring.md) — Statistical drift detection, PSI/KS methodology, thresholds, and report schemas
- 🧪 [Testing](docs/testing.md) — Test taxonomy and automated test execution

## Medical Safety Boundary

⚠️ **Important:** MedMLOps-X is an educational machine-learning demonstration and is NOT a certified clinical diagnostic or prescribing tool. Predictions and SHAP feature attributions are generated for instructional MLOps demonstration only and must never be utilized as medical advice.

