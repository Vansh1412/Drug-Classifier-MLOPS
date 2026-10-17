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
        ┌────────────┴────────────┬────────────────────────┬────────────────────────┐
        ↓                         ↓                        ↓                        ↓
  Core Inference           Explainability            Observability              Monitoring
 (/predict, /health)      (SHAP /explain)         (/metrics, /version)      (/monitoring/drift)
```

## 2. Component Directory

| Component | Path | Responsibility |
|---|---|---|
| **Data Layer (Structured)** | `Data/drug.csv` | Canonical patient tabular dataset (~200 records) |
| **Data Layer (Reviews NLP)** | `Data/reviews/drugsComTest_raw.csv` | Raw patient review dataset (~53,766 records) |
| **Processed Reviews** | `Data/reviews/processed/` | Preprocessed review parquet dataset with length features |
| **Review Validation** | `src/nlp_validation.py` | Pandera schema and quality validation for review dataset |
| **Review Preprocessing** | `src/nlp_preprocessing.py` | HTML decoding, text cleaning, and leakage-safe splitting |
| **Data Versioning** | `Data/drug.csv.dvc`, `Data/reviews/drugsComTest_raw.csv.dvc`, `dvc.yaml` | DVC dataset tracking and stage reproducibility (`dvc repro`) |
| **Experiment Tracking** | `src/tracking.py`, `mlruns.db` | MLflow run tracking, parameter/metric logging, and model logging |
| **Data Validation** | `src/data_validation.py` | Pandera schema enforcing clinical value ranges and datatypes |
| **Model Registry** | `src/model_registry.py` | Factory for competition candidates & ModelRegistry manager (atomic promote, rollback) |
| **Model Registry Manifest** | `Model/registry_manifest.json` | Registry manifest tracking active champion, archived champion, and promotion history |
| **Retraining Engine** | `src/retraining.py` | Automated retraining trigger, drift certifier, state machine, and dual quality gate |
| **Preprocessing** | `src/preprocessing.py` | Shared ColumnTransformer (OrdinalEncoder + StandardScaler) |
| **Evaluation Engine** | `src/evaluate.py` | 5-Fold Stratified CV and multi-class metrics computation |
| **Champion Selection** | `src/champion_selection.py` | Quality gate thresholding and primary metric leaderboard |
| **Explainability Service** | `src/explainability.py` | SHAP TreeExplainer service and domain feature attribution mapping |
| **Model Monitor** | `src/monitoring.py` | Statistical drift engine (KS test, Wasserstein, PSI, Chi-Square, Pandera) |
| **Monitoring Reports** | `monitoring/reports/` | Structured JSON drift reports with lineage metadata |
| **API Metrics Collector** | `src/api_metrics.py` | In-memory thread-safe operational metrics and latency tracker |
| **Pipeline Runner** | `train.py` | Deterministic end-to-end training and competition execution |
| **Config Layer** | `configs/model_config.yaml` | Centralized hyperparams, seeds, thresholds, retraining/promotion, and review configs |
| **Model Artifact** | `Model/drug_pipeline.skops` | Serialized active champion scikit-learn Pipeline using `skops` |
| **Model Archive** | `Model/archive/` | Historical archived champion snapshots for zero-downtime rollback |
| **Model Metadata** | `Model/model_metadata.json` | Active model metadata, hyperparameters, and evaluation records |
| **Structured Metrics** | `Results/metrics.json` | Machine-readable metrics for all candidates |
| **API Layer** | `api/main.py` | FastAPI production REST server (/health, /predict, /explain, /version, /metrics, /monitoring, /retraining, /models) |
| **UI Layer** | `App/drug_app.py` | Gradio interface for Hugging Face Spaces |
| **NLP Competition Engine** | `src/nlp_competition.py` | TF-IDF feature pipeline, StratifiedGroupKFold CV, quality gate, and competition runner |
| **NLP Tracking** | `src/nlp_tracking.py` | Dedicated MLflow experiment tracking (`MedMLOps-X-NLP-Drug-Review`) and lineage logging |
| **NLP Artifacts** | `Model/nlp/` | Isolated NLP champion and candidate model storage area |
| **NLP Results** | `Results/nlp_competition_report.json`, `Results/nlp_metrics.json` | Comprehensive NLP model competition and metrics reports |
| **Patient Experience Engine** | `src/patient_experience.py` | Descriptive sentiment, drug/condition aggregations, aspect extraction, lexical similarity |
| **Patient Experience Reports** | `Results/patient_experience_report.json`, `Results/patient_experience_summary.json` | Machine-readable full and compact patient experience intelligence reports |
| **NLP Monitoring Engine** | `src/nlp_monitoring.py` | Statistical drift (length, vocabulary, TF-IDF, sentiment, rating, aspect, category, prediction) and certification |
| **NLP Monitoring Reports** | `Results/nlp_drift_report.json`, `monitoring/nlp_reports/` | Comprehensive NLP drift and data quality reports |
| **NLP Retraining Engine** | `src/nlp_retraining.py` | Automated retraining trigger, drift certification, dual quality gate, and competition runner |
| **NLP Model Registry** | `src/nlp_model_registry.py` | Dedicated NLP model registry, bootstrap promotion, atomic rollback, and smoke testing |
| **NLP Retraining History** | `monitoring/nlp_retraining_history.json` | Auditable trigger history, processed idempotency hashes, and candidate metrics |
| **Test Suite** | `tests/` | 288 automated pytest tests covering baseline, Sprint 1 through Sprint 11 (RC1) |
| **Container** | `Dockerfile`, `docker-compose.yml` | Hardened multi-stage build, non-root user (appuser), active healthcheck |
| **CI Automation** | `.github/workflows/ci.yml` | Continuous integration: black, ruff, dvc, pytest, Docker build & smoke tests |
| **CD Automation** | `.github/workflows/cd.yml` | Branch-guarded continuous deployment to HF Spaces |
| **Host Deploy** | `deploy.sh` | Shell script for zero-downtime Linux container rollout |
| **Tool Config** | `pyproject.toml` | Unified configuration for pytest, black, ruff, coverage |
| **Container Docs** | `docs/docker.md` | Multi-stage architecture, security controls, and CLI guide |
| **CI/CD Docs** | `docs/ci-cd.md` | Automated quality gates, smoke testing protocol, and permissions |
| **Monitoring Docs** | `docs/monitoring.md` | Structured ML statistical drift methodology, PSI/KS thresholds, and report schemas |
| **Retraining Docs** | `docs/retraining.md` | Drift certification, dual quality gate, atomic promotion, and rollback guide |
| **NLP Foundation Docs** | `docs/nlp-data-foundation.md` | Review data foundation, schema validation, and leakage-safe splitting |
| **NLP Competition Docs** | `docs/nlp-model-competition.md` | Classical NLP model competition, TF-IDF, CV Macro F1, and quality gate |
| **Patient Experience Docs** | `docs/patient-experience-intelligence.md` | Patient experience analytics, aspect extraction, and TF-IDF similarity |
| **NLP Monitoring Docs** | `docs/nlp-monitoring.md` | NLP statistical drift methodology, baseline reference, aspect tracking, and API guide |
| **NLP Retraining Docs** | `docs/nlp-retraining.md` | NLP automated retraining architecture, bootstrap promotion, quality gates, and rollback guide |

## 3. Medical Safety Boundary

⚠️ **Important:** MedMLOps-X is an educational machine-learning demonstration and is NOT a certified clinical diagnostic, prescribing, or treatment recommendation tool. Predictions, patient review sentiment classifications, aspect frequency extractions, and SHAP feature attributions are generated for instructional MLOps demonstration only and must never be utilized as medical advice, clinical trial efficacy evidence, or clinical decision support.
