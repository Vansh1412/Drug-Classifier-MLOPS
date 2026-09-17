# CI/CD Pipeline & Automated Quality Gates Guide

## 1. Overview

**MedMLOps-X** employs a hardened, automated Continuous Integration and Continuous Deployment (CI/CD) lifecycle built on GitHub Actions. The pipeline enforces rigorous code quality, pipeline reproducibility, end-to-end regression testing, hardened Docker containerization, and live container smoke validation before deployment.

---

## 2. CI/CD Architecture

```
                               ┌────────────────────────────────┐
                               │       Git Push / PR (main)     │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Continuous Integration (CI)     │
                              │  (.github/workflows/ci.yml)      │
                              └────────────────┬─────────────────┘
                                               │
      ┌──────────────────┬─────────────────────┼─────────────────────┬──────────────────┐
      ▼                  ▼                     ▼                     ▼                  ▼
[Format Check]      [Lint Check]       [DVC Integrity]      [Test Suite]         [Docker Build]
 black --check      ruff check            dvc status           pytest -v           docker build
      │                  │                     │                     │                  │
      └──────────────────┴─────────────────────┼─────────────────────┴──────────────────┘
                                               ▼
                                  ┌──────────────────────────┐
                                  │ Container Smoke Test     │
                                  │  • /health (200 OK)      │
                                  │  • /predict (Inference)  │
                                  │  • /explain (SHAP)       │
                                  │  • /version (Lineage)    │
                                  │  • /metrics (Telemetry)  │
                                  │  • Non-root verification │
                                  └────────────┬─────────────┘
                                               │
                                               ▼
                                  ┌──────────────────────────┐
                                  │ CML Evaluation Report    │
                                  │ Metrics & Confusion Plot │
                                  └────────────┬─────────────┘
                                               │ (Workflow Completed on main)
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Continuous Deployment (CD)      │
                              │  (.github/workflows/cd.yml)      │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │   Hugging Face Spaces Sync       │
                              │   (App, Model, Metrics)          │
                              └──────────────────────────────────┘
```

---

## 3. Continuous Integration (CI) Stages

| Stage | Command / Action | Purpose |
|---|---|---|
| **1. Checkout** | `actions/checkout@v4` | Check out repository code at triggered ref |
| **2. Python Setup** | `actions/setup-python@v5` | Initialize reproducible Python 3.13 runtime |
| **3. Dependencies** | `pip install -r requirements.txt` | Install locked production & test dependencies |
| **4. Formatting Check** | `black --check .` | Verify code follows deterministic black formatting (88 cols) |
| **5. Linting Check** | `ruff check .` | Perform fast static analysis, import sorting, and error checking |
| **6. DVC Pipeline Status** | `dvc status` | Verify dataset integrity and pipeline DAG synchronization |
| **7. Test Suite** | `pytest -v` | Execute all 81 automated unit, integration, and security tests |
| **8. Docker Build** | `docker build -t medmlops-x:local .` | Compile hardened multi-stage production container |
| **9. Container Smoke Test** | Curl & health polling loop | Validate runtime container endpoints and verify non-root user execution |
| **10. CML Report** | CML report generation | Post confusion matrix and evaluation metrics to job summary |

---

## 4. Container Smoke Testing Protocol

Inside the CI runner, the freshly built image is executed and tested against live HTTP traffic before the build is approved:

1. **Daemon Launch**: `docker run -d --name medmlops-container -p 8000:8000 medmlops-x:local`
2. **Health Poll**: Loop queries `http://localhost:8000/health` until HTTP 200 is confirmed (max 30s).
3. **Endpoint Validation**:
   - `GET /health` → Assert status is `healthy`.
   - `POST /predict` → Assert `predicted_drug` is present in response.
   - `POST /explain` → Assert SHAP `feature_attributions` and `medical_disclaimer` are present.
   - `GET /version` → Assert model name and git/dvc lineage metadata exist.
   - `GET /metrics` → Assert operational counters incremented.
4. **Security Assertion**:
   ```bash
   CONTAINER_USER=$(docker exec medmlops-container whoami)
   if [ "$CONTAINER_USER" = "root" ]; then
     echo "ERROR: Container is executing as root!"
     exit 1
   fi
   ```
5. **Safe Tear Down**: Container logs are dumped to CI stdout, followed by `docker stop` and `docker rm`.

---

## 5. Continuous Deployment (CD)

The deployment workflow (`.github/workflows/cd.yml`) triggers only upon successful completion of the CI workflow on the `main` branch:

- **Branch Guard**: Requires `head_branch == 'main'` and `conclusion == 'success'`.
- **Secret Isolation**: Utilizes GitHub secrets (`HF`, `HF_SPACE`). If not configured, it issues a graceful notice without failing.
- **Synchronized Artifacts**: Syncs `App/` (Gradio UI), `Model/` (Champion Skops pipeline), and `Results/` (Evaluation metrics) directly to Hugging Face Spaces.

---

## 6. Pipeline Security Controls

- **Least Privilege Permissions**: CI permissions are explicitly restricted:
  ```yaml
  permissions:
    contents: read
    pull-requests: write
    issues: write
  ```
- **No Secret Leakage**: Secrets are never echoed, logged, or exported.
- **Immutable Context**: Container build excludes `.env`, `mlruns.db`, and local credentials.
- **Fork Protection**: External pull requests cannot access repository secrets.
