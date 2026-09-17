# Production Docker Architecture & Container Hardening Guide

## 1. Overview

In Sprint 4, **MedMLOps-X** hardened its containerization architecture to provide an enterprise-grade, secure, deterministic, and self-monitoring runtime container for the FastAPI production serving layer.

Key hardening features:
- **Multi-Stage Build**: Isolates build-time dependencies (compilers, build wheels) in a builder stage to produce an ultra-lean runtime container.
- **Least-Privilege Non-Root Execution**: Runs under a dedicated unprivileged user (`appuser`, UID 10001, GID 10001) instead of root.
- **Hardened Build Context**: Strict `.dockerignore` prevents leakage of git history, virtual environments, development caches, MLflow databases, DVC caches, and environment secrets.
- **Active Container Healthcheck**: Uses native Docker `HEALTHCHECK` polling the FastAPI `/health` endpoint to verify genuine model loading and inference readiness.
- **Deterministic Entrypoint**: Explicit `uvicorn` startup configuration binding to port 8000 with production environment parameters.

---

## 2. Multi-Stage Dockerfile Architecture

```
                    ┌───────────────────────────────────┐
                    │     python:3.13-slim (Builder)    │
                    ├───────────────────────────────────┤
                    │ • Install gcc, g++, build tools   │
                    │ • Compile/install pip dependencies│
                    │   into user directory (~/.local)  │
                    └─────────────────┬─────────────────┘
                                      │ Copy ~/.local
                                      ▼
                    ┌───────────────────────────────────┐
                    │     python:3.13-slim (Runtime)    │
                    ├───────────────────────────────────┤
                    │ • Create non-root user (appuser)  │
                    │ • Copy required application files:│
                    │   api/, src/, Model/, configs/    │
                    │ • Set ownership to appuser        │
                    │ • Configure HEALTHCHECK           │
                    │ • USER appuser                    │
                    │ • ENTRYPOINT [uvicorn]            │
                    └───────────────────────────────────┘
```

### Stage 1: Builder
- Base Image: `python:3.13-slim`
- Installs minimal system build dependencies (`gcc`, `g++`) needed for C extensions.
- Upgrades `pip` and executes `pip install --no-cache-dir --user -r /tmp/requirements.txt`.
- Keeps development build tools strictly contained within the build phase.

### Stage 2: Runtime
- Base Image: `python:3.13-slim`
- Installs runtime utilities: `curl` (for health checking).
- Creates non-root system group `appgroup` (GID 10001) and system user `appuser` (UID 10001) without password or interactive shell.
- Copies pre-built Python packages from `/root/.local` to `/home/appuser/.local`.
- Sets working directory to `/app`.
- Copies strictly required application directories with `--chown=appuser:appgroup`:
  - `api/`: FastAPI server application and schemas
  - `src/`: Core domain logic (`explainability.py`, `api_metrics.py`, `config.py`)
  - `Model/`: Champion model artifact (`drug_pipeline.skops`) and metadata
  - `configs/`: Pipeline and model configuration (`model_config.yaml`)
- Explicitly drops superuser privileges with `USER appuser`.

---

## 3. Container Security Controls

| Security Control | Implementation | Verification Command |
|---|---|---|
| **Non-Root Execution** | `USER appuser` (UID: 10001) | `docker exec <container> whoami` → `appuser` |
| **No Build Tools in Runtime** | Multi-stage build copies only compiled wheels | `which gcc` inside runtime returns empty |
| **Minimal Attack Surface** | Minimal Debian-slim base, no superfluous tools | Runtime image contains only required serving assets |
| **Secrets & Cache Isolation** | `.dockerignore` rejects `.git`, `.env`, `mlruns.db`, `.dvc/cache` | Inspect build context / layer history |
| **Read-Only / Principle of Least Privilege** | Application files owned by `appuser:appgroup` | Container has no access to host filesystem |
| **Safe Model Deserialization** | Skops trusted-type loading enforced | Reject unknown pickled payloads |

---

## 4. Docker Usage Commands

### Build Image
```bash
docker build -t medmlops-x:local .
```

### Run Container
```bash
docker run -d \
  --name medmlops-api \
  -p 8000:8000 \
  medmlops-x:local
```

### Verify Container Status and Non-Root User
```bash
# Check container status
docker ps -f name=medmlops-api

# Confirm non-root execution
docker exec medmlops-api whoami
# Output: appuser

docker exec medmlops-api id
# Output: uid=10001(appuser) gid=10001(appgroup) groups=10001(appgroup)
```

### Inspect Container Health
```bash
# Check health status
docker inspect --format='{{json .State.Health}}' medmlops-api
```

### Test API Endpoints via Container
```bash
# 1. Healthcheck
curl http://localhost:8000/health

# 2. Predict endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "sex": "M", "blood_pressure": "HIGH", "cholesterol": "NORMAL", "na_to_k_ratio": 15.4}'

# 3. Explain endpoint (SHAP)
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "sex": "M", "blood_pressure": "HIGH", "cholesterol": "NORMAL", "na_to_k_ratio": 15.4}'

# 4. Version & Lineage endpoint
curl http://localhost:8000/version

# 5. Operational Metrics endpoint
curl http://localhost:8000/metrics
```

### Container Logs, Stop & Cleanup
```bash
# View application logs
docker logs medmlops-api

# Stop container
docker stop medmlops-api

# Remove container
docker rm medmlops-api
```

---

## 5. Docker Compose

For local development and integration testing, MedMLOps-X provides a production-hardened `docker-compose.yml`:

```bash
# Launch containerized service in background
docker compose up -d

# Verify service health and port mappings
docker compose ps

# View real-time service logs
docker compose logs -f api

# Stop and remove containers
docker compose down
```
