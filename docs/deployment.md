# Deployment Guide

MedMLOps-X provides three complementary deployment pathways:

---

## 1. Local Development Server

Run the FastAPI Uvicorn server locally with hot-reloading:

```bash
make run-api
# or directly:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- Health endpoint: `http://localhost:8000/health`
- Interactive OpenAPI Swagger docs: `http://localhost:8000/docs`
- Predict endpoint: `http://localhost:8000/predict`

---

## 2. Docker Containerization

The production container uses a **multi-stage Dockerfile**:
- **Stage 1 (`builder`):** Installs compiler tools (`gcc`), upgrades pip, and compiles Python wheels from `api/requirements.txt`.
- **Stage 2 (`runtime`):** Uses a clean `python:3.10-slim` base, copying only installed packages and application code.
- **Health Check:** Evaluates `GET /health` every 30s.

### Commands:

```bash
# Build Docker image
docker build -t drug-classifier .

# Run container
docker run -d -p 8000:8000 --name drug-classifier-api drug-classifier

# Or using Docker Compose:
docker compose up -d
```

---

## 3. Linux Host Deployment (`deploy.sh`)

For self-hosted Linux instances, `deploy.sh` executes a 5-step zero-downtime deployment:

```bash
bash deploy.sh [IMAGE_NAME] [CONTAINER_NAME] [PORT]
```

1. Pulls latest commit from Git (`git pull origin main`)
2. Builds the Docker image
3. Gracefully stops and cleans up the existing container
4. Launches the new container with automatic restart policies
5. Executes automated health check polling against `http://localhost:PORT/health`

---

## 4. Hugging Face Spaces Deployment (CD)

Automated continuous deployment via `.github/workflows/cd.yml`:
- Triggered on successful CI workflow completion restricted to `main` branch.
- Syncs `App/`, `Model/`, and `Results/` to Hugging Face Spaces using the `huggingface_hub` API.
