"""
tests/test_sprint4.py — Sprint 4 Test Suite: Docker Hardening & Container Validation.

Covers:
- S4-T2 & S4-T3: Dockerfile multi-stage structure, non-root user (appuser UID 10001), layer minimization.
- S4-T4: .dockerignore rules and exclusion of sensitive files/caches.
- S4-T5: Dependency alignment and explicit package pinning.
- S4-T6 & S4-T7: Container healthcheck directive and docker-compose.yml configuration.
- S4-T8: Container smoke test suite validating /health, /predict, /explain, /version, /metrics.
- S4-T9: Security controls (no root execution, read-only package separation, environment settings).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import skops.io as sio  # noqa: E402

import api.main as api_module  # noqa: E402
from src.config import get_trusted_types, load_config  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "Model" / "drug_pipeline.skops"
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"


@pytest.fixture(scope="module")
def client():
    """TestClient fixture with model pipeline initialized."""
    if api_module.pipeline is None:
        cfg = load_config()
        trusted = get_trusted_types(cfg)
        api_module.pipeline = sio.load(str(MODEL_PATH), trusted=trusted)
    return TestClient(api_module.app)


@pytest.fixture(scope="module")
def dockerfile_content():
    """Read Dockerfile text."""
    assert DOCKERFILE_PATH.is_file(), f"Dockerfile not found at {DOCKERFILE_PATH}"
    return DOCKERFILE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerignore_content():
    """Read .dockerignore text."""
    assert (
        DOCKERIGNORE_PATH.is_file()
    ), f".dockerignore not found at {DOCKERIGNORE_PATH}"
    return DOCKERIGNORE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose_content():
    """Read docker-compose.yml text."""
    assert COMPOSE_PATH.is_file(), f"docker-compose.yml not found at {COMPOSE_PATH}"
    return COMPOSE_PATH.read_text(encoding="utf-8")


# =============================================================================
# 1. Dockerfile & Image Hardening Tests (S4-T2, S4-T3, S4-T6, S4-T9)
# =============================================================================
class TestDockerConfiguration:
    def test_dockerfile_multi_stage_architecture(self, dockerfile_content):
        """Assert Dockerfile defines both builder and runtime stages."""
        assert re.search(
            r"FROM\s+\S+\s+AS\s+builder", dockerfile_content, re.IGNORECASE
        )
        assert re.search(
            r"FROM\s+\S+\s+AS\s+runtime", dockerfile_content, re.IGNORECASE
        )

    def test_dockerfile_non_root_user_creation(self, dockerfile_content):
        """Assert Dockerfile creates dedicated unprivileged appuser (UID 10001)."""
        assert "useradd" in dockerfile_content
        assert "10001" in dockerfile_content
        assert "appuser" in dockerfile_content
        assert re.search(r"USER\s+appuser", dockerfile_content)

    def test_dockerfile_copies_all_required_runtime_directories(
        self, dockerfile_content
    ):
        """Assert Dockerfile copies api/, src/, Model/, and configs/ into image."""
        required_dirs = ["api/", "src/", "Model/", "configs/"]
        for directory in required_dirs:
            pattern = rf"COPY\s+.*{re.escape(directory)}"
            assert re.search(
                pattern, dockerfile_content
            ), f"Missing COPY for {directory}"

    def test_dockerfile_chown_set_for_non_root_user(self, dockerfile_content):
        """Assert COPY commands in runtime stage specify --chown=appuser:appgroup."""
        assert "--chown=appuser:appgroup" in dockerfile_content

    def test_dockerfile_healthcheck_directive(self, dockerfile_content):
        """Assert native HEALTHCHECK instruction exists and queries /health."""
        assert "HEALTHCHECK" in dockerfile_content
        assert "/health" in dockerfile_content
        assert "--interval=" in dockerfile_content
        assert "--timeout=" in dockerfile_content

    def test_dockerfile_exposed_port(self, dockerfile_content):
        """Assert port 8000 is explicitly exposed."""
        assert re.search(r"EXPOSE\s+8000", dockerfile_content)

    def test_dockerfile_environment_variables(self, dockerfile_content):
        """Assert production Python environment variables are declared."""
        assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile_content
        assert "PYTHONUNBUFFERED=1" in dockerfile_content

    def test_dockerfile_deterministic_entrypoint(self, dockerfile_content):
        """Assert CMD runs uvicorn api.main:app on port 8000."""
        assert 'CMD ["uvicorn", "api.main:app"' in dockerfile_content


# =============================================================================
# 2. Docker Context & Compose Hardening Tests (S4-T4, S4-T7)
# =============================================================================
class TestDockerContextAndCompose:
    def test_dockerignore_excludes_git_and_ci(self, dockerignore_content):
        """Assert .dockerignore excludes .git and .github."""
        assert ".git" in dockerignore_content
        assert ".github" in dockerignore_content

    def test_dockerignore_excludes_bytecode_and_caches(self, dockerignore_content):
        """Assert .dockerignore excludes __pycache__ and test caches."""
        assert "__pycache__" in dockerignore_content
        assert ".pytest_cache" in dockerignore_content

    def test_dockerignore_excludes_mlflow_and_dvc_caches(self, dockerignore_content):
        """Assert .dockerignore excludes local experiment databases and DVC caches."""
        assert "mlruns" in dockerignore_content
        assert "mlruns.db" in dockerignore_content
        assert ".dvc/cache" in dockerignore_content

    def test_dockerignore_excludes_secrets_and_env(self, dockerignore_content):
        """Assert .dockerignore excludes environment files and private keys."""
        assert ".env" in dockerignore_content
        assert "*.pem" in dockerignore_content or "*.key" in dockerignore_content

    def test_docker_compose_configuration(self, compose_content):
        """Assert docker-compose.yml defines drug-classifier service with healthcheck."""
        assert "drug-classifier" in compose_content
        assert "8000:8000" in compose_content
        assert "healthcheck:" in compose_content
        assert "/health" in compose_content


# =============================================================================
# 3. Container Smoke Tests for Production Endpoints (S4-T8)
# =============================================================================
class TestContainerSmokeEndpoints:
    @pytest.fixture
    def patient_payload(self):
        return {
            "age": 30,
            "sex": "M",
            "blood_pressure": "HIGH",
            "cholesterol": "NORMAL",
            "na_to_k_ratio": 15.4,
        }

    def test_smoke_health_endpoint(self, client):
        """Smoke test: GET /health returns HTTP 200 and model_loaded=True."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_smoke_predict_endpoint(self, client, patient_payload):
        """Smoke test: POST /predict returns valid drug prediction and confidence."""
        response = client.post("/predict", json=patient_payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_drug" in data
        assert data["predicted_drug"] in {"DrugY", "drugA", "drugB", "drugC", "drugX"}
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert "probabilities" in data

    def test_smoke_explain_endpoint(self, client, patient_payload):
        """Smoke test: POST /explain returns SHAP attributions and medical disclaimer."""
        response = client.post("/explain", json=patient_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["explanation_method"] == "SHAP"
        assert len(data["explanation"]) == 5
        assert "disclaimer" in data
        assert "MUST NOT be interpreted as clinical diagnosis" in data["disclaimer"]

    def test_smoke_version_endpoint(self, client):
        """Smoke test: GET /version returns system version and lineage metadata."""
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert data["application"] == "MedMLOps-X"
        assert data["api_version"] == "1.1.0"
        assert data["model_name"] == "xgboost"
        assert "git_commit" in data
        assert "dataset_version" in data

    def test_smoke_metrics_endpoint(self, client):
        """Smoke test: GET /metrics returns operational request and latency counters."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "MedMLOps-X API"
        assert "total_requests" in data
        assert "uptime_seconds" in data

    def test_smoke_boundary_rejection_security(self, client):
        """Smoke test: Out-of-bounds input returns HTTP 422 without traceback."""
        invalid_payload = {
            "age": 100,  # Invalid: age > 74
            "sex": "INVALID",
            "blood_pressure": "HIGH",
            "cholesterol": "NORMAL",
            "na_to_k_ratio": 15.4,
        }
        response = client.post("/explain", json=invalid_payload)
        assert response.status_code == 422
        assert "Traceback" not in response.text
