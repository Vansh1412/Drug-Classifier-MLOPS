"""
tests/test_sprint3.py — Sprint 3 Comprehensive Test Suite.

Covers:
- S3-T2 & S3-T3: SHAP explainability service, multiclass attribution, feature mapping, determinism.
- S3-T4: POST /explain endpoint schema, response structure, and validity.
- S3-T5: Enhanced POST /predict with confidence, probabilities, and backward compatibility.
- S3-T6: GET /version lineage, git commit, and dataset hash metadata.
- S3-T7: GET /metrics operational API observability and live request counters.
- S3-T8: API input boundary validation, error handling, security, and medical disclaimer enforcement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import skops.io as sio
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import api.main as api_module  # noqa: E402
from src.config import get_trusted_types, load_config  # noqa: E402
from src.explainability import (  # noqa: E402
    MEDICAL_EXPLANATION_DISCLAIMER,
    ModelExplainerService,
)

VALID_DRUGS = {"DrugY", "drugA", "drugB", "drugC", "drugX"}
MODEL_PATH = PROJECT_ROOT / "Model" / "drug_pipeline.skops"


@pytest.fixture(scope="module")
def loaded_pipeline():
    """Load the champion model pipeline with trusted types."""
    cfg = load_config()
    trusted = get_trusted_types(cfg)
    pipeline = sio.load(str(MODEL_PATH), trusted=trusted)
    return pipeline


@pytest.fixture(scope="module")
def explainer_service(loaded_pipeline):
    """Instantiate the explainability service on the champion pipeline."""
    return ModelExplainerService(loaded_pipeline, model_name="xgboost")


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with model loaded."""
    if api_module.pipeline is None:
        cfg = load_config()
        trusted = get_trusted_types(cfg)
        api_module.pipeline = sio.load(str(MODEL_PATH), trusted=trusted)
    return TestClient(api_module.app)


@pytest.fixture
def valid_patient_payload():
    """Return a standard valid patient feature payload."""
    return {
        "age": 30,
        "sex": "M",
        "blood_pressure": "HIGH",
        "cholesterol": "NORMAL",
        "na_to_k_ratio": 15.4,
    }


# =============================================================================
# 1. SHAP Explainability Service Tests (S3-T2, S3-T3)
# =============================================================================
class TestExplainabilityService:
    def test_service_initialization(self, explainer_service):
        """Assert that ModelExplainerService initializes with valid components."""
        assert explainer_service.pipeline is not None
        assert explainer_service.preprocessor is not None
        assert explainer_service.estimator is not None
        assert explainer_service.classes == [
            "DrugY",
            "drugA",
            "drugB",
            "drugC",
            "drugX",
        ]
        assert explainer_service.explainer is not None

    def test_service_rejects_invalid_pipeline(self):
        """Assert that initializing with an invalid pipeline raises ValueError."""
        with pytest.raises(ValueError):
            ModelExplainerService(object())

    def test_explain_output_schema(self, explainer_service):
        """Assert that explain() returns all required schema keys."""
        sample = [[30, "M", "HIGH", "NORMAL", 15.4]]
        res = explainer_service.explain(sample)

        expected_keys = {
            "prediction",
            "confidence",
            "probabilities",
            "explained_class",
            "explanation",
            "explanation_method",
            "model_name",
            "disclaimer",
        }
        assert expected_keys.issubset(res.keys())
        assert res["prediction"] in VALID_DRUGS
        assert res["explanation_method"] == "SHAP"

    def test_explain_multiclass_probabilities_validity(self, explainer_service):
        """Assert that class probabilities sum to ~1.0 and cover all 5 drug classes."""
        sample = [[45, "F", "LOW", "HIGH", 10.5]]
        res = explainer_service.explain(sample)

        probs = res["probabilities"]
        assert set(probs.keys()) == VALID_DRUGS
        assert 0.99 <= sum(probs.values()) <= 1.01
        assert res["confidence"] == max(probs.values())
        assert 0.0 <= res["confidence"] <= 1.0

    def test_explain_feature_attributions_integrity(self, explainer_service):
        """Assert that feature attributions map cleanly to domain features."""
        sample = [[50, "M", "HIGH", "NORMAL", 22.0]]
        res = explainer_service.explain(sample)

        explanation = res["explanation"]
        assert len(explanation) == 5

        features_in_explanation = {item["feature"] for item in explanation}
        expected_features = {"Age", "Sex", "BP", "Cholesterol", "Na_to_K"}
        assert features_in_explanation == expected_features

        for item in explanation:
            assert "feature" in item
            assert "value" in item
            assert "contribution" in item
            assert isinstance(item["contribution"], float)

    def test_explain_determinism(self, explainer_service):
        """Assert that identical inputs yield identical SHAP explanations."""
        sample = [[28, "F", "HIGH", "NORMAL", 18.2]]
        res1 = explainer_service.explain(sample)
        res2 = explainer_service.explain(sample)

        assert res1["prediction"] == res2["prediction"]
        assert res1["confidence"] == res2["confidence"]
        assert res1["explanation"] == res2["explanation"]

    def test_explain_alternative_target_class(self, explainer_service):
        """Assert that explainer supports targeting an explicit class."""
        sample = [[30, "M", "HIGH", "NORMAL", 15.4]]
        res = explainer_service.explain(sample, target_class="drugA")

        assert res["explained_class"] == "drugA"
        assert len(res["explanation"]) == 5

    def test_medical_disclaimer_content(self, explainer_service):
        """Assert that the medical safety disclaimer is included and accurate."""
        sample = [[30, "M", "HIGH", "NORMAL", 15.4]]
        res = explainer_service.explain(sample)

        assert "disclaimer" in res
        assert "MUST NOT be interpreted as clinical diagnosis" in res["disclaimer"]
        assert res["disclaimer"] == MEDICAL_EXPLANATION_DISCLAIMER


# =============================================================================
# 2. Production API Endpoints Tests (S3-T4, S3-T5, S3-T6, S3-T7)
# =============================================================================
class TestApiEndpoints:
    def test_predict_endpoint_enhancements(self, client, valid_patient_payload):
        """Assert /predict maintains backward compatibility and returns confidence."""
        response = client.post("/predict", json=valid_patient_payload)
        assert response.status_code == 200

        data = response.json()
        assert "predicted_drug" in data
        assert data["predicted_drug"] in VALID_DRUGS
        assert "message" in data

        # Enhanced Sprint 3 fields
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
        assert "probabilities" in data
        assert set(data["probabilities"].keys()) == VALID_DRUGS
        assert "model_version" in data

    def test_explain_endpoint_happy_path(self, client, valid_patient_payload):
        """Assert /explain returns HTTP 200 with structured SHAP explanation."""
        response = client.post("/explain", json=valid_patient_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["prediction"] in VALID_DRUGS
        assert 0.0 <= data["confidence"] <= 1.0
        assert set(data["probabilities"].keys()) == VALID_DRUGS
        assert data["explanation_method"] == "SHAP"
        assert len(data["explanation"]) == 5
        assert "disclaimer" in data

    def test_explain_endpoint_high_na_to_k_yields_drug_y(self, client):
        """Verify biomedical logic: Na_to_K > 15 yields DrugY with positive contribution."""
        payload = {
            "age": 40,
            "sex": "M",
            "blood_pressure": "HIGH",
            "cholesterol": "HIGH",
            "na_to_k_ratio": 25.0,
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["prediction"] == "DrugY"

        # Na_to_K should have a positive contribution towards DrugY
        na_item = next(
            item for item in data["explanation"] if item["feature"] == "Na_to_K"
        )
        assert na_item["contribution"] > 0.0

    def test_version_endpoint_metadata(self, client):
        """Assert /version returns full reproducibility lineage and metadata."""
        response = client.get("/version")
        assert response.status_code == 200

        data = response.json()
        assert data["application"] == "MedMLOps-X"
        assert data["api_version"] == "1.1.0"
        assert data["model_name"] == "xgboost"
        assert data["model_type"] == "XGBClassifier"
        assert "git_commit" in data
        assert "dataset_version" in data
        assert "pipeline_stages" in data
        assert isinstance(data["pipeline_stages"], list)

    def test_metrics_endpoint_data(self, client):
        """Assert /metrics returns operational observability metrics."""
        response = client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "MedMLOps-X API"
        assert "uptime_seconds" in data
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "requests_by_endpoint" in data
        assert "latency_ms" in data

    def test_metrics_counters_increment(self, client, valid_patient_payload):
        """Assert that requests to /predict and /explain increment live counters."""
        metrics_before = client.get("/metrics").json()
        prev_pred = metrics_before.get("prediction_requests", 0)
        prev_exp = metrics_before.get("explanation_requests", 0)

        # Fire requests
        client.post("/predict", json=valid_patient_payload)
        client.post("/explain", json=valid_patient_payload)

        metrics_after = client.get("/metrics").json()
        assert metrics_after["prediction_requests"] == prev_pred + 1
        assert metrics_after["explanation_requests"] == prev_exp + 1


# =============================================================================
# 3. API Validation & Security Boundary Tests (S3-T8)
# =============================================================================
class TestApiValidationAndSecurity:
    def test_invalid_categorical_sex_rejected(self, client, valid_patient_payload):
        """Assert invalid Sex values return HTTP 422."""
        payload = {**valid_patient_payload, "sex": "OTHER"}
        response = client.post("/explain", json=payload)
        assert response.status_code == 422

    def test_invalid_categorical_bp_rejected(self, client, valid_patient_payload):
        """Assert invalid Blood Pressure values return HTTP 422."""
        payload = {**valid_patient_payload, "blood_pressure": "CRITICAL"}
        response = client.post("/explain", json=payload)
        assert response.status_code == 422

    def test_invalid_categorical_cholesterol_rejected(
        self, client, valid_patient_payload
    ):
        """Assert invalid Cholesterol values return HTTP 422."""
        payload = {**valid_patient_payload, "cholesterol": "LOW"}
        response = client.post("/explain", json=payload)
        assert response.status_code == 422

    def test_out_of_range_age_rejected(self, client, valid_patient_payload):
        """Assert Age outside [15, 74] returns HTTP 422."""
        payload_under = {**valid_patient_payload, "age": 10}
        response_under = client.post("/explain", json=payload_under)
        assert response_under.status_code == 422

        payload_over = {**valid_patient_payload, "age": 85}
        response_over = client.post("/explain", json=payload_over)
        assert response_over.status_code == 422

    def test_out_of_range_na_to_k_rejected(self, client, valid_patient_payload):
        """Assert Na_to_K outside [6.2, 38.2] returns HTTP 422."""
        payload_low = {**valid_patient_payload, "na_to_k_ratio": 3.0}
        response_low = client.post("/explain", json=payload_low)
        assert response_low.status_code == 422

        payload_high = {**valid_patient_payload, "na_to_k_ratio": 50.0}
        response_high = client.post("/explain", json=payload_high)
        assert response_high.status_code == 422

    def test_missing_required_field_rejected(self, client, valid_patient_payload):
        """Assert missing fields return HTTP 422."""
        payload = valid_patient_payload.copy()
        del payload["na_to_k_ratio"]
        response = client.post("/explain", json=payload)
        assert response.status_code == 422

    def test_malformed_json_rejected(self, client):
        """Assert malformed request payloads return HTTP 422 without traceback."""
        response = client.post(
            "/explain",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        body = response.text
        assert "Traceback" not in body

    def test_no_internal_traceback_leak_on_error(self, client):
        """Assert that unexpected errors do not expose Python stack traces."""
        response = client.post("/explain", json={"invalid": "structure"})
        assert response.status_code == 422
        assert "Traceback (most recent call last)" not in response.text

    def test_openapi_schema_contains_endpoints_and_disclaimer(self, client):
        """Assert that OpenAPI specification documents all endpoints and disclaimer."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        paths = schema.get("paths", {})
        assert "/health" in paths
        assert "/predict" in paths
        assert "/explain" in paths
        assert "/version" in paths
        assert "/metrics" in paths
        assert "MEDICAL SAFETY DISCLAIMER" in schema.get("info", {}).get(
            "description", ""
        )
