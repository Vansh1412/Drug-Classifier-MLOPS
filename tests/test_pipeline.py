"""
Unit tests for the Drug Classification ML pipeline.
Tests model file existence, loading, prediction output, and API endpoints.
"""

import os
import sys
import warnings

import pytest
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Add project root to path so imports resolve correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_PATH = "./Model/drug_pipeline.skops"
VALID_DRUGS = {"DrugY", "drugA", "drugB", "drugC", "drugX"}

TRUSTED_TYPES = [
    "sklearn.pipeline.Pipeline",
    "sklearn.preprocessing.OneHotEncoder",
    "sklearn.preprocessing.StandardScaler",
    "sklearn.compose.ColumnTransformer",
    "sklearn.preprocessing.OrdinalEncoder",
    "sklearn.impute.SimpleImputer",
    "sklearn.tree.DecisionTreeClassifier",
    "sklearn.ensemble.RandomForestClassifier",
    "numpy.dtype",
]


# ─────────────────────────────────────────────
# Test 1: Model file exists
# ─────────────────────────────────────────────
def test_model_file_exists():
    """Assert that the trained model file is present on disk."""
    assert os.path.exists(MODEL_PATH), (
        f"Model file not found at '{MODEL_PATH}'. Run 'make train' to generate it."
    )


# ─────────────────────────────────────────────
# Test 2: Model loads without error
# ─────────────────────────────────────────────
def test_model_loads():
    """Assert that the skops model file loads cleanly into a sklearn pipeline."""
    import skops.io as sio
    from sklearn.pipeline import Pipeline

    pipeline = sio.load(MODEL_PATH, trusted=TRUSTED_TYPES)
    assert pipeline is not None, "Pipeline loaded as None."
    assert isinstance(pipeline, Pipeline), (
        f"Expected sklearn Pipeline, got {type(pipeline)}"
    )


# ─────────────────────────────────────────────
# Test 3: Prediction returns a single valid drug label
# ─────────────────────────────────────────────
def test_prediction_output():
    """Assert that the pipeline predicts a known, valid drug label for a known input."""
    import skops.io as sio

    pipeline = sio.load(MODEL_PATH, trusted=TRUSTED_TYPES)

    sample = [[30, "M", "HIGH", "NORMAL", 15.4]]
    prediction = pipeline.predict(sample)

    assert len(prediction) == 1, f"Expected 1 prediction, got {len(prediction)}"
    assert prediction[0] in VALID_DRUGS, (
        f"Predicted drug '{prediction[0]}' is not in valid drug set {VALID_DRUGS}"
    )


# ─────────────────────────────────────────────
# Test 4: Multiple predictions return correct shape
# ─────────────────────────────────────────────
def test_prediction_shape():
    """Assert that batch predictions return the correct number of results."""
    import skops.io as sio

    pipeline = sio.load(MODEL_PATH, trusted=TRUSTED_TYPES)

    samples = [
        [30, "M", "HIGH", "NORMAL", 15.4],
        [35, "F", "LOW", "NORMAL", 8.0],
        [50, "M", "HIGH", "HIGH", 34.0],
    ]
    predictions = pipeline.predict(samples)

    assert len(predictions) == len(samples), (
        f"Expected {len(samples)} predictions, got {len(predictions)}"
    )
    for pred in predictions:
        assert pred in VALID_DRUGS, (
            f"Predicted drug '{pred}' is not in valid drug set {VALID_DRUGS}"
        )


# ─────────────────────────────────────────────
# Test 5: FastAPI /health endpoint returns 200
# ─────────────────────────────────────────────
def test_api_health_endpoint():
    """Assert that the FastAPI /health endpoint returns HTTP 200 with status 'ok'."""
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200, (
        f"Health endpoint returned {response.status_code}, expected 200"
    )
    data = response.json()
    assert data["status"] == "ok", f"Expected status 'ok', got '{data['status']}'"


# ─────────────────────────────────────────────
# Test 6: FastAPI /predict endpoint returns valid drug
# ─────────────────────────────────────────────
def test_api_predict_endpoint():
    """Assert that the FastAPI /predict endpoint returns a valid drug prediction."""
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    payload = {
        "age": 30,
        "sex": "M",
        "blood_pressure": "HIGH",
        "cholesterol": "NORMAL",
        "na_to_k_ratio": 15.4,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200, (
        f"Predict endpoint returned {response.status_code}: {response.text}"
    )
    data = response.json()
    assert "predicted_drug" in data, "Response missing 'predicted_drug' field"
    assert data["predicted_drug"] in VALID_DRUGS, (
        f"Predicted drug '{data['predicted_drug']}' not in valid drug set {VALID_DRUGS}"
    )
