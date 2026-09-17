import json
import logging
import os
import time
import warnings
from typing import Any, Literal

import pandas as pd
import skops.io as sio
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.exceptions import InconsistentVersionWarning

from src.api_metrics import metrics_collector
from src.explainability import (
    MEDICAL_EXPLANATION_DISCLAIMER,
    ModelExplainerService,
)
from src.monitoring import get_monitor

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
logger = logging.getLogger("api.main")

# --- App Setup ---
APP_TITLE = "MedMLOps-X Drug Classification API"
APP_DESCRIPTION = (
    "Production REST API providing explainable drug recommendations based on patient biometric features.\n\n"
    "### Capabilities\n"
    "- **Prediction**: High-performance classification using the champion model.\n"
    "- **Explainability**: Local SHAP feature attribution quantifying biometric influences.\n"
    "- **Observability**: Live API request tracking, latency metrics, and version lineage.\n\n"
    "⚠️ **CRITICAL MEDICAL SAFETY DISCLAIMER**: This system is strictly an educational machine-learning "
    "engineering demonstration and is NOT a medical device, clinical decision support system, or prescribing tool. "
    "Predictions and SHAP feature attributions represent statistical model weights and must NEVER be used as medical advice."
)
API_VERSION = "1.1.0"

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=API_VERSION,
)

# --- Model Loading & Setup ---
try:
    from src.config import get_trusted_types, load_config

    _cfg = load_config()
    TRUSTED_TYPES = get_trusted_types(_cfg)
except Exception:
    _cfg = {}
    TRUSTED_TYPES = [
        "sklearn.pipeline.Pipeline",
        "sklearn.preprocessing.OneHotEncoder",
        "sklearn.preprocessing.StandardScaler",
        "sklearn.compose.ColumnTransformer",
        "sklearn.preprocessing.OrdinalEncoder",
        "sklearn.impute.SimpleImputer",
        "sklearn.tree.DecisionTreeClassifier",
        "sklearn.tree._classes.DecisionTreeClassifier",
        "sklearn.ensemble.RandomForestClassifier",
        "sklearn.ensemble._forest.RandomForestClassifier",
        "sklearn.ensemble.ExtraTreesClassifier",
        "sklearn.ensemble._forest.ExtraTreesClassifier",
        "sklearn.tree.ExtraTreeClassifier",
        "sklearn.tree._classes.ExtraTreeClassifier",
        "sklearn.linear_model.LogisticRegression",
        "sklearn.linear_model._logistic.LogisticRegression",
        "src.model_registry.XGBStringClassifier",
        "xgboost.sklearn.XGBClassifier",
        "xgboost.core.Booster",
        "sklearn.preprocessing.LabelEncoder",
        "sklearn.preprocessing._label.LabelEncoder",
        "numpy.ndarray",
        "numpy.dtype",
    ]

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_BASE_DIR, "Model", "drug_pipeline.skops")
METADATA_PATH = os.path.join(_BASE_DIR, "Model", "model_metadata.json")

pipeline = None
try:
    if os.path.exists(MODEL_PATH):
        pipeline = sio.load(MODEL_PATH, trusted=TRUSTED_TYPES)
except Exception as err:
    logger.warning(f"Initial model loading failed: {err}")
    pipeline = None

# Global explainer service cache
_explainer_service: ModelExplainerService | None = None
_last_pipeline_instance = None
_cached_mlflow_run_id: str | None = None


def get_explainer_service() -> ModelExplainerService | None:
    """Retrieve or lazily initialize the ModelExplainerService for the active pipeline."""
    global _explainer_service, _last_pipeline_instance
    if pipeline is None:
        return None
    if _explainer_service is None or _last_pipeline_instance is not pipeline:
        try:
            model_name = "xgboost"
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                    model_name = meta.get("model_name", "xgboost")
            _explainer_service = ModelExplainerService(pipeline, model_name=model_name)
            _last_pipeline_instance = pipeline
        except Exception as e:
            logger.error(f"Failed to initialize ModelExplainerService: {e}")
            _explainer_service = None
    return _explainer_service


# --- Metrics & Logging Middleware ---
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    try:
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        metrics_collector.record_request(
            request.url.path, response.status_code, duration_ms
        )
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        metrics_collector.record_request(request.url.path, 500, duration_ms)
        logger.exception(
            f"Unhandled exception during request {request.url.path}: {exc}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal server error occurred. Please contact the administrator."
            },
        )


# --- Exception Handlers (Prevent Internal Leakage) ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    metrics_collector.record_request(request.url.path, 422, 0.0)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


# --- Pydantic Schemas ---
class PatientFeatures(BaseModel):
    age: int = Field(
        ..., ge=15, le=74, description="Patient age in years (valid range: 15–74)"
    )
    sex: Literal["M", "F"] = Field(
        ..., description="Patient biological sex ('M' or 'F')"
    )
    blood_pressure: Literal["HIGH", "LOW", "NORMAL"] = Field(
        ...,
        alias="blood_pressure",
        description="Blood pressure level ('HIGH', 'LOW', 'NORMAL')",
    )
    cholesterol: Literal["HIGH", "NORMAL"] = Field(
        ..., description="Cholesterol level ('HIGH' or 'NORMAL')"
    )
    na_to_k_ratio: float = Field(
        ...,
        ge=6.2,
        le=38.2,
        description="Sodium-to-potassium ratio in blood (valid range: 6.2–38.2)",
    )

    model_config = {"populate_by_name": True}


class DriftEvaluationRequest(BaseModel):
    records: list[PatientFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Batch of patient records to evaluate for population drift against baseline (1 to 1000 items)",
    )


class PredictionResponse(BaseModel):
    predicted_drug: str = Field(
        ..., description="Recommended drug label predicted by the model"
    )
    message: str = Field(..., description="Human-readable summary of prediction")
    confidence: float | None = Field(
        default=None, description="Model prediction probability for the predicted class"
    )
    probabilities: dict[str, float] | None = Field(
        default=None, description="Probability distribution across all drug classes"
    )
    model_version: str | None = Field(
        default=None, description="Version of the model serving predictions"
    )
    model_name: str | None = Field(
        default=None, description="Identifier of the champion model architecture"
    )


class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Domain feature name (e.g. 'Na_to_K', 'Age')")
    value: Any = Field(..., description="Input value supplied for this feature")
    contribution: float = Field(
        ..., description="SHAP attribution value towards the predicted class"
    )


class ExplanationResponse(BaseModel):
    prediction: str = Field(..., description="Predicted drug class")
    confidence: float = Field(..., description="Model confidence score (0.0 to 1.0)")
    probabilities: dict[str, float] = Field(
        ..., description="Class probability distribution across all 5 drug classes"
    )
    explained_class: str = Field(
        ..., description="Drug class being explained by the SHAP values"
    )
    explanation: list[FeatureContribution] = Field(
        ..., description="Ranked list of feature contributions to the prediction"
    )
    explanation_method: str = Field(
        default="SHAP",
        description="Explainability algorithm employed (SHAP TreeExplainer)",
    )
    model_name: str = Field(..., description="Champion model identifier")
    model_version: str = Field(..., description="Semantic version of the serving model")
    disclaimer: str = Field(
        default=MEDICAL_EXPLANATION_DISCLAIMER,
        description="Mandatory medical safety boundary notice",
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status ('ok')")
    model_loaded: bool = Field(
        ..., description="Whether the ML pipeline artifact is loaded in memory"
    )


class VersionResponse(BaseModel):
    application: str = Field(..., description="Application name")
    api_version: str = Field(..., description="FastAPI service semantic version")
    model_name: str = Field(..., description="Active champion model name")
    model_type: str = Field(..., description="Underlying model class")
    model_version: str = Field(..., description="Model project release version")
    training_timestamp: str | None = Field(
        default=None, description="UTC ISO timestamp of model training"
    )
    git_commit: str = Field(..., description="Git commit SHA representing code lineage")
    dataset_version: str = Field(
        ..., description="DVC dataset md5 hash representing data lineage"
    )
    mlflow_run_id: str | None = Field(
        default=None, description="MLflow run identifier if available"
    )
    pipeline_stages: list[str] = Field(
        default_factory=lambda: [
            "data_validation",
            "preprocessing",
            "champion_inference",
            "shap_explanation",
        ],
        description="Active pipeline stages in the prediction and explanation lifecycle",
    )


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint to verify service liveness and model availability."""
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_drug(patient: PatientFeatures):
    """
    Predict the recommended drug type based on patient biometric features.

    Maintains 100% backward compatibility with Sprint 0/1/2 contracts while exposing
    model confidence and class probability distributions.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Ensure '{MODEL_PATH}' exists. Run 'make train' first.",
        )

    features = [
        [
            patient.age,
            patient.sex,
            patient.blood_pressure,
            patient.cholesterol,
            patient.na_to_k_ratio,
        ]
    ]

    try:
        prediction = str(pipeline.predict(features)[0])
    except Exception as exc:
        logger.error(f"Inference execution failed: {exc}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")

    # Calculate probabilities and confidence
    confidence = None
    probabilities = None
    classes = ["DrugY", "drugA", "drugB", "drugC", "drugX"]

    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        model_step = pipeline.named_steps["model"]
        if hasattr(model_step, "classes_") and model_step.classes_ is not None:
            classes = list(model_step.classes_)

    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(features)[0]
            confidence = round(float(max(probs)), 4)
            probabilities = {
                cls_name: round(float(p), 4) for cls_name, p in zip(classes, probs)
            }
        except Exception:
            pass

    # Model metadata resolution
    model_version = "0.2.0"
    model_name = "xgboost"
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
                model_name = meta.get("model_name", model_name)
        except Exception:
            pass

    return PredictionResponse(
        predicted_drug=prediction,
        message=f"Based on the provided patient features, the recommended drug is: {prediction}",
        confidence=confidence,
        probabilities=probabilities,
        model_version=model_version,
        model_name=model_name,
    )


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
def explain_prediction(patient: PatientFeatures):
    """
    Generate SHAP-based local explainability for a patient prediction.

    Evaluates the champion model and computes feature attributions quantifying how
    each patient biometric contributed towards the model's recommended drug class.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Ensure '{MODEL_PATH}' exists. Run 'make train' first.",
        )

    service = get_explainer_service()
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability service is currently unavailable.",
        )

    features = [
        [
            patient.age,
            patient.sex,
            patient.blood_pressure,
            patient.cholesterol,
            patient.na_to_k_ratio,
        ]
    ]

    try:
        explanation_dict = service.explain(features)
    except Exception as exc:
        logger.error(f"SHAP explanation calculation failed: {exc}")
        raise HTTPException(
            status_code=500, detail="Explainability computation failed."
        )

    model_version = "0.2.0"
    return ExplanationResponse(
        prediction=explanation_dict["prediction"],
        confidence=explanation_dict["confidence"],
        probabilities=explanation_dict["probabilities"],
        explained_class=explanation_dict["explained_class"],
        explanation=[
            FeatureContribution(**item) for item in explanation_dict["explanation"]
        ],
        explanation_method=explanation_dict["explanation_method"],
        model_name=explanation_dict["model_name"],
        model_version=model_version,
        disclaimer=explanation_dict["disclaimer"],
    )


@app.get("/version", response_model=VersionResponse, tags=["Metadata"])
def get_version():
    """
    Retrieve system, model, and data lineage metadata.

    Exposes semantic versioning, model architecture, Git commit hash,
    and DVC dataset MD5 hash for full reproducibility auditability.
    """
    model_name = "xgboost"
    model_class = "XGBClassifier"
    training_timestamp = None
    git_commit = "unknown"
    dataset_version = "unknown"
    mlflow_run_id = None

    # Lineage retrieval
    try:
        from src.tracking import get_dvc_dataset_hash, get_git_commit_sha

        git_commit = get_git_commit_sha()
        dataset_version = get_dvc_dataset_hash()
    except Exception:
        pass

    # Model metadata retrieval
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
                model_name = meta.get("model_name", model_name)
                model_class = meta.get("model_class", model_class)
                training_timestamp = meta.get("training_timestamp")
        except Exception:
            pass

    # Optional MLflow run ID retrieval if experiment database is reachable
    global _cached_mlflow_run_id
    if _cached_mlflow_run_id is not None:
        mlflow_run_id = _cached_mlflow_run_id
    else:
        try:
            import mlflow

            from src.config import load_config

            cfg = load_config()
            tracking_uri = cfg.get("mlflow", {}).get(
                "tracking_uri", "sqlite:///mlruns.db"
            )
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name("MedMLOps-X-Drug-Classifier")
            if exp is not None:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string="tags.is_champion = 'true'",
                    max_results=1,
                )
                if runs:
                    _cached_mlflow_run_id = runs[0].info.run_id
                    mlflow_run_id = _cached_mlflow_run_id
        except Exception:
            pass

    return VersionResponse(
        application="MedMLOps-X",
        api_version=API_VERSION,
        model_name=model_name,
        model_type=model_class,
        model_version="0.2.0",
        training_timestamp=training_timestamp,
        git_commit=git_commit,
        dataset_version=dataset_version,
        mlflow_run_id=mlflow_run_id,
    )


@app.get("/metrics", tags=["Observability"])
def get_operational_metrics():
    """
    Expose real-time API operational and service observability metrics.

    Includes total requests, endpoint hits, status code distribution,
    latency statistics (average, p95), and uptime.
    """
    return metrics_collector.get_metrics()


@app.post("/monitoring/drift", tags=["Monitoring"])
def evaluate_drift(request_data: DriftEvaluationRequest):
    """
    Evaluate a batch of patient records for population data drift, prediction drift,
    and schema/data quality anomalies against reference baseline.
    """
    try:
        data_dicts = [
            {
                "Age": p.age,
                "Sex": p.sex,
                "BP": p.blood_pressure,
                "Cholesterol": p.cholesterol,
                "Na_to_K": p.na_to_k_ratio,
            }
            for p in request_data.records
        ]
        current_df = pd.DataFrame(data_dicts)
        monitor = get_monitor()
        report = monitor.compute_drift(
            current_df, include_predictions=True, save_report=True
        )
        return report
    except Exception as exc:
        logger.error(f"Drift monitoring evaluation failed: {exc}")
        raise HTTPException(
            status_code=500, detail="Drift monitoring computation failed."
        )


@app.get("/monitoring/latest", tags=["Monitoring"])
def get_latest_monitoring_report():
    """
    Retrieve the most recently computed and persisted monitoring report.
    """
    monitor = get_monitor()
    report = monitor.get_latest_report()
    if report is None:
        report = monitor.compute_drift(
            monitor.reference_df, include_predictions=True, save_report=True
        )
    return report


# --- Entry Point ---
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
