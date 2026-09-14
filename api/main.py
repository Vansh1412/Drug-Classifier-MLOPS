import os
import warnings
from typing import Literal

import skops.io as sio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --- App Setup ---
app = FastAPI(
    title="Drug Classification API",
    description=(
        "REST API to predict recommended drug types based on patient biometric features "
        "using a trained scikit-learn RandomForest pipeline.\n\n"
        "⚠️ **MEDICAL SAFETY DISCLAIMER**: This system is an educational machine-learning demonstration "
        "and is NOT a clinical diagnostic or prescribing tool. Predictions must never be used as medical advice."
    ),
    version="1.0.0",
)

# --- Model Loading ---
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

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Model",
    "drug_pipeline.skops",
)

try:
    pipeline = sio.load(MODEL_PATH, trusted=TRUSTED_TYPES)
except FileNotFoundError:
    pipeline = None


# --- Pydantic Schemas ---
class PatientFeatures(BaseModel):
    age: int = Field(..., ge=15, le=74, description="Patient age (15–74)")
    sex: Literal["M", "F"] = Field(..., description="Patient sex: M or F")
    blood_pressure: Literal["HIGH", "LOW", "NORMAL"] = Field(
        ..., alias="blood_pressure", description="Blood pressure level"
    )
    cholesterol: Literal["HIGH", "NORMAL"] = Field(..., description="Cholesterol level")
    na_to_k_ratio: float = Field(
        ..., ge=6.2, le=38.2, description="Sodium to potassium ratio in blood"
    )

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    predicted_drug: str
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint to verify the service and model are ready."""
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_drug(patient: PatientFeatures):
    """
    Predict the recommended drug type for a patient.

    Accepts patient features and returns the predicted drug label.
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

    prediction = pipeline.predict(features)[0]
    return PredictionResponse(
        predicted_drug=prediction,
        message=f"Based on the provided patient features, the recommended drug is: {prediction}",
    )


# --- Entry Point ---
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
