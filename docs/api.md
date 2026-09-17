# REST API Documentation — MedMLOps-X

The MedMLOps-X REST API is built on FastAPI and exposes endpoints for inference, local explainability (SHAP), version lineage, and live operational observability.

Interactive documentation is available when running locally at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## 1. Endpoints Overview

| Method | Path | Tag | Description |
|---|---|---|---|
| `GET` | `/health` | Health | Service readiness and model loading status. |
| `POST` | `/predict` | Prediction | Predict recommended drug class with confidence & probabilities. |
| `POST` | `/explain` | Explainability | Local SHAP feature attributions for a patient prediction. |
| `GET` | `/version` | Metadata | Code, model, and data lineage metadata (Git SHA, DVC MD5). |
| `GET` | `/metrics` | Observability | Live API operational metrics, request counters, and latencies. |

---

## 2. Endpoint Details

### 2.1 `GET /health`
Verifies service liveness and confirms the trained champion pipeline is loaded in memory.

**Response `200 OK`:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### 2.2 `POST /predict`
Predicts the recommended drug class based on patient biometric features. Maintains 100% backward compatibility with Sprint 0/1/2 schemas while supplying model confidence and probability distributions.

**Request Body:**
```json
{
  "age": 30,
  "sex": "M",
  "blood_pressure": "HIGH",
  "cholesterol": "NORMAL",
  "na_to_k_ratio": 15.4
}
```

**Validation Constraints:**
- `age`: integer, `15 <= age <= 74`
- `sex`: string enum, `"M"` or `"F"`
- `blood_pressure`: string enum, `"HIGH"`, `"LOW"`, `"NORMAL"`
- `cholesterol`: string enum, `"HIGH"`, `"NORMAL"`
- `na_to_k_ratio`: float, `6.2 <= na_to_k_ratio <= 38.2`

**Response `200 OK`:**
```json
{
  "predicted_drug": "DrugY",
  "message": "Based on the provided patient features, the recommended drug is: DrugY",
  "confidence": 0.9944,
  "probabilities": {
    "DrugY": 0.9944,
    "drugA": 0.0026,
    "drugB": 0.001,
    "drugC": 0.0007,
    "drugX": 0.0013
  },
  "model_version": "0.2.0",
  "model_name": "xgboost"
}
```

---

### 2.3 `POST /explain`
Computes SHAP feature contributions explaining why the champion model produced a specific recommendation.

**Request Body:** Same schema as `/predict`.

**Response `200 OK`:**
```json
{
  "prediction": "DrugY",
  "confidence": 0.9944,
  "probabilities": {
    "DrugY": 0.9944,
    "drugA": 0.0026,
    "drugB": 0.001,
    "drugC": 0.0007,
    "drugX": 0.0013
  },
  "explained_class": "DrugY",
  "explanation": [
    {
      "feature": "Na_to_K",
      "value": 15.4,
      "contribution": 3.3437
    },
    {
      "feature": "Age",
      "value": 30,
      "contribution": 0.0
    },
    {
      "feature": "Sex",
      "value": "M",
      "contribution": 0.0
    },
    {
      "feature": "BP",
      "value": "HIGH",
      "contribution": 0.0
    },
    {
      "feature": "Cholesterol",
      "value": "NORMAL",
      "contribution": 0.0
    }
  ],
  "explanation_method": "SHAP",
  "model_name": "xgboost",
  "model_version": "0.2.0",
  "disclaimer": "This explanation represents statistical feature attribution for model transparency. It reflects machine-learning model decision weights and MUST NOT be interpreted as clinical diagnosis, biological etiology, or medical treatment advice."
}
```

---

### 2.4 `GET /version`
Exposes end-to-end reproducibility lineage linking the serving application to its source code and data version.

**Response `200 OK`:**
```json
{
  "application": "MedMLOps-X",
  "api_version": "1.1.0",
  "model_name": "xgboost",
  "model_type": "XGBClassifier",
  "model_version": "0.2.0",
  "training_timestamp": "2026-09-14T17:14:02.125871+00:00",
  "git_commit": "968058e6d4b30f1a06f7270c199df1a7b0e35319",
  "dataset_version": "7a10a9eb01e3d5664125ea1dd179c8c9",
  "mlflow_run_id": "baa060737ef34f2d889a78257312b061",
  "pipeline_stages": [
    "data_validation",
    "preprocessing",
    "champion_inference",
    "shap_explanation"
  ]
}
```

---

### 2.5 `GET /metrics`
Exposes live operational metrics collected by the in-memory `APIMetricsCollector`.

**Response `200 OK`:**
```json
{
  "service": "MedMLOps-X API",
  "uptime_seconds": 128.45,
  "start_timestamp": "2026-09-17T14:20:45.376575+00:00",
  "total_requests": 25,
  "successful_requests": 24,
  "client_errors_4xx": 1,
  "server_errors_5xx": 0,
  "prediction_requests": 10,
  "explanation_requests": 8,
  "requests_by_status": {
    "200": 24,
    "422": 1
  },
  "requests_by_endpoint": {
    "/health": 3,
    "/predict": 10,
    "/explain": 8,
    "/version": 2,
    "/metrics": 2
  },
  "latency_ms": {
    "/predict": {
      "avg_latency_ms": 8.42,
      "p95_latency_ms": 11.20,
      "sample_count": 10
    },
    "/explain": {
      "avg_latency_ms": 14.50,
      "p95_latency_ms": 22.10,
      "sample_count": 8
    }
  }
}
```

---

## 3. Operational Metrics vs Model Metrics

To maintain architectural clarity, MedMLOps-X strictly separates:
- **Model Evaluation Metrics:** (Accuracy, Macro F1, Recall, Confusion Matrix) — evaluated offline during training, tracked in MLflow (`mlruns.db`) and persisted in `Results/metrics.json`.
- **API Operational Metrics:** (Request volumes, error rates, p95 latencies, service uptime) — tracked live in memory during server execution and exposed at `/metrics`.

---

## 4. Error Responses

| Status Code | Meaning | Example Cause |
|---|---|---|
| `400 Bad Request` | Malformed request body | Non-JSON string sent with JSON header |
| `422 Unprocessable Entity` | Schema validation violation | `age: 5` (outside `[15, 74]`), `sex: "INVALID"` |
| `500 Internal Server Error` | Unexpected backend error | Pipeline compute failure (sanitized error returned) |
| `503 Service Unavailable` | Model artifact unready | Model file missing on disk |

All error responses return structured JSON and never leak internal stack traces or environment variables:
```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "age"],
      "msg": "Input should be less than or equal to 74",
      "input": 90
    }
  ]
}
```
