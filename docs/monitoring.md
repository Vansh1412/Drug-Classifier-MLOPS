# Continuous Model Monitoring & Data Drift Detection Guide

## 1. Overview

In Sprint 5, **MedMLOps-X** added an enterprise-grade **Continuous Model Monitoring and Data Drift Detection** layer. Moving beyond static model serving, the platform actively evaluates the statistical alignment between **Reference Baseline Data** (training distribution) and **Current Serving Data** (incoming patient populations).

The monitoring engine identifies:
1. **Input Feature Drift**: Distributional changes in patient biometrics (`Age`, `Sex`, `BP`, `Cholesterol`, `Na_to_K`).
2. **Prediction Output Drift**: Frequency shifts in recommended drug classes and confidence score degradation.
3. **Schema Drift**: Column absences, unexpected extra features, or datatype mismatches.
4. **Data Quality Anomalies**: Spikes in missing values, clinical range violations, and unseen categorical levels.

---

## 2. Monitoring Architecture

```
                    Reference Data (Data/drug.csv)
                                  │
                                  ▼
                     ┌───────────────────────────┐
                     │  Baseline Feature Profile │ (Means, Quantiles, Frequencies)
                     └─────────────┬─────────────┘
                                   │
Current Serving Data (Batch of Patients or Aggregated Payload)
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │     Monitoring Engine     │
                     │    (src/monitoring.py)    │
                     ├───────────────────────────┤
                     │ 1. Schema Validation      │
                     │ 2. Data Quality Checks    │
                     │ 3. Numerical Drift (KS)   │
                     │ 4. Categorical Drift (PSI)│
                     │ 5. Prediction Drift (PSI) │
                     │ 6. Severity Classifier    │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │   Structured JSON Report  │
                     │ monitoring/reports/*.json │
                     │ • Model/Git/DVC Lineage   │
                     │ • Zero PII / Patient Data │
                     │ • Medical Safety Boundary │
                     └─────────────┬─────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
          FastAPI Serving                     MLflow Tracking
(/monitoring/drift, /monitoring/latest)      (Lineage Artifacts)
```

---

## 3. Drift Detection Methodology

### 3.1 Numerical Feature Drift (`Age`, `Na_to_K`, `confidence`)
- **Two-Sample Kolmogorov-Smirnov (KS) Test**:
  - Non-parametric test comparing the empirical cumulative distribution functions (ECDFs) of the reference and current data:
    $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$$
  - $p \ge 0.05$: `PASS` (Low Severity)
  - $0.01 \le p < 0.05$: `WARNING` (Medium Severity)
  - $p < 0.01$: `DRIFT` (High Severity)
- **1D Wasserstein Distance** (Earth Mover's Distance):
  - Quantifies the minimum work required to transform one distribution into the other:
    $$W_1(u, v) = \int_{-\infty}^{\infty} |U(x) - V(x)| dx$$
  - Normalized by reference standard deviation to measure physical shift magnitude.

### 3.2 Categorical Feature Drift (`Sex`, `BP`, `Cholesterol`)
- **Population Stability Index (PSI)**:
  - Industry-standard metric measuring divergence between categorical or binned probability distributions:
    $$\text{PSI} = \sum_{i=1}^k (A_i - E_i) \times \ln\left(\frac{A_i}{E_i}\right)$$
  - where $A_i$ is Actual (Current) proportion and $E_i$ is Expected (Reference) proportion.
  - $\text{PSI} < 0.10$: `PASS` (Negligible change)
  - $0.10 \le \text{PSI} < 0.20$: `WARNING` (Moderate population shift)
  - $\text{PSI} \ge 0.20$: `DRIFT` (Significant population shift)

### 3.3 Prediction Output Drift
- Evaluates model recommendation distribution (`predicted_drug`) against reference target distribution using PSI.
- Evaluates model confidence score distribution shifts using the KS test.

### 3.4 Data Quality & Schema Monitoring
- **Missingness**: Tracks null percentage per column against a 0.0% baseline tolerance.
- **Range Violations**: Evaluates inputs violating Pandera boundaries (`Age` $\in [15, 74]$, `Na_to_K` $\in [6.0, 39.0]$).
- **Unseen Categories**: Flags categorical levels never observed during baseline training (e.g. `BP="VERY_HIGH"`).

---

## 4. Concept Drift Boundary

> [!IMPORTANT]
> **Prediction Drift $\neq$ Concept Drift**:
> In healthcare machine learning, **feature drift** indicates demographic or biometric shifts in incoming patient cohorts, and **prediction drift** indicates shifts in model decision frequencies.
> Without confirmed **ground-truth clinical outcomes** (e.g. lab confirmations, physician notes) for serving patients, model accuracy degradation **cannot** be measured.
> MedMLOps-X strictly reports statistical distribution variance and never makes unsupported claims regarding clinical efficacy.

---

## 5. Threshold Strategy & Severity Scale

Configured centrally in `configs/model_config.yaml`:

```yaml
monitoring:
  enabled: true
  reference_dataset: "Data/drug.csv"
  reports_dir: "monitoring/reports"
  max_batch_size: 1000
  thresholds:
    numerical:
      method: "ks"
      p_value_warning: 0.05
      p_value_drift: 0.01
      wasserstein_warning: 0.15
    categorical:
      method: "psi"
      psi_warning: 0.10
      psi_drift: 0.20
      chi2_p_value_drift: 0.05
    prediction:
      class_psi_warning: 0.10
      class_psi_drift: 0.20
      confidence_ks_warning: 0.05
      confidence_ks_drift: 0.01
    quality:
      max_null_rate: 0.0
      max_out_of_range_rate: 0.0
```

| Status | Severity | Criteria | Action |
|---|---|---|---|
| **PASS** | `LOW` | All $p \ge 0.05$, all $\text{PSI} < 0.10$, quality clean | Normal serving operation |
| **WARNING** | `MEDIUM` | Any $0.01 \le p < 0.05$, $0.10 \le \text{PSI} < 0.20$, or nulls detected | Flag for developer inspection |
| **DRIFT** | `HIGH` | Any $p < 0.01$, $\text{PSI} \ge 0.20$, or unseen categories | Trigger candidate retraining pipeline |
| **ERROR** | `HIGH` | Missing columns or datatype incompatibilities | Reject batch; investigate schema breach |

---

## 6. Structured Report Schema

Monitoring reports are stored as JSON in `monitoring/reports/`:

```json
{
  "report_id": "drift_20260917_153000_a1b2c3d4",
  "timestamp": "2026-09-17T15:30:00+00:00",
  "model_metadata": {
    "model_name": "xgboost",
    "model_version": "0.2.0",
    "git_commit": "aaffd67",
    "dvc_dataset_hash": "7a10a9e"
  },
  "summary": {
    "overall_status": "PASS",
    "overall_severity": "LOW",
    "feature_drift_detected": false,
    "prediction_drift_detected": false,
    "schema_status": "PASS",
    "data_quality_status": "PASS",
    "drifted_features": [],
    "drifted_features_count": 0,
    "total_features_monitored": 5,
    "reference_samples": 200,
    "current_samples": 50
  },
  "schema_validation": { "status": "PASS", ... },
  "data_quality": { "status": "PASS", ... },
  "features": {
    "Age": {
      "type": "numerical",
      "status": "PASS",
      "severity": "LOW",
      "ks_p_value": 0.4521,
      "wasserstein_distance": 0.0412,
      "psi": 0.0215
    },
    ...
  },
  "prediction_drift": {
    "status": "PASS",
    "class_psi": 0.0341,
    ...
  },
  "disclaimer": "MLOps statistical monitoring signals only. Not clinical or diagnostic evaluation."
}
```

---

## 7. REST API Endpoints

### 7.1 Evaluate Batch Drift (`POST /monitoring/drift`)
```bash
curl -X POST http://localhost:8000/monitoring/drift \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"age": 30, "sex": "M", "blood_pressure": "HIGH", "cholesterol": "NORMAL", "na_to_k_ratio": 15.4},
      {"age": 45, "sex": "F", "blood_pressure": "LOW", "cholesterol": "HIGH", "na_to_k_ratio": 18.2},
      {"age": 60, "sex": "M", "blood_pressure": "NORMAL", "cholesterol": "NORMAL", "na_to_k_ratio": 24.1}
    ]
  }'
```

### 7.2 Retrieve Latest Report (`GET /monitoring/latest`)
```bash
curl http://localhost:8000/monitoring/latest
```

---

## 8. Privacy & Security Safeguards

1. **Zero Raw Patient Data Logging**: Reports record strictly aggregate statistics (quantiles, means, category percentages, test p-values). Individual patient biometric records are never persisted in monitoring reports.
2. **Path Traversal Protection**: File writes strictly validate destination paths against `monitoring/reports/`, rejecting any relative `../` directory escapes.
3. **Bounded Input Batching**: The API restricts batch sizes to a maximum of 1,000 records to prevent memory exhaustion and DoS attacks.
4. **Sanitized Error Responses**: Internal computation exceptions are logged server-side and masked as sanitized HTTP 500 responses without exposing stack traces.
