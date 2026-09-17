# Automated Retraining, Model Quality Gate & Promotion/Rollback Guide

## 1. Overview

In Sprint 6, **MedMLOps-X** completed the closed-loop MLOps lifecycle by connecting the Sprint 5 Continuous Monitoring drift signals to an **Automated Retraining Trigger Engine**, **Champion/Challenger Quality Gate**, and a **Reversible Model Registry with Atomic Promotion and Rollback**.

The system operationalizes model updates while adhering to strict safety principles:
- **Champion Protection**: The active production model is never modified, deleted, or overwritten during candidate training.
- **Isolated Challenger Staging**: All retraining candidates and evaluation artifacts are built into an isolated challenger workspace (`Model/challengers/`).
- **Drift Certification**: Raw drift reports are certified before triggering retraining, rejecting schema mismatches, corrupted payloads, or small-sample noise.
- **Idempotency Guard**: Prevents duplicate retraining runs for the same drift event, dataset hash, and champion version.
- **Dual Quality Gate**: Requires absolute baseline performance AND comparative non-regression against the active champion.
- **Atomic Promotion**: Archives the existing champion before replacing production artifacts.
- **Automatic Rollback**: Immediately restores the previous champion if post-promotion smoke tests fail.

---

## 2. End-to-End Lifecycle Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. MONITORING & DRIFT DETECTION (Sprint 5)                  │
│    Reference Data vs Serving Data ──► Structured Report     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. DRIFT CERTIFICATION POLICY (src/retraining.py)           │
│    • overall_status == "DRIFT"                              │
│    • schema_status == "PASS" (Schema breaches rejected)     │
│    • data_quality_status != "ERROR"                         │
│    • current_samples >= min_samples (N >= 20)               │
│    • Core predictive features drifted                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. IDEMPOTENT TRIGGER ENGINE                                │
│    Hash = SHA256(report_id :: champion_ver :: dvc_hash)     │
│    • If already processed ──► Return SKIPPED                │
│    • Else ──► Transition to RETRAINING                      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. ISOLATED CANDIDATE TRAINING & COMPETITION                │
│    • Trains: LR, RF, ExtraTrees, XGBoost                    │
│    • Outputs ONLY to Model/challengers/                     │
│    • Production champion Model/drug_pipeline.skops UNTOUCHED│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. DUAL QUALITY GATE (Absolute + Comparative)               │
│    • Gate 1 (Absolute): Accuracy >= 0.80, Macro F1 >= 0.75  │
│    • Gate 2 (Comparative):                                  │
│        Challenger F1 >= Champion F1 - 0.02                  │
│        Challenger CV F1 >= Champion CV F1 - 0.03            │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
         [ Gate PASS ]                  [ Gate FAIL ]
               │                              │
               ▼                              ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│ 6. ATOMIC PROMOTION          │ │ REJECT CHALLENGER          │
│    • Archive old champion    │ │ • Champion 100% UNCHANGED  │
│      to Model/archive/       │ │ • Challenger logged as     │
│    • Atomic file replacement │ │   rejected                 │
│    • Update registry manifest│ └────────────────────────────┘
└──────────────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. POST-PROMOTION SMOKE TEST                                │
│    • Verify Skops trusted deserialization                   │
│    • Execute sample inference validation                    │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
          [ Verified ]                   [ Failed ]
               │                              │
               ▼                              ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│ PROMOTION ACCEPTED           │ │ AUTOMATIC ROLLBACK         │
│ • Active champion updated    │ │ • Revert to archived model │
│ • API serving cache reloaded │ │ • Active champion restored │
│ • Lineage logged to MLflow   │ │ • API serving cache reset  │
└──────────────────────────────┘ └────────────────────────────┘
```

---

## 3. Drift Certification Policy

The `DriftCertifier` evaluates monitoring reports before permitting automated retraining:

| Rule | Requirement | Rationale |
|---|---|---|
| **Overall Status** | `overall_status == "DRIFT"` | Only actionable drift triggers pipeline execution. |
| **Schema Integrity** | `schema_status == "PASS"` | A schema breach (e.g. missing column or bad datatype) requires a code/contract fix; retraining on invalid schemas corrupts models. |
| **Data Quality** | `data_quality_status != "ERROR"` | Corrupted, zero-row, or malformed payloads are rejected at the gate. |
| **Sample Size** | `current_samples >= 20` | Prevents stochastic sampling noise from small batch sizes triggering expensive retraining runs. |
| **Feature Relevance** | Drift $\cap$ `{Age, Sex, BP, Cholesterol, Na_to_K}` $\neq \emptyset$ | Guarantees drift affects core predictive attributes. |

---

## 4. Dual Quality Gate

Before a retrained challenger can replace the champion, it must pass two independent validation gates:

### Gate 1: Absolute Minimum Performance Gate
- **Accuracy**: $\ge 0.80$
- **Macro F1 Score**: $\ge 0.75$
- **Macro Recall**: $\ge 0.70$

### Gate 2: Comparative Non-Regression Gate
- **Macro F1 Non-Regression**:
  $$\text{F1}_{\text{challenger}} \ge \text{F1}_{\text{champion}} - 0.02$$
- **Cross-Validation Stability**:
  $$\text{CV F1}_{\text{challenger}} \ge \text{CV F1}_{\text{champion}} - 0.03$$
- **Integrity Validation**: Artifact must load successfully via `skops.io` with trusted types whitelist verification.

If any check fails, the challenger is marked `REJECTED`, and the production champion remains untouched.

---

## 5. Model Registry & Rollback Protocol

The `ModelRegistry` ([src/model_registry.py](file:///d:/Chrome%20Downloads/CICD-for-Machine-Learning-main/CICD-for-Machine-Learning-main/src/model_registry.py)) manages version lineage and file operations:

- **Manifest File** (`Model/registry_manifest.json`):
  Tracks `active_champion`, `previous_champion`, and audit history.
- **Archive Directory** (`Model/archive/`):
  Stores snapshot copies (`champion_v{ver}_{timestamp}.skops`, `model_metadata_v{ver}_{timestamp}.json`).
- **Atomic Promotion**:
  1. Current champion is copied to `Model/archive/`.
  2. Challenger is staged into temporary files (`.promotetmp`) and atomically replaced.
  3. Manifest is updated with version transitions.
  4. Post-promotion test runs; on failure, `rollback()` is triggered automatically.
- **Manual / API Rollback**:
  An administrator can trigger rollback at any time via `POST /models/rollback` or CLI, restoring the archived model and updating the manifest.

---

## 6. Management API Endpoints

All lifecycle operations are exposed via FastAPI and secured with API Key authentication (`X-API-Key`):

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| `POST` | `/retraining/trigger` | Yes (`X-API-Key`) | Certify drift report and trigger isolated retraining workflow. |
| `GET` | `/retraining/status` | No | Observability endpoint returning state machine state and history. |
| `GET` | `/models/champion` | No | Returns active champion details, lineage, and version manifest. |
| `POST` | `/models/rollback` | Yes (`X-API-Key`) | Restores previously archived champion model. |

---

## 7. Security Controls

1. **Endpoint Authorization**:
   - Lifecycle operations require the `X-API-Key` header.
   - Configurable via `MEDMLOPS_ADMIN_API_KEY` environment variable.
2. **Subprocess Injection Defense**:
   - Subprocesses pass argument lists directly; `shell=True` is strictly avoided.
3. **Path Traversal Guards**:
   - All filesystem operations validate `path.is_relative_to(PROJECT_ROOT / "Model")`.
4. **Trusted Deserialization**:
   - All model loads use `skops.io` with an explicit whitelist of trusted sklearn/xgboost classes.
5. **Idempotency Guard**:
   - Duplicate requests with identical report fingerprints are skipped to prevent resource exhaustion.

---

## 8. Limitations & Clinical Safety Boundary

> [!CAUTION]
> **MLOps Quality Gate $\neq$ Clinical Efficacy Validation**:
> Automated retraining, champion/challenger competition, and quality gate evaluations are engineering safeguards ensuring mathematical non-regression.
> Passing the quality gate does NOT constitute clinical approval or medical safety validation.
> This repository is strictly an educational MLOps platform demonstration.
