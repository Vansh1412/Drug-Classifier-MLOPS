# NLP Automated Retraining, Quality Gate, Promotion & Rollback Guide

## 1. Overview

In Sprint 11, **MedMLOps-X** extends the closed-loop MLOps architecture to the **Drug Review Intelligence / NLP Workload**. By connecting Sprint 10 NLP Continuous Monitoring drift signals (`Results/nlp_drift_report.json`) to an **NLP Automated Retraining Trigger Engine**, **NLP Dual Quality Gate**, and an **Isolated NLP Model Registry with Atomic Promotion and Rollback**, the platform enables resilient lifecycle automation for patient-reported review analytics.

### Safety & Architecture Guarantees
- **Workload Isolation**: Structured model artifacts (`Model/drug_pipeline.skops`, `Model/registry_manifest.json`) are strictly protected and never touched, overwritten, or modified.
- **NLP Champion = NONE Initial State**: Gracefully handles the absence of an initial champion (from Sprint 8 quality gate failure) via **Bootstrap Promotion**.
- **Drift Certification**: Retraining is triggered **ONLY on certified `DRIFT`**; `PASS`, `WARNING`, and `ERROR` certifications are strictly rejected.
- **Leakage-Safe Retraining**: TF-IDF vectorization is fitted strictly inside cross-validation folds within `Pipeline([('tfidf', ...), ('classifier', ...)])`. Splits are group-aware (`StratifiedGroupKFold` on `review_clean_hash`) ensuring zero test review leakage.
- **Isolated Challenger Workspace**: Challengers train in dedicated workspaces (`Model/nlp/challengers/<retraining_id>/`).
- **Dual Quality Gate**: Requires absolute thresholds (`Macro F1 >= 0.70`, `Accuracy >= 0.70`, `All recalls >= 0.40`) and comparative non-regression when a champion exists.
- **Atomic Promotion**: Stages candidate artifacts with atomic file swaps and updates `Model/nlp/registry_manifest.json`.
- **Smoke Testing & Rollback**: Validates Skops deserialization and prediction on representative patient review texts; immediately rolls back if smoke testing fails.

---

## 2. End-to-End NLP Lifecycle Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. NLP MONITORING & DRIFT DETECTION (Sprint 10)             │
│    Reference vs Serving Reviews ──► Results/nlp_drift_report│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. NLP DRIFT CERTIFICATION (src/nlp_retraining.py)          │
│    • overall_status == "DRIFT" (PASS/WARNING/ERROR rejected) │
│    • data_quality.status != "ERROR"                         │
│    • current.rows >= min_samples (N >= 20)                  │
│    • drift_detected == True or signals present              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. IDEMPOTENT TRIGGER ENGINE                                │
│    Hash = SHA256(report_id :: champion_ver :: dvc_hash)     │
│    • If duplicate ──► Return SKIPPED                        │
│    • Else ──► Transition to RETRAINING                      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. ISOLATED NLP CHALLENGER TRAINING                         │
│    • Data: Data/reviews/processed/reviews_cleaned.parquet   │
│    • Leakage-safe StratifiedGroupKFold on review_clean_hash │
│    • Candidates: LogisticRegression, LinearSVC, NaiveBayes  │
│    • Output: Model/nlp/challengers/<retraining_id>/         │
│    • Lineage logged to MLflow (MedMLOps-X-NLP-Drug-Review)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. NLP QUALITY GATE (Absolute + Comparative)                │
│    • Absolute Gate:                                         │
│        CV Macro F1 >= 0.70, Test Macro F1 >= 0.70           │
│        Test Accuracy >= 0.70, All class recalls >= 0.40     │
│    • Comparative Gate (Only if champion exists):            │
│        Challenger Macro F1 >= Champion Macro F1 - 0.02      │
│    • If Champion == NONE: Comparative gate bypassed         │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
         [ Gate PASS ]                  [ Gate FAIL ]
               │                              │
               ▼                              ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│ 6. ATOMIC PROMOTION          │ │ REJECT CHALLENGER          │
│    • Archive old champion    │ │ • Champion unchanged       │
│      to Model/nlp/archive/   │ │ • Status: REJECTED         │
│    • Stage & atomically swap │ │ • Audit logged in history  │
│    • Update Model/nlp/       │ └────────────────────────────┘
│      registry_manifest.json  │
└──────────────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. POST-PROMOTION SMOKE TEST                                │
│    • Load Model/nlp/champion/nlp_champion_pipeline.skops    │
│    • Test inference on representative drug review samples   │
│    • Verify predictions in {"low", "neutral", "high"}       │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
          [ Verified ]                   [ Failed ]
               │                              │
               ▼                              ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│ PROMOTION ACCEPTED           │ │ AUTOMATIC IMMEDIATE        │
│ • Active champion updated    │ │ ROLLBACK                   │
│ • Manifest status = ACTIVE   │ │ • If replacement: restore │
│ • History logged             │ │   previous champion        │
│                              │ │ • If bootstrap: reset to   │
│                              │ │   status = NONE            │
└──────────────────────────────┘ └────────────────────────────┘
```

---

## 3. NLP Champion = NONE Bootstrap Behavior

At the start of Sprint 11, the NLP workload has **no production champion** (`status: "NONE"`) because all Sprint 8 candidate models failed the quality gate (Macro F1 ~0.63 < 0.70).

The system handles this special initial condition deterministically:
1. **Initial Manifest**: `Model/nlp/registry_manifest.json` initializes with `status: "NONE"`, `active_champion: null`, `previous_champion: null`.
2. **API Endpoint**: `GET /models/nlp/champion` returns:
   ```json
   {
     "workload": "nlp",
     "status": "NONE",
     "message": "No NLP model has passed the production quality gate. NLP Champion is currently NONE.",
     "active_champion": null,
     "manifest": { ... }
   }
   ```
3. **Bootstrap Promotion**: When a challenger candidate is evaluated while champion is `NONE`:
   - Evaluates **Absolute Quality Gate only** (`Macro F1 >= 0.70`, `Accuracy >= 0.70`, `All recalls >= 0.40`).
   - Comparative non-regression gate is **bypassed** (no comparative metrics are fabricated against a nonexistent champion).
   - If candidate passes and smoke test verifies: promoted as `v1.0.0` with `is_bootstrap: true`.
   - If smoke test fails: automatically reverted to `status: "NONE"`.

---

## 4. Drift Certification Policy

The `NLPDriftCertifier` inspects drift monitoring reports before allowing automated retraining:

| Rule | Requirement | Rationale |
|---|---|---|
| **Overall Status** | `certification.overall_status == "DRIFT"` | Only confirmed drift triggers retraining; `PASS` and `WARNING` are rejected. |
| **Data Quality** | `data_quality.status != "ERROR"` | Payload errors, corrupt parquet files, or malformed records must not trigger retraining. |
| **Sample Size** | `current.rows >= min_samples` (default: 20) | Prevents stochastic small-sample noise from triggering retraining. |
| **Drift Signals** | `drift_detected == True` or `signals` present | Confirms actionable drift in vocabulary, text length, TF-IDF, sentiment, or aspect distributions. |

---

## 5. Dual Quality Gate

| Metric | Threshold | Scope | Rationale |
|---|---|---|---|
| **CV Macro F1** | `cv_macro_f1_mean >= 0.70` | 5-fold StratifiedGroupKFold | Prevents minority sentiment class collapse across folds. |
| **Test Macro F1** | `test_macro_f1 >= 0.70` | Held-out test set (20%) | Primary production performance gate. |
| **Test Accuracy** | `test_accuracy >= 0.70` | Held-out test set (20%) | Overall accuracy standard. |
| **Per-Class Recall** | `recall >= 0.40` for all classes (`low`, `neutral`, `high`) | Held-out test set (20%) | Enforces neutral sentiment class viability. |
| **Comparative F1 Margin** | `challenger_f1 >= champion_f1 - 0.02` | Comparative (when champion exists) | Non-regression safeguard against model degradation. |

---

## 6. Model Registry & Artifact Structure

```text
Model/
├── drug_pipeline.skops              <-- STRUCTURED CHAMPION (PROTECTED, UNTOUCHED)
├── model_metadata.json              <-- Structured metadata
├── registry_manifest.json           <-- Structured registry manifest
├── archive/                         <-- Structured champion archives
├── challengers/                     <-- Structured challenger workspaces
│
└── nlp/                             <-- ISOLATED NLP WORKLOAD DIRECTORY
    ├── registry_manifest.json       <-- Dedicated NLP registry manifest
    ├── nlp_model_metadata.json      <-- NLP champion metadata
    ├── champion/
    │   └── nlp_champion_pipeline.skops
    ├── archive/                     <-- Previous NLP champions
    └── challengers/                 <-- Isolated NLP candidate workspaces
        └── nlp_retrain_<id>/
            ├── nlp_challenger_pipeline.skops
            └── training_manifest.json
```

---

## 7. API Management Endpoints

All model management endpoints are privileged administrative operations protected by `X-API-Key`:

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/models/nlp/champion` | Public | Returns active champion details or `status: "NONE"`. |
| `GET` | `/nlp/retraining/status` | Public | Returns retraining state machine status, active champion, and audit count. |
| `POST` | `/nlp/retraining/trigger` | `X-API-Key` | Triggers certified retraining pipeline on DRIFT report. |
| `POST` | `/models/nlp/rollback` | `X-API-Key` | Restores previous champion from archive; returns 400 if no previous champion. |

---

## 8. Clinical Limitations & Medical Safety Boundaries

1. **Patient-Reported Reviews are Not Clinical Evidence**: Sentiment and ratings reflect individual patient experiences and cannot substitute for randomized controlled clinical trials.
2. **No Medical Advice**: The system does NOT diagnose conditions, prescribe medications, or assert clinical superiority of any drug.
3. **Lexical Limitations of TF-IDF**: Classical TF-IDF n-gram pipelines have vocabulary bounds and cannot infer deep contextual semantic subtleties.
4. **Quality Gate Integrity**: When candidates do not achieve Macro F1 >= 0.70, **NLP Champion = NONE is the valid and expected outcome**. Gates are never lowered to manufacture a champion.
