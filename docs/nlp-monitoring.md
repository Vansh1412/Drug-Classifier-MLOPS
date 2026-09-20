# MedMLOps-X — NLP Monitoring & Drift Detection Architecture

## 1. Executive Summary & Overview

Sprint 10 introduces the **NLP Monitoring & Drift Detection** layer for MedMLOps-X, extending the production monitoring architecture to support the **Patient Experience Intelligence / Drug Review** workload (53,766 patient-reported review records).

The system monitors incoming review batches across ten distinct dimensions:
1. **Schema & Data Quality**: Missing/unexpected columns, null rates, empty reviews, out-of-bounds ratings (1–10), and invalid dates.
2. **Review Length Drift**: Character, word, and sentence count distributions using two-sample Kolmogorov-Smirnov (KS) test, normalized Wasserstein distance (Earth Mover's Distance), and Population Stability Index (PSI).
3. **Vocabulary Drift**: Lexical overlap, Jaccard similarity, new token (out-of-vocabulary) rate, and top token frequency shifts.
4. **TF-IDF Feature Distribution Drift**: Fixed reference vectorizer (never re-fitted on incoming data), transform-only execution on current batches, and mean feature vector cosine distance with top feature distribution PSI.
5. **Sentiment Distribution Drift**: 3-class categorical distribution tracking (`low`, `neutral`, `high`) using categorical PSI and Chi-Square contingency testing.
6. **Rating Distribution Drift**: 1–10 numeric ratings, descriptive statistics (mean, median, std, quantiles), KS test, and Wasserstein distance.
7. **Aspect Drift**: Clinical domain mention frequencies across 8 predefined domains (`side_effects`, `effectiveness`, `dosage_administration`, `cost_affordability`, `sleep_drowsiness`, `mental_mood`, `weight_appetite`, `withdrawal_dependence`).
8. **Drug & Condition Category Drift**: Frequency distributions for `drugName` and `condition`, explicit handling of `"Unknown"` as a missing placeholder, and stable PSI grouping.
9. **NLP Prediction Drift**: Supports `NOT_AVAILABLE` status when no NLP champion exists (Sprint 8 produced `NLP Champion = NONE`), and evaluates categorical PSI when a prediction stream is explicitly provided.
10. **Overall Drift Certification**: Deterministic state machine producing `PASS`, `WARNING`, `DRIFT`, or `ERROR` certification.

---

## 2. Medical Safety Boundary

> [!CAUTION]
> **CRITICAL MEDICAL SAFETY DISCLAIMER**:
> This system analyzes **Patient-Reported Drug Experience Analytics**.
> Statistical drift in patient-reported review data indicates a change in the **observed data distribution**.
> It does **NOT** establish:
> - Clinical efficacy
> - Drug safety or adverse event rate changes
> - Causality or pharmacovigilance confirmation
> - Diagnosis or clinical outcome degradation
> - Treatment or prescription recommendations
>
> An increase in negative review sentiment or side-effect keyword mentions means only that the **lexical frequency of patient-reported text shifted**. It does NOT mean the medication became less safe or less effective.

---

## 3. Architecture & Unified Monitoring

```text
                        MedMLOps-X Monitoring
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
           Structured ML                         NLP ML
                 │                                 │
           Sprint 5 Engine                   Sprint 10 Engine
         (Age, BP, Na_to_K)             (Text, Vocab, Aspects)
                 │                                 │
                 └────────────────┬────────────────┘
                                  │
                          Unified Monitoring
                                  │
                      ┌───────────┴───────────┐
                      ▼                       ▼
              Structured Report          NLP Report
            (monitoring/reports)    (monitoring/nlp_reports)
                      │                       │
                      ▼                       ▼
                 FastAPI /               FastAPI /
              Structured API              NLP API
                      │                       │
                      ▼                       ▼
               Sprint 6 Retrain        Sprint 11 Retrain
             (Structured Champion)     (Future NLP Retrain)
```

---

## 4. Reference Baseline

The reference distribution is fixed and versioned:
- **Reference Dataset**: `Data/reviews/processed/reviews_cleaned.parquet`
- **Rows**: 53,766 preprocessed patient reviews
- **Dataset SHA-256**: `1057924d53c81f85802c58ee60247abaaf70174c977233a8adfc37d744b81b2c`
- **DVC Dataset Hash**: `9e9461e2200485dd1f6743ed17dff4c6`
- **Git Commit SHA**: `faac1e53d1f9c9de30a6060cf4b28ea4d8a5a57c`
- **Reference Immutability**: The reference dataset cannot be silently modified or replaced during runtime monitoring. All drift comparisons evaluate `reference` vs `current`.

---

## 5. Statistical Methodologies & Drift Metrics

### 5.1 Text Length Drift (S10-T4)
Evaluates `review_char_count`, `review_word_count`, and `review_sentence_count`:
- **Two-Sample Kolmogorov-Smirnov (KS) Test**:
  $$\text{KS Statistic} = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$$
  Flags `WARNING` if $p < 0.05$; flags `DRIFT` if $p < 0.01$.
- **Normalized Wasserstein Distance**:
  $$W_1(u, v) = \int_{-\infty}^{\infty} |U(x) - V(x)| dx$$
  Normalized by $\sigma_{\text{ref}}$. Flags `WARNING` if $W \ge 0.15$; flags `DRIFT` if $W \ge 0.30$.
- **Numerical PSI**: Binned using 10 reference quantile intervals. Flags `WARNING` if $\text{PSI} \ge 0.10$; flags `DRIFT` if $\text{PSI} \ge 0.20$.

### 5.2 Vocabulary Drift (S10-T5)
Extracts deterministic word tokens ($\ge 2$ characters):
- **New Token Rate**: $\frac{|V_{\text{curr}} \setminus V_{\text{ref}}|}{|V_{\text{curr}}|}$ (fraction of batch tokens never seen in baseline).
- **Top Tokens Missing Rate**: $\frac{|T_{\text{ref, top}} \setminus V_{\text{curr}}|}{|T_{\text{ref, top}}|}$ (fraction of the top 50 baseline tokens absent in batch).
- **Jaccard Similarity**: $\frac{|V_{\text{ref}} \cap V_{\text{curr}}|}{|V_{\text{ref}} \cup V_{\text{curr}}|}$.
- Evaluated as purely lexical monitoring without medical inferences.

### 5.3 TF-IDF Feature Distribution Drift (S10-T6)
- The reference `TfidfVectorizer` ($N=5000$ max features, sublinear TF) is **fit once on the reference dataset**.
- Incoming batches are processed via `.transform()` **only**; `fit` and `fit_transform` are strictly forbidden during monitoring.
- **Cosine Distance**: $1.0 - \cos(\bar{v}_{\text{ref}}, \bar{v}_{\text{curr}})$. Flags `WARNING` if $\ge 0.15$; flags `DRIFT` if $\ge 0.30$.
- Tracks top-10 individual feature distribution PSIs.

### 5.4 Sentiment & Rating Drift (S10-T7 & S10-T8)
- **Sentiment Classes**: `low` (1–4), `neutral` (5–7), `high` (8–10).
- **Categorical PSI**: $\sum (A_i - E_i) \ln(A_i / E_i)$. Flags `WARNING` if $\ge 0.10$; flags `DRIFT` if $\ge 0.20$.
- **Chi-Square Contingency Test**: Assesses distribution independence.
- **Rating Summary**: Mean, median, std, min, max, and quantiles compared via KS test and Wasserstein distance.

### 5.5 Aspect Drift (S10-T9)
Tracks mention percentage across 8 clinical domains established in Sprint 9:
- `side_effects` (Baseline: 19.06%)
- `effectiveness` (Baseline: 18.83%)
- `dosage_administration` (Baseline: 31.81%)
- `cost_affordability` (Baseline: 5.86%)
- `sleep_drowsiness` (Baseline: 15.95%)
- `mental_mood` (Baseline: 24.11%)
- `weight_appetite` (Baseline: 14.55%)
- `withdrawal_dependence` (Baseline: 4.18%)
- Flags `WARNING` if absolute change $\ge 5.0$ percentage points; flags `DRIFT` if $\ge 10.0$ percentage points.

### 5.6 Category Drift (S10-T10)
- Monitors `drugName` and `condition`.
- `"Unknown"` is explicitly treated as an unassigned/missing condition placeholder.
- To prevent small-sample cardinality distortions, clinical conditions and drugs are evaluated on the Top 15 categories + an `'Other'` bucket.

### 5.7 NLP Prediction Drift (S10-T11)
- Sprint 8 resulted in `NLP Champion = NONE`.
- If no prediction stream is provided, prediction drift reports:
  ```json
  {
    "status": "NOT_AVAILABLE",
    "drift_detected": false,
    "reason": "Sprint 8 produced no accepted NLP champion model; prediction stream not supplied."
  }
  ```
- When predictions are provided, evaluates categorical PSI against baseline classes.

---

## 6. Overall Certification State Machine (S10-T12)

```text
       ┌──────────────────────────────────────────────┐
       │ Incoming Batch & Data Quality Checks        │
       └──────────────────────┬───────────────────────┘
                              │
               Critical Schema / Rating Error?
                              │
               YES ───────────┴─────────── NO
                │                           │
                ▼                           ▼
            [ ERROR ]             Statistical Distributions
         (CRITICAL Sev)            (Length, Vocab, TF-IDF,
                                    Sentiment, Rating, Aspect)
                                            │
                                  Any Metric >= DRIFT?
                                            │
                             YES ───────────┴─────────── NO
                              │                           │
                              ▼                           ▼
                          [ DRIFT ]              Any Metric >= WARN?
                         (HIGH Sev)                       │
                                           YES ───────────┴─────────── NO
                                            │                           │
                                            ▼                           ▼
                                       [ WARNING ]                  [ PASS ]
                                      (MEDIUM Sev)                  (LOW Sev)
```

**Certification Policy**:
- `ERROR`: Critical data quality failure (missing columns, invalid rating range $<1$ or $>10$).
- `DRIFT`: One or more critical distributions drift beyond drift thresholds.
- `WARNING`: Non-critical shift (e.g., moderate vocabulary novelty or slight length variance).
- `PASS`: All metrics remain within normal baseline variance.
- **Sprint 10 Scope Rule**: `automated_retraining_triggered: false` is strictly enforced.

---

## 7. API Integration

### `POST /monitoring/nlp/drift`
Evaluates an incoming batch of patient reviews (up to 1,000 items) against the reference baseline.
- **Request Body**:
  ```json
  {
    "reviews": [
      {
        "uniqueID": 1001,
        "drugName": "Lipitor",
        "condition": "High Cholesterol",
        "review": "Very effective medication, lowered my cholesterol significantly.",
        "rating": 9,
        "date": "28-Feb-12",
        "usefulCount": 5
      }
    ],
    "predictions": ["high"]
  }
  ```
- **Response**: Full structured certification report.
- **Privacy Guarantee**: Zero raw patient review text is returned in the API response.

### `GET /monitoring/nlp/latest`
Retrieves the most recently computed and persisted NLP drift report (`Results/nlp_drift_report.json`).

---

## 8. DVC Pipeline Integration

The NLP monitoring stage is declared in `dvc.yaml`:
```yaml
  nlp_monitoring:
    cmd: python -c "from src.nlp_monitoring import run_nlp_monitoring; run_nlp_monitoring()"
    deps:
      - Data/reviews/processed/reviews_cleaned.parquet
      - configs/model_config.yaml
      - src/nlp_monitoring.py
    outs:
      - Results/nlp_drift_report.json:
          cache: false
```
- Completely isolated from structured ML stages (`validate`, `train`).
- Consumes preprocessed review parquet and produces canonical JSON report.

---

## 9. Security Review & Privacy Guarantees

1. **No Raw Review Text in Reports or Responses**: Only statistical aggregates, distributions, token shifts, and hashes are stored.
2. **Path Traversal Protection**: `save_report` validates that destination paths resolve strictly within `monitoring/nlp_reports/`.
3. **No Automated Retraining**: Automated retraining triggers are disabled for NLP in Sprint 10.
4. **No Secrets / Credentials**: Zero API keys or tokens are stored in reports or code.
5. **Deterministic Testing**: Synthetic current batches are created without modifying raw data.
