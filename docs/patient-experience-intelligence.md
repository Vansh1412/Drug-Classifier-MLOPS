# MedMLOps-X — Sprint 9: Patient Experience Intelligence

## 1. Executive Summary & Purpose

Sprint 9 builds an explainable, leakage-safe **Patient Experience Intelligence** layer on top of the existing Drug Review Intelligence workload (`Data/reviews/processed/reviews_cleaned.parquet`, 53,766 records).

While Sprint 8 established the classical NLP benchmark and strictly enforced quality gates (concluding with `NLP Champion = NONE`), Sprint 9 transforms the cleaned patient review corpus into actionable, descriptive healthcare intelligence:
- **Sentiment Analytics**: Overall distribution, rating breakdown, and descriptive statistics.
- **Drug-Level Experience**: Aggregations for drugs with sufficient review volume ($\ge 20$ reviews).
- **Condition-Level Experience**: Aggregations per condition, explicitly handling `"Unknown"` categories.
- **Temporal Experience**: Yearly volume, mean rating, and sentiment evolution over time.
- **Aspect / Theme Intelligence**: Interpretable, keyword-guided aspect extraction across 8 core clinical domains.
- **Lexical Review Similarity**: In-memory TF-IDF cosine similarity search for retrieval of similar patient experiences.

---

## 2. Medical Safety Boundary

> [!IMPORTANT]
> **Strict Medical Safety Boundary**:
> The analytics describe patterns in the supplied patient-reported review dataset and should **NOT** be interpreted as clinical evidence, medical advice, drug safety conclusions, efficacy conclusions, or prescribing recommendations.
>
> This system is designed for **Patient-Reported Drug Experience Analytics**.
> It is **NOT**:
> - A clinical decision-support system
> - A diagnostic tool
> - A prescription recommendation system
> - Evidence of drug efficacy or safety
> - Clinical trial evidence
>
> All outputs are strictly descriptive (e.g., *"Reviews in this dataset show..."*, *"Among drugs with at least 20 reviews..."*). Never use evaluative or promotional labels such as *"best drug"*, *"safest drug"*, or *"recommended drug"*.

---

## 3. Sentiment Analytics Methodology (S9-T2)

### 3.1 Target Definition & Mapping

Sentiment classes are deterministically derived from the numerical review `rating` (1–10) established in Sprint 7 and Sprint 8:

$$\text{sentiment\_class} = \begin{cases} \text{"low"} & \text{if } 1 \le \text{rating} \le 4 \\ \text{"neutral"} & \text{if } 5 \le \text{rating} \le 7 \\ \text{"high"} & \text{if } 8 \le \text{rating} \le 10 \end{cases}$$

The original numeric `rating` is preserved alongside the derived `sentiment_class` for auditability and lineage.

### 3.2 Dataset-Wide Distribution

Across the 53,766 patient reviews:
- **High (8–10)**: 32,349 reviews (**60.17%**)
- **Low (1–4)**: 13,497 reviews (**25.10%**)
- **Neutral (5–7)**: 7,920 reviews (**14.73%**)
- **Mean Rating**: **6.9769** ($\pm 3.28$ std, median: 8.0)
- **Mean Useful Count**: **28.0** ($\pm 36.3$ std, median: 16.0)

---

## 4. Drug-Level Experience Intelligence (S9-T3)

### 4.1 Minimum Review Threshold Policy

To prevent misleading conclusions drawn from tiny sample sizes (e.g., a drug with a single 10/10 review appearing as 100% positive), a configurable review threshold is enforced:

$$\text{min\_reviews\_threshold} = 20$$

- **Total Unique Drugs**: 2,637
- **Drugs Meeting Threshold**: **521 drugs**

### 4.2 Top-Reviewed Drug Samples (Descriptive Only)

| Drug Name | Review Count | Mean Rating | Median | Low % | Neutral % | High % | Mean Useful |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Levonorgestrel** | 1,273 | 7.37 | 9.0 | 23.33% | 12.88% | 63.79% | 29.5 |
| **Etonogestrel** | 1,085 | 5.89 | 6.0 | 39.82% | 17.69% | 42.49% | 23.1 |
| **Ethinyl estradiol / norethindrone** | 903 | 5.58 | 6.0 | 44.52% | 16.06% | 39.42% | 23.8 |
| **Nexplanon** | 736 | 5.56 | 6.0 | 45.38% | 16.98% | 37.64% | 17.4 |
| **Ethinyl estradiol / norgestimate** | 673 | 5.63 | 6.0 | 43.83% | 16.20% | 39.97% | 26.2 |
| **Phentermine** | 542 | 8.75 | 10.0 | 8.30% | 5.90% | 85.79% | 53.4 |
| **Sertraline** | 508 | 7.33 | 9.0 | 20.28% | 14.17% | 65.55% | 48.7 |
| **Escitalopram** | 455 | 7.79 | 9.0 | 17.80% | 11.21% | 70.99% | 45.2 |

*Note*: Metrics represent patient-reported sentiment within this dataset and are not clinical efficacy measures.

---

## 5. Condition-Level Experience Intelligence (S9-T4)

### 5.1 Condition Handling & Missing Value Protocol

- **Total Unique Conditions**: 708
- **Condition Threshold**: $\ge 10$ reviews
- **Missing Condition Category**: The 295 records with missing conditions are explicitly grouped under `"Unknown"` and flagged with `is_missing_category: true`.
- **Zero Clinical Inference**: Missing conditions are never inferred from drug names, review text, or co-occurrence patterns.

### 5.2 Top-Reviewed Conditions (Descriptive Only)

| Condition | Review Count | Mean Rating | Median | Low % | Neutral % | High % | Missing Category? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Birth Control** | 9,648 | 6.04 | 7.0 | 38.61% | 17.06% | 44.33% | No |
| **Depression** | 3,095 | 7.10 | 8.0 | 23.36% | 13.96% | 62.68% | No |
| **Pain** | 2,100 | 7.58 | 9.0 | 20.33% | 12.19% | 67.48% | No |
| **Anxiety** | 1,908 | 7.68 | 9.0 | 18.08% | 12.63% | 69.29% | No |
| **Acne** | 1,847 | 7.37 | 9.0 | 22.85% | 12.29% | 64.86% | No |
| **Bipolar Disorder** | 1,380 | 7.02 | 8.0 | 25.07% | 14.71% | 60.22% | No |
| **Weight Loss** | 1,248 | 7.97 | 9.0 | 15.63% | 9.53% | 74.84% | No |
| **Insomnia** | 1,231 | 6.60 | 8.0 | 31.68% | 14.13% | 54.18% | No |
| **Unknown** | 295 | 6.95 | 8.0 | 25.76% | 14.58% | 59.66% | **Yes** |

---

## 6. Temporal Experience Trends (S9-T5)

The dataset spans 10 years of patient-reported experiences (2008–2017). Yearly review volume expanded significantly over time while aggregate mean rating remained stable between 6.8 and 7.2.

| Year | Review Volume | Mean Rating | Low % | Neutral % | High % |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2008** | 1,029 | 7.13 | 24.10% | 13.99% | 61.90% |
| **2009** | 2,492 | 6.94 | 25.56% | 14.89% | 59.55% |
| **2010** | 2,477 | 6.99 | 24.63% | 15.30% | 60.07% |
| **2011** | 3,382 | 6.98 | 24.78% | 15.35% | 59.88% |
| **2012** | 3,962 | 6.88 | 26.55% | 15.14% | 58.30% |
| **2013** | 3,923 | 6.86 | 26.92% | 14.50% | 58.58% |
| **2014** | 4,142 | 6.92 | 26.00% | 14.58% | 59.42% |
| **2015** | 7,678 | 6.89 | 26.24% | 15.42% | 58.34% |
| **2016** | 12,329 | 7.02 | 24.84% | 14.37% | 60.78% |
| **2017** | 12,352 | 7.05 | 24.08% | 14.41% | 61.50% |

---

## 7. Aspect / Theme Extraction (S9-T6, S9-T7)

### 7.1 Methodology

Aspect intelligence utilizes transparent, interpretable domain keyword dictionaries combined with word-boundary regular expressions (`\b(?:...)\b`). This eliminates the opacity and computational overhead of black-box language models while ensuring 100% determinism.

### 7.2 Clinical Aspect Findings

| Aspect Domain | Key Terms | Mention Count | % of Reviews | Mean Rating | High % | Low % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dosage & Administration** | dose, dosage, pill, tablet, injection, mg, daily | 17,102 | **31.81%** | 6.93 | 59.5% | 25.4% |
| **Mental & Mood** | anxiety, depression, panic, mood, calm, irritable | 12,962 | **24.11%** | 7.03 | 61.2% | 24.7% |
| **Side Effects** | side effect, nausea, headache, dizziness, fatigue | 10,248 | **19.06%** | 6.32 | 53.0% | 31.6% |
| **Effectiveness** | effective, cured, worked wonders, relief, miracle | 10,123 | **18.83%** | 8.35 | 79.5% | 12.0% |
| **Sleep & Drowsiness** | insomnia, drowsy, sleep, sleeping, tired, nightmare | 8,574 | **15.95%** | 6.92 | 59.8% | 25.2% |
| **Weight & Appetite** | weight gain, weight loss, appetite, pounds, hungry | 7,823 | **14.55%** | 6.88 | 59.1% | 26.0% |
| **Cost & Affordability** | cost, expensive, insurance, copay, affordable, price | 3,150 | **5.86%** | 6.64 | 56.4% | 28.5% |
| **Withdrawal & Dependence** | withdrawal, taper, dependence, wean, stopping | 2,246 | **4.18%** | 6.45 | 53.9% | 32.1% |

---

## 8. Lexical Review Similarity Engine (S9-T8)

The `ReviewSimilarityEngine` provides transparent lexical retrieval using TF-IDF feature matrices and cosine similarity:

```text
Query: "severe headache nausea after morning dose"
                ↓
    TfidfVectorizer.transform([query])
                ↓
    Cosine Similarity vs Corpus Matrix (53,766 documents)
                ↓
    Optional Filtering (by drugName, condition)
                ↓
    Top-k Ranking & Exemplar Extraction
```

### Key Features:
- **Zero External Dependencies**: Operates in-memory using Scikit-learn without external vector databases (no FAISS, Pinecone, Chroma).
- **Sub-10ms Latency**: Sparse matrix vectorization and dot-product similarity execute in milliseconds.
- **Robust Edge-Case Handling**: Empty queries, whitespace-only inputs, and out-of-vocabulary terms return empty lists cleanly without exceptions.

---

## 9. Privacy, Safety, and PII Controls

- **No Full Review Dumping**: Generated summary reports (`Results/patient_experience_summary.json`) contain only aggregated counts, percentages, and metrics.
- **Traceable Exemplars**: Full report exemplars store `uniqueID` and `review_clean_hash` with bounded, truncated text snippets ($\le 120$ characters) for contextual verification without full text duplication.
- **Log Sanitation**: Raw review text and patient identifiers are never emitted to logs, exception traces, or MLflow parameters.

---

## 10. Reproducibility & CLI Execution

To reproduce the complete Patient Experience Intelligence analysis:
```bash
# Run analysis and generate reports
python -c "from src.patient_experience import generate_patient_experience_report; generate_patient_experience_report()"

# Or reproduce via DVC
dvc repro patient_experience_analysis
```

---

## 11. Limitations & Future Work

1. **Self-Selection Bias**: Patient-reported review datasets inherently suffer from self-selection bias (patients with extremely positive or negative experiences are more likely to submit reviews).
2. **Lexical Keyword Limitations**: Transparent keyword matching does not capture complex negations (e.g., *"did not cause nausea"*) as effectively as contextual embeddings.
3. **No Clinical Attribution**: Adverse experience mentions (e.g. *"nausea"*) reflect co-occurrence in patient reports, not clinically verified adverse drug events.
