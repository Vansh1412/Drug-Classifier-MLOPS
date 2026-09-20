# MedMLOps-X — Sprint 8: NLP Model Competition & Baseline Evaluation

## 1. Executive Summary

Sprint 8 establishes a reproducible, leakage-safe **NLP model competition pipeline** for the **Drug Review Intelligence** workload. The primary objective is to benchmark classical NLP baselines (**Logistic Regression**, **Linear SVM**, and **Multinomial Naive Bayes**) under strict leakage-prevention constraints and evaluate them against an automated production **Quality Gate**.

### Medical Safety Boundary

> [!IMPORTANT]
> **Medical Safety Boundary**:
> This system is designed for **Patient-Reported Drug Experience Analytics**.
> It is **NOT**:
> - A clinical decision-support system
> - A prescription recommendation system
> - A diagnostic system
> - A medical advice system
> - Evidence of drug efficacy or safety
> - A clinical trial analysis
>
> *Model performance on patient-reported drug reviews should never be interpreted as clinical efficacy, safety, diagnosis, or prescribing evidence.*

---

## 2. Review Dataset & Target Definition (S8-T1)

### 2.1 Dataset Specifications

- **Source**: `Data/reviews/drugsComTest_raw.csv` (tracked via `Data/reviews/drugsComTest_raw.csv.dvc`)
- **Processed Artifact**: `Data/reviews/processed/reviews_cleaned.parquet`
- **Total Records**: 53,766 patient review records
- **Features Used**: `review_clean` (strictly text-only benchmark)
- **Features Excluded from Training**: `rating` (used only for target derivation), `usefulCount`, `condition`, `drugName`

### 2.2 Target Transformation

The prediction target (`sentiment_class`) is deterministically derived from the numerical patient rating (1–10) using the standardized Sprint 7 rating bins:

$$\text{sentiment\_class} = \begin{cases} \text{"low"} & \text{if } 1 \le \text{rating} \le 4 \\ \text{"neutral"} & \text{if } 5 \le \text{rating} \le 7 \\ \text{"high"} & \text{if } 8 \le \text{rating} \le 10 \end{cases}$$

The original numeric `rating` is strictly preserved in the DataFrame for auditability, lineage, and downstream calibration.

### 2.3 Class Distribution & Severe Imbalance

| Sentiment Class | Rating Range | Count | Proportion |
| :--- | :---: | :---: | :---: |
| **High** | 8–10 | 32,349 | 60.17% |
| **Low** | 1–4 | 13,497 | 25.10% |
| **Neutral** | 5–7 | 7,920 | 14.73% |
| **Total** | 1–10 | 53,766 | 100.00% |

> [!NOTE]
> The target distribution is heavily bimodal and imbalanced: positive experiences ("high") comprise over 60% of reports, while neutral experiences comprise only 14.7%. As an engineering principle, we do not synthetically generate or oversample review text. Consequently, **Macro F1** is the authoritative metric for ranking and quality gating.

---

## 3. Leakage Prevention & Dataset Splitting (S8-T2, S8-T3)

### 3.1 Identical Review Text De-duplication & Grouping

In the raw dataset, **5,486 records** share identical review texts across different patients and conditions. A standard random split would leak identical texts between train and test sets.

We utilize `split_reviews_leakage_safe(strategy="stratified_group", group_col="review_clean_hash")`:
- **Hashing**: Deterministic SHA-256 hash (`review_clean_hash`) computed on normalized lowercase text.
- **Grouping**: Entire groups of identical text are assigned together to either train or test via `StratifiedGroupKFold`.
- **Guarantee**: $\text{Train}_{\text{text}} \cap \text{Test}_{\text{text}} = \emptyset$ (strictly 0 text overlap).
- **Split Sizes**:
  - **Train**: 43,052 rows (80.07%)
  - **Test**: 10,714 rows (19.93%)
  - **Verified Overlap**: 0 records

### 3.2 TF-IDF Pipeline Encapsulation

To guarantee that vocabulary, inverse document frequencies, and feature scalings never leak from test data or cross-validation holdouts:
- Feature extraction and classification are unified inside an `sklearn.pipeline.Pipeline`:
  ```python
  Pipeline([
      ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.90, sublinear_tf=True, max_features=10000)),
      ("classifier", clf)
  ])
  ```
- The vectorizer is **fitted only on the training split/fold**, never on the combined dataset.
- Unseen n-grams in the test set are cleanly ignored at inference time.

---

## 4. Model Competition & Cross-Validation (S8-T4, S8-T5)

### 4.1 Candidate Architectures

Three classical NLP models were evaluated using the identical dataset split, preprocessing pipeline, and evaluation metrics:

1. **Logistic Regression**:
   - Class weight: `balanced` (compensates for class skew)
   - Solver: `lbfgs`, `C=1.0`, `max_iter=1000`, `random_state=125`
2. **Linear SVM (`LinearSVC`)**:
   - Class weight: `balanced`
   - `C=1.0`, `max_iter=2000`, `random_state=125`
3. **Multinomial Naive Bayes (`MultinomialNB`)**:
   - Smoothing parameter: $\alpha = 0.5$

### 4.2 Group-Aware Cross-Validation Protocol

- **Protocol**: 5-fold `StratifiedGroupKFold` on the 43,052 training samples using `review_clean_hash` as grouping.
- **Primary Metric**: **Macro F1** (unweighted average of F1 across Low, Neutral, and High).
- **Secondary Metrics**: Accuracy, Macro Precision, Macro Recall, Class-wise Recall.

---

## 5. Competition Results & Evaluation

### 5.1 Model Comparison Table

| Candidate Model | CV Macro F1 | CV Std | Test Macro F1 | Test Accuracy | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.6242** | $\pm 0.0045$ | **0.6332** | 70.40% | **REJECTED (Gate Fail)** |
| **Linear SVM** | 0.6033 | $\pm 0.0052$ | 0.6170 | 71.71% | **REJECTED (Gate Fail)** |
| **Multinomial Naive Bayes** | 0.5149 | $\pm 0.0063$ | 0.5208 | **72.36%** | **REJECTED (Gate Fail)** |

### 5.2 Critical Analysis of Class Imbalance & The "Accuracy Trap"

The competition results illustrate a classic ML phenomenon:
- **Naive Bayes achieved the highest raw accuracy (72.36%)**, but its **Macro F1 was only 0.5208**. Why?
  - High class recall: **95.77%**
  - Low class recall: **56.35%**
  - Neutral class recall: **5.34%** (F1: 0.0956)
  - Naive Bayes achieved high accuracy by almost completely ignoring the minority neutral class (14.73% of data) and defaulting to the majority "high" class.
- **Logistic Regression with Balanced Class Weights** achieved the highest Macro F1 (**0.6242 CV / 0.6332 Test**):
  - High class recall: 75.33% (F1: 0.8121)
  - Low class recall: 71.68% (F1: 0.7028)
  - Neutral class recall: 48.40% (F1: 0.3849)
  - It balanced predictions across all classes, but the difficulty of separating nuanced neutral drug sentiment using unigram/bigram TF-IDF bounded overall Macro F1 to ~0.63.

---

## 6. Quality Gate & Champion Selection (S8-T6)

### 6.1 Quality Gate Criteria

The NLP Quality Gate enforces three absolute requirements:
1. $\text{CV Macro F1} \ge 0.70$
2. $\text{Test Macro F1} \ge 0.70$
3. $\text{Test Accuracy} \ge 0.70$
4. $\text{Per-Class Recall} \ge 0.40$ (prevents minority class collapse)

### 6.2 Quality Gate Outcome: NO NLP CHAMPION

| Model | Gate Pass? | Gate Failure Reasons |
| :--- | :---: | :--- |
| **Logistic Regression** | **FAIL** | CV Macro F1 (0.6242) < 0.7000; Test Macro F1 (0.6332) < 0.7000 |
| **Linear SVM** | **FAIL** | CV Macro F1 (0.6033) < 0.7000; Test Macro F1 (0.6170) < 0.7000; Neutral Recall (0.3427) < 0.4000 |
| **Multinomial Naive Bayes** | **FAIL** | CV Macro F1 (0.5149) < 0.7000; Test Macro F1 (0.5208) < 0.7000; Neutral Recall (0.0534) < 0.4000 |

### Outcome:
```text
NLP CHAMPION: NONE
```

> [!NOTE]
> **Engineering Integrity Principle**:
> We do NOT lower quality gate thresholds or manipulate metrics to manufacture a champion. Classical bag-of-words / TF-IDF representations lack the contextual depth required to accurately distinguish nuanced neutral sentiment in patient reviews. This honest benchmark establishes the quantitative necessity for contextual representation in future sprints.

---

## 7. MLflow Experiment Tracking (S8-T7)

- **Dedicated Experiment**: `MedMLOps-X-NLP-Drug-Review` (Experiment ID: `2`)
- **Tracking URI**: `sqlite:///mlruns.db`
- **Runs Logged**:
  - `nlp-candidate-logistic_regression` (Run ID: `1213a88e44904b6aa17308b2ee2c7342`)
  - `nlp-candidate-linear_svm` (Run ID: `b5fe935c22ed40b8910583caac0d1ccc`)
  - `nlp-candidate-naive_bayes` (Run ID: `4918647fa5374e6caaf4f7da57ad56d0`)
- **Logged Parameters**: `model_name`, `tfidf_ngram_range`, `tfidf_max_features`, `tfidf_sublinear_tf`, `classifier_C`, `classifier_solver`, `classifier_class_weight`, `classifier_alpha`, `split_strategy`, `cv_folds`.
- **Logged Metrics**: `cv_macro_f1_mean`, `cv_macro_f1_std`, `cv_accuracy_mean`, `cv_accuracy_std`, `test_macro_f1`, `test_accuracy`, `test_precision_macro`, `test_recall_macro`, `test_f1_low`, `test_f1_neutral`, `test_f1_high`, `test_recall_neutral`.
- **Tags**: `workload: nlp`, `sprint: 8`, `dataset: drugsComTest_raw`, `git_commit`, `dvc_dataset_hash`.

---

## 8. Artifact Isolation & Integrity (S8-T8)

The structured tabular ML champion (`Model/drug_pipeline.skops`, `Model/model_metadata.json`, `Data/drug.csv`) remains **100% untouched and protected**.

Sprint 8 artifacts are strictly isolated:
```text
Model/
├── drug_pipeline.skops          <-- [PROTECTED] Tabular XGBoost Champion
├── model_metadata.json          <-- [PROTECTED] Tabular Metadata
├── registry_manifest.json       <-- [PROTECTED] Tabular Registry
└── nlp/                         <-- [NEW] Isolated NLP directory
    └── champion/                <-- (Empty: No champion promoted)
Results/
├── competition_report.json      <-- [PROTECTED] Tabular Competition Report
├── metrics.json                 <-- [PROTECTED] Tabular Metrics
├── nlp_competition_report.json  <-- [NEW] Full NLP Competition Report
└── nlp_metrics.json             <-- [NEW] Summary NLP Competition Metrics
```

---

## 9. Reproducibility & CLI Execution

To reproduce the NLP model competition pipeline:
```bash
# Run NLP model competition end-to-end
python -c "from src.nlp_competition import run_nlp_competition; run_nlp_competition()"

# Or execute via DVC
dvc repro nlp_competition
```
