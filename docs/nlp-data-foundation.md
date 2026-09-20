# MedMLOps-X — Multi-Modal Drug Review Data Foundation

**Sprint:** 7 — Multi-Modal Data Foundation  
**Workload:** Drug Review Intelligence / Patient-Reported Drug Experience Analytics  
**Primary Dataset:** `Data/reviews/drugsComTest_raw.csv`  
**Processed Dataset:** `Data/reviews/processed/reviews_cleaned.parquet`  

---

## 1. Overview & Workload Purpose

MedMLOps-X is evolving from a single-workload tabular classification system into a **multi-workload MLOps platform**. 

```
                         MedMLOps-X
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
       STRUCTURED ML                         NLP ML
             │                                 │
      Data/drug.csv             Data/reviews/drugsComTest_raw.csv
             │                                 │
    Drug Classification               Review Intelligence
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                      Shared MLOps Layer
                              │
                  DVC / MLflow / Registry
                              │
                     Quality Gates
                              │
                       Monitoring
                              │
                       Retraining
                              │
                   Promotion/Rollback
```

While the structured pipeline predicts categorical medication recommendations based on physiological patient biomarkers (`Age`, `Sex`, `BP`, `Cholesterol`, `Na_to_K`), the **Drug Review Intelligence** workload establishes the foundational data architecture for analyzing natural language patient-reported experiences, sentiment, and treatment satisfaction.

---

## 2. Raw Dataset Profile (`drugsComTest_raw.csv`)

The raw review dataset is tracked via DVC at `Data/reviews/drugsComTest_raw.csv.dvc` and maintained as an immutable raw artifact:

| Metric | Empirical Value | Description / Findings |
|---|---|---|
| **Row Count** | **53,766** | Total patient-reported review records |
| **Column Count** | **7** | `uniqueID`, `drugName`, `condition`, `review`, `rating`, `date`, `usefulCount` |
| **File Size** | **27.6 MB** (27,637,580 bytes) | Stored under canonical `Data/reviews/` root |
| **Unique Drugs** | **2,637** | Top drugs: Levonorgestrel (1,273), Etonogestrel (1,085), Nexplanon (736) |
| **Unique Conditions** | **708** | Top conditions: Birth Control (9,648), Depression (3,095), Pain (2,100) |
| **Missing Values** | `condition`: **295** (0.55%) | All other 6 columns have 0 missing values |
| **Duplicate Rows** | **0** | No exact 7-column duplicate records |
| **Duplicate uniqueID** | **0** | `uniqueID` is strictly unique (0 to 232,291) |
| **Duplicate Reviews** | **5,486** | Identical review text shared across multiple records |
| **HTML Entity Rate** | **35,219** (65.5%) | Raw text contains HTML escape sequences (`&#039;`, `&amp;`, `&quot;`) |
| **Rating Range** | **1 to 10** | Integer user rating (Mean: 7.00, Median: 8.0) |
| **Rating Distribution** | Positive (8–10): **60.2%**<br>Negative (1–3): **22.0%**<br>Neutral (4–7): **17.8%** | Severe bimodal distribution reflecting user review self-selection bias |
| **Date Span** | **2008-02-25 → 2017-12-12** | ~10-year span (format: `%d-%b-%y`) |

---

## 3. Schema & Data Quality Validation (`src/nlp_validation.py`)

Schema and data quality validation is implemented using **Pandera** (`DataFrameSchema`):

```python
from src.nlp_validation import validate_review_dataset, validate_review_schema

# Load and validate raw dataset
df = validate_review_dataset("Data/reviews/drugsComTest_raw.csv")
```

### 3.1 Validation Rules & Constraints
- **`uniqueID`**: Integer, non-null, strictly unique across the dataset.
- **`drugName`**: String, non-empty, non-null.
- **`condition`**: String, nullable. The 295 missing values (0.55%) are explicitly permitted per contract.
- **`review`**: String, non-empty, non-null.
- **`rating`**: Integer, bounded strictly within $[1, 10]$.
- **`date`**: Non-null string parseable via `%d-%b-%y`.
- **`usefulCount`**: Integer, $\ge 0$.

### 3.2 Missing Condition Strategy
- In the raw dataset, missing conditions represent unrecorded patient indications.
- **Validation Stage**: Treated as a `WARNING` (not an `ERROR`), confirming missingness is below the 5% threshold (`max_missing_condition_rate: 0.05`).
- **Downstream Preprocessing**: The raw condition is preserved in `condition_raw`, while `condition_clean` fills missing values with the neutral placeholder `"Unknown"`. Condition inference from text is strictly prohibited to avoid fabricating diagnostic labels.

---

## 4. NLP Preprocessing Pipeline (`src/nlp_preprocessing.py`)

Preprocessing transforms raw text into a clean, reproducible dataset without aggressive linguistic reduction:

```
Raw Review Text
      │
      ▼
HTML Entity Decoding (html.unescape)
      │
      ▼
Surrounding Quote Removal
      │
      ▼
Control Character Cleanup ([\x00-\x08\x0B\x0C\x0E-\x1F\x7F])
      │
      ▼
Whitespace Normalization (\r\n, \t, multiple spaces -> single space)
      │
      ▼
Linguistic Preservation (keep negations, medical terms, drug names)
      │
      ▼
Metadata & Feature Extraction (char_count, word_count, sentence_count, clean_hash)
      │
      ▼
Processed Dataset (Data/reviews/processed/reviews_cleaned.parquet)
```

### 4.1 Linguistic Preservation Principles
- **Negations Preserved**: Words such as `not`, `no`, `never`, `none`, `without` are critical for medical sentiment and are never removed.
- **Medical Terminology Preserved**: Symptoms (e.g., `nausea`, `headache`, `insomnia`), conditions (`depression`, `bipolar`), and drug names are preserved intact.
- **Punctuation & Case Preserved**: Standard punctuation and capitalization are retained for sentence boundary detection and future sentiment modeling.

### 4.2 Derived Metadata Features
- **`review_char_count`**: Total character length of cleaned review.
- **`review_word_count`**: Word count delimited by whitespace.
- **`review_sentence_count`**: Estimated sentence count based on terminal punctuation (`.!?`).
- **`review_clean_hash`**: SHA-256 fingerprint of normalized lowercase review text for grouping identical reviews.

---

## 5. Data Leakage Defense & Splitting Strategies

Because **5,486 reviews share identical text** across different records (e.g., cross-posted reviews across brand/generic drugs or boilerplate feedback), naive random train/test splits would result in significant data leakage and artificially inflated evaluation metrics.

`src/nlp_preprocessing.py` implements two leakage-safe splitting strategies:

### 5.1 Strategy A: Stratified-Group Split (`stratified_group`)
- **Group Definition**: Reviews are grouped by `review_clean_hash`. All rows sharing the same review text are assigned as an indivisible unit to either train or test.
- **Stratification**: Uses `StratifiedGroupKFold` across binned rating categories (Low: 1–4, Neutral: 5–7, High: 8–10) with automatic fallback to `GroupKFold` on edge cases.
- **Leakage Guarantee**:
  $$\text{Train } \cap \text{ Test } = \emptyset \quad (\text{zero identical clean review texts in both splits})$$

### 5.2 Strategy B: Temporal Split (`temporal`)
- **Chronological Boundary**: Splits records by posting date (e.g., Train $\le$ 2016-12-31, Test $\ge$ 2017-01-01).
- **Leakage Purge**: Any test review whose text appeared during the training period is automatically purged from the test set, preventing temporal lookahead leakage.
- **Leakage Guarantee**:
  $$\text{Train } \cap \text{ Test } = \emptyset \quad (\text{zero duplicate texts from historical period})$$

---

## 6. DVC & Docker Architecture Integration

### 6.1 DVC Pipeline Stages (`dvc.yaml`)
Review data foundation stages are fully integrated into DVC while maintaining complete isolation from the structured classification pipeline:

```yaml
  validate_reviews:
    cmd: python -c "from src.nlp_validation import validate_review_dataset; validate_review_dataset('Data/reviews/drugsComTest_raw.csv')"
    deps:
      - Data/reviews/drugsComTest_raw.csv
      - configs/model_config.yaml
      - src/nlp_validation.py

  preprocess_reviews:
    cmd: python -c "from src.nlp_preprocessing import run_review_preprocessing; run_review_preprocessing('Data/reviews/drugsComTest_raw.csv')"
    deps:
      - Data/reviews/drugsComTest_raw.csv
      - configs/model_config.yaml
      - src/nlp_preprocessing.py
    outs:
      - Data/reviews/processed/reviews_cleaned.parquet:
          cache: false
```

**Isolation Guarantee**: `validate_reviews` and `preprocess_reviews` do not depend on `Data/drug.csv`, `Model/drug_pipeline.skops`, or `train.py`. Running review stages will never trigger retraining of the structured XGBoost champion.

### 6.2 Docker Context Hardening (`.dockerignore`)
The raw review CSV (27.6 MB) and processed review artifacts are strictly excluded from the production serving container context in `.dockerignore`:
```dockerignore
# Multi-Modal Review Data (excluded from production serving container)
Data/reviews/
Data/drug_reviews/
```
The structured dataset (`Data/drug.csv`, 5.8 KB) remains included to support continuous drift monitoring and serving metadata.

---

## 7. Future NLP Roadmap (Sprints 8–11)

| Sprint | Focus Area | Planned Scope |
|---|---|---|
| **Sprint 8** | **NLP Model Competition** | TF-IDF feature extraction, Logistic Regression, Linear SVM, Naive Bayes baselines, MLflow experiment tracking, NLP quality gate. |
| **Sprint 9** | **Review Experience Intelligence** | Multi-class sentiment classification, aspect extraction, drug-condition satisfaction analytics, semantic similarity. |
| **Sprint 10** | **NLP Monitoring & Text Drift** | Text-length drift, vocabulary drift, sentiment distribution shift, embedding/prediction drift detection. |
| **Sprint 11** | **NLP Retraining & Lifecycle** | Drift certification, automated challenger training, quality gates, atomic promotion, rollback for NLP models. |

---

## 8. Medical Safety Boundary

> ⚠️ **Important Medical Disclaimer:**  
> MedMLOps-X Drug Review Intelligence is an educational MLOps demonstration platform analyzing patient-reported text. It is NOT a certified clinical decision support system, diagnostic tool, or medical prescribing guide. Patient reviews reflect subjective, unverified individual experiences and self-selection bias. Never interpret patient sentiment as medical advice, clinical trial evidence, or pharmacological efficacy.
