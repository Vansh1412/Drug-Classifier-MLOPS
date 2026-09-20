"""
src/nlp_monitoring.py — NLP Monitoring & Drift Detection for MedMLOps-X.

Provides continuous monitoring and drift detection for the Drug Review Intelligence
workload (53k+ patient-reported drug reviews):
1. Schema & Data Quality Monitoring: Missing columns, null rates, invalid ratings/dates.
2. Review Length Drift: Character, word, and sentence count distributions (KS, Wasserstein, PSI).
3. Vocabulary Drift: Token overlap, Jaccard similarity, new/removed token rates.
4. TF-IDF Feature Distribution Drift: Fixed reference vectorizer, transform-only on current batch.
5. Sentiment Distribution Drift: 3-class distribution tracking (PSI, Chi-Square).
6. Rating Distribution Drift: 1–10 numeric ratings, quantiles, statistical tests.
7. Aspect Drift: Clinical domain mention frequencies across 8 predefined aspects.
8. Category Drift: Drug name and condition distributions, explicit 'Unknown' handling.
9. Prediction Drift: Supports NOT_AVAILABLE fallback (Sprint 8 produced no accepted NLP champion).
10. Overall NLP Drift Certification: Deterministic PASS / WARNING / DRIFT / ERROR certification.
11. Secure Report Persistence: Path traversal prevention, zero raw review text exposure.

Medical Safety Boundary:
    Statistical drift in patient-reported review data indicates a change in the observed
    data distribution. It does NOT establish a change in drug safety, clinical efficacy,
    adverse events, diagnosis, or prescribing recommendations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import load_config
from src.nlp_competition import derive_sentiment_target
from src.nlp_preprocessing import clean_text, extract_length_features
from src.nlp_tracking import get_dvc_dataset_hash, get_git_commit_sha

logger = logging.getLogger("src.nlp_monitoring")

MONITORING_DISCLAIMER = (
    "MLOps statistical monitoring signals only. Not clinical or diagnostic evaluation. "
    "Drift in review sentiment, vocabulary, aspects, or ratings indicates a change in the observed "
    "data distribution; it does not establish clinical efficacy, drug safety, adverse events, or "
    "prescribing guidance without verified clinical outcomes."
)

_TOKEN_REGEX = re.compile(r"\b[a-zA-Z]{2,}\b")


# =============================================================================
# Core Statistical Helper Functions
# =============================================================================


def calculate_ks(
    reference: np.ndarray | pd.Series, current: np.ndarray | pd.Series
) -> dict[str, float]:
    """Calculate Two-Sample Kolmogorov-Smirnov test between reference and current samples.

    Args:
        reference: Reference numerical distribution.
        current: Current numerical distribution.

    Returns:
        Dictionary with 'statistic' (KS distance) and 'p_value'.
    """
    ref_clean = pd.Series(reference).dropna().to_numpy()
    curr_clean = pd.Series(current).dropna().to_numpy()

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    res = stats.ks_2samp(ref_clean, curr_clean)
    return {
        "statistic": round(float(res.statistic), 4),
        "p_value": round(float(res.pvalue), 6),
    }


def calculate_wasserstein(
    reference: np.ndarray | pd.Series, current: np.ndarray | pd.Series
) -> float:
    """Calculate 1D Wasserstein Distance normalized by reference standard deviation.

    Args:
        reference: Reference numerical distribution.
        current: Current numerical distribution.

    Returns:
        Normalized Wasserstein distance (float >= 0.0).
    """
    ref_clean = pd.Series(reference).dropna().to_numpy()
    curr_clean = pd.Series(current).dropna().to_numpy()

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    raw_dist = float(stats.wasserstein_distance(ref_clean, curr_clean))
    ref_std = float(np.std(ref_clean))
    norm_dist = raw_dist / ref_std if ref_std > 1e-6 else raw_dist
    return round(norm_dist, 4)


def calculate_psi(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    is_categorical: bool = True,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current distributions.

    PSI Formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))

    Args:
        reference: Reference baseline series.
        current: Current serving series.
        is_categorical: Whether data is categorical.
        num_buckets: Quantile bucket count for numerical features.
        epsilon: Small smoothing constant to avoid log(0) or division by zero.

    Returns:
        Population Stability Index (float >= 0.0).
    """
    ref_s = pd.Series(reference).dropna()
    curr_s = pd.Series(current).dropna()

    if len(ref_s) == 0 or len(curr_s) == 0:
        return 0.0

    if is_categorical:
        categories = sorted(set(ref_s.unique()).union(set(curr_s.unique())))
        ref_counts = ref_s.value_counts(normalize=True)
        curr_counts = curr_s.value_counts(normalize=True)

        psi_val = 0.0
        for cat in categories:
            e = ref_counts.get(cat, 0.0) + epsilon
            a = curr_counts.get(cat, 0.0) + epsilon
            psi_val += (a - e) * math.log(a / e)
        return round(float(psi_val), 4)

    try:
        quantiles = np.linspace(0, 1, num_buckets + 1)
        bins = np.percentile(ref_s, quantiles * 100)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_binned = pd.cut(ref_s, bins=bins)
        curr_binned = pd.cut(curr_s, bins=bins)

        ref_dist = ref_binned.value_counts(normalize=True).sort_index()
        curr_dist = curr_binned.value_counts(normalize=True).sort_index()

        psi_val = 0.0
        for b in ref_dist.index:
            e = ref_dist.get(b, 0.0) + epsilon
            a = curr_dist.get(b, 0.0) + epsilon
            psi_val += (a - e) * math.log(a / e)
        return round(float(psi_val), 4)
    except Exception as exc:
        logger.warning(f"Failed to compute numerical PSI: {exc}")
        return 0.0


def calculate_numerical_summary(s: pd.Series) -> dict[str, float]:
    """Return descriptive summary statistics for a numerical series."""
    clean_s = s.dropna()
    if len(clean_s) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": round(float(clean_s.mean()), 4),
        "std": round(float(clean_s.std(ddof=1)), 4) if len(clean_s) > 1 else 0.0,
        "min": round(float(clean_s.min()), 4),
        "max": round(float(clean_s.max()), 4),
        "median": round(float(clean_s.median()), 4),
    }


def calculate_categorical_distribution(s: pd.Series) -> dict[str, float]:
    """Return category percentages rounded to 4 decimals."""
    clean_s = s.dropna()
    if len(clean_s) == 0:
        return {}
    vc = clean_s.value_counts(normalize=True)
    return {str(k): round(float(v), 4) for k, v in vc.items()}


def tokenize_text(text: str) -> list[str]:
    """Deterministic tokenization for vocabulary drift detection."""
    if not isinstance(text, str):
        return []
    return [m.lower() for m in _TOKEN_REGEX.findall(text)]


# =============================================================================
# NLPMonitor Class
# =============================================================================


class NLPMonitor:
    """Production-grade NLP monitoring & drift detection service for MedMLOps-X."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        reference_df: pd.DataFrame | None = None,
    ) -> None:
        self.config = config or load_config()
        self.nlp_mon_cfg = self.config.get("nlp_monitoring", {})
        self.rev_cfg = self.config.get("reviews", {})
        self.val_cfg = self.rev_cfg.get("validation", {})
        self.aspects_cfg = self.rev_cfg.get("patient_experience", {}).get("aspects", {})

        # 1. Load reference baseline dataset
        ref_path_str = self.nlp_mon_cfg.get(
            "reference_dataset", "Data/reviews/processed/reviews_cleaned.parquet"
        )
        self.ref_path = Path(ref_path_str)

        if reference_df is not None:
            self.reference_df = reference_df.copy()
        else:
            if not self.ref_path.exists():
                raise FileNotFoundError(
                    f"NLP Reference dataset not found at: {self.ref_path.resolve()}"
                )
            if self.ref_path.suffix == ".parquet":
                self.reference_df = pd.read_parquet(self.ref_path)
            else:
                self.reference_df = pd.read_csv(self.ref_path)

        # 2. Ensure length features and sentiment target exist on reference
        self._prepare_reference_dataset()

        # 3. Reports directory
        reports_dir_str = self.nlp_mon_cfg.get("reports_dir", "monitoring/nlp_reports")
        self.reports_dir = Path(reports_dir_str)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 4. Fingerprint reference snapshot
        self.reference_snapshot = self._compute_reference_snapshot()

        # 5. Precompute reference profiles & fixed TF-IDF vectorizer
        self._build_reference_profiles()

    def _prepare_reference_dataset(self) -> None:
        """Ensure reference DataFrame contains all required derived columns."""
        if "review_clean" not in self.reference_df.columns:
            if "review" in self.reference_df.columns:
                self.reference_df["review_clean"] = self.reference_df["review"].apply(
                    clean_text
                )
            else:
                self.reference_df["review_clean"] = ""

        if "review_char_count" not in self.reference_df.columns:
            self.reference_df = extract_length_features(
                self.reference_df, text_col="review_clean"
            )

        if "sentiment_class" not in self.reference_df.columns:
            if "rating" in self.reference_df.columns:
                try:
                    self.reference_df = derive_sentiment_target(self.reference_df)
                except Exception as exc:
                    logger.warning(
                        f"Could not derive sentiment target on reference: {exc}"
                    )

    def _compute_reference_snapshot(self) -> dict[str, Any]:
        """Generate deterministic fingerprint metadata for reference baseline."""
        file_hash = "unknown"
        if self.ref_path.exists():
            try:
                hasher = hashlib.sha256()
                with open(self.ref_path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(65536), b""):
                        hasher.update(chunk)
                file_hash = hasher.hexdigest()
            except Exception as exc:
                logger.warning(f"Failed to compute reference file hash: {exc}")

        git_sha = "unknown"
        dvc_hash = "unknown"
        try:
            git_sha = get_git_commit_sha()
            dvc_hash = get_dvc_dataset_hash("Data/reviews/drugsComTest_raw.csv.dvc")
        except Exception:
            pass

        return {
            "dataset_path": str(self.ref_path),
            "rows": len(self.reference_df),
            "file_sha256": file_hash,
            "dvc_hash": dvc_hash,
            "git_commit": git_sha,
            "created_timestamp": datetime.now(timezone.utc).isoformat(),
            "config_version": self.config.get("api", {}).get("version", "1.1.0"),
        }

    def _build_reference_profiles(self) -> None:
        """Precompute reference baseline profiles for length, vocabulary, and TF-IDF."""
        # 1. Length summaries
        self.ref_lengths = {}
        for col in [
            "review_char_count",
            "review_word_count",
            "review_sentence_count",
        ]:
            if col in self.reference_df.columns:
                self.ref_lengths[col] = calculate_numerical_summary(
                    self.reference_df[col]
                )

        # 2. Vocabulary profile
        ref_tokens = []
        for text in self.reference_df["review_clean"].dropna():
            ref_tokens.extend(tokenize_text(text))
        self.ref_token_series = pd.Series(ref_tokens)
        self.ref_vocab = set(self.ref_token_series.unique())
        self.ref_vocab_size = len(self.ref_vocab)
        self.ref_top_tokens = (
            self.ref_token_series.value_counts(normalize=True).head(50).to_dict()
        )

        # 3. Fixed TF-IDF Vectorizer (fit ONCE on reference)
        tfidf_cfg = self.nlp_mon_cfg.get("tfidf_drift", {})
        max_feat = tfidf_cfg.get("max_features", 5000)
        self.tfidf_top_n = tfidf_cfg.get("top_n_features", 500)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_feat,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        ref_clean_texts = self.reference_df["review_clean"].fillna("").astype(str)
        self.ref_tfidf_matrix = self.tfidf_vectorizer.fit_transform(ref_clean_texts)
        # Precompute reference mean feature vector
        ref_mean_arr = np.asarray(self.ref_tfidf_matrix.mean(axis=0)).ravel()
        self.ref_tfidf_mean = ref_mean_arr
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()

        # 4. Sentiment profile
        self.ref_sentiment_dist = {}
        if "sentiment_class" in self.reference_df.columns:
            self.ref_sentiment_dist = calculate_categorical_distribution(
                self.reference_df["sentiment_class"]
            )

        # 5. Rating profile
        self.ref_rating_summary = {}
        self.ref_rating_dist = {}
        if "rating" in self.reference_df.columns:
            self.ref_rating_summary = calculate_numerical_summary(
                self.reference_df["rating"]
            )
            self.ref_rating_dist = calculate_categorical_distribution(
                self.reference_df["rating"]
            )

        # 6. Aspect profile
        self.ref_aspect_rates = {}
        texts_lower = self.reference_df["review_clean"].str.lower()
        ref_total = len(self.reference_df)
        for aspect_name, kws in self.aspects_cfg.items():
            if kws:
                escaped = [re.escape(k.lower()) for k in kws]
                pattern = r"\b(?:" + "|".join(escaped) + r")\b"
                matches = texts_lower.str.contains(pattern, regex=True, na=False).sum()
                pct = (
                    round(float(matches / ref_total) * 100, 2) if ref_total > 0 else 0.0
                )
                self.ref_aspect_rates[aspect_name] = {
                    "count": int(matches),
                    "percentage": pct,
                }

        # 7. Category profiles
        self.ref_drugs_dist = calculate_categorical_distribution(
            self.reference_df["drugName"]
        )
        self.ref_conditions_dist = calculate_categorical_distribution(
            self.reference_df["condition"].fillna("Unknown")
        )

    # =========================================================================
    # S10-T3: Schema & Data Quality Monitoring
    # =========================================================================

    def validate_batch_quality(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate schema compliance and data quality of incoming review batch.

        Monitors:
        - Missing and unexpected columns
        - Null rates per column
        - Empty review text rate
        - Rating range violations (1–10)
        - Invalid date formats
        - Condition and drugName validity
        """
        q_cfg = self.nlp_mon_cfg.get("quality", {})
        expected_cols = q_cfg.get(
            "expected_columns",
            [
                "uniqueID",
                "drugName",
                "condition",
                "review",
                "rating",
                "date",
                "usefulCount",
            ],
        )
        max_null_rate = float(q_cfg.get("max_null_rate", 0.05))
        max_empty_rate = float(q_cfg.get("max_empty_review_rate", 0.0))
        max_invalid_rating_rate = float(q_cfg.get("max_invalid_rating_rate", 0.0))

        total_rows = len(current_df)
        if total_rows == 0:
            return {
                "status": "ERROR",
                "total_rows": 0,
                "issues": ["Batch is empty (0 records)."],
            }

        issues = []
        status = "PASS"

        # 1. Schema columns
        current_cols = list(current_df.columns)
        missing_cols = [c for c in expected_cols if c not in current_cols]
        unexpected_cols = [
            c
            for c in current_cols
            if c not in expected_cols
            and c
            not in [
                "review_clean",
                "review_raw",
                "review_clean_hash",
                "condition_raw",
                "condition_clean",
                "drugName_clean",
                "date_parsed",
                "year",
                "review_char_count",
                "review_word_count",
                "review_sentence_count",
                "sentiment_class",
                "prediction",
            ]
        ]

        if missing_cols:
            status = "ERROR"
            issues.append(f"Missing required columns: {missing_cols}")

        if unexpected_cols:
            issues.append(f"Unexpected extra columns: {unexpected_cols}")
            if status != "ERROR":
                status = "WARNING"

        # 2. Null rates
        null_rates = {
            col: round(float(current_df[col].isnull().sum() / total_rows), 4)
            for col in current_df.columns
        }
        # Note: condition is allowed to be missing/Unknown up to max_null_rate or placeholder
        for col, n_rate in null_rates.items():
            if col == "condition":
                continue
            if n_rate > max_null_rate:
                issues.append(
                    f"Null rate for '{col}' is {n_rate} (exceeds threshold {max_null_rate})"
                )
                if status != "ERROR":
                    status = "WARNING"

        # 3. Empty review rate
        empty_reviews = 0
        review_col = (
            "review_clean" if "review_clean" in current_df.columns else "review"
        )
        if review_col in current_df.columns:
            empty_reviews = (
                current_df[review_col].fillna("").astype(str).str.strip() == ""
            ).sum()
        empty_rate = round(float(empty_reviews / total_rows), 4)
        if empty_rate > max_empty_rate:
            issues.append(
                f"Empty review rate is {empty_rate} (exceeds threshold {max_empty_rate})"
            )
            if status != "ERROR":
                status = "DRIFT" if empty_rate > 0.05 else "WARNING"

        # 4. Rating range checks (1 to 10)
        invalid_ratings = 0
        if "rating" in current_df.columns and pd.api.types.is_numeric_dtype(
            current_df["rating"]
        ):
            invalid_ratings = (
                (current_df["rating"] < 1) | (current_df["rating"] > 10)
            ).sum()
            invalid_rating_rate = round(float(invalid_ratings / total_rows), 4)
            if invalid_rating_rate > max_invalid_rating_rate:
                issues.append(
                    f"Invalid ratings count: {invalid_ratings} ({invalid_rating_rate * 100}%)"
                )
                status = "ERROR"
        elif "rating" in current_df.columns:
            status = "ERROR"
            issues.append("Rating column is not numeric.")

        # 5. Invalid date rate
        invalid_dates = 0
        if "date" in current_df.columns:
            # Fast parsing trying d-b-y first, then Y-m-d, then fallback
            parsed = pd.to_datetime(
                current_df["date"], format="%d-%b-%y", errors="coerce"
            )
            if parsed.isnull().any():
                parsed = parsed.fillna(
                    pd.to_datetime(
                        current_df["date"], format="%Y-%m-%d", errors="coerce"
                    )
                )
            if parsed.isnull().any():
                parsed = parsed.fillna(
                    pd.to_datetime(current_df["date"], errors="coerce")
                )
            invalid_dates = parsed.isnull().sum()
            invalid_date_rate = round(float(invalid_dates / total_rows), 4)
            if invalid_date_rate > 0.0:
                issues.append(
                    f"Invalid date strings count: {invalid_dates} ({invalid_date_rate * 100}%)"
                )
                if status != "ERROR":
                    status = "WARNING"

        return {
            "status": status,
            "total_rows": total_rows,
            "missing_columns": missing_cols,
            "unexpected_columns": unexpected_cols,
            "null_rates": null_rates,
            "empty_review_count": int(empty_reviews),
            "empty_review_rate": empty_rate,
            "invalid_ratings_count": int(invalid_ratings),
            "invalid_dates_count": int(invalid_dates),
            "issues": issues,
        }

    # =========================================================================
    # S10-T4: Review Length Drift
    # =========================================================================

    def compute_length_drift(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate review length distribution drift across char, word, and sentence counts."""
        cfg = self.nlp_mon_cfg.get("length_drift", {})
        p_warn = cfg.get("ks_p_value_warning", 0.05)
        p_drift = cfg.get("ks_p_value_drift", 0.01)
        w_warn = cfg.get("wasserstein_warning", 0.15)
        w_drift = cfg.get("wasserstein_drift", 0.30)
        psi_warn = cfg.get("psi_warning", 0.10)
        psi_drift = cfg.get("psi_drift", 0.20)

        # Ensure length features are computed on current
        work_df = current_df.copy()
        text_col = "review_clean" if "review_clean" in work_df.columns else "review"
        if "review_char_count" not in work_df.columns and text_col in work_df.columns:
            work_df = extract_length_features(work_df, text_col=text_col)

        features = [
            "review_char_count",
            "review_word_count",
            "review_sentence_count",
        ]
        results = {}
        drift_count = 0
        warning_count = 0

        for feat in features:
            if (
                feat in work_df.columns
                and feat in self.reference_df.columns
                and len(work_df[feat].dropna()) > 0
            ):
                ref_s = self.reference_df[feat]
                curr_s = work_df[feat]

                ks_res = calculate_ks(ref_s, curr_s)
                w_dist = calculate_wasserstein(ref_s, curr_s)
                psi_val = calculate_psi(ref_s, curr_s, is_categorical=False)

                p_val = ks_res["p_value"]
                status = "PASS"
                if p_val < p_drift or w_dist >= w_drift or psi_val >= psi_drift:
                    status = "DRIFT"
                    drift_count += 1
                elif p_val < p_warn or w_dist >= w_warn or psi_val >= psi_warn:
                    status = "WARNING"
                    warning_count += 1

                results[feat] = {
                    "status": status,
                    "ks_statistic": ks_res["statistic"],
                    "ks_p_value": ks_res["p_value"],
                    "wasserstein_distance": w_dist,
                    "psi": psi_val,
                    "reference_summary": self.ref_lengths.get(feat, {}),
                    "current_summary": calculate_numerical_summary(curr_s),
                }

        overall = "PASS"
        if drift_count > 0:
            overall = "DRIFT"
        elif warning_count > 0:
            overall = "WARNING"

        return {
            "status": overall,
            "drift_detected": drift_count > 0,
            "features": results,
        }

    # =========================================================================
    # S10-T5: Vocabulary Drift
    # =========================================================================

    def compute_vocabulary_drift(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate vocabulary drift between reference and current review text."""
        cfg = self.nlp_mon_cfg.get("vocabulary_drift", {})
        new_token_warn = cfg.get("new_token_rate_warning", 0.15)
        new_token_drift = cfg.get("new_token_rate_drift", 0.30)
        top_n = cfg.get("top_n_tokens", 50)

        text_col = "review_clean" if "review_clean" in current_df.columns else "review"
        if text_col not in current_df.columns or len(current_df) == 0:
            return {
                "status": "PASS",
                "reason": "No text column available for vocabulary drift.",
            }

        curr_tokens = []
        for t in current_df[text_col].dropna():
            curr_tokens.extend(tokenize_text(t))

        if not curr_tokens:
            return {"status": "WARNING", "reason": "No valid tokens found."}

        curr_token_series = pd.Series(curr_tokens)
        curr_vocab = set(curr_token_series.unique())
        curr_vocab_size = len(curr_vocab)

        intersection = self.ref_vocab.intersection(curr_vocab)
        union = self.ref_vocab.union(curr_vocab)
        new_tokens = curr_vocab - self.ref_vocab
        removed_tokens = self.ref_vocab - curr_vocab

        jaccard = round(float(len(intersection) / len(union)), 4) if union else 1.0
        overlap_rate = (
            round(float(len(intersection) / curr_vocab_size), 4)
            if curr_vocab_size
            else 0.0
        )
        new_token_rate = (
            round(float(len(new_tokens) / curr_vocab_size), 4)
            if curr_vocab_size
            else 0.0
        )
        removed_token_rate = (
            round(float(len(removed_tokens) / self.ref_vocab_size), 4)
            if self.ref_vocab_size
            else 0.0
        )

        top_tokens_set = set(list(self.ref_top_tokens.keys())[:top_n])
        missing_top_tokens = len(top_tokens_set - curr_vocab)
        missing_top_rate = (
            round(float(missing_top_tokens / len(top_tokens_set)), 4)
            if top_tokens_set
            else 0.0
        )

        status = "PASS"
        if new_token_rate >= new_token_drift or missing_top_rate >= 0.50:
            status = "DRIFT"
        elif new_token_rate >= new_token_warn or missing_top_rate >= 0.25:
            status = "WARNING"

        # Top tokens frequency shift
        curr_top_tokens = (
            curr_token_series.value_counts(normalize=True).head(top_n).to_dict()
        )
        top_shifts = {}
        for tok, ref_pct in list(self.ref_top_tokens.items())[:15]:
            curr_pct = curr_top_tokens.get(tok, 0.0)
            top_shifts[tok] = {
                "reference_freq": round(float(ref_pct), 4),
                "current_freq": round(float(curr_pct), 4),
                "shift": round(float(curr_pct - ref_pct), 4),
            }

        return {
            "status": status,
            "drift_detected": status == "DRIFT",
            "jaccard_similarity": jaccard,
            "vocabulary_overlap_rate": overlap_rate,
            "reference_vocab_size": self.ref_vocab_size,
            "current_vocab_size": curr_vocab_size,
            "new_token_count": len(new_tokens),
            "new_token_rate": new_token_rate,
            "removed_token_count": len(removed_tokens),
            "removed_token_rate": removed_token_rate,
            "top_token_shifts": top_shifts,
            "medical_disclaimer": "Vocabulary changes reflect lexical variations in review texts and do not establish new clinical events or safety signals.",
        }

    # =========================================================================
    # S10-T6: TF-IDF Feature Distribution Drift
    # =========================================================================

    def compute_tfidf_drift(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate TF-IDF feature distribution drift using fixed reference vectorizer."""
        cfg = self.nlp_mon_cfg.get("tfidf_drift", {})
        cos_warn = cfg.get("cosine_distance_warning", 0.15)
        cos_drift = cfg.get("cosine_distance_drift", 0.30)
        feat_psi_drift = cfg.get("feature_psi_drift", 0.20)

        text_col = "review_clean" if "review_clean" in current_df.columns else "review"
        if text_col not in current_df.columns or len(current_df) == 0:
            return {"status": "PASS", "reason": "No text column available."}

        # IMPORTANT: transform ONLY, never fit_transform!
        clean_texts = current_df[text_col].fillna("").astype(str)
        curr_tfidf = self.tfidf_vectorizer.transform(clean_texts)

        curr_mean_arr = np.asarray(curr_tfidf.mean(axis=0)).ravel()

        # Cosine distance between mean vectors
        cos_sim = float(
            cosine_similarity(
                self.ref_tfidf_mean.reshape(1, -1),
                curr_mean_arr.reshape(1, -1),
            )[0, 0]
        )
        cos_dist = round(1.0 - cos_sim, 4)

        # Identify top drifted features by absolute difference in mean weight
        diffs = np.abs(self.ref_tfidf_mean - curr_mean_arr)
        top_indices = np.argsort(diffs)[::-1][: self.tfidf_top_n]

        top_drifted_features = []
        high_drift_features = 0
        for idx in top_indices[:10]:
            feat_name = self.feature_names[idx]
            ref_w = round(float(self.ref_tfidf_mean[idx]), 4)
            curr_w = round(float(curr_mean_arr[idx]), 4)
            diff = round(float(curr_w - ref_w), 4)

            # Sample feature distribution PSI
            ref_col = np.asarray(self.ref_tfidf_matrix[:, idx].todense()).ravel()
            curr_col = np.asarray(curr_tfidf[:, idx].todense()).ravel()
            f_psi = calculate_psi(ref_col, curr_col, is_categorical=False)
            if f_psi >= feat_psi_drift:
                high_drift_features += 1

            top_drifted_features.append(
                {
                    "feature": feat_name,
                    "reference_mean_weight": ref_w,
                    "current_mean_weight": curr_w,
                    "weight_difference": diff,
                    "feature_psi": f_psi,
                }
            )

        status = "PASS"
        if cos_dist >= cos_drift or high_drift_features >= 3:
            status = "DRIFT"
        elif cos_dist >= cos_warn or high_drift_features >= 1:
            status = "WARNING"

        return {
            "status": status,
            "drift_detected": status == "DRIFT",
            "cosine_distance": cos_dist,
            "cosine_similarity": round(cos_sim, 4),
            "features_monitored": len(self.feature_names),
            "top_drifted_features": top_drifted_features,
        }

    # =========================================================================
    # S10-T7: Sentiment Distribution Drift
    # =========================================================================

    def compute_sentiment_drift(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate categorical sentiment distribution drift across low, neutral, high."""
        cfg = self.nlp_mon_cfg.get("sentiment_drift", {})
        psi_warn = cfg.get("psi_warning", 0.10)
        psi_drift = cfg.get("psi_drift", 0.20)
        chi2_drift_p = cfg.get("chi2_p_value_drift", 0.01)

        work_df = current_df.copy()
        if "sentiment_class" not in work_df.columns and "rating" in work_df.columns:
            try:
                work_df = derive_sentiment_target(work_df)
            except Exception as exc:
                logger.warning(f"Could not derive sentiment target: {exc}")

        if "sentiment_class" not in work_df.columns or len(work_df) == 0:
            return {
                "status": "PASS",
                "reason": "sentiment_class not available.",
            }

        curr_sentiment_dist = calculate_categorical_distribution(
            work_df["sentiment_class"]
        )
        psi_val = calculate_psi(
            self.reference_df["sentiment_class"],
            work_df["sentiment_class"],
            is_categorical=True,
        )

        # Chi-square test on counts
        chi2_p = 1.0
        try:
            classes = ["low", "neutral", "high"]
            ref_c = [
                int((self.reference_df["sentiment_class"] == c).sum()) for c in classes
            ]
            curr_c = [int((work_df["sentiment_class"] == c).sum()) for c in classes]
            if sum(curr_c) > 0 and all(c > 0 for c in ref_c):
                res = stats.chi2_contingency([ref_c, curr_c])
                chi2_p = round(float(res.pvalue), 6)
        except Exception as exc:
            logger.warning(f"Chi-square calculation error: {exc}")

        status = "PASS"
        if psi_val >= psi_drift or chi2_p < chi2_drift_p:
            status = "DRIFT"
        elif psi_val >= psi_warn:
            status = "WARNING"

        return {
            "status": status,
            "drift_detected": status == "DRIFT",
            "psi": psi_val,
            "chi2_p_value": chi2_p,
            "reference_distribution": self.ref_sentiment_dist,
            "current_distribution": curr_sentiment_dist,
            "description": (
                f"Sentiment PSI={psi_val}; observed distribution shifted from reference."
                if status != "PASS"
                else "Sentiment distribution matches baseline within expected variance."
            ),
        }

    # =========================================================================
    # S10-T8: Rating Distribution Drift
    # =========================================================================

    def compute_rating_drift(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate numerical 1–10 rating distribution drift."""
        cfg = self.nlp_mon_cfg.get("rating_drift", {})
        p_warn = cfg.get("ks_p_value_warning", 0.05)
        p_drift = cfg.get("ks_p_value_drift", 0.01)
        w_warn = cfg.get("wasserstein_warning", 0.15)
        w_drift = cfg.get("wasserstein_drift", 0.30)
        psi_warn = cfg.get("psi_warning", 0.10)
        psi_drift = cfg.get("psi_drift", 0.20)

        if "rating" not in current_df.columns or len(current_df) == 0:
            return {"status": "PASS", "reason": "rating column not available."}

        ref_s = self.reference_df["rating"]
        curr_s = current_df["rating"].dropna()

        ks_res = calculate_ks(ref_s, curr_s)
        w_dist = calculate_wasserstein(ref_s, curr_s)
        psi_val = calculate_psi(ref_s, curr_s, is_categorical=False)

        p_val = ks_res["p_value"]
        status = "PASS"
        if p_val < p_drift or w_dist >= w_drift or psi_val >= psi_drift:
            status = "DRIFT"
        elif p_val < p_warn or w_dist >= w_warn or psi_val >= psi_warn:
            status = "WARNING"

        curr_summary = calculate_numerical_summary(curr_s)
        desc = "Rating distribution matches baseline."
        if status != "PASS":
            ref_m = self.ref_rating_summary.get("mean", 0.0)
            curr_m = curr_summary.get("mean", 0.0)
            if curr_m < ref_m:
                desc = f"Observed rating distribution shifted toward lower ratings (mean {curr_m} vs {ref_m})."
            else:
                desc = f"Observed rating distribution shifted toward higher ratings (mean {curr_m} vs {ref_m})."

        return {
            "status": status,
            "drift_detected": status == "DRIFT",
            "ks_statistic": ks_res["statistic"],
            "ks_p_value": ks_res["p_value"],
            "wasserstein_distance": w_dist,
            "psi": psi_val,
            "reference_summary": self.ref_rating_summary,
            "current_summary": curr_summary,
            "reference_distribution": self.ref_rating_dist,
            "current_distribution": calculate_categorical_distribution(curr_s),
            "description": desc,
        }

    # =========================================================================
    # S10-T9: Aspect Drift
    # =========================================================================

    def compute_aspect_drift(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate clinical aspect mention frequency drift across 8 predefined domains."""
        cfg = self.nlp_mon_cfg.get("aspect_drift", {})
        warn_pct = cfg.get("warning_pct_change", 5.0)
        drift_pct = cfg.get("drift_pct_change", 10.0)

        text_col = "review_clean" if "review_clean" in current_df.columns else "review"
        if text_col not in current_df.columns or len(current_df) == 0:
            return {"status": "PASS", "aspects": {}}

        texts_lower = current_df[text_col].fillna("").astype(str).str.lower()
        curr_total = len(current_df)

        aspect_results = {}
        drift_count = 0
        warning_count = 0

        for aspect_name, kws in self.aspects_cfg.items():
            if not kws:
                continue

            escaped = [re.escape(k.lower()) for k in kws]
            pattern = r"\b(?:" + "|".join(escaped) + r")\b"
            matches = int(texts_lower.str.contains(pattern, regex=True, na=False).sum())
            curr_pct = (
                round(float(matches / curr_total) * 100, 2) if curr_total > 0 else 0.0
            )

            ref_info = self.ref_aspect_rates.get(
                aspect_name, {"count": 0, "percentage": 0.0}
            )
            ref_pct = ref_info["percentage"]

            abs_diff = round(curr_pct - ref_pct, 2)
            rel_diff = round((abs_diff / ref_pct) * 100, 2) if ref_pct > 0 else 0.0

            status = "PASS"
            if abs(abs_diff) >= drift_pct:
                status = "DRIFT"
                drift_count += 1
            elif abs(abs_diff) >= warn_pct:
                status = "WARNING"
                warning_count += 1

            aspect_results[aspect_name] = {
                "status": status,
                "reference_percentage": ref_pct,
                "current_percentage": curr_pct,
                "absolute_change": abs_diff,
                "relative_change_pct": rel_diff,
                "mention_count": matches,
            }

        overall = "PASS"
        if drift_count > 0:
            overall = "DRIFT"
        elif warning_count > 0:
            overall = "WARNING"

        return {
            "status": overall,
            "drift_detected": drift_count > 0,
            "aspects": aspect_results,
            "medical_disclaimer": "Aspect frequency changes describe lexical shifts in review text and do NOT constitute clinical adverse event reports or safety changes.",
        }

    # =========================================================================
    # S10-T10: Drug & Condition Category Drift
    # =========================================================================

    def compute_category_drift(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate categorical distribution drift for drugName and condition."""
        cfg = self.nlp_mon_cfg.get("categorical_drift", {})
        psi_warn = cfg.get("psi_warning", 0.10)
        psi_drift = cfg.get("psi_drift", 0.20)
        new_cat_warn = cfg.get("new_category_rate_warning", 0.05)
        new_cat_drift = cfg.get("new_category_rate_drift", 0.15)

        total_rows = len(current_df)
        if total_rows == 0:
            return {"status": "PASS"}

        res = {}
        overall_drift = False
        overall_warn = False

        for col, ref_dist in [
            ("drugName", self.ref_drugs_dist),
            ("condition", self.ref_conditions_dist),
        ]:
            if col in current_df.columns:
                curr_s = current_df[col].fillna("Unknown").astype(str).str.strip()
                curr_dist = calculate_categorical_distribution(curr_s)

                ref_cats = set(ref_dist.keys())
                curr_cats = set(curr_dist.keys())

                new_cats = list(curr_cats - ref_cats)
                disappeared_cats = list(ref_cats - curr_cats)

                new_cat_rate = (
                    round(float(len(new_cats) / len(curr_cats)), 4)
                    if curr_cats
                    else 0.0
                )

                # For condition, handle 'Unknown' as an explicit missing placeholder
                unknown_rate = 0.0
                if col == "condition":
                    unknown_count = int((curr_s == "Unknown").sum())
                    unknown_rate = round(float(unknown_count / total_rows), 4)
                    # Filter out Unknown for clinical condition PSI
                    clinical_ref_dist = {
                        k: v for k, v in ref_dist.items() if k != "Unknown"
                    }
                    clinical_curr_s = curr_s[curr_s != "Unknown"]
                    clinical_ref_s = self.reference_df[col].dropna()
                    clinical_ref_s = clinical_ref_s[clinical_ref_s != "Unknown"]
                else:
                    clinical_ref_dist = ref_dist
                    clinical_curr_s = curr_s
                    clinical_ref_s = self.reference_df[col].dropna()

                # Group high-cardinality categories into Top N + 'Other' for stable PSI
                top_cats = [
                    cat
                    for cat, _ in sorted(
                        clinical_ref_dist.items(), key=lambda x: x[1], reverse=True
                    )[:15]
                ]

                ref_binned = clinical_ref_s.apply(
                    lambda x: x if x in top_cats else "Other"
                )
                curr_binned = clinical_curr_s.apply(
                    lambda x: x if x in top_cats else "Other"
                )

                cat_psi = 0.0
                if len(ref_binned) > 0 and len(curr_binned) > 0:
                    cat_psi = calculate_psi(
                        ref_binned, curr_binned, is_categorical=True
                    )

                status = "PASS"
                if (
                    cat_psi >= psi_drift
                    or new_cat_rate >= new_cat_drift
                    or unknown_rate >= 0.15
                ):
                    status = "DRIFT"
                    overall_drift = True
                elif (
                    cat_psi >= psi_warn
                    or new_cat_rate >= new_cat_warn
                    or unknown_rate >= 0.05
                ):
                    status = "WARNING"
                    overall_warn = True

                col_res = {
                    "status": status,
                    "psi": cat_psi,
                    "new_categories_count": len(new_cats),
                    "new_category_rate": new_cat_rate,
                    "disappeared_categories_count": len(disappeared_cats),
                    "total_unique_categories": len(curr_cats),
                }
                if col == "condition":
                    col_res["unknown_condition_rate"] = unknown_rate
                res[col] = col_res

        overall_status = "PASS"
        if overall_drift:
            overall_status = "DRIFT"
        elif overall_warn:
            overall_status = "WARNING"

        return {
            "status": overall_status,
            "drift_detected": overall_drift,
            "categories": res,
        }

    # =========================================================================
    # S10-T11: Prediction Drift
    # =========================================================================

    def compute_prediction_drift(
        self,
        current_predictions: pd.Series | np.ndarray | list[str] | None = None,
        reference_predictions: pd.Series | np.ndarray | list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate NLP prediction distribution drift when predictions are supplied.

        Sprint 8 produced NO accepted NLP champion model. Therefore, if predictions
        are not supplied, status is NOT_AVAILABLE.
        """
        if current_predictions is None:
            return {
                "status": "NOT_AVAILABLE",
                "drift_detected": False,
                "reason": "Sprint 8 produced no accepted NLP champion model; prediction stream not supplied.",
            }

        curr_p = pd.Series(current_predictions).dropna()
        if len(curr_p) == 0:
            return {
                "status": "NOT_AVAILABLE",
                "drift_detected": False,
                "reason": "Current predictions series is empty.",
            }

        # Reference predictions fallback to sentiment_class if not explicitly passed
        if reference_predictions is not None:
            ref_p = pd.Series(reference_predictions).dropna()
        elif "sentiment_class" in self.reference_df.columns:
            ref_p = self.reference_df["sentiment_class"].dropna()
        else:
            return {
                "status": "NOT_AVAILABLE",
                "drift_detected": False,
                "reason": "Reference predictions not available.",
            }

        psi_val = calculate_psi(ref_p, curr_p, is_categorical=True)
        pred_cfg = self.nlp_mon_cfg.get("prediction_drift", {})
        psi_warn = pred_cfg.get("psi_warning", 0.10)
        psi_drift = pred_cfg.get("psi_drift", 0.20)

        status = "PASS"
        if psi_val >= psi_drift:
            status = "DRIFT"
        elif psi_val >= psi_warn:
            status = "WARNING"

        return {
            "status": status,
            "drift_detected": status == "DRIFT",
            "psi": psi_val,
            "reference_distribution": calculate_categorical_distribution(ref_p),
            "current_distribution": calculate_categorical_distribution(curr_p),
        }

    # =========================================================================
    # S10-T12 to S10-T14: Overall Certification & Reporting
    # =========================================================================

    def certify_drift(
        self,
        current_df: pd.DataFrame,
        current_predictions: pd.Series | None = None,
        save_report: bool = True,
    ) -> dict[str, Any]:
        """Execute end-to-end NLP monitoring and produce deterministic certification report."""
        timestamp = datetime.now(timezone.utc).isoformat()
        report_id = f"nlp_drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # 1. Schema & Data Quality
        quality_res = self.validate_batch_quality(current_df)

        # 2. Text Length Drift
        length_res = self.compute_length_drift(current_df)

        # 3. Vocabulary Drift
        vocab_res = self.compute_vocabulary_drift(current_df)

        # 4. TF-IDF Drift
        tfidf_res = self.compute_tfidf_drift(current_df)

        # 5. Sentiment Drift
        sentiment_res = self.compute_sentiment_drift(current_df)

        # 6. Rating Drift
        rating_res = self.compute_rating_drift(current_df)

        # 7. Aspect Drift
        aspect_res = self.compute_aspect_drift(current_df)

        # 8. Category Drift
        cat_res = self.compute_category_drift(current_df)

        # 9. Prediction Drift
        pred_res = self.compute_prediction_drift(current_predictions)

        # S10-T12: Overall Certification Logic
        signals = []
        if quality_res["status"] == "ERROR":
            signals.append("critical_data_quality_or_schema_failure")
        if quality_res["status"] == "WARNING":
            signals.append("data_quality_warning")

        if length_res.get("status") == "DRIFT":
            signals.append("review_length_distribution_drift")
        elif length_res.get("status") == "WARNING":
            signals.append("review_length_distribution_warning")

        if vocab_res.get("status") == "DRIFT":
            signals.append("vocabulary_distribution_drift")
        elif vocab_res.get("status") == "WARNING":
            signals.append("vocabulary_distribution_warning")

        if tfidf_res.get("status") == "DRIFT":
            signals.append("tfidf_feature_distribution_drift")
        elif tfidf_res.get("status") == "WARNING":
            signals.append("tfidf_feature_distribution_warning")

        if sentiment_res.get("status") == "DRIFT":
            signals.append("sentiment_distribution_drift")
        elif sentiment_res.get("status") == "WARNING":
            signals.append("sentiment_distribution_warning")

        if rating_res.get("status") == "DRIFT":
            signals.append("rating_distribution_drift")
        elif rating_res.get("status") == "WARNING":
            signals.append("rating_distribution_warning")

        if aspect_res.get("status") == "DRIFT":
            signals.append("aspect_frequency_drift")
        elif aspect_res.get("status") == "WARNING":
            signals.append("aspect_frequency_warning")

        if cat_res.get("status") == "DRIFT":
            signals.append("category_distribution_drift")
        elif cat_res.get("status") == "WARNING":
            signals.append("category_distribution_warning")

        if pred_res.get("status") == "DRIFT":
            signals.append("nlp_prediction_distribution_drift")

        # Determine overall certification status
        if quality_res["status"] == "ERROR":
            overall_status = "ERROR"
            overall_severity = "CRITICAL"
        elif any("drift" in s for s in signals):
            overall_status = "DRIFT"
            overall_severity = "HIGH"
        elif any("warning" in s for s in signals):
            overall_status = "WARNING"
            overall_severity = "MEDIUM"
        else:
            overall_status = "PASS"
            overall_severity = "LOW"

        report: dict[str, Any] = {
            "report_id": report_id,
            "timestamp": timestamp,
            "certification": {
                "overall_status": overall_status,
                "overall_severity": overall_severity,
                "drift_detected": overall_status == "DRIFT",
                "signals": signals,
                "automated_retraining_triggered": False,  # S10 strictly forbids automated retraining
            },
            "reference": self.reference_snapshot,
            "current": {
                "rows": len(current_df),
                "timestamp": timestamp,
            },
            "data_quality": quality_res,
            "text_length_drift": length_res,
            "vocabulary_drift": vocab_res,
            "tfidf_drift": tfidf_res,
            "sentiment_drift": sentiment_res,
            "rating_drift": rating_res,
            "aspect_drift": aspect_res,
            "category_drift": cat_res,
            "prediction_drift": pred_res,
            "disclaimer": MONITORING_DISCLAIMER,
        }

        if save_report:
            self.save_report(report)

        return report

    def save_report(self, report: dict[str, Any]) -> Path:
        """Persist structured report to disk safely with path traversal protection."""
        report_id = report.get("report_id", f"nlp_drift_{uuid.uuid4().hex[:8]}")
        safe_filename = (
            f"{report_id}.json".replace("..", "").replace("/", "").replace("\\", "")
        )
        dest_path = (self.reports_dir / safe_filename).resolve()

        resolved_reports_dir = self.reports_dir.resolve()
        if not str(dest_path).startswith(str(resolved_reports_dir)):
            raise ValueError(f"Path traversal detected: {dest_path}")

        # Also write to canonical Results/nlp_drift_report.json only if it's a full certification report
        if "certification" in report:
            canonical_path = Path(
                self.nlp_mon_cfg.get("report_path", "Results/nlp_drift_report.json")
            )
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            with open(canonical_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)

        return dest_path

    def get_latest_report(self) -> dict[str, Any] | None:
        """Retrieve the most recently generated drift report from disk."""
        canonical_path = Path(
            self.nlp_mon_cfg.get("report_path", "Results/nlp_drift_report.json")
        )
        if canonical_path.exists():
            try:
                with open(canonical_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.warning(f"Could not read canonical report: {exc}")

        if not self.reports_dir.exists():
            return None

        report_files = sorted(
            self.reports_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not report_files:
            return None

        try:
            with open(report_files[0], "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.error(f"Failed to read latest report {report_files[0]}: {exc}")
            return None

    # =========================================================================
    # S10-T18: Synthetic Current Batch Generation for Testing
    # =========================================================================

    def create_synthetic_current_batch(
        self,
        size: int = 500,
        shift_type: str | None = None,
        random_state: int = 125,
    ) -> pd.DataFrame:
        """Create a deterministic synthetic current review batch without modifying raw data.

        Supported shift_types:
        - None: Clean baseline sample (PASS)
        - 'sentiment_shift': Subsamples predominantly negative/low rating reviews (DRIFT)
        - 'length_shift': Subsamples exceptionally long or truncated reviews (DRIFT)
        - 'quality_degradation': Injects nulls, empty strings, invalid ratings (ERROR/DRIFT)
        - 'aspect_shift': Subsamples reviews mentioning specific aspects (e.g. side effects)
        """
        sample_df = self.reference_df.sample(
            n=min(size * 3, len(self.reference_df)),
            random_state=random_state,
        ).copy()

        if shift_type is None:
            return sample_df.head(size).copy()

        if shift_type == "sentiment_shift":
            # Overweight low rating reviews
            low_df = sample_df[sample_df["rating"] <= 4]
            high_df = sample_df[sample_df["rating"] >= 8]
            n_low = int(size * 0.8)
            n_high = size - n_low
            shifted = pd.concat(
                [
                    low_df.sample(
                        n=min(n_low, len(low_df)),
                        replace=True,
                        random_state=random_state,
                    ),
                    high_df.sample(
                        n=min(n_high, len(high_df)),
                        replace=True,
                        random_state=random_state,
                    ),
                ]
            )
            return shifted.sample(frac=1.0, random_state=random_state).reset_index(
                drop=True
            )

        if shift_type == "length_shift":
            # Select extreme short reviews
            short_df = sample_df[sample_df["review_word_count"] < 10]
            if len(short_df) < size:
                short_df = pd.concat([short_df] * (size // len(short_df) + 1))
            return short_df.head(size).reset_index(drop=True)

        if shift_type == "quality_degradation":
            batch = sample_df.head(size).copy()
            # Inject invalid ratings and empty reviews
            batch.loc[batch.index[:10], "rating"] = 99
            batch.loc[batch.index[10:30], "review"] = ""
            batch.loc[batch.index[10:30], "review_clean"] = ""
            return batch

        if shift_type == "aspect_shift":
            # Filter reviews discussing side effects
            side_effect_kws = self.aspects_cfg.get("side_effects", ["nausea"])
            escaped = [re.escape(k.lower()) for k in side_effect_kws]
            pattern = r"\b(?:" + "|".join(escaped) + r")\b"
            matched = sample_df[
                sample_df["review_clean"]
                .str.lower()
                .str.contains(pattern, regex=True, na=False)
            ]
            if len(matched) < size:
                matched = pd.concat([matched] * (size // len(matched) + 1))
            return matched.head(size).reset_index(drop=True)

        return sample_df.head(size).copy()


# Module-level singleton
_nlp_monitor_instance: NLPMonitor | None = None


def get_nlp_monitor() -> NLPMonitor:
    """Retrieve or lazily initialize the global NLPMonitor singleton."""
    global _nlp_monitor_instance
    if _nlp_monitor_instance is None:
        _nlp_monitor_instance = NLPMonitor()
    return _nlp_monitor_instance


def run_nlp_monitoring() -> dict[str, Any]:
    """DVC stage entrypoint: run baseline NLP monitoring and generate report."""
    logger.info("Initializing NLP monitoring execution...")
    monitor = get_nlp_monitor()
    # In DVC baseline stage, certify reference dataset against its baseline profile
    logger.info(f"Certifying NLP baseline with {len(monitor.reference_df)} reviews...")
    report = monitor.certify_drift(monitor.reference_df, save_report=True)
    logger.info(
        f"NLP monitoring complete. Overall Status: {report['certification']['overall_status']}"
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_nlp_monitoring()
