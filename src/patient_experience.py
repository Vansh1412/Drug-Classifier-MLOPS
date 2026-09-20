"""
src/patient_experience.py — Patient Experience Intelligence Layer for MedMLOps-X.

Provides explainable analytics and intelligence over patient-reported drug reviews:
1. Sentiment Analytics: Overall distribution, rating breakdown, and descriptive statistics.
2. Drug-Level Experience: Aggregations for drugs meeting a minimum review threshold.
3. Condition-Level Experience: Aggregations per condition, explicitly handling "Unknown".
4. Temporal Experience: Yearly volume, mean rating, and sentiment shifts over time.
5. Aspect / Theme Extraction: Interpretable keyword + n-gram extraction across clinical domains.
6. Lexical Review Similarity: TF-IDF cosine similarity search for finding similar patient experiences.
7. Machine-Readable Reporting: Produces comprehensive and compact analytics summaries.

Medical Safety Boundary:
    Analytics describe patterns in the patient-reported review dataset and should NOT
    be interpreted as clinical efficacy, safety, diagnosis, or prescribing recommendations.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import load_config
from src.nlp_competition import derive_sentiment_target
from src.nlp_tracking import get_dvc_dataset_hash, get_git_commit_sha

logger = logging.getLogger(__name__)


def analyze_overall_sentiment(df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall sentiment and rating distribution metrics.

    Args:
        df: DataFrame containing 'rating' and 'sentiment_class' columns.

    Returns:
        Dictionary of overall sentiment counts, percentages, and rating statistics.
    """
    total = len(df)
    if total == 0:
        return {
            "total_reviews": 0,
            "sentiment_distribution": {},
            "rating_distribution": {},
            "rating_summary": {},
            "useful_count_summary": {},
        }

    # Sentiment distribution
    sentiment_counts = df["sentiment_class"].value_counts().to_dict()
    sentiment_dist = {}
    for cls in ["high", "low", "neutral"]:
        count = int(sentiment_counts.get(cls, 0))
        pct = round((count / total) * 100, 2)
        sentiment_dist[cls] = {"count": count, "percentage": pct}

    # Rating distribution (1 through 10)
    rating_counts = df["rating"].value_counts().to_dict()
    rating_dist = {}
    for r in range(1, 11):
        count = int(rating_counts.get(r, 0))
        pct = round((count / total) * 100, 2)
        rating_dist[str(r)] = {"count": count, "percentage": pct}

    # Summary statistics for rating
    ratings = df["rating"]
    rating_summary = {
        "mean": round(float(ratings.mean()), 4),
        "median": round(float(ratings.median()), 4),
        "std": round(float(ratings.std(ddof=1)), 4) if len(ratings) > 1 else 0.0,
        "min": int(ratings.min()),
        "max": int(ratings.max()),
    }

    # Summary statistics for usefulCount
    useful = df["usefulCount"] if "usefulCount" in df.columns else pd.Series([0])
    useful_summary = {
        "mean": round(float(useful.mean()), 4),
        "median": round(float(useful.median()), 4),
        "std": round(float(useful.std(ddof=1)), 4) if len(useful) > 1 else 0.0,
        "max": int(useful.max()),
    }

    return {
        "total_reviews": total,
        "sentiment_distribution": sentiment_dist,
        "rating_distribution": rating_dist,
        "rating_summary": rating_summary,
        "useful_count_summary": useful_summary,
    }


def analyze_drug_experience(
    df: pd.DataFrame,
    min_reviews: int = 20,
) -> list[dict[str, Any]]:
    """Compute descriptive statistics per drug for drugs meeting min_reviews.

    Strictly descriptive: does NOT rank drugs as 'best' or 'worst'.

    Args:
        df: DataFrame containing 'drugName', 'rating', 'sentiment_class', 'usefulCount'.
        min_reviews: Minimum number of reviews required for inclusion.

    Returns:
        List of drug experience records sorted by review_count (descending).
    """
    if len(df) == 0:
        return []

    drug_counts = df["drugName"].value_counts()
    eligible_drugs = drug_counts[drug_counts >= min_reviews].index.tolist()

    results: list[dict[str, Any]] = []
    for drug in eligible_drugs:
        sub = df[df["drugName"] == drug]
        n_reviews = len(sub)
        sent_counts = sub["sentiment_class"].value_counts().to_dict()

        low_c = int(sent_counts.get("low", 0))
        neutral_c = int(sent_counts.get("neutral", 0))
        high_c = int(sent_counts.get("high", 0))

        ratings = sub["rating"]
        useful = sub["usefulCount"] if "usefulCount" in sub.columns else pd.Series([0])

        results.append(
            {
                "drugName": drug,
                "review_count": n_reviews,
                "mean_rating": round(float(ratings.mean()), 4),
                "median_rating": round(float(ratings.median()), 4),
                "rating_std": (
                    round(float(ratings.std(ddof=1)), 4) if len(ratings) > 1 else 0.0
                ),
                "low_count": low_c,
                "neutral_count": neutral_c,
                "high_count": high_c,
                "low_pct": round((low_c / n_reviews) * 100, 2),
                "neutral_pct": round((neutral_c / n_reviews) * 100, 2),
                "high_pct": round((high_c / n_reviews) * 100, 2),
                "mean_useful_count": round(float(useful.mean()), 4),
            }
        )

    # Sort deterministically: review_count desc, mean_rating desc, drugName asc
    results.sort(key=lambda x: (-x["review_count"], -x["mean_rating"], x["drugName"]))
    return results


def analyze_condition_experience(
    df: pd.DataFrame,
    min_reviews: int = 10,
) -> list[dict[str, Any]]:
    """Compute descriptive statistics per condition for conditions meeting min_reviews.

    Explicitly labels 'Unknown' conditions as missing/unavailable categories without
    attempting clinical inference.

    Args:
        df: DataFrame containing 'condition' (or 'condition_clean'), 'rating', 'sentiment_class'.
        min_reviews: Minimum number of reviews required for inclusion.

    Returns:
        List of condition experience records sorted by review_count (descending).
    """
    if len(df) == 0:
        return []

    # Prefer condition_clean if present, else normalize condition
    cond_series = (
        df["condition_clean"]
        if "condition_clean" in df.columns
        else df["condition"].fillna("Unknown").astype(str).str.strip()
    )

    work_df = df.copy()
    work_df["_cond"] = cond_series

    cond_counts = work_df["_cond"].value_counts()
    eligible_conds = cond_counts[cond_counts >= min_reviews].index.tolist()

    results: list[dict[str, Any]] = []
    for cond in eligible_conds:
        sub = work_df[work_df["_cond"] == cond]
        n_reviews = len(sub)
        sent_counts = sub["sentiment_class"].value_counts().to_dict()

        low_c = int(sent_counts.get("low", 0))
        neutral_c = int(sent_counts.get("neutral", 0))
        high_c = int(sent_counts.get("high", 0))

        ratings = sub["rating"]

        results.append(
            {
                "condition": cond,
                "is_missing_category": (cond == "Unknown"),
                "review_count": n_reviews,
                "mean_rating": round(float(ratings.mean()), 4),
                "median_rating": round(float(ratings.median()), 4),
                "rating_std": (
                    round(float(ratings.std(ddof=1)), 4) if len(ratings) > 1 else 0.0
                ),
                "low_count": low_c,
                "neutral_count": neutral_c,
                "high_count": high_c,
                "low_pct": round((low_c / n_reviews) * 100, 2),
                "neutral_pct": round((neutral_c / n_reviews) * 100, 2),
                "high_pct": round((high_c / n_reviews) * 100, 2),
            }
        )

    # Sort deterministically: review_count desc, mean_rating desc, condition asc
    results.sort(key=lambda x: (-x["review_count"], -x["mean_rating"], x["condition"]))
    return results


def analyze_temporal_experience(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute yearly volume, mean rating, and sentiment distribution.

    Args:
        df: DataFrame containing date information ('year' or 'date_parsed' or 'date').

    Returns:
        List of yearly experience statistics sorted chronologically.
    """
    if len(df) == 0:
        return []

    work_df = df.copy()
    if "year" not in work_df.columns:
        if "date_parsed" in work_df.columns:
            work_df["year"] = pd.to_datetime(
                work_df["date_parsed"], errors="coerce"
            ).dt.year
        else:
            work_df["year"] = pd.to_datetime(
                work_df["date"], format="%d-%b-%y", errors="coerce"
            ).dt.year

    valid_years_df = work_df.dropna(subset=["year"]).copy()
    valid_years_df["year"] = valid_years_df["year"].astype(int)

    results: list[dict[str, Any]] = []
    for yr in sorted(valid_years_df["year"].unique()):
        sub = valid_years_df[valid_years_df["year"] == yr]
        n_reviews = len(sub)
        sent_counts = sub["sentiment_class"].value_counts().to_dict()

        low_c = int(sent_counts.get("low", 0))
        neutral_c = int(sent_counts.get("neutral", 0))
        high_c = int(sent_counts.get("high", 0))

        ratings = sub["rating"]

        results.append(
            {
                "year": int(yr),
                "review_count": n_reviews,
                "mean_rating": round(float(ratings.mean()), 4),
                "median_rating": round(float(ratings.median()), 4),
                "low_count": low_c,
                "neutral_count": neutral_c,
                "high_count": high_c,
                "low_pct": round((low_c / n_reviews) * 100, 2),
                "neutral_pct": round((neutral_c / n_reviews) * 100, 2),
                "high_pct": round((high_c / n_reviews) * 100, 2),
            }
        )

    return results


def extract_aspect_intelligence(
    df: pd.DataFrame,
    aspect_keywords: dict[str, list[str]],
    max_exemplars: int = 3,
) -> dict[str, Any]:
    """Extract clinical/experiential aspects using transparent keyword dictionaries.

    Computes mention frequency, percentage of total reviews, and conditional
    sentiment breakdown for each aspect. Exemplars are referenced by ID and hash
    to avoid dumping full review text.

    Args:
        df: DataFrame containing 'review_clean', 'rating', 'sentiment_class', 'uniqueID', 'review_clean_hash'.
        aspect_keywords: Dictionary mapping aspect name to list of keywords.
        max_exemplars: Maximum number of exemplar review references to retain per aspect.

    Returns:
        Dictionary of aspect metrics.
    """
    total = len(df)
    if total == 0:
        return {}

    texts_lower = df["review_clean"].astype(str).str.lower()
    aspect_results: dict[str, Any] = {}

    for aspect_name, keywords in aspect_keywords.items():
        if not keywords:
            continue

        # Build regex pattern with word boundaries for precision
        escaped_kws = [re.escape(k.lower()) for k in keywords]
        pattern = r"\b(?:" + "|".join(escaped_kws) + r")\b"
        mask = texts_lower.str.contains(pattern, regex=True, na=False)

        match_count = int(mask.sum())
        match_pct = round((match_count / total) * 100, 2)

        if match_count > 0:
            matched_df = df[mask]
            sent_counts = matched_df["sentiment_class"].value_counts().to_dict()

            low_c = int(sent_counts.get("low", 0))
            neutral_c = int(sent_counts.get("neutral", 0))
            high_c = int(sent_counts.get("high", 0))

            ratings = matched_df["rating"]

            # Select exemplars (highest usefulCount or first records)
            exemplar_sub = matched_df.head(max_exemplars)
            exemplars = []
            for _, row in exemplar_sub.iterrows():
                clean_text = str(row.get("review_clean", ""))
                # Provide a bounded, truncated snippet for context without full text exposure
                snippet = (
                    clean_text[:120] + "..." if len(clean_text) > 120 else clean_text
                )
                exemplars.append(
                    {
                        "uniqueID": int(row.get("uniqueID", 0)),
                        "review_clean_hash": str(row.get("review_clean_hash", "")),
                        "rating": int(row.get("rating", 0)),
                        "sentiment_class": str(row.get("sentiment_class", "")),
                        "snippet": snippet,
                    }
                )

            aspect_results[aspect_name] = {
                "keywords": keywords,
                "mention_count": match_count,
                "mention_percentage": match_pct,
                "mean_rating": round(float(ratings.mean()), 4),
                "median_rating": round(float(ratings.median()), 4),
                "sentiment_distribution": {
                    "low_count": low_c,
                    "neutral_count": neutral_c,
                    "high_count": high_c,
                    "low_pct": round((low_c / match_count) * 100, 2),
                    "neutral_pct": round((neutral_c / match_count) * 100, 2),
                    "high_pct": round((high_c / match_count) * 100, 2),
                },
                "exemplars": exemplars,
            }
        else:
            aspect_results[aspect_name] = {
                "keywords": keywords,
                "mention_count": 0,
                "mention_percentage": 0.0,
                "mean_rating": 0.0,
                "median_rating": 0.0,
                "sentiment_distribution": {
                    "low_count": 0,
                    "neutral_count": 0,
                    "high_count": 0,
                    "low_pct": 0.0,
                    "neutral_pct": 0.0,
                    "high_pct": 0.0,
                },
                "exemplars": [],
            }

    return aspect_results


class ReviewSimilarityEngine:
    """Lexical semantic similarity engine based on TF-IDF and cosine similarity.

    Provides transparent, reproducible similarity search without requiring
    external vector databases or neural embedding frameworks.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tfidf_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize and fit TF-IDF vectorizer over the review corpus.

        Args:
            df: DataFrame containing 'review_clean', 'uniqueID', 'review_clean_hash',
                'drugName', 'condition', 'rating', 'sentiment_class'.
            tfidf_params: Optional dictionary of TF-IDF hyperparameters.
        """
        self.df = df.reset_index(drop=True)
        params = {
            "ngram_range": (1, 2),
            "min_df": 3,
            "max_df": 0.90,
            "max_features": 10000,
            "sublinear_tf": True,
        }
        if tfidf_params:
            params.update(tfidf_params)

        self.vectorizer = TfidfVectorizer(**params)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["review_clean"])
        logger.info(
            f"ReviewSimilarityEngine initialized with {self.tfidf_matrix.shape[0]} documents "
            f"and {self.tfidf_matrix.shape[1]} features."
        )

    def find_similar(
        self,
        query_text: str,
        top_k: int = 5,
        filter_drug: str | None = None,
        filter_condition: str | None = None,
        exclude_self_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Find the top-k most lexically similar reviews for a query string.

        Args:
            query_text: Review text query to search.
            top_k: Number of similar reviews to return.
            filter_drug: Optional drugName to filter results.
            filter_condition: Optional condition to filter results.
            exclude_self_id: Optional uniqueID to exclude from results (e.g. self-match).

        Returns:
            List of matching review records with similarity scores.
        """
        if not query_text or not isinstance(query_text, str) or not query_text.strip():
            return []

        query_vec = self.vectorizer.transform([query_text])

        # If query contains no known terms (all zeros), return empty
        if query_vec.nnz == 0:
            return []

        scores = cosine_similarity(query_vec, self.tfidf_matrix).ravel()

        # Apply candidate filtering if specified
        valid_indices = np.arange(len(self.df))
        if filter_drug is not None:
            valid_indices = valid_indices[
                self.df["drugName"].iloc[valid_indices] == filter_drug
            ]
        if filter_condition is not None:
            cond_col = (
                "condition_clean"
                if "condition_clean" in self.df.columns
                else "condition"
            )
            valid_indices = valid_indices[
                self.df[cond_col].iloc[valid_indices] == filter_condition
            ]
        if exclude_self_id is not None and "uniqueID" in self.df.columns:
            valid_indices = valid_indices[
                self.df["uniqueID"].iloc[valid_indices] != exclude_self_id
            ]

        if len(valid_indices) == 0:
            return []

        filtered_scores = scores[valid_indices]
        # Sort indices by score descending
        sorted_rel_order = np.argsort(-filtered_scores)[:top_k]
        top_indices = valid_indices[sorted_rel_order]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            row = self.df.iloc[idx]
            clean_text = str(row.get("review_clean", ""))
            snippet = clean_text[:120] + "..." if len(clean_text) > 120 else clean_text
            results.append(
                {
                    "uniqueID": int(row.get("uniqueID", 0)),
                    "review_clean_hash": str(row.get("review_clean_hash", "")),
                    "drugName": str(row.get("drugName", "")),
                    "condition": str(row.get("condition", "")),
                    "rating": int(row.get("rating", 0)),
                    "sentiment_class": str(row.get("sentiment_class", "")),
                    "similarity_score": round(score, 4),
                    "snippet": snippet,
                }
            )

        return results


def generate_patient_experience_report(
    cfg: dict[str, Any] | None = None,
    sample_n: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the complete Patient Experience Intelligence pipeline and write reports.

    Args:
        cfg: Optional configuration dictionary. If None, loaded via load_config().
        sample_n: Optional row limit for accelerated testing.

    Returns:
        Tuple of (full_report_dict, summary_report_dict).
    """
    if cfg is None:
        cfg = load_config()

    rev_cfg = cfg.get("reviews", {})
    exp_cfg = rev_cfg.get("patient_experience", {})
    reports_cfg = exp_cfg.get("reports", {})

    processed_data_path = _PROJECT_ROOT / rev_cfg.get(
        "processed_data_path", "Data/reviews/processed/reviews_cleaned.parquet"
    )

    logger.info(f"Loading cleaned review dataset from: {processed_data_path}")
    df = pd.read_parquet(processed_data_path)

    if sample_n is not None and sample_n < len(df):
        logger.info(f"Sampling {sample_n} rows for execution.")
        df = df.sample(n=sample_n, random_state=125).reset_index(drop=True)

    # Ensure sentiment_class target is present
    if "sentiment_class" not in df.columns:
        df = derive_sentiment_target(df)

    min_drug_reviews = int(exp_cfg.get("min_reviews_threshold", 20))
    min_cond_reviews = int(exp_cfg.get("condition_min_reviews_threshold", 10))
    aspect_kws = exp_cfg.get("aspects", {})

    # 1. Overall sentiment analytics
    logger.info("Computing overall sentiment analytics...")
    overall_sentiment = analyze_overall_sentiment(df)

    # 2. Drug-level analytics
    logger.info(f"Computing drug-level analytics (min_reviews={min_drug_reviews})...")
    drug_analytics = analyze_drug_experience(df, min_reviews=min_drug_reviews)

    # 3. Condition-level analytics
    logger.info(
        f"Computing condition-level analytics (min_reviews={min_cond_reviews})..."
    )
    condition_analytics = analyze_condition_experience(df, min_reviews=min_cond_reviews)

    # 4. Temporal analytics
    logger.info("Computing temporal analytics...")
    temporal_analytics = analyze_temporal_experience(df)

    # 5. Aspect intelligence
    logger.info("Extracting aspect intelligence...")
    aspect_analytics = extract_aspect_intelligence(df, aspect_kws)

    # Lineage metadata
    dvc_hash = get_dvc_dataset_hash("Data/reviews/drugsComTest_raw.csv.dvc")
    git_sha = get_git_commit_sha()
    iso_now = datetime.now(timezone.utc).isoformat()

    # Dates
    dates = pd.to_datetime(df["date_parsed"], errors="coerce").dropna()
    min_date = str(dates.min().strftime("%Y-%m-%d")) if not dates.empty else "unknown"
    max_date = str(dates.max().strftime("%Y-%m-%d")) if not dates.empty else "unknown"

    # Full report
    full_report: dict[str, Any] = {
        "workload": "patient_experience_intelligence",
        "sprint": 9,
        "dataset_metadata": {
            "total_records": len(df),
            "unique_drugs": int(df["drugName"].nunique()),
            "unique_conditions": int(df["condition"].nunique()),
            "date_range": {"min_date": min_date, "max_date": max_date},
            "dvc_dataset_hash": dvc_hash,
            "git_commit": git_sha,
        },
        "overall_sentiment": overall_sentiment,
        "drug_experience": {
            "min_reviews_threshold": min_drug_reviews,
            "drugs_analyzed_count": len(drug_analytics),
            "drugs": drug_analytics,
        },
        "condition_experience": {
            "min_reviews_threshold": min_cond_reviews,
            "conditions_analyzed_count": len(condition_analytics),
            "conditions": condition_analytics,
        },
        "temporal_experience": {
            "years_covered": [t["year"] for t in temporal_analytics],
            "timeline": temporal_analytics,
        },
        "aspect_intelligence": aspect_analytics,
        "similarity_configuration": exp_cfg.get("similarity", {}),
        "medical_safety_boundary": (
            "Patient-reported drug experience analytics only. "
            "These analytics describe patterns in the collected review dataset and do NOT "
            "constitute clinical trial evidence, medical advice, prescribing recommendations, "
            "or proof of drug safety and efficacy."
        ),
        "generated_at": iso_now,
    }

    # Compact summary report for API / dashboard consumption
    summary_report: dict[str, Any] = {
        "workload": "patient_experience_intelligence",
        "sprint": 9,
        "total_reviews": len(df),
        "unique_drugs": int(df["drugName"].nunique()),
        "unique_conditions": int(df["condition"].nunique()),
        "overall_sentiment": overall_sentiment.get("sentiment_distribution", {}),
        "mean_rating": overall_sentiment.get("rating_summary", {}).get("mean", 0.0),
        "drugs_meeting_threshold": len(drug_analytics),
        "top_reviewed_drugs": [
            {
                "drugName": d["drugName"],
                "review_count": d["review_count"],
                "mean_rating": d["mean_rating"],
                "high_pct": d["high_pct"],
            }
            for d in drug_analytics[:10]
        ],
        "top_reviewed_conditions": [
            {
                "condition": c["condition"],
                "review_count": c["review_count"],
                "mean_rating": c["mean_rating"],
                "is_missing_category": c["is_missing_category"],
            }
            for c in condition_analytics[:10]
        ],
        "aspect_mention_percentages": {
            k: v["mention_percentage"] for k, v in aspect_analytics.items()
        },
        "lineage": {
            "dvc_dataset_hash": dvc_hash,
            "git_commit": git_sha,
            "generated_at": iso_now,
        },
    }

    # Persist reports
    full_report_path = _PROJECT_ROOT / reports_cfg.get(
        "full_report_path", "Results/patient_experience_report.json"
    )
    summary_report_path = _PROJECT_ROOT / reports_cfg.get(
        "summary_report_path", "Results/patient_experience_summary.json"
    )

    full_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_report_path, "w", encoding="utf-8") as fh:
        json.dump(full_report, fh, indent=2)
    logger.info(f"Saved full patient experience report to: {full_report_path}")

    summary_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_report_path, "w", encoding="utf-8") as fh:
        json.dump(summary_report, fh, indent=2)
    logger.info(f"Saved summary patient experience report to: {summary_report_path}")

    return full_report, summary_report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    generate_patient_experience_report()
