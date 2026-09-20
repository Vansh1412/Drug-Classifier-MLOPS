"""
src/nlp_preprocessing.py — Reproducible NLP Preprocessing & Leakage-Safe Splitting for Drug Reviews.

Transforms raw drug reviews into a structured, clean dataset:
1. Decodes HTML entities (e.g., &#039;, &amp;, &quot;) using html.unescape.
2. Normalizes whitespace and strips control characters while strictly preserving
   negations, medical terms, symptoms, and drug names.
3. Preserves both raw and clean review text for lineage and reproducibility.
4. Computes lightweight text-length features (char count, word count, sentence count).
5. Documents and applies neutral placeholder handling for missing conditions.
6. Implements leakage-safe splitting (Stratified-Group and Temporal) to ensure
   identical review texts (5,486 occurrences) never cross the train/test boundary.

Usage:
    from src.nlp_preprocessing import run_review_preprocessing, split_reviews_leakage_safe
    processed_path = run_review_preprocessing("Data/reviews/drugsComTest_raw.csv")
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path for direct script execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import load_config
from src.nlp_validation import validate_review_dataset

logger = logging.getLogger(__name__)

# Control characters regex (excluding \t, \n, \r which are normalized separately)
_CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_WHITESPACE_REGEX = re.compile(r"\s+")
_SENTENCE_SPLIT_REGEX = re.compile(r"[.!?]+(?:\s+|$)")


def decode_html_entities(text: str) -> str:
    """Decode HTML escape entities (e.g. &#039;, &amp;, &quot;) into standard characters.

    Args:
        text: Input string potentially containing HTML entities.

    Returns:
        Decoded string.
    """
    if not isinstance(text, str):
        return ""
    return html.unescape(text)


def clean_text(text: str) -> str:
    """Normalize text while preserving all linguistic, medical, and negation content.

    Transformations applied:
    - HTML entity decoding (html.unescape)
    - Surrounding quote removal
    - Control character removal
    - Newline and tab replacement with single space
    - Multiple whitespace collapsing
    - Leading/trailing whitespace stripping

    Note: Does NOT remove negations (not, no, never), punctuation, stop words,
    drug names, or clinical terms.

    Args:
        text: Raw review string.

    Returns:
        Cleaned review string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Decode HTML entities
    decoded = decode_html_entities(text)

    # 2. Strip surrounding quotes if wrapped
    stripped = decoded.strip().strip('"').strip("'")

    # 3. Strip control characters
    no_control = _CONTROL_CHAR_REGEX.sub(" ", stripped)

    # 4. Collapse multiple whitespace and normalize newlines/tabs
    normalized = _WHITESPACE_REGEX.sub(" ", no_control).strip()

    return normalized


def compute_text_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized lowercase review text.

    Used for grouping identical review texts to prevent data leakage during splitting.

    Args:
        text: Review text.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    normalized_key = text.strip().lower().encode("utf-8")
    return hashlib.sha256(normalized_key).hexdigest()


def extract_length_features(
    df: pd.DataFrame, text_col: str = "review_clean"
) -> pd.DataFrame:
    """Extract lightweight metadata features from the cleaned review text.

    Features computed:
    - review_char_count: Length of clean text in characters
    - review_word_count: Number of whitespace-delimited words
    - review_sentence_count: Number of sentences estimated by terminal punctuation

    Args:
        df: DataFrame containing text_col.
        text_col: Name of the text column to analyze.

    Returns:
        DataFrame with added feature columns.
    """
    texts = df[text_col].astype(str)

    df["review_char_count"] = texts.str.len().astype(int)
    df["review_word_count"] = texts.apply(lambda t: len(t.split())).astype(int)

    def count_sentences(t: str) -> int:
        if not t:
            return 0
        sentences = [s for s in _SENTENCE_SPLIT_REGEX.split(t) if s.strip()]
        return max(1, len(sentences))

    df["review_sentence_count"] = texts.apply(count_sentences).astype(int)
    return df


def preprocess_review_dataframe(
    df: pd.DataFrame, cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Execute end-to-end preprocessing on the review dataset.

    Preserves raw columns, creates clean columns, generates text hashes for
    leakage-safe grouping, and extracts text-length features.

    Args:
        df: Raw review DataFrame.
        cfg: Optional configuration dictionary.

    Returns:
        Processed pandas DataFrame.
    """
    if cfg is None:
        cfg = load_config()

    rev_cfg = cfg.get("reviews", {})
    val_cfg = rev_cfg.get("validation", {})
    missing_placeholder = val_cfg.get("missing_condition_placeholder", "Unknown")

    processed = df.copy()

    # 1. Preserve raw review and create review_clean
    processed["review_raw"] = processed["review"]
    processed["review_clean"] = processed["review"].apply(clean_text)

    # 2. Preserve raw condition and create condition_clean
    processed["condition_raw"] = processed["condition"]
    processed["condition_clean"] = (
        processed["condition"].fillna(missing_placeholder).astype(str).str.strip()
    )

    # 3. Clean drugName
    processed["drugName_clean"] = processed["drugName"].astype(str).str.strip()

    # 4. Generate deterministic text hash for duplicate grouping / leakage defense
    processed["review_clean_hash"] = processed["review_clean"].apply(compute_text_hash)

    # 5. Parse date and extract ISO date and year
    parsed_dates = pd.to_datetime(processed["date"], format="%d-%b-%y", errors="coerce")
    processed["date_parsed"] = parsed_dates.dt.strftime("%Y-%m-%d")
    processed["year"] = parsed_dates.dt.year.astype(int)

    # 6. Extract length features
    processed = extract_length_features(processed, text_col="review_clean")

    return processed


def split_reviews_leakage_safe(
    df: pd.DataFrame,
    strategy: str = "stratified_group",
    test_size: float = 0.2,
    random_state: int = 125,
    stratify_col: str = "rating",
    group_col: str = "review_clean_hash",
    temporal_threshold_date: str = "2017-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split review dataset into train and test sets with strict data leakage prevention.

    Guarantees:
    - Zero overlap in cleaned review texts between train and test sets
      (set(train[review_clean]).intersection(set(test[review_clean])) == empty).

    Strategies:
    - 'stratified_group': Groups identical reviews by group_col (hash) and splits
      entire groups using StratifiedGroupKFold, preserving rating distribution.
    - 'temporal': Splits by date (< threshold vs >= threshold) and removes any
      test records whose review text already appeared in the training set.

    Args:
        df: Preprocessed DataFrame containing group_col, stratify_col, date_parsed.
        strategy: 'stratified_group' or 'temporal'.
        test_size: Fraction of groups to allocate to test (for stratified_group).
        random_state: Seed for reproducible group splitting.
        stratify_col: Column to preserve distribution across splits (e.g. 'rating').
        group_col: Column defining groups of identical reviews (e.g. 'review_clean_hash').
        temporal_threshold_date: Date string 'YYYY-MM-DD' for temporal splitting.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If an unknown strategy is specified or required columns are missing.
    """
    if group_col not in df.columns:
        raise ValueError(f"Required group column '{group_col}' missing from DataFrame.")

    if strategy == "stratified_group":
        # Calculate number of folds corresponding to test_size (e.g. test_size=0.2 -> 5 folds)
        n_splits = max(2, int(round(1.0 / test_size)))
        groups = df[group_col]

        # Use binned stratification (Low 1-4, Neutral 5-7, High 8-10) for robust group stratification
        try:
            if stratify_col in df.columns and df[stratify_col].nunique() > 3:
                y = pd.cut(
                    df[stratify_col], bins=[-np.inf, 4, 7, np.inf], labels=[0, 1, 2]
                ).astype(int)
            else:
                y = (
                    df[stratify_col].astype(str)
                    if stratify_col in df.columns
                    else np.zeros(len(df), dtype=int)
                )

            sgkf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            train_idx, test_idx = next(sgkf.split(df, y, groups))
        except ValueError:
            # Fallback to GroupKFold if rare classes prevent stratification
            from sklearn.model_selection import GroupKFold

            gkf = GroupKFold(n_splits=n_splits)
            train_idx, test_idx = next(gkf.split(df, groups=groups))

        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    elif strategy == "temporal":
        if "date_parsed" not in df.columns:
            raise ValueError(
                "Temporal splitting requires 'date_parsed' column in DataFrame."
            )

        train_mask = df["date_parsed"] < temporal_threshold_date
        test_mask = df["date_parsed"] >= temporal_threshold_date

        train_df = df[train_mask].copy().reset_index(drop=True)
        candidate_test_df = df[test_mask].copy().reset_index(drop=True)

        # Leakage prevention: purge test reviews whose clean text already exists in train
        train_hashes = set(train_df[group_col])
        test_df = (
            candidate_test_df[~candidate_test_df[group_col].isin(train_hashes)]
            .copy()
            .reset_index(drop=True)
        )

        purged_count = len(candidate_test_df) - len(test_df)
        if purged_count > 0:
            logger.info(
                f"Temporal split: Purged {purged_count} test records to prevent "
                f"identical review text leakage from training period."
            )
    else:
        raise ValueError(
            f"Unknown splitting strategy '{strategy}'. Supported: 'stratified_group', 'temporal'."
        )

    # Verification: assert zero text leakage
    train_texts = set(train_df["review_clean"])
    test_texts = set(test_df["review_clean"])
    overlap = train_texts.intersection(test_texts)
    if len(overlap) > 0:
        raise RuntimeError(
            f"Data leakage detected! {len(overlap)} review texts present in both train and test sets."
        )

    logger.info(
        f"Leakage-safe split complete ({strategy}): "
        f"Train={len(train_df)} rows, Test={len(test_df)} rows, Overlap=0"
    )
    return train_df, test_df


def run_review_preprocessing(
    input_path: str | Path = "Data/reviews/drugsComTest_raw.csv",
    output_path: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Execute complete review preprocessing and persist processed dataset to disk.

    Args:
        input_path: Path to raw review CSV.
        output_path: Optional path for output parquet file. If None, loaded from config.
        cfg: Optional configuration dictionary.

    Returns:
        Path to the saved processed parquet file.
    """
    if cfg is None:
        cfg = load_config()

    rev_cfg = cfg.get("reviews", {})
    if output_path is None:
        output_path = rev_cfg.get(
            "processed_data_path", "Data/reviews/processed/reviews_cleaned.parquet"
        )

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Validate raw dataset
    raw_df = validate_review_dataset(input_path, cfg)

    # 2. Preprocess
    logger.info("Executing review preprocessing pipeline...")
    processed_df = preprocess_review_dataframe(raw_df, cfg)

    # 3. Persist to Parquet
    logger.info(f"Saving processed dataset to {out_p}")
    processed_df.to_parquet(out_p, index=False, engine="pyarrow")

    logger.info(
        f"Preprocessing complete: {processed_df.shape[0]} rows, "
        f"{processed_df.shape[1]} columns saved to {out_p}"
    )
    return out_p


if __name__ == "__main__":
    import sys

    in_path = sys.argv[1] if len(sys.argv) > 1 else "Data/reviews/drugsComTest_raw.csv"
    out_file = run_review_preprocessing(in_path)
    print(f"Review preprocessing SUCCESS. Output: {out_file}")
