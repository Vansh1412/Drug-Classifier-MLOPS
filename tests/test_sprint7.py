"""
tests/test_sprint7.py — Comprehensive Test Suite for Sprint 7 Multi-Modal Data Foundation.

Validates:
1. Review dataset existence, columns, and row count sanity.
2. Pandera review schema validation (bounds, types, null handling, uniqueID uniqueness).
3. Review data quality assessment (missing condition handling, duplicate detection, HTML entities).
4. NLP preprocessing pipeline (HTML entity decoding, whitespace normalization, length features).
5. Data leakage prevention (Stratified-Group and Temporal splitting with zero text overlap).
6. DVC pipeline integration and stage isolation from the structured champion model.
7. Docker context exclusion for raw review datasets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
RAW_REVIEWS_PATH = PROJECT_ROOT / "Data" / "reviews" / "drugsComTest_raw.csv"
REVIEWS_DVC_PATH = PROJECT_ROOT / "Data" / "reviews" / "drugsComTest_raw.csv.dvc"
PROCESSED_PARQUET_PATH = (
    PROJECT_ROOT / "Data" / "reviews" / "processed" / "reviews_cleaned.parquet"
)

from src.nlp_preprocessing import (
    clean_text,
    compute_text_hash,
    decode_html_entities,
    extract_length_features,
    preprocess_review_dataframe,
    split_reviews_leakage_safe,
)
from src.nlp_validation import (
    validate_review_data_quality,
    validate_review_schema,
)


# =============================================================================
# 1. Dataset Existence & Integrity
# =============================================================================
class TestReviewDatasetIntegrity:
    def test_raw_dataset_exists(self):
        """Assert that Data/reviews/drugsComTest_raw.csv exists and is non-empty."""
        assert RAW_REVIEWS_PATH.is_file(), f"Missing dataset: {RAW_REVIEWS_PATH}"
        assert (
            RAW_REVIEWS_PATH.stat().st_size > 20_000_000
        ), "Dataset unexpectedly small"

    def test_raw_dataset_columns_and_row_count(self):
        """Assert that the raw review dataset contains expected columns and row count."""
        df = pd.read_csv(RAW_REVIEWS_PATH, nrows=10)
        expected_cols = [
            "uniqueID",
            "drugName",
            "condition",
            "review",
            "rating",
            "date",
            "usefulCount",
        ]
        assert list(df.columns) == expected_cols


# =============================================================================
# 2. Schema and Data Quality Validation
# =============================================================================
class TestReviewSchemaValidation:
    @pytest.fixture
    def sample_review_df(self) -> pd.DataFrame:
        """Provide a minimal valid review DataFrame."""
        return pd.DataFrame(
            {
                "uniqueID": [101, 102, 103],
                "drugName": ["Aspirin", "Ibuprofen", "Paracetamol"],
                "condition": ["Headache", "Pain", None],
                "review": [
                    "Works well for headaches.",
                    "Relieved my pain quickly.",
                    "No noticeable side effects.",
                ],
                "rating": [8, 9, 7],
                "date": ["28-Feb-12", "17-May-09", "29-Sep-17"],
                "usefulCount": [12, 45, 0],
            }
        )

    def test_valid_schema_passes(self, sample_review_df):
        """Assert that a compliant DataFrame passes schema validation."""
        is_valid, errors = validate_review_schema(sample_review_df)
        assert is_valid, f"Expected valid schema but got errors: {errors}"
        assert len(errors) == 0

    def test_missing_required_column_fails(self, sample_review_df):
        """Assert that omitting a required column (e.g. rating) fails schema validation."""
        df_invalid = sample_review_df.drop(columns=["rating"])
        is_valid, errors = validate_review_schema(df_invalid)
        assert not is_valid
        assert len(errors) > 0

    def test_out_of_bounds_rating_fails(self, sample_review_df):
        """Assert that ratings outside [1, 10] fail validation."""
        df_invalid = sample_review_df.copy()
        df_invalid.loc[0, "rating"] = 11
        is_valid, errors = validate_review_schema(df_invalid)
        assert not is_valid

        df_invalid.loc[0, "rating"] = 0
        is_valid, errors = validate_review_schema(df_invalid)
        assert not is_valid

    def test_negative_useful_count_fails(self, sample_review_df):
        """Assert that negative usefulCount fails validation."""
        df_invalid = sample_review_df.copy()
        df_invalid.loc[0, "usefulCount"] = -1
        is_valid, errors = validate_review_schema(df_invalid)
        assert not is_valid

    def test_empty_review_text_fails(self, sample_review_df):
        """Assert that empty or whitespace-only reviews fail validation."""
        df_invalid = sample_review_df.copy()
        df_invalid.loc[0, "review"] = "   "
        is_valid, errors = validate_review_schema(df_invalid)
        assert not is_valid

    def test_duplicate_unique_id_fails(self, sample_review_df):
        """Assert that duplicate uniqueIDs fail schema validation."""
        df_invalid = sample_review_df.copy()
        df_invalid.loc[1, "uniqueID"] = df_invalid.loc[0, "uniqueID"]
        is_valid, errors = validate_review_schema(df_invalid)
        assert not is_valid

    def test_data_quality_missing_condition_yields_warning(self, sample_review_df):
        """Assert that missing conditions produce a WARNING status without failing."""
        cfg = {"reviews": {"validation": {"max_missing_condition_rate": 0.40}}}
        quality = validate_review_data_quality(sample_review_df, cfg=cfg)
        assert quality["status"] == "WARNING"
        assert quality["missing_condition_count"] == 1
        assert any("missing conditions" in w for w in quality["warnings"])


# =============================================================================
# 3. NLP Preprocessing & Feature Extraction
# =============================================================================
class TestReviewPreprocessing:
    def test_decode_html_entities(self):
        """Assert that HTML entities are properly decoded."""
        raw = "I&#039;ve tried this &amp; that &quot;pill&quot;."
        decoded = decode_html_entities(raw)
        assert decoded == 'I\'ve tried this & that "pill".'

    def test_clean_text_normalizes_whitespace_and_control_chars(self):
        """Assert that whitespace is collapsed and control characters are stripped."""
        raw = '  "Great medicine!\r\n\tWorked   well.\x00  " '
        cleaned = clean_text(raw)
        assert cleaned == "Great medicine! Worked well."

    def test_clean_text_preserves_medical_terms_and_negations(self):
        """Assert that negations and medical terminology are strictly preserved."""
        raw = (
            "Did NOT work for my depression or bipolar symptoms; no relief whatsoever."
        )
        cleaned = clean_text(raw)
        assert "NOT" in cleaned
        assert "depression" in cleaned
        assert "bipolar" in cleaned
        assert "no relief" in cleaned

    def test_extract_length_features(self):
        """Assert that character, word, and sentence counts are calculated correctly."""
        df = pd.DataFrame(
            {
                "review_clean": [
                    "First sentence. Second sentence! Third sentence?",
                    "Single sentence review.",
                ]
            }
        )
        df_feats = extract_length_features(df, text_col="review_clean")
        assert "review_char_count" in df_feats.columns
        assert "review_word_count" in df_feats.columns
        assert "review_sentence_count" in df_feats.columns
        assert df_feats.loc[0, "review_sentence_count"] == 3
        assert df_feats.loc[1, "review_sentence_count"] == 1
        assert df_feats.loc[0, "review_word_count"] == 6

    def test_preprocess_preserves_raw_and_clean(self):
        """Assert that preprocessing retains review_raw and creates review_clean."""
        df = pd.DataFrame(
            {
                "uniqueID": [1],
                "drugName": ["DrugA"],
                "condition": [None],
                "review": ["It&#039;s effective."],
                "rating": [9],
                "date": ["10-Aug-16"],
                "usefulCount": [5],
            }
        )
        processed = preprocess_review_dataframe(df)
        assert processed.loc[0, "review_raw"] == "It&#039;s effective."
        assert processed.loc[0, "review_clean"] == "It's effective."
        assert processed.loc[0, "condition_raw"] is None
        assert processed.loc[0, "condition_clean"] == "Unknown"
        assert "review_clean_hash" in processed.columns

    def test_compute_text_hash_deterministic(self):
        """Assert that identical review text produces identical SHA-256 hash."""
        h1 = compute_text_hash("Great drug for migraine.")
        h2 = compute_text_hash("  great drug for migraine.  ")
        assert h1 == h2
        assert len(h1) == 64


# =============================================================================
# 4. Data Leakage Defense & Splitting Strategy
# =============================================================================
class TestLeakageSafeSplitting:
    @pytest.fixture
    def mock_reviews_with_duplicates(self) -> pd.DataFrame:
        """Provide a dataset with duplicate review texts across multiple rows."""
        data = [
            (1, "DrugA", "Pain", "Excellent painkiller.", 10, "10-Jan-15"),
            (
                2,
                "DrugB",
                "Pain",
                "Excellent painkiller.",
                9,
                "15-Feb-15",
            ),  # Duplicate text
            (3, "DrugC", "Anxiety", "Helped with anxiety.", 8, "20-Mar-15"),
            (
                4,
                "DrugD",
                "Anxiety",
                "Helped with anxiety.",
                8,
                "25-Apr-16",
            ),  # Duplicate text
            (5, "DrugE", "Insomnia", "Sleep was restored.", 9, "05-May-16"),
            (6, "DrugF", "Depression", "No improvement.", 2, "12-Jun-16"),
            (7, "DrugG", "Depression", "Severe nausea.", 1, "18-Jul-16"),
            (8, "DrugH", "Acne", "Clear skin finally.", 10, "22-Aug-16"),
            (9, "DrugI", "Weight", "Reduced appetite.", 7, "30-Sep-17"),
            (
                10,
                "DrugJ",
                "Weight",
                "Reduced appetite.",
                7,
                "15-Oct-17",
            ),  # Duplicate text in 2017
        ]
        df = pd.DataFrame(
            data,
            columns=["uniqueID", "drugName", "condition", "review", "rating", "date"],
        )
        df["usefulCount"] = 5
        return preprocess_review_dataframe(df)

    def test_stratified_group_split_zero_leakage(self, mock_reviews_with_duplicates):
        """Assert that identical review texts NEVER cross the train/test boundary in group split."""
        train_df, test_df = split_reviews_leakage_safe(
            mock_reviews_with_duplicates, strategy="stratified_group", test_size=0.3
        )
        train_texts = set(train_df["review_clean"])
        test_texts = set(test_df["review_clean"])
        overlap = train_texts.intersection(test_texts)
        assert len(overlap) == 0, f"Data leakage detected! Overlapping texts: {overlap}"

    def test_temporal_split_zero_leakage(self, mock_reviews_with_duplicates):
        """Assert that temporal splitting eliminates review text overlap between periods."""
        train_df, test_df = split_reviews_leakage_safe(
            mock_reviews_with_duplicates,
            strategy="temporal",
            temporal_threshold_date="2017-01-01",
        )
        train_texts = set(train_df["review_clean"])
        test_texts = set(test_df["review_clean"])
        overlap = train_texts.intersection(test_texts)
        assert len(overlap) == 0, f"Temporal text leakage detected: {overlap}"
        assert all(train_df["date_parsed"] < "2017-01-01")
        assert all(test_df["date_parsed"] >= "2017-01-01")


# =============================================================================
# 5. DVC Review Stages & Isolation
# =============================================================================
class TestDVCReviewStages:
    def test_reviews_dvc_metadata_file_exists(self):
        """Assert that Data/reviews/drugsComTest_raw.csv.dvc exists with valid hash."""
        assert REVIEWS_DVC_PATH.is_file(), f"Missing DVC file: {REVIEWS_DVC_PATH}"
        with open(REVIEWS_DVC_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert "outs" in data
        assert len(data["outs"]) == 1
        entry = data["outs"][0]
        assert entry["path"] == "drugsComTest_raw.csv"
        assert entry["size"] == 27637580
        assert len(entry["md5"]) == 32

    def test_dvc_yaml_review_stages_defined(self):
        """Assert that dvc.yaml contains validate_reviews and preprocess_reviews stages."""
        dvc_yaml = PROJECT_ROOT / "dvc.yaml"
        with open(dvc_yaml, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        stages = cfg.get("stages", {})
        assert "validate_reviews" in stages
        assert "preprocess_reviews" in stages

    def test_dvc_yaml_review_stages_isolated_from_structured_model(self):
        """Assert that review stages do NOT depend on structured dataset or model artifacts."""
        dvc_yaml = PROJECT_ROOT / "dvc.yaml"
        with open(dvc_yaml, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        stages = cfg.get("stages", {})

        for stage_name in ["validate_reviews", "preprocess_reviews"]:
            deps = stages[stage_name].get("deps", [])
            assert (
                "Data/drug.csv" not in deps
            ), f"{stage_name} must not depend on Data/drug.csv"
            assert (
                "Model/drug_pipeline.skops" not in deps
            ), f"{stage_name} must not depend on champion model"
            assert "train.py" not in deps, f"{stage_name} must not depend on train.py"


# =============================================================================
# 6. Docker Context Exclusion
# =============================================================================
class TestDockerReviewExclusion:
    def test_dockerignore_excludes_review_data(self):
        """Assert that .dockerignore excludes raw review data to prevent image bloat."""
        dockerignore_path = PROJECT_ROOT / ".dockerignore"
        content = dockerignore_path.read_text(encoding="utf-8")
        assert "Data/reviews/" in content
        assert "Data/drug_reviews/" in content

    def test_dockerignore_preserves_structured_dataset(self):
        """Assert that .dockerignore does NOT exclude the structured dataset needed for serving."""
        dockerignore_path = PROJECT_ROOT / ".dockerignore"
        content = dockerignore_path.read_text(encoding="utf-8")
        assert "Data/drug.csv" not in content
        assert "!Data/drug.csv" not in content or "Data/drug.csv" not in content
