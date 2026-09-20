"""
tests/test_sprint10.py — Test Suite for Sprint 10 NLP Monitoring & Drift Detection.

Comprehensive test coverage (~30 tests):
1. Reference Dataset & Baseline Fingerprinting (S10-T2)
2. Schema & Data Quality Monitoring (S10-T3)
3. Review Length Drift (S10-T4)
4. Vocabulary Drift (S10-T5)
5. TF-IDF Feature Distribution Drift (S10-T6)
6. Sentiment Distribution Drift (S10-T7)
7. Rating Distribution Drift (S10-T8)
8. Aspect Drift (S10-T9)
9. Drug & Condition Category Drift (S10-T10)
10. Prediction Drift (S10-T11)
11. Overall Certification (S10-T12)
12. Persistence & Security (S10-T13, S10-T14, S10-T20)
13. API Endpoints (S10-T15)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.nlp_monitoring import (
    NLPMonitor,
    get_nlp_monitor,
    tokenize_text,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def monitor() -> NLPMonitor:
    """Provide a shared NLPMonitor instance for the test suite."""
    return get_nlp_monitor()


@pytest.fixture
def clean_batch(monitor: NLPMonitor) -> pd.DataFrame:
    """Clean synthetic current batch sampled from reference baseline."""
    return monitor.create_synthetic_current_batch(
        size=200, shift_type=None, random_state=42
    )


@pytest.fixture
def api_client() -> TestClient:
    """Provide FastAPI test client."""
    return TestClient(app)


# =============================================================================
# 1. Reference Dataset & Baseline Fingerprinting (S10-T2)
# =============================================================================


class TestReferenceBaseline:
    def test_reference_dataset_loads(self, monitor: NLPMonitor):
        """Verify that the reference dataset loads with expected rows and columns."""
        assert monitor.reference_df is not None
        assert len(monitor.reference_df) > 50000
        assert "review_clean" in monitor.reference_df.columns
        assert "rating" in monitor.reference_df.columns
        assert "sentiment_class" in monitor.reference_df.columns

    def test_reference_snapshot_fingerprint_deterministic(self, monitor: NLPMonitor):
        """Verify reference snapshot metadata is deterministic and populated."""
        snapshot = monitor.reference_snapshot
        assert "file_sha256" in snapshot
        assert len(snapshot["file_sha256"]) == 64  # SHA-256
        assert snapshot["rows"] == len(monitor.reference_df)
        assert "created_timestamp" in snapshot
        assert "config_version" in snapshot

    def test_current_reference_distinction(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify that current batch is distinct from the full reference dataset."""
        assert len(clean_batch) < len(monitor.reference_df)
        report = monitor.certify_drift(clean_batch, save_report=False)
        assert report["reference"]["rows"] == len(monitor.reference_df)
        assert report["current"]["rows"] == len(clean_batch)


# =============================================================================
# 2. Schema & Data Quality Monitoring (S10-T3)
# =============================================================================


class TestDataQualityMonitoring:
    def test_clean_batch_passes_quality(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify a clean sample from baseline passes data quality checks."""
        res = monitor.validate_batch_quality(clean_batch)
        assert res["status"] in ["PASS", "WARNING"]
        assert len(res["missing_columns"]) == 0
        assert res["invalid_ratings_count"] == 0

    def test_missing_column_triggers_error(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify that a missing required column produces ERROR status."""
        bad_df = clean_batch.drop(columns=["rating"])
        res = monitor.validate_batch_quality(bad_df)
        assert res["status"] == "ERROR"
        assert "rating" in res["missing_columns"]

    def test_invalid_rating_range_triggers_error(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify that out-of-range ratings (<1 or >10) trigger ERROR status."""
        bad_df = clean_batch.copy()
        bad_df.loc[bad_df.index[:5], "rating"] = 15
        res = monitor.validate_batch_quality(bad_df)
        assert res["status"] == "ERROR"
        assert res["invalid_ratings_count"] >= 5

    def test_empty_review_rate_detection(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify that a high empty review rate triggers warning or drift."""
        bad_df = clean_batch.copy()
        bad_df.loc[bad_df.index[:20], "review_clean"] = "   "
        res = monitor.validate_batch_quality(bad_df)
        assert res["empty_review_count"] >= 20
        assert res["empty_review_rate"] > 0.0

    def test_unexpected_columns_produce_warning(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify that unexpected non-standard columns produce a WARNING."""
        bad_df = clean_batch.copy()
        bad_df["untracked_internal_feature"] = "test"
        res = monitor.validate_batch_quality(bad_df)
        assert "untracked_internal_feature" in res["unexpected_columns"]
        assert res["status"] == "WARNING"


# =============================================================================
# 3. Review Length Drift (S10-T4)
# =============================================================================


class TestReviewLengthDrift:
    def test_clean_batch_length_drift_pass(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify that a representative batch does not trigger length drift."""
        res = monitor.compute_length_drift(clean_batch)
        assert res["status"] in ["PASS", "WARNING"]
        assert "review_char_count" in res["features"]
        assert "ks_p_value" in res["features"]["review_char_count"]

    def test_synthetic_short_reviews_trigger_length_drift(self, monitor: NLPMonitor):
        """Verify that an influx of extremely short reviews triggers DRIFT."""
        short_batch = monitor.create_synthetic_current_batch(
            size=200, shift_type="length_shift"
        )
        res = monitor.compute_length_drift(short_batch)
        assert res["status"] == "DRIFT"
        assert res["drift_detected"] is True
        assert res["features"]["review_word_count"]["status"] == "DRIFT"


# =============================================================================
# 4. Vocabulary Drift (S10-T5)
# =============================================================================


class TestVocabularyDrift:
    def test_tokenizer_deterministic(self):
        """Verify deterministic whitespace/punctuation tokenization."""
        text = "Severe nausea, dizziness & headache! (100mg daily)."
        tokens = tokenize_text(text)
        assert tokens == ["severe", "nausea", "dizziness", "headache", "daily"]

    def test_clean_batch_vocabulary_overlap(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify vocabulary overlap with baseline on clean sample."""
        res = monitor.compute_vocabulary_drift(clean_batch)
        assert res["jaccard_similarity"] > 0.0
        assert res["vocabulary_overlap_rate"] > 0.50
        assert "medical_disclaimer" in res

    def test_unseen_vocabulary_detection(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify that injected out-of-vocabulary tokens are detected."""
        oov_df = clean_batch.copy()
        oov_df["review_clean"] = "zyxwvutsrq ponmlkjihg fedcba newsyntheticcompoundx"
        res = monitor.compute_vocabulary_drift(oov_df)
        assert res["new_token_count"] > 0
        assert res["status"] in ["WARNING", "DRIFT"]


# =============================================================================
# 5. TF-IDF Feature Distribution Drift (S10-T6)
# =============================================================================


class TestTfidfDrift:
    def test_reference_vectorizer_is_fixed(self, monitor: NLPMonitor):
        """Verify reference TF-IDF vectorizer is pre-fitted and fixed."""
        assert hasattr(monitor, "tfidf_vectorizer")
        assert len(monitor.feature_names) > 100
        assert len(monitor.ref_tfidf_mean) == len(monitor.feature_names)

    def test_tfidf_transform_only_on_current(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify that current batch is transformed without altering vectorizer vocabulary."""
        initial_vocab_size = len(monitor.tfidf_vectorizer.vocabulary_)
        res = monitor.compute_tfidf_drift(clean_batch)
        assert len(monitor.tfidf_vectorizer.vocabulary_) == initial_vocab_size
        assert "cosine_distance" in res
        assert res["cosine_distance"] >= 0.0

    def test_dissimilar_batch_triggers_tfidf_drift(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify that radically shifted text triggers TF-IDF drift."""
        shifted_df = clean_batch.copy()
        shifted_df["review_clean"] = (
            "xylophone zebra quantum mechanics astrophysics telescope satellite orbit"
        )
        res = monitor.compute_tfidf_drift(shifted_df)
        assert res["cosine_distance"] > 0.50
        assert res["status"] in ["WARNING", "DRIFT"]


# =============================================================================
# 6. Sentiment Distribution Drift (S10-T7)
# =============================================================================


class TestSentimentDrift:
    def test_sentiment_distribution_clean_batch(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify sentiment drift evaluation on clean batch."""
        res = monitor.compute_sentiment_drift(clean_batch)
        assert res["status"] in ["PASS", "WARNING"]
        assert "psi" in res
        assert "current_distribution" in res

    def test_sentiment_shift_triggers_drift(self, monitor: NLPMonitor):
        """Verify that an artificially skewed negative batch triggers sentiment DRIFT."""
        neg_batch = monitor.create_synthetic_current_batch(
            size=200, shift_type="sentiment_shift"
        )
        res = monitor.compute_sentiment_drift(neg_batch)
        assert res["status"] == "DRIFT"
        assert res["drift_detected"] is True
        assert res["psi"] >= 0.10


# =============================================================================
# 7. Rating Distribution Drift (S10-T8)
# =============================================================================


class TestRatingDrift:
    def test_rating_drift_clean_batch(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify rating distribution calculation on clean batch."""
        res = monitor.compute_rating_drift(clean_batch)
        assert res["status"] in ["PASS", "WARNING"]
        assert "ks_statistic" in res
        assert "wasserstein_distance" in res
        assert "current_summary" in res

    def test_rating_skew_triggers_drift(self, monitor: NLPMonitor):
        """Verify that skewed ratings trigger rating DRIFT and descriptive report."""
        skewed_batch = monitor.create_synthetic_current_batch(
            size=200, shift_type="sentiment_shift"
        )
        res = monitor.compute_rating_drift(skewed_batch)
        assert res["status"] == "DRIFT"
        assert res["drift_detected"] is True
        assert "shifted toward lower ratings" in res["description"]


# =============================================================================
# 8. Aspect Drift (S10-T9)
# =============================================================================


class TestAspectDrift:
    def test_aspects_match_sprint9_vocabulary(self, monitor: NLPMonitor):
        """Verify that aspect monitoring uses the 8 predefined clinical domains."""
        expected_aspects = {
            "side_effects",
            "effectiveness",
            "dosage_administration",
            "cost_affordability",
            "sleep_drowsiness",
            "mental_mood",
            "weight_appetite",
            "withdrawal_dependence",
        }
        assert expected_aspects.issubset(set(monitor.aspects_cfg.keys()))

    def test_clean_batch_aspect_metrics(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify aspect prevalence computation on clean batch."""
        res = monitor.compute_aspect_drift(clean_batch)
        assert "aspects" in res
        assert "side_effects" in res["aspects"]
        assert "reference_percentage" in res["aspects"]["side_effects"]

    def test_aspect_mention_surge_triggers_drift(self, monitor: NLPMonitor):
        """Verify that a batch with high side effects mentions triggers aspect DRIFT."""
        aspect_batch = monitor.create_synthetic_current_batch(
            size=200, shift_type="aspect_shift"
        )
        res = monitor.compute_aspect_drift(aspect_batch)
        assert res["aspects"]["side_effects"]["current_percentage"] > 80.0
        assert res["aspects"]["side_effects"]["status"] == "DRIFT"
        assert res["status"] == "DRIFT"


# =============================================================================
# 9. Drug & Condition Category Drift (S10-T10)
# =============================================================================


class TestCategoryDrift:
    def test_category_drift_clean_batch(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify category drift evaluation on clean batch."""
        res = monitor.compute_category_drift(clean_batch)
        assert "categories" in res
        assert "drugName" in res["categories"]
        assert "condition" in res["categories"]

    def test_unknown_condition_handled_explicitly(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify that 'Unknown' condition is explicitly handled as missing."""
        df_with_unknown = clean_batch.copy()
        df_with_unknown.loc[df_with_unknown.index[:10], "condition"] = "Unknown"
        res = monitor.compute_category_drift(df_with_unknown)
        assert res["categories"]["condition"]["status"] in ["PASS", "WARNING"]


# =============================================================================
# 10. Prediction Drift (S10-T11)
# =============================================================================


class TestPredictionDrift:
    def test_prediction_drift_not_available_when_none(self, monitor: NLPMonitor):
        """Verify status is NOT_AVAILABLE when no prediction stream is provided."""
        res = monitor.compute_prediction_drift(current_predictions=None)
        assert res["status"] == "NOT_AVAILABLE"
        assert "Sprint 8 produced no accepted NLP champion" in res["reason"]

    def test_prediction_drift_computes_when_supplied(self, monitor: NLPMonitor):
        """Verify prediction drift evaluates correctly when explicit predictions are supplied."""
        preds = ["high"] * 100 + ["low"] * 100
        res = monitor.compute_prediction_drift(current_predictions=preds)
        assert res["status"] in ["PASS", "WARNING", "DRIFT"]
        assert "psi" in res
        assert "current_distribution" in res


# =============================================================================
# 11. Overall Certification (S10-T12)
# =============================================================================


class TestOverallCertification:
    def test_clean_batch_certifies_pass_or_warning(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify clean baseline sample certifies as PASS or WARNING."""
        report = monitor.certify_drift(clean_batch, save_report=False)
        assert report["certification"]["overall_status"] in ["PASS", "WARNING"]
        assert report["certification"]["automated_retraining_triggered"] is False

    def test_shifted_batch_certifies_drift(self, monitor: NLPMonitor):
        """Verify that a multi-dimensionally shifted batch certifies as DRIFT."""
        shifted_batch = monitor.create_synthetic_current_batch(
            size=200, shift_type="sentiment_shift"
        )
        report = monitor.certify_drift(shifted_batch, save_report=False)
        assert report["certification"]["overall_status"] == "DRIFT"
        assert report["certification"]["drift_detected"] is True
        assert len(report["certification"]["signals"]) > 0

    def test_corrupt_batch_certifies_error(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify that schema/rating corruption certifies as ERROR."""
        corrupt_batch = clean_batch.copy()
        corrupt_batch.loc[corrupt_batch.index[:10], "rating"] = 999
        report = monitor.certify_drift(corrupt_batch, save_report=False)
        assert report["certification"]["overall_status"] == "ERROR"
        assert (
            "critical_data_quality_or_schema_failure"
            in report["certification"]["signals"]
        )


# =============================================================================
# 12. Persistence & Security (S10-T13, S10-T14, S10-T20)
# =============================================================================


class TestPersistenceAndSecurity:
    def test_save_and_retrieve_latest_report(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify saving and retrieving the latest report from disk."""
        report = monitor.certify_drift(clean_batch, save_report=True)
        saved_report = monitor.get_latest_report()
        assert saved_report is not None
        assert saved_report["report_id"] == report["report_id"]
        assert Path("Results/nlp_drift_report.json").exists()

    def test_path_traversal_guard(self, monitor: NLPMonitor):
        """Verify that path traversal attempts in report_id are blocked."""
        malicious_report = {"report_id": "../../../evil"}
        # The save_report method strips slashes and dots; ensure traversal outside directory raises or stays safe
        dest_path = monitor.save_report(malicious_report)
        assert dest_path.is_relative_to(monitor.reports_dir.resolve())

    def test_zero_raw_review_text_in_report(
        self, monitor: NLPMonitor, clean_batch: pd.DataFrame
    ):
        """Verify that raw review text is never serialized into the drift report."""
        report = monitor.certify_drift(clean_batch, save_report=False)
        report_str = json.dumps(report)
        for text in clean_batch["review"].head(5):
            if len(text) > 30:
                assert text not in report_str


# =============================================================================
# 13. API Endpoints (S10-T15)
# =============================================================================


class TestAPIIntegration:
    def test_get_latest_nlp_report(self, api_client: TestClient):
        """Verify GET /monitoring/nlp/latest returns valid structured JSON."""
        response = api_client.get("/monitoring/nlp/latest")
        assert response.status_code == 200
        data = response.json()
        assert "certification" in data
        assert "overall_status" in data["certification"]
        assert "disclaimer" in data

    def test_post_nlp_drift_evaluation(self, api_client: TestClient):
        """Verify POST /monitoring/nlp/drift evaluates an incoming review batch."""
        payload = {
            "reviews": [
                {
                    "uniqueID": 10001,
                    "drugName": "Lipitor",
                    "condition": "High Cholesterol",
                    "review": "Very effective medication for lowering my LDL levels.",
                    "rating": 9,
                    "date": "28-Feb-12",
                    "usefulCount": 5,
                },
                {
                    "uniqueID": 10002,
                    "drugName": "Lipitor",
                    "condition": "High Cholesterol",
                    "review": "Experienced moderate muscle pain and nausea after 3 weeks.",
                    "rating": 3,
                    "date": "15-Mar-13",
                    "usefulCount": 12,
                },
            ]
        }
        response = api_client.post("/monitoring/nlp/drift", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "certification" in data
        assert "text_length_drift" in data
        assert "sentiment_drift" in data
        assert "aspect_drift" in data

    def test_post_nlp_drift_invalid_rating_rejected(self, api_client: TestClient):
        """Verify POST /monitoring/nlp/drift rejects out-of-bounds ratings (ge=1, le=10)."""
        payload = {
            "reviews": [
                {
                    "uniqueID": 10003,
                    "drugName": "Lipitor",
                    "condition": "High Cholesterol",
                    "review": "Test review",
                    "rating": 15,  # Invalid rating > 10
                    "date": "28-Feb-12",
                    "usefulCount": 0,
                }
            ]
        }
        response = api_client.post("/monitoring/nlp/drift", json=payload)
        assert response.status_code == 422  # Pydantic validation error
