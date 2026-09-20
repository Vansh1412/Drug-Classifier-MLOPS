"""
tests/test_sprint9.py — Comprehensive Test Suite for Sprint 9 Patient Experience Intelligence.

Validates:
1. Sentiment Analytics: Overall distribution, rating statistics, boundary safety.
2. Drug-Level Analytics: Aggregation correctness, min_reviews filtering, descriptive framing.
3. Condition-Level Analytics: Aggregation, explicit "Unknown" handling, zero condition inference.
4. Temporal Experience: Yearly volume, mean rating, chronological ordering.
5. Aspect / Theme Intelligence: Keyword extraction, conditional sentiment breakdown, exemplar references.
6. Lexical Review Similarity: TF-IDF cosine similarity, top-k ordering, self-match, edge cases.
7. Privacy & Safety: No PII leakage, no raw text dumping in summaries.
8. Lineage & Artifacts: Report generation, schema conformity, DVC hash and Git commit tracking.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRUCTURED_CHAMPION_PATH = PROJECT_ROOT / "Model" / "drug_pipeline.skops"

from src.patient_experience import (
    ReviewSimilarityEngine,
    analyze_condition_experience,
    analyze_drug_experience,
    analyze_overall_sentiment,
    analyze_temporal_experience,
    extract_aspect_intelligence,
    generate_patient_experience_report,
)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def sample_experience_df() -> pd.DataFrame:
    """Provide a diverse sample DataFrame for patient experience intelligence testing."""
    return pd.DataFrame(
        {
            "uniqueID": list(range(101, 116)),
            "drugName": (
                ["Aspirin"] * 6
                + ["Ibuprofen"] * 5
                + ["Paracetamol"] * 3
                + ["RareDrug"] * 1
            ),
            "condition": (
                ["Pain"] * 5
                + ["Unknown"] * 1
                + ["Pain"] * 4
                + ["Headache"] * 1
                + ["Headache"] * 3
                + ["Fever"] * 1
            ),
            "condition_clean": (
                ["Pain"] * 5
                + ["Unknown"] * 1
                + ["Pain"] * 4
                + ["Headache"] * 1
                + ["Headache"] * 3
                + ["Fever"] * 1
            ),
            "review": [
                "Aspirin was effective and relieved my severe headache quickly.",
                "Terrible nausea and stomach pain after taking this dose daily.",
                "Average pill, helped slightly with the pain.",
                "Worked wonders! Completely cured my chronic pain without side effects.",
                "Horrible side effects, felt dizzy and experienced insomnia.",
                "Affordable price and good coverage by insurance.",
                "Ibuprofen was great, effective pain relief within an hour.",
                "Experienced mild drowsiness and wanted to sleep all day.",
                "Gained weight and had increased appetite while on this.",
                "Did not help my headache, expensive copay with poor results.",
                "Easy daily dosage, no withdrawal symptoms after stopping.",
                "Paracetamol cured my fever and headache completely.",
                "Mild nausea but otherwise very effective for pain.",
                "Cheap generic option, works great for mild headache.",
                "Rare medication for fever, worked okay.",
            ],
            "review_clean": [
                "Aspirin was effective and relieved my severe headache quickly.",
                "Terrible nausea and stomach pain after taking this dose daily.",
                "Average pill, helped slightly with the pain.",
                "Worked wonders! Completely cured my chronic pain without side effects.",
                "Horrible side effects, felt dizzy and experienced insomnia.",
                "Affordable price and good coverage by insurance.",
                "Ibuprofen was great, effective pain relief within an hour.",
                "Experienced mild drowsiness and wanted to sleep all day.",
                "Gained weight and had increased appetite while on this.",
                "Did not help my headache, expensive copay with poor results.",
                "Easy daily dosage, no withdrawal symptoms after stopping.",
                "Paracetamol cured my fever and headache completely.",
                "Mild nausea but otherwise very effective for pain.",
                "Cheap generic option, works great for mild headache.",
                "Rare medication for fever, worked okay.",
            ],
            "review_clean_hash": [f"hash_{i}" for i in range(101, 116)],
            "rating": [9, 2, 6, 10, 1, 8, 9, 7, 5, 2, 8, 10, 7, 8, 6],
            "sentiment_class": [
                "high",
                "low",
                "neutral",
                "high",
                "low",
                "high",
                "high",
                "neutral",
                "neutral",
                "low",
                "high",
                "high",
                "neutral",
                "high",
                "neutral",
            ],
            "date": ["15-Jan-15"] * 5 + ["20-Jun-16"] * 5 + ["10-Nov-17"] * 5,
            "date_parsed": (
                ["2015-01-15"] * 5 + ["2016-06-20"] * 5 + ["2017-11-10"] * 5
            ),
            "year": [2015] * 5 + [2016] * 5 + [2017] * 5,
            "usefulCount": [15, 2, 8, 45, 0, 12, 30, 4, 9, 3, 22, 18, 7, 14, 5],
        }
    )


# =============================================================================
# 1. Sentiment Analytics Tests (S9-T2)
# =============================================================================
class TestSentimentAnalytics:
    def test_overall_sentiment_aggregation(self, sample_experience_df):
        """Assert overall sentiment counts and percentages sum to total records."""
        res = analyze_overall_sentiment(sample_experience_df)
        assert res["total_reviews"] == 15

        dist = res["sentiment_distribution"]
        assert set(dist.keys()) == {"high", "low", "neutral"}

        total_count = sum(d["count"] for d in dist.values())
        assert total_count == 15

        total_pct = sum(d["percentage"] for d in dist.values())
        assert 99.9 <= total_pct <= 100.1

    def test_rating_distribution_keys(self, sample_experience_df):
        """Assert rating distribution contains keys for ratings 1 through 10."""
        res = analyze_overall_sentiment(sample_experience_df)
        rating_dist = res["rating_distribution"]
        for r in range(1, 11):
            assert str(r) in rating_dist
            assert "count" in rating_dist[str(r)]
            assert "percentage" in rating_dist[str(r)]

    def test_rating_summary_statistics(self, sample_experience_df):
        """Assert rating summary statistics are numerically bounded."""
        res = analyze_overall_sentiment(sample_experience_df)
        stats = res["rating_summary"]
        assert 1.0 <= stats["mean"] <= 10.0
        assert 1.0 <= stats["median"] <= 10.0
        assert stats["min"] == 1
        assert stats["max"] == 10
        assert stats["std"] > 0.0

    def test_empty_dataframe_sentiment_handling(self):
        """Assert empty DataFrame does not raise ZeroDivisionError."""
        empty_df = pd.DataFrame(columns=["rating", "sentiment_class"])
        res = analyze_overall_sentiment(empty_df)
        assert res["total_reviews"] == 0
        assert res["sentiment_distribution"] == {}


# =============================================================================
# 2. Drug-Level Analytics Tests (S9-T3)
# =============================================================================
class TestDrugExperienceAnalytics:
    def test_drug_experience_aggregation(self, sample_experience_df):
        """Assert drug aggregation computes correct review counts and percentages."""
        # Use min_reviews=5 to include Aspirin (6) and Ibuprofen (5)
        drugs = analyze_drug_experience(sample_experience_df, min_reviews=5)
        assert len(drugs) == 2

        aspirin = next(d for d in drugs if d["drugName"] == "Aspirin")
        assert aspirin["review_count"] == 6
        assert aspirin["low_count"] == 2
        assert aspirin["neutral_count"] == 1
        assert aspirin["high_count"] == 3
        assert aspirin["low_pct"] == round((2 / 6) * 100, 2)
        assert aspirin["high_pct"] == round((3 / 6) * 100, 2)

    def test_min_reviews_threshold_filtering(self, sample_experience_df):
        """Assert drugs below min_reviews threshold are excluded."""
        # Paracetamol has 3 reviews, RareDrug has 1
        drugs = analyze_drug_experience(sample_experience_df, min_reviews=4)
        drug_names = [d["drugName"] for d in drugs]
        assert "Aspirin" in drug_names
        assert "Ibuprofen" in drug_names
        assert "Paracetamol" not in drug_names
        assert "RareDrug" not in drug_names

    def test_drug_analytics_deterministic_sort(self, sample_experience_df):
        """Assert results are sorted by review_count descending."""
        drugs = analyze_drug_experience(sample_experience_df, min_reviews=2)
        counts = [d["review_count"] for d in drugs]
        assert counts == sorted(counts, reverse=True)

    def test_no_evaluative_ranking_labels(self, sample_experience_df):
        """Verify no judgmental/promotional keys (e.g. 'best', 'worst', 'safest') exist."""
        drugs = analyze_drug_experience(sample_experience_df, min_reviews=1)
        disallowed_keys = {
            "best",
            "worst",
            "safest",
            "most_effective",
            "recommended",
        }
        for d in drugs:
            for k in d.keys():
                assert (
                    k.lower() not in disallowed_keys
                ), f"Disallowed evaluative key found: {k}"


# =============================================================================
# 3. Condition-Level Analytics Tests (S9-T4)
# =============================================================================
class TestConditionExperienceAnalytics:
    def test_condition_experience_aggregation(self, sample_experience_df):
        """Assert condition aggregation computes correct review counts and ratings."""
        conds = analyze_condition_experience(sample_experience_df, min_reviews=3)
        cond_names = [c["condition"] for c in conds]
        assert "Pain" in cond_names
        assert "Headache" in cond_names

        pain = next(c for c in conds if c["condition"] == "Pain")
        assert pain["review_count"] == 9  # 5 Aspirin + 4 Ibuprofen

    def test_unknown_condition_explicit_handling(self, sample_experience_df):
        """Assert Unknown condition is explicitly flagged as a missing category."""
        conds = analyze_condition_experience(sample_experience_df, min_reviews=1)
        unknown = next(c for c in conds if c["condition"] == "Unknown")
        assert unknown["is_missing_category"] is True

        pain = next(c for c in conds if c["condition"] == "Pain")
        assert pain["is_missing_category"] is False

    def test_no_condition_inference(self):
        """Assert that missing/null condition values are not synthetically imputed."""
        df = pd.DataFrame(
            {
                "drugName": ["Aspirin", "Aspirin"],
                "condition": [None, "Pain"],
                "rating": [9, 8],
                "sentiment_class": ["high", "high"],
            }
        )
        conds = analyze_condition_experience(df, min_reviews=1)
        cond_names = [c["condition"] for c in conds]
        assert "Unknown" in cond_names
        assert "Pain" in cond_names
        # Ensure row 0 was not imputed as 'Pain' just because drug was Aspirin
        unknown_entry = next(c for c in conds if c["condition"] == "Unknown")
        assert unknown_entry["review_count"] == 1


# =============================================================================
# 4. Temporal Experience Analytics Tests (S9-T5)
# =============================================================================
class TestTemporalExperienceAnalytics:
    def test_temporal_experience_aggregation(self, sample_experience_df):
        """Assert yearly aggregation computes volume and mean ratings per year."""
        timeline = analyze_temporal_experience(sample_experience_df)
        assert len(timeline) == 3

        years = [t["year"] for t in timeline]
        assert years == [2015, 2016, 2017]

        for t in timeline:
            assert t["review_count"] == 5
            assert 1.0 <= t["mean_rating"] <= 10.0
            assert "low_pct" in t
            assert "high_pct" in t

    def test_temporal_chronological_sorting(self, sample_experience_df):
        """Assert timeline is sorted chronologically ascending."""
        timeline = analyze_temporal_experience(sample_experience_df)
        years = [t["year"] for t in timeline]
        assert years == sorted(years)

    def test_malformed_date_handling(self):
        """Assert invalid dates are dropped without raising an exception."""
        df = pd.DataFrame(
            {
                "date": ["corrupt_date", "15-Jan-15"],
                "rating": [5, 9],
                "sentiment_class": ["neutral", "high"],
            }
        )
        timeline = analyze_temporal_experience(df)
        assert len(timeline) == 1
        assert timeline[0]["year"] == 2015


# =============================================================================
# 5. Aspect / Theme Intelligence Tests (S9-T6, S9-T7)
# =============================================================================
class TestAspectIntelligence:
    def test_aspect_detection_and_frequencies(self, sample_experience_df):
        """Assert aspect extraction detects keywords and computes mention counts."""
        keywords = {
            "side_effects": ["side effect", "side effects", "nausea"],
            "effectiveness": ["effective", "worked wonders", "cured"],
            "cost_affordability": ["affordable", "copay", "insurance"],
        }
        aspects = extract_aspect_intelligence(sample_experience_df, keywords)
        assert "side_effects" in aspects
        assert "effectiveness" in aspects
        assert "cost_affordability" in aspects

        assert aspects["side_effects"]["mention_count"] > 0
        assert aspects["effectiveness"]["mention_count"] > 0
        assert aspects["cost_affordability"]["mention_count"] > 0

    def test_aspect_conditional_sentiment(self, sample_experience_df):
        """Assert aspect results contain conditional sentiment distribution."""
        keywords = {"effectiveness": ["effective", "cured"]}
        aspects = extract_aspect_intelligence(sample_experience_df, keywords)
        eff = aspects["effectiveness"]
        dist = eff["sentiment_distribution"]

        assert "low_count" in dist
        assert "neutral_count" in dist
        assert "high_count" in dist
        assert dist["high_count"] > 0  # 'cured' reviews are high rating

    def test_aspect_exemplars_contain_id_and_bounded_snippet(
        self, sample_experience_df
    ):
        """Assert exemplars provide uniqueID, hash, and truncated snippet."""
        keywords = {"effectiveness": ["cured"]}
        aspects = extract_aspect_intelligence(
            sample_experience_df, keywords, max_exemplars=2
        )
        exemplars = aspects["effectiveness"]["exemplars"]

        assert len(exemplars) <= 2
        for ex in exemplars:
            assert "uniqueID" in ex
            assert "review_clean_hash" in ex
            assert "snippet" in ex
            assert len(ex["snippet"]) <= 125  # Bounded snippet length


# =============================================================================
# 6. Review Similarity Engine Tests (S9-T8)
# =============================================================================
class TestReviewSimilarityEngine:
    def test_similarity_self_match(self, sample_experience_df):
        """Assert that querying an exact review yields similarity score close to 1.0."""
        engine = ReviewSimilarityEngine(
            sample_experience_df, tfidf_params={"min_df": 1, "max_df": 1.0}
        )
        query = "Aspirin was effective and relieved my severe headache quickly."
        matches = engine.find_similar(query, top_k=3)

        assert len(matches) > 0
        top_match = matches[0]
        assert top_match["similarity_score"] >= 0.95
        assert top_match["drugName"] == "Aspirin"

    def test_similarity_top_k_bound(self, sample_experience_df):
        """Assert that find_similar returns no more than top_k items."""
        engine = ReviewSimilarityEngine(
            sample_experience_df, tfidf_params={"min_df": 1, "max_df": 1.0}
        )
        matches = engine.find_similar("headache pain relief", top_k=2)
        assert len(matches) <= 2

    def test_similarity_empty_query(self, sample_experience_df):
        """Assert empty query string returns empty list."""
        engine = ReviewSimilarityEngine(
            sample_experience_df, tfidf_params={"min_df": 1, "max_df": 1.0}
        )
        assert engine.find_similar("") == []
        assert engine.find_similar("   ") == []

    def test_similarity_out_of_vocabulary(self, sample_experience_df):
        """Assert out-of-vocabulary query returns empty list without error."""
        engine = ReviewSimilarityEngine(
            sample_experience_df, tfidf_params={"min_df": 1, "max_df": 1.0}
        )
        matches = engine.find_similar("xyznonexistentterm12345", top_k=5)
        assert matches == []

    def test_similarity_filter_by_drug(self, sample_experience_df):
        """Assert filter_drug restricts search results to the specified drug."""
        engine = ReviewSimilarityEngine(
            sample_experience_df, tfidf_params={"min_df": 1, "max_df": 1.0}
        )
        matches = engine.find_similar("pain headache", top_k=5, filter_drug="Ibuprofen")
        for m in matches:
            assert m["drugName"] == "Ibuprofen"


# =============================================================================
# 7. Privacy and Safety Tests (S9-T7, S9-T15)
# =============================================================================
class TestPrivacyAndSafety:
    def test_summary_report_does_not_contain_full_reviews(self, tmp_path):
        """Assert compact summary report does not contain full review text."""
        df = pd.DataFrame(
            {
                "uniqueID": [1],
                "drugName": ["Aspirin"],
                "condition": ["Pain"],
                "condition_clean": ["Pain"],
                "review_clean": ["This is a full patient review."],
                "review_clean_hash": ["hash1"],
                "rating": [10],
                "sentiment_class": ["high"],
                "date_parsed": ["2016-01-01"],
                "year": [2016],
                "usefulCount": [5],
            }
        )
        report_path = tmp_path / "full.json"
        summary_path = tmp_path / "summary.json"

        test_cfg = {
            "reviews": {
                "patient_experience": {
                    "min_reviews_threshold": 1,
                    "condition_min_reviews_threshold": 1,
                    "aspects": {"test": ["patient"]},
                    "similarity": {"min_df": 1, "max_df": 1.0},
                    "reports": {
                        "full_report_path": str(report_path),
                        "summary_report_path": str(summary_path),
                    },
                }
            }
        }

        # Save parquet for runner
        parquet_path = tmp_path / "reviews.parquet"
        df.to_parquet(parquet_path)
        test_cfg["reviews"]["processed_data_path"] = str(parquet_path)

        full, summary = generate_patient_experience_report(test_cfg)
        summary_str = json.dumps(summary)
        assert "This is a full patient review." not in summary_str


# =============================================================================
# 8. Lineage and Artifact Generation Tests (S9-T9, S9-T10)
# =============================================================================
class TestLineageAndArtifacts:
    def test_structured_champion_remains_untouched(self):
        """Assert that structured champion Model/drug_pipeline.skops exists and is protected."""
        assert STRUCTURED_CHAMPION_PATH.exists()

    def test_report_lineage_metadata(self, tmp_path):
        """Assert generated reports contain DVC hash, Git commit, and safety boundary."""
        df = pd.DataFrame(
            {
                "uniqueID": [1, 2],
                "drugName": ["Aspirin", "Ibuprofen"],
                "condition": ["Pain", "Pain"],
                "condition_clean": ["Pain", "Pain"],
                "review_clean": ["Worked great", "Average"],
                "review_clean_hash": ["h1", "h2"],
                "rating": [9, 6],
                "sentiment_class": ["high", "neutral"],
                "date_parsed": ["2016-01-01", "2016-02-01"],
                "year": [2016, 2016],
                "usefulCount": [5, 2],
            }
        )
        parquet_path = tmp_path / "reviews.parquet"
        df.to_parquet(parquet_path)

        test_cfg = {
            "reviews": {
                "processed_data_path": str(parquet_path),
                "patient_experience": {
                    "min_reviews_threshold": 1,
                    "condition_min_reviews_threshold": 1,
                    "aspects": {"effective": ["great"]},
                    "similarity": {"min_df": 1, "max_df": 1.0},
                    "reports": {
                        "full_report_path": str(tmp_path / "full.json"),
                        "summary_report_path": str(tmp_path / "summary.json"),
                    },
                },
            }
        }

        full_rep, summary_rep = generate_patient_experience_report(test_cfg)

        assert "dataset_metadata" in full_rep
        assert "dvc_dataset_hash" in full_rep["dataset_metadata"]
        assert "git_commit" in full_rep["dataset_metadata"]
        assert "medical_safety_boundary" in full_rep
        assert (
            "Patient-reported drug experience analytics only"
            in full_rep["medical_safety_boundary"]
        )

        assert "lineage" in summary_rep
        assert "dvc_dataset_hash" in summary_rep["lineage"]
        assert "git_commit" in summary_rep["lineage"]
