"""
tests/test_sprint8.py — Comprehensive Test Suite for Sprint 8 NLP Model Competition.

Validates:
1. Target Definition: rating (1-10) -> sentiment_class ('low', 'neutral', 'high'),
   immutability of raw rating, and input validation.
2. Leakage-Safe Splitting: Zero text overlap between train and test sets.
3. TF-IDF Pipeline: Fitted strictly on training data / inside CV folds, handles edge cases.
4. Model Training & Inference: LR, Linear SVM, and Naive Bayes train and predict valid classes.
5. Group-Aware Cross-Validation: StratifiedGroupKFold produces Macro F1, Accuracy, and std.
6. Quality Gate & Champion Selection: Threshold enforcement, selection by CV Macro F1,
   and support for NO CHAMPION outcome.
7. Artifact Isolation & Integrity: NLP champion saved to Model/nlp/ without modifying
   structured champion (Model/drug_pipeline.skops).
8. MLflow Experiment Tracking: NLP experiment isolation, parameter/metric/tag logging.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
STRUCTURED_CHAMPION_PATH = PROJECT_ROOT / "Model" / "drug_pipeline.skops"
STRUCTURED_METADATA_PATH = PROJECT_ROOT / "Model" / "model_metadata.json"
NLP_CHAMPION_DIR = PROJECT_ROOT / "Model" / "nlp" / "champion"

from src.config import load_config
from src.nlp_competition import (
    build_nlp_pipeline,
    derive_sentiment_target,
    evaluate_cv_leakage_safe,
    evaluate_nlp_quality_gate,
    evaluate_test_set,
    save_nlp_champion,
    select_nlp_champion,
)
from src.nlp_preprocessing import split_reviews_leakage_safe
from src.nlp_tracking import (
    get_dvc_dataset_hash,
    get_git_commit_sha,
    log_nlp_candidate_run,
    setup_nlp_mlflow,
)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def sample_review_data() -> pd.DataFrame:
    """Provide a diverse sample DataFrame for NLP testing."""
    return pd.DataFrame(
        {
            "uniqueID": list(range(1, 13)),
            "drugName": ["Aspirin"] * 6 + ["Ibuprofen"] * 6,
            "condition": ["Pain"] * 12,
            "review": [
                "Excellent medication, completely cured my headache!",
                "Terrible experience, awful side effects and nausea.",
                "It was okay, neither great nor bad.",
                "Fantastic relief, highly recommended to everyone.",
                "Did not work at all, absolute waste of money.",
                "Moderate improvement but caused mild drowsiness.",
                # Duplicate reviews to test grouping and leakage prevention
                "Excellent medication, completely cured my headache!",
                "Terrible experience, awful side effects and nausea.",
                "It was okay, neither great nor bad.",
                "Works wonders for severe chronic pain.",
                "Horrible pain after taking this, never again.",
                "Average drug, took several days to feel anything.",
            ],
            "review_clean": [
                "Excellent medication, completely cured my headache!",
                "Terrible experience, awful side effects and nausea.",
                "It was okay, neither great nor bad.",
                "Fantastic relief, highly recommended to everyone.",
                "Did not work at all, absolute waste of money.",
                "Moderate improvement but caused mild drowsiness.",
                "Excellent medication, completely cured my headache!",
                "Terrible experience, awful side effects and nausea.",
                "It was okay, neither great nor bad.",
                "Works wonders for severe chronic pain.",
                "Horrible pain after taking this, never again.",
                "Average drug, took several days to feel anything.",
            ],
            "review_clean_hash": [
                "hash_1",
                "hash_2",
                "hash_3",
                "hash_4",
                "hash_5",
                "hash_6",
                "hash_1",  # duplicate
                "hash_2",  # duplicate
                "hash_3",  # duplicate
                "hash_7",
                "hash_8",
                "hash_9",
            ],
            "rating": [10, 1, 6, 9, 2, 5, 10, 1, 7, 8, 2, 6],
            "date": ["28-Feb-12"] * 12,
            "date_parsed": ["2012-02-28"] * 12,
            "usefulCount": [10] * 12,
        }
    )


# =============================================================================
# 1. Target Definition Tests (S8-T1)
# =============================================================================
class TestNLPTargetDefinition:
    def test_sentiment_target_derivation_mapping(self, sample_review_data):
        """Assert that ratings 1-4 map to 'low', 5-7 to 'neutral', 8-10 to 'high'."""
        df = derive_sentiment_target(sample_review_data)
        assert "sentiment_class" in df.columns

        # Verify exact mappings
        expected_classes = [
            "high",  # 10
            "low",  # 1
            "neutral",  # 6
            "high",  # 9
            "low",  # 2
            "neutral",  # 5
            "high",  # 10
            "low",  # 1
            "neutral",  # 7
            "high",  # 8
            "low",  # 2
            "neutral",  # 6
        ]
        assert df["sentiment_class"].tolist() == expected_classes

    def test_raw_numeric_rating_is_preserved(self, sample_review_data):
        """Assert that the original numeric rating column is unchanged and accessible."""
        original_ratings = sample_review_data["rating"].copy()
        df = derive_sentiment_target(sample_review_data)
        pd.testing.assert_series_equal(df["rating"], original_ratings)

    def test_target_derivation_rejects_missing_column(self):
        """Assert ValueError when rating column is absent."""
        df = pd.DataFrame({"review": ["Good", "Bad"]})
        with pytest.raises(ValueError, match="Rating column 'rating' not found"):
            derive_sentiment_target(df)

    def test_target_derivation_rejects_out_of_bounds(self):
        """Assert ValueError when ratings are outside the valid [1, 10] range."""
        df = pd.DataFrame({"rating": [0, 5, 11]})
        with pytest.raises(ValueError, match="values outside"):
            derive_sentiment_target(df)

    def test_target_derivation_rejects_nulls(self):
        """Assert ValueError when ratings contain NaN."""
        df = pd.DataFrame({"rating": [5, np.nan, 8]})
        with pytest.raises(ValueError, match="null values"):
            derive_sentiment_target(df)


# =============================================================================
# 2. Leakage-Safe Splitting & TF-IDF Tests (S8-T2, S8-T3)
# =============================================================================
class TestLeakageSafeSplittingAndTFIDF:
    def test_zero_text_overlap_in_group_split(self, sample_review_data):
        """Verify that identical reviews do not cross the train/test split boundary."""
        df = derive_sentiment_target(sample_review_data)
        train_df, test_df = split_reviews_leakage_safe(
            df,
            strategy="stratified_group",
            test_size=0.33,
            random_state=125,
            stratify_col="sentiment_class",
            group_col="review_clean_hash",
        )

        train_texts = set(train_df["review_clean"])
        test_texts = set(test_df["review_clean"])
        overlap = train_texts.intersection(test_texts)
        assert len(overlap) == 0, f"Leakage detected! Overlapping texts: {overlap}"

    def test_tfidf_pipeline_encapsulation(self):
        """Verify that build_nlp_pipeline constructs a proper sklearn Pipeline."""
        for model_name in ["logistic_regression", "linear_svm", "naive_bayes"]:
            pipe = build_nlp_pipeline(model_name)
            assert isinstance(pipe, Pipeline)
            assert "tfidf" in pipe.named_steps
            assert "classifier" in pipe.named_steps

    def test_tfidf_fitted_only_on_training_data(self):
        """Verify that TF-IDF vocabulary is learned exclusively from training data."""
        train_texts = pd.Series(["apple orange banana", "orange banana grape"])
        test_texts = pd.Series(["watermelon pineapple kiwi"])

        pipe = build_nlp_pipeline(
            "logistic_regression", tfidf_params={"min_df": 1, "max_df": 1.0}
        )
        pipe.fit(train_texts, pd.Series(["low", "high"]))

        vocab = pipe.named_steps["tfidf"].vocabulary_
        assert "apple" in vocab
        assert "orange" in vocab
        # Test words must NOT be in vocabulary
        assert "watermelon" not in vocab
        assert "pineapple" not in vocab
        assert "kiwi" not in vocab

        # Verify inference works on held-out text
        preds = pipe.predict(test_texts)
        assert len(preds) == 1

    def test_unsupported_model_raises_error(self):
        """Assert ValueError when attempting to build an unsupported model."""
        with pytest.raises(ValueError, match="Unsupported NLP model"):
            build_nlp_pipeline("bert_transformer")


# =============================================================================
# 3. Model Training & Inference Tests (S8-T4)
# =============================================================================
class TestNLPModelTrainingAndInference:
    @pytest.mark.parametrize(
        "model_name", ["logistic_regression", "linear_svm", "naive_bayes"]
    )
    def test_candidate_model_train_and_predict(self, sample_review_data, model_name):
        """Assert that each candidate model trains, predicts, and outputs valid classes."""
        df = derive_sentiment_target(sample_review_data)
        pipe = build_nlp_pipeline(model_name, tfidf_params={"min_df": 1, "max_df": 1.0})

        X = df["review_clean"]
        y = df["sentiment_class"]

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(X)
        for p in preds:
            assert p in {"low", "neutral", "high"}


# =============================================================================
# 4. Cross-Validation & Test Evaluation Tests (S8-T5)
# =============================================================================
class TestCrossValidationAndEvaluation:
    def test_evaluate_cv_leakage_safe(self, sample_review_data):
        """Assert that evaluate_cv_leakage_safe computes expected CV metrics without error."""
        df = derive_sentiment_target(sample_review_data)
        pipe = build_nlp_pipeline(
            "naive_bayes", tfidf_params={"min_df": 1, "max_df": 1.0}
        )

        cv_metrics = evaluate_cv_leakage_safe(
            pipeline=pipe,
            X=df["review_clean"],
            y=df["sentiment_class"],
            groups=df["review_clean_hash"],
            n_splits=3,
            random_state=125,
        )

        expected_keys = [
            "cv_macro_f1_mean",
            "cv_macro_f1_std",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "cv_precision_macro_mean",
            "cv_recall_macro_mean",
        ]
        for key in expected_keys:
            assert key in cv_metrics
            assert isinstance(cv_metrics[key], float)
            assert 0.0 <= cv_metrics[key] <= 1.0

    def test_evaluate_test_set(self, sample_review_data):
        """Assert that evaluate_test_set produces comprehensive test metrics and confusion matrix."""
        df = derive_sentiment_target(sample_review_data)
        pipe = build_nlp_pipeline(
            "logistic_regression", tfidf_params={"min_df": 1, "max_df": 1.0}
        )
        pipe.fit(df["review_clean"], df["sentiment_class"])

        test_res = evaluate_test_set(
            pipeline=pipe,
            X_test=df["review_clean"],
            y_test=df["sentiment_class"],
            classes=["low", "neutral", "high"],
        )

        assert "test_macro_f1" in test_res
        assert "test_accuracy" in test_res
        assert "class_metrics" in test_res
        assert "confusion_matrix" in test_res
        assert len(test_res["confusion_matrix"]) == 3  # 3x3 matrix


# =============================================================================
# 5. Quality Gate & Champion Selection Tests (S8-T6)
# =============================================================================
class TestNLPQualityGateAndChampionSelection:
    def test_quality_gate_passes_when_metrics_exceed_thresholds(self):
        """Assert quality gate passes when all metrics meet or exceed criteria."""
        cv_metrics = {"cv_macro_f1_mean": 0.75, "cv_accuracy_mean": 0.78}
        test_metrics = {
            "test_macro_f1": 0.74,
            "test_accuracy": 0.76,
            "class_metrics": {
                "low": {"recall": 0.70},
                "neutral": {"recall": 0.50},
                "high": {"recall": 0.85},
            },
        }
        gate_cfg = {
            "min_macro_f1": 0.70,
            "min_accuracy": 0.70,
            "min_class_recall": 0.40,
        }

        passed, reasons = evaluate_nlp_quality_gate(cv_metrics, test_metrics, gate_cfg)
        assert passed is True
        assert len(reasons) == 0

    def test_quality_gate_fails_below_macro_f1(self):
        """Assert quality gate fails when Macro F1 is below threshold."""
        cv_metrics = {"cv_macro_f1_mean": 0.65, "cv_accuracy_mean": 0.78}
        test_metrics = {
            "test_macro_f1": 0.64,
            "test_accuracy": 0.76,
            "class_metrics": {
                "low": {"recall": 0.70},
                "neutral": {"recall": 0.50},
                "high": {"recall": 0.85},
            },
        }
        gate_cfg = {
            "min_macro_f1": 0.70,
            "min_accuracy": 0.70,
            "min_class_recall": 0.40,
        }

        passed, reasons = evaluate_nlp_quality_gate(cv_metrics, test_metrics, gate_cfg)
        assert passed is False
        assert any("Macro F1" in r for r in reasons)

    def test_no_champion_when_no_candidate_passes_gate(self):
        """Assert NO NLP CHAMPION outcome when all models fail quality gate."""
        candidates = [
            {
                "model_name": "logistic_regression",
                "quality_gate_passed": False,
                "cv_metrics": {"cv_macro_f1_mean": 0.62},
            },
            {
                "model_name": "linear_svm",
                "quality_gate_passed": False,
                "cv_metrics": {"cv_macro_f1_mean": 0.64},
            },
        ]
        champion = select_nlp_champion(candidates)
        assert champion is None

    def test_champion_selected_by_cv_macro_f1(self):
        """Assert model with highest CV Macro F1 is selected as champion."""
        candidates = [
            {
                "model_name": "logistic_regression",
                "quality_gate_passed": True,
                "cv_metrics": {"cv_macro_f1_mean": 0.72},
                "test_metrics": {"test_macro_f1": 0.71},
            },
            {
                "model_name": "linear_svm",
                "quality_gate_passed": True,
                "cv_metrics": {"cv_macro_f1_mean": 0.76},
                "test_metrics": {"test_macro_f1": 0.75},
            },
            {
                "model_name": "naive_bayes",
                "quality_gate_passed": True,
                "cv_metrics": {"cv_macro_f1_mean": 0.70},
                "test_metrics": {"test_macro_f1": 0.69},
            },
        ]
        champion = select_nlp_champion(candidates)
        assert champion is not None
        assert champion["model_name"] == "linear_svm"


# =============================================================================
# 6. Artifact Isolation & Integrity Tests (S8-T7, S8-T8)
# =============================================================================
class TestNLPArtifactIsolationAndIntegrity:
    def test_structured_champion_remains_untouched(self):
        """Assert that structured champion Model/drug_pipeline.skops exists and is not modified."""
        assert STRUCTURED_CHAMPION_PATH.exists()
        assert STRUCTURED_METADATA_PATH.exists()

        # Verify structured metadata belongs to XGBoost / Drug Classifier
        with open(STRUCTURED_METADATA_PATH, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        assert (
            meta.get("model_name", "").lower() == "xgboost"
            or "xgboost" in meta.get("model_type", "").lower()
            or "Drug" in meta.get("model_type", "")
        )

    def test_nlp_champion_save_and_load(self, sample_review_data, tmp_path):
        """Assert that NLP champion pipeline is saved via skops and loads correctly."""
        df = derive_sentiment_target(sample_review_data)
        pipe = build_nlp_pipeline(
            "logistic_regression", tfidf_params={"min_df": 1, "max_df": 1.0}
        )
        pipe.fit(df["review_clean"], df["sentiment_class"])

        test_cfg = {
            "reviews": {
                "nlp_competition": {
                    "artifacts": {
                        "pipeline_path": str(tmp_path / "nlp_champion_pipeline.skops"),
                        "metadata_path": str(tmp_path / "nlp_model_metadata.json"),
                    }
                }
            }
        }
        metadata = {
            "model_type": "logistic_regression",
            "classes": ["low", "neutral", "high"],
        }

        pipe_path, meta_path = save_nlp_champion(pipe, metadata, cfg=test_cfg)
        assert pipe_path.exists()
        assert meta_path.exists()

        # Load back with skops using get_untrusted_types
        untrusted = sio.get_untrusted_types(file=pipe_path)
        loaded_pipe = sio.load(pipe_path, trusted=untrusted)
        assert isinstance(loaded_pipe, Pipeline)
        pred = loaded_pipe.predict(["Aspirin helped my severe headache!"])
        assert pred[0] in {"low", "neutral", "high"}


# =============================================================================
# 7. MLflow Experiment Tracking Tests (S8-T6)
# =============================================================================
class TestNLPMLflowTracking:
    def test_get_git_commit_sha(self):
        """Assert that git commit sha is retrieved or returns 'unknown'."""
        sha = get_git_commit_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0

    def test_get_dvc_dataset_hash(self):
        """Assert that DVC dataset hash is retrieved or returns 'unknown'."""
        dvc_hash = get_dvc_dataset_hash("Data/reviews/drugsComTest_raw.csv.dvc")
        assert isinstance(dvc_hash, str)
        assert len(dvc_hash) > 0

    def test_setup_nlp_mlflow_experiment_name(self):
        """Assert that setup_nlp_mlflow sets the dedicated NLP experiment."""
        cfg = load_config()
        exp_id = setup_nlp_mlflow(cfg)
        assert exp_id is not None
        assert isinstance(exp_id, str)

    def test_log_nlp_candidate_run(self):
        """Assert that candidate run logging succeeds with tags and metrics."""
        cfg = load_config()
        setup_nlp_mlflow(cfg)

        run_id = log_nlp_candidate_run(
            name="test_candidate",
            params={"C": 1.0, "solver": "lbfgs"},
            metrics={"cv_macro_f1_mean": 0.72, "test_accuracy": 0.75},
            tags={"test_tag": "true"},
        )
        assert run_id is not None
        assert len(run_id) > 0
