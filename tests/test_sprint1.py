"""
Sprint 1 Test Suite — MedMLOps-X
Covers:
  - Data validation (positive & negative cases)
  - Deterministic splitting
  - Metrics JSON schema & persistence
  - Model registry & candidate instantiation
  - Champion selection policy & quality gate rejection
  - Pipeline reproducibility
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors as pa_err
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.champion_selection import (
    NoValidChampionError,
    format_leaderboard,
    select_champion,
)
from src.config import load_config
from src.data_validation import load_and_validate, validate_dataset
from src.evaluate import train_and_evaluate
from src.model_registry import build_model_candidates
from src.preprocessing import build_preprocessor


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def raw_df(config):
    return pd.read_csv(config["data"]["data_path"])


# =============================================================================
# 1. Data Validation Tests (Task S1-T2)
# =============================================================================
class TestDataValidation:
    def test_valid_dataset_passes(self, config):
        """Assert that the production Data/drug.csv passes validation cleanly."""
        df = load_and_validate(config["data"]["data_path"], config)
        assert df is not None
        assert len(df) == 200
        assert set(df.columns) == set(config["data"]["columns"])

    def test_missing_required_column_fails(self, config, raw_df):
        """Assert that omitting a required column raises a schema error."""
        df_invalid = raw_df.drop(columns=["Age"])
        with pytest.raises((pa_err.SchemaError, pa_err.SchemaErrors)):
            validate_dataset(df_invalid, config)

    def test_extra_column_fails(self, config, raw_df):
        """Assert that extra unexpected columns are rejected (strict schema)."""
        df_invalid = raw_df.copy()
        df_invalid["UnexpectedColumn"] = 123
        with pytest.raises((pa_err.SchemaError, pa_err.SchemaErrors)):
            validate_dataset(df_invalid, config)

    def test_invalid_datatype_fails(self, config, raw_df):
        """Assert that wrong data types (e.g. string for numeric Age) are rejected."""
        df_invalid = raw_df.copy()
        df_invalid["Age"] = df_invalid["Age"].astype(str)
        with pytest.raises((pa_err.SchemaError, pa_err.SchemaErrors)):
            validate_dataset(df_invalid, config)

    def test_invalid_category_fails(self, config, raw_df):
        """Assert that out-of-domain categorical values are rejected."""
        df_invalid = raw_df.copy()
        df_invalid.loc[0, "BP"] = "EXTREME_HIGH"
        with pytest.raises((pa_err.SchemaError, pa_err.SchemaErrors)):
            validate_dataset(df_invalid, config)

    def test_invalid_numeric_range_fails(self, config, raw_df):
        """Assert that values exceeding biological boundaries are rejected."""
        df_invalid = raw_df.copy()
        df_invalid.loc[0, "Age"] = 150  # Max allowed is 74
        with pytest.raises((pa_err.SchemaError, pa_err.SchemaErrors)):
            validate_dataset(df_invalid, config)

    def test_nan_or_infinite_value_fails(self, config, raw_df):
        """Assert that infinite or NaN values in numerical columns are rejected."""
        df_invalid = raw_df.copy()
        df_invalid.loc[0, "Na_to_K"] = np.nan
        with pytest.raises((pa_err.SchemaError, pa_err.SchemaErrors)):
            validate_dataset(df_invalid, config)


# =============================================================================
# 2. Deterministic Splitting Tests (Task S1-T1)
# =============================================================================
class TestDeterministicSplitting:
    def test_two_consecutive_runs_produce_identical_splits(self, config, raw_df):
        """Assert that identical config random states produce 100% identical splits."""
        shuffle_seed = config["data"]["shuffle_random_state"]
        split_seed = config["data"]["split_random_state"]
        test_size = config["data"]["test_size"]
        target = config["data"]["target_column"]

        # Run 1
        df1 = raw_df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
        X1 = df1.drop(target, axis=1).values
        y1 = df1[target].values
        X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
            X1, y1, test_size=test_size, random_state=split_seed, stratify=y1
        )

        # Run 2
        df2 = raw_df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
        X2 = df2.drop(target, axis=1).values
        y2 = df2[target].values
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
            X2, y2, test_size=test_size, random_state=split_seed, stratify=y2
        )

        assert np.array_equal(X_tr1, X_tr2)
        assert np.array_equal(X_te1, X_te2)
        assert np.array_equal(y_tr1, y_tr2)
        assert np.array_equal(y_te1, y_te2)

    def test_different_seed_produces_different_split(self, config, raw_df):
        """Assert that changing the split seed changes the resulting split."""
        target = config["data"]["target_column"]
        X = raw_df.drop(target, axis=1).values
        y = raw_df[target].values

        X_tr1, _, _, _ = train_test_split(
            X, y, test_size=0.3, random_state=125, stratify=y
        )
        X_tr2, _, _, _ = train_test_split(
            X, y, test_size=0.3, random_state=999, stratify=y
        )

        assert not np.array_equal(X_tr1, X_tr2)


# =============================================================================
# 3. Model Registry & Label Contract Tests (Tasks S1-T4, S1-T3)
# =============================================================================
class TestModelRegistry:
    def test_candidate_instantiation(self, config):
        """Assert that all candidates defined in config can be instantiated."""
        candidates = build_model_candidates(config)
        expected = set(config["models"].keys())
        assert set(candidates.keys()) == expected
        for name, estimator in candidates.items():
            assert hasattr(estimator, "fit")
            assert hasattr(estimator, "predict")

    def test_unknown_model_class_raises_key_error(self, config):
        """Assert that unmapped model classes raise KeyError."""
        bad_config = {
            "models": {"unknown": {"class": "NonExistentModel", "params": {}}}
        }
        with pytest.raises(KeyError):
            build_model_candidates(bad_config)

    def test_all_candidates_return_string_labels(self, config, raw_df):
        """Assert that every candidate model directly outputs string drug labels."""
        candidates = build_model_candidates(config)
        target = config["data"]["target_column"]
        X = raw_df.drop(target, axis=1).values
        y = raw_df[target].values
        valid_drugs = set(config["validation"]["valid_drug_classes"])

        sample = [[30, "M", "HIGH", "NORMAL", 15.4]]

        for name, model in candidates.items():
            pipeline = Pipeline(
                [
                    ("preprocessor", build_preprocessor(config)),
                    ("model", model),
                ]
            )
            pipeline.fit(X, y)
            pred = pipeline.predict(sample)

            assert len(pred) == 1
            assert isinstance(
                pred[0], str
            ), f"{name} returned non-string: {type(pred[0])}"
            assert (
                pred[0] in valid_drugs
            ), f"{name} returned unexpected label: {pred[0]}"


# =============================================================================
# 4. Champion Selection & Quality Gate Tests (Task S1-T5)
# =============================================================================
class TestChampionSelection:
    def test_champion_selected_by_primary_metric(self, config):
        """Assert that candidate with highest primary metric among passing models is champion."""
        results = [
            {
                "model_name": "modA",
                "accuracy": 0.85,
                "macro_f1": 0.80,
                "macro_recall": 0.75,
            },
            {
                "model_name": "modB",
                "accuracy": 0.90,
                "macro_f1": 0.95,
                "macro_recall": 0.88,
            },
            {
                "model_name": "modC",
                "accuracy": 0.88,
                "macro_f1": 0.86,
                "macro_recall": 0.80,
            },
        ]
        champion, challengers = select_champion(results, config)
        assert champion["model_name"] == "modB"
        assert len(challengers) == 2
        assert challengers[0]["model_name"] == "modC"

    def test_quality_gate_rejects_failing_models(self, config):
        """Assert that models below minimum threshold are excluded even if accuracy is high."""
        results = [
            # High accuracy but fails macro_f1 threshold (min 0.75)
            {
                "model_name": "failing_model",
                "accuracy": 0.95,
                "macro_f1": 0.60,
                "macro_recall": 0.80,
            },
            # Passes all thresholds
            {
                "model_name": "passing_model",
                "accuracy": 0.85,
                "macro_f1": 0.82,
                "macro_recall": 0.75,
            },
        ]
        champion, challengers = select_champion(results, config)
        assert champion["model_name"] == "passing_model"
        assert len(challengers) == 0

    def test_no_valid_champion_raises_error(self, config):
        """Assert that NoValidChampionError is raised if all models fail the quality gate."""
        results = [
            {
                "model_name": "bad1",
                "accuracy": 0.50,
                "macro_f1": 0.40,
                "macro_recall": 0.40,
            },
            {
                "model_name": "bad2",
                "accuracy": 0.60,
                "macro_f1": 0.55,
                "macro_recall": 0.50,
            },
        ]
        with pytest.raises(NoValidChampionError) as excinfo:
            select_champion(results, config)
        assert "NO VALID CHAMPION" in str(excinfo.value)

    def test_leaderboard_formatting(self):
        """Assert that format_leaderboard creates a readable summary."""
        champ = {
            "model_name": "m1",
            "accuracy": 0.95,
            "macro_f1": 0.92,
            "macro_recall": 0.90,
        }
        chal = [
            {
                "model_name": "m2",
                "accuracy": 0.85,
                "macro_f1": 0.80,
                "macro_recall": 0.78,
            }
        ]
        failing = [
            {
                "model_name": "m3",
                "accuracy": 0.50,
                "macro_f1": 0.40,
                "macro_recall": 0.40,
            }
        ]

        board = format_leaderboard(champ, chal, failing)
        assert "[CHAMPION] m1" in board
        assert "#2 CHALLENGER: m2" in board
        assert "[FAILED QUALITY GATE]:" in board


# =============================================================================
# 5. Metrics JSON Persistence Tests (Task S1-T3)
# =============================================================================
class TestMetricsPersistence:
    def test_metrics_json_structure_and_types(self, config):
        """Assert that Results/metrics.json is valid and conforms to expected schema."""
        metrics_path = Path(config["results"]["metrics_json"])
        assert metrics_path.exists(), f"Metrics JSON missing at {metrics_path}"

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "run_timestamp" in data
        assert "total_runtime_seconds" in data
        assert "dataset" in data
        assert "champion" in data
        assert "all_candidates" in data

        # Check champion metrics
        champ = data["champion"]
        for key in [
            "model_name",
            "accuracy",
            "macro_f1",
            "macro_recall",
            "quality_gate",
        ]:
            assert key in champ, f"Champion metric missing: {key}"

        # Check all candidates list
        assert len(data["all_candidates"]) >= 4
        model_names = [c["model_name"] for c in data["all_candidates"]]
        assert "xgboost" in model_names
        assert "random_forest" in model_names
        assert "extra_trees" in model_names
        assert "logistic_regression" in model_names

    def test_model_metadata_json_exists(self, config):
        """Assert that Model/model_metadata.json is generated with model details."""
        meta_path = Path(config["champion"]["metadata_path"])
        assert meta_path.exists(), f"Model metadata missing at {meta_path}"

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert "model_name" in meta
        assert "model_class" in meta
        assert "artifact_path" in meta
        assert "accuracy" in meta
        assert "macro_f1" in meta


# =============================================================================
# 6. Reproducibility Test (Task S1-T7)
# =============================================================================
class TestReproducibility:
    def test_pipeline_produces_identical_evaluation(self, config, raw_df):
        """Assert that training two pipelines with identical seed produces exact same metrics."""
        shuffle_seed = config["data"]["shuffle_random_state"]
        split_seed = config["data"]["split_random_state"]
        target = config["data"]["target_column"]

        def run_pipeline():
            df = raw_df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
            X = df.drop(target, axis=1).values
            y = df[target].values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=split_seed, stratify=y
            )
            model_cfg = config["models"]["random_forest"]
            candidates = build_model_candidates(config)
            pipe = Pipeline(
                [
                    ("preprocessor", build_preprocessor(config)),
                    ("model", candidates["random_forest"]),
                ]
            )
            return train_and_evaluate(
                pipe,
                X_train,
                y_train,
                X_test,
                y_test,
                config,
                "random_forest",
                model_cfg["params"],
            )

        res1 = run_pipeline()
        res2 = run_pipeline()

        assert res1["accuracy"] == res2["accuracy"]
        assert res1["macro_f1"] == res2["macro_f1"]
        assert res1["macro_recall"] == res2["macro_recall"]
        assert res1["cv_macro_f1_mean"] == res2["cv_macro_f1_mean"]
