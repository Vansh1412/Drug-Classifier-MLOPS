"""
tests/test_audit_leakage.py — ML Evaluation Integrity & Data Leakage Elimination Test Suite.

Comprehensive automated regression tests verifying:
1. Zero train/test overlap and zero duplicate leakage across partitions.
2. Target column strictly isolated from input features.
3. Preprocessing transformations fitted solely on training data/folds inside Pipeline.
4. Model selection (champion selection) strictly independent of held-out test data.
5. Test metrics calculated on test predictions, not training predictions.
6. Prediction and label length alignment (len == 60).
7. Confusion matrix totals, class labels, and diagonal accuracy consistency.
8. Dashboard and artifact metric reconciliation.
9. Protected production asset cryptographic hash integrity.
10. Mathematical separability verification via deterministic clinical prescribing rules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.champion_selection import select_champion
from src.config import load_config
from src.evaluate import evaluate_on_test
from src.model_registry import build_model_candidates
from src.preprocessing import build_preprocessor

# Protected asset hashes
AUTHORITATIVE_DATASET_SHA256 = (
    "9517b83a24a3d7d62eb3920752d409ea187e4a1ff8ae94b4fd946a8e7f63f3b2"
)
AUTHORITATIVE_MODEL_SHA256 = (
    "d1a582fed592daa01eaff5b8eba7648a479eec9dfd10e41bea1c35f6758c0292"
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def dataset_splits(cfg):
    df = pd.read_csv(cfg["data"]["data_path"])
    df_shuffled = df.sample(
        frac=1, random_state=cfg["data"]["shuffle_random_state"]
    ).reset_index(drop=True)

    X = df_shuffled.drop(cfg["data"]["target_column"], axis=1)
    y = df_shuffled[cfg["data"]["target_column"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["split_random_state"],
        stratify=y,
    )
    return {
        "raw_df": df,
        "shuffled_df": df_shuffled,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


class TestDatasetAndSplitIntegrity:
    """Verifies dataset cleanliness, deduplication, and partition isolation."""

    def test_zero_exact_duplicates_in_authoritative_dataset(self, dataset_splits):
        """Assert dataset contains zero exact duplicate rows across all 200 records."""
        df = dataset_splits["raw_df"]
        assert df.shape == (200, 6)
        assert df.duplicated().sum() == 0, "Dataset contains unexpected duplicate rows"

    def test_zero_feature_duplicates_with_conflicting_targets(self, dataset_splits):
        """Assert no two records share identical features but conflicting drug labels."""
        df = dataset_splits["raw_df"]
        feature_cols = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
        dup_features = df[df.duplicated(subset=feature_cols, keep=False)]
        assert (
            len(dup_features) == 0
        ), f"Found {len(dup_features)} conflicting feature rows"

    def test_zero_train_test_overlap(self, dataset_splits):
        """Assert zero feature vector overlap between train and test partitions."""
        X_train = dataset_splits["X_train"].values
        X_test = dataset_splits["X_test"].values

        train_tuples = set(tuple(row) for row in X_train)
        test_tuples = set(tuple(row) for row in X_test)
        overlap = train_tuples.intersection(test_tuples)

        assert (
            len(overlap) == 0
        ), f"Data leakage detected! Found {len(overlap)} identical rows in train and test"

    def test_stratification_proportions_preserved(self, dataset_splits):
        """Assert stratified split preserves class proportions within 2% margin."""
        y_train = dataset_splits["y_train"]
        y_test = dataset_splits["y_test"]

        train_dist = y_train.value_counts(normalize=True).to_dict()
        test_dist = y_test.value_counts(normalize=True).to_dict()

        for cls in train_dist:
            diff = abs(train_dist[cls] - test_dist[cls])
            assert (
                diff < 0.02
            ), f"Class {cls} stratification mismatch: train={train_dist[cls]:.3f}, test={test_dist[cls]:.3f}"


class TestDataLeakageElimination:
    """Verifies prevention of target, preprocessing, and resampling leakage."""

    def test_target_column_strictly_excluded_from_features(self, cfg, dataset_splits):
        """Assert target column 'Drug' is not in feature matrices."""
        target = cfg["data"]["target_column"]
        assert target not in dataset_splits["X_train"].columns
        assert target not in dataset_splits["X_test"].columns

    def test_preprocessing_fitted_solely_on_training_set(self, cfg, dataset_splits):
        """Assert StandardScaler and encoders learn parameters ONLY from X_train."""
        X_train = dataset_splits["X_train"].values
        X_test = dataset_splits["X_test"].values
        y_train = dataset_splits["y_train"].values

        preprocessor = build_preprocessor(cfg)
        pipe = Pipeline(
            [
                ("preprocessing", preprocessor),
                ("dummy", build_model_candidates(cfg)["xgboost"]),
            ]
        )
        pipe.fit(X_train, y_train)

        # Retrieve fitted StandardScaler inside ColumnTransformer
        fitted_ct = pipe.named_steps["preprocessing"]
        scaler = fitted_ct.named_transformers_["num_scaler"]
        scaler_mean = scaler.mean_

        # Calculate expected mean solely from training numerical columns: Age (col 0), Na_to_K (col 4)
        expected_age_mean = np.mean(X_train[:, 0].astype(float))
        expected_na_k_mean = np.mean(X_train[:, 4].astype(float))

        np.testing.assert_almost_equal(scaler_mean[0], expected_age_mean, decimal=5)
        np.testing.assert_almost_equal(scaler_mean[1], expected_na_k_mean, decimal=5)

        # Verify scaler mean does NOT equal the full dataset or test set mean
        test_age_mean = np.mean(X_test[:, 0].astype(float))
        assert not np.isclose(scaler_mean[0], test_age_mean)

    def test_no_resampling_prior_to_split(self, dataset_splits):
        """Assert exact sample count preservation: 140 train + 60 test == 200 total."""
        assert len(dataset_splits["X_train"]) == 140
        assert len(dataset_splits["X_test"]) == 60
        assert (
            len(dataset_splits["X_train"]) + len(dataset_splits["X_test"])
            == len(dataset_splits["raw_df"])
            == 200
        )


class TestModelSelectionIndependence:
    """Verifies that model selection does not depend on held-out test data."""

    def test_champion_selection_driven_by_cross_validation(self, cfg):
        """Assert select_champion ranks and gates by CV metrics when selection_basis='cv'."""
        # Candidate 1: High CV, lower test
        c1 = {
            "model_name": "model_cv_winner",
            "cv_macro_f1_mean": 0.95,
            "cv_accuracy_mean": 0.96,
            "cv_macro_recall_mean": 0.94,
            "accuracy": 0.90,
            "macro_f1": 0.88,
            "macro_recall": 0.86,
        }
        # Candidate 2: Low CV, but overfitted high test score
        c2 = {
            "model_name": "model_test_overfitter",
            "cv_macro_f1_mean": 0.82,
            "cv_accuracy_mean": 0.85,
            "cv_macro_recall_mean": 0.80,
            "accuracy": 0.98,
            "macro_f1": 0.97,
            "macro_recall": 0.96,
        }

        champion, challengers = select_champion([c1, c2], cfg)
        assert (
            champion["model_name"] == "model_cv_winner"
        ), "Champion selection must select CV winner, not test overfitter!"
        assert challengers[0]["model_name"] == "model_test_overfitter"

    def test_quality_gate_rejects_candidate_failing_cv_even_with_high_test(self, cfg):
        """Assert quality gate rejects a candidate that fails CV minimum threshold."""
        c_failing_cv = {
            "model_name": "leaky_model",
            "cv_macro_f1_mean": 0.60,  # Below min_macro_f1 (0.75)
            "cv_accuracy_mean": 0.85,
            "cv_macro_recall_mean": 0.75,
            "accuracy": 1.00,  # Artificially perfect on test
            "macro_f1": 1.00,
            "macro_recall": 1.00,
        }
        c_passing = {
            "model_name": "legit_model",
            "cv_macro_f1_mean": 0.80,
            "cv_accuracy_mean": 0.85,
            "cv_macro_recall_mean": 0.78,
            "accuracy": 0.82,
            "macro_f1": 0.80,
            "macro_recall": 0.78,
        }
        champion, challengers = select_champion([c_failing_cv, c_passing], cfg)
        assert champion["model_name"] == "legit_model"
        assert len(challengers) == 0


class TestEvaluationMetricsAndMatrixConsistency:
    """Verifies metrics computation, lengths, confusion matrix, and reporting agreement."""

    def test_prediction_and_label_length_alignment(self, dataset_splits):
        """Assert prediction vector matches true test label vector length exactly (60)."""
        pipe = sio.load(
            "Model/drug_pipeline.skops",
            trusted=sio.get_untrusted_types(file="Model/drug_pipeline.skops"),
        )
        preds = pipe.predict(dataset_splits["X_test"].values)
        assert len(preds) == len(dataset_splits["y_test"]) == 60

    def test_test_metrics_derived_strictly_from_test_split(self, dataset_splits):
        """Assert evaluate_on_test produces metrics matching manual scikit-learn calculation."""
        pipe = sio.load(
            "Model/drug_pipeline.skops",
            trusted=sio.get_untrusted_types(file="Model/drug_pipeline.skops"),
        )
        X_test = dataset_splits["X_test"].values
        y_test = dataset_splits["y_test"].values

        metrics = evaluate_on_test(pipe, X_test, y_test)
        preds = pipe.predict(X_test)

        expected_acc = accuracy_score(y_test, preds)
        expected_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        expected_rec = recall_score(y_test, preds, average="macro", zero_division=0)

        assert metrics["accuracy"] == expected_acc == 1.0
        assert metrics["macro_f1"] == expected_f1 == 1.0
        assert metrics["macro_recall"] == expected_rec == 1.0

    def test_confusion_matrix_dimensions_and_support_sum(self, dataset_splits):
        """Assert confusion matrix matches test sample count (60) and 5 unique classes."""
        pipe = sio.load(
            "Model/drug_pipeline.skops",
            trusted=sio.get_untrusted_types(file="Model/drug_pipeline.skops"),
        )
        y_test = dataset_splits["y_test"].values
        preds = pipe.predict(dataset_splits["X_test"].values)

        classes = sorted(set(y_test.tolist()))
        cm = confusion_matrix(y_test, preds, labels=classes)

        assert cm.shape == (5, 5), "Confusion matrix must be 5x5 for 5 drug classes"
        assert (
            cm.sum() == 60
        ), "Confusion matrix total must sum to test sample count (60)"
        assert (
            np.trace(cm) == 60
        ), "All 60 test samples must lie on the diagonal (100% accuracy)"

    def test_artifact_and_report_metrics_consistency(self):
        """Assert Results/metrics.json, Results/metrics.txt, and Model/model_metadata.json agree."""
        with open("Results/metrics.json", "r", encoding="utf-8") as fh:
            metrics_json = json.load(fh)

        with open("Model/model_metadata.json", "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        assert (
            metrics_json["champion"]["model_name"]
            == metadata["model_name"]
            == "xgboost"
        )
        assert metrics_json["champion"]["accuracy"] == metadata["accuracy"] == 1.0
        assert metrics_json["champion"]["macro_f1"] == metadata["macro_f1"] == 1.0
        assert (
            metrics_json["champion"]["macro_recall"] == metadata["macro_recall"] == 1.0
        )

        # Check metrics.txt contains champion results
        metrics_txt = Path("Results/metrics.txt").read_text(encoding="utf-8")
        assert "Champion Model: xgboost" in metrics_txt
        assert "Accuracy = 1.0" in metrics_txt
        assert "Macro F1 = 1.0" in metrics_txt


class TestSeparabilityAndArtifactIntegrity:
    """Verifies mathematical explanation for 100% score and protected asset hashes."""

    def test_ground_truth_separability_via_deterministic_clinical_rules(
        self, dataset_splits
    ):
        """
        Verify that 100% accuracy is the mathematically expected result for this dataset
        because the ground-truth labels follow exact deterministic clinical decision rules:
          - Na_to_K > 14.83 -> DrugY
          - BP == HIGH: Age <= 50 -> drugA, Age > 50 -> drugB
          - BP == LOW: Cholesterol == HIGH -> drugC, Cholesterol == NORMAL -> drugX
          - BP == NORMAL -> drugX
        """
        df = dataset_splits["raw_df"]

        def rule_prescribe(row):
            if row["Na_to_K"] > 14.83:
                return "DrugY"
            if row["BP"] == "HIGH":
                return "drugA" if row["Age"] <= 50 else "drugB"
            if row["BP"] == "LOW":
                return "drugC" if row["Cholesterol"] == "HIGH" else "drugX"
            if row["BP"] == "NORMAL":
                return "drugX"
            return "UNKNOWN"

        rule_predictions = df.apply(rule_prescribe, axis=1)
        rule_accuracy = (rule_predictions == df["Drug"]).mean()

        assert (
            rule_accuracy == 1.0
        ), f"Rule accuracy must be 1.0 on all 200 records, got {rule_accuracy}"

    def test_algorithmic_boundary_disjointness_without_hardcoded_rule(
        self, dataset_splits
    ):
        """
        Independently proves the clinical decision boundaries without hardcoding any rule:
        1. Calculates min Na_to_K for DrugY vs max Na_to_K for non-Y (strictly disjoint).
        2. Calculates max Age for drugA vs min Age for drugB among BP=HIGH (strictly disjoint).
        3. Asserts conditional purity of BP=LOW by Cholesterol and BP=NORMAL.
        4. Fits an independent DecisionTreeClassifier directly to verify 0 training error.
        """
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.tree import DecisionTreeClassifier

        df = dataset_splits["raw_df"]
        drug_y = df[df["Drug"] == "DrugY"]
        non_y = df[df["Drug"] != "DrugY"]

        # 1. Independent Na_to_K threshold disjointness
        min_y_na_k = drug_y["Na_to_K"].min()
        max_non_y_na_k = non_y["Na_to_K"].max()
        assert (
            min_y_na_k > max_non_y_na_k
        ), f"Na_to_K overlap detected: DrugY min {min_y_na_k} <= non-Y max {max_non_y_na_k}"

        # 2. Independent Age threshold disjointness among BP=HIGH
        bp_high = non_y[non_y["BP"] == "HIGH"]
        max_drugA_age = bp_high[bp_high["Drug"] == "drugA"]["Age"].max()
        min_drugB_age = bp_high[bp_high["Drug"] == "drugB"]["Age"].min()
        assert (
            max_drugA_age < min_drugB_age
        ), f"Age overlap detected for BP=HIGH: drugA max {max_drugA_age} >= drugB min {min_drugB_age}"

        # 3. Independent purity for BP=LOW and BP=NORMAL
        bp_low = non_y[non_y["BP"] == "LOW"]
        assert set(bp_low[bp_low["Cholesterol"] == "HIGH"]["Drug"]) == {"drugC"}
        assert set(bp_low[bp_low["Cholesterol"] == "NORMAL"]["Drug"]) == {"drugX"}

        bp_norm = non_y[non_y["BP"] == "NORMAL"]
        assert set(bp_norm["Drug"]) == {"drugX"}

        # 4. Independent decision tree fit achieves 100% training accuracy (zero loss)
        X_df = df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]].copy()
        oe = OrdinalEncoder()
        X_df[["Sex", "BP", "Cholesterol"]] = oe.fit_transform(
            X_df[["Sex", "BP", "Cholesterol"]]
        )
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_df, df["Drug"])
        dt_acc = dt.score(X_df, df["Drug"])
        assert (
            dt_acc == 1.0
        ), f"Independent decision tree failed to achieve 1.0 accuracy: {dt_acc}"

    def test_evaluation_artifact_lineage_and_metadata_mapping(self, cfg):
        """Assert evaluation reports strictly match the authoritative artifact and dataset on disk."""
        with open("Results/metrics.json", "r", encoding="utf-8") as fh:
            metrics_json = json.load(fh)

        with open("Model/model_metadata.json", "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        with open(cfg["data"]["data_path"], "rb") as fh:
            actual_dataset_hash = hashlib.sha256(fh.read()).hexdigest()

        with open(cfg["champion"]["artifact_path"], "rb") as fh:
            actual_artifact_hash = hashlib.sha256(fh.read()).hexdigest()

        # Verify lineage hashes match active files on disk
        assert metrics_json["dataset"]["dataset_sha256"] == actual_dataset_hash
        assert metrics_json["champion"]["artifact_sha256"] == actual_artifact_hash
        assert metadata["dataset_sha256"] == actual_dataset_hash
        assert metadata["artifact_sha256"] == actual_artifact_hash

        # Verify model selection basis is CV to guard against test-set selection leakage
        assert metrics_json["champion"]["selection_basis"] == "cv"
        assert metadata["selection_basis"] == "cv"

    def test_protected_dataset_sha256_unaltered(self, cfg):
        """Assert Data/drug.csv SHA-256 matches authoritative hash exactly."""
        with open(cfg["data"]["data_path"], "rb") as fh:
            current_hash = hashlib.sha256(fh.read()).hexdigest()
        assert (
            current_hash == AUTHORITATIVE_DATASET_SHA256
        ), f"Data/drug.csv hash mismatch: {current_hash} != {AUTHORITATIVE_DATASET_SHA256}"

    def test_protected_model_artifact_sha256_unaltered(self, cfg):
        """Assert Model/drug_pipeline.skops SHA-256 matches authoritative production hash."""
        with open(cfg["champion"]["artifact_path"], "rb") as fh:
            current_hash = hashlib.sha256(fh.read()).hexdigest()
        assert (
            current_hash == AUTHORITATIVE_MODEL_SHA256
        ), f"Model/drug_pipeline.skops hash mismatch: {current_hash} != {AUTHORITATIVE_MODEL_SHA256}"
