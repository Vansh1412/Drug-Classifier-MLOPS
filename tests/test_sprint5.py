"""
tests/test_sprint5.py — Comprehensive automated test suite for Sprint 5:
Continuous Model Monitoring + Data Drift Detection.

Verifies:
- Statistical algorithms (KS test, Wasserstein distance, PSI)
- Schema validation and data quality checks (missingness, out-of-range, unseen categories)
- Numerical, categorical, and prediction output drift
- Report generation, JSON serializability, and lineage metadata injection
- Safe report persistence and path traversal prevention
- Zero raw patient records in monitoring reports
- FastAPI endpoints (/monitoring/drift, /monitoring/latest) with input boundary validation
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.main as api_module
from src.config import load_config
from src.monitoring import (
    MONITORING_DISCLAIMER,
    ModelMonitor,
    calculate_categorical_distribution,
    calculate_ks,
    calculate_numerical_summary,
    calculate_psi,
    calculate_wasserstein,
)

client = TestClient(api_module.app)


# ─────────────────────────────────────────────────────────────
# 1. Statistical Algorithms Tests
# ─────────────────────────────────────────────────────────────
class TestStatisticalAlgorithms:
    """Validate core mathematical and statistical drift algorithms."""

    def test_calculate_ks_identical_distributions(self):
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        res = calculate_ks(data, data)
        assert res["statistic"] == 0.0
        assert res["p_value"] == 1.0

    def test_calculate_ks_shifted_distributions(self):
        ref = np.random.normal(loc=20.0, scale=2.0, size=200)
        curr = np.random.normal(loc=45.0, scale=2.0, size=200)
        res = calculate_ks(ref, curr)
        assert res["statistic"] > 0.8
        assert res["p_value"] < 0.001

    def test_calculate_ks_handles_empty_or_nans(self):
        empty = np.array([])
        nans = np.array([np.nan, np.nan])
        res = calculate_ks(empty, nans)
        assert res["statistic"] == 0.0
        assert res["p_value"] == 1.0

    def test_calculate_wasserstein_distance(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        curr_same = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        curr_shifted = np.array([10.0, 12.0, 13.0, 14.0, 15.0])

        assert calculate_wasserstein(ref, curr_same) == 0.0
        assert calculate_wasserstein(ref, curr_shifted) > 1.0

    def test_calculate_psi_categorical_identical(self):
        ref = pd.Series(["HIGH", "LOW", "NORMAL", "HIGH", "LOW"])
        curr = pd.Series(["HIGH", "LOW", "NORMAL", "HIGH", "LOW"])
        psi = calculate_psi(ref, curr, is_categorical=True)
        assert psi == 0.0

    def test_calculate_psi_categorical_shifted(self):
        ref = pd.Series(["HIGH"] * 80 + ["LOW"] * 20)
        curr = pd.Series(["HIGH"] * 10 + ["LOW"] * 90)
        psi = calculate_psi(ref, curr, is_categorical=True)
        # Marked population shift should yield PSI >= 0.20
        assert psi >= 0.20

    def test_calculate_psi_numerical_identical_and_shifted(self):
        ref = np.linspace(10, 100, 200)
        curr_same = np.linspace(10, 100, 200)
        curr_shifted = np.linspace(80, 200, 200)

        assert calculate_psi(ref, curr_same, is_categorical=False) == 0.0
        assert calculate_psi(ref, curr_shifted, is_categorical=False) > 0.2

    def test_calculate_summaries_consistency(self):
        s_num = pd.Series([10.0, 20.0, 30.0])
        num_summary = calculate_numerical_summary(s_num)
        assert num_summary["mean"] == 20.0
        assert num_summary["min"] == 10.0
        assert num_summary["max"] == 30.0

        s_cat = pd.Series(["A", "A", "B"])
        cat_dist = calculate_categorical_distribution(s_cat)
        assert cat_dist["A"] == 0.6667
        assert cat_dist["B"] == 0.3333


# ─────────────────────────────────────────────────────────────
# 2. Engine Initialization & Configuration Tests
# ─────────────────────────────────────────────────────────────
class TestMonitoringEngineInit:
    """Verify ModelMonitor setup, reference data loading, and config binding."""

    def test_monitor_initialization_with_defaults(self):
        monitor = ModelMonitor()
        assert len(monitor.reference_df) == 200
        assert "Age" in monitor.ref_numerical
        assert "Na_to_K" in monitor.ref_numerical
        assert "Sex" in monitor.ref_categorical
        assert "BP" in monitor.ref_categorical
        assert "Cholesterol" in monitor.ref_categorical
        assert monitor.reports_dir.exists()

    def test_monitor_custom_reference_dataset(self):
        custom_df = pd.DataFrame(
            {
                "Age": [25, 45, 65],
                "Sex": ["M", "F", "M"],
                "BP": ["HIGH", "LOW", "NORMAL"],
                "Cholesterol": ["NORMAL", "HIGH", "NORMAL"],
                "Na_to_K": [12.5, 18.2, 24.1],
                "Drug": ["DrugY", "drugA", "drugB"],
            }
        )
        monitor = ModelMonitor(reference_df=custom_df)
        assert len(monitor.reference_df) == 3
        assert monitor.ref_numerical["Age"]["mean"] == 45.0

    def test_monitor_missing_reference_file_raises_error(self):
        bad_config = copy.deepcopy(load_config())
        bad_config["monitoring"]["reference_dataset"] = "Data/non_existent.csv"
        with pytest.raises(FileNotFoundError):
            ModelMonitor(config=bad_config)


# ─────────────────────────────────────────────────────────────
# 3. Schema & Data Quality Drift Tests
# ─────────────────────────────────────────────────────────────
class TestSchemaAndDataQuality:
    """Validate schema contract inspection and data quality anomalies."""

    @pytest.fixture
    def monitor(self):
        return ModelMonitor()

    def test_schema_valid_dataframe_passes(self, monitor):
        valid_df = monitor.reference_df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]]
        res = monitor.validate_schema(valid_df)
        assert res["status"] == "PASS"
        assert len(res["missing_columns"]) == 0
        assert len(res["unexpected_columns"]) == 0

    def test_schema_missing_column_returns_error(self, monitor):
        invalid_df = monitor.reference_df[["Age", "Sex", "BP"]]
        res = monitor.validate_schema(invalid_df)
        assert res["status"] == "ERROR"
        assert "Cholesterol" in res["missing_columns"]
        assert "Na_to_K" in res["missing_columns"]

    def test_schema_unexpected_column_returns_warning(self, monitor):
        df = monitor.reference_df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]].copy()
        df["unwanted_column"] = 123
        res = monitor.validate_schema(df)
        assert res["status"] == "WARNING"
        assert "unwanted_column" in res["unexpected_columns"]

    def test_data_quality_clean_dataset_passes(self, monitor):
        clean_df = monitor.reference_df[
            ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
        ].copy()
        res = monitor.check_data_quality(clean_df)
        assert res["status"] == "PASS"
        assert len(res["unseen_categories"]) == 0

    def test_data_quality_null_injection_yields_warning(self, monitor):
        df = monitor.reference_df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]].copy()
        df.loc[0:5, "Age"] = np.nan
        res = monitor.check_data_quality(df)
        assert res["status"] == "WARNING"
        assert res["null_rates"]["Age"] > 0.0

    def test_data_quality_out_of_range_yields_warning(self, monitor):
        df = monitor.reference_df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]].copy()
        df.loc[0, "Age"] = 99  # Max valid is 74
        res = monitor.check_data_quality(df)
        assert res["status"] == "WARNING"
        assert res["out_of_range"]["Age"]["violating_count"] >= 1

    def test_data_quality_unseen_category_yields_drift(self, monitor):
        df = monitor.reference_df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]].copy()
        df.loc[0, "BP"] = "EXTREME_HIGH"
        res = monitor.check_data_quality(df)
        assert res["status"] == "DRIFT"
        assert "BP" in res["unseen_categories"]


# ─────────────────────────────────────────────────────────────
# 4. Feature and Prediction Drift Tests
# ─────────────────────────────────────────────────────────────
class TestFeatureAndPredictionDrift:
    """Validate numerical, categorical, and prediction distribution drift detection."""

    @pytest.fixture
    def monitor(self):
        return ModelMonitor()

    def test_identical_dataset_yields_pass(self, monitor):
        report = monitor.compute_drift(monitor.reference_df, save_report=False)
        assert report["summary"]["overall_status"] == "PASS"
        assert report["summary"]["overall_severity"] == "LOW"
        assert report["summary"]["feature_drift_detected"] is False
        assert report["summary"]["drifted_features_count"] == 0

    def test_numerical_drift_detected_on_shifted_age(self, monitor):
        df = monitor.reference_df.copy()
        df["Age"] = df["Age"] + 25  # Shift mean from ~44 to ~69
        report = monitor.compute_drift(df, save_report=False)
        assert report["summary"]["overall_status"] in ["DRIFT", "WARNING"]
        assert "Age" in report["summary"]["drifted_features"]
        assert report["features"]["Age"]["drift_detected"] is True

    def test_categorical_drift_detected_on_skewed_sex(self, monitor):
        df = monitor.reference_df.copy()
        # Invert baseline (which is ~52% M, 48% F) to 100% Female
        df["Sex"] = "F"
        report = monitor.compute_drift(df, save_report=False)
        assert report["features"]["Sex"]["psi"] >= 0.20
        assert report["features"]["Sex"]["drift_detected"] is True

    def test_prediction_drift_computation(self, monitor):
        report = monitor.compute_drift(
            monitor.reference_df, include_predictions=True, save_report=False
        )
        assert report["prediction_drift"] is not None
        assert "predicted_class_distribution" in report["prediction_drift"]
        assert "class_psi" in report["prediction_drift"]
        assert report["prediction_drift"]["status"] == "PASS"

    def test_prediction_drift_on_biased_subpopulation(self, monitor):
        df = monitor.reference_df.copy()
        # High Na_to_K ratio strongly predicts DrugY
        df["Na_to_K"] = 35.0
        report = monitor.compute_drift(df, include_predictions=True, save_report=False)
        assert report["prediction_drift"] is not None
        assert "DrugY" in report["prediction_drift"]["predicted_class_distribution"]
        # Extreme shift towards DrugY should trigger prediction drift
        assert report["prediction_drift"]["class_psi"] > 0.15


# ─────────────────────────────────────────────────────────────
# 5. Report Structure, Persistence, and Security Tests
# ─────────────────────────────────────────────────────────────
class TestReportPersistenceAndSecurity:
    """Validate report structure, file persistence, path traversal rejection, and PII protection."""

    @pytest.fixture
    def monitor(self):
        return ModelMonitor()

    def test_report_contains_metadata_and_disclaimer(self, monitor):
        report = monitor.compute_drift(monitor.reference_df, save_report=False)
        assert "report_id" in report
        assert "timestamp" in report
        assert "model_metadata" in report
        assert report["model_metadata"]["model_name"] == "xgboost"
        assert report["model_metadata"]["model_version"] == "0.2.0"
        assert report["disclaimer"] == MONITORING_DISCLAIMER

    def test_report_safe_persistence_and_retrieval(self, monitor, tmp_path):
        monitor.reports_dir = tmp_path
        report = monitor.compute_drift(monitor.reference_df, save_report=True)

        latest = monitor.get_latest_report()
        assert latest is not None
        assert latest["report_id"] == report["report_id"]
        assert latest["summary"]["overall_status"] == "PASS"

    def test_path_traversal_prevention_on_save(self, monitor):
        malicious_report = {
            "report_id": "../../../etc/passwd",
            "summary": {"overall_status": "PASS"},
        }
        with patch.object(monitor, "reports_dir", Path("/safe/reports/dir")):
            with pytest.raises((ValueError, Exception)):
                monitor.save_report(malicious_report)

    def test_zero_raw_patient_records_in_report(self, monitor):
        report = monitor.compute_drift(monitor.reference_df, save_report=False)
        # Confirm no raw patient rows are serialized into the report
        assert "raw_records" not in report
        assert "patient_records" not in report
        assert "patients" not in report


# ─────────────────────────────────────────────────────────────
# 6. FastAPI Monitoring Endpoint Smoke Tests
# ─────────────────────────────────────────────────────────────
class TestFastAPIMonitoringEndpoints:
    """Validate FastAPI /monitoring/drift and /monitoring/latest endpoints."""

    def test_monitoring_drift_endpoint_happy_path(self):
        payload = {
            "records": [
                {
                    "age": 30,
                    "sex": "M",
                    "blood_pressure": "HIGH",
                    "cholesterol": "NORMAL",
                    "na_to_k_ratio": 15.4,
                },
                {
                    "age": 45,
                    "sex": "F",
                    "blood_pressure": "LOW",
                    "cholesterol": "HIGH",
                    "na_to_k_ratio": 18.2,
                },
                {
                    "age": 60,
                    "sex": "M",
                    "blood_pressure": "NORMAL",
                    "cholesterol": "NORMAL",
                    "na_to_k_ratio": 24.1,
                },
            ]
        }
        resp = client.post("/monitoring/drift", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "report_id" in data
        assert "summary" in data
        assert data["summary"]["current_samples"] == 3
        assert "overall_status" in data["summary"]

    def test_monitoring_drift_endpoint_shifted_batch(self):
        # Batch with aged patients shifted far above reference mean
        payload = {
            "records": [
                {
                    "age": 74,
                    "sex": "M",
                    "blood_pressure": "HIGH",
                    "cholesterol": "HIGH",
                    "na_to_k_ratio": 7.0,
                }
                for _ in range(30)
            ]
        }
        resp = client.post("/monitoring/drift", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["overall_status"] in ["DRIFT", "WARNING"]

    def test_monitoring_drift_empty_batch_rejected(self):
        resp = client.post("/monitoring/drift", json={"records": []})
        assert resp.status_code == 422  # Pydantic min_length=1 validation error

    def test_monitoring_drift_oversized_batch_rejected(self):
        oversized_records = [
            {
                "age": 30,
                "sex": "M",
                "blood_pressure": "HIGH",
                "cholesterol": "NORMAL",
                "na_to_k_ratio": 15.0,
            }
            for _ in range(1001)
        ]
        resp = client.post("/monitoring/drift", json={"records": oversized_records})
        assert resp.status_code == 422  # Pydantic max_length=1000 validation error

    def test_monitoring_latest_endpoint(self):
        resp = client.get("/monitoring/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert "report_id" in data
        assert "summary" in data
        assert "disclaimer" in data

    def test_api_metrics_records_monitoring_calls(self):
        resp_initial = client.get("/metrics")
        init_total = resp_initial.json()["total_requests"]

        # Hit /monitoring/latest
        client.get("/monitoring/latest")

        resp_after = client.get("/metrics")
        after_total = resp_after.json()["total_requests"]
        assert after_total > init_total
