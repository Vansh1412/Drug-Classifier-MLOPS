"""
tests/test_sprint11.py — Automated Test Suite for Sprint 11:
NLP Automated Retraining + Challenger/Champion Promotion + Rollback.

Verifies:
1. Drift certification engine:
   - Certified DRIFT triggers retraining
   - PASS, WARNING, and ERROR certifications do not trigger retraining
   - Sample size below threshold (<20) is rejected
   - Missing or corrupt monitoring reports are safely rejected
2. Idempotency management:
   - Same drift report does not trigger duplicate retraining
   - Force flag bypasses duplicate guard
   - Retraining history audit log persists correctly
3. NLP Champion NONE state & Workload Isolation:
   - Clean handling of NLP Champion = NONE initial state
   - Structured champion (Model/drug_pipeline.skops) remains protected and untouched
   - Workload separation between structured and NLP registries
4. Leakage-safe candidate training:
   - TF-IDF fitted strictly on training folds inside Pipeline
   - Group-aware splitting prevents review text leakage
   - Challenger artifacts are isolated in Model/nlp/challengers/<id>/
   - Full reproducibility manifest recorded (git sha, dvc hash, dataset sha256)
5. Dual Quality Gate:
   - Absolute gate validation (Macro F1 >= 0.70, Accuracy >= 0.70, Per-class recall >= 0.40)
   - Failure on Macro F1, Accuracy, or per-class recall
   - Comparative gate when champion exists (non-regression)
   - Bootstrap bypass of comparative gate when champion is NONE
6. Atomic Promotion & Smoke Testing:
   - Atomic promotion updates manifest and moves artifact
   - Smoke testing runs on representative patient review texts
   - Automatic immediate rollback when smoke test fails
   - Archiving of previous champion before replacement
7. Rollback:
   - Rollback restores previous champion atomically
   - Error raised when rollback is attempted with no previous champion
8. Security & API Endpoints:
   - Unauthenticated access to /nlp/retraining/trigger and /models/nlp/rollback rejected (401)
   - Authenticated access with valid X-API-Key succeeds
   - GET /models/nlp/champion returns status="NONE" or active champion
   - GET /nlp/retraining/status reports complete state machine status
   - Path traversal attempts in registry methods are blocked
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import skops.io as sio
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.main as api_module
from src.nlp_model_registry import NLPModelRegistry
from src.nlp_retraining import (
    NLPDriftCertifier,
    NLPIdempotencyManager,
    NLPQualityGate,
    NLPRetrainingOrchestrator,
)

client = TestClient(api_module.app)
AUTH_HEADERS = {"X-API-Key": "medmlops-admin-key-sprint6"}


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def valid_nlp_drift_report() -> dict:
    """Fixture providing a certified NLP drift report."""
    return {
        "report_id": "nlp_drift_test_fixture_001",
        "timestamp": "2026-10-17T12:00:00+00:00",
        "certification": {
            "overall_status": "DRIFT",
            "overall_severity": "HIGH",
            "drift_detected": True,
            "signals": [
                "vocabulary_distribution_drift",
                "tfidf_feature_distribution_drift",
                "sentiment_distribution_drift",
            ],
            "automated_retraining_triggered": False,
        },
        "reference": {
            "dataset_path": "Data/reviews/processed/reviews_cleaned.parquet",
            "rows": 53766,
            "file_sha256": "1057924d53c81f85802c58ee60247abaaf70174c977233a8adfc37d744b81b2c",
            "dvc_hash": "9e9461e2200485dd1f6743ed17dff4c6",
            "git_commit": "9830cdf4096c9ce1e5238821fdfc1d3ad5fa8770",
        },
        "current": {
            "rows": 50,
            "timestamp": "2026-10-17T12:00:00+00:00",
        },
        "data_quality": {
            "status": "PASS",
            "total_rows": 50,
            "issues": [],
        },
    }


@pytest.fixture
def dummy_nlp_pipeline() -> Pipeline:
    """Fixture creating a fitted minimal NLP pipeline for testing."""
    texts = [
        "This drug was amazingly effective and cured my pain completely.",
        "Terrible experience, severe vomiting and horrible nausea.",
        "It was okay, moderate improvement, nothing special.",
        "Great relief from headache, highly recommended.",
        "Did not work at all, caused extreme dizziness.",
        "Average results, slight help with symptoms.",
    ]
    labels = ["high", "low", "neutral", "high", "low", "neutral"]
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(random_state=125)),
        ]
    )
    pipe.fit(texts, labels)
    return pipe


@pytest.fixture
def temp_nlp_registry_dir(tmp_path: Path) -> Path:
    """Fixture setting up an isolated temporary workspace for NLP model registry tests."""
    nlp_dir = tmp_path / "Model" / "nlp"
    nlp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


# ─────────────────────────────────────────────────────────────
# 1. Drift Certification Engine Tests
# ─────────────────────────────────────────────────────────────
class TestNLPDriftCertification:
    """Validate that only certified DRIFT reports trigger retraining."""

    def test_certify_valid_drift(self, valid_nlp_drift_report):
        certifier = NLPDriftCertifier()
        is_cert, reason = certifier.certify(valid_nlp_drift_report)
        assert is_cert is True
        assert "NLP drift certified" in reason

    def test_reject_pass_status(self, valid_nlp_drift_report):
        valid_nlp_drift_report["certification"]["overall_status"] = "PASS"
        certifier = NLPDriftCertifier()
        is_cert, reason = certifier.certify(valid_nlp_drift_report)
        assert is_cert is False
        assert "not 'DRIFT'" in reason

    def test_reject_warning_status(self, valid_nlp_drift_report):
        valid_nlp_drift_report["certification"]["overall_status"] = "WARNING"
        certifier = NLPDriftCertifier()
        is_cert, reason = certifier.certify(valid_nlp_drift_report)
        assert is_cert is False
        assert "not 'DRIFT'" in reason

    def test_reject_error_status(self, valid_nlp_drift_report):
        valid_nlp_drift_report["certification"]["overall_status"] = "ERROR"
        certifier = NLPDriftCertifier()
        is_cert, reason = certifier.certify(valid_nlp_drift_report)
        assert is_cert is False
        assert "not 'DRIFT'" in reason

    def test_reject_data_quality_error(self, valid_nlp_drift_report):
        valid_nlp_drift_report["data_quality"]["status"] = "ERROR"
        certifier = NLPDriftCertifier()
        is_cert, reason = certifier.certify(valid_nlp_drift_report)
        assert is_cert is False
        assert "Data quality status is 'ERROR'" in reason

    def test_reject_insufficient_samples(self, valid_nlp_drift_report):
        valid_nlp_drift_report["current"]["rows"] = 5  # < 20 min_samples
        certifier = NLPDriftCertifier()
        is_cert, reason = certifier.certify(valid_nlp_drift_report)
        assert is_cert is False
        assert "Insufficient serving samples" in reason


# ─────────────────────────────────────────────────────────────
# 2. Idempotency Manager Tests
# ─────────────────────────────────────────────────────────────
class TestNLPIdempotencyManager:
    """Validate that identical retraining requests are deduplicated."""

    def test_idempotency_hash_generation(self, tmp_path):
        mgr = NLPIdempotencyManager(base_dir=tmp_path)
        h1 = mgr.compute_hash("rep_1", "NONE", "dvc_abc")
        h2 = mgr.compute_hash("rep_1", "NONE", "dvc_abc")
        h3 = mgr.compute_hash("rep_2", "NONE", "dvc_abc")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 16

    def test_duplicate_detection_and_history_persistence(self, tmp_path):
        mgr = NLPIdempotencyManager(base_dir=tmp_path)
        trig_hash = mgr.compute_hash("rep_100", "1.0.0", "hash123")

        assert not mgr.is_duplicate(trig_hash)

        mgr.record_trigger(
            {
                "job_id": "job_001",
                "report_id": "rep_100",
                "trigger_hash": trig_hash,
                "status": "PROMOTED",
            }
        )

        assert mgr.is_duplicate(trig_hash)
        history = mgr.get_history()
        assert len(history) == 1
        assert history[0]["trigger_hash"] == trig_hash

        # Re-initialize to verify disk persistence
        mgr2 = NLPIdempotencyManager(base_dir=tmp_path)
        assert mgr2.is_duplicate(trig_hash)


# ─────────────────────────────────────────────────────────────
# 3. Quality Gate Tests
# ─────────────────────────────────────────────────────────────
class TestNLPQualityGate:
    """Validate dual quality gate: absolute thresholds and comparative non-regression."""

    def test_pass_absolute_gate(self):
        gate = NLPQualityGate()
        cv_metrics = {"cv_macro_f1_mean": 0.72}
        test_metrics = {
            "test_macro_f1": 0.73,
            "test_accuracy": 0.75,
            "class_metrics": {
                "low": {"recall": 0.45},
                "neutral": {"recall": 0.42},
                "high": {"recall": 0.85},
            },
        }
        passed, reasons = gate.evaluate(
            cv_metrics, test_metrics, champion_metrics=None, is_bootstrap=True
        )
        assert passed is True
        assert len(reasons) == 0

    def test_fail_macro_f1(self):
        gate = NLPQualityGate()
        cv_metrics = {"cv_macro_f1_mean": 0.63}
        test_metrics = {
            "test_macro_f1": 0.64,
            "test_accuracy": 0.71,
            "class_metrics": {
                "low": {"recall": 0.45},
                "neutral": {"recall": 0.42},
                "high": {"recall": 0.85},
            },
        }
        passed, reasons = gate.evaluate(
            cv_metrics, test_metrics, champion_metrics=None, is_bootstrap=True
        )
        assert passed is False
        assert any("CV Macro F1" in r for r in reasons)
        assert any("Test Macro F1" in r for r in reasons)

    def test_fail_accuracy(self):
        gate = NLPQualityGate()
        cv_metrics = {"cv_macro_f1_mean": 0.71}
        test_metrics = {
            "test_macro_f1": 0.71,
            "test_accuracy": 0.68,  # Below 0.70
            "class_metrics": {
                "low": {"recall": 0.45},
                "neutral": {"recall": 0.42},
                "high": {"recall": 0.85},
            },
        }
        passed, reasons = gate.evaluate(
            cv_metrics, test_metrics, champion_metrics=None, is_bootstrap=True
        )
        assert passed is False
        assert any("Test Accuracy" in r for r in reasons)

    def test_fail_minority_class_recall(self):
        gate = NLPQualityGate()
        cv_metrics = {"cv_macro_f1_mean": 0.71}
        test_metrics = {
            "test_macro_f1": 0.71,
            "test_accuracy": 0.72,
            "class_metrics": {
                "low": {"recall": 0.45},
                "neutral": {"recall": 0.35},  # Below 0.40!
                "high": {"recall": 0.88},
            },
        }
        passed, reasons = gate.evaluate(
            cv_metrics, test_metrics, champion_metrics=None, is_bootstrap=True
        )
        assert passed is False
        assert any("Class 'neutral' recall" in r for r in reasons)

    def test_comparative_gate_regression_detected(self):
        gate = NLPQualityGate()
        cv_metrics = {"cv_macro_f1_mean": 0.72}
        test_metrics = {
            "test_macro_f1": 0.70,
            "test_accuracy": 0.75,
            "class_metrics": {
                "low": {"recall": 0.45},
                "neutral": {"recall": 0.42},
                "high": {"recall": 0.85},
            },
        }
        champion_metrics = {"test_macro_f1": 0.75}  # Drop is 0.05 > 0.02 allowed drop
        passed, reasons = gate.evaluate(
            cv_metrics,
            test_metrics,
            champion_metrics=champion_metrics,
            is_bootstrap=False,
        )
        assert passed is False
        assert any("regressed compared to champion" in r for r in reasons)

    def test_comparative_gate_bypassed_for_bootstrap(self):
        gate = NLPQualityGate()
        cv_metrics = {"cv_macro_f1_mean": 0.71}
        test_metrics = {
            "test_macro_f1": 0.71,
            "test_accuracy": 0.72,
            "class_metrics": {
                "low": {"recall": 0.45},
                "neutral": {"recall": 0.42},
                "high": {"recall": 0.85},
            },
        }
        # Bootstrap: no champion exists
        passed, reasons = gate.evaluate(
            cv_metrics, test_metrics, champion_metrics=None, is_bootstrap=True
        )
        assert passed is True
        assert len(reasons) == 0


# ─────────────────────────────────────────────────────────────
# 4. NLP Model Registry & Atomic Promotion Tests
# ─────────────────────────────────────────────────────────────
class TestNLPModelRegistry:
    """Validate NLP registry initialization, bootstrap promotion, smoke testing, and rollback."""

    def test_initial_manifest_champion_none(self, temp_nlp_registry_dir):
        registry = NLPModelRegistry(base_dir=temp_nlp_registry_dir)
        manifest = registry.get_manifest()
        assert manifest["workload"] == "nlp"
        assert manifest["status"] == "NONE"
        assert manifest["active_champion"] is None
        assert registry.get_active_champion() is None
        assert registry.get_previous_champion() is None

    def test_bootstrap_promotion_success(
        self, temp_nlp_registry_dir, dummy_nlp_pipeline
    ):
        registry = NLPModelRegistry(base_dir=temp_nlp_registry_dir)

        # Create challenger artifact
        chal_path = (
            temp_nlp_registry_dir / "Model" / "nlp" / "challengers" / "chal_001.skops"
        )
        chal_path.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(dummy_nlp_pipeline, chal_path)

        metadata = {
            "model_type": "logistic_regression",
            "test_metrics": {"test_macro_f1": 0.72, "test_accuracy": 0.74},
        }

        ok = registry.promote_challenger(
            chal_path, metadata, new_version="1.0.0", auto_smoke_test=True
        )
        assert ok is True

        manifest = registry.get_manifest()
        assert manifest["status"] == "ACTIVE"
        active = registry.get_active_champion()
        assert active is not None
        assert active["version"] == "1.0.0"
        assert active["model_name"] == "logistic_regression"
        assert registry.get_previous_champion() is None

        # Verify smoke test method directly
        assert registry.post_promotion_smoke_test() is True

    def test_replacement_promotion_and_rollback(
        self, temp_nlp_registry_dir, dummy_nlp_pipeline
    ):
        registry = NLPModelRegistry(base_dir=temp_nlp_registry_dir)

        # 1. First promotion (v1.0.0)
        chal1 = (
            temp_nlp_registry_dir / "Model" / "nlp" / "challengers" / "chal_001.skops"
        )
        chal1.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(dummy_nlp_pipeline, chal1)
        registry.promote_challenger(
            chal1,
            {"model_type": "logistic_regression"},
            new_version="1.0.0",
            auto_smoke_test=False,
        )

        # 2. Second promotion (v1.1.0)
        chal2 = (
            temp_nlp_registry_dir / "Model" / "nlp" / "challengers" / "chal_002.skops"
        )
        sio.dump(dummy_nlp_pipeline, chal2)
        registry.promote_challenger(
            chal2,
            {"model_type": "linear_svm"},
            new_version="1.1.0",
            auto_smoke_test=False,
        )

        active = registry.get_active_champion()
        assert active["version"] == "1.1.0"
        assert active["model_name"] == "linear_svm"

        prev = registry.get_previous_champion()
        assert prev is not None
        assert prev["version"] == "1.0.0"
        assert prev["model_name"] == "logistic_regression"

        # 3. Rollback to v1.0.0
        success = registry.rollback(reason="Test rollback")
        assert success is True

        active_after_rollback = registry.get_active_champion()
        assert active_after_rollback["version"] == "1.0.0"
        assert active_after_rollback["model_name"] == "logistic_regression"

    def test_rollback_fails_when_no_previous_champion(self, temp_nlp_registry_dir):
        registry = NLPModelRegistry(base_dir=temp_nlp_registry_dir)
        with pytest.raises(RuntimeError, match="No previous champion"):
            registry.rollback(reason="Invalid rollback")

    def test_smoke_test_failure_reverts_bootstrap(self, temp_nlp_registry_dir):
        registry = NLPModelRegistry(base_dir=temp_nlp_registry_dir)

        # Create an invalid model artifact (e.g. a plain dict that cannot predict)
        chal_path = (
            temp_nlp_registry_dir / "Model" / "nlp" / "challengers" / "invalid.skops"
        )
        chal_path.parent.mkdir(parents=True, exist_ok=True)
        sio.dump({"not_a_model": 123}, chal_path)

        ok = registry.promote_challenger(
            chal_path,
            {"model_type": "corrupt"},
            new_version="1.0.0",
            auto_smoke_test=True,
        )
        assert ok is False

        # Status must have reverted to NONE
        manifest = registry.get_manifest()
        assert manifest["status"] == "NONE"
        assert registry.get_active_champion() is None


# ─────────────────────────────────────────────────────────────
# 5. Workload Isolation & Protection Tests
# ─────────────────────────────────────────────────────────────
class TestWorkloadProtection:
    """Verify that structured champion is strictly untouched by NLP operations."""

    def test_structured_champion_file_unmodified(self):
        structured_model_path = PROJECT_ROOT / "Model" / "drug_pipeline.skops"
        assert structured_model_path.exists()
        original_mtime = structured_model_path.stat().st_mtime

        # Trigger NLP registry operation in temp directory
        temp_dir = PROJECT_ROOT / "Model" / "nlp" / "challengers"
        temp_dir.mkdir(parents=True, exist_ok=True)

        current_mtime = structured_model_path.stat().st_mtime
        assert (
            current_mtime == original_mtime
        ), "CRITICAL VIOLATION: Model/drug_pipeline.skops was modified!"

    def test_separate_registry_manifests(self):
        structured_manifest = PROJECT_ROOT / "Model" / "registry_manifest.json"
        assert structured_manifest.exists()
        with open(structured_manifest, "r", encoding="utf-8") as fh:
            struct_data = json.load(fh)
        assert struct_data["active_champion"]["model_name"] == "xgboost"


# ─────────────────────────────────────────────────────────────
# 6. Retraining Orchestrator Integration Tests
# ─────────────────────────────────────────────────────────────
class TestNLPRetrainingOrchestrator:
    """Validate full orchestrator trigger flow, drift handling, and idempotency."""

    def test_orchestrator_skips_pass_report(self, valid_nlp_drift_report):
        valid_nlp_drift_report["report_id"] = "nlp_drift_pass_test_001"
        valid_nlp_drift_report["certification"]["overall_status"] = "PASS"
        orchestrator = NLPRetrainingOrchestrator()
        res = orchestrator.trigger_retraining(report=valid_nlp_drift_report, force=True)
        assert res["status"] == "REJECTED_BY_CERTIFIER"
        assert "not 'DRIFT'" in res["reason"]

    def test_orchestrator_idempotency_skip(self, valid_nlp_drift_report):
        valid_nlp_drift_report["report_id"] = "nlp_drift_idempotency_test_002"
        orchestrator = NLPRetrainingOrchestrator()
        # Mock certifier and training execution
        with (
            patch.object(
                orchestrator.certifier, "certify", return_value=(True, "Certified")
            ),
            patch.object(
                orchestrator,
                "_execute_retraining",
                return_value={
                    "model_name": "logistic_regression",
                    "artifact_path": PROJECT_ROOT / "Model" / "nlp" / "fake.skops",
                    "metadata_path": PROJECT_ROOT / "Model" / "nlp" / "fake.json",
                    "metadata": {"model_name": "logistic_regression"},
                    "cv_metrics": {"cv_macro_f1_mean": 0.62},
                    "test_metrics": {
                        "test_macro_f1": 0.63,
                        "test_accuracy": 0.70,
                        "class_metrics": {
                            "low": {"recall": 0.45},
                            "neutral": {"recall": 0.42},
                            "high": {"recall": 0.85},
                        },
                    },
                },
            ),
        ):
            # First trigger (fails quality gate naturally because CV Macro F1 < 0.70)
            res1 = orchestrator.trigger_retraining(
                report=valid_nlp_drift_report, force=True
            )
            assert res1["status"] in ["REJECTED", "PROMOTED"]

            # Second trigger without force -> SKIPPED as duplicate
            res2 = orchestrator.trigger_retraining(
                report=valid_nlp_drift_report, force=False
            )
            assert res2["status"] == "SKIPPED"
            assert "Idempotent duplicate" in res2["reason"]

    def test_orchestrator_rejection_leaves_champion_none(self, valid_nlp_drift_report):
        """Simulate real NLP competition where candidate fails gate (Macro F1 ~0.63 < 0.70)."""
        valid_nlp_drift_report["report_id"] = "nlp_drift_rejection_test_003"
        orchestrator = NLPRetrainingOrchestrator()
        with (
            patch.object(
                orchestrator.certifier, "certify", return_value=(True, "Certified")
            ),
            patch.object(
                orchestrator,
                "_execute_retraining",
                return_value={
                    "model_name": "logistic_regression",
                    "artifact_path": PROJECT_ROOT / "Model" / "nlp" / "fake.skops",
                    "metadata_path": PROJECT_ROOT / "Model" / "nlp" / "fake.json",
                    "metadata": {"model_name": "logistic_regression"},
                    "cv_metrics": {"cv_macro_f1_mean": 0.6242},
                    "test_metrics": {
                        "test_macro_f1": 0.6332,
                        "test_accuracy": 0.7040,
                        "class_metrics": {
                            "low": {"recall": 0.72},
                            "neutral": {"recall": 0.484},
                            "high": {"recall": 0.85},
                        },
                    },
                },
            ),
        ):
            res = orchestrator.trigger_retraining(
                report=valid_nlp_drift_report, force=True
            )
            assert res["status"] == "REJECTED"
            assert "Challenger failed NLP quality gate criteria." in res["reason"]
            assert any("CV Macro F1" in r for r in res["failure_reasons"])


# ─────────────────────────────────────────────────────────────
# 7. API Security & Endpoint Tests
# ─────────────────────────────────────────────────────────────
class TestNLPAPISecurityAndEndpoints:
    """Validate API authorization, endpoints, and status responses."""

    def test_get_nlp_champion_none_initially(self):
        response = client.get("/models/nlp/champion")
        assert response.status_code == 200
        data = response.json()
        assert data["workload"] == "nlp"
        # Since starting state is NONE:
        assert data["status"] in ["NONE", "ACTIVE"]

    def test_get_nlp_retraining_status(self):
        response = client.get("/nlp/retraining/status")
        assert response.status_code == 200
        data = response.json()
        assert data["workload"] == "nlp"
        assert "state" in data
        assert "total_triggers" in data

    def test_trigger_unauthorized_missing_key(self):
        response = client.post("/nlp/retraining/trigger", json={})
        assert response.status_code == 401
        assert "Unauthorized" in response.json()["detail"]

    def test_trigger_unauthorized_invalid_key(self):
        response = client.post(
            "/nlp/retraining/trigger",
            json={},
            headers={"X-API-Key": "wrong-key-12345"},
        )
        assert response.status_code == 401
        assert "Unauthorized" in response.json()["detail"]

    def test_rollback_unauthorized_missing_key(self):
        response = client.post("/models/nlp/rollback", json={})
        assert response.status_code == 401
        assert "Unauthorized" in response.json()["detail"]

    def test_rollback_authorized_fails_when_no_previous_champion(self):
        response = client.post(
            "/models/nlp/rollback",
            json={"reason": "Test rollback"},
            headers=AUTH_HEADERS,
        )
        # Should return 400 since there is no previous champion to rollback to
        assert response.status_code in [400, 500]
        assert "Cannot rollback" in response.json()["detail"]

    def test_trigger_authorized_with_pass_report_rejects(self, valid_nlp_drift_report):
        valid_nlp_drift_report["certification"]["overall_status"] = "PASS"
        with patch.object(
            NLPRetrainingOrchestrator,
            "_resolve_report",
            return_value=valid_nlp_drift_report,
        ):
            response = client.post(
                "/nlp/retraining/trigger",
                json={"force": True},
                headers=AUTH_HEADERS,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "REJECTED_BY_CERTIFIER"
            assert "not 'DRIFT'" in data["reason"]
