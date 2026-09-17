"""
tests/test_sprint6.py — Automated Test Suite for Sprint 6:
Automated Retraining Trigger + Champion/Challenger Quality Gate + Promotion/Rollback.

Verifies:
- Drift certification engine (valid drift, benign drift, small sample, schema failure)
- Idempotency manager and duplicate prevention
- Dual Quality Gate (absolute thresholds + comparative champion non-regression)
- Model registry manifest management, atomic archiving, promotion, and rollback
- Automatic rollback on post-promotion verification failure
- Isolated challenger training without champion mutation
- API security authorization (X-API-Key verification on lifecycle endpoints)
- Management endpoints (/retraining/trigger, /retraining/status, /models/champion, /models/rollback)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.main as api_module
from src.config import load_config
from src.model_registry import ModelRegistry
from src.retraining import (
    DriftCertifier,
    IdempotencyManager,
    QualityGate,
    RetrainingOrchestrator,
    RetrainingState,
)

client = TestClient(api_module.app)
AUTH_HEADERS = {"X-API-Key": "medmlops-admin-key-sprint6"}


@pytest.fixture
def sample_drift_report() -> dict:
    """Fixture providing a certified drift report."""
    return {
        "report_id": "drift_test_fixture_001",
        "timestamp": "2026-09-17T20:00:00+00:00",
        "model_metadata": {
            "model_name": "xgboost",
            "model_version": "0.2.0",
            "git_commit": "abcdef123456",
            "dvc_dataset_hash": "dvc789hash",
        },
        "summary": {
            "overall_status": "DRIFT",
            "overall_severity": "HIGH",
            "feature_drift_detected": True,
            "prediction_drift_detected": True,
            "schema_status": "PASS",
            "data_quality_status": "PASS",
            "drifted_features": ["Age", "Na_to_K"],
            "drifted_features_count": 2,
            "total_features_monitored": 5,
            "reference_samples": 200,
            "current_samples": 50,
        },
        "schema_validation": {
            "status": "PASS",
            "expected_columns": ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"],
            "missing_columns": [],
            "unexpected_columns": [],
            "type_mismatches": [],
        },
        "data_quality": {
            "status": "PASS",
            "null_rates": {
                "Age": 0.0,
                "Sex": 0.0,
                "BP": 0.0,
                "Cholesterol": 0.0,
                "Na_to_K": 0.0,
            },
            "out_of_range": {},
            "unseen_categories": {},
        },
    }


# ─────────────────────────────────────────────────────────────
# 1. Drift Certification Tests
# ─────────────────────────────────────────────────────────────
class TestDriftCertification:
    """Validate that only genuine, actionable drift triggers retraining."""

    @pytest.fixture
    def certifier(self):
        return DriftCertifier()

    def test_certify_valid_drift_report(self, certifier, sample_drift_report):
        is_certified, reason = certifier.certify(sample_drift_report)
        assert is_certified is True
        assert "Drift certified" in reason

    def test_reject_non_drift_report(self, certifier, sample_drift_report):
        sample_drift_report["summary"]["overall_status"] = "PASS"
        is_certified, reason = certifier.certify(sample_drift_report)
        assert is_certified is False
        assert "not 'DRIFT'" in reason

    def test_reject_schema_failure(self, certifier, sample_drift_report):
        sample_drift_report["summary"]["schema_status"] = "ERROR"
        sample_drift_report["schema_validation"]["status"] = "ERROR"
        is_certified, reason = certifier.certify(sample_drift_report)
        assert is_certified is False
        assert "Schema mismatch" in reason

    def test_reject_data_quality_error(self, certifier, sample_drift_report):
        sample_drift_report["summary"]["data_quality_status"] = "ERROR"
        sample_drift_report["data_quality"]["status"] = "ERROR"
        is_certified, reason = certifier.certify(sample_drift_report)
        assert is_certified is False
        assert "Data quality status is 'ERROR'" in reason

    def test_reject_insufficient_sample_size(self, certifier, sample_drift_report):
        sample_drift_report["summary"]["current_samples"] = 5  # Below threshold 20
        is_certified, reason = certifier.certify(sample_drift_report)
        assert is_certified is False
        assert "Insufficient serving samples" in reason

    def test_reject_no_core_feature_drift(self, certifier, sample_drift_report):
        sample_drift_report["summary"]["drifted_features"] = ["unrelated_meta_col"]
        sample_drift_report["summary"]["prediction_drift_detected"] = False
        is_certified, reason = certifier.certify(sample_drift_report)
        assert is_certified is False
        assert "No core predictive features" in reason


# ─────────────────────────────────────────────────────────────
# 2. Idempotency Manager Tests
# ─────────────────────────────────────────────────────────────
class TestIdempotencyManager:
    """Validate duplicate trigger prevention."""

    @pytest.fixture
    def manager(self, tmp_path):
        cfg = {
            "retraining": {
                "trigger_history_path": str(tmp_path / "test_retraining_history.json")
            }
        }
        return IdempotencyManager(config=cfg, base_dir=tmp_path)

    def test_compute_hash_deterministic(self, manager):
        h1 = manager.compute_hash("rep1", "0.2.0", "hash123")
        h2 = manager.compute_hash("rep1", "0.2.0", "hash123")
        assert h1 == h2

    def test_record_and_detect_duplicate(self, manager):
        trigger_hash = manager.compute_hash("rep1", "0.2.0", "hash123")
        assert manager.is_duplicate(trigger_hash) is False

        manager.record_trigger(
            {
                "trigger_hash": trigger_hash,
                "report_id": "rep1",
                "status": "PROMOTED",
            }
        )
        assert manager.is_duplicate(trigger_hash) is True


# ─────────────────────────────────────────────────────────────
# 3. Quality Gate Tests
# ─────────────────────────────────────────────────────────────
class TestQualityGate:
    """Validate dual quality gate: absolute thresholds + champion comparison."""

    @pytest.fixture
    def quality_gate(self):
        return QualityGate()

    def test_quality_gate_passes_equal_or_better(self, quality_gate):
        champ_metrics = {
            "accuracy": 0.98,
            "macro_f1": 0.97,
            "macro_recall": 0.96,
            "cv_macro_f1_mean": 0.95,
        }
        chal_metrics = {
            "accuracy": 0.99,
            "macro_f1": 0.98,
            "macro_recall": 0.97,
            "cv_macro_f1_mean": 0.96,
        }
        passed, failures = quality_gate.evaluate(chal_metrics, champ_metrics)
        assert passed is True
        assert len(failures) == 0

    def test_quality_gate_fails_below_absolute_minimum(self, quality_gate):
        champ_metrics = {"accuracy": 0.95, "macro_f1": 0.90}
        chal_metrics = {
            "accuracy": 0.70,  # Below 0.80
            "macro_f1": 0.65,  # Below 0.75
            "macro_recall": 0.60,  # Below 0.70
            "cv_macro_f1_mean": 0.65,
        }
        passed, failures = quality_gate.evaluate(chal_metrics, champ_metrics)
        assert passed is False
        assert any("Accuracy" in f for f in failures)
        assert any("Macro F1" in f for f in failures)

    def test_quality_gate_fails_on_macro_f1_regression(self, quality_gate):
        champ_metrics = {
            "accuracy": 0.95,
            "macro_f1": 0.95,
            "cv_macro_f1_mean": 0.92,
        }
        chal_metrics = {
            "accuracy": 0.92,
            "macro_f1": 0.91,  # Drop of 0.04 > allowed drop of 0.02
            "macro_recall": 0.90,
            "cv_macro_f1_mean": 0.91,
        }
        passed, failures = quality_gate.evaluate(chal_metrics, champ_metrics)
        assert passed is False
        assert any("regressed compared to champion" in f for f in failures)


# ─────────────────────────────────────────────────────────────
# 4. Model Registry & Rollback Tests
# ─────────────────────────────────────────────────────────────
class TestModelRegistry:
    """Validate model manifest, atomic archiving, promotion, and rollback."""

    @pytest.fixture
    def registry(self, tmp_path):
        # Create mock model directory structure
        model_dir = tmp_path / "Model"
        model_dir.mkdir(parents=True)

        # Copy active champion artifact
        real_art = PROJECT_ROOT / "Model" / "drug_pipeline.skops"
        mock_art = model_dir / "drug_pipeline.skops"
        shutil.copy2(real_art, mock_art)

        # Create mock active metadata
        meta = {
            "model_name": "xgboost",
            "model_class": "XGBClassifier",
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "cv_macro_f1_mean": 0.9633,
            "training_timestamp": "2026-09-17T12:00:00+00:00",
        }
        with open(model_dir / "model_metadata.json", "w") as fh:
            json.dump(meta, fh)

        cfg = {
            "champion": {
                "artifact_path": "Model/drug_pipeline.skops",
                "metadata_path": "Model/model_metadata.json",
            },
            "retraining": {
                "manifest_path": "Model/registry_manifest.json",
                "archive_dir": "Model/archive",
                "challengers_dir": "Model/challengers",
            },
            "security": load_config().get("security", {}),
        }
        return ModelRegistry(config=cfg, base_dir=tmp_path)

    def test_manifest_initialization(self, registry):
        manifest = registry.get_manifest()
        assert manifest["active_champion"]["version"] == "0.2.0"
        assert manifest["active_champion"]["model_name"] == "xgboost"
        assert manifest["previous_champion"] is None

    def test_archive_champion(self, registry):
        archived = registry.archive_champion("0.2.0")
        assert (registry.base_dir / archived["artifact_path"]).exists()
        assert (registry.base_dir / archived["metadata_path"]).exists()

    def test_promote_challenger_atomic(self, registry, tmp_path):
        # Create dummy challenger artifact
        chal_art = registry.challengers_dir / "challenger_pipeline.skops"
        shutil.copy2(registry.active_artifact_path, chal_art)

        chal_meta = {
            "model_name": "extra_trees",
            "accuracy": 0.98,
            "macro_f1": 0.97,
            "cv_macro_f1_mean": 0.95,
        }

        success = registry.promote_challenger(
            challenger_artifact_path=chal_art,
            challenger_metadata=chal_meta,
            new_version="0.2.1",
            auto_smoke_test=True,
        )
        assert success is True

        manifest = registry.get_manifest()
        assert manifest["active_champion"]["version"] == "0.2.1"
        assert manifest["active_champion"]["model_name"] == "extra_trees"
        assert manifest["previous_champion"]["version"] == "0.2.0"

    def test_rollback_restores_previous_champion(self, registry):
        # Promote first
        chal_art = registry.challengers_dir / "challenger_pipeline.skops"
        shutil.copy2(registry.active_artifact_path, chal_art)
        chal_meta = {"model_name": "extra_trees", "macro_f1": 0.98}
        registry.promote_challenger(chal_art, chal_meta, "0.2.1")

        # Rollback
        success = registry.rollback(reason="Test rollback")
        assert success is True

        manifest = registry.get_manifest()
        assert manifest["active_champion"]["version"] == "0.2.0"
        assert manifest["active_champion"]["model_name"] == "xgboost"

    def test_rollback_without_previous_raises_error(self, registry):
        with pytest.raises(RuntimeError, match="No previous champion"):
            registry.rollback()

    def test_post_promotion_smoke_test(self, registry):
        assert registry.post_promotion_smoke_test() is True


# ─────────────────────────────────────────────────────────────
# 5. Retraining Orchestrator Tests
# ─────────────────────────────────────────────────────────────
class TestRetrainingOrchestrator:
    """Validate full end-to-end retraining state machine and workflow."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        # Setup mock directories in tmp_path
        model_dir = tmp_path / "Model"
        model_dir.mkdir(parents=True)
        shutil.copy2(
            PROJECT_ROOT / "Model" / "drug_pipeline.skops",
            model_dir / "drug_pipeline.skops",
        )
        shutil.copy2(
            PROJECT_ROOT / "Model" / "model_metadata.json",
            model_dir / "model_metadata.json",
        )

        cfg = load_config()
        cfg["retraining"]["manifest_path"] = "Model/registry_manifest.json"
        cfg["retraining"]["archive_dir"] = "Model/archive"
        cfg["retraining"]["challengers_dir"] = "Model/challengers"
        cfg["retraining"]["trigger_history_path"] = "monitoring/retraining_history.json"

        return RetrainingOrchestrator(config=cfg, base_dir=tmp_path)

    def test_orchestrator_initial_status(self, orchestrator):
        status = orchestrator.get_status()
        assert status["state"] == RetrainingState.IDLE.value
        assert "active_champion" in status

    def test_orchestrator_rejects_uncertified_report(
        self, orchestrator, sample_drift_report
    ):
        sample_drift_report["summary"]["overall_status"] = "PASS"
        res = orchestrator.trigger_retraining(report=sample_drift_report)
        assert res["status"] == "REJECTED"
        assert "certification failed" in res["reason"]

    def test_orchestrator_skips_idempotent_duplicate(
        self, orchestrator, sample_drift_report
    ):
        # Mock training to speed up test execution
        with patch.object(
            orchestrator,
            "_train_isolated_candidates",
            return_value=(
                {
                    "model_name": "xgboost",
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                    "macro_recall": 1.0,
                    "cv_macro_f1_mean": 0.9633,
                },
                MagicMock(),
            ),
        ):
            # Ensure mock candidate artifact exists
            mock_art = (
                orchestrator.registry.challengers_dir / "challenger_pipeline.skops"
            )
            shutil.copy2(orchestrator.registry.active_artifact_path, mock_art)

            # First execution
            res1 = orchestrator.trigger_retraining(
                report=sample_drift_report, auto_promote=False
            )
            assert res1["status"] == "READY_FOR_PROMOTION"

            # Duplicate execution
            res2 = orchestrator.trigger_retraining(
                report=sample_drift_report, force=False
            )
            assert res2["status"] == "SKIPPED"
            assert "Idempotent duplicate" in res2["reason"]

    def test_orchestrator_auto_rollback_on_smoke_failure(
        self, orchestrator, sample_drift_report
    ):
        with patch.object(
            orchestrator,
            "_train_isolated_candidates",
            return_value=(
                {
                    "model_name": "xgboost",
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                    "macro_recall": 1.0,
                    "cv_macro_f1_mean": 0.9633,
                },
                MagicMock(),
            ),
        ):
            # Mock registry.post_promotion_smoke_test to fail
            with patch.object(
                orchestrator.registry,
                "post_promotion_smoke_test",
                return_value=False,
            ):
                # Create mock candidate artifact
                mock_art = (
                    orchestrator.registry.challengers_dir / "challenger_pipeline.skops"
                )
                shutil.copy2(orchestrator.registry.active_artifact_path, mock_art)

                res = orchestrator.trigger_retraining(
                    report=sample_drift_report, force=True, auto_promote=True
                )
                assert res["status"] == "ROLLED_BACK"
                assert "rolled back" in res["reason"]


# ─────────────────────────────────────────────────────────────
# 6. Management API Endpoints & Security Tests
# ─────────────────────────────────────────────────────────────
class TestManagementAPI:
    """Validate FastAPI endpoints for retraining, rollback, and champion lineage."""

    def test_get_champion_details_public(self):
        resp = client.get("/models/champion")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_champion" in data
        assert data["active_champion"]["model_name"] == "xgboost"

    def test_get_retraining_status_public(self):
        resp = client.get("/retraining/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "state" in data
        assert "active_champion" in data

    def test_trigger_retraining_requires_authorization(self):
        resp = client.post("/retraining/trigger", json={})
        assert resp.status_code == 401
        assert "Unauthorized" in resp.json()["detail"]

    def test_rollback_requires_authorization(self):
        resp = client.post("/models/rollback", json={"reason": "Testing auth gate"})
        assert resp.status_code == 401
        assert "Unauthorized" in resp.json()["detail"]

    def test_trigger_retraining_authorized_execution(self, sample_drift_report):
        with patch.object(
            RetrainingOrchestrator,
            "_resolve_report",
            return_value=sample_drift_report,
        ):
            with patch.object(
                RetrainingOrchestrator,
                "_train_isolated_candidates",
                return_value=(
                    {
                        "model_name": "xgboost",
                        "accuracy": 1.0,
                        "macro_f1": 1.0,
                        "macro_recall": 1.0,
                        "cv_macro_f1_mean": 0.9633,
                    },
                    MagicMock(),
                ),
            ):
                resp = client.post(
                    "/retraining/trigger",
                    headers=AUTH_HEADERS,
                    json={"force": True, "auto_promote": False},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] in ["READY_FOR_PROMOTION", "PROMOTED"]
