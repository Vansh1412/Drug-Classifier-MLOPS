"""
src/retraining.py — Automated Retraining Trigger Engine + Quality Gate + Lifecycle Management.

Connects Sprint 5 Continuous Monitoring drift reports to a controlled, reproducible,
and reversible model retraining and promotion lifecycle.

Guarantees:
1. Current champion is protected at all costs during retraining.
2. Retraining outputs to an isolated challenger workspace (Model/challengers/).
3. Idempotent trigger guards prevent duplicate retraining runs.
4. Dual Quality Gate enforces absolute standards and comparative non-regression.
5. Atomic promotion preserves previous champion in Model/archive/.
6. Automatic immediate rollback if post-promotion validation fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import skops.io as sio
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.champion_selection import select_champion
from src.config import load_config
from src.data_validation import load_and_validate
from src.evaluate import train_and_evaluate
from src.model_registry import build_model_candidates, get_registry
from src.preprocessing import build_preprocessor

logger = logging.getLogger("src.retraining")


class RetrainingState(str, Enum):
    """Lifecycle states for the MedMLOps-X retraining state machine."""

    IDLE = "IDLE"
    TRIGGERED = "TRIGGERED"
    VALIDATING = "VALIDATING"
    RETRAINING = "RETRAINING"
    EVALUATING = "EVALUATING"
    QUALITY_GATE = "QUALITY_GATE"
    READY_FOR_PROMOTION = "READY_FOR_PROMOTION"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"


class DriftCertifier:
    """Evaluates drift reports to determine if automated retraining is justified."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.retrain_cfg = self.config.get("retraining", {})
        self.min_samples = self.retrain_cfg.get("min_monitoring_samples", 20)
        self.core_features = set(
            self.config.get(
                "data",
                {},
            ).get(
                "columns",
                ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"],
            )
        ) - {"Drug"}

    def certify(self, report: dict[str, Any]) -> tuple[bool, str]:
        """Certify whether a monitoring report warrants automated retraining.

        Args:
            report: Monitoring report dictionary matching Sprint 5 schema.

        Returns:
            Tuple of (is_certified, certification_reason).
        """
        summary = report.get("summary", {})
        overall_status = summary.get("overall_status", "PASS")

        # 1. Must have overall drift flag
        if overall_status != "DRIFT":
            return (
                False,
                f"Overall status is '{overall_status}', not 'DRIFT'. Retraining not required.",
            )

        # 2. Schema status must be PASS (schema breaches require pipeline/contract fixes, not retraining)
        schema_status = summary.get("schema_status") or report.get(
            "schema_validation", {}
        ).get("status", "PASS")
        if schema_status != "PASS":
            return (
                False,
                f"Schema status is '{schema_status}'. Schema mismatch must be resolved before retraining.",
            )

        # 3. Data quality must not be ERROR
        quality_status = summary.get("data_quality_status") or report.get(
            "data_quality", {}
        ).get("status", "PASS")
        if quality_status == "ERROR":
            return (
                False,
                "Data quality status is 'ERROR'. Corrupt or empty data cannot trigger retraining.",
            )

        # 4. Minimum sample size constraint to avoid small-sample stochastic noise
        current_samples = summary.get("current_samples", 0)
        if current_samples < self.min_samples:
            return (
                False,
                f"Insufficient serving samples ({current_samples} < {self.min_samples}) for reliable certification.",
            )

        # 5. Core feature drift intersection
        drifted_features = set(summary.get("drifted_features", []))
        core_drifted = drifted_features.intersection(self.core_features)
        if not core_drifted and not summary.get("prediction_drift_detected"):
            return (
                False,
                "No core predictive features or predictions exhibited certified drift.",
            )

        return (
            True,
            f"Drift certified: {len(core_drifted)} core features drifted ({sorted(core_drifted)}) with N={current_samples} samples.",
        )


class IdempotencyManager:
    """Prevents duplicate retraining triggers for identical drift events."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        base_dir: Path | str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent.parent)
        history_rel = self.config.get("retraining", {}).get(
            "trigger_history_path", "monitoring/retraining_history.json"
        )
        self.history_path = (self.base_dir / history_rel).resolve()
        self._ensure_history()

    def _ensure_history(self) -> dict[str, Any]:
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.warning(f"Failed to read trigger history: {exc}")
        history: dict[str, Any] = {"triggers": [], "processed_hashes": []}
        self._save_history(history)
        return history

    def _save_history(self, data: dict[str, Any]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.history_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        tmp.replace(self.history_path)

    def compute_hash(
        self, report_id: str, champion_version: str, dataset_hash: str
    ) -> str:
        """Compute SHA256 idempotency fingerprint."""
        raw = f"{report_id}::{champion_version}::{dataset_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def is_duplicate(self, trigger_hash: str) -> bool:
        """Check whether trigger hash has already been executed."""
        history = self._ensure_history()
        return trigger_hash in history.get("processed_hashes", [])

    def record_trigger(self, record: dict[str, Any]) -> None:
        """Record completed or attempted trigger in audit log."""
        history = self._ensure_history()
        history["triggers"].append(record)
        trigger_hash = record.get("trigger_hash")
        if trigger_hash and trigger_hash not in history.get("processed_hashes", []):
            history["processed_hashes"].append(trigger_hash)
        self._save_history(history)

    def get_history(self) -> list[dict[str, Any]]:
        """Return history of all retraining trigger records."""
        return self._ensure_history().get("triggers", [])


class QualityGate:
    """Evaluates candidates against absolute standards and champion non-regression."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.qg_cfg = self.config.get("retraining", {}).get("quality_gate", {})
        self.min_accuracy = self.qg_cfg.get("min_accuracy", 0.80)
        self.min_macro_f1 = self.qg_cfg.get("min_macro_f1", 0.75)
        self.min_recall = self.qg_cfg.get("min_recall", 0.70)
        self.max_f1_drop = self.qg_cfg.get("max_allowed_f1_drop", 0.02)
        self.max_cv_drop = self.qg_cfg.get("max_allowed_cv_f1_drop", 0.03)

    def evaluate(
        self,
        challenger_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Evaluate dual quality gate.

        Returns:
            Tuple of (passed, list_of_failure_reasons).
        """
        failures = []

        # Gate 1: Absolute Minimum Performance
        acc = challenger_metrics.get("accuracy", 0.0)
        f1 = challenger_metrics.get("macro_f1", 0.0)
        rec = challenger_metrics.get("macro_recall", 0.0)

        if acc < self.min_accuracy:
            failures.append(
                f"Accuracy {acc:.4f} below absolute threshold {self.min_accuracy:.4f}"
            )
        if f1 < self.min_macro_f1:
            failures.append(
                f"Macro F1 {f1:.4f} below absolute threshold {self.min_macro_f1:.4f}"
            )
        if rec < self.min_recall:
            failures.append(
                f"Macro Recall {rec:.4f} below absolute threshold {self.min_recall:.4f}"
            )

        # Gate 2: Comparative Non-Regression vs Active Champion
        champ_f1 = champion_metrics.get("macro_f1", 0.0)
        champ_cv = champion_metrics.get("cv_macro_f1_mean", 0.0)
        chal_cv = challenger_metrics.get("cv_macro_f1_mean", 0.0)

        if f1 < (champ_f1 - self.max_f1_drop):
            failures.append(
                f"Macro F1 {f1:.4f} regressed compared to champion {champ_f1:.4f} (drop > {self.max_f1_drop:.4f})"
            )

        if champ_cv > 0.0 and chal_cv < (champ_cv - self.max_cv_drop):
            failures.append(
                f"CV Macro F1 {chal_cv:.4f} regressed compared to champion {champ_cv:.4f} (drop > {self.max_cv_drop:.4f})"
            )

        return (len(failures) == 0, failures)


class RetrainingOrchestrator:
    """Orchestrates end-to-end retraining, competition, quality gates, and promotion."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        base_dir: Path | str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent.parent)
        self.registry = get_registry(config=self.config, base_dir=self.base_dir)
        self.certifier = DriftCertifier(config=self.config)
        self.idempotency = IdempotencyManager(
            config=self.config, base_dir=self.base_dir
        )
        self.quality_gate = QualityGate(config=self.config)
        self.state = RetrainingState.IDLE
        self.current_job: dict[str, Any] | None = None

    def get_status(self) -> dict[str, Any]:
        """Return current status of the retraining state machine."""
        return {
            "state": self.state.value,
            "current_job": self.current_job,
            "active_champion": self.registry.get_active_champion(),
            "previous_champion": self.registry.get_previous_champion(),
            "total_triggers": len(self.idempotency.get_history()),
        }

    def trigger_retraining(
        self,
        report: dict[str, Any] | None = None,
        report_id: str | None = None,
        force: bool = False,
        auto_promote: bool = True,
    ) -> dict[str, Any]:
        """Execute the certified retraining lifecycle.

        Workflow:
          1. Resolve drift report (from dict, report_id, or latest on disk).
          2. Check idempotency.
          3. Certify drift.
          4. Execute isolated retraining into Model/challengers/.
          5. Evaluate dual quality gate.
          6. If passes, promote challenger and execute smoke test.
          7. If smoke test fails, rollback immediately.
        """
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        job_id = f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(timestamp.encode()).hexdigest()[:6]}"

        self.state = RetrainingState.TRIGGERED
        self.current_job = {
            "job_id": job_id,
            "state": self.state.value,
            "started_at": timestamp,
        }

        try:
            # Step 1: Resolve Report
            resolved_report = self._resolve_report(report, report_id)
            if not resolved_report:
                self.state = RetrainingState.FAILED
                res = {
                    "job_id": job_id,
                    "status": "FAILED",
                    "reason": "No valid monitoring report could be resolved.",
                }
                self.current_job = res
                return res

            target_report_id = resolved_report.get("report_id", "unknown_report")
            self.current_job["report_id"] = target_report_id

            # Step 2: Idempotency Check
            active_champ = self.registry.get_active_champion()
            champ_ver = active_champ.get("version", "0.2.0")
            dvc_hash = resolved_report.get("model_metadata", {}).get(
                "dvc_dataset_hash", "unknown_dvc"
            )

            trigger_hash = self.idempotency.compute_hash(
                target_report_id, champ_ver, dvc_hash
            )
            self.current_job["trigger_hash"] = trigger_hash

            if not force and self.idempotency.is_duplicate(trigger_hash):
                self.state = RetrainingState.IDLE
                res = {
                    "job_id": job_id,
                    "status": "SKIPPED",
                    "reason": f"Idempotent duplicate: report '{target_report_id}' was already processed for champion v{champ_ver}.",
                    "trigger_hash": trigger_hash,
                }
                self.current_job = res
                return res

            # Step 3: Drift Certification
            self.state = RetrainingState.VALIDATING
            is_certified, cert_reason = self.certifier.certify(resolved_report)
            if not is_certified:
                self.state = RetrainingState.REJECTED
                res = {
                    "job_id": job_id,
                    "status": "REJECTED",
                    "reason": f"Drift certification failed: {cert_reason}",
                    "trigger_hash": trigger_hash,
                }
                self.idempotency.record_trigger(
                    {**res, "timestamp": timestamp, "certified": False}
                )
                self.current_job = res
                return res

            # Step 4: Isolated Retraining & Candidate Competition
            self.state = RetrainingState.RETRAINING
            challenger_result, challenger_pipeline = self._train_isolated_candidates()

            # Step 5: Dual Quality Gate
            self.state = RetrainingState.QUALITY_GATE
            qg_passed, qg_failures = self.quality_gate.evaluate(
                challenger_metrics=challenger_result,
                champion_metrics=active_champ.get("metrics", {}),
            )

            if not qg_passed:
                self.state = RetrainingState.REJECTED
                res = {
                    "job_id": job_id,
                    "status": "REJECTED",
                    "reason": f"Quality Gate failed: {'; '.join(qg_failures)}",
                    "challenger": challenger_result["model_name"],
                    "challenger_metrics": {
                        "accuracy": challenger_result.get("accuracy"),
                        "macro_f1": challenger_result.get("macro_f1"),
                    },
                    "trigger_hash": trigger_hash,
                }
                self.idempotency.record_trigger(
                    {
                        **res,
                        "timestamp": timestamp,
                        "certified": True,
                        "promoted": False,
                    }
                )
                self.current_job = res
                return res

            # Step 6: Promotion or Staging
            self.state = RetrainingState.READY_FOR_PROMOTION
            new_version = self._compute_next_version(champ_ver)

            if not auto_promote:
                res = {
                    "job_id": job_id,
                    "status": "READY_FOR_PROMOTION",
                    "new_version": new_version,
                    "challenger": challenger_result["model_name"],
                    "challenger_metrics": challenger_result,
                    "trigger_hash": trigger_hash,
                }
                self.idempotency.record_trigger(
                    {
                        **res,
                        "timestamp": timestamp,
                        "certified": True,
                        "promoted": False,
                    }
                )
                self.current_job = res
                return res

            # Execute Atomic Promotion
            promoted = self.registry.promote_challenger(
                challenger_artifact_path=self.registry.challengers_dir
                / "challenger_pipeline.skops",
                challenger_metadata=challenger_result,
                new_version=new_version,
                auto_smoke_test=True,
            )

            duration = round(time.perf_counter() - start_time, 2)
            if promoted:
                self.state = RetrainingState.PROMOTED
                res = {
                    "job_id": job_id,
                    "status": "PROMOTED",
                    "new_version": new_version,
                    "champion_model": challenger_result["model_name"],
                    "metrics": {
                        "accuracy": challenger_result.get("accuracy"),
                        "macro_f1": challenger_result.get("macro_f1"),
                        "cv_macro_f1_mean": challenger_result.get("cv_macro_f1_mean"),
                    },
                    "duration_seconds": duration,
                    "trigger_hash": trigger_hash,
                }
                self.idempotency.record_trigger(
                    {
                        **res,
                        "timestamp": timestamp,
                        "certified": True,
                        "promoted": True,
                    }
                )
                self.current_job = res
                return res
            else:
                self.state = RetrainingState.ROLLED_BACK
                res = {
                    "job_id": job_id,
                    "status": "ROLLED_BACK",
                    "reason": "Post-promotion smoke test failed; automatically rolled back to previous champion.",
                    "trigger_hash": trigger_hash,
                }
                self.idempotency.record_trigger(
                    {
                        **res,
                        "timestamp": timestamp,
                        "certified": True,
                        "promoted": False,
                    }
                )
                self.current_job = res
                return res

        except Exception as exc:
            logger.exception(f"Retraining workflow encountered error: {exc}")
            self.state = RetrainingState.FAILED
            res = {
                "job_id": job_id,
                "status": "FAILED",
                "error": str(exc),
            }
            self.current_job = res
            return res

    def _resolve_report(
        self,
        report_dict: dict[str, Any] | None,
        report_id: str | None,
    ) -> dict[str, Any] | None:
        """Resolve monitoring report from parameters or disk."""
        if report_dict is not None:
            return report_dict

        reports_dir = (
            self.base_dir
            / self.config.get("monitoring", {}).get("reports_dir", "monitoring/reports")
        ).resolve()
        if not reports_dir.exists():
            return None

        if report_id is not None:
            target_file = (reports_dir / f"{report_id}.json").resolve()
            if target_file.is_relative_to(reports_dir) and target_file.exists():
                with open(target_file, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            return None

        # Fallback to latest report on disk
        report_files = sorted(
            reports_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if report_files:
            with open(report_files[0], "r", encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def _train_isolated_candidates(self) -> tuple[dict[str, Any], Any]:
        """Execute multi-model competition writing exclusively to Model/challengers/."""
        data_path = (self.base_dir / self.config["data"]["data_path"]).resolve()
        drug_df = load_and_validate(data_path, self.config)

        shuffle_seed = self.config["data"]["shuffle_random_state"]
        split_seed = self.config["data"]["split_random_state"]
        test_size = self.config["data"]["test_size"]

        drug_df = drug_df.sample(frac=1, random_state=shuffle_seed).reset_index(
            drop=True
        )
        X = drug_df.drop(self.config["data"]["target_column"], axis=1).values
        y = drug_df[self.config["data"]["target_column"]].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed, stratify=y
        )

        candidates = build_model_candidates(self.config)
        all_results = []

        for name, estimator in candidates.items():
            preprocessor = build_preprocessor(self.config)
            pipeline = Pipeline(
                [
                    ("preprocessing", preprocessor),
                    ("model", estimator),
                ]
            )
            model_params = {
                k: v
                for k, v in self.config["models"][name]["params"].items()
                if not k.startswith("#")
            }
            res = train_and_evaluate(
                pipeline,
                X_train,
                y_train,
                X_test,
                y_test,
                self.config,
                name,
                model_params,
            )
            all_results.append({**res, "_pipeline": pipeline})

        serializable_results = [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in all_results
        ]
        pipeline_map = {r["model_name"]: r["_pipeline"] for r in all_results}

        champion_res, _ = select_champion(serializable_results, self.config)
        champion_pipeline = pipeline_map[champion_res["model_name"]]

        # Persist ONLY to isolated challengers directory
        chal_art = self.registry.challengers_dir / "challenger_pipeline.skops"
        chal_meta = self.registry.challengers_dir / "challenger_metadata.json"

        sio.dump(champion_pipeline, chal_art)
        with open(chal_meta, "w", encoding="utf-8") as fh:
            json.dump(champion_res, fh, indent=2)

        return (champion_res, champion_pipeline)

    def _compute_next_version(self, current_version: str) -> str:
        """Increment patch version (e.g. 0.2.0 -> 0.2.1)."""
        parts = current_version.split(".")
        if len(parts) == 3 and parts[2].isdigit():
            return f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
        return f"{current_version}.1"


_orchestrator_instance: RetrainingOrchestrator | None = None


def get_orchestrator(
    config: dict[str, Any] | None = None,
    base_dir: Path | str | None = None,
) -> RetrainingOrchestrator:
    """Retrieve or initialize the global RetrainingOrchestrator singleton."""
    global _orchestrator_instance
    if _orchestrator_instance is None or config is not None or base_dir is not None:
        _orchestrator_instance = RetrainingOrchestrator(
            config=config, base_dir=base_dir
        )
    return _orchestrator_instance
