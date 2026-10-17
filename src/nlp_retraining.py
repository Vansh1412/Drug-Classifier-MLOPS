"""
src/nlp_retraining.py — Automated NLP Retraining Trigger Engine, Quality Gate & Promotion.

Connects Sprint 10 NLP Continuous Monitoring drift reports to a controlled, reproducible,
and reversible NLP model retraining and promotion lifecycle.

Guarantees:
1. Retraining triggers ONLY on certified DRIFT (rejects PASS, WARNING, ERROR).
2. Existing structured champion (Model/drug_pipeline.skops) is NEVER modified.
3. Dedicated isolated challenger workspace: Model/nlp/challengers/<retraining_id>/.
4. Leakage-safe group-aware splitting and cross-validation (StratifiedGroupKFold on review_clean_hash).
5. Supports initial state: NLP Champion = NONE (bootstrap promotion without comparative gate).
6. Dual Quality Gate: absolute standards (Macro F1 >= 0.70, Accuracy >= 0.70, Class recall >= 0.40)
   plus comparative champion non-regression when a champion exists.
7. Atomic promotion with immediate automatic rollback upon smoke test failure.
8. Complete audit history persisted to monitoring/nlp_retraining_history.json.
9. Full reproducibility metadata logged to training manifest and MLflow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import skops.io as sio

from src.config import load_config
from src.nlp_competition import (
    build_nlp_pipeline,
    derive_sentiment_target,
    evaluate_cv_leakage_safe,
    evaluate_test_set,
)
from src.nlp_model_registry import get_nlp_registry
from src.nlp_preprocessing import split_reviews_leakage_safe
from src.nlp_tracking import (
    get_dvc_dataset_hash,
    get_git_commit_sha,
    log_nlp_candidate_run,
    setup_nlp_mlflow,
)

logger = logging.getLogger("src.nlp_retraining")


class NLPRetrainingState(str, Enum):
    """Lifecycle states for the MedMLOps-X NLP retraining state machine."""

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


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(path)
    if not p.exists():
        return "unknown"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class NLPDriftCertifier:
    """Evaluates NLP drift reports to determine if automated retraining is justified."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.retrain_cfg = self.config.get("nlp_retraining", {})
        self.min_samples = self.retrain_cfg.get("min_monitoring_samples", 20)

    def certify(self, report: dict[str, Any]) -> tuple[bool, str]:
        """Certify whether an NLP monitoring report warrants automated retraining.

        Rules:
        - overall_status must be 'DRIFT' (PASS, WARNING, ERROR rejected).
        - data_quality status must not be 'ERROR'.
        - current serving samples >= min_samples.
        """
        cert = report.get("certification", {})
        overall_status = cert.get("overall_status", "PASS")

        # 1. Must have overall DRIFT certification
        if overall_status != "DRIFT":
            return (
                False,
                f"Overall status is '{overall_status}', not 'DRIFT'. Retraining not required.",
            )

        # 2. Data quality must not be ERROR
        quality = report.get("data_quality", {})
        quality_status = quality.get("status", "PASS")
        if quality_status == "ERROR":
            return (
                False,
                "Data quality status is 'ERROR'. Corrupt or invalid review data cannot trigger retraining.",
            )

        # 3. Minimum sample size constraint
        current_data = report.get("current", {})
        current_samples = current_data.get("rows", 0)
        if current_samples < self.min_samples:
            return (
                False,
                f"Insufficient serving samples ({current_samples} < {self.min_samples}) for reliable retraining.",
            )

        # 4. Confirm drift detected flag and signals
        drift_detected = cert.get("drift_detected", False)
        signals = cert.get("signals", [])
        if not drift_detected and not signals:
            return (
                False,
                "No drift signals or drift detection flag recorded in certification.",
            )

        return (
            True,
            f"NLP drift certified: {len(signals)} signals detected ({', '.join(signals[:3])}) with N={current_samples} samples.",
        )


class NLPIdempotencyManager:
    """Prevents duplicate retraining triggers for identical drift events."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        base_dir: Path | str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent.parent)
        history_rel = self.config.get("nlp_retraining", {}).get(
            "trigger_history_path", "monitoring/nlp_retraining_history.json"
        )
        self.history_path = (self.base_dir / history_rel).resolve()
        self._ensure_history()

    def _ensure_history(self) -> dict[str, Any]:
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.warning(f"Failed to read NLP trigger history: {exc}")
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
        """Check whether trigger hash has already been processed."""
        history = self._ensure_history()
        return trigger_hash in history.get("processed_hashes", [])

    def record_trigger(self, record: dict[str, Any]) -> None:
        """Record completed or attempted retraining trigger in audit log."""
        history = self._ensure_history()
        history["triggers"].append(record)
        trigger_hash = record.get("trigger_hash")
        if trigger_hash and trigger_hash not in history.get("processed_hashes", []):
            history["processed_hashes"].append(trigger_hash)
        self._save_history(history)

    def get_history(self) -> list[dict[str, Any]]:
        """Return full history of all retraining trigger records."""
        return self._ensure_history().get("triggers", [])


class NLPQualityGate:
    """Evaluates NLP candidate models against absolute thresholds and champion non-regression."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.qg_cfg = self.config.get("nlp_retraining", {}).get("quality_gate", {})
        self.min_macro_f1 = float(self.qg_cfg.get("min_macro_f1", 0.70))
        self.min_accuracy = float(self.qg_cfg.get("min_accuracy", 0.70))
        self.min_class_recall = float(self.qg_cfg.get("min_class_recall", 0.40))
        self.max_allowed_f1_drop = float(self.qg_cfg.get("max_allowed_f1_drop", 0.02))

    def evaluate(
        self,
        cv_metrics: dict[str, Any],
        test_metrics: dict[str, Any],
        champion_metrics: dict[str, Any] | None = None,
        is_bootstrap: bool = False,
    ) -> tuple[bool, list[str]]:
        """Evaluate candidate model.

        If is_bootstrap or champion_metrics is None:
            Evaluates absolute quality gate only.
        If active champion exists:
            Evaluates absolute quality gate + comparative non-regression.

        Returns:
            Tuple of (passed: bool, failure_reasons: list[str]).
        """
        failures: list[str] = []

        # 1. Absolute Gate
        cv_f1 = cv_metrics.get("cv_macro_f1_mean", 0.0)
        if cv_f1 < self.min_macro_f1:
            failures.append(
                f"CV Macro F1 ({cv_f1:.4f}) below absolute threshold ({self.min_macro_f1:.4f})"
            )

        test_f1 = test_metrics.get("test_macro_f1", 0.0)
        if test_f1 < self.min_macro_f1:
            failures.append(
                f"Test Macro F1 ({test_f1:.4f}) below absolute threshold ({self.min_macro_f1:.4f})"
            )

        test_acc = test_metrics.get("test_accuracy", 0.0)
        if test_acc < self.min_accuracy:
            failures.append(
                f"Test Accuracy ({test_acc:.4f}) below absolute threshold ({self.min_accuracy:.4f})"
            )

        class_metrics = test_metrics.get("class_metrics", {})
        for cls_name, c_data in class_metrics.items():
            recall_val = c_data.get("recall", 0.0)
            if recall_val < self.min_class_recall:
                failures.append(
                    f"Class '{cls_name}' recall ({recall_val:.4f}) below minimum threshold ({self.min_class_recall:.4f})"
                )

        # 2. Comparative Non-Regression Gate (Only when champion exists)
        if not is_bootstrap and champion_metrics is not None:
            champ_f1 = float(
                champion_metrics.get("test_macro_f1")
                or champion_metrics.get("macro_f1", 0.0)
            )
            if test_f1 < (champ_f1 - self.max_allowed_f1_drop):
                failures.append(
                    f"Test Macro F1 ({test_f1:.4f}) regressed compared to champion ({champ_f1:.4f}) beyond margin ({self.max_allowed_f1_drop:.4f})"
                )

        return (len(failures) == 0, failures)


class NLPRetrainingOrchestrator:
    """Orchestrates end-to-end NLP retraining, model competition, quality gates, and promotion."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        base_dir: Path | str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent.parent)
        self.registry = get_nlp_registry(config=self.config, base_dir=self.base_dir)
        self.certifier = NLPDriftCertifier(config=self.config)
        self.idempotency = NLPIdempotencyManager(
            config=self.config, base_dir=self.base_dir
        )
        self.quality_gate = NLPQualityGate(config=self.config)
        self.state = NLPRetrainingState.IDLE
        self.current_job: dict[str, Any] | None = None

    def get_status(self) -> dict[str, Any]:
        """Return current status of the NLP retraining state machine."""
        return {
            "workload": "nlp",
            "state": self.state.value,
            "current_job": self.current_job,
            "active_champion": self.registry.get_active_champion(),
            "previous_champion": self.registry.get_previous_champion(),
            "manifest_status": self.registry.get_manifest().get("status", "NONE"),
            "total_triggers": len(self.idempotency.get_history()),
        }

    def _resolve_report(
        self,
        report: dict[str, Any] | None = None,
        report_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Resolve monitoring report from dict, specific ID, or latest on disk."""
        if report is not None:
            return report

        results_dir = (self.base_dir / "Results").resolve()
        latest_file = results_dir / "nlp_drift_report.json"

        if report_id:
            specific_file = results_dir / f"{report_id}.json"
            if specific_file.exists():
                try:
                    with open(specific_file, "r", encoding="utf-8") as fh:
                        return json.load(fh)
                except Exception as exc:
                    logger.warning(f"Failed to read report {specific_file}: {exc}")

        if latest_file.exists():
            try:
                with open(latest_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if report_id is None or data.get("report_id") == report_id:
                        return data
            except Exception as exc:
                logger.warning(f"Failed to read latest NLP report {latest_file}: {exc}")

        return None

    def trigger_retraining(
        self,
        report: dict[str, Any] | None = None,
        report_id: str | None = None,
        force: bool = False,
        auto_promote: bool = True,
    ) -> dict[str, Any]:
        """Execute the certified NLP retraining lifecycle.

        Workflow:
          1. Resolve NLP drift report.
          2. Check idempotency.
          3. Certify drift (reject PASS, WARNING, ERROR).
          4. Execute isolated challenger retraining into Model/nlp/challengers/<retraining_id>/.
          5. Evaluate NLP Quality Gate (supports bootstrap when champion=NONE).
          6. If passes, promote challenger to Model/nlp/champion/ and smoke test.
          7. If smoke test fails, rollback immediately.
        """
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        job_id = f"nlp_retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(timestamp.encode()).hexdigest()[:6]}"

        self.state = NLPRetrainingState.TRIGGERED
        self.current_job = {
            "job_id": job_id,
            "state": self.state.value,
            "started_at": timestamp,
        }

        try:
            # Step 1: Resolve Report
            resolved_report = self._resolve_report(report, report_id)
            if not resolved_report:
                self.state = NLPRetrainingState.FAILED
                res = {
                    "job_id": job_id,
                    "status": "FAILED",
                    "reason": "No valid NLP monitoring report could be resolved.",
                }
                self.current_job = res
                return res

            target_report_id = resolved_report.get("report_id", "unknown_report")
            self.current_job["report_id"] = target_report_id

            # Step 2: Idempotency Check
            active_champ = self.registry.get_active_champion()
            champ_ver = active_champ.get("version", "NONE") if active_champ else "NONE"
            dvc_hash = resolved_report.get("reference", {}).get(
                "dvc_hash", "unknown_dvc"
            )

            trigger_hash = self.idempotency.compute_hash(
                target_report_id, champ_ver, dvc_hash
            )
            self.current_job["trigger_hash"] = trigger_hash

            if not force and self.idempotency.is_duplicate(trigger_hash):
                self.state = NLPRetrainingState.IDLE
                res = {
                    "job_id": job_id,
                    "status": "SKIPPED",
                    "reason": f"Idempotent duplicate: NLP report '{target_report_id}' was already processed for champion v{champ_ver}.",
                    "trigger_hash": trigger_hash,
                }
                self.current_job = res
                return res

            # Step 3: Drift Certification
            self.state = NLPRetrainingState.VALIDATING
            is_certified, cert_reason = self.certifier.certify(resolved_report)
            if not is_certified:
                self.state = NLPRetrainingState.IDLE
                res = {
                    "job_id": job_id,
                    "status": "REJECTED_BY_CERTIFIER",
                    "reason": cert_reason,
                    "trigger_hash": trigger_hash,
                }
                self.current_job = res
                self.idempotency.record_trigger(
                    {
                        "job_id": job_id,
                        "report_id": target_report_id,
                        "status": "REJECTED_BY_CERTIFIER",
                        "reason": cert_reason,
                        "trigger_hash": trigger_hash,
                        "timestamp": timestamp,
                    }
                )
                return res

            # Step 4: Retraining in Isolated Challenger Workspace
            self.state = NLPRetrainingState.RETRAINING
            challenger_res = self._execute_retraining(job_id, resolved_report)

            # Step 5: Quality Gate
            self.state = NLPRetrainingState.QUALITY_GATE
            is_bootstrap = active_champ is None
            champ_metrics = active_champ.get("metrics") if active_champ else None

            gate_passed, gate_reasons = self.quality_gate.evaluate(
                cv_metrics=challenger_res["cv_metrics"],
                test_metrics=challenger_res["test_metrics"],
                champion_metrics=champ_metrics,
                is_bootstrap=is_bootstrap,
            )

            challenger_res["quality_gate"] = {
                "passed": gate_passed,
                "reasons": gate_reasons,
                "is_bootstrap": is_bootstrap,
            }

            if not gate_passed:
                self.state = NLPRetrainingState.REJECTED
                res = {
                    "job_id": job_id,
                    "status": "REJECTED",
                    "reason": "Challenger failed NLP quality gate criteria.",
                    "failure_reasons": gate_reasons,
                    "challenger": challenger_res,
                    "trigger_hash": trigger_hash,
                    "duration_seconds": round(time.perf_counter() - start_time, 2),
                }
                self.current_job = res
                self.idempotency.record_trigger(
                    {
                        "job_id": job_id,
                        "report_id": target_report_id,
                        "status": "REJECTED",
                        "candidate": challenger_res["model_name"],
                        "failure_reasons": gate_reasons,
                        "trigger_hash": trigger_hash,
                        "timestamp": timestamp,
                    }
                )
                return res

            # Step 6: Promotion & Smoke Test
            new_version = self._generate_candidate_version(champ_ver)
            if not auto_promote:
                self.state = NLPRetrainingState.READY_FOR_PROMOTION
                res = {
                    "job_id": job_id,
                    "status": "READY_FOR_PROMOTION",
                    "challenger": challenger_res,
                    "new_version": new_version,
                    "trigger_hash": trigger_hash,
                }
                self.current_job = res
                return res

            self.state = NLPRetrainingState.PROMOTED
            promotion_ok = self.registry.promote_challenger(
                challenger_artifact_path=challenger_res["artifact_path"],
                challenger_metadata=challenger_res["metadata"],
                new_version=new_version,
                auto_smoke_test=True,
            )

            if not promotion_ok:
                self.state = (
                    NLPRetrainingState.ROLLED_BACK
                    if not is_bootstrap
                    else NLPRetrainingState.FAILED
                )
                res = {
                    "job_id": job_id,
                    "status": "ROLLED_BACK" if not is_bootstrap else "FAILED",
                    "reason": "Post-promotion smoke test failed; state restored.",
                    "trigger_hash": trigger_hash,
                }
                self.current_job = res
                return res

            res = {
                "job_id": job_id,
                "status": "PROMOTED",
                "promoted_version": new_version,
                "model_name": challenger_res["model_name"],
                "is_bootstrap": is_bootstrap,
                "metrics": challenger_res["test_metrics"],
                "trigger_hash": trigger_hash,
                "duration_seconds": round(time.perf_counter() - start_time, 2),
            }
            self.current_job = res
            self.idempotency.record_trigger(
                {
                    "job_id": job_id,
                    "report_id": target_report_id,
                    "status": "PROMOTED",
                    "version": new_version,
                    "model_name": challenger_res["model_name"],
                    "trigger_hash": trigger_hash,
                    "timestamp": timestamp,
                }
            )
            return res

        except Exception as exc:
            logger.error(f"NLP Retraining pipeline crashed: {exc}", exc_info=True)
            self.state = NLPRetrainingState.FAILED
            res = {
                "job_id": job_id,
                "status": "FAILED",
                "error": str(exc),
            }
            self.current_job = res
            return res

    def _execute_retraining(
        self, job_id: str, report: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute isolated model competition and training on processed reviews dataset."""
        dataset_rel = self.config.get("nlp_retraining", {}).get(
            "dataset_path", "Data/reviews/processed/reviews_cleaned.parquet"
        )
        data_path = (self.base_dir / dataset_rel).resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Cleaned review dataset not found at {data_path}")

        dataset_sha256 = compute_file_sha256(data_path)
        dvc_hash = get_dvc_dataset_hash()
        git_sha = get_git_commit_sha()

        # Load processed review data
        df = pd.read_parquet(data_path)
        df_target = derive_sentiment_target(df)

        # Leakage-safe group-aware train/test split
        split_res = split_reviews_leakage_safe(
            df_target,
            group_col="review_clean_hash",
            test_size=0.20,
            random_state=125,
        )
        train_df = split_res["train"]
        test_df = split_res["test"]

        X_tr = train_df["review_clean"]
        y_tr = train_df["sentiment_class"]
        groups_tr = train_df["review_clean_hash"]

        X_te = test_df["review_clean"]
        y_te = test_df["sentiment_class"]

        # MLflow setup
        setup_nlp_mlflow(self.config)

        # Train candidates (Logistic Regression, Linear SVM, Multinomial Naive Bayes)
        candidates = ["logistic_regression", "linear_svm", "naive_bayes"]
        candidate_results = {}

        for cand_name in candidates:
            pipe = build_nlp_pipeline(cand_name, cfg=self.config)
            cv_res = evaluate_cv_leakage_safe(
                pipe, X_tr, y_tr, groups=groups_tr, n_splits=5, random_state=125
            )

            # Fit on full training set and evaluate on test set
            pipe.fit(X_tr, y_tr)
            test_res = evaluate_test_set(pipe, X_te, y_te)

            candidate_results[cand_name] = {
                "pipeline": pipe,
                "cv_metrics": cv_res,
                "test_metrics": test_res,
            }

            # Log candidate run to MLflow
            try:
                log_nlp_candidate_run(
                    name=cand_name,
                    params={"retraining_job_id": job_id},
                    metrics={
                        "cv_macro_f1": cv_res["cv_macro_f1_mean"],
                        "test_macro_f1": test_res["test_macro_f1"],
                        "test_accuracy": test_res["test_accuracy"],
                    },
                    tags={
                        "retraining_job_id": job_id,
                        "dataset_sha256": dataset_sha256,
                    },
                )
            except Exception as mlf_err:
                logger.warning(
                    f"MLflow logging for candidate {cand_name} failed: {mlf_err}"
                )

        # Select best candidate by CV Macro F1
        best_cand_name = max(
            candidate_results.keys(),
            key=lambda k: candidate_results[k]["cv_metrics"]["cv_macro_f1_mean"],
        )
        best_cand = candidate_results[best_cand_name]

        # Save candidate artifact into isolated challenger directory
        chal_dir = self.registry.challengers_dir / job_id
        chal_dir.mkdir(parents=True, exist_ok=True)
        art_path = chal_dir / "nlp_challenger_pipeline.skops"
        meta_path = chal_dir / "training_manifest.json"

        sio.dump(best_cand["pipeline"], art_path)

        metadata = {
            "retraining_id": job_id,
            "model_type": best_cand_name,
            "model_name": best_cand_name,
            "dataset_path": str(data_path.relative_to(self.base_dir)).replace(
                "\\", "/"
            ),
            "dataset_sha256": dataset_sha256,
            "dvc_dataset_hash": dvc_hash,
            "git_commit": git_sha,
            "python_version": sys.version.split()[0],
            "os_platform": platform.platform(),
            "target_definition": "rating 1-4: low, 5-7: neutral, 8-10: high",
            "split_strategy": "leakage-safe StratifiedGroupKFold on review_clean_hash",
            "random_seed": 125,
            "cv_metrics": best_cand["cv_metrics"],
            "test_metrics": best_cand["test_metrics"],
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        return {
            "model_name": best_cand_name,
            "artifact_path": art_path,
            "metadata_path": meta_path,
            "metadata": metadata,
            "cv_metrics": best_cand["cv_metrics"],
            "test_metrics": best_cand["test_metrics"],
        }

    def _generate_candidate_version(self, current_version: str) -> str:
        """Bump patch version, or default to 1.0.0 if initial bootstrap."""
        if current_version == "NONE":
            return "1.0.0"
        try:
            parts = [int(p) for p in current_version.split(".")]
            parts[-1] += 1
            return ".".join(str(p) for p in parts)
        except Exception:
            return f"{current_version}.1"


_nlp_orchestrator_instance: NLPRetrainingOrchestrator | None = None


def get_nlp_orchestrator(
    config: dict[str, Any] | None = None,
    base_dir: Path | str | None = None,
) -> NLPRetrainingOrchestrator:
    """Retrieve or initialize the global NLPRetrainingOrchestrator singleton."""
    global _nlp_orchestrator_instance
    if _nlp_orchestrator_instance is None or config is not None or base_dir is not None:
        _nlp_orchestrator_instance = NLPRetrainingOrchestrator(
            config=config, base_dir=base_dir
        )
    return _nlp_orchestrator_instance


def run_nlp_retraining_cli() -> None:
    """CLI entrypoint for NLP retraining execution."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting MedMLOps-X NLP Automated Retraining CLI...")
    orchestrator = get_nlp_orchestrator()
    res = orchestrator.trigger_retraining()
    logger.info(f"Retraining Result: {json.dumps(res, indent=2, default=str)}")


if __name__ == "__main__":
    run_nlp_retraining_cli()
