"""
src/tracking.py — MLflow experiment tracking & lineage utilities for MedMLOps-X.

Provides structured logging for candidate models, champion promotion,
lineage tracking (Git commit SHA, DVC dataset hash), and artifact persistence.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None  # type: ignore

import yaml

from src.config import get_trusted_types


def get_git_commit_sha() -> str:
    """Retrieve the current Git commit SHA, or 'unknown' if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_dvc_dataset_hash(dvc_path: str | Path = "Data/drug.csv.dvc") -> str:
    """Extract the dataset md5 hash from the DVC metadata file."""
    path = Path(dvc_path)
    if not path.exists():
        return "unknown"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        outs = data.get("outs", [])
        if outs and isinstance(outs[0], dict):
            return str(outs[0].get("md5", "unknown"))
    except Exception:
        pass
    return "unknown"


def get_lineage_metadata(cfg: dict[str, Any]) -> dict[str, str]:
    """Assemble reproducible lineage metadata tags."""
    return {
        "git_commit": get_git_commit_sha(),
        "dvc_dataset_hash": get_dvc_dataset_hash(),
        "python_version": sys.version.split()[0],
        "os_platform": platform.platform(),
        "shuffle_random_state": str(cfg["data"]["shuffle_random_state"]),
        "split_random_state": str(cfg["data"]["split_random_state"]),
        "cv_random_state": str(cfg["data"]["cv_random_state"]),
    }


def setup_mlflow(cfg: dict[str, Any]) -> str:
    """Initialize MLflow tracking URI and experiment.

    Args:
        cfg: Configuration dictionary containing `mlflow` section.

    Returns:
        The active experiment ID.
    """
    if mlflow is None:
        raise ImportError(
            "mlflow is required for MLflow experiment tracking but is not installed."
        )

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlruns.db")
    experiment_name = mlflow_cfg.get("experiment_name", "MedMLOps-X-Drug-Classifier")

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id


def log_candidate_run(
    name: str,
    estimator: Any,
    result: dict[str, Any],
    cfg: dict[str, Any],
    lineage: dict[str, str],
) -> str:
    """Log an individual candidate model run in MLflow.

    Args:
        name: Candidate model identifier (e.g. 'xgboost').
        estimator: The candidate estimator object.
        result: Evaluated metrics dictionary from `train_and_evaluate`.
        cfg: Full configuration dictionary.
        lineage: Lineage metadata tags dict.

    Returns:
        The MLflow run_id.
    """
    if mlflow is None:
        raise ImportError(
            "mlflow is required for MLflow candidate logging but is not installed."
        )

    with mlflow.start_run(run_name=f"candidate-{name}", nested=True) as run:
        # 1. Tags
        tags = {
            **lineage,
            "model_name": name,
            "model_type": "candidate",
            "model_class": cfg["models"][name]["class"],
        }
        mlflow.set_tags(tags)

        # 2. Parameters
        params = {
            "candidate_name": name,
            "model_class": cfg["models"][name]["class"],
            "test_size": cfg["data"]["test_size"],
            "cv_folds": cfg["data"]["cv_folds"],
            "split_random_state": cfg["data"]["split_random_state"],
            "shuffle_random_state": cfg["data"]["shuffle_random_state"],
            "cv_random_state": cfg["data"]["cv_random_state"],
        }
        for p_key, p_val in result.get("model_params", {}).items():
            if not p_key.startswith("#"):
                params[f"param_{p_key}"] = str(p_val)
        mlflow.log_params(params)

        # 3. Metrics
        metric_keys = [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_precision",
            "macro_recall",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "cv_macro_f1_mean",
            "cv_macro_f1_std",
            "cv_macro_precision_mean",
            "cv_macro_recall_mean",
            "training_time_seconds",
        ]
        for mk in metric_keys:
            if mk in result and result[mk] is not None:
                mlflow.log_metric(mk, float(result[mk]))

        return run.info.run_id


def log_champion_run(
    champion_result: dict[str, Any],
    challengers: list[dict[str, Any]],
    failing: list[dict[str, Any]],
    champion_pipeline: Any,
    cfg: dict[str, Any],
    lineage: dict[str, str],
) -> str:
    """Log the champion model selection, artifacts, and MLflow model.

    Args:
        champion_result: Metrics dict of the selected champion.
        challengers: Ranked list of passing challenger metrics.
        failing: List of candidates that failed the quality gate.
        champion_pipeline: Fitted champion scikit-learn Pipeline.
        cfg: Full configuration dictionary.
        lineage: Lineage metadata tags dict.

    Returns:
        The champion MLflow run_id.
    """
    if mlflow is None:
        raise ImportError(
            "mlflow is required for MLflow champion logging but is not installed."
        )

    champ_name = champion_result["model_name"]

    with mlflow.start_run(run_name=f"champion-{champ_name}", nested=False) as run:
        # 1. Tags
        tags = {
            **lineage,
            "is_champion": "true",
            "champion_model": champ_name,
            "quality_gate": "PASSED",
            "total_candidates": str(1 + len(challengers) + len(failing)),
            "challengers_count": str(len(challengers)),
            "failing_count": str(len(failing)),
        }
        mlflow.set_tags(tags)

        # 2. Parameters
        params = {
            "champion_name": champ_name,
            "champion_class": cfg["models"][champ_name]["class"],
            "primary_metric": cfg["champion"]["primary_metric"],
            "min_accuracy": cfg["champion"]["min_accuracy"],
            "min_macro_f1": cfg["champion"]["min_macro_f1"],
            "min_recall": cfg["champion"]["min_recall"],
        }
        for p_key, p_val in champion_result.get("model_params", {}).items():
            if not p_key.startswith("#"):
                params[f"champion_param_{p_key}"] = str(p_val)
        mlflow.log_params(params)

        # 3. Metrics
        metric_keys = [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_precision",
            "macro_recall",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "cv_macro_f1_mean",
            "cv_macro_f1_std",
            "training_time_seconds",
        ]
        for mk in metric_keys:
            if mk in champion_result and champion_result[mk] is not None:
                mlflow.log_metric(f"champion_{mk}", float(champion_result[mk]))

        # 4. Artifacts
        artifact_files = [
            cfg["results"]["metrics_json"],
            cfg["results"]["competition_report_json"],
            cfg["results"]["metrics_txt"],
            cfg["results"]["confusion_matrix_png"],
            cfg["champion"]["metadata_path"],
            "configs/model_config.yaml",
        ]
        for art_path in artifact_files:
            if os.path.exists(art_path):
                mlflow.log_artifact(art_path, artifact_path="results")

        # 5. Model logging via MLflow with skops trusted types
        if cfg.get("mlflow", {}).get("log_models", True):
            trusted = get_trusted_types(cfg)
            try:
                mlflow.sklearn.log_model(
                    champion_pipeline,
                    name="model",
                    skops_trusted_types=trusted,
                )
            except Exception as e:
                # Log warning if MLflow model serialization encounters environment constraint
                mlflow.set_tag("mlflow_model_logging_warning", str(e))

        return run.info.run_id
