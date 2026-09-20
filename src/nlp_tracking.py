"""
src/nlp_tracking.py — MLflow experiment tracking & lineage for NLP Review Intelligence.

Provides structured logging for Sprint 8 NLP candidate models and champion pipeline,
with isolation from the structured drug classifier MLflow experiment.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
import yaml

logger = logging.getLogger(__name__)


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


def get_dvc_dataset_hash(
    dvc_path: str | Path = "Data/reviews/drugsComTest_raw.csv.dvc",
) -> str:
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


def setup_nlp_mlflow(cfg: dict[str, Any]) -> str:
    """Initialize MLflow tracking URI and NLP experiment.

    Args:
        cfg: Configuration dictionary containing `reviews.nlp_competition.mlflow` section.

    Returns:
        The active experiment ID string.
    """
    nlp_cfg = cfg.get("reviews", {}).get("nlp_competition", {})
    mlflow_cfg = nlp_cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlruns.db")
    experiment_name = mlflow_cfg.get("experiment_name", "MedMLOps-X-NLP-Drug-Review")

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    logger.info(
        f"NLP MLflow initialized: experiment='{experiment_name}' (id={experiment.experiment_id}) "
        f"uri='{tracking_uri}'"
    )
    return str(experiment.experiment_id)


def log_nlp_candidate_run(
    name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> str:
    """Log an individual NLP candidate model run in MLflow.

    Args:
        name: Name of the model candidate (e.g. 'logistic_regression').
        params: Dictionary of parameters to log.
        metrics: Dictionary of numerical metrics to log.
        tags: Optional metadata tags.
        nested: Whether to start as a nested MLflow run.

    Returns:
        The MLflow run ID string.
    """
    with mlflow.start_run(run_name=f"nlp-candidate-{name}", nested=nested) as run:
        # Default tags
        run_tags = {
            "workload": "nlp",
            "sprint": "8",
            "dataset": "drugsComTest_raw",
            "model_type": name,
            "git_commit": get_git_commit_sha(),
            "dvc_dataset_hash": get_dvc_dataset_hash(),
            "python_version": sys.version.split()[0],
            "os_platform": platform.platform(),
        }
        if tags:
            run_tags.update(tags)
        mlflow.set_tags(run_tags)

        # Log parameters
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                clean_params[k] = str(v)
            elif v is None:
                clean_params[k] = "None"
            else:
                clean_params[k] = v
        mlflow.log_params(clean_params)

        # Log metrics
        for mk, mv in metrics.items():
            if isinstance(mv, (int, float)):
                mlflow.log_metric(mk, float(mv))

        return str(run.info.run_id)


def log_nlp_champion_run(
    champion_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[str | Path] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log the NLP champion model selection and associated artifacts in MLflow.

    Args:
        champion_name: Name of the winning model.
        params: Key parameters of the champion pipeline.
        metrics: Validation and test metrics for the champion.
        artifacts: File paths of artifacts to attach to the run.
        tags: Optional metadata tags.

    Returns:
        The MLflow champion run ID string.
    """
    with mlflow.start_run(
        run_name=f"nlp-champion-{champion_name}", nested=False
    ) as run:
        champ_tags = {
            "workload": "nlp",
            "sprint": "8",
            "is_champion": "true",
            "champion_name": champion_name,
            "git_commit": get_git_commit_sha(),
            "dvc_dataset_hash": get_dvc_dataset_hash(),
            "python_version": sys.version.split()[0],
            "os_platform": platform.platform(),
        }
        if tags:
            champ_tags.update(tags)
        mlflow.set_tags(champ_tags)

        clean_params = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                clean_params[k] = str(v)
            elif v is None:
                clean_params[k] = "None"
            else:
                clean_params[k] = v
        mlflow.log_params(clean_params)

        for mk, mv in metrics.items():
            if isinstance(mv, (int, float)):
                mlflow.log_metric(f"champion_{mk}", float(mv))

        if artifacts:
            for art in artifacts:
                art_path = Path(art)
                if art_path.exists():
                    mlflow.log_artifact(str(art_path), artifact_path="artifacts")
                else:
                    logger.warning(f"Artifact not found for MLflow logging: {art_path}")

        return str(run.info.run_id)
