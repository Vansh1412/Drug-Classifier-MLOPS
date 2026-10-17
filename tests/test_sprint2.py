"""
Sprint 2 Test Suite — MedMLOps-X
Covers:
  - DVC foundation, dataset versioning, and pipeline definition (dvc.yaml, dvc.lock)
  - MLflow experiment tracking, candidate parameters, and metrics logging
  - Lineage tracking (Git commit SHA, DVC dataset hash)
  - Champion tracking and quality gate tagging in MLflow
  - MLflow model logging & string prediction contract
  - Backward compatibility with existing pipeline and Skops artifacts
"""

import os
import sys
from pathlib import Path

import mlflow
import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_trusted_types, load_config
from src.tracking import get_dvc_dataset_hash, get_git_commit_sha


def _bootstrap_sprint2_mlflow_runs(cfg: dict, experiment_id: str) -> None:
    """Bootstrap deterministic candidate and champion runs into MLflow for testing.

    Ensures MLflow tracking and lineage tests have real runs, parameters,
    metrics, tags, and artifacts without depending on a developer's local state.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    from src.champion_selection import select_champion
    from src.data_validation import load_and_validate
    from src.evaluate import train_and_evaluate
    from src.model_registry import build_model_candidates
    from src.preprocessing import build_preprocessor
    from src.tracking import get_lineage_metadata, log_candidate_run, log_champion_run

    data_path = cfg["data"]["data_path"]
    drug_df = load_and_validate(data_path, cfg)
    shuffle_seed = cfg["data"]["shuffle_random_state"]
    split_seed = cfg["data"]["split_random_state"]
    test_size = cfg["data"]["test_size"]

    drug_df = drug_df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    X = drug_df.drop(cfg["data"]["target_column"], axis=1).values
    y = drug_df[cfg["data"]["target_column"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_seed, stratify=y
    )

    candidates = build_model_candidates(cfg)
    lineage = get_lineage_metadata(cfg)
    all_results = []

    mlflow.set_experiment(experiment_id=experiment_id)

    for name, estimator in candidates.items():
        preprocessor = build_preprocessor(cfg)
        pipeline = Pipeline([("preprocessing", preprocessor), ("model", estimator)])
        model_params = {
            k: v
            for k, v in cfg["models"][name]["params"].items()
            if not k.startswith("#")
        }
        res = train_and_evaluate(
            pipeline, X_train, y_train, X_test, y_test, cfg, name, model_params
        )
        all_results.append({**res, "_pipeline": pipeline})
        log_candidate_run(name, estimator, res, cfg, lineage)

    serializable_results = [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in all_results
    ]
    pipeline_map = {r["model_name"]: r["_pipeline"] for r in all_results}

    champion_result, challengers = select_champion(serializable_results, cfg)
    champion_pipeline = pipeline_map[champion_result["model_name"]]

    passing_names = {champion_result["model_name"]} | {
        c["model_name"] for c in challengers
    }
    failing_results = [
        r for r in serializable_results if r["model_name"] not in passing_names
    ]

    log_champion_run(
        champion_result,
        challengers,
        failing_results,
        champion_pipeline,
        cfg,
        lineage,
    )


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def mlflow_client(config):
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        config.get("mlflow", {}).get("tracking_uri", "sqlite:///mlruns.db"),
    )
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


@pytest.fixture
def mlflow_experiment(config, mlflow_client):
    exp_name = config.get("mlflow", {}).get(
        "experiment_name", "MedMLOps-X-Drug-Classifier"
    )
    exp = mlflow_client.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow_client.create_experiment(exp_name)
        exp = mlflow_client.get_experiment(exp_id)

    runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.model_type = 'candidate'",
    )
    if len(runs) < 4:
        _bootstrap_sprint2_mlflow_runs(config, exp.experiment_id)

    return exp


# =============================================================================
# 1. DVC Foundation & Dataset Versioning Tests (Tasks S2-T2, S2-T3, S2-T4)
# =============================================================================
class TestDVCVersioning:
    def test_dvc_directory_structure(self):
        """Assert that .dvc directory and core configuration files exist."""
        dvc_dir = PROJECT_ROOT / ".dvc"
        assert dvc_dir.is_dir(), ".dvc directory missing"
        assert (dvc_dir / "config").is_file(), ".dvc/config file missing"
        assert (PROJECT_ROOT / ".dvcignore").is_file(), ".dvcignore missing"

    def test_dataset_dvc_metadata_file_exists(self):
        """Assert that Data/drug.csv.dvc exists and contains valid md5 hash."""
        dvc_meta = PROJECT_ROOT / "Data" / "drug.csv.dvc"
        assert dvc_meta.is_file(), "Data/drug.csv.dvc metadata file missing"

        with open(dvc_meta, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        assert "outs" in data
        assert len(data["outs"]) == 1
        entry = data["outs"][0]
        assert entry["path"] == "drug.csv"
        assert entry["size"] == 5828
        assert "md5" in entry
        assert len(entry["md5"]) == 32

    def test_dvc_pipeline_definition(self):
        """Assert that dvc.yaml defines validate and train stages with correct dependencies."""
        dvc_yaml_path = PROJECT_ROOT / "dvc.yaml"
        assert dvc_yaml_path.is_file(), "dvc.yaml missing"

        with open(dvc_yaml_path, "r", encoding="utf-8") as fh:
            pipeline_def = yaml.safe_load(fh)

        assert "stages" in pipeline_def
        stages = pipeline_def["stages"]
        assert "validate" in stages
        assert "train" in stages

        # Check validate stage
        assert "Data/drug.csv" in stages["validate"]["deps"]
        assert "src/data_validation.py" in stages["validate"]["deps"]

        # Check train stage
        assert "Data/drug.csv" in stages["train"]["deps"]
        assert "train.py" in stages["train"]["deps"]
        assert "outs" in stages["train"]

    def test_dvc_lock_file_exists(self):
        """Assert that dvc.lock exists after pipeline execution."""
        dvc_lock = PROJECT_ROOT / "dvc.lock"
        assert dvc_lock.is_file(), "dvc.lock missing. Run 'dvc repro' first."

        with open(dvc_lock, "r", encoding="utf-8") as fh:
            lock_data = yaml.safe_load(fh)

        assert "stages" in lock_data
        assert "validate" in lock_data["stages"]
        assert "train" in lock_data["stages"]


# =============================================================================
# 2. MLflow Experiment Tracking Tests (Tasks S2-T5, S2-T6, S2-T7, S2-T8)
# =============================================================================
class TestMLflowTracking:
    def test_mlflow_experiment_exists(self, mlflow_experiment):
        """Assert that the MedMLOps-X MLflow experiment is initialized and active."""
        assert mlflow_experiment.name == "MedMLOps-X-Drug-Classifier"
        assert mlflow_experiment.lifecycle_stage == "active"

    def test_candidate_runs_logged_with_parameters(
        self, mlflow_client, mlflow_experiment
    ):
        """Assert that candidate model runs log configuration and hyperparameters."""
        runs = mlflow_client.search_runs(
            experiment_ids=[mlflow_experiment.experiment_id],
            filter_string="tags.model_type = 'candidate'",
        )
        assert len(runs) >= 4, f"Expected at least 4 candidate runs, found {len(runs)}"

        logged_candidates = {r.data.params.get("candidate_name") for r in runs}
        expected_candidates = {
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "xgboost",
        }
        assert expected_candidates.issubset(logged_candidates)

        for run in runs:
            params = run.data.params
            assert "test_size" in params
            assert "cv_folds" in params
            assert "split_random_state" in params

    def test_metrics_logged_for_runs(self, mlflow_client, mlflow_experiment):
        """Assert that accuracy, macro F1, and cross-validation metrics are recorded."""
        runs = mlflow_client.search_runs(
            experiment_ids=[mlflow_experiment.experiment_id]
        )
        assert len(runs) > 0

        for run in runs:
            metrics = run.data.metrics
            assert "accuracy" in metrics or "champion_accuracy" in metrics
            assert "macro_f1" in metrics or "champion_macro_f1" in metrics
            assert (
                "cv_macro_f1_mean" in metrics or "champion_cv_macro_f1_mean" in metrics
            )


# =============================================================================
# 3. Experiment Lineage Tests (Task S2-T10)
# =============================================================================
class TestExperimentLineage:
    def test_git_commit_hash_retrieval(self):
        """Assert that Git commit SHA is resolved to a valid commit hash."""
        sha = get_git_commit_sha()
        assert sha != "unknown"
        assert len(sha) == 40

    def test_dvc_dataset_hash_retrieval(self):
        """Assert that DVC dataset hash is extracted from Data/drug.csv.dvc."""
        dvc_hash = get_dvc_dataset_hash()
        assert dvc_hash != "unknown"
        assert len(dvc_hash) == 32

    def test_lineage_metadata_logged_in_runs(self, mlflow_client, mlflow_experiment):
        """Assert that every MLflow run contains git_commit and dvc_dataset_hash tags."""
        runs = mlflow_client.search_runs(
            experiment_ids=[mlflow_experiment.experiment_id]
        )
        for run in runs:
            tags = run.data.tags
            assert (
                "git_commit" in tags
            ), f"Run {run.info.run_name} missing git_commit tag"
            assert (
                "dvc_dataset_hash" in tags
            ), f"Run {run.info.run_name} missing dvc_dataset_hash tag"
            assert len(tags["git_commit"]) > 0
            assert len(tags["dvc_dataset_hash"]) > 0


# =============================================================================
# 4. Champion Tracking & Quality Gate Tests (Task S2-T11)
# =============================================================================
class TestChampionTracking:
    def test_champion_run_identified_and_tagged(self, mlflow_client, mlflow_experiment):
        """Assert that champion run has is_champion=true and quality_gate=PASSED."""
        champion_runs = mlflow_client.search_runs(
            experiment_ids=[mlflow_experiment.experiment_id],
            filter_string="tags.is_champion = 'true'",
        )
        assert len(champion_runs) >= 1, "Champion run not found in MLflow"
        champ = champion_runs[0]
        assert champ.data.tags.get("is_champion") == "true"
        assert champ.data.tags.get("quality_gate") == "PASSED"
        assert champ.data.params.get("champion_name") == "xgboost"

    def test_champion_artifacts_logged_in_mlflow(
        self, mlflow_client, mlflow_experiment
    ):
        """Assert that metrics and reports are saved as artifacts in the champion run."""
        champion_runs = mlflow_client.search_runs(
            experiment_ids=[mlflow_experiment.experiment_id],
            filter_string="tags.is_champion = 'true'",
        )
        champ = champion_runs[0]
        artifacts = mlflow_client.list_artifacts(champ.info.run_id, path="results")
        artifact_names = {a.path.split("/")[-1] for a in artifacts}

        assert "metrics.json" in artifact_names
        assert "competition_report.json" in artifact_names
        assert "model_metadata.json" in artifact_names


# =============================================================================
# 5. Model Logging & String Label Contract Tests (Task S2-T9)
# =============================================================================
class TestModelLogging:
    def test_mlflow_logged_model_loads_and_predicts(
        self, config, mlflow_client, mlflow_experiment
    ):
        """Assert that MLflow logged model loads with skops trusted types and emits string labels."""
        champion_runs = mlflow_client.search_runs(
            experiment_ids=[mlflow_experiment.experiment_id],
            filter_string="tags.is_champion = 'true'",
        )
        champ = champion_runs[0]
        model_uri = f"runs:/{champ.info.run_id}/model"

        # Load model from MLflow run (flavor configuration automatically resolves trusted types)
        loaded_model = mlflow.sklearn.load_model(model_uri)

        sample = [[30, "M", "HIGH", "NORMAL", 15.4]]
        pred = loaded_model.predict(sample)
        assert len(pred) == 1
        assert isinstance(pred[0], str)
        assert pred[0] in {"DrugY", "drugA", "drugB", "drugC", "drugX"}

    def test_skops_disk_artifact_remains_functional(self):
        """Assert that Model/drug_pipeline.skops remains directly usable by FastAPI."""
        import skops.io as sio

        cfg = load_config()
        trusted = get_trusted_types(cfg)
        model = sio.load("Model/drug_pipeline.skops", trusted=trusted)
        sample = [[30, "M", "HIGH", "NORMAL", 15.4]]
        pred = model.predict(sample)
        assert pred[0] in {"DrugY", "drugA", "drugB", "drugC", "drugX"}
