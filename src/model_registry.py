"""
src/model_registry.py — Model candidate factory for MedMLOps-X competition pipeline.

Instantiates all configured model candidates from configs/model_config.yaml.
Each candidate is returned as an unfitted estimator ready for pipeline integration.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.config import get_trusted_types, load_config

logger = logging.getLogger("src.model_registry")


class XGBStringClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost classifier wrapper that handles string target labels internally.

    XGBClassifier requires integer class labels [0, 1, ...]. This wrapper transparently
    encodes string labels during fit() and decodes them back to original strings
    during predict(), conforming to the scikit-learn string-label contract.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.estimator = XGBClassifier(**kwargs)
        self.classes_ = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self.kwargs.copy()

    def set_params(self, **params: Any) -> XGBStringClassifier:
        self.kwargs.update(params)
        self.estimator = XGBClassifier(**self.kwargs)
        return self

    def fit(self, X: Any, y: Any) -> XGBStringClassifier:
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.estimator.fit(X, y_enc)
        return self

    def predict(self, X: Any) -> Any:
        y_pred_enc = self.estimator.predict(X)
        return self.label_encoder_.inverse_transform(y_pred_enc)

    def predict_proba(self, X: Any) -> Any:
        return self.estimator.predict_proba(X)


_MODEL_CLASS_MAP: dict[str, type[BaseEstimator]] = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "XGBClassifier": XGBStringClassifier,
    "XGBStringClassifier": XGBStringClassifier,
}


def build_model_candidates(
    cfg: dict[str, Any],
) -> dict[str, BaseEstimator]:
    """Instantiate all model candidates defined in configuration.

    Args:
        cfg: Full configuration dictionary (from load_config()).

    Returns:
        Ordered dict mapping candidate name -> unfitted sklearn estimator.

    Raises:
        KeyError: If a model class in config is not registered in _MODEL_CLASS_MAP.
    """
    candidates: dict[str, BaseEstimator] = {}
    models_cfg = cfg.get("models", {})

    for name, model_cfg in models_cfg.items():
        class_name = model_cfg["class"]
        if class_name not in _MODEL_CLASS_MAP:
            raise KeyError(
                f"Unknown model class '{class_name}' for candidate '{name}'. "
                f"Registered classes: {list(_MODEL_CLASS_MAP.keys())}"
            )
        params = model_cfg.get("params", {})
        estimator = _MODEL_CLASS_MAP[class_name](**params)
        candidates[name] = estimator

    return candidates


class ModelRegistry:
    """Manages model versioning, manifest tracking, atomic promotion, and rollback."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        base_dir: Path | str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent.parent)
        retrain_cfg = self.config.get("retraining", {})

        manifest_rel = retrain_cfg.get("manifest_path", "Model/registry_manifest.json")
        self.manifest_path = (self.base_dir / manifest_rel).resolve()

        archive_rel = retrain_cfg.get("archive_dir", "Model/archive")
        self.archive_dir = (self.base_dir / archive_rel).resolve()
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        challengers_rel = retrain_cfg.get("challengers_dir", "Model/challengers")
        self.challengers_dir = (self.base_dir / challengers_rel).resolve()
        self.challengers_dir.mkdir(parents=True, exist_ok=True)

        champ_cfg = self.config.get("champion", {})
        self.active_artifact_path = (
            self.base_dir / champ_cfg.get("artifact_path", "Model/drug_pipeline.skops")
        ).resolve()
        self.active_metadata_path = (
            self.base_dir / champ_cfg.get("metadata_path", "Model/model_metadata.json")
        ).resolve()

        self._ensure_manifest()

    def _ensure_manifest(self) -> dict[str, Any]:
        """Ensure registry manifest exists and is initialized."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.warning(f"Error reading manifest, re-initializing: {exc}")

        # Initialize default manifest from active model metadata
        active_meta = {}
        if self.active_metadata_path.exists():
            try:
                with open(self.active_metadata_path, "r", encoding="utf-8") as fh:
                    active_meta = json.load(fh)
            except Exception:
                pass

        default_version = self.config.get("project", {}).get("version", "0.2.0")
        manifest = {
            "active_champion": {
                "version": default_version,
                "model_name": active_meta.get("model_name", "xgboost"),
                "artifact_path": str(
                    self.active_artifact_path.relative_to(self.base_dir)
                ).replace("\\", "/"),
                "metadata_path": str(
                    self.active_metadata_path.relative_to(self.base_dir)
                ).replace("\\", "/"),
                "promoted_at": active_meta.get(
                    "training_timestamp", datetime.now(timezone.utc).isoformat()
                ),
                "metrics": {
                    "accuracy": active_meta.get("accuracy", 1.0),
                    "macro_f1": active_meta.get("macro_f1", 1.0),
                    "cv_macro_f1_mean": active_meta.get("cv_macro_f1_mean", 0.9633),
                },
            },
            "previous_champion": None,
            "history": [
                {
                    "version": default_version,
                    "model_name": active_meta.get("model_name", "xgboost"),
                    "action": "initial_champion",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }
        self._save_manifest(manifest)
        return manifest

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        """Atomically persist manifest to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.manifest_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        tmp_path.replace(self.manifest_path)

    def get_manifest(self) -> dict[str, Any]:
        """Return the current registry manifest."""
        return self._ensure_manifest()

    def get_active_champion(self) -> dict[str, Any]:
        """Return metadata for the currently active champion."""
        return self.get_manifest().get("active_champion", {})

    def get_previous_champion(self) -> dict[str, Any] | None:
        """Return metadata for the previous champion, if available."""
        return self.get_manifest().get("previous_champion")

    def archive_champion(self, version: str) -> dict[str, str]:
        """Snapshot current active champion to archive directory before promotion."""
        if not self.active_artifact_path.exists():
            raise FileNotFoundError(
                f"Active champion artifact not found at {self.active_artifact_path}"
            )

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_ver = version.replace(".", "_")

        archived_art_name = f"champion_v{safe_ver}_{timestamp_str}.skops"
        archived_meta_name = f"model_metadata_v{safe_ver}_{timestamp_str}.json"

        dest_art = (self.archive_dir / archived_art_name).resolve()
        dest_meta = (self.archive_dir / archived_meta_name).resolve()

        if not dest_art.is_relative_to(self.base_dir) or not dest_meta.is_relative_to(
            self.base_dir
        ):
            raise ValueError("Path traversal attempt in archive_champion")

        shutil.copy2(self.active_artifact_path, dest_art)
        if self.active_metadata_path.exists():
            shutil.copy2(self.active_metadata_path, dest_meta)

        logger.info(f"Archived champion v{version} to {dest_art.name}")
        return {
            "artifact_path": str(dest_art.relative_to(self.base_dir)).replace(
                "\\", "/"
            ),
            "metadata_path": str(dest_meta.relative_to(self.base_dir)).replace(
                "\\", "/"
            ),
        }

    def promote_challenger(
        self,
        challenger_artifact_path: Path | str,
        challenger_metadata: dict[str, Any],
        new_version: str,
        auto_smoke_test: bool = True,
    ) -> bool:
        """Atomically promote validated challenger to active champion."""
        chal_path = Path(challenger_artifact_path)
        if not chal_path.is_absolute():
            chal_path = (self.base_dir / chal_path).resolve()
        if not chal_path.exists():
            raise FileNotFoundError(f"Challenger artifact does not exist: {chal_path}")

        manifest = self.get_manifest()
        current_champ = manifest.get("active_champion", {})
        current_ver = current_champ.get("version", "0.2.0")

        # 1. Archive current champion
        archived_paths = self.archive_champion(current_ver)

        # 2. Stage new files to temporary destinations then replace atomically
        tmp_art = self.active_artifact_path.with_suffix(".promotetmp")
        tmp_meta = self.active_metadata_path.with_suffix(".promotetmp")

        shutil.copy2(chal_path, tmp_art)

        updated_meta = challenger_metadata.copy()
        updated_meta["model_version"] = new_version
        updated_meta["promoted_at"] = datetime.now(timezone.utc).isoformat()
        updated_meta["parent_champion_version"] = current_ver

        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(updated_meta, fh, indent=2)

        tmp_art.replace(self.active_artifact_path)
        tmp_meta.replace(self.active_metadata_path)

        # 3. Update manifest
        manifest["previous_champion"] = {
            **current_champ,
            "archived_artifact_path": archived_paths["artifact_path"],
            "archived_metadata_path": archived_paths["metadata_path"],
            "archived_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest["active_champion"] = {
            "version": new_version,
            "model_name": updated_meta.get("model_name", "xgboost"),
            "artifact_path": str(
                self.active_artifact_path.relative_to(self.base_dir)
            ).replace("\\", "/"),
            "metadata_path": str(
                self.active_metadata_path.relative_to(self.base_dir)
            ).replace("\\", "/"),
            "promoted_at": updated_meta["promoted_at"],
            "metrics": {
                "accuracy": updated_meta.get("accuracy"),
                "macro_f1": updated_meta.get("macro_f1"),
                "cv_macro_f1_mean": updated_meta.get("cv_macro_f1_mean"),
            },
        }
        manifest["history"].append(
            {
                "version": new_version,
                "model_name": updated_meta.get("model_name", "xgboost"),
                "action": "promoted",
                "timestamp": updated_meta["promoted_at"],
            }
        )
        self._save_manifest(manifest)

        # 4. Post-promotion validation
        if auto_smoke_test:
            if not self.post_promotion_smoke_test():
                logger.error(
                    "Post-promotion smoke test failed! Triggering automatic rollback."
                )
                self.rollback(reason="Post-promotion smoke test failed")
                return False

        logger.info(
            f"Successfully promoted challenger {updated_meta.get('model_name')} to v{new_version}"
        )
        return True

    def post_promotion_smoke_test(self, test_features: list[Any] | None = None) -> bool:
        """Validate that newly promoted champion loads and produces valid predictions."""
        try:
            import skops.io as sio

            trusted = get_trusted_types(self.config)
            loaded_model = sio.load(self.active_artifact_path, trusted=trusted)
            sample = test_features or [[30, "M", "HIGH", "NORMAL", 15.4]]
            preds = loaded_model.predict(sample)
            if len(preds) > 0 and str(preds[0]) != "":
                return True
            return False
        except Exception as exc:
            logger.error(f"Post-promotion smoke test encountered exception: {exc}")
            return False

    def rollback(self, reason: str = "Rollback requested") -> bool:
        """Restore the previously archived champion model."""
        manifest = self.get_manifest()
        prev = manifest.get("previous_champion")
        if not prev:
            raise RuntimeError(
                "Cannot rollback: No previous champion found in registry manifest."
            )

        archived_art_rel = prev.get("archived_artifact_path")
        archived_meta_rel = prev.get("archived_metadata_path")

        if not archived_art_rel:
            raise FileNotFoundError(
                "Previous champion archive artifact path not recorded in manifest."
            )

        archived_art = (self.base_dir / archived_art_rel).resolve()
        if not archived_art.exists():
            raise FileNotFoundError(
                f"Archived artifact missing from disk: {archived_art}"
            )

        # Restore artifact and metadata
        shutil.copy2(archived_art, self.active_artifact_path)
        if archived_meta_rel:
            archived_meta = (self.base_dir / archived_meta_rel).resolve()
            if archived_meta.exists():
                shutil.copy2(archived_meta, self.active_metadata_path)

        # Swap active and previous in manifest
        rolled_back_version = manifest["active_champion"]["version"]
        restored_version = prev["version"]

        manifest["active_champion"] = {
            "version": restored_version,
            "model_name": prev["model_name"],
            "artifact_path": str(
                self.active_artifact_path.relative_to(self.base_dir)
            ).replace("\\", "/"),
            "metadata_path": str(
                self.active_metadata_path.relative_to(self.base_dir)
            ).replace("\\", "/"),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "metrics": prev.get("metrics", {}),
        }
        manifest["history"].append(
            {
                "version": restored_version,
                "action": f"rolled_back_from_v{rolled_back_version}",
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_manifest(manifest)
        logger.info(
            f"Successfully rolled back champion to v{restored_version} ({reason})"
        )
        return True


_registry_instance: ModelRegistry | None = None


def get_registry(
    config: dict[str, Any] | None = None,
    base_dir: Path | str | None = None,
) -> ModelRegistry:
    """Retrieve or initialize the global ModelRegistry singleton."""
    global _registry_instance
    if _registry_instance is None or config is not None or base_dir is not None:
        _registry_instance = ModelRegistry(config=config, base_dir=base_dir)
    return _registry_instance
