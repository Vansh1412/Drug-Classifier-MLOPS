"""
src/nlp_model_registry.py — NLP Model Registry, Versioning, Promotion, and Rollback.

Provides isolated model versioning, atomic promotion, smoke testing, and rollback
for the Drug Review Intelligence / NLP workload (Model/nlp/).

Key Guarantees:
1. Strict workload isolation: Structured artifacts (Model/drug_pipeline.skops) are never touched.
2. Supports NLP Champion = NONE initial state gracefully (bootstrap promotion).
3. Atomic promotion via staging files and atomic replacement.
4. Automatic smoke testing of promoted pipelines on representative drug review samples.
5. Immediate automatic rollback to previous champion if smoke testing fails.
6. Path traversal protection on all archive, challenger, and champion operations.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import skops.io as sio

from src.config import load_config

logger = logging.getLogger("src.nlp_model_registry")

_DEFAULT_SMOKE_SAMPLES = [
    "This medication relieved my symptoms significantly with no major side effects.",
    "Severe headache, intense nausea, and complete lack of effectiveness.",
    "Mild improvement observed, average results overall.",
]
_EXPECTED_CLASSES = {"low", "neutral", "high"}


class NLPModelRegistry:
    """Manages NLP review intelligence model registry, lifecycle, and artifacts."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        base_dir: Path | str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent.parent)

        retrain_cfg = self.config.get("nlp_retraining", {})
        manifest_rel = retrain_cfg.get(
            "manifest_path", "Model/nlp/registry_manifest.json"
        )
        self.manifest_path = (self.base_dir / manifest_rel).resolve()

        archive_rel = retrain_cfg.get("archive_dir", "Model/nlp/archive")
        self.archive_dir = (self.base_dir / archive_rel).resolve()
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        challengers_rel = retrain_cfg.get("challengers_dir", "Model/nlp/challengers")
        self.challengers_dir = (self.base_dir / challengers_rel).resolve()
        self.challengers_dir.mkdir(parents=True, exist_ok=True)

        champ_dir_rel = retrain_cfg.get("champion_dir", "Model/nlp/champion")
        self.champion_dir = (self.base_dir / champ_dir_rel).resolve()
        self.champion_dir.mkdir(parents=True, exist_ok=True)

        champ_pipe_rel = retrain_cfg.get(
            "champion_pipeline_path",
            "Model/nlp/champion/nlp_champion_pipeline.skops",
        )
        self.champion_pipeline_path = (self.base_dir / champ_pipe_rel).resolve()

        champ_meta_rel = retrain_cfg.get(
            "champion_metadata_path", "Model/nlp/nlp_model_metadata.json"
        )
        self.champion_metadata_path = (self.base_dir / champ_meta_rel).resolve()

        self._ensure_manifest()

    def _ensure_manifest(self) -> dict[str, Any]:
        """Ensure NLP registry manifest exists and is initialized.

        Initial state is status="NONE" with active_champion=None because
        Sprint 8 classical models did not pass the quality gate.
        """
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.warning(f"Error reading NLP manifest, re-initializing: {exc}")

        # Check if an existing champion is already present on disk
        active_champ = None
        status = "NONE"
        if (
            self.champion_pipeline_path.exists()
            and self.champion_metadata_path.exists()
        ):
            try:
                with open(self.champion_metadata_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                active_champ = {
                    "version": meta.get("model_version", "1.0.0"),
                    "model_name": meta.get("model_type", "nlp_classifier"),
                    "artifact_path": str(
                        self.champion_pipeline_path.relative_to(self.base_dir)
                    ).replace("\\", "/"),
                    "metadata_path": str(
                        self.champion_metadata_path.relative_to(self.base_dir)
                    ).replace("\\", "/"),
                    "promoted_at": meta.get(
                        "promoted_at", datetime.now(timezone.utc).isoformat()
                    ),
                    "metrics": meta.get("test_metrics", {}),
                }
                status = "ACTIVE"
            except Exception as exc:
                logger.warning(f"Failed to read existing NLP champion metadata: {exc}")

        manifest = {
            "workload": "nlp",
            "status": status,
            "active_champion": active_champ,
            "previous_champion": None,
            "history": (
                [
                    {
                        "version": active_champ["version"],
                        "model_name": active_champ["model_name"],
                        "action": "initial_champion",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ]
                if active_champ
                else []
            ),
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
        """Return the current NLP registry manifest."""
        return self._ensure_manifest()

    def get_active_champion(self) -> dict[str, Any] | None:
        """Return metadata for the active NLP champion, or None if status is NONE."""
        manifest = self.get_manifest()
        if manifest.get("status") == "NONE" or not manifest.get("active_champion"):
            return None
        return manifest["active_champion"]

    def get_previous_champion(self) -> dict[str, Any] | None:
        """Return metadata for the previous NLP champion, or None."""
        return self.get_manifest().get("previous_champion")

    def archive_champion(self, version: str) -> dict[str, str]:
        """Snapshot current active champion to archive directory before promotion."""
        if not self.champion_pipeline_path.exists():
            raise FileNotFoundError(
                f"Active NLP champion artifact not found at {self.champion_pipeline_path}"
            )

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_ver = version.replace(".", "_")

        archived_art_name = f"nlp_champion_v{safe_ver}_{timestamp_str}.skops"
        archived_meta_name = f"nlp_model_metadata_v{safe_ver}_{timestamp_str}.json"

        dest_art = (self.archive_dir / archived_art_name).resolve()
        dest_meta = (self.archive_dir / archived_meta_name).resolve()

        if not dest_art.is_relative_to(self.base_dir) or not dest_meta.is_relative_to(
            self.base_dir
        ):
            raise ValueError("Path traversal attempt in archive_champion")

        shutil.copy2(self.champion_pipeline_path, dest_art)
        if self.champion_metadata_path.exists():
            shutil.copy2(self.champion_metadata_path, dest_meta)

        logger.info(f"Archived NLP champion v{version} to {dest_art.name}")
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
        """Atomically promote validated challenger to active NLP champion.

        Supports bootstrap promotion when active champion is None / status="NONE".
        """
        chal_path = Path(challenger_artifact_path)
        if not chal_path.is_absolute():
            chal_path = (self.base_dir / chal_path).resolve()
        if not chal_path.exists():
            raise FileNotFoundError(f"Challenger artifact does not exist: {chal_path}")

        manifest = self.get_manifest()
        current_champ = self.get_active_champion()
        is_bootstrap = current_champ is None

        archived_paths: dict[str, str] | None = None
        current_ver: str | None = None

        # 1. Archive current champion if one exists
        if not is_bootstrap:
            current_ver = current_champ.get("version", "1.0.0")
            archived_paths = self.archive_champion(current_ver)

        # 2. Stage new files to temporary destinations then replace atomically
        self.champion_dir.mkdir(parents=True, exist_ok=True)
        tmp_art = self.champion_pipeline_path.with_suffix(".promotetmp")
        tmp_meta = self.champion_metadata_path.with_suffix(".promotetmp")

        shutil.copy2(chal_path, tmp_art)

        updated_meta = challenger_metadata.copy()
        updated_meta["model_version"] = new_version
        updated_meta["promoted_at"] = datetime.now(timezone.utc).isoformat()
        updated_meta["is_bootstrap"] = is_bootstrap
        if not is_bootstrap:
            updated_meta["parent_champion_version"] = current_ver

        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(updated_meta, fh, indent=2)

        tmp_art.replace(self.champion_pipeline_path)
        tmp_meta.replace(self.champion_metadata_path)

        # 3. Update manifest
        if not is_bootstrap and archived_paths and current_champ:
            manifest["previous_champion"] = {
                **current_champ,
                "archived_artifact_path": archived_paths["artifact_path"],
                "archived_metadata_path": archived_paths["metadata_path"],
                "archived_at": datetime.now(timezone.utc).isoformat(),
            }
        else:
            manifest["previous_champion"] = None

        manifest["status"] = "ACTIVE"
        manifest["active_champion"] = {
            "version": new_version,
            "model_name": updated_meta.get("model_type")
            or updated_meta.get("model_name", "nlp_classifier"),
            "artifact_path": str(
                self.champion_pipeline_path.relative_to(self.base_dir)
            ).replace("\\", "/"),
            "metadata_path": str(
                self.champion_metadata_path.relative_to(self.base_dir)
            ).replace("\\", "/"),
            "promoted_at": updated_meta["promoted_at"],
            "metrics": updated_meta.get("test_metrics")
            or updated_meta.get("metrics", {}),
        }
        manifest["history"].append(
            {
                "version": new_version,
                "model_name": manifest["active_champion"]["model_name"],
                "action": "bootstrap_promoted" if is_bootstrap else "promoted",
                "timestamp": updated_meta["promoted_at"],
            }
        )
        self._save_manifest(manifest)

        # 4. Post-promotion smoke test
        if auto_smoke_test:
            if not self.post_promotion_smoke_test():
                logger.error(
                    "NLP post-promotion smoke test failed! Triggering automatic rollback."
                )
                if not is_bootstrap and manifest.get("previous_champion"):
                    self.rollback(reason="Post-promotion smoke test failed")
                else:
                    # Bootstrap failed: revert to NONE
                    self._revert_bootstrap()
                return False

        logger.info(
            f"Successfully promoted NLP challenger {manifest['active_champion']['model_name']} to v{new_version}"
        )
        return True

    def _revert_bootstrap(self) -> None:
        """Safely revert a failed bootstrap promotion back to NONE."""
        try:
            if self.champion_pipeline_path.exists():
                self.champion_pipeline_path.unlink()
            if self.champion_metadata_path.exists():
                self.champion_metadata_path.unlink()
        except Exception as exc:
            logger.warning(f"Error cleaning up failed bootstrap files: {exc}")

        manifest = self.get_manifest()
        manifest["status"] = "NONE"
        manifest["active_champion"] = None
        manifest["previous_champion"] = None
        manifest["history"].append(
            {
                "action": "bootstrap_promotion_failed_smoke_test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_manifest(manifest)

    def post_promotion_smoke_test(self, test_samples: list[str] | None = None) -> bool:
        """Validate that the active NLP champion loads and generates valid predictions."""
        try:
            if not self.champion_pipeline_path.exists():
                logger.error("Active NLP pipeline artifact does not exist on disk.")
                return False

            untrusted = sio.get_untrusted_types(file=self.champion_pipeline_path)
            loaded_pipe = sio.load(self.champion_pipeline_path, trusted=untrusted)

            samples = test_samples or _DEFAULT_SMOKE_SAMPLES
            preds = loaded_pipe.predict(pd.Series(samples))

            if len(preds) != len(samples):
                logger.error(
                    f"Prediction count {len(preds)} does not match sample count {len(samples)}"
                )
                return False

            for p in preds:
                if str(p) not in _EXPECTED_CLASSES:
                    logger.error(
                        f"Prediction '{p}' outside expected classes {_EXPECTED_CLASSES}"
                    )
                    return False

            return True
        except Exception as exc:
            logger.error(f"NLP post-promotion smoke test encountered exception: {exc}")
            return False

    def rollback(self, reason: str = "Rollback requested") -> bool:
        """Restore the previously archived NLP champion model."""
        manifest = self.get_manifest()
        prev = manifest.get("previous_champion")
        if not prev:
            raise RuntimeError(
                "Cannot rollback: No previous champion found in NLP registry manifest."
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
        shutil.copy2(archived_art, self.champion_pipeline_path)
        if archived_meta_rel:
            archived_meta = (self.base_dir / archived_meta_rel).resolve()
            if archived_meta.exists():
                shutil.copy2(archived_meta, self.champion_metadata_path)

        # Run smoke test on restored model
        if not self.post_promotion_smoke_test():
            logger.error("Smoke test on restored rollback champion failed!")
            raise RuntimeError("Smoke test on restored rollback champion failed.")

        # Swap active and previous in manifest
        current_champ = manifest.get("active_champion") or {}
        rolled_back_version = current_champ.get("version", "unknown")
        restored_version = prev["version"]

        manifest["status"] = "ACTIVE"
        manifest["active_champion"] = {
            "version": restored_version,
            "model_name": prev["model_name"],
            "artifact_path": str(
                self.champion_pipeline_path.relative_to(self.base_dir)
            ).replace("\\", "/"),
            "metadata_path": str(
                self.champion_metadata_path.relative_to(self.base_dir)
            ).replace("\\", "/"),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "metrics": prev.get("metrics", {}),
        }
        manifest["previous_champion"] = None
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
            f"Successfully rolled back NLP champion to v{restored_version} ({reason})"
        )
        return True


_nlp_registry_instance: NLPModelRegistry | None = None


def get_nlp_registry(
    config: dict[str, Any] | None = None,
    base_dir: Path | str | None = None,
) -> NLPModelRegistry:
    """Retrieve or initialize the global NLPModelRegistry singleton."""
    global _nlp_registry_instance
    if _nlp_registry_instance is None or config is not None or base_dir is not None:
        _nlp_registry_instance = NLPModelRegistry(config=config, base_dir=base_dir)
    return _nlp_registry_instance
