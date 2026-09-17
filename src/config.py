"""
src/config.py — Centralized configuration loader for MedMLOps-X.

Loads and validates configs/model_config.yaml and provides typed access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Project root is one level above src/
_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "model_config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        config_path: Path to the YAML config. Defaults to configs/model_config.yaml.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file must be a YAML mapping, got: {type(cfg)}")
    return cfg


def get_trusted_types(cfg: dict[str, Any]) -> list[str]:
    """Return the list of trusted skops deserialization types."""
    return cfg["security"]["trusted_types"]
