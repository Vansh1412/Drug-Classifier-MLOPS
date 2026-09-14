"""
src/data_validation.py — Pandera schema validation for the Drug classification dataset.

Validates Data/drug.csv (or any equivalent DataFrame) against the dataset contract
defined in configs/model_config.yaml. Validation errors produce actionable messages.

Usage:
    from src.data_validation import validate_dataset, load_and_validate
    df = validate_dataset(df, cfg)           # validate a loaded DataFrame
    df = load_and_validate("Data/drug.csv", cfg)  # load + validate in one step
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check


def build_schema(cfg: dict[str, Any]) -> DataFrameSchema:
    """Build a Pandera DataFrameSchema from configuration constraints.

    Args:
        cfg: Full configuration dictionary (from load_config()).

    Returns:
        Pandera DataFrameSchema for the drug dataset.
    """
    v = cfg["validation"]

    schema = DataFrameSchema(
        columns={
            "Age": Column(
                dtype=int,
                checks=[
                    Check.greater_than_or_equal_to(v["age_min"],
                        error=f"Age must be >= {v['age_min']}"),
                    Check.less_than_or_equal_to(v["age_max"],
                        error=f"Age must be <= {v['age_max']}"),
                ],
                nullable=False,
                description="Patient age in years",
            ),
            "Sex": Column(
                dtype=str,
                checks=Check.isin(v["sex_categories"],
                    error=f"Sex must be one of {v['sex_categories']}"),
                nullable=False,
                description="Patient biological sex",
            ),
            "BP": Column(
                dtype=str,
                checks=Check.isin(v["bp_categories"],
                    error=f"BP must be one of {v['bp_categories']}"),
                nullable=False,
                description="Blood pressure level",
            ),
            "Cholesterol": Column(
                dtype=str,
                checks=Check.isin(v["cholesterol_categories"],
                    error=f"Cholesterol must be one of {v['cholesterol_categories']}"),
                nullable=False,
                description="Cholesterol level",
            ),
            "Na_to_K": Column(
                dtype=float,
                checks=[
                    Check.greater_than_or_equal_to(v["na_to_k_min"],
                        error=f"Na_to_K must be >= {v['na_to_k_min']}"),
                    Check.less_than_or_equal_to(v["na_to_k_max"],
                        error=f"Na_to_K must be <= {v['na_to_k_max']}"),
                    Check(lambda s: s.apply(lambda x: x == x and x != float("inf")),
                        error="Na_to_K must be finite and non-NaN"),
                ],
                nullable=False,
                description="Sodium to potassium ratio in blood",
            ),
            "Drug": Column(
                dtype=str,
                checks=Check.isin(v["valid_drug_classes"],
                    error=f"Drug must be one of {v['valid_drug_classes']}"),
                nullable=False,
                description="Recommended drug class (target label)",
            ),
        },
        strict=True,  # Reject columns not in schema
        coerce=False,  # Do not silently coerce types
        name="DrugDatasetSchema",
    )
    return schema


def validate_dataset(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Validate a DataFrame against the drug dataset schema.

    Args:
        df: DataFrame to validate.
        cfg: Full configuration dictionary.

    Returns:
        The validated DataFrame (unchanged if valid).

    Raises:
        pandera.errors.SchemaError: If any validation check fails.
        pandera.errors.SchemaErrors: If multiple validation failures occur.
    """
    schema = build_schema(cfg)
    validated = schema.validate(df, lazy=True)
    return validated


def load_and_validate(data_path: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """Load a CSV file and validate it against the drug dataset schema.

    Args:
        data_path: Path to the CSV file.
        cfg: Full configuration dictionary.

    Returns:
        Validated DataFrame.

    Raises:
        FileNotFoundError: If the data file does not exist.
        pandera.errors.SchemaError: If validation fails.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    return validate_dataset(df, cfg)
