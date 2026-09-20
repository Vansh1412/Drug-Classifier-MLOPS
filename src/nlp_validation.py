"""
src/nlp_validation.py — Pandera schema and data quality validation for Drug Review dataset.

Validates Data/reviews/drugsComTest_raw.csv against the review dataset contract defined
in configs/model_config.yaml. Validates data types, ranges, uniqueness of uniqueID,
and documents neutral handling for missing medical conditions.

Usage:
    from src.nlp_validation import validate_review_dataset, validate_review_schema, validate_review_data_quality
    df = validate_review_dataset("Data/reviews/drugsComTest_raw.csv")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path for direct script execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import pandera.errors as pa_err
from pandera import Check, Column, DataFrameSchema

from src.config import load_config

logger = logging.getLogger(__name__)


def build_review_schema(cfg: dict[str, Any] | None = None) -> DataFrameSchema:
    """Build a Pandera DataFrameSchema for the raw drug reviews dataset.

    Args:
        cfg: Optional configuration dictionary. If None, default configs are loaded.

    Returns:
        Pandera DataFrameSchema enforcing column types, non-null constraints,
        and rating bounds [1, 10].
    """
    if cfg is None:
        cfg = load_config()

    rev_cfg = cfg.get("reviews", {})
    val_cfg = rev_cfg.get("validation", {})
    rating_min = val_cfg.get("rating_min", 1)
    rating_max = val_cfg.get("rating_max", 10)
    useful_min = val_cfg.get("useful_count_min", 0)

    schema = DataFrameSchema(
        columns={
            "uniqueID": Column(
                dtype=int,
                nullable=False,
                unique=True,
                description="Unique review identifier",
            ),
            "drugName": Column(
                dtype=str,
                checks=Check(
                    lambda s: s.astype(str).str.strip().str.len() > 0,
                    error="drugName must be non-empty string",
                ),
                nullable=False,
                description="Name of the medication reviewed",
            ),
            "condition": Column(
                dtype=str,
                nullable=True,  # 295 missing conditions in raw dataset; allowed per contract
                coerce=False,
                description="Reported patient condition or indication (nullable)",
            ),
            "review": Column(
                dtype=str,
                checks=Check(
                    lambda s: s.astype(str).str.strip().str.len() > 0,
                    error="review must be non-empty string",
                ),
                nullable=False,
                description="Patient review text",
            ),
            "rating": Column(
                dtype=int,
                checks=[
                    Check.greater_than_or_equal_to(
                        rating_min, error=f"rating must be >= {rating_min}"
                    ),
                    Check.less_than_or_equal_to(
                        rating_max, error=f"rating must be <= {rating_max}"
                    ),
                ],
                nullable=False,
                description="User rating from 1 to 10",
            ),
            "date": Column(
                dtype=str,
                checks=Check(
                    lambda s: pd.to_datetime(
                        s, format="%d-%b-%y", errors="coerce"
                    ).notnull(),
                    error="date must be parseable as a valid datetime string (format: %d-%b-%y)",
                ),
                nullable=False,
                description="Date the review was posted",
            ),
            "usefulCount": Column(
                dtype=int,
                checks=Check.greater_than_or_equal_to(
                    useful_min, error=f"usefulCount must be >= {useful_min}"
                ),
                nullable=False,
                description="Number of users who found this review useful",
            ),
        },
        strict=True,
        coerce=False,
    )
    return schema


def validate_review_schema(
    df: pd.DataFrame, cfg: dict[str, Any] | None = None
) -> tuple[bool, list[str]]:
    """Validate a DataFrame against the Pandera review schema contract.

    Args:
        df: Review DataFrame to validate.
        cfg: Optional configuration dictionary.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    schema = build_review_schema(cfg)
    try:
        schema.validate(df, lazy=True)
        return True, []
    except pa_err.SchemaErrors as exc:
        errors = [str(err) for err in exc.failure_cases["check"].tolist()]
        return False, errors
    except Exception as exc:
        return False, [str(exc)]


def validate_review_data_quality(
    df: pd.DataFrame, cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Perform comprehensive data quality and anomaly inspection on reviews.

    Args:
        df: Review DataFrame to assess.
        cfg: Optional configuration dictionary.

    Returns:
        Structured dictionary reporting data quality metrics, status (VALID, WARNING, ERROR),
        warnings, and errors.
    """
    if cfg is None:
        cfg = load_config()

    rev_cfg = cfg.get("reviews", {})
    val_cfg = rev_cfg.get("validation", {})
    max_missing_cond_rate = val_cfg.get("max_missing_condition_rate", 0.05)

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Schema check
    is_schema_valid, schema_errors = validate_review_schema(df, cfg)
    if not is_schema_valid:
        errors.extend(schema_errors)

    row_count, col_count = df.shape

    # 2. Missing condition rate
    missing_cond_count = (
        int(df["condition"].isnull().sum()) if "condition" in df.columns else 0
    )
    missing_cond_rate = float(missing_cond_count / row_count) if row_count > 0 else 0.0

    if missing_cond_count > 0:
        if missing_cond_rate > max_missing_cond_rate:
            errors.append(
                f"Missing condition rate {missing_cond_rate:.4f} exceeds allowed limit {max_missing_cond_rate}"
            )
        else:
            warnings.append(
                f"{missing_cond_count} records ({missing_cond_rate:.2%}) have missing conditions; "
                f"documented neutral handling will be applied during preprocessing."
            )

    # 3. Duplicate checks
    dup_unique_id = (
        int(df["uniqueID"].duplicated().sum()) if "uniqueID" in df.columns else 0
    )
    if dup_unique_id > 0:
        errors.append(f"Found {dup_unique_id} duplicate uniqueIDs (must be unique)")

    dup_reviews = int(df["review"].duplicated().sum()) if "review" in df.columns else 0
    if dup_reviews > 0:
        warnings.append(
            f"Found {dup_reviews} duplicate review texts across records; "
            f"requires leakage-safe group splitting."
        )

    # 4. HTML entity detection
    html_entity_count = 0
    if "review" in df.columns:
        reviews_str = df["review"].astype(str)
        html_entity_count = int(
            reviews_str.str.contains(r"&#\d+;|&[a-zA-Z]+;", regex=True).sum()
        )
        if html_entity_count > 0:
            warnings.append(
                f"Found {html_entity_count} reviews with HTML escape entities; "
                f"requires html.unescape() during preprocessing."
            )

    # 5. Rating distribution
    rating_dist = {}
    if "rating" in df.columns:
        rating_dist = df["rating"].value_counts().sort_index().to_dict()

    # Determine overall status
    if errors:
        status = "ERROR"
    elif warnings:
        status = "WARNING"
    else:
        status = "VALID"

    return {
        "status": status,
        "row_count": row_count,
        "col_count": col_count,
        "missing_condition_count": missing_cond_count,
        "missing_condition_rate": missing_cond_rate,
        "duplicate_unique_id_count": dup_unique_id,
        "duplicate_review_text_count": dup_reviews,
        "html_entity_review_count": html_entity_count,
        "rating_distribution": rating_dist,
        "warnings": warnings,
        "errors": errors,
    }


def validate_review_dataset(
    csv_path: str | Path = "Data/reviews/drugsComTest_raw.csv",
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load and validate the review dataset.

    Args:
        csv_path: Path to the raw review CSV.
        cfg: Optional configuration dictionary.

    Returns:
        Validated pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If schema or data quality validation produces errors.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Review dataset not found at: {path}")

    logger.info(f"Loading review dataset from {path}")
    df = pd.read_csv(path)

    quality_report = validate_review_data_quality(df, cfg)
    if quality_report["status"] == "ERROR":
        error_msg = "; ".join(quality_report["errors"])
        raise ValueError(f"Review dataset validation FAILED: {error_msg}")

    for warning in quality_report["warnings"]:
        logger.warning(f"Review Data Quality Warning: {warning}")

    logger.info(
        f"Review dataset validated successfully: {df.shape[0]} rows, "
        f"{df.shape[1]} columns (status: {quality_report['status']})"
    )
    return df


if __name__ == "__main__":
    import sys

    target_path = (
        sys.argv[1] if len(sys.argv) > 1 else "Data/reviews/drugsComTest_raw.csv"
    )
    validated_df = validate_review_dataset(target_path)
    print(f"Validation PASSED: {validated_df.shape[0]} records loaded and validated.")
