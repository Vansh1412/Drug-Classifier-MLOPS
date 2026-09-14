"""
src/preprocessing.py — Shared sklearn ColumnTransformer factory for MedMLOps-X.

All model candidates use this identical preprocessing contract to ensure
a fair apples-to-apples comparison in the model competition.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def build_preprocessor(cfg: dict[str, Any]) -> ColumnTransformer:
    """Build the shared feature preprocessing ColumnTransformer.

    Categorical columns are ordinal-encoded.
    Numerical columns are median-imputed then standard-scaled.

    Args:
        cfg: Full configuration dictionary (from load_config()).

    Returns:
        A configured, unfitted ColumnTransformer.
    """
    cat_indices = cfg["data"]["categorical_indices"]
    num_indices = cfg["data"]["numerical_indices"]

    transformer = ColumnTransformer(
        transformers=[
            ("encoder", OrdinalEncoder(), cat_indices),
            ("num_imputer", SimpleImputer(strategy="median"), num_indices),
            ("num_scaler", StandardScaler(), num_indices),
        ]
    )
    return transformer
