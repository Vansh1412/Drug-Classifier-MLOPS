"""
src/model_registry.py — Model candidate factory for MedMLOps-X competition pipeline.

Instantiates all configured model candidates from configs/model_config.yaml.
Each candidate is returned as an unfitted estimator ready for pipeline integration.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


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
