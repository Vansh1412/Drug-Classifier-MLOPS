"""
src/evaluate.py — Model evaluation utilities for MedMLOps-X.

Computes standard classification metrics for a fitted pipeline.
All metrics are derived from actual model predictions — never hardcoded.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics on the held-out test set.

    Args:
        pipeline: A fitted sklearn Pipeline.
        X_test: Test feature matrix.
        y_test: Test target vector.

    Returns:
        Dictionary of {metric_name: value}.
    """
    preds = pipeline.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_test, preds, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_test, preds, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, preds, average="macro", zero_division=0)
        ),
    }


def cross_validate_pipeline(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Run stratified k-fold cross-validation on the training set.

    Args:
        pipeline: An unfitted sklearn Pipeline.
        X_train: Training feature matrix.
        y_train: Training target vector.
        cfg: Full configuration dictionary.

    Returns:
        Dictionary of mean/std CV metrics.
    """
    cv = StratifiedKFold(
        n_splits=cfg["data"]["cv_folds"],
        shuffle=True,
        random_state=cfg["data"]["cv_random_state"],
    )

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "macro_precision": "precision_macro",
        "macro_recall": "recall_macro",
    }

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,  # n_jobs=1 for determinism
        return_train_score=False,
    )

    return {
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "cv_macro_f1_mean": float(np.mean(cv_results["test_macro_f1"])),
        "cv_macro_f1_std": float(np.std(cv_results["test_macro_f1"])),
        "cv_macro_precision_mean": float(np.mean(cv_results["test_macro_precision"])),
        "cv_macro_recall_mean": float(np.mean(cv_results["test_macro_recall"])),
    }


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict[str, Any],
    model_name: str,
    model_params: dict[str, Any],
) -> dict[str, Any]:
    """Full evaluation run: CV on train, final fit + test metrics.

    Args:
        pipeline: Unfitted sklearn Pipeline.
        X_train, y_train: Training split.
        X_test, y_test: Test split.
        cfg: Configuration dictionary.
        model_name: Human-readable candidate identifier.
        model_params: Estimator hyperparameters dict.

    Returns:
        Complete evaluation result dictionary.
    """
    # Cross-validation (on training set only — no leakage)
    cv_metrics = cross_validate_pipeline(pipeline, X_train, y_train, cfg)

    # Final fit on full training set + evaluate on test
    t_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time_s = time.perf_counter() - t_start

    test_metrics = evaluate_on_test(pipeline, X_test, y_test)

    return {
        "model_name": model_name,
        "model_params": model_params,
        "training_time_seconds": round(training_time_s, 4),
        **cv_metrics,
        **test_metrics,
    }
