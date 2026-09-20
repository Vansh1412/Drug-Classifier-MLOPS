"""
src/nlp_competition.py — NLP Model Competition Pipeline for Drug Review Intelligence.

Implements Sprint 8 NLP competition:
1. Target Derivation: rating (1-10) -> sentiment_class ('low' 1-4, 'neutral' 5-7, 'high' 8-10).
2. Leakage-Safe Splitting: Group-aware splitting ensuring zero review text overlap.
3. TF-IDF Feature Extraction: Fitted strictly on training folds to eliminate data leakage.
4. Model Competition: Logistic Regression, Linear SVM, and Multinomial Naive Bayes.
5. Group-Aware Cross-Validation: StratifiedGroupKFold evaluating Macro F1 and Accuracy.
6. Quality Gate: Enforces Macro F1 >= 0.70 and Accuracy >= 0.70; supports NO CHAMPION outcome.
7. MLflow Experiment Tracking: Dedicated NLP experiment logging.
8. Champion Artifact: Serialized Pipeline in isolated Model/nlp/ directory.

Usage:
    from src.nlp_competition import run_nlp_competition
    report = run_nlp_competition()
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import load_config
from src.nlp_preprocessing import (
    run_review_preprocessing,
    split_reviews_leakage_safe,
)
from src.nlp_tracking import (
    get_dvc_dataset_hash,
    get_git_commit_sha,
    log_nlp_candidate_run,
    log_nlp_champion_run,
    setup_nlp_mlflow,
)

logger = logging.getLogger(__name__)


def derive_sentiment_target(
    df: pd.DataFrame,
    rating_col: str = "rating",
    target_col: str = "sentiment_class",
) -> pd.DataFrame:
    """Derive 3-class sentiment target from numerical drug review rating.

    Bins:
        1–4  -> 'low'
        5–7  -> 'neutral'
        8–10 -> 'high'

    Preserves the original numeric rating column for auditability and lineage.

    Args:
        df: Input DataFrame containing rating_col.
        rating_col: Name of the column with numeric ratings (1-10).
        target_col: Name of the resulting target column to create.

    Returns:
        DataFrame with the target_col added.

    Raises:
        ValueError: If rating_col is missing or contains values outside [1, 10].
    """
    if rating_col not in df.columns:
        raise ValueError(
            f"Rating column '{rating_col}' not found in DataFrame columns: {list(df.columns)}"
        )

    out = df.copy()
    ratings = out[rating_col]

    if ratings.isnull().any():
        raise ValueError(f"Rating column '{rating_col}' contains null values.")

    invalid_ratings = ratings[(ratings < 1) | (ratings > 10)]
    if not invalid_ratings.empty:
        raise ValueError(
            f"Rating column contains values outside [1, 10]: {invalid_ratings.unique().tolist()}"
        )

    out[target_col] = pd.cut(
        ratings,
        bins=[0, 4, 7, 10],
        labels=["low", "neutral", "high"],
    ).astype(str)

    return out


def build_nlp_pipeline(
    model_name: str,
    cfg: dict[str, Any] | None = None,
    tfidf_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build an sklearn Pipeline combining TfidfVectorizer and a classifier.

    Supported models:
    - 'logistic_regression'
    - 'linear_svm'
    - 'naive_bayes'

    Args:
        model_name: Identifier for the model architecture.
        cfg: Configuration dictionary. If None, loaded via load_config().
        tfidf_params: Optional dictionary to override TF-IDF vectorizer settings.

    Returns:
        Configured sklearn Pipeline instance.

    Raises:
        ValueError: If model_name is not supported.
    """
    if cfg is None:
        cfg = load_config()

    nlp_cfg = cfg.get("reviews", {}).get("nlp_competition", {})
    tfidf_cfg = dict(nlp_cfg.get("tfidf", {}))
    if tfidf_params:
        tfidf_cfg.update(tfidf_params)
    models_cfg = nlp_cfg.get("models", {})

    ngram_range = tuple(tfidf_cfg.get("ngram_range", [1, 2]))
    min_df = tfidf_cfg.get("min_df", 3)
    max_df = tfidf_cfg.get("max_df", 0.90)
    sublinear_tf = tfidf_cfg.get("sublinear_tf", True)
    max_features = tfidf_cfg.get("max_features", 10000)

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
    )

    if model_name == "logistic_regression":
        m_cfg = models_cfg.get("logistic_regression", {})
        clf = LogisticRegression(
            C=float(m_cfg.get("C", 1.0)),
            max_iter=int(m_cfg.get("max_iter", 1000)),
            solver=str(m_cfg.get("solver", "lbfgs")),
            class_weight=m_cfg.get("class_weight", "balanced"),
            random_state=int(m_cfg.get("random_state", 125)),
        )
    elif model_name == "linear_svm":
        m_cfg = models_cfg.get("linear_svm", {})
        clf = LinearSVC(
            C=float(m_cfg.get("C", 1.0)),
            max_iter=int(m_cfg.get("max_iter", 2000)),
            class_weight=m_cfg.get("class_weight", "balanced"),
            random_state=int(m_cfg.get("random_state", 125)),
        )
    elif model_name == "naive_bayes":
        m_cfg = models_cfg.get("naive_bayes", {})
        clf = MultinomialNB(
            alpha=float(m_cfg.get("alpha", 0.5)),
        )
    else:
        raise ValueError(
            f"Unsupported NLP model '{model_name}'. "
            f"Supported: 'logistic_regression', 'linear_svm', 'naive_bayes'."
        )

    return Pipeline([("tfidf", vectorizer), ("classifier", clf)])


def evaluate_cv_leakage_safe(
    pipeline: Pipeline,
    X: pd.Series,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    random_state: int = 125,
) -> dict[str, float]:
    """Perform group-aware stratified cross-validation without data leakage.

    Because the vectorizer is inside the Pipeline, fitting on each fold's training
    indices ensures that vocabulary and IDF statistics never leak from validation folds.
    Identical reviews are kept within the same fold via groups.

    Args:
        pipeline: Sklearn Pipeline containing ('tfidf', 'classifier').
        X: Series of review text.
        y: Series of target classes.
        groups: Series of group identifiers (e.g. review_clean_hash).
        n_splits: Number of CV folds.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary of mean and std for Macro F1, Accuracy, Precision, and Recall.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    f1_scores: list[float] = []
    accuracy_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_pipe = clone(pipeline)
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        fold_pipe.fit(X_tr, y_tr)
        y_pred = fold_pipe.predict(X_val)

        f1_scores.append(
            float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        )
        accuracy_scores.append(float(accuracy_score(y_val, y_pred)))
        precision_scores.append(
            float(precision_score(y_val, y_pred, average="macro", zero_division=0))
        )
        recall_scores.append(
            float(recall_score(y_val, y_pred, average="macro", zero_division=0))
        )

    return {
        "cv_macro_f1_mean": float(np.mean(f1_scores)),
        "cv_macro_f1_std": float(np.std(f1_scores)),
        "cv_accuracy_mean": float(np.mean(accuracy_scores)),
        "cv_accuracy_std": float(np.std(accuracy_scores)),
        "cv_precision_macro_mean": float(np.mean(precision_scores)),
        "cv_precision_macro_std": float(np.std(precision_scores)),
        "cv_recall_macro_mean": float(np.mean(recall_scores)),
        "cv_recall_macro_std": float(np.std(recall_scores)),
    }


def evaluate_test_set(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    classes: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate fitted NLP pipeline on held-out test set.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test: Series of test review texts.
        y_test: Series of ground-truth test labels.
        classes: Ordered list of class names. Defaults to ['low', 'neutral', 'high'].

    Returns:
        Dictionary containing overall metrics, per-class metrics, and confusion matrix.
    """
    if classes is None:
        classes = ["low", "neutral", "high"]

    y_pred = pipeline.predict(X_test)

    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro = float(
        precision_score(y_test, y_pred, average="macro", zero_division=0)
    )
    recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_test, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    class_metrics = {}
    for cls in classes:
        cls_data = report.get(cls, {})
        class_metrics[cls] = {
            "precision": float(cls_data.get("precision", 0.0)),
            "recall": float(cls_data.get("recall", 0.0)),
            "f1": float(cls_data.get("f1-score", 0.0)),
            "support": int(cls_data.get("support", 0)),
        }

    return {
        "test_macro_f1": macro_f1,
        "test_accuracy": accuracy,
        "test_precision_macro": precision_macro,
        "test_recall_macro": recall_macro,
        "class_metrics": class_metrics,
        "confusion_matrix": cm,
        "classes": classes,
    }


def evaluate_nlp_quality_gate(
    cv_metrics: dict[str, float],
    test_metrics: dict[str, Any],
    gate_cfg: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Evaluate whether candidate model meets production NLP quality gate criteria.

    Criteria:
    1. CV Macro F1 >= min_macro_f1 (default 0.70)
    2. Test Macro F1 >= min_macro_f1 (default 0.70)
    3. Test Accuracy >= min_accuracy (default 0.70)
    4. Per-class recall >= min_class_recall (default 0.40) to prevent minority class collapse

    Args:
        cv_metrics: CV metrics dictionary.
        test_metrics: Test metrics dictionary.
        gate_cfg: Quality gate configuration.

    Returns:
        Tuple of (passed: bool, failure_reasons: list[str]).
    """
    min_macro_f1 = float(gate_cfg.get("min_macro_f1", 0.70))
    min_accuracy = float(gate_cfg.get("min_accuracy", 0.70))
    min_class_recall = float(gate_cfg.get("min_class_recall", 0.40))

    reasons: list[str] = []

    cv_f1 = cv_metrics.get("cv_macro_f1_mean", 0.0)
    if cv_f1 < min_macro_f1:
        reasons.append(
            f"CV Macro F1 ({cv_f1:.4f}) below threshold ({min_macro_f1:.4f})"
        )

    test_f1 = test_metrics.get("test_macro_f1", 0.0)
    if test_f1 < min_macro_f1:
        reasons.append(
            f"Test Macro F1 ({test_f1:.4f}) below threshold ({min_macro_f1:.4f})"
        )

    test_acc = test_metrics.get("test_accuracy", 0.0)
    if test_acc < min_accuracy:
        reasons.append(
            f"Test Accuracy ({test_acc:.4f}) below threshold ({min_accuracy:.4f})"
        )

    class_metrics = test_metrics.get("class_metrics", {})
    for cls_name, m in class_metrics.items():
        recall = m.get("recall", 0.0)
        if recall < min_class_recall:
            reasons.append(
                f"Class '{cls_name}' recall ({recall:.4f}) below threshold ({min_class_recall:.4f})"
            )

    passed = len(reasons) == 0
    return passed, reasons


def select_nlp_champion(
    candidates: list[dict[str, Any]],
    gate_cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Select champion NLP model based on CV Macro F1 among gate-passing models.

    If no models pass the quality gate, returns None.

    Args:
        candidates: List of evaluated candidate result dictionaries.
        gate_cfg: Quality gate configuration.

    Returns:
        Winning candidate dictionary or None if no candidate passed.
    """
    passing_candidates = [c for c in candidates if c.get("quality_gate_passed", False)]

    if not passing_candidates:
        logger.warning(
            "No candidate passed the NLP quality gate. Outcome: NO NLP CHAMPION."
        )
        return None

    # Rank primary by cv_macro_f1_mean (descending), secondary by test_macro_f1 (descending)
    sorted_candidates = sorted(
        passing_candidates,
        key=lambda c: (
            c.get("cv_metrics", {}).get("cv_macro_f1_mean", 0.0),
            c.get("test_metrics", {}).get("test_macro_f1", 0.0),
        ),
        reverse=True,
    )
    champion = sorted_candidates[0]
    logger.info(
        f"NLP Champion selected: {champion['model_name']} "
        f"(CV Macro F1: {champion['cv_metrics']['cv_macro_f1_mean']:.4f}, "
        f"Test Macro F1: {champion['test_metrics']['test_macro_f1']:.4f})"
    )
    return champion


def save_nlp_champion(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist the NLP champion pipeline and its metadata to the isolated Model/nlp/ directory.

    Guarantees:
    - Never touches or overwrites Model/drug_pipeline.skops or Model/model_metadata.json.

    Args:
        pipeline: Trained champion Pipeline.
        metadata: Metadata dictionary to save alongside the model.
        cfg: Configuration dictionary containing artifact paths.

    Returns:
        Tuple of (pipeline_path, metadata_path).
    """
    art_cfg = cfg.get("reviews", {}).get("nlp_competition", {}).get("artifacts", {})
    pipeline_path = _PROJECT_ROOT / art_cfg.get(
        "pipeline_path", "Model/nlp/champion/nlp_champion_pipeline.skops"
    )
    metadata_path = _PROJECT_ROOT / art_cfg.get(
        "metadata_path", "Model/nlp/nlp_model_metadata.json"
    )

    # Ensure parent directories exist
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Save pipeline safely with skops
    sio.dump(pipeline, pipeline_path)
    logger.info(f"Saved NLP champion pipeline to: {pipeline_path}")

    # Save metadata as formatted JSON
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info(f"Saved NLP champion metadata to: {metadata_path}")

    return pipeline_path, metadata_path


def run_nlp_competition(
    cfg: dict[str, Any] | None = None,
    save_champion: bool = True,
    sample_n: int | None = None,
) -> dict[str, Any]:
    """Run the complete NLP Model Competition pipeline.

    Steps:
    1. Load processed reviews dataset (or trigger preprocessing if missing).
    2. Derive sentiment target (1-4 -> 'low', 5-7 -> 'neutral', 8-10 -> 'high').
    3. Perform leakage-safe stratified group splitting.
    4. Initialize dedicated NLP MLflow experiment.
    5. Evaluate Logistic Regression, Linear SVM, and Multinomial Naive Bayes
       via 5-fold StratifiedGroupKFold CV and test set evaluation.
    6. Apply NLP quality gate.
    7. Select and persist NLP champion artifact.
    8. Generate competition report and metrics files.

    Args:
        cfg: Optional configuration dictionary.
        save_champion: Whether to persist winning pipeline and metadata.
        sample_n: Optional row limit for accelerated testing.

    Returns:
        Complete competition summary report dictionary.
    """
    if cfg is None:
        cfg = load_config()

    nlp_cfg = cfg.get("reviews", {}).get("nlp_competition", {})
    target_cfg = nlp_cfg.get("target", {})
    split_cfg = cfg.get("reviews", {}).get("splitting", {})
    gate_cfg = nlp_cfg.get("quality_gate", {})
    art_cfg = nlp_cfg.get("artifacts", {})

    processed_data_path = _PROJECT_ROOT / cfg.get("reviews", {}).get(
        "processed_data_path", "Data/reviews/processed/reviews_cleaned.parquet"
    )

    # 1. Dataset loading / generation
    if not processed_data_path.exists():
        raw_path = _PROJECT_ROOT / cfg.get("reviews", {}).get(
            "raw_data_path", "Data/reviews/drugsComTest_raw.csv"
        )
        logger.info(f"Cleaned parquet not found. Generating from raw: {raw_path}")
        run_review_preprocessing(raw_path, cfg=cfg)

    logger.info(f"Loading cleaned reviews from: {processed_data_path}")
    df = pd.read_parquet(processed_data_path)

    if sample_n is not None and sample_n < len(df):
        logger.info(f"Sampling {sample_n} rows for execution.")
        df = df.sample(n=sample_n, random_state=125).reset_index(drop=True)

    # 2. Derive sentiment target
    rating_col = target_cfg.get("rating_col", "rating")
    target_col = target_cfg.get("name", "sentiment_class")
    df = derive_sentiment_target(df, rating_col=rating_col, target_col=target_col)

    class_counts = df[target_col].value_counts().to_dict()
    total_rows = len(df)
    class_distribution = {
        k: {"count": int(v), "percentage": round((v / total_rows) * 100, 2)}
        for k, v in class_counts.items()
    }
    logger.info(f"Target distribution ({total_rows} rows): {class_distribution}")

    # 3. Leakage-safe dataset splitting
    train_df, test_df = split_reviews_leakage_safe(
        df,
        strategy=split_cfg.get("default_strategy", "stratified_group"),
        test_size=float(split_cfg.get("test_size", 0.2)),
        random_state=int(split_cfg.get("random_state", 125)),
        stratify_col=target_col,
        group_col=split_cfg.get("group_column", "review_clean_hash"),
    )

    # Verify zero text overlap
    train_texts = set(train_df["review_clean"])
    test_texts = set(test_df["review_clean"])
    text_overlap = len(train_texts.intersection(test_texts))
    if text_overlap > 0:
        raise RuntimeError(
            f"CRITICAL: Data leakage detected! {text_overlap} identical review texts in both train and test."
        )

    logger.info(
        f"Leakage-safe split complete: train={len(train_df)} rows, test={len(test_df)} rows. "
        f"Text overlap={text_overlap} (0 required)."
    )

    # 4. Initialize MLflow
    setup_nlp_mlflow(cfg)

    # 5. Model competition
    models_to_evaluate = ["logistic_regression", "linear_svm", "naive_bayes"]
    candidates: list[dict[str, Any]] = []

    X_train = train_df["review_clean"]
    y_train = train_df[target_col]
    groups_train = train_df[split_cfg.get("group_column", "review_clean_hash")]

    X_test = test_df["review_clean"]
    y_test = test_df[target_col]

    cv_folds = int(nlp_cfg.get("cv", {}).get("folds", 5))
    cv_seed = int(nlp_cfg.get("cv", {}).get("random_state", 125))

    for model_name in models_to_evaluate:
        logger.info(f"Evaluating candidate: {model_name}...")
        pipeline = build_nlp_pipeline(model_name, cfg=cfg)

        # Cross-validation
        cv_metrics = evaluate_cv_leakage_safe(
            pipeline=pipeline,
            X=X_train,
            y=y_train,
            groups=groups_train,
            n_splits=cv_folds,
            random_state=cv_seed,
        )

        # Fit on full training set and evaluate on test set
        pipeline.fit(X_train, y_train)
        test_metrics = evaluate_test_set(
            pipeline=pipeline,
            X_test=X_test,
            y_test=y_test,
            classes=target_cfg.get("classes", ["low", "neutral", "high"]),
        )

        # Quality gate check
        gate_passed, gate_reasons = evaluate_nlp_quality_gate(
            cv_metrics=cv_metrics,
            test_metrics=test_metrics,
            gate_cfg=gate_cfg,
        )

        # Log candidate run to MLflow
        model_params = dict(pipeline.get_params())
        # Filter non-serializable params
        clean_params = {
            "model_name": model_name,
            "tfidf_ngram_range": str(nlp_cfg.get("tfidf", {}).get("ngram_range")),
            "tfidf_max_features": nlp_cfg.get("tfidf", {}).get("max_features"),
            "tfidf_sublinear_tf": nlp_cfg.get("tfidf", {}).get("sublinear_tf"),
            "split_strategy": split_cfg.get("default_strategy", "stratified_group"),
            "cv_folds": cv_folds,
        }
        for pk in ["C", "max_iter", "solver", "class_weight", "alpha"]:
            if f"classifier__{pk}" in model_params:
                clean_params[f"classifier_{pk}"] = str(
                    model_params[f"classifier__{pk}"]
                )

        run_metrics = {
            "cv_macro_f1_mean": cv_metrics["cv_macro_f1_mean"],
            "cv_macro_f1_std": cv_metrics["cv_macro_f1_std"],
            "cv_accuracy_mean": cv_metrics["cv_accuracy_mean"],
            "cv_accuracy_std": cv_metrics["cv_accuracy_std"],
            "test_macro_f1": test_metrics["test_macro_f1"],
            "test_accuracy": test_metrics["test_accuracy"],
            "test_precision_macro": test_metrics["test_precision_macro"],
            "test_recall_macro": test_metrics["test_recall_macro"],
        }
        for cls_name, cm in test_metrics["class_metrics"].items():
            run_metrics[f"test_f1_{cls_name}"] = cm["f1"]
            run_metrics[f"test_recall_{cls_name}"] = cm["recall"]

        run_id = log_nlp_candidate_run(
            name=model_name,
            params=clean_params,
            metrics=run_metrics,
            tags={"quality_gate_passed": str(gate_passed)},
        )

        candidate_record = {
            "model_name": model_name,
            "mlflow_run_id": run_id,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "quality_gate_passed": gate_passed,
            "quality_gate_reasons": gate_reasons,
            "pipeline": pipeline,
        }
        candidates.append(candidate_record)

        logger.info(
            f"Candidate '{model_name}': CV Macro F1={cv_metrics['cv_macro_f1_mean']:.4f}, "
            f"Test Macro F1={test_metrics['test_macro_f1']:.4f}, "
            f"Test Acc={test_metrics['test_accuracy']:.4f}, Gate={'PASS' if gate_passed else 'FAIL'}"
        )

    # 6. Champion selection
    champion = select_nlp_champion(candidates, gate_cfg=gate_cfg)

    # 7. Champion artifact persistence & metadata
    champion_info: dict[str, Any] | None = None
    if champion is not None:
        champ_name = champion["model_name"]
        champ_pipeline = champion["pipeline"]

        metadata = {
            "workload": "nlp",
            "model_type": champ_name,
            "model_version": "1.0.0",
            "task": "patient_reported_sentiment_classification",
            "classes": target_cfg.get("classes", ["low", "neutral", "high"]),
            "target_mapping": {
                "low": "ratings 1-4",
                "neutral": "ratings 5-7",
                "high": "ratings 8-10",
            },
            "training_dataset": {
                "raw_path": cfg.get("reviews", {}).get("raw_data_path"),
                "rows_train": len(train_df),
                "rows_test": len(test_df),
                "dvc_hash": get_dvc_dataset_hash(),
                "git_commit": get_git_commit_sha(),
                "split_strategy": split_cfg.get("default_strategy", "stratified_group"),
                "text_overlap": 0,
            },
            "metrics": {
                "cv_macro_f1_mean": champion["cv_metrics"]["cv_macro_f1_mean"],
                "cv_macro_f1_std": champion["cv_metrics"]["cv_macro_f1_std"],
                "cv_accuracy_mean": champion["cv_metrics"]["cv_accuracy_mean"],
                "cv_accuracy_std": champion["cv_metrics"]["cv_accuracy_std"],
                "test_macro_f1": champion["test_metrics"]["test_macro_f1"],
                "test_accuracy": champion["test_metrics"]["test_accuracy"],
                "test_precision_macro": champion["test_metrics"][
                    "test_precision_macro"
                ],
                "test_recall_macro": champion["test_metrics"]["test_recall_macro"],
                "class_metrics": champion["test_metrics"]["class_metrics"],
            },
            "quality_gate": {
                "passed": True,
                "thresholds": gate_cfg,
            },
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "medical_safety_boundary": (
                "Patient-reported drug experience analytics only. "
                "Not for clinical decision-support, diagnosis, or prescribing."
            ),
        }

        if save_champion:
            pipe_path, meta_path = save_nlp_champion(champ_pipeline, metadata, cfg=cfg)
            # Log champion run to MLflow
            log_nlp_champion_run(
                champion_name=champ_name,
                params={
                    "model_type": champ_name,
                    "split_strategy": split_cfg.get("default_strategy"),
                    "dvc_hash": get_dvc_dataset_hash(),
                },
                metrics={
                    "cv_macro_f1_mean": champion["cv_metrics"]["cv_macro_f1_mean"],
                    "test_macro_f1": champion["test_metrics"]["test_macro_f1"],
                    "test_accuracy": champion["test_metrics"]["test_accuracy"],
                },
                artifacts=[pipe_path, meta_path],
            )

        champion_info = {
            "model_name": champ_name,
            "cv_macro_f1_mean": champion["cv_metrics"]["cv_macro_f1_mean"],
            "test_macro_f1": champion["test_metrics"]["test_macro_f1"],
            "test_accuracy": champion["test_metrics"]["test_accuracy"],
            "mlflow_run_id": champion["mlflow_run_id"],
        }
    else:
        logger.warning("No champion selected because no model passed the quality gate.")

    # 8. Report compilation
    report: dict[str, Any] = {
        "workload": "nlp",
        "sprint": 8,
        "dataset_rows": total_rows,
        "class_distribution": class_distribution,
        "split_strategy": split_cfg.get("default_strategy", "stratified_group"),
        "text_overlap": text_overlap,
        "quality_gate_thresholds": gate_cfg,
        "champion": champion_info,
        "candidates": [
            {
                "model_name": c["model_name"],
                "mlflow_run_id": c["mlflow_run_id"],
                "cv_metrics": c["cv_metrics"],
                "test_metrics": c["test_metrics"],
                "quality_gate_passed": c["quality_gate_passed"],
                "quality_gate_reasons": c["quality_gate_reasons"],
            }
            for c in candidates
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "medical_safety_boundary": (
            "Model performance on patient-reported drug reviews should not be "
            "interpreted as clinical efficacy, safety, diagnosis, or prescribing evidence."
        ),
    }

    # Save reports
    report_path = _PROJECT_ROOT / art_cfg.get(
        "report_path", "Results/nlp_competition_report.json"
    )
    metrics_path = _PROJECT_ROOT / art_cfg.get(
        "metrics_path", "Results/nlp_metrics.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info(f"Saved NLP competition report to: {report_path}")

    # Metrics summary
    metrics_summary = {
        "champion": champion_info["model_name"] if champion_info else "NONE",
        "quality_gate_passed": champion_info is not None,
        "candidates": {
            c["model_name"]: {
                "cv_macro_f1": c["cv_metrics"]["cv_macro_f1_mean"],
                "cv_f1_std": c["cv_metrics"]["cv_macro_f1_std"],
                "test_macro_f1": c["test_metrics"]["test_macro_f1"],
                "test_accuracy": c["test_metrics"]["test_accuracy"],
                "gate_passed": c["quality_gate_passed"],
            }
            for c in candidates
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_summary, fh, indent=2)
    logger.info(f"Saved NLP metrics summary to: {metrics_path}")

    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_nlp_competition()
