"""
src/monitoring.py — Continuous Model Monitoring and Data Drift Detection for MedMLOps-X.

Provides production-grade statistical drift detection, data quality monitoring,
schema validation, and prediction distribution tracking comparing reference baseline
data with serving/current data.

Uses SciPy, NumPy, Pandas, and Pandera for zero-heavy-dependency, deterministic,
and reproducible statistical evaluations.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import get_trusted_types, load_config
from src.tracking import get_dvc_dataset_hash, get_git_commit_sha

logger = logging.getLogger("src.monitoring")

MONITORING_DISCLAIMER = (
    "MLOps statistical monitoring signals only. Not clinical or diagnostic evaluation. "
    "Drift indicates population variance between reference and serving distributions; "
    "it does not establish medical validity or clinical outcome degradation without ground-truth labels."
)


def calculate_ks(
    reference: np.ndarray | pd.Series, current: np.ndarray | pd.Series
) -> dict[str, float]:
    """Calculate Two-Sample Kolmogorov-Smirnov test between reference and current samples.

    Args:
        reference: Reference numerical distribution.
        current: Current numerical distribution.

    Returns:
        Dictionary with 'statistic' (KS distance) and 'p_value'.
    """
    ref_clean = pd.Series(reference).dropna().to_numpy()
    curr_clean = pd.Series(current).dropna().to_numpy()

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    res = stats.ks_2samp(ref_clean, curr_clean)
    return {
        "statistic": round(float(res.statistic), 4),
        "p_value": round(float(res.pvalue), 6),
    }


def calculate_wasserstein(
    reference: np.ndarray | pd.Series, current: np.ndarray | pd.Series
) -> float:
    """Calculate 1D Wasserstein Distance (Earth Mover's Distance) between distributions.

    Args:
        reference: Reference numerical distribution.
        current: Current numerical distribution.

    Returns:
        Wasserstein distance (normalized by reference standard deviation if non-zero).
    """
    ref_clean = pd.Series(reference).dropna().to_numpy()
    curr_clean = pd.Series(current).dropna().to_numpy()

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    raw_dist = float(stats.wasserstein_distance(ref_clean, curr_clean))
    ref_std = float(np.std(ref_clean))
    norm_dist = raw_dist / ref_std if ref_std > 1e-6 else raw_dist
    return round(norm_dist, 4)


def calculate_psi(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    is_categorical: bool = True,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current distributions.

    PSI Formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))

    Args:
        reference: Reference baseline series.
        current: Current serving series.
        is_categorical: Whether data is categorical.
        num_buckets: Quantile bucket count for numerical features.
        epsilon: Small smoothing constant to avoid log(0) or division by zero.

    Returns:
        Population Stability Index (float >= 0.0).
    """
    ref_s = pd.Series(reference).dropna()
    curr_s = pd.Series(current).dropna()

    if len(ref_s) == 0 or len(curr_s) == 0:
        return 0.0

    if is_categorical:
        # Categorical frequencies
        categories = sorted(set(ref_s.unique()).union(set(curr_s.unique())))
        ref_counts = ref_s.value_counts(normalize=True)
        curr_counts = curr_s.value_counts(normalize=True)

        psi_val = 0.0
        for cat in categories:
            e = ref_counts.get(cat, 0.0) + epsilon
            a = curr_counts.get(cat, 0.0) + epsilon
            psi_val += (a - e) * math.log(a / e)
        return round(float(psi_val), 4)

    # Numerical binning by quantiles of reference
    try:
        quantiles = np.linspace(0, 1, num_buckets + 1)
        bins = np.percentile(ref_s, quantiles * 100)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_binned = pd.cut(ref_s, bins=bins)
        curr_binned = pd.cut(curr_s, bins=bins)

        ref_dist = ref_binned.value_counts(normalize=True).sort_index()
        curr_dist = curr_binned.value_counts(normalize=True).sort_index()

        psi_val = 0.0
        for b in ref_dist.index:
            e = ref_dist.get(b, 0.0) + epsilon
            a = curr_dist.get(b, 0.0) + epsilon
            psi_val += (a - e) * math.log(a / e)
        return round(float(psi_val), 4)
    except Exception as exc:
        logger.warning(f"Failed to compute numerical PSI, falling back to 0.0: {exc}")
        return 0.0


def calculate_categorical_distribution(s: pd.Series) -> dict[str, float]:
    """Return category percentages rounded to 4 decimals."""
    clean_s = s.dropna()
    if len(clean_s) == 0:
        return {}
    vc = clean_s.value_counts(normalize=True)
    return {str(k): round(float(v), 4) for k, v in vc.items()}


def calculate_numerical_summary(s: pd.Series) -> dict[str, float]:
    """Return descriptive summary statistics for a numerical series."""
    clean_s = s.dropna()
    if len(clean_s) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": round(float(clean_s.mean()), 2),
        "std": round(float(clean_s.std()), 2) if len(clean_s) > 1 else 0.0,
        "min": round(float(clean_s.min()), 2),
        "max": round(float(clean_s.max()), 2),
        "median": round(float(clean_s.median()), 2),
    }


class ModelMonitor:
    """Production-grade model monitoring service for MedMLOps-X."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        reference_df: pd.DataFrame | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or load_config()
        self.mon_cfg = self.config.get("monitoring", {})
        self.val_cfg = self.config.get("validation", {})
        self.model = model

        # Load reference baseline
        if reference_df is not None:
            self.reference_df = reference_df.copy()
        else:
            ref_path_str = self.mon_cfg.get("reference_dataset", "Data/drug.csv")
            ref_path = Path(ref_path_str)
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"Reference dataset not found at: {ref_path.resolve()}"
                )
            self.reference_df = pd.read_csv(ref_path)

        # Base directory for reports
        reports_dir_str = self.mon_cfg.get("reports_dir", "monitoring/reports")
        self.reports_dir = Path(reports_dir_str)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Precompute reference profiles
        self._build_reference_profiles()

    def _build_reference_profiles(self) -> None:
        """Precompute distribution summaries for reference data."""
        self.ref_numerical = {}
        for col in ["Age", "Na_to_K"]:
            if col in self.reference_df.columns:
                self.ref_numerical[col] = calculate_numerical_summary(
                    self.reference_df[col]
                )

        self.ref_categorical = {}
        for col in ["Sex", "BP", "Cholesterol"]:
            if col in self.reference_df.columns:
                self.ref_categorical[col] = calculate_categorical_distribution(
                    self.reference_df[col]
                )

        self.ref_predictions = {}
        if "Drug" in self.reference_df.columns:
            self.ref_predictions["Drug"] = calculate_categorical_distribution(
                self.reference_df["Drug"]
            )

    def _ensure_model_loaded(self) -> Any:
        """Ensure champion model pipeline is loaded for prediction drift checks."""
        if self.model is not None:
            return self.model
        try:
            import skops.io as sio

            model_path = Path(
                self.config.get("champion", {}).get(
                    "artifact_path", "Model/drug_pipeline.skops"
                )
            )
            if model_path.exists():
                trusted = get_trusted_types(self.config)
                self.model = sio.load(model_path, trusted=trusted)
        except Exception as exc:
            logger.warning(f"Could not load model artifact for prediction drift: {exc}")
            self.model = None
        return self.model

    def validate_schema(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Verify schema compliance of incoming dataset."""
        expected_cols = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
        current_cols = list(current_df.columns)

        missing_cols = [col for col in expected_cols if col not in current_cols]
        unexpected_cols = [
            col
            for col in current_cols
            if col not in expected_cols and col not in ["Drug", "prediction"]
        ]

        # Datatype checks
        type_mismatches = []
        if "Age" in current_df.columns and not pd.api.types.is_numeric_dtype(
            current_df["Age"]
        ):
            type_mismatches.append(
                {
                    "column": "Age",
                    "expected": "numeric",
                    "actual": str(current_df["Age"].dtype),
                }
            )
        if "Na_to_K" in current_df.columns and not pd.api.types.is_numeric_dtype(
            current_df["Na_to_K"]
        ):
            type_mismatches.append(
                {
                    "column": "Na_to_K",
                    "expected": "numeric",
                    "actual": str(current_df["Na_to_K"].dtype),
                }
            )

        status = "PASS"
        if missing_cols or type_mismatches:
            status = "ERROR"
        elif unexpected_cols:
            status = "WARNING"

        return {
            "status": status,
            "expected_columns": expected_cols,
            "missing_columns": missing_cols,
            "unexpected_columns": unexpected_cols,
            "type_mismatches": type_mismatches,
        }

    def check_data_quality(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Check missingness, range violations, and unseen categories."""
        total_rows = len(current_df)
        if total_rows == 0:
            return {
                "status": "ERROR",
                "missing_values": {},
                "out_of_range": {},
                "unseen_categories": {},
            }

        # 1. Missingness
        null_counts = current_df.isnull().sum().to_dict()
        null_rates = {
            col: round(float(count / total_rows), 4)
            for col, count in null_counts.items()
        }

        # 2. Range violations
        out_of_range = {}
        if "Age" in current_df.columns and pd.api.types.is_numeric_dtype(
            current_df["Age"]
        ):
            age_min = self.val_cfg.get("age_min", 15)
            age_max = self.val_cfg.get("age_max", 74)
            violating_age = (
                (current_df["Age"] < age_min) | (current_df["Age"] > age_max)
            ).sum()
            out_of_range["Age"] = {
                "violating_count": int(violating_age),
                "violating_rate": round(float(violating_age / total_rows), 4),
                "expected_range": [age_min, age_max],
            }

        if "Na_to_K" in current_df.columns and pd.api.types.is_numeric_dtype(
            current_df["Na_to_K"]
        ):
            nak_min = self.val_cfg.get("na_to_k_min", 6.0)
            nak_max = self.val_cfg.get("na_to_k_max", 39.0)
            violating_nak = (
                (current_df["Na_to_K"] < nak_min) | (current_df["Na_to_K"] > nak_max)
            ).sum()
            out_of_range["Na_to_K"] = {
                "violating_count": int(violating_nak),
                "violating_rate": round(float(violating_nak / total_rows), 4),
                "expected_range": [nak_min, nak_max],
            }

        # 3. Unseen categories
        unseen_cats = {}
        for col in ["Sex", "BP", "Cholesterol"]:
            if col in current_df.columns:
                valid_cats = set(
                    self.val_cfg.get(
                        f"{col.lower()}_categories",
                        self.reference_df[col].unique().tolist(),
                    )
                )
                curr_cats = set(current_df[col].dropna().unique())
                diff = list(curr_cats - valid_cats)
                if diff:
                    unseen_cats[col] = diff

        # Quality status
        quality_status = "PASS"
        has_nulls = any(rate > 0 for rate in null_rates.values())
        has_range_violation = any(
            v["violating_count"] > 0 for v in out_of_range.values()
        )

        if unseen_cats:
            quality_status = "DRIFT"
        elif has_nulls or has_range_violation:
            quality_status = "WARNING"

        return {
            "status": quality_status,
            "null_rates": null_rates,
            "out_of_range": out_of_range,
            "unseen_categories": unseen_cats,
        }

    def compute_feature_drift(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Compute numerical and categorical drift for each feature against reference."""
        thresholds = self.mon_cfg.get("thresholds", {})
        num_thresh = thresholds.get("numerical", {})
        cat_thresh = thresholds.get("categorical", {})

        p_val_drift = num_thresh.get("p_value_drift", 0.01)
        p_val_warn = num_thresh.get("p_value_warning", 0.05)
        w_warn = num_thresh.get("wasserstein_warning", 0.15)
        psi_drift_thresh = cat_thresh.get("psi_drift", 0.20)
        psi_warn_thresh = cat_thresh.get("psi_warning", 0.10)

        feature_reports = {}

        # 1. Numerical Features (Age, Na_to_K)
        for col in ["Age", "Na_to_K"]:
            if col in current_df.columns and col in self.reference_df.columns:
                ref_series = self.reference_df[col]
                curr_series = current_df[col]

                ks_res = calculate_ks(ref_series, curr_series)
                w_dist = calculate_wasserstein(ref_series, curr_series)
                num_psi = calculate_psi(ref_series, curr_series, is_categorical=False)

                p_val = ks_res["p_value"]
                status = "PASS"
                severity = "LOW"

                if p_val < p_val_drift or w_dist >= w_warn * 1.5:
                    status = "DRIFT"
                    severity = "HIGH"
                elif p_val < p_val_warn or w_dist >= w_warn:
                    status = "WARNING"
                    severity = "MEDIUM"

                feature_reports[col] = {
                    "type": "numerical",
                    "status": status,
                    "severity": severity,
                    "drift_detected": status == "DRIFT",
                    "ks_statistic": ks_res["statistic"],
                    "ks_p_value": ks_res["p_value"],
                    "wasserstein_distance": w_dist,
                    "psi": num_psi,
                    "reference_summary": self.ref_numerical.get(col, {}),
                    "current_summary": calculate_numerical_summary(curr_series),
                }

        # 2. Categorical Features (Sex, BP, Cholesterol)
        for col in ["Sex", "BP", "Cholesterol"]:
            if col in current_df.columns and col in self.reference_df.columns:
                ref_series = self.reference_df[col]
                curr_series = current_df[col]

                psi_val = calculate_psi(ref_series, curr_series, is_categorical=True)

                status = "PASS"
                severity = "LOW"
                if psi_val >= psi_drift_thresh:
                    status = "DRIFT"
                    severity = "HIGH"
                elif psi_val >= psi_warn_thresh:
                    status = "WARNING"
                    severity = "MEDIUM"

                feature_reports[col] = {
                    "type": "categorical",
                    "status": status,
                    "severity": severity,
                    "drift_detected": status == "DRIFT",
                    "psi": psi_val,
                    "reference_distribution": self.ref_categorical.get(col, {}),
                    "current_distribution": calculate_categorical_distribution(
                        curr_series
                    ),
                }

        return feature_reports

    def compute_prediction_drift(
        self, current_df: pd.DataFrame
    ) -> dict[str, Any] | None:
        """Evaluate prediction output drift if model is available."""
        model = self._ensure_model_loaded()
        if model is None or len(current_df) == 0:
            return None

        # Build feature matrix in correct order
        features = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
        if not all(col in current_df.columns for col in features):
            return None

        clean_current = current_df[features].dropna()
        if len(clean_current) == 0:
            return None

        try:
            preds = model.predict(clean_current.values)
            pred_series = pd.Series([str(p) for p in preds])

            confidences = []
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(clean_current.values)
                confidences = [float(np.max(row)) for row in probs]

            # Reference predictions (using target Drug from reference dataset)
            ref_drug_series = self.reference_df["Drug"]
            psi_val = calculate_psi(ref_drug_series, pred_series, is_categorical=True)

            pred_thresh = self.mon_cfg.get("thresholds", {}).get("prediction", {})
            class_drift_th = pred_thresh.get("class_psi_drift", 0.20)
            class_warn_th = pred_thresh.get("class_psi_warning", 0.10)

            status = "PASS"
            severity = "LOW"
            if psi_val >= class_drift_th:
                status = "DRIFT"
                severity = "HIGH"
            elif psi_val >= class_warn_th:
                status = "WARNING"
                severity = "MEDIUM"

            pred_report = {
                "status": status,
                "severity": severity,
                "drift_detected": status == "DRIFT",
                "class_psi": psi_val,
                "reference_class_distribution": self.ref_predictions.get("Drug", {}),
                "predicted_class_distribution": calculate_categorical_distribution(
                    pred_series
                ),
            }

            if confidences:
                conf_series = pd.Series(confidences)
                pred_report["confidence_summary"] = calculate_numerical_summary(
                    conf_series
                )

            return pred_report
        except Exception as exc:
            logger.error(f"Prediction drift evaluation failed: {exc}")
            return None

    def compute_drift(
        self,
        current_df: pd.DataFrame,
        include_predictions: bool = True,
        save_report: bool = True,
    ) -> dict[str, Any]:
        """Execute end-to-end monitoring analysis and generate structured report.

        Args:
            current_df: DataFrame of current serving records.
            include_predictions: Whether to compute prediction output drift.
            save_report: Whether to persist report to disk.

        Returns:
            Dictionary matching the MedMLOps-X Drift Report Schema.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        report_id = f"drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # 1. Schema Validation
        schema_res = self.validate_schema(current_df)

        # 2. Data Quality Checks
        quality_res = self.check_data_quality(current_df)

        # 3. Feature Drift
        feature_drift = self.compute_feature_drift(current_df)

        # 4. Prediction Drift
        pred_drift = None
        if include_predictions:
            pred_drift = self.compute_prediction_drift(current_df)

        # 5. Overall Status & Severity Determination
        all_statuses = [schema_res["status"], quality_res["status"]]
        all_statuses.extend(f["status"] for f in feature_drift.values())
        if pred_drift:
            all_statuses.append(pred_drift["status"])

        if "ERROR" in all_statuses:
            overall_status = "ERROR"
            overall_severity = "HIGH"
        elif "DRIFT" in all_statuses:
            overall_status = "DRIFT"
            overall_severity = "HIGH"
        elif "WARNING" in all_statuses:
            overall_status = "WARNING"
            overall_severity = "MEDIUM"
        else:
            overall_status = "PASS"
            overall_severity = "LOW"

        drifted_features = [
            col for col, res in feature_drift.items() if res.get("drift_detected")
        ]

        # Lineage metadata
        git_sha = "unknown"
        dvc_hash = "unknown"
        try:
            git_sha = get_git_commit_sha()
            dvc_hash = get_dvc_dataset_hash()
        except Exception:
            pass

        report: dict[str, Any] = {
            "report_id": report_id,
            "timestamp": timestamp,
            "model_metadata": {
                "model_name": "xgboost",
                "model_version": "0.2.0",
                "git_commit": git_sha,
                "dvc_dataset_hash": dvc_hash,
            },
            "summary": {
                "overall_status": overall_status,
                "overall_severity": overall_severity,
                "feature_drift_detected": len(drifted_features) > 0,
                "prediction_drift_detected": bool(
                    pred_drift and pred_drift.get("drift_detected")
                ),
                "schema_status": schema_res["status"],
                "data_quality_status": quality_res["status"],
                "drifted_features": drifted_features,
                "drifted_features_count": len(drifted_features),
                "total_features_monitored": len(feature_drift),
                "reference_samples": len(self.reference_df),
                "current_samples": len(current_df),
            },
            "schema_validation": schema_res,
            "data_quality": quality_res,
            "features": feature_drift,
            "prediction_drift": pred_drift,
            "disclaimer": MONITORING_DISCLAIMER,
        }

        if save_report:
            self.save_report(report)

        return report

    def save_report(self, report: dict[str, Any]) -> Path:
        """Persist structured report to disk safely with path traversal protection.

        Args:
            report: Monitoring report dictionary.

        Returns:
            Resolved Path of the saved report.
        """
        report_id = report.get("report_id", f"drift_{uuid.uuid4().hex[:8]}")
        # Sanitize filename
        safe_filename = (
            f"{report_id}.json".replace("..", "").replace("/", "").replace("\\", "")
        )
        dest_path = (self.reports_dir / safe_filename).resolve()

        # Path traversal guard
        if not str(dest_path).startswith(str(self.reports_dir.resolve())):
            raise ValueError(f"Path traversal detected: {dest_path}")

        with open(dest_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

        return dest_path

    def get_latest_report(self) -> dict[str, Any] | None:
        """Retrieve the most recently generated drift report from disk."""
        if not self.reports_dir.exists():
            return None

        report_files = sorted(
            self.reports_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not report_files:
            return None

        try:
            with open(report_files[0], "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.error(f"Failed to read latest report {report_files[0]}: {exc}")
            return None


# Module-level singleton
_monitor_instance: ModelMonitor | None = None


def get_monitor() -> ModelMonitor:
    """Retrieve or initialize the global ModelMonitor singleton."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelMonitor()
    return _monitor_instance
