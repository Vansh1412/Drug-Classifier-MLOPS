"""
src/explainability.py — Production SHAP Explainability Service for MedMLOps-X.

Provides model-agnostic and TreeExplainer-based local feature attributions for
the champion pipeline, translating transformed feature SHAP values back to
human-interpretable domain features (Age, Sex, BP, Cholesterol, Na_to_K).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Medical safety disclaimer constant
MEDICAL_EXPLANATION_DISCLAIMER = (
    "This explanation represents statistical feature attribution for model transparency. "
    "It reflects machine-learning model decision weights and MUST NOT be interpreted "
    "as clinical diagnosis, biological etiology, or medical treatment advice."
)


class ModelExplainerService:
    """Service providing SHAP explanations for the MedMLOps-X champion model pipeline."""

    def __init__(self, pipeline: Pipeline, model_name: str = "xgboost") -> None:
        """Initialize the explainability service with a fitted pipeline.

        Args:
            pipeline: Fitted scikit-learn Pipeline containing 'preprocessing' and 'model' steps.
            model_name: Identifier for the model.
        """
        self.pipeline = pipeline
        self.model_name = model_name

        if not hasattr(pipeline, "named_steps"):
            raise ValueError("Provided pipeline must have 'named_steps' attribute.")

        self.preprocessor = pipeline.named_steps.get("preprocessing")
        self.model_step = pipeline.named_steps.get("model")

        if self.preprocessor is None or self.model_step is None:
            raise ValueError("Pipeline must contain 'preprocessing' and 'model' steps.")

        # Resolve underlying estimator (handling XGBStringClassifier or raw estimators)
        if hasattr(self.model_step, "estimator"):
            self.estimator = self.model_step.estimator
        else:
            self.estimator = self.model_step

        # Resolve class labels
        if (
            hasattr(self.model_step, "classes_")
            and self.model_step.classes_ is not None
        ):
            self.classes = list(self.model_step.classes_)
        elif (
            hasattr(self.estimator, "classes_") and self.estimator.classes_ is not None
        ):
            self.classes = list(self.estimator.classes_)
        else:
            self.classes = ["DrugY", "drugA", "drugB", "drugC", "drugX"]

        # Initialize and cache SHAP TreeExplainer
        self._init_explainer()

    def _init_explainer(self) -> None:
        """Initialize the SHAP TreeExplainer on the core estimator."""
        try:
            self.explainer = shap.TreeExplainer(self.estimator)
            logger.info("SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            logger.warning(
                f"TreeExplainer initialization failed: {e}. Falling back to generic/None."
            )
            self.explainer = None

    def _map_feature_contributions(
        self,
        shap_values_sample: np.ndarray,
        target_class_idx: int,
        feature_values: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Map raw transformed SHAP values back to human-readable domain features.

        ColumnTransformer structure:
          0: encoder__x1 (Sex)
          1: encoder__x2 (BP)
          2: encoder__x3 (Cholesterol)
          3: num_imputer__x0 (Age)
          4: num_imputer__x4 (Na_to_K)
          5: num_scaler__x0 (Age scaled)
          6: num_scaler__x4 (Na_to_K scaled)

        Args:
            shap_values_sample: Array of shape (n_transformed_features, n_classes) or (n_features,)
            target_class_idx: Index of the target class being explained.
            feature_values: Original dictionary of input features.

        Returns:
            Sorted list of feature contribution dictionaries.
        """
        # Map original feature name -> indices in preprocessor output
        feature_index_map = {
            "Age": [3, 5],
            "Sex": [0],
            "BP": [1],
            "Cholesterol": [2],
            "Na_to_K": [4, 6],
        }

        contributions: list[dict[str, Any]] = []

        is_multiclass = shap_values_sample.ndim == 2 and shap_values_sample.shape[1] > 1

        for feat_name, col_indices in feature_index_map.items():
            if is_multiclass:
                total_contrib = sum(
                    float(shap_values_sample[idx, target_class_idx])
                    for idx in col_indices
                    if idx < shap_values_sample.shape[0]
                )
            else:
                total_contrib = sum(
                    float(shap_values_sample[idx])
                    for idx in col_indices
                    if idx < len(shap_values_sample)
                )

            contributions.append(
                {
                    "feature": feat_name,
                    "value": feature_values.get(feat_name),
                    "contribution": round(total_contrib, 4),
                }
            )

        # Sort descending by absolute contribution impact
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions

    def explain(
        self,
        features: list[list[Any]],
        feature_names: list[str] | None = None,
        target_class: str | None = None,
    ) -> dict[str, Any]:
        """Generate prediction, confidence, class probabilities, and SHAP explanation.

        Args:
            features: 2D list of sample features [[age, sex, bp, cholesterol, na_to_k]].
            feature_names: Names corresponding to columns in features.
            target_class: Optional specific class to explain. Defaults to predicted class.

        Returns:
            Structured dictionary with prediction, confidence, probabilities,
            feature contributions, model metadata, and medical disclaimer.
        """
        if feature_names is None:
            feature_names = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]

        sample_raw = features[0]
        input_dict = dict(zip(feature_names, sample_raw))

        # 1. Prediction
        predicted_drug = str(self.pipeline.predict(features)[0])

        # 2. Probabilities & Confidence
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(features)[0]
            confidence = round(float(np.max(probs)), 4)
            probabilities = {
                cls_name: round(float(prob), 4)
                for cls_name, prob in zip(self.classes, probs)
            }
        else:
            confidence = 1.0
            probabilities = {predicted_drug: 1.0}

        # 3. Determine target class index for SHAP explanation
        explained_class = target_class if target_class else predicted_drug
        if explained_class in self.classes:
            class_idx = self.classes.index(explained_class)
        else:
            class_idx = 0

        # 4. Compute SHAP Values
        explanation_list: list[dict[str, Any]] = []
        if self.explainer is not None:
            transformed = self.preprocessor.transform(features)
            shap_output = self.explainer(transformed)
            sample_shap = shap_output.values[0]
            explanation_list = self._map_feature_contributions(
                sample_shap, class_idx, input_dict
            )

        return {
            "prediction": predicted_drug,
            "confidence": confidence,
            "probabilities": probabilities,
            "explained_class": explained_class,
            "explanation": explanation_list,
            "explanation_method": "SHAP",
            "model_name": self.model_name,
            "disclaimer": MEDICAL_EXPLANATION_DISCLAIMER,
        }
