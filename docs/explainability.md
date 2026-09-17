# Explainability Guide — SHAP in MedMLOps-X

This document describes the explainability architecture, methodology, and interpretation guidelines for MedMLOps-X.

---

## 1. Why SHAP (SHapley Additive exPlanations)?

In healthcare and clinical machine learning demonstrations, black-box predictions carry inherent opacity risks. Even when models demonstrate high statistical accuracy, stakeholders must understand:
1. Which biometric attributes drove a specific recommendation.
2. How different feature values push model logits towards or away from candidate therapies.
3. Whether the decision boundaries align with established pharmacological intuition.

MedMLOps-X integrates **SHAP (SHapley Additive exPlanations)** based on cooperative game theory:
- **Local Accuracy / Additivity:** The sum of feature contributions equals the difference between the model's raw output and the expected baseline output.
- **Missingness:** Features with zero impact on the outcome receive a contribution of zero.
- **Consistency:** If a model changes so that the marginal contribution of a feature increases, that feature's attribution cannot decrease.

For tree-based ensembles (such as the champion XGBoost model), MedMLOps-X employs **TreeExplainer**, providing exact polynomial-time SHAP computation without sampling approximation error.

---

## 2. Architecture & Service Integration

Explainability operates on the **actual champion model** and the exact scikit-learn preprocessing pipeline:

```text
Patient Features (Age, Sex, BP, Cholesterol, Na_to_K)
                         ↓
            Pydantic Edge Validation
                         ↓
       ColumnTransformer Preprocessing Pipeline
   (OrdinalEncoder + SimpleImputer + StandardScaler)
                         ↓
         Transformed Feature Vector (1, 7)
                         ↓
           Champion XGBClassifier Model
       ┌─────────────────┴─────────────────┐
       ↓                                   ↓
Raw Margin & Probabilities         SHAP TreeExplainer
       ↓                                   ↓
Predicted Drug & Confidence       Raw SHAP Matrix (1, 7, 5)
       └─────────────────┬─────────────────┘
                         ↓
        Domain Feature Attribution Mapper
      (Maps transformed indices back to 5 biometrics)
                         ↓
      Structured Explanation Response (/explain)
```

### Explainer Caching
To maintain low latency in production serving, `ModelExplainerService` instantiates and caches the `shap.TreeExplainer` upon application startup. Subsequent calls to `/explain` evaluate in ~2–5 ms per sample.

---

## 3. Feature Attribution Mapping

The preprocessing pipeline produces 7 transformed columns from the 5 domain inputs:
- `encoder__x1` → `Sex` (Index 0)
- `encoder__x2` → `BP` (Index 1)
- `encoder__x3` → `Cholesterol` (Index 2)
- `num_imputer__x0` → `Age` (Index 3)
- `num_imputer__x4` → `Na_to_K` (Index 4)
- `num_scaler__x0` → `Age` (Index 5)
- `num_scaler__x4` → `Na_to_K` (Index 6)

`ModelExplainerService` maps these columns back to their human-interpretable parent features:
$$\text{Contrib}(\text{Age}) = \phi_3 + \phi_5$$
$$\text{Contrib}(\text{Na\_to\_K}) = \phi_4 + \phi_6$$
$$\text{Contrib}(\text{Sex}) = \phi_0, \quad \text{Contrib}(\text{BP}) = \phi_1, \quad \text{Contrib}(\text{Cholesterol}) = \phi_2$$

Attributions are evaluated with respect to the **predicted drug class** (or a requested alternative class) and sorted in descending order of absolute magnitude.

---

## 4. What SHAP Attributions Mean vs What They Do NOT Mean

| Property | What It Means | What It Does NOT Mean |
|---|---|---|
| **Feature Contribution** | Mathematical weight of a feature in moving the model's decision margin towards a specific label. | Medical or biological causation for disease or treatment efficacy. |
| **Positive Value ($>0$)** | The observed biometric pushed the model toward predicting this drug class. | Clinical proof that this drug is therapeutic or medically indicated. |
| **Negative Value ($<0$)** | The observed biometric pushed the model away from predicting this drug class. | Clinical proof that this drug is contraindicated. |
| **Zero Value ($=0$)** | The feature did not influence the decision boundary for this specific sample. | The feature is unimportant in general medicine. |

---

## 5. Medical Safety Boundary

> [!CAUTION]
> **CRITICAL MEDICAL DISCLAIMER**  
> SHAP values explain **algorithmic behavior**, not **biological etiology**.  
> MedMLOps-X is an educational engineering demonstration. Under no circumstances should feature attributions be used by patients or healthcare practitioners for clinical diagnosis, treatment decisions, or medical advice.
