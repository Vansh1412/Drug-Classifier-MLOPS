# ADR-001: Model Serialization Strategy via Skops

## Status
Accepted

## Context
Machine learning models trained in Python with `scikit-learn` are traditionally serialized using standard `pickle` or `joblib.dump()`. However, standard Python pickle deserialization is fundamentally unsafe: an attacker who tampers with or crafts a serialized model file can achieve arbitrary remote code execution upon deserialization (`pickle.load`).

In production MLOps systems where models are transferred across networks, stored in artifact repositories, and loaded by API servers, secure serialization is a mandatory security baseline.

## Decision
MedMLOps-X uses the **`skops`** persistence format (`skops.io.dump` and `skops.io.load`) with an explicit whitelist of trusted types (`trusted_types`) for all model artifacts.

The trusted types list is centralized in `configs/model_config.yaml`:
- `sklearn.pipeline.Pipeline`
- `sklearn.preprocessing.OneHotEncoder`
- `sklearn.preprocessing.StandardScaler`
- `sklearn.compose.ColumnTransformer`
- `sklearn.preprocessing.OrdinalEncoder`
- `sklearn.impute.SimpleImputer`
- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.ensemble.RandomForestClassifier`
- `numpy.dtype`

## Alternatives Considered

1. **Standard `pickle` (`pickle.dump` / `pickle.load`):**
   - *Rejected:* High risk of arbitrary code execution. Zero type filtering mechanism.
2. **`joblib` (`joblib.dump` / `joblib.load`):**
   - *Rejected:* Inherits standard pickle security limitations under the hood.
3. **`ONNX` (`onnxruntime`):**
   - *Deferred:* Offers portability and inference speed, but introduces extra conversion complexity for scikit-learn custom column transformers and tabular string inputs.

## Consequences

### Positive
- Prevents arbitrary code execution upon model loading.
- Human-verifiable schema and trusted type checking.
- Native support for complex `scikit-learn` pipelines including encoders, imputers, scalers, and ensemble classifiers.

### Negative
- Any new transformer or estimator added to the pipeline in future sprints must be explicitly registered in `trusted_types`.
- Mild dependency on the `skops` library.
