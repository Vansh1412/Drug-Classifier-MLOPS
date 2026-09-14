# Testing Strategy & Guidelines

## 1. Test Suite Architecture

Automated testing is powered by `pytest` and configured centrally via `pyproject.toml`.

The suite is located in `tests/test_pipeline.py` and validates both offline model integrity and online REST API contracts:

| Test Name | Target | Verification |
|---|---|---|
| `test_model_file_exists` | Disk artifact | Asserts `Model/drug_pipeline.skops` is present on disk |
| `test_model_loads` | Serialization | Verifies skops loads without corruption into an sklearn Pipeline |
| `test_prediction_output` | ML Inference | Validates single sample returns a valid drug class |
| `test_prediction_shape` | Batch Inference | Validates multiple inputs return correct dimensional output |
| `test_api_health_endpoint` | FastAPI Service | Asserts `GET /health` returns HTTP 200 and `model_loaded: true` |
| `test_api_predict_endpoint` | FastAPI Contract | Asserts `POST /predict` accepts patient schema and returns valid prediction |

## 2. Running Tests

```bash
# Run full suite
pytest

# Run with verbose output
pytest -v

# Run via Makefile
make test
```

## 3. Warning Filters & Quality Controls

Inconsistent version warnings from unpickling across minor scikit-learn releases are suppressed cleanly in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
filterwarnings = [
    "ignore::sklearn.exceptions.InconsistentVersionWarning",
]
```
