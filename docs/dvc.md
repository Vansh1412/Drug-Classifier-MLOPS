# DVC Data & Pipeline Versioning Guide

## Overview

**MedMLOps-X** integrates **Data Version Control (DVC)** to manage dataset versioning, pipeline stage reproducibility, and artifact lineage. DVC decouples large dataset files from Git version control while maintaining deterministic pointers in the repository.

---

## Key Concepts & Responsibilities

| Responsibility | Tool | Artifacts |
|---|---|---|
| Code Versioning | Git | Commit SHA (`git rev-parse --short HEAD`) |
| Dataset Versioning | DVC | `Data/drug.csv.dvc` (`7a10a9eb01e3d5664125ea1dd179c8c9`) |
| Pipeline Reproducibility | DVC | `dvc.yaml`, `dvc.lock` |
| Experiment Tracking & Lineage | MLflow | `sqlite:///mlruns.db` (candidate & champion runs) |

---

## Dataset Versioning

The canonical dataset `Data/drug.csv` (200 records, 6 features) is managed by DVC:

1. **Untracked in Git**: `Data/.gitignore` ignores `/drug.csv`.
2. **Tracked via Metadata**: `Data/drug.csv.dvc` stores the md5 hash of the dataset:
   ```yaml
   outs:
   - md5: 7a10a9eb01e3d5664125ea1dd179c8c9
     size: 5971
     hash: md5
     path: drug.csv
   ```
3. **Reproducibility**: Any team member or CI runner with access to remote storage can synchronize data via:
   ```bash
   dvc pull
   ```
   Or push modified data versions using:
   ```bash
   dvc add Data/drug.csv
   dvc push
   git add Data/drug.csv.dvc
   git commit -m "chore: update dataset version"
   ```

---

## DVC Pipeline Architecture

The pipeline is defined declaratively in `dvc.yaml` and executes two decoupled stages:

```
+---------------------+
| Data/drug.csv.dvc   |
+----------+----------+
           |
           v
+---------------------+
| Stage: validate     | -> Pandera schema verification
+----------+----------+
           |
           v
+---------------------+
| Stage: train        | -> Multi-model competition, evaluation,
+----------+----------+    quality gate, MLflow logging, Skops save
           |
           +--> Model/drug_pipeline.skops
           +--> Results/metrics.json
           +--> Results/competition_report.json
           +--> Results/model_results.png
```

### Stage Definitions (`dvc.yaml`)

```yaml
stages:
  validate:
    cmd: python -c "from src.config import load_config; from src.data_validation import load_and_validate; load_and_validate('Data/drug.csv', load_config())"
    deps:
      - Data/drug.csv
      - src/data_validation.py
      - src/config.py
      - configs/model_config.yaml

  train:
    cmd: python train.py
    deps:
      - Data/drug.csv
      - configs/model_config.yaml
      - src/config.py
      - src/data_validation.py
      - src/preprocessing.py
      - src/model_registry.py
      - src/evaluate.py
      - src/champion_selection.py
      - src/tracking.py
      - train.py
    outs:
      - Model/drug_pipeline.skops:
          cache: false
      - Model/model_metadata.json:
          cache: false
      - Results/metrics.json:
          cache: false
      - Results/competition_report.json:
          cache: false
      - Results/model_results.png:
          cache: false
```

> **Design Decision**: Outputs in `dvc.yaml` specify `cache: false` so that lightweight runtime artifacts (`Model/`, `Results/`) remain tracked in Git for continuous integration and immediate FastAPI serving without requiring a remote DVC pull in container environments.

---

## Reproducing the Pipeline

### Check Pipeline Status
```bash
dvc status
```
Output when up to date:
```text
Data and pipelines are up to date.
```

### Run Pipeline Reproduction
```bash
dvc repro
```
If dependencies have not changed, DVC skips completed stages. To force re-execution of all stages:
```bash
dvc repro -f
```

---

## Lineage Integration

When `train.py` executes, `src/tracking.py` extracts the active DVC dataset hash from `Data/drug.csv.dvc` via `get_dvc_dataset_hash()`. This hash is recorded directly as a tag on all MLflow runs:

```text
tags.dvc_dataset_hash: "7a10a9eb01e3d5664125ea1dd179c8c9"
tags.git_commit: "9ebf4c8"
```

This guarantees end-to-end provenance between code, data, and experimental runs.
