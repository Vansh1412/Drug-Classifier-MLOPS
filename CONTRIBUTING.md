# Contributing to MedMLOps-X

Welcome to MedMLOps-X! This guide outlines local setup, coding standards, and the PR workflow for contributing to the repository.

---

## 1. Local Environment Setup

Ensure you have Python 3.10+ installed.

```bash
# Clone repository
git clone https://github.com/Vansh1412/Drug-Classifier-MLOPS.git
cd Drug-Classifier-MLOPS

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Git Branching & Commit Conventions

### Branch Naming:
```text
feat/<feature-name>       # e.g., feat/model-competition
fix/<bug-description>     # e.g., fix/gradio-dependency
docs/<documentation-name> # e.g., docs/adr-002
security/<fix-name>       # e.g., security/non-root-docker
```

### Commit Message Format:
Follow conventional commits with structured prefixes:
- `feat:` — New user-facing feature or model capability
- `fix:` — Bug fix
- `refactor:` — Code restructuring without functional change
- `test:` — Adding or updating tests
- `docs:` — Documentation improvements or ADRs
- `ci:` — GitHub Actions workflow changes
- `security:` — Security hardening or dependency patches
- `chore:` — Maintenance, dependency pinning, or formatting

---

## 3. Engineering Guidelines

1. **Centralize Configuration:** Avoid magic numbers. Use `configs/model_config.yaml` for thresholds, seeds, and hyperparameters.
2. **Deterministic ML:** Always pin random seeds (`random_state=125`) across data splits, preprocessing, and model initializers.
3. **Safe Serialization:** Use `skops` with explicit `trusted_types` whitelists. Never introduce raw `pickle.load`.
4. **Never Break Working Tests:** The baseline test suite in `tests/test_pipeline.py` must pass on every commit.
5. **Medical Safety:** Ensure user-facing outputs clearly state that the system is an educational ML demonstration, not clinical advice.

---

## 4. Pre-PR Verification

Run these commands before pushing your branch:

```bash
# 1. Run tests
pytest

# 2. Format code
black train.py api/ tests/

# 3. Test API endpoints locally
uvicorn api.main:app --port 8000
# In a separate terminal:
curl http://localhost:8000/health
```

All Pull Requests require completion of the Definition of Done checklist in the PR template.
