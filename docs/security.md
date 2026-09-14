# Security Policy & Hardening

MedMLOps-X incorporates security controls across the ML, API, container, and CI/CD layers.

---

## 1. Secure Model Deserialization (`skops`)

Standard Python `pickle` and `joblib` serializers are inherently vulnerable to arbitrary code execution attacks. MedMLOps-X uses `skops`:
- Models are deserialized with an explicit whitelist of trusted types (`configs/model_config.yaml`).
- Any attempt to load untrusted classes triggers an immediate security exception.
- See `docs/adr/ADR-001-model-serialization.md` for architectural context.

---

## 2. API Contract & Input Validation

The FastAPI serving layer enforces strict input boundary checks via Pydantic:
- `age`: integer constrained to `[15, 74]`
- `sex`: literal enum `["M", "F"]`
- `blood_pressure`: literal enum `["HIGH", "LOW", "NORMAL"]`
- `cholesterol`: literal enum `["HIGH", "NORMAL"]`
- `na_to_k_ratio`: float constrained to `[6.2, 38.2]`

Requests outside these boundaries are rejected at the edge with HTTP 422 Unprocessable Entity.

---

## 3. CI/CD Deployment Security

- **Branch Protection:** Continuous Deployment (`.github/workflows/cd.yml`) is restricted strictly to successful CI completions originating from the `main` branch.
- **Pull Request Isolation:** Untrusted PRs and fork-originated workflows cannot trigger automated production deployments or access deployment credentials (`HF_TOKEN`).
- **Secret Management:** Deployment secrets (`HF`, `HF_SPACE`) are managed through GitHub Actions Secrets and never hardcoded in scripts or commits.

---

## 4. Medical Safety Boundary

⚠️ The system explicitly states in all user-facing documentation, API docs (`/docs`), and Gradio UI headers that it is an **educational machine-learning demonstration** and **not medical advice**.
