# ─────────────────────────────────────────────────────────────
# MedMLOps-X — Hardened Production Multi-Stage Dockerfile
# ─────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies for compiling any native wheel extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python production dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Hardened Runtime ─────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Security: Create dedicated non-root system user and group (UID/GID 10001)
RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --shell /bin/false --no-create-home appuser

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app

# Copy installed Python packages and entrypoint binaries from builder
COPY --from=builder /usr/local /usr/local

# Copy only required application source directories with non-root ownership
COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup Model/ ./Model/
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup Data/ ./Data/
COPY --chown=appuser:appgroup monitoring/ ./monitoring/

# Expose service port and switch to unprivileged user
EXPOSE 8000
USER appuser

# Health check to ensure container is live and responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start FastAPI production server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
