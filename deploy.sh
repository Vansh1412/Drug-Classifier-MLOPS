#!/bin/bash
# ─────────────────────────────────────────────────────────────
# deploy.sh — Linux Deployment Script for Drug Classifier API
# Usage: bash deploy.sh [IMAGE_NAME] [CONTAINER_NAME] [PORT]
# ─────────────────────────────────────────────────────────────

set -e  # Exit immediately on any error

# ── Config ────────────────────────────────────────────────────
IMAGE_NAME="${1:-drug-classifier}"
CONTAINER_NAME="${2:-drug-classifier-api}"
PORT="${3:-8000}"
HEALTH_URL="http://localhost:${PORT}/health"

echo "======================================"
echo "  Drug Classifier — Linux Deployment  "
echo "======================================"
echo "Image:     $IMAGE_NAME"
echo "Container: $CONTAINER_NAME"
echo "Port:      $PORT"
echo "--------------------------------------"

# ── Step 1: Pull latest code ──────────────────────────────────
echo "[1/5] Pulling latest code from git..."
git pull origin main
echo "  ✔ Code updated."

# ── Step 2: Build Docker image ────────────────────────────────
echo "[2/5] Building Docker image: $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" .
echo "  ✔ Docker image built successfully."

# ── Step 3: Stop & remove existing container ──────────────────
echo "[3/5] Stopping existing container (if running)..."
if docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
    docker stop "$CONTAINER_NAME"
    echo "  ✔ Stopped container: $CONTAINER_NAME"
fi
if docker ps -aq --filter "name=$CONTAINER_NAME" | grep -q .; then
    docker rm "$CONTAINER_NAME"
    echo "  ✔ Removed container: $CONTAINER_NAME"
fi

# ── Step 4: Run new container ─────────────────────────────────
echo "[4/5] Starting new container: $CONTAINER_NAME..."
docker run -d \
    --name "$CONTAINER_NAME" \
    -p "${PORT}:8000" \
    --restart unless-stopped \
    "$IMAGE_NAME"
echo "  ✔ Container started on port $PORT."

# ── Step 5: Health check ──────────────────────────────────────
echo "[5/5] Running health check..."
sleep 5  # Give the server a moment to start

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" || true)

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "  ✔ Health check PASSED (HTTP $HTTP_STATUS)"
    echo ""
    echo "======================================"
    echo "  ✅ Deployment Successful!"
    echo "  API running at: http://localhost:$PORT"
    echo "  Docs at:        http://localhost:$PORT/docs"
    echo "======================================"
else
    echo "  ✘ Health check FAILED (HTTP $HTTP_STATUS)"
    echo "  Printing container logs:"
    docker logs "$CONTAINER_NAME" --tail 30
    exit 1
fi
