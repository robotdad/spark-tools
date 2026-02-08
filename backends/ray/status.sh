#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/lib/spark-common.sh"
spark_load_config

CONTAINER="spark-ray"
PORT="${SPARK_PORT:-8000}"

echo "=== Spark Cluster Status (ray + vllm) ==="
echo ""
spark_show_config
echo ""

# 1. Ray head container
echo "--- Ray Head (${SPARK_PRIMARY_HOST}) ---"
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "  Container: running"
    # Ray cluster status
    RAY_STATUS=$(docker exec "$CONTAINER" ray status 2>/dev/null || echo "  ray status failed")
    NODE_COUNT=$(echo "$RAY_STATUS" | grep -c "node_" || echo "0")
    echo "  Ray nodes: $NODE_COUNT"
else
    echo "  Container: NOT running"
fi
echo ""

# 2. Ray worker
echo "--- Ray Worker (${SPARK_SECONDARY_HOST}) ---"
if spark_ssh "$SPARK_SECONDARY_HOST" "docker ps --format '{{.Names}}' | grep -q '^${CONTAINER}$'" 2>/dev/null; then
    echo "  Container: running"
else
    echo "  Container: NOT running"
fi
echo ""

# 3. GPU
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"
echo ""

# 4. API endpoint
echo "--- API Endpoint (port ${PORT}) ---"
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
if [[ "$HEALTH_STATUS" == "200" ]]; then
    echo "  Health: OK (HTTP 200)"
    MODELS=$(curl -s --max-time 5 "http://localhost:${PORT}/v1/models" 2>/dev/null || true)
    if [[ -n "$MODELS" ]]; then
        MODEL_ID=$(echo "$MODELS" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
        [[ -n "$MODEL_ID" ]] && echo "  Serving: $MODEL_ID"
    fi
else
    echo "  Health: NOT responding (HTTP ${HEALTH_STATUS})"
fi
echo ""
echo "=== Status Complete ==="
