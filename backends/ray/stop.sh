#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/lib/spark-common.sh"
spark_load_config

CONTAINER="spark-ray"

echo "=== Stopping Ray Cluster ==="
echo ""

# Stop vLLM processes in head container first
spark_info "Stopping vLLM processes..."
docker exec "$CONTAINER" bash -c 'pkill -f "vllm serve" 2>/dev/null || true' 2>/dev/null || true

# Stop and remove containers
spark_info "Stopping Ray head on ${SPARK_PRIMARY_HOST}..."
docker stop "$CONTAINER" 2>/dev/null || true
docker rm -f "$CONTAINER" 2>/dev/null || true
echo "  Done."

spark_info "Stopping Ray worker on ${SPARK_SECONDARY_HOST}..."
spark_ssh "$SPARK_SECONDARY_HOST" "docker stop ${CONTAINER} 2>/dev/null || true; docker rm -f ${CONTAINER} 2>/dev/null || true" 2>/dev/null || true
echo "  Done."

echo ""
echo "=== Ray Cluster Stopped ==="
