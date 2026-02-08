#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/lib/spark-common.sh"
spark_load_config

PORT="$(spark_effective_port)"

echo "=== Spark Cluster Status (swarm + ${SPARK_ENGINE}) ==="
echo ""
spark_show_config
echo ""

# 1. Docker Swarm health
echo "--- Docker Swarm ---"
if docker info --format '{{.Swarm.LocalNodeState}}' 2>/dev/null | grep -q "active"; then
    echo "  Swarm: active"
    docker node ls --format "  {{.Hostname}}: {{.Status}} ({{.Availability}})" 2>/dev/null || echo "  Could not list nodes"
else
    echo "  Swarm: NOT active"
fi
echo ""

# 2. Engine-specific checks
case "$SPARK_ENGINE" in
    trtllm)
        echo "--- TRT-LLM Containers ---"
        CID="$("${SPARK_TOOLS_DIR}/trtllm/trtllm-container.sh" 2>/dev/null || true)"
        if [[ -n "$CID" ]]; then
            echo "  Local container: $CID (running)"
        else
            echo "  Local container: not found"
        fi

        echo ""
        echo "--- SSH to ${SPARK_SECONDARY_HOST} ---"
        if spark_ssh "$SPARK_SECONDARY_HOST" hostname &>/dev/null; then
            REMOTE_CID="$(spark_ssh "$SPARK_SECONDARY_HOST" "cd ~/spark-tools/trtllm && ./trtllm-container.sh 2>/dev/null || true")"
            if [[ -n "$REMOTE_CID" ]]; then
                echo "  Remote container: $REMOTE_CID (running)"
            else
                echo "  Remote container: not found"
            fi
        else
            echo "  Cannot reach ${SPARK_SECONDARY_HOST}"
        fi
        ;;
    vllm)
        echo "--- vLLM Containers ---"
        FOUND=false
        for name in vllm-server vllm-glm47; do
            if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
                echo "  $name: running"
                FOUND=true
            fi
        done
        if [[ "$FOUND" != true ]]; then
            echo "  No vLLM containers running"
        fi
        ;;
esac
echo ""

# 3. GPU status
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"
echo ""

# 4. API endpoint health
echo "--- API Endpoint (port ${PORT}) ---"
HEALTH_URL="http://localhost:${PORT}/health"
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$HEALTH_URL" 2>/dev/null || echo "000")
if [[ "$HEALTH_STATUS" == "200" ]]; then
    echo "  Health: OK (HTTP 200)"
    # Try to get model info
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
