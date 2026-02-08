#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/lib/spark-common.sh"
spark_load_config

# Parse optional flags to pass through
STOP_FLAGS="${1:-}"

case "$SPARK_ENGINE" in
    trtllm)
        spark_info "Stopping TRT-LLM..."
        exec "${SPARK_TOOLS_DIR}/trtllm/trtllm-stop.sh" ${STOP_FLAGS:+"$STOP_FLAGS"}
        ;;
    vllm)
        spark_info "Stopping vLLM..."
        # Stop ALL known vllm containers (vllm-server AND vllm-glm47)
        for name in vllm-server vllm-glm47; do
            if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
                echo "  Stopping container: $name"
                docker stop "$name" 2>/dev/null || true
                docker rm "$name" 2>/dev/null || true
            fi
        done
        echo "Done."
        ;;
    *)
        spark_die "Unknown engine: $SPARK_ENGINE"
        ;;
esac
