#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/lib/spark-common.sh"
spark_load_config

if [[ "$SPARK_ENGINE" != "vllm" ]]; then
    spark_die "Ray mode only supports vllm engine (got: $SPARK_ENGINE)"
fi

IMAGE="${SPARK_RAY_VLLM_IMAGE:-scitrera/dgx-spark-vllm:0.15.1-t5}"
CONTAINER="spark-ray"
PORT="${SPARK_PORT:-8000}"
QSFP="${SPARK_QSFP_IFACE:-enp1s0f0np0}"

# Detect primary IP on QSFP interface for Ray binding
PRIMARY_IP=$(ip -4 addr show "$QSFP" 2>/dev/null | grep -oP 'inet \K[\d.]+' || true)
if [[ -z "$PRIMARY_IP" ]]; then
    spark_warn "No IP on $QSFP, falling back to hostname IP"
    PRIMARY_IP=$(hostname -I | awk '{print $1}')
fi

# Environment vars for NCCL/UCX to use QSFP
RAY_ENV=(
    -e "UCX_NET_DEVICES=${QSFP}"
    -e "NCCL_SOCKET_IFNAME=${QSFP}"
    -e "OMPI_MCA_btl_tcp_if_include=${QSFP}"
    -e "GLOO_SOCKET_IFNAME=${QSFP}"
    -e "TP_SOCKET_IFNAME=${QSFP}"
    -e "RAY_memory_monitor_refresh_ms=0"
    -e "MASTER_ADDR=${PRIMARY_IP}"
)

start_ray_container() {
    local host="$1"
    local role="$2"  # head or worker
    local run_cmd

    # Docker run base
    run_cmd="docker run -d --name ${CONTAINER} \
        --gpus all --network host --shm-size 64g \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v \$HOME/.cache/huggingface:/root/.cache/huggingface"

    # Add env vars
    for env_flag in "${RAY_ENV[@]}"; do
        run_cmd+=" $env_flag"
    done

    # Add image and Ray command
    if [[ "$role" == "head" ]]; then
        run_cmd+=" ${IMAGE} ray start --block --head --port=6379"
    else
        run_cmd+=" ${IMAGE} ray start --block --address=${PRIMARY_IP}:6379"
    fi

    if [[ "$host" == "$SPARK_PRIMARY_HOST" || "$host" == "$(hostname)" ]]; then
        # Local
        docker rm -f "$CONTAINER" 2>/dev/null || true
        eval "$run_cmd"
    else
        # Remote
        spark_ssh "$host" "docker rm -f ${CONTAINER} 2>/dev/null || true; ${run_cmd}"
    fi
}

echo "=== Starting Ray Cluster ==="
echo "Image: $IMAGE"
echo "Primary: $SPARK_PRIMARY_HOST ($PRIMARY_IP)"
echo "Secondary: $SPARK_SECONDARY_HOST"
echo ""

# Step 1: Start Ray head
spark_info "Starting Ray head on ${SPARK_PRIMARY_HOST}..."
start_ray_container "$SPARK_PRIMARY_HOST" "head"
echo "  Container started."

# Step 2: Start Ray worker
spark_info "Starting Ray worker on ${SPARK_SECONDARY_HOST}..."
start_ray_container "$SPARK_SECONDARY_HOST" "worker"
echo "  Container started."

# Step 3: Wait for Ray cluster
spark_info "Waiting for Ray cluster to form..."
MAX_WAIT=60
WAITED=0
while [[ $WAITED -lt $MAX_WAIT ]]; do
    NODE_COUNT=$(docker exec "$CONTAINER" ray status 2>/dev/null | grep -c "node_" || echo "0")
    if [[ "$NODE_COUNT" -ge 2 ]]; then
        echo "  Ray cluster ready ($NODE_COUNT nodes)"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo "  Waiting... (${WAITED}s, nodes: ${NODE_COUNT})"
done

if [[ $WAITED -ge $MAX_WAIT ]]; then
    spark_warn "Ray cluster did not form within ${MAX_WAIT}s. Proceeding anyway..."
fi
echo ""

# Step 4: Build vLLM serve command
VLLM_CMD="vllm serve ${MODEL_NAME} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --dtype auto \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --distributed-executor-backend ray"

# Add served model name if set
if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then
    VLLM_CMD+=" --served-model-name ${SERVED_MODEL_NAME}"
fi

# Add extra args
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    VLLM_CMD+=" ${VLLM_EXTRA_ARGS}"
fi

spark_info "Starting vLLM in Ray head container..."
echo "  Command: $VLLM_CMD"
echo ""

# Run vLLM in background inside the container
docker exec -d "$CONTAINER" bash -c "$VLLM_CMD"

echo ""
echo "=== Ray + vLLM Starting ==="
echo ""
echo "Monitor logs:"
echo "  docker exec $CONTAINER bash -c 'tail -f /tmp/vllm*.log 2>/dev/null || echo No log yet'"
echo "  docker logs -f $CONTAINER"
echo ""
echo "Check status:"
echo "  spark-status"
echo ""
echo "API endpoint: http://localhost:${PORT}/v1"
