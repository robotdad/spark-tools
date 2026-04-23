#!/bin/bash
# vLLM server startup script for DGX Spark
# Serves models via OpenAI-compatible API with full Blackwell optimizations.
#
# Loads the same config chain as the systemd service template:
#   model.env → node.env  (system-level, then user-level overrides)
#
# All VLLM_EXTRA_ARGS (FP8 KV cache, FlashInfer, prefix caching, etc.)
# are passed through to vllm serve. --enforce-eager is stripped automatically
# since this script runs standalone (TP=1) where CUDA graphs work fine.

set -euo pipefail

# ── Defaults (overridden by config files if present) ─────────────────────────
DEFAULT_MODEL="Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_PORT=8000
DEFAULT_GPU_MEM_UTIL=0.9
DEFAULT_MAX_MODEL_LEN=4096
CONTAINER_IMAGE="${SPARK_VLLM_IMAGE:-nvcr.io/nvidia/vllm:25.09-py3}"
CONTAINER_NAME="vllm-server"

# ── Load config chain (same order as systemd template) ───────────────────────
for f in \
    /etc/spark-tools/model.env \
    "${HOME}/.config/spark-tools/model.env" \
    /etc/spark-tools/node.env \
    "${HOME}/.config/spark-tools/node.env"; do
    # shellcheck disable=SC1090
    [ -f "$f" ] && source "$f"
done

# ── Resolve final values (config > positional args > defaults) ───────────────
MODEL="${MODEL_NAME:-${1:-$DEFAULT_MODEL}}"
PORT="${2:-${SPARK_PORT:-$DEFAULT_PORT}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-${3:-$DEFAULT_GPU_MEM_UTIL}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${4:-$DEFAULT_MAX_MODEL_LEN}}"

# ── Strip --enforce-eager (only needed for multi-node TP=2) ──────────────────
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_EXTRA_ARGS="$(echo "$VLLM_EXTRA_ARGS" | sed 's/--enforce-eager//g')"

# Get local IP for remote access instructions
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "=================================================="
echo "Starting vLLM Server"
echo "=================================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU Memory: ${GPU_MEM_UTIL} utilization"
echo "Max Context: ${MAX_MODEL_LEN} tokens"
if [ -n "$VLLM_EXTRA_ARGS" ]; then
    echo "Extra args: $VLLM_EXTRA_ARGS"
fi
echo ""
echo "Local access:  http://localhost:$PORT"
echo "Remote access: http://$LOCAL_IP:$PORT"
echo "=================================================="
echo ""

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Create HuggingFace cache directory if it doesn't exist
mkdir -p ~/.cache/huggingface

# ── Build volume mounts ──────────────────────────────────────────────────────
VOLUMES="-v $HOME/.cache/huggingface:/root/.cache/huggingface"
# Mount bundled tiktoken encodings if available (avoids runtime download)
if [ -d /etc/encodings ]; then
    VOLUMES="$VOLUMES -v /etc/encodings:/etc/encodings"
fi

# ── Build environment variables ──────────────────────────────────────────────
ENV_VARS=""
if [ -d /etc/encodings ]; then
    ENV_VARS="$ENV_VARS -e TIKTOKEN_ENCODINGS_BASE=/etc/encodings"
fi
ENV_VARS="$ENV_VARS -e VLLM_FLASHINFER_MOE_BACKEND=latency"

# Start the server
echo "Starting vLLM server..."
echo ""

# shellcheck disable=SC2086
docker run -d \
    --name "$CONTAINER_NAME" \
    --network host \
    --gpus all \
    --shm-size 10.24g \
    $VOLUMES \
    $ENV_VARS \
    "$CONTAINER_IMAGE" \
    vllm serve "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        $VLLM_EXTRA_ARGS

echo ""
echo "✓ Container started: $CONTAINER_NAME"
echo ""
echo "Monitor logs:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "Wait for 'Application startup complete' before sending requests"
echo ""
echo "Stop server:"
echo "  vllm-stop.sh"
echo ""
echo "=================================================="
