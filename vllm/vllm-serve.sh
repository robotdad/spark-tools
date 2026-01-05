#!/bin/bash
# vLLM server startup script for DGX Spark
# Serves vision-language models via OpenAI-compatible API

set -e

# Default configuration
DEFAULT_MODEL="Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_PORT=8000
DEFAULT_GPU_MEM_UTIL=0.9
DEFAULT_MAX_MODEL_LEN=4096
CONTAINER_IMAGE="nvcr.io/nvidia/vllm:25.09-py3"
CONTAINER_NAME="vllm-server"

# Parse arguments
MODEL="${1:-$DEFAULT_MODEL}"
PORT="${2:-$DEFAULT_PORT}"
GPU_MEM_UTIL="${3:-$DEFAULT_GPU_MEM_UTIL}"
MAX_MODEL_LEN="${4:-$DEFAULT_MAX_MODEL_LEN}"

# Get local IP for remote access instructions
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "=================================================="
echo "Starting vLLM Server"
echo "=================================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU Memory: ${GPU_MEM_UTIL} utilization"
echo "Max Context: ${MAX_MODEL_LEN} tokens"
echo ""
echo "Local access:  http://localhost:$PORT"
echo "Remote access: http://$LOCAL_IP:$PORT"
echo "=================================================="
echo ""

# Check if Docker image exists
if ! docker images | grep -q "nvidia/vllm.*25.09-py3"; then
    echo "ERROR: vLLM Docker image not found"
    echo "Pull with: docker pull $CONTAINER_IMAGE"
    exit 1
fi

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# Create HuggingFace cache directory if it doesn't exist
mkdir -p ~/.cache/huggingface

# Start the server
echo "Starting vLLM server..."
echo "First run will download model (~14GB for Qwen2-VL-7B)"
echo ""

docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p $PORT:$PORT \
    $CONTAINER_IMAGE \
    vllm serve $MODEL \
        --host 0.0.0.0 \
        --port $PORT \
        --dtype auto \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --trust-remote-code

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
echo "Test endpoint:"
echo "  vllm-validate.sh $MODEL $PORT"
echo ""
echo "=================================================="
