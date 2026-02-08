#!/bin/bash
# GLM-4.7-Flash server startup script for DGX Spark
# Serves GLM-4.7-Flash via OpenAI-compatible API
#
# Model options:
#   - zai-org/GLM-4.7-Flash (full BF16, ~62GB)
#   - QuantTrio/GLM-4.7-AWQ (AWQ 4-bit, ~16GB, recommended)
#
# Based on: https://huggingface.co/zai-org/GLM-4.7-Flash

set -e

# Default configuration
DEFAULT_MODEL="QuantTrio/GLM-4.7-AWQ"  # AWQ quant recommended for single Spark
DEFAULT_PORT=8000
DEFAULT_GPU_MEM_UTIL=0.9
DEFAULT_MAX_MODEL_LEN=65536
CONTAINER_IMAGE="nvcr.io/nvidia/vllm:25.12.post1-py3"
CONTAINER_NAME="vllm-glm47"

# Parse arguments
MODEL="${1:-$DEFAULT_MODEL}"
PORT="${2:-$DEFAULT_PORT}"
GPU_MEM_UTIL="${3:-$DEFAULT_GPU_MEM_UTIL}"
MAX_MODEL_LEN="${4:-$DEFAULT_MAX_MODEL_LEN}"

# Get local IP for remote access instructions
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "=================================================="
echo "Starting GLM-4.7-Flash Server"
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
if ! docker images | grep -q "nvidia/vllm.*25.12"; then
    echo "WARNING: Recommended vLLM image not found"
    echo "Pulling: $CONTAINER_IMAGE"
    echo "This may take a few minutes..."
    docker pull $CONTAINER_IMAGE
fi

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# Create HuggingFace cache directory if it doesn't exist
mkdir -p ~/.cache/huggingface

# Build vLLM serve command with GLM-4.7 specific options
VLLM_CMD="vllm serve $MODEL \
    --host 0.0.0.0 \
    --port $PORT \
    --dtype auto \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --trust-remote-code \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-4.7-flash"

# Add MTP speculative decoding for better performance (optional, can be enabled)
# Uncomment the following to enable MTP:
# VLLM_CMD="$VLLM_CMD --speculative-config.method mtp --speculative-config.num_speculative_tokens 1"

# Start the server
echo "Starting vLLM server..."
echo "First run will download model (~16GB for AWQ, ~62GB for full)"
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
    $VLLM_CMD

echo ""
echo "Container started: $CONTAINER_NAME"
echo ""
echo "Monitor logs:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "Wait for 'Application startup complete' before sending requests"
echo ""
echo "Stop server:"
echo "  docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
echo ""
echo "Test endpoint:"
echo "  curl http://localhost:$PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"glm-4.7-flash\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
echo ""
echo "=================================================="
echo ""
echo "NOTES:"
echo "  - Model alias: glm-4.7-flash (use in API calls)"
echo "  - Tool calling: Enabled with glm47 parser"
echo "  - Reasoning: deepseek_r1 parser for thinking tags"
echo ""
echo "To use the full BF16 model instead:"
echo "  $0 zai-org/GLM-4.7-Flash"
echo "=================================================="
