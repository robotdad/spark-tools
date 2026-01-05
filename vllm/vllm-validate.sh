#!/bin/bash
# Validate vLLM server health on DGX Spark

CONTAINER_NAME="vllm-server"

# Parse arguments
MODEL="${1:-Qwen/Qwen2-VL-7B-Instruct}"
PORT="${2:-8000}"

echo "=================================================="
echo "vLLM Health Check"
echo "=================================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo ""

# Check if container is running
echo "1. Checking container status..."
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "   ✓ Container running: $CONTAINER_NAME"
else
    echo "   ✗ Container not running"
    echo ""
    echo "Start with: vllm-serve.sh"
    exit 1
fi

# Check GPU status
echo ""
echo "2. Checking GPU status..."
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# Check if server is responding
echo ""
echo "3. Checking health endpoint..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null || echo "000")

if [ "$HEALTH_STATUS" = "200" ]; then
    echo "   ✓ Health endpoint responding (HTTP 200)"
else
    echo "   ✗ Health endpoint not responding (HTTP $HEALTH_STATUS)"
    echo ""
    echo "Check logs: docker logs $CONTAINER_NAME"
    echo "Server may still be starting up (wait for 'Application startup complete')"
    exit 1
fi

# Check models endpoint
echo ""
echo "4. Testing /v1/models endpoint..."
MODELS_RESPONSE=$(curl -s http://localhost:$PORT/v1/models 2>/dev/null)

if echo "$MODELS_RESPONSE" | grep -q "\"id\""; then
    echo "   ✓ Models endpoint responding"
    echo "$MODELS_RESPONSE" | grep -o '"id":"[^"]*"' | head -1
else
    echo "   ✗ Models endpoint not responding correctly"
fi

# Test inference with a simple prompt
echo ""
echo "5. Testing inference endpoint..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say 'OK' if you are working.\"}],
        \"max_tokens\": 10
    }" 2>/dev/null)

if echo "$TEST_RESPONSE" | grep -q "\"content\""; then
    CONTENT=$(echo "$TEST_RESPONSE" | grep -o '"content":"[^"]*"' | head -1 | cut -d'"' -f4)
    echo "   ✓ Inference working"
    echo "   Response: $CONTENT"
else
    echo "   ✗ Inference test failed"
    echo "   Response: $TEST_RESPONSE"
fi

echo ""
echo "=================================================="
echo "✓ Validation complete"
echo "=================================================="
