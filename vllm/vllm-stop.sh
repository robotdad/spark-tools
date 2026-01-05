#!/bin/bash
# Stop vLLM server on DGX Spark

CONTAINER_NAME="vllm-server"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping vLLM container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    echo "✓ vLLM server stopped"
else
    echo "No running vLLM container found"
    
    # Check if container exists but is stopped
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Removing stopped container: $CONTAINER_NAME"
        docker rm $CONTAINER_NAME
    fi
fi
