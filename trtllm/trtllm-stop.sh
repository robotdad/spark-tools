#!/usr/bin/env bash
set -euo pipefail

# Host naming:
# - PRIMARY_HOST: this node (where you run the script), or override via SPARK_PRIMARY_HOST
# - SECONDARY_HOST: the other Spark node, override via SPARK_SECONDARY_HOST (defaults to 'dyad')
PRIMARY_HOST="${SPARK_PRIMARY_HOST:-$(hostname)}"
SECONDARY_HOST="${SPARK_SECONDARY_HOST:-dyad}"
echo "Primary host:   ${PRIMARY_HOST}"
echo "Secondary host: ${SECONDARY_HOST}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="kill"
STACK_NAME="trtllm-multinode"

if [[ $# -gt 0 ]]; then
  case "$1" in
    --kill-only) MODE="kill" ;;
    --stack)     MODE="stack" ;;
    --swarm)     MODE="swarm" ;;
    *)
      echo "Usage: $0 [--kill-only|--stack|--swarm]" >&2
      exit 1
      ;;
  esac
fi

echo "=== TRT-LLM stop ==="
echo "Mode: $MODE"
echo

stop_in_node () {
  local node="$1"
  local label="$2"

  if [[ "$node" == "$PRIMARY_HOST" ]]; then
    # Local node
    local CID
    CID="$("${SCRIPT_DIR}/trtllm-container.sh" 2>/dev/null || true)"
    if [[ -z "$CID" ]]; then
      echo "$label no container found"
      return
    fi
    echo "$label container: $CID"

    docker exec "$CID" bash -lc '
      if ! pgrep -f "trtllm-llmapi-launch|trtllm-serve|uvicorn" >/dev/null 2>&1; then
        echo "  no trtllm processes"
        exit 0
      fi
      echo "  killing trtllm-related processes..."
      pkill -f trtllm-llmapi-launch || true
      pkill -f trtllm-serve || true
      pkill -f uvicorn || true
    ' || echo "$label docker exec failed"
  else
    # Remote node
    ssh "$node" "cd \$HOME/spark-tools/trtllm && \
      CID=\$(./trtllm-container.sh 2>/dev/null || true); \
      if [ -z \"\$CID\" ]; then
        echo \"$label no container found\"
        exit 0
      fi
      echo \"$label container: \$CID\"
      docker exec \"\$CID\" bash -lc '
        if ! pgrep -f \"trtllm-llmapi-launch|trtllm-serve|uvicorn\" >/dev/null 2>&1; then
          echo \"  no trtllm processes\"
          exit 0
        fi
        echo \"  killing trtllm-related processes...\"
        pkill -f trtllm-llmapi-launch || true
        pkill -f trtllm-serve || true
        pkill -f uvicorn || true
      '
    " || echo "$label ssh/docker exec failed"
  fi
}

if [[ "$MODE" == "kill" || "$MODE" == "stack" || "$MODE" == "swarm" ]]; then
  echo "-- Killing TRT-LLM server processes in containers --"
  stop_in_node "$PRIMARY_HOST"   "[$PRIMARY_HOST]"
  stop_in_node "$SECONDARY_HOST" "[$SECONDARY_HOST]"
  echo
fi

if [[ "$MODE" == "stack" || "$MODE" == "swarm" ]]; then
  echo "-- Removing docker stack '$STACK_NAME' on ${PRIMARY_HOST} --"
  # We assume this script is run on PRIMARY_HOST
  if docker stack ls | awk 'NR>1 {print $1}' | grep -qx "$STACK_NAME"; then
    docker stack rm "$STACK_NAME"
    echo "  stack remove requested; containers will stop shortly."
  else
    echo "  stack '$STACK_NAME' not found."
  fi
  echo
fi

if [[ "$MODE" == "swarm" ]]; then
  echo "-- Leaving swarm on ${SECONDARY_HOST} (worker) and ${PRIMARY_HOST} (manager) --"
  ssh "$SECONDARY_HOST" 'docker swarm leave --force || echo "  secondary: swarm leave failed"' \
    || echo "  ssh to ${SECONDARY_HOST} failed"
  docker swarm leave --force || echo "  primary: swarm leave failed"
  echo
fi

echo "Done."