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

if [[ $# < 1 ]]; then
  echo "Usage: $0 MODEL" >&2
  echo "  MODEL - Hugging Face model id, e.g. nvidia/Qwen3-14B-FP4" >&2
  exit 1
fi

MODEL="$1"
MODEL_DIR="models--${MODEL//\//--}"

check_node_local() {
  echo "============================================================"
  echo "Node: $PRIMARY_HOST (local)"

  local CID
  CID="$("${SCRIPT_DIR}/trtllm-container.sh")"
  echo "Container: $CID"

  docker exec -i "$CID" bash -lc '
    MODEL_DIR="'"$MODEL_DIR"'"
    BASE="/root/.cache/huggingface/hub/$MODEL_DIR"

    echo "Model cache path: $BASE"

    if [[ ! -d "$BASE" ]]; then
      echo "Status: NOT_STARTED (cache dir missing)"
      exit 0
    fi

    INC_COUNT=$(find "$BASE" -name "*.incomplete" | wc -l || echo 0)
    BLOB_COUNT=$(find "$BASE" -type f | wc -l || echo 0)
    SIZE=$(du -sh "$BASE" 2>/dev/null | awk "{print \$1}")

    if [[ "$INC_COUNT" -eq 0 && "$BLOB_COUNT" -gt 0 ]]; then
      STATUS="COMPLETE"
    elif [[ "$INC_COUNT" -gt 0 ]]; then
      STATUS="IN_PROGRESS"
    else
      STATUS="UNKNOWN"
    fi

    echo "Status: $STATUS"
    echo "  Total files:      $BLOB_COUNT"
    echo "  Incomplete files: $INC_COUNT"
    echo "  Approx size:      $SIZE"
  '

  echo
}

check_node_remote() {
  local NODE="$1"
  echo "============================================================"
  echo "Node: $NODE (via ssh)"

  ssh "$NODE" "cd \$HOME/spark-tools/trtllm && CID=\$(./trtllm-container.sh 2>/dev/null || echo 'UNKNOWN'); \
    echo \"Container: \$CID\"; \
    docker exec -i \"\$CID\" bash -lc '
      BASE=\"/root/.cache/huggingface/hub/$MODEL_DIR\"

      echo \"Model cache path: \$BASE\"

      if [[ ! -d \"\$BASE\" ]]; then
        echo \"Status: NOT_STARTED (cache dir missing)\"
        exit 0
      fi

      INC_COUNT=\$(find \"\$BASE\" -name \"*.incomplete\" | wc -l || echo 0)
      BLOB_COUNT=\$(find \"\$BASE\" -type f | wc -l || echo 0)
      SIZE=\$(du -sh \"\$BASE\" 2>/dev/null | awk \"{print \\$1}\")

      if [[ \"\$INC_COUNT\" -eq 0 && \"\$BLOB_COUNT\" -gt 0 ]]; then
        STATUS=\"COMPLETE\"
      elif [[ \"\$INC_COUNT\" -gt 0 ]]; then
        STATUS=\"IN_PROGRESS\"
      else
        STATUS=\"UNKNOWN\"
      fi

      echo \"Status: \$STATUS\"
      echo \"  Total files:      \$BLOB_COUNT\"
      echo \"  Incomplete files: \$INC_COUNT\"
      echo \"  Approx size:      \$SIZE\"
    '"

  echo
}

# primary (local)
check_node_local

# secondary (remote)
check_node_remote "$SECONDARY_HOST"
