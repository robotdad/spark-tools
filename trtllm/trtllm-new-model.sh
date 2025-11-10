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

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL [PORT] [TP_SIZE]" >&2
  echo "  MODEL   - Hugging Face model id, e.g. nvidia/Qwen3-14B-FP4" >&2
  echo "  PORT    - (optional) serving port, default 8355" >&2
  echo "  TP_SIZE - (optional) tensor parallel size, default 2" >&2
  exit 1
fi

MODEL="$1"
PORT="${2:-8355}"
TP_SIZE="${3:-2}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set in your environment." >&2
  echo "Export it first, e.g.:" >&2
  echo "  export HF_TOKEN=hf_..." >&2
  exit 1
fi

echo "=== New TRT-LLM model setup ==="
echo "Model:   $MODEL"
echo "Port:    $PORT"
echo "TP size: $TP_SIZE"
echo

echo "Step 1: Kick off cluster download for '$MODEL' on ${PRIMARY_HOST} and ${SECONDARY_HOST}..."
"${SCRIPT_DIR}/cluster-download.sh" "$MODEL"
echo

echo "Cluster download has been started in the background inside the TRT-LLM containers."
echo
echo "Next steps:"
echo "  1) Check download status on both nodes:"
echo "       ${SCRIPT_DIR}/trtllm-model-status.sh \"$MODEL\""
echo
echo "  2) Once both nodes show:  Incomplete files: 0"
echo "     start the server with:"
echo "       ${SCRIPT_DIR}/trtllm-serve.sh \"$MODEL\" $PORT $TP_SIZE"
echo
echo "Tip: you can re-run trtllm-model-status.sh at any time to see log tails and cache state."
