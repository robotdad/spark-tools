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
  echo "Usage: $0 <huggingface-model-id>" >&2
  exit 1
fi

MODEL="$1"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set in your environment. Export HF_TOKEN before running this." >&2
  exit 1
fi

for NODE in "${PRIMARY_HOST}" "${SECONDARY_HOST}"; do
  echo "=== Node: ${NODE} ==="

  if [[ "${NODE}" == "${PRIMARY_HOST}" ]]; then
    echo "Starting background download on local node (${NODE})..."
    "${SCRIPT_DIR}/trtllm-download.sh" download "${MODEL}"
  else
    echo "Starting background download on remote node (${NODE})..."
    ssh "${NODE}" "HF_TOKEN='${HF_TOKEN}' bash -lc 'cd \"\$HOME/spark-tools/trtllm\" && ./trtllm-download.sh download \"${MODEL}\"'"
  fi

  echo
done

echo "Cluster download kicked off for model: ${MODEL}"
echo "To check status for this model across the cluster, run on ${PRIMARY_HOST}:"
echo "  trtllm-model-status.sh ${MODEL}"
