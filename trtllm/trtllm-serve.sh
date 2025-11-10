#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <huggingface-model-id> [port] [tp_size]" >&2
  exit 1
fi

MODEL="$1"
PORT="${2:-8355}"
TP_SIZE="${3:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CID="$("${SCRIPT_DIR}/trtllm-container.sh")"

# Recreate the /tmp config file inside the container (in case it was lost)
docker exec "${CID}" bash -lc '
  cat <<CFG > /tmp/extra-llm-api-config.yml
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.75
cuda_graph_config:
  enable_padding: true
CFG
'

echo "Starting TRT-LLM server for '${MODEL}' on port ${PORT} (tp_size=${TP_SIZE}) in container ${CID}..."

docker exec \
  -e MODEL="${MODEL}" \
  -it "${CID}" bash -lc "
    set -euo pipefail
    mpirun trtllm-llmapi-launch trtllm-serve \"${MODEL}\" \
      --tp_size ${TP_SIZE} \
      --backend pytorch \
      --max_num_tokens 32768 \
      --max_batch_size 4 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml \
      --port ${PORT}
  "
