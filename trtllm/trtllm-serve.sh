#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <huggingface-model-id> [port] [tp_size] [gpu_mem_fraction]" >&2
  echo "  gpu_mem_fraction defaults to 0.9 (NVIDIA recommended)" >&2
  exit 1
fi

MODEL="$1"
PORT="${2:-8355}"
TP_SIZE="${3:-2}"
GPU_MEM_FRACTION="${4:-0.9}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CID="$("${SCRIPT_DIR}/trtllm-container.sh")"

# Validate model is fully downloaded before attempting to serve
echo "Validating model '${MODEL}' is ready..."
MODEL_DIR="models--${MODEL//\//--}"

MODEL_STATUS=$(docker exec "${CID}" bash -lc "
  BASE=\"/root/.cache/huggingface/hub/${MODEL_DIR}\"
  
  if [[ ! -d \"\$BASE\" ]]; then
    echo \"NOT_FOUND\"
    exit 0
  fi
  
  INC_COUNT=\$(find \"\$BASE\" -name \"*.incomplete\" 2>/dev/null | wc -l || echo 0)
  BLOB_COUNT=\$(find \"\$BASE\" -type f 2>/dev/null | wc -l || echo 0)
  
  if [[ \"\$INC_COUNT\" -eq 0 && \"\$BLOB_COUNT\" -gt 0 ]]; then
    echo \"COMPLETE\"
  elif [[ \"\$INC_COUNT\" -gt 0 ]]; then
    echo \"IN_PROGRESS\"
  else
    echo \"UNKNOWN\"
  fi
" || echo "ERROR")

if [[ "${MODEL_STATUS}" != "COMPLETE" ]]; then
  echo "ERROR: Model '${MODEL}' is not ready (status: ${MODEL_STATUS})" >&2
  echo >&2
  if [[ "${MODEL_STATUS}" == "NOT_FOUND" ]]; then
    echo "Model not found in cache. Download it first with:" >&2
    echo "  ${SCRIPT_DIR}/trtllm-download.sh download ${MODEL}" >&2
  elif [[ "${MODEL_STATUS}" == "IN_PROGRESS" ]]; then
    echo "Model download is still in progress. Check status with:" >&2
    echo "  ${SCRIPT_DIR}/trtllm-model-status.sh ${MODEL}" >&2
  else
    echo "Model status is unclear. Check with:" >&2
    echo "  ${SCRIPT_DIR}/trtllm-model-status.sh ${MODEL}" >&2
  fi
  exit 1
fi

echo "Model validated. Proceeding to serve..."
echo

# Recreate the /tmp config file inside the container (in case it was lost)
docker exec "${CID}" bash -lc "
  cat <<CFG > /tmp/extra-llm-api-config.yml
print_iter_log: false
kv_cache_config:
  dtype: \"auto\"
  free_gpu_memory_fraction: ${GPU_MEM_FRACTION}
cuda_graph_config:
  enable_padding: true
CFG
"

echo "Starting TRT-LLM server for '${MODEL}' on port ${PORT} (tp_size=${TP_SIZE}, gpu_mem=${GPU_MEM_FRACTION}) in container ${CID}..."

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
