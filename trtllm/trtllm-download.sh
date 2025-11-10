#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  $0 download <huggingface-model-id>"
  echo "  $0 status   <huggingface-model-id>"
  echo
  echo "Examples:"
  echo "  $0 download nvidia/Qwen3-235B-A22B-FP4"
  echo "  $0 status   nvidia/Qwen3-235B-A22B-FP4"
  exit 1
}

if [[ $# -lt 2 ]]; then
  usage
fi

COMMAND="$1"
MODEL="$2"

if [[ "${COMMAND}" != "download" && "${COMMAND}" != "status" ]]; then
  usage
fi

if [[ "${COMMAND}" == "download" && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set in your environment. Export HF_TOKEN before running download." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CID="$("${SCRIPT_DIR}/trtllm-container.sh")"

# Sanitize model name for log file (replace / with _)
SAFE_MODEL="${MODEL//\//_}"
LOG_PATH="/tmp/hf_download_${SAFE_MODEL}.log"

if [[ "${COMMAND}" == "download" ]]; then
  echo "Starting background download of '${MODEL}' in container ${CID}."
  echo "  Log file (inside container): ${LOG_PATH}"
  echo
  echo "This will keep running even if your SSH session disconnects."
  echo "You can check progress with:"
  echo "  ${0} status ${MODEL}"
  echo

  docker exec \
    -e MODEL="${MODEL}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -d "${CID}" bash -lc "
      set -euo pipefail
      echo \"[HF DL] starting download of \$MODEL\" > '${LOG_PATH}'
      nohup bash -lc 'huggingface-cli download \"\$MODEL\"' >> '${LOG_PATH}' 2>&1 &
    "

  echo "Background download started."
  exit 0
fi

if [[ "${COMMAND}" == "status" ]]; then
  echo "Showing last 40 lines of download log for '${MODEL}' from ${CID}:"
  echo "  Log file: ${LOG_PATH}"
  echo

  docker exec -i "${CID}" bash -lc "
    if [[ -f '${LOG_PATH}' ]]; then
      tail -n 40 '${LOG_PATH}'
    else
      echo 'No log file found at ${LOG_PATH}. Maybe download has not started yet?'
    fi
  "
fi
