#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/lib/spark-common.sh"
spark_load_config

PORT="$(spark_effective_port)"

case "$SPARK_ENGINE" in
    trtllm)
        spark_info "Starting TRT-LLM via Swarm (model: ${MODEL_NAME}, port: ${PORT}, tp: ${TP_SIZE})"
        exec "${SPARK_TOOLS_DIR}/trtllm/trtllm-serve.sh" \
            "$MODEL_NAME" "$PORT" "${TP_SIZE}" "${TRTLLM_GPU_MEM_FRACTION}"
        ;;
    vllm)
        spark_info "Starting vLLM standalone (model: ${MODEL_NAME}, port: ${PORT})"
        exec "${SPARK_TOOLS_DIR}/vllm/vllm-serve.sh" \
            "$MODEL_NAME" "$PORT" "${GPU_MEM_UTIL}" "${MAX_MODEL_LEN}"
        ;;
    *)
        spark_die "Unknown engine: $SPARK_ENGINE"
        ;;
esac
