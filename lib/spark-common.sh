#!/usr/bin/env bash
# spark-common.sh - Shared library for spark-tools
# Source this file; do not execute directly.

# Repo root (parent of lib/)
SPARK_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Config paths
SPARK_CONFIG_DIR="${SPARK_CONFIG_DIR:-$HOME/.config/spark-tools}"
SPARK_CLUSTER_ENV="${SPARK_CONFIG_DIR}/cluster.env"
SPARK_MODEL_ENV="${SPARK_CONFIG_DIR}/model.env"

# --- Color helpers ---
_red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
_green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
_yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
_bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

# --- Logging ---
spark_info()  { echo "==> $*"; }
spark_warn()  { _yellow "WARNING: $*" >&2; }
spark_error() { _red "ERROR: $*" >&2; }
spark_die()   { spark_error "$@"; exit 1; }

# --- Config Loading ---
spark_load_config() {
    # Load cluster config
    if [[ ! -f "$SPARK_CLUSTER_ENV" ]]; then
        spark_die "Cluster config not found: $SPARK_CLUSTER_ENV
Run 'spark-init' to create it."
    fi
    # shellcheck source=/dev/null
    source "$SPARK_CLUSTER_ENV"

    # Load model config (optional - some commands don't need it)
    if [[ -f "$SPARK_MODEL_ENV" ]]; then
        # shellcheck source=/dev/null
        source "$SPARK_MODEL_ENV"
    fi

    # Apply env var overrides (env vars beat config file)
    SPARK_MODE="${SPARK_MODE:-swarm}"
    SPARK_ENGINE="${SPARK_ENGINE:-trtllm}"
    SPARK_PRIMARY_HOST="${SPARK_PRIMARY_HOST:-$(hostname)}"
    SPARK_SECONDARY_HOST="${SPARK_SECONDARY_HOST:-dyad}"
    SPARK_PORT="${SPARK_PORT:-8000}"
    SPARK_QSFP_IFACE="${SPARK_QSFP_IFACE:-enp1s0f0np0}"

    # Export for child processes
    export SPARK_MODE SPARK_ENGINE SPARK_PRIMARY_HOST SPARK_SECONDARY_HOST
    export SPARK_PORT SPARK_QSFP_IFACE SPARK_TOOLS_DIR
    export MODEL_NAME TP_SIZE MAX_MODEL_LEN GPU_MEM_UTIL
    export VLLM_EXTRA_ARGS TRTLLM_PORT TRTLLM_MAX_BATCH_SIZE
    export TRTLLM_MAX_NUM_TOKENS TRTLLM_GPU_MEM_FRACTION
    export SERVED_MODEL_NAME
}

# --- Validation ---
spark_validate_mode() {
    case "$SPARK_MODE" in
        swarm|ray) ;;
        *) spark_die "Invalid SPARK_MODE='$SPARK_MODE'. Must be 'swarm' or 'ray'." ;;
    esac

    case "$SPARK_ENGINE" in
        trtllm|vllm) ;;
        *) spark_die "Invalid SPARK_ENGINE='$SPARK_ENGINE'. Must be 'trtllm' or 'vllm'." ;;
    esac

    if [[ "$SPARK_MODE" == "ray" && "$SPARK_ENGINE" == "trtllm" ]]; then
        spark_die "Invalid combination: ray + trtllm is not supported.
TRT-LLM requires Docker Swarm + MPI. Use SPARK_MODE=swarm or SPARK_ENGINE=vllm."
    fi
}

# --- Dispatch ---
# Routes to the correct backend script: backends/{mode}/{action}.sh
spark_dispatch() {
    local action="$1"
    shift

    spark_validate_mode

    local backend_script="${SPARK_TOOLS_DIR}/backends/${SPARK_MODE}/${action}.sh"
    if [[ ! -x "$backend_script" ]]; then
        spark_die "Backend not found: ${SPARK_MODE}/${action}.sh
Is ${SPARK_MODE} mode fully installed?"
    fi

    exec "$backend_script" "$@"
}

# --- Info Display ---
spark_show_config() {
    _bold "Spark Tools Configuration"
    echo "  Mode:      ${SPARK_MODE}"
    echo "  Engine:    ${SPARK_ENGINE}"
    echo "  Primary:   ${SPARK_PRIMARY_HOST}"
    echo "  Secondary: ${SPARK_SECONDARY_HOST}"
    echo "  QSFP:      ${SPARK_QSFP_IFACE}"
    echo "  API Port:  ${SPARK_PORT}"
    if [[ -n "${MODEL_NAME:-}" ]]; then
        echo "  Model:     ${MODEL_NAME}"
        echo "  TP Size:   ${TP_SIZE:-2}"
        echo "  Context:   ${MAX_MODEL_LEN:-32768}"
        echo "  GPU Mem:   ${GPU_MEM_UTIL:-0.85}"
    fi
}

# --- Effective port ---
# TRT-LLM uses its own port; vLLM uses SPARK_PORT
spark_effective_port() {
    if [[ "$SPARK_ENGINE" == "trtllm" ]]; then
        echo "${TRTLLM_PORT:-8355}"
    else
        echo "${SPARK_PORT:-8000}"
    fi
}

# --- SSH helper ---
spark_ssh() {
    local host="$1"
    shift
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$host" "$@"
}
