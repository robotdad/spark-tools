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
#
# Load order (last value for a given key wins):
#   1. cluster.env  — orchestration mode, engine, host names, network, images
#   2. model.env    — model identity, TP size, memory, vLLM/TRT-LLM flags
#   3. node.env     — per-node overrides (TP_SIZE=1, GPU_MEM_UTIL, etc.)
#
# node.env is loaded by spark_load_node_env(), called automatically at the
# end of spark_load_config(). It sources (in ascending priority):
#   /etc/spark-tools/node.env                    — system-level node override
#   ~/.config/spark-tools/node.env               — user-level generic override
#   ~/.config/spark-tools/node.env.<hostname>    — user-level host-specific override
#
spark_load_config() {
    # 1. Load cluster config (required)
    if [[ ! -f "$SPARK_CLUSTER_ENV" ]]; then
        spark_die "Cluster config not found: $SPARK_CLUSTER_ENV
Run 'spark-init' to create it."
    fi
    # shellcheck source=/dev/null
    source "$SPARK_CLUSTER_ENV"

    # 2. Load model config (optional — some commands don't need it)
    if [[ -f "$SPARK_MODEL_ENV" ]]; then
        # shellcheck source=/dev/null
        source "$SPARK_MODEL_ENV"
    fi

    # 3. Load node-specific overrides (optional — overrides model.env values)
    spark_load_node_env

    # Apply defaults for any vars not set by any config file or environment
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

# --- Node-specific env loading ---
# Sources per-node override files in ascending priority order so that the
# most specific file (host-named) always wins.  All files are optional.
# Call this after model.env is loaded; node.env values intentionally shadow
# model.env settings (e.g. dropping TP_SIZE from 2 → 1 in split mode).
spark_load_node_env() {
    local node_hostname
    node_hostname="$(hostname -s)"

    # System-level generic override (lowest priority among node files)
    if [[ -f "/etc/spark-tools/node.env" ]]; then
        # shellcheck source=/dev/null
        source "/etc/spark-tools/node.env"
    fi

    # User-level generic override
    if [[ -f "${SPARK_CONFIG_DIR}/node.env" ]]; then
        # shellcheck source=/dev/null
        source "${SPARK_CONFIG_DIR}/node.env"
    fi

    # User-level host-specific override (highest priority)
    # e.g. ~/.config/spark-tools/node.env.monad or node.env.dyad
    if [[ -f "${SPARK_CONFIG_DIR}/node.env.${node_hostname}" ]]; then
        # shellcheck source=/dev/null
        source "${SPARK_CONFIG_DIR}/node.env.${node_hostname}"
    fi

    # Re-export model-related vars in case node.env changed them
    export MODEL_NAME TP_SIZE MAX_MODEL_LEN GPU_MEM_UTIL VLLM_EXTRA_ARGS SERVED_MODEL_NAME
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

# --- Port availability check ---
# Returns 0 if the port is free, 1 if occupied.
# When occupied, prints the process holding it to stderr.
spark_check_port() {
    local port="${1:-$(spark_effective_port)}"
    local host="${2:-localhost}"
    local holder
    holder="$(ss -tlnp "sport = :${port}" 2>/dev/null | tail -n +2)"
    if [[ -z "$holder" ]]; then
        return 0
    fi
    local proc_info
    proc_info="$(echo "$holder" | grep -oP 'users:\(\("\K[^"]+' | head -1)"
    echo "Port ${port} on ${host} is already in use by: ${proc_info:-unknown process}" >&2
    echo "  ${holder}" >&2
    return 1
}

# --- SSH helpers ---
# When running under sudo, drop back to the real operator so SSH can
# access their ~/.ssh/ keys and agent socket (root has neither).
spark_ssh() {
    local host="$1"
    shift
    if [[ -n "${SUDO_USER:-}" ]]; then
        sudo -u "$SUDO_USER" ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$host" "$@"
    else
        ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$host" "$@"
    fi
}

# SSH with TTY allocation — use for commands that need remote sudo
# when the remote user may not have NOPASSWD.  Do NOT use for piped input.
spark_ssh_tty() {
    local host="$1"
    shift
    if [[ -n "${SUDO_USER:-}" ]]; then
        sudo -u "$SUDO_USER" ssh -t -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$host" "$@"
    else
        ssh -t -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$host" "$@"
    fi
}
