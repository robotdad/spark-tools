#!/usr/bin/env bash
# spark-common.sh - Shared library for spark-tools
# Source this file; do not execute directly.

# Repo root (parent of lib/)
SPARK_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Config paths — resolve the real operator's home under sudo
_SPARK_HOME="${HOME}"
if [[ -n "${SUDO_USER:-}" ]]; then
    _SPARK_HOME="$(eval echo "~${SUDO_USER}")"
fi
SPARK_CONFIG_DIR="${SPARK_CONFIG_DIR:-${_SPARK_HOME}/.config/spark-tools}"
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
    export VLLM_EXTRA_ARGS VLLM_IMAGE VLLM_SERVE_CMD
    export TRTLLM_PORT TRTLLM_MAX_BATCH_SIZE
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
    export MODEL_NAME TP_SIZE MAX_MODEL_LEN GPU_MEM_UTIL
    export VLLM_EXTRA_ARGS VLLM_IMAGE VLLM_SERVE_CMD SERVED_MODEL_NAME
}

# --- Model Profile Resolution ---
# Profiles live in $SPARK_TOOLS_DIR/profiles/ as env files keyed by model name.
# Model name "Org/Model" maps to file "Org--Model.env" (slashes → double dash).
#
# Profiles carry VLLM_IMAGE, VLLM_SERVE_CMD, VLLM_EXTRA_ARGS, and recommended
# defaults (PROFILE_DEFAULT_*) for TP_SIZE, MAX_MODEL_LEN, GPU_MEM_UTIL.
#
# Usage:
#   spark_resolve_profile "google/gemma-4-26B-A4B-it"
#     → prints the profile file path and returns 0, or returns 1 if not found
#
#   spark_apply_profile "google/gemma-4-26B-A4B-it" /path/to/model.env
#     → applies VLLM_IMAGE, VLLM_SERVE_CMD, VLLM_EXTRA_ARGS from the profile
#       to the given model.env file via sed substitution.
#       Optionally applies PROFILE_DEFAULT_* values if the corresponding
#       model.env value was not explicitly set by the user (--tp, --ctx, --mem).

SPARK_PROFILES_DIR="${SPARK_TOOLS_DIR}/profiles"

# Resolve a model name to a profile file path.  Returns 0 and prints the
# path if found, returns 1 if no matching profile exists.
spark_resolve_profile() {
    local model_name="$1"
    local profile_file="${SPARK_PROFILES_DIR}/$(echo "${model_name}" | sed 's|/|--|g').env"
    if [[ -f "$profile_file" ]]; then
        echo "$profile_file"
        return 0
    fi
    return 1
}

# Apply a profile's settings to model.env.
# Arguments:
#   $1  model name (HuggingFace repo ID)
#   $2  path to model.env to update
#   $3  (optional) "defaults" — also apply PROFILE_DEFAULT_* values
spark_apply_profile() {
    local model_name="$1"
    local model_env="$2"
    local apply_defaults="${3:-}"

    local profile_file
    if ! profile_file=$(spark_resolve_profile "$model_name"); then
        return 1
    fi

    # Source profile into a subshell to avoid polluting current env
    local p_vllm_image p_vllm_serve_cmd p_vllm_extra_args
    local p_default_tp p_default_ctx p_default_mem
    eval "$(
        source "$profile_file"
        echo "p_vllm_image='${VLLM_IMAGE:-}'"
        echo "p_vllm_serve_cmd='${VLLM_SERVE_CMD:-}'"
        echo "p_vllm_extra_args='${VLLM_EXTRA_ARGS:-}'"
        echo "p_default_tp='${PROFILE_DEFAULT_TP_SIZE:-}'"
        echo "p_default_ctx='${PROFILE_DEFAULT_MAX_MODEL_LEN:-}'"
        echo "p_default_mem='${PROFILE_DEFAULT_GPU_MEM_UTIL:-}'"
    )"

    # Apply VLLM_IMAGE
    if [[ -n "$p_vllm_image" ]]; then
        if grep -q '^VLLM_IMAGE=' "$model_env"; then
            sed -i "s|^VLLM_IMAGE=.*|VLLM_IMAGE=${p_vllm_image}|" "$model_env"
        else
            echo "VLLM_IMAGE=${p_vllm_image}" >> "$model_env"
        fi
    fi

    # Apply VLLM_SERVE_CMD
    if grep -q '^VLLM_SERVE_CMD=' "$model_env"; then
        sed -i "s|^VLLM_SERVE_CMD=.*|VLLM_SERVE_CMD=${p_vllm_serve_cmd}|" "$model_env"
    else
        echo "VLLM_SERVE_CMD=${p_vllm_serve_cmd}" >> "$model_env"
    fi

    # Apply VLLM_EXTRA_ARGS
    if [[ -n "$p_vllm_extra_args" ]]; then
        sed -i "s|^VLLM_EXTRA_ARGS=.*|VLLM_EXTRA_ARGS=\"${p_vllm_extra_args}\"|" "$model_env"
    fi

    # Optionally apply recommended defaults
    if [[ "$apply_defaults" == "defaults" ]]; then
        [[ -n "$p_default_tp" ]]  && sed -i "s/^TP_SIZE=.*/TP_SIZE=${p_default_tp}/" "$model_env"
        [[ -n "$p_default_ctx" ]] && sed -i "s/^MAX_MODEL_LEN=.*/MAX_MODEL_LEN=${p_default_ctx}/" "$model_env"
        [[ -n "$p_default_mem" ]] && sed -i "s/^GPU_MEM_UTIL=.*/GPU_MEM_UTIL=${p_default_mem}/" "$model_env"
    fi

    return 0
}

# List all available profiles.  Prints "Org/Model" names, one per line.
spark_list_profiles() {
    if [[ ! -d "$SPARK_PROFILES_DIR" ]]; then
        return 0
    fi
    local f
    for f in "${SPARK_PROFILES_DIR}"/*.env; do
        [[ -f "$f" ]] || continue
        local basename="${f##*/}"        # "Org--Model.env"
        basename="${basename%.env}"       # "Org--Model"
        echo "${basename/--//}"          # "Org/Model"
    done
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

# --- Topology ---
# Read the current topology (split | cluster). Defaults to "cluster".
spark_read_topology() {
    local topo_file="${SPARK_CONFIG_DIR}/topology"
    if [[ -f "$topo_file" ]]; then
        cat "$topo_file"
    else
        echo "cluster"
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

# --- Port availability ---
# Returns 0 if the port is free, 1 if occupied.
spark_port_free() {
    local port="$1"
    [[ -z "$(ss -tlnp "sport = :${port}" 2>/dev/null | tail -n +2)" ]]
}

# Find a free port starting from the configured one, auto-incrementing if needed.
# Updates SPARK_PORT (and cluster.env) when it has to bump.
spark_resolve_port() {
    local port="${SPARK_PORT:-8000}"
    local max_tries=10
    local original="$port"

    if spark_port_free "$port"; then
        return 0
    fi

    local holder
    holder="$(ss -tlnp "sport = :${port}" 2>/dev/null | tail -n +2 | head -1)"
    local proc_info
    proc_info="$(echo "$holder" | grep -oP 'users:\(\("\K[^"]+' | head -1)"
    spark_warn "Port ${port} in use by ${proc_info:-another process}, scanning for a free port..."

    local i=1
    while (( i < max_tries )); do
        port=$(( original + i ))
        if spark_port_free "$port"; then
            spark_info "Using port ${port} instead of ${original}"
            export SPARK_PORT="$port"
            # Persist so subsequent commands (spark-check, proxy, etc.) use the same port.
            local _cfg="${SPARK_CONFIG_DIR}/cluster.env"
            if [[ -f "$_cfg" ]]; then
                sed -i "s/^SPARK_PORT=.*/SPARK_PORT=${port}/" "$_cfg"
                spark_info "Updated SPARK_PORT=${port} in ${_cfg}"
            fi
            return 0
        fi
        (( i++ ))
    done

    spark_die "No free port found in range ${original}-${port}. Free a port or set SPARK_PORT manually."
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

# --- Training Config ---
# Separate from spark_load_config because training.env is optional and
# only needed by spark-train-* commands (not every spark-* command).
SPARK_TRAINING_ENV="${SPARK_CONFIG_DIR}/training.env"
SPARK_TRAINING_LOG_DIR="${_SPARK_HOME}/.local/share/spark-tools/training"
SPARK_TRAINING_PID="/tmp/spark-train.pid"

spark_load_training_config() {
    if [[ ! -f "$SPARK_TRAINING_ENV" ]]; then
        spark_die "Training config not found: $SPARK_TRAINING_ENV
Run 'spark-train-setup' to create it, or copy config/training.env.example:
  cp ${SPARK_TOOLS_DIR}/config/training.env.example ${SPARK_TRAINING_ENV}"
    fi
    # shellcheck source=/dev/null
    source "$SPARK_TRAINING_ENV"

    # Validate required variables
    [[ -z "${TRAINING_SCRIPT_DIR:-}" ]] && spark_die "TRAINING_SCRIPT_DIR not set in $SPARK_TRAINING_ENV"
    [[ -z "${TRAINING_DATA_DIR:-}" ]]   && spark_die "TRAINING_DATA_DIR not set in $SPARK_TRAINING_ENV"
    [[ -z "${TRAINING_CHECKPOINT_DIR:-}" ]] && spark_die "TRAINING_CHECKPOINT_DIR not set in $SPARK_TRAINING_ENV"

    # Apply defaults for optional vars
    TRAINING_MEMORY_MAX="${TRAINING_MEMORY_MAX:-100G}"
    CHECKPOINT_TIMEOUT="${CHECKPOINT_TIMEOUT:-120}"

    export TRAINING_SCRIPT_DIR TRAINING_DATA_DIR TRAINING_CHECKPOINT_DIR
    export TRAINING_MEMORY_MAX CHECKPOINT_TIMEOUT
}

# --- Training Process Detection ---
# Check if training is currently running (PID file exists and process alive).
# Returns 0 if training is running, 1 otherwise.
spark_training_running() {
    if [[ -f "$SPARK_TRAINING_PID" ]]; then
        local pid
        pid=$(cat "$SPARK_TRAINING_PID" 2>/dev/null)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        # Stale PID file — process is dead, clean up
        rm -f "$SPARK_TRAINING_PID"
    fi
    return 1
}

# Get the training PID (if running). Prints PID to stdout, returns 1 if not running.
spark_training_pid() {
    if spark_training_running; then
        cat "$SPARK_TRAINING_PID"
        return 0
    fi
    return 1
}

# Write training PID file
spark_training_write_pid() {
    local pid="$1"
    echo "$pid" > "$SPARK_TRAINING_PID"
}

# Remove training PID file
spark_training_remove_pid() {
    rm -f "$SPARK_TRAINING_PID"
}

# --- OOM Protection ---
# Apply memory safety protections on a node. Idempotent — safe to call every time.
# Usage: spark_apply_oom_protection [node]
#   node = "local" (default) or a hostname (runs via spark_ssh)
spark_apply_oom_protection() {
    local node="${1:-local}"
    local mem_max="${TRAINING_MEMORY_MAX:-100G}"

    local cmds="
        # 1. Disable swap
        sudo swapoff -a 2>/dev/null || true

        # 2. Protect SSH from OOM killer
        sudo mkdir -p /etc/systemd/system/ssh.service.d
        echo '[Service]
OOMScoreAdjust=-1000
MemoryMin=512M' | sudo tee /etc/systemd/system/ssh.service.d/oom.conf >/dev/null
        sudo systemctl daemon-reload
        sudo systemctl restart ssh 2>/dev/null || sudo systemctl restart sshd 2>/dev/null || true

        # 3. Drop filesystem caches
        sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    "

    if [[ "$node" == "local" ]]; then
        eval "$cmds"
    else
        spark_ssh "$node" "$cmds"
    fi
}

# Check if inference is running on a node.
# Returns 0 if any inference process detected, 1 otherwise.
# Usage: spark_inference_running [node]
#   node = "local" (default) or a hostname
spark_inference_running() {
    local node="${1:-local}"

    local check_cmd='
        # Check systemd services
        for svc in spark-vllm-standalone vllm vllm-ray-head spark-ray-vllm; do
            if systemctl is-active --quiet "$svc" 2>/dev/null; then
                exit 0
            fi
        done
        # Check Docker stack
        if docker stack ls 2>/dev/null | grep -q "^spark"; then
            exit 0
        fi
        exit 1
    '

    if [[ "$node" == "local" ]]; then
        bash -c "$check_cmd" 2>/dev/null
    else
        spark_ssh "$node" "$check_cmd" 2>/dev/null
    fi
}
