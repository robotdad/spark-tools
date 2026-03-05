#!/usr/bin/env bash
set -euo pipefail

#=============================================================================
# spark-tools — Self-Contained Installer
#=============================================================================
# Installs and configures spark-tools on a two-node DGX Spark cluster.
# Combines the logic formerly split across spark-init, spark-install-services,
# and spark-install-extras into a single self-contained script.
#
# Cluster topology:
#   monad  (head)   — cluster manager, Ray head / Swarm manager, vLLM API
#   dyad   (worker) — compute worker, Ray worker / Swarm worker
#
# Usage:
#   ./install.sh [OPTIONS]
#
# Options:
#   --node-type TYPE      Node role: head or worker (default: auto-detect from hostname)
#   --mode MODE           Orchestration backend: swarm or ray (default: swarm)
#   --head-ip IP          Head node IP on QSFP network (default: auto-detect)
#   --worker-ip IP        Worker node IP on QSFP network (default: auto-detect)
#   --qsfp-iface IFACE    High-speed inter-node interface name (default: auto-detect)
#   --user USERNAME       Username to configure services for (default: current user)
#   --force               Overwrite existing config files (default: skip existing)
#   --dry-run             Show what would be done without making any changes
#   --help                Show this help message
#
# Examples:
#   ./install.sh                              # Auto-detect everything
#   ./install.sh --node-type head             # Install monad as head (Ray or Swarm)
#   ./install.sh --node-type worker           # Install dyad as worker
#   ./install.sh --node-type head --mode ray  # Head node with Ray backend
#   ./install.sh --dry-run                    # Preview without changes
#   ./install.sh --force                      # Re-install, overwriting configs
#   ./install.sh --node-type head --head-ip 192.168.10.1 --worker-ip 192.168.10.2
#
# What this installs:
#   ~/.config/spark-tools/cluster.env    — cluster settings (mode, hosts, network)
#   ~/.config/spark-tools/model.env      — model settings (model, parallelism, memory)
#   /usr/local/bin/spark-*               — symlinks to bin/ commands
#   /etc/systemd/system/spark-*.service  — service units for chosen mode + node role
#   /etc/sysctl.conf                     — vm.swappiness=1 (GB10 UMA tuning)
#   /etc/hosts                           — monad + dyad hostname entries (if absent)
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARK_TOOLS_DIR="$SCRIPT_DIR"

#-----------------------------------------------------------------------------
# Defaults
#-----------------------------------------------------------------------------
USERNAME="${USER}"
HEAD_IP=""
WORKER_IP=""
QSFP_IFACE=""
NODE_TYPE=""          # empty = auto-detect from hostname
SPARK_MODE="swarm"
DRY_RUN=false
FORCE=false

#-----------------------------------------------------------------------------
# Usage
#-----------------------------------------------------------------------------
usage() {
    cat <<'EOF'
install.sh — unified spark-tools installer for a two-node DGX Spark cluster

Installs and fully configures spark-tools on a single node in one step. Covers
config files, system PATH symlinks, systemd service units, kernel tuning, and
/etc/hosts entries. Run independently on each node (monad then dyad, or dyad
then monad — order does not matter).

WHY: Rather than calling spark-init, spark-install-services, and
spark-install-extras by hand and risk missing a step, install.sh executes all
eight setup phases atomically with a consistent set of auto-detected values.
Use --dry-run to preview every action before anything is written.

Usage:
  ./install.sh [OPTIONS]

  Run from the spark-tools repo root. sudo access is required (unless
  --dry-run is given).

Cluster topology:
  monad  (head)    cluster manager — Ray head / Swarm manager, vLLM API server
  dyad   (worker)  compute worker  — Ray worker / Swarm worker node

Options:
  --node-type TYPE    Node role for this machine: head or worker.
                      Default: auto-detected from hostname.
                        monad → head
                        dyad  → worker
                        other → head (with a warning)

  --mode MODE         Orchestration backend: swarm or ray.
                      Default: swarm.
                        swarm — Docker Swarm + TRT-LLM (recommended)
                        ray   — Ray cluster + vLLM (forces SPARK_ENGINE=vllm)

  --head-ip IP        IP address of monad on the QSFP high-speed link.
                      Default: auto-detected (see Auto-detection below).
                      Used in /etc/hosts and systemd service templates.

  --worker-ip IP      IP address of dyad on the QSFP high-speed link.
                      Default: auto-detected (see Auto-detection below).
                      Used in /etc/hosts entries.

  --qsfp-iface IFACE  Name of the high-speed inter-node network interface.
                      Default: auto-detected by probing enp1s0f0np0,
                      enp1s0f1np0, enp1s0f1np1, ib0 in order. Falls back
                      to any interface with a 169.254.x.x address, then
                      hardcodes enp1s0f0np0 as a last resort.

  --user USERNAME     System user to configure services for (home dir,
                      config paths, service ExecStart user).
                      Default: current user ($USER).

  --force             Overwrite existing cluster.env and model.env. Without
                      this flag, existing config files are left untouched.
                      Does NOT affect systemd units or /etc/hosts (always
                      written/updated).

  --dry-run           Print every action that would be taken without writing
                      any files, running systemctl, or modifying /etc/hosts
                      or /etc/sysctl.conf. Does not require sudo. Use to
                      verify the installer's decisions before committing.

  --help, -h          Show this message and exit.

Auto-detection logic:
  Node type   hostname -s == 'monad' → head; 'dyad' → worker; else → head.
  QSFP iface  First of enp1s0f0np0, enp1s0f1np0, enp1s0f1np1, ib0 that
              exists. Falls back to any 169.254.x.x interface, then the
              hardcoded default.
  HEAD_IP     On monad: IP on the QSFP iface, or primary outbound LAN IP.
              On dyad:  looked up from /etc/hosts (monad or spark-1 entry).
  WORKER_IP   On dyad:  IP on the QSFP iface, or primary outbound LAN IP.
              On monad: looked up from /etc/hosts (dyad or spark-2 entry);
              if absent, guessed as HEAD_IP + 1 on the same /24 subnet.

The eight installation steps:
  [1/8] Pre-flight checks
        Verifies lib/spark-common.sh, systemd/, and config/ exist in the repo.
        Confirms sudo access (skipped in --dry-run). Checks for ip, awk, sed,
        grep. Fails fast with a clear error if any prerequisite is missing.

  [2/8] Creating config directories
        mkdir -p ~/.config/spark-tools/ (or SPARK_CONFIG_DIR if overridden).

  [3/8] Writing cluster.env
        Copies config/cluster.env.example to ~/.config/spark-tools/cluster.env,
        substituting detected/specified values for SPARK_MODE, SPARK_ENGINE,
        SPARK_PRIMARY_HOST (monad), SPARK_SECONDARY_HOST (dyad), SPARK_QSFP_IFACE,
        and SPARK_PORT. Skipped if the file exists and --force is not given.

  [4/8] Writing model.env
        Copies config/model.env.example to ~/.config/spark-tools/model.env
        unchanged (model choice, TP size, memory, and engine flags must be
        edited manually). Default model: nvidia/Qwen3-235B-A22B-FP4.
        Skipped if the file exists and --force is not given.

  [5/8] Installing bin scripts → /usr/local/bin
        Creates sudo ln -sf symlinks for every spark-* command in bin/ into
        /usr/local/bin/, making them available system-wide on the PATH. Also
        runs chmod +x on each script in the repo.

  [6/8] Installing systemd service units
        Processes .service.template files from systemd/, substituting
        {{USERNAME}}, {{SPARK_TOOLS_DIR}}, {{HEAD_IP}}, {{WORKER_IP}},
        {{QSFP_IFACE}}, {{PORT}}, {{RAY_VLLM_IMAGE}}, etc., then writes
        rendered units to /etc/systemd/system/.

        Services installed by mode and node role:
          swarm / head    spark-swarm-stack.service   (deploy the Swarm stack)
                          spark-proxy.service         (optional auth proxy)
          swarm / worker  (no unit — dyad joins via 'docker swarm join')
          ray   / head    spark-ray-head.service      (start Ray runtime)
                          spark-ray-vllm.service      (launch vLLM on Ray)
                          spark-proxy.service         (optional auth proxy)
          ray   / worker  spark-ray-worker.service    (join Ray cluster)

        spark-proxy is installed on head nodes but NOT auto-enabled. See
        post-install instructions for how to activate it.

  [7/8] System tuning
        vm.swappiness=1: applied at runtime (sysctl) and persisted in
        /etc/sysctl.conf. The GB10 GPU uses unified memory (CPU + GPU share
        RAM); aggressive swapping causes severe thrashing under model load.

        transparent_hugepage=madvise: applied at runtime only (does not
        persist across reboots). Allows PyTorch and GPU allocators to opt
        into huge pages rather than taking them globally, reducing GC latency
        spikes during inference.

  [8/8] Updating /etc/hosts and reloading systemd
        Adds 'HEAD_IP  monad' and 'WORKER_IP  dyad' to /etc/hosts if not
        already present. Required so Ray, NCCL, and Docker can resolve node
        names without external DNS.
        Runs 'sudo systemctl daemon-reload' to register the new unit files.

Files written:
  ~/.config/spark-tools/cluster.env    cluster settings (mode, hosts, network)
  ~/.config/spark-tools/model.env      model settings (model, parallelism, mem)
  /usr/local/bin/spark-*               symlinks to bin/ commands
  /etc/systemd/system/spark-*.service  service units for chosen mode + role
  /etc/sysctl.conf                     vm.swappiness=1 (persistent)
  /etc/hosts                           monad + dyad entries (if absent)
  /sys/kernel/mm/transparent_hugepage  madvise (runtime only)

Exit codes:
  0  Installation completed successfully (or dry-run preview completed).
  1  Pre-flight check failed, invalid option value, or required command absent.

Examples:
  # Auto-detect everything and install (typical usage):
  ./install.sh

  # Preview every action without writing anything (no sudo needed):
  ./install.sh --dry-run

  # Explicitly install monad as a Ray head node:
  ./install.sh --node-type head --mode ray

  # Explicitly install dyad as a Ray worker:
  ./install.sh --node-type worker --mode ray

  # Re-install, overwriting existing config files:
  ./install.sh --force

  # Specify IPs explicitly (useful when /etc/hosts is empty on dyad):
  ./install.sh --node-type head --head-ip 192.168.10.1 --worker-ip 192.168.10.2

  # Install for a different user (e.g., running as root):
  ./install.sh --user robotdad

After installation — head node (swarm mode):
  1. Edit config:  $EDITOR ~/.config/spark-tools/cluster.env
                   $EDITOR ~/.config/spark-tools/model.env
  2. Init swarm:   docker swarm init --advertise-addr <HEAD_IP>
                   # Run the printed 'docker swarm join' command on dyad
  3. Start:        sudo systemctl enable --now spark-swarm-stack
  4. Monitor:      journalctl -u spark-swarm-stack -f
  5. Verify API:   curl http://localhost:8355/v1/models   # trtllm default port

After installation — head node (ray mode):
  1. Edit config:  $EDITOR ~/.config/spark-tools/cluster.env
  2. Start:        sudo systemctl enable spark-ray-head spark-ray-vllm
                   sudo systemctl start spark-ray-head
                   sudo systemctl start spark-ray-vllm
  3. On dyad:      sudo systemctl enable --now spark-ray-worker
  4. Verify:       curl http://localhost:8000/health

See also:
  spark-init              re-run config initialisation only
  spark-install-services  re-install service units only
  spark-install-extras    install MOTD and auth proxy secret
  spark-status            cluster health check
  spark-help              full command reference
EOF
}

#-----------------------------------------------------------------------------
# Argument Parsing
#-----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-type)
            NODE_TYPE="$2"
            shift 2
            ;;
        --mode)
            SPARK_MODE="$2"
            shift 2
            ;;
        --head-ip)
            HEAD_IP="$2"
            shift 2
            ;;
        --worker-ip)
            WORKER_IP="$2"
            shift 2
            ;;
        --qsfp-iface)
            QSFP_IFACE="$2"
            shift 2
            ;;
        --user)
            USERNAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information." >&2
            exit 1
            ;;
    esac
done

#-----------------------------------------------------------------------------
# Color / Logging Helpers
#-----------------------------------------------------------------------------
_bold()   { printf '\033[1m%s\033[0m\n' "$*"; }
_green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
_yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
_cyan()   { printf '\033[0;36m%s\033[0m\n' "$*"; }
_red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
_step()   { printf '\n\033[1;34m[%s]\033[0m \033[1m%s\033[0m\n' "$1" "$2"; }

die()  { _red "ERROR: $*" >&2; exit 1; }
warn() { _yellow "WARNING: $*" >&2; }
info() { echo "  $*"; }

#-----------------------------------------------------------------------------
# Auto-Detection Functions
#-----------------------------------------------------------------------------

# Derive node role from well-known hostname. monad = head, dyad = worker.
detect_node_type() {
    local hn
    hn="$(hostname -s 2>/dev/null || hostname)"
    case "$hn" in
        monad) echo "head" ;;
        dyad)  echo "worker" ;;
        *)
            warn "Hostname '$hn' is not 'monad' or 'dyad'; defaulting to head."
            echo "head"
            ;;
    esac
}

# Primary outbound LAN IP (used for /etc/hosts and fallback IP detection).
detect_primary_ip() {
    ip route get 8.8.8.8 2>/dev/null | grep -oP 'src \K[\d.]+' || \
    ip addr show 2>/dev/null \
        | grep -oP 'inet \K[\d.]+' \
        | grep -v '^127\.' \
        | grep -v '^169\.254\.' \
        | grep -v '^172\.' \
        | head -1
}

# IP assigned to a specific interface (e.g., the QSFP link).
detect_iface_ip() {
    local iface="$1"
    ip -4 addr show "$iface" 2>/dev/null | grep -oP 'inet \K[\d.]+' | head -1
}

# Find the QSFP / high-speed inter-node interface by probing known names.
detect_qsfp_iface() {
    local candidates=(enp1s0f0np0 enp1s0f1np0 enp1s0f1np1 ib0)
    local iface
    for iface in "${candidates[@]}"; do
        if ip link show "$iface" &>/dev/null; then
            echo "$iface"
            return
        fi
    done
    # Fallback: any interface with a link-local address may be the direct cable.
    ip addr 2>/dev/null \
        | grep -B2 '169\.254\.' \
        | grep -oP '^\d+: \K[^:@]+' \
        | head -1
}

# Look up a hostname in /etc/hosts and return its IP.
hosts_ip_for() {
    grep -E "^[0-9].*[[:space:]]${1}([[:space:]]|$)" /etc/hosts 2>/dev/null \
        | awk '{print $1}' \
        | head -1
}

#-----------------------------------------------------------------------------
# Apply Auto-Detection
#-----------------------------------------------------------------------------

# Node type
if [[ -z "$NODE_TYPE" ]]; then
    NODE_TYPE="$(detect_node_type)"
    info "Auto-detected node type: $NODE_TYPE"
fi

[[ "$NODE_TYPE" == "head" || "$NODE_TYPE" == "worker" ]] \
    || die "--node-type must be 'head' or 'worker', got: '$NODE_TYPE'"

[[ "$SPARK_MODE" == "swarm" || "$SPARK_MODE" == "ray" ]] \
    || die "--mode must be 'swarm' or 'ray', got: '$SPARK_MODE'"

# QSFP interface
if [[ -z "$QSFP_IFACE" ]]; then
    QSFP_IFACE="$(detect_qsfp_iface)"
    if [[ -n "$QSFP_IFACE" ]]; then
        info "Auto-detected QSFP interface: $QSFP_IFACE"
    else
        QSFP_IFACE="enp1s0f0np0"
        warn "No QSFP interface found; using fallback: $QSFP_IFACE"
    fi
fi

# HEAD_IP — the IP on the QSFP link on monad (used in service templates and /etc/hosts).
if [[ -z "$HEAD_IP" ]]; then
    if [[ "$NODE_TYPE" == "head" ]]; then
        # We ARE monad — try QSFP IP, then primary LAN IP.
        HEAD_IP="$(detect_iface_ip "$QSFP_IFACE")"
        [[ -z "$HEAD_IP" ]] && HEAD_IP="$(detect_primary_ip)"
        [[ -n "$HEAD_IP" ]] && info "Auto-detected HEAD_IP: $HEAD_IP"
    else
        # We are dyad — look up monad in /etc/hosts (or its alias spark-1).
        HEAD_IP="$(hosts_ip_for "monad")"
        [[ -z "$HEAD_IP" ]] && HEAD_IP="$(hosts_ip_for "spark-1")"
        if [[ -n "$HEAD_IP" ]]; then
            info "Auto-detected HEAD_IP from /etc/hosts: $HEAD_IP"
        else
            warn "Cannot determine HEAD_IP; set it manually in cluster.env after install."
            HEAD_IP=""
        fi
    fi
fi

# WORKER_IP — the IP on the QSFP link on dyad.
if [[ -z "$WORKER_IP" ]]; then
    if [[ "$NODE_TYPE" == "worker" ]]; then
        # We ARE dyad — try QSFP IP, then primary LAN IP.
        WORKER_IP="$(detect_iface_ip "$QSFP_IFACE")"
        [[ -z "$WORKER_IP" ]] && WORKER_IP="$(detect_primary_ip)"
        [[ -n "$WORKER_IP" ]] && info "Auto-detected WORKER_IP: $WORKER_IP"
    else
        # We are monad — look up dyad in /etc/hosts.
        WORKER_IP="$(hosts_ip_for "dyad")"
        [[ -z "$WORKER_IP" ]] && WORKER_IP="$(hosts_ip_for "spark-2")"
        if [[ -n "$WORKER_IP" ]]; then
            info "Auto-detected WORKER_IP from /etc/hosts: $WORKER_IP"
        elif [[ -n "$HEAD_IP" ]]; then
            # Guess: worker is head+1 on the same subnet.
            WORKER_IP="$(echo "$HEAD_IP" | awk -F. '{print $1"."$2"."$3"."$4+1}')"
            warn "WORKER_IP not found in /etc/hosts; guessing $WORKER_IP (head + 1)"
        fi
    fi
fi

# Derived values (read from cluster.env.example defaults; overridable via env).
HOME_DIR="$(eval echo "~${USERNAME}")"
SPARK_CONFIG_DIR="${SPARK_CONFIG_DIR:-${HOME_DIR}/.config/spark-tools}"
RAY_VLLM_IMAGE="${RAY_VLLM_IMAGE:-scitrera/dgx-spark-vllm:0.15.1-t5}"
SPARK_PORT="${SPARK_PORT:-8000}"
SPARK_PROXY_PORT="${SPARK_PROXY_PORT:-9000}"

# Engine: ray mode requires vllm; swarm defaults to trtllm.
if [[ "$SPARK_MODE" == "ray" ]]; then
    SPARK_ENGINE="vllm"
else
    SPARK_ENGINE="trtllm"
fi

#-----------------------------------------------------------------------------
# Display Configuration
#-----------------------------------------------------------------------------
echo ""
echo "=============================================="
_bold "  spark-tools Installation"
echo "=============================================="
echo ""
echo "  Repo:         $SPARK_TOOLS_DIR"
echo "  Username:     $USERNAME"
echo "  Home dir:     $HOME_DIR"
echo "  Config dir:   $SPARK_CONFIG_DIR"
echo "  Node type:    $NODE_TYPE"
echo "  Mode:         $SPARK_MODE  (engine: $SPARK_ENGINE)"
echo "  Head IP:      ${HEAD_IP:-<unknown>}"
echo "  Worker IP:    ${WORKER_IP:-<unknown>}"
echo "  QSFP iface:   $QSFP_IFACE"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    _yellow "[DRY RUN — no changes will be made]"
    echo ""
fi

#-----------------------------------------------------------------------------
# Helpers
#-----------------------------------------------------------------------------

# run: execute a command, or print it in dry-run mode.
run() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [dry-run] $*"
    else
        "$@"
    fi
}

# process_template: substitute all {{PLACEHOLDER}} tokens in a template file
# and emit the result to stdout.
process_template() {
    local template="$1"
    sed \
        -e "s|{{USERNAME}}|${USERNAME}|g" \
        -e "s|{{HOME_DIR}}|${HOME_DIR}|g" \
        -e "s|{{SPARK_TOOLS_DIR}}|${SPARK_TOOLS_DIR}|g" \
        -e "s|{{COMPOSE_FILE}}|${HOME_DIR}/docker-compose.yml|g" \
        -e "s|{{QSFP_IFACE}}|${QSFP_IFACE}|g" \
        -e "s|{{HEAD_IP}}|${HEAD_IP:-127.0.0.1}|g" \
        -e "s|{{WORKER_IP}}|${WORKER_IP:-127.0.0.1}|g" \
        -e "s|{{RAY_VLLM_IMAGE}}|${RAY_VLLM_IMAGE}|g" \
        -e "s|{{PORT}}|${SPARK_PORT}|g" \
        -e "s|{{UPSTREAM_PORT}}|${SPARK_PORT}|g" \
        -e "s|{{PROXY_PORT}}|${SPARK_PROXY_PORT}|g" \
        "$template"
}

# install_service: process one .service.template and write it to systemd.
install_service() {
    local template_name="$1"
    local tmpl_path="${SPARK_TOOLS_DIR}/systemd/${template_name}"
    local svc_name="${template_name%.template}"      # strip .template suffix
    local dest="/etc/systemd/system/${svc_name}"

    if [[ ! -f "$tmpl_path" ]]; then
        warn "Template not found: $tmpl_path (skipping)"
        return
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [dry-run] would install $dest"
        echo "  --- BEGIN ${svc_name} ---"
        process_template "$tmpl_path" | sed 's/^/  /'
        echo "  --- END ${svc_name} ---"
        echo ""
    else
        process_template "$tmpl_path" | sudo tee "$dest" > /dev/null
        echo "  Installed: $dest"
    fi
}

#-----------------------------------------------------------------------------
# [1/8] Pre-flight Checks
#-----------------------------------------------------------------------------
_step "1/8" "Pre-flight checks"

# Verify we're running from the repo root.
[[ -f "${SPARK_TOOLS_DIR}/lib/spark-common.sh" ]] \
    || die "lib/spark-common.sh not found — run install.sh from the spark-tools repo root."

[[ -d "${SPARK_TOOLS_DIR}/systemd" ]] \
    || die "systemd/ directory not found in repo."

[[ -f "${SPARK_TOOLS_DIR}/config/cluster.env.example" ]] \
    || die "config/cluster.env.example not found in repo."

# Verify sudo access (not needed in dry-run).
if [[ "$DRY_RUN" == false ]]; then
    sudo -v 2>/dev/null || die "sudo access is required for installation."
fi

# Verify required system commands.
for cmd in ip awk sed grep; do
    command -v "$cmd" &>/dev/null || die "Required command not found: $cmd"
done

info "OK — running as ${USERNAME} from ${SPARK_TOOLS_DIR}"

#-----------------------------------------------------------------------------
# [2/8] Config Directories
#-----------------------------------------------------------------------------
_step "2/8" "Creating config directories"

run mkdir -p "$SPARK_CONFIG_DIR"
info "$SPARK_CONFIG_DIR"

#-----------------------------------------------------------------------------
# [3/8] Write cluster.env
#-----------------------------------------------------------------------------
_step "3/8" "Writing cluster.env"

CLUSTER_ENV_DEST="${SPARK_CONFIG_DIR}/cluster.env"

if [[ -f "$CLUSTER_ENV_DEST" && "$FORCE" != true ]]; then
    info "Already exists: $CLUSTER_ENV_DEST"
    info "(use --force to overwrite)"
else
    if [[ "$DRY_RUN" == false ]]; then
        sed \
            -e "s|^SPARK_MODE=.*|SPARK_MODE=${SPARK_MODE}|" \
            -e "s|^SPARK_ENGINE=.*|SPARK_ENGINE=${SPARK_ENGINE}|" \
            -e "s|^SPARK_PRIMARY_HOST=.*|SPARK_PRIMARY_HOST=monad|" \
            -e "s|^SPARK_SECONDARY_HOST=.*|SPARK_SECONDARY_HOST=dyad|" \
            -e "s|^SPARK_QSFP_IFACE=.*|SPARK_QSFP_IFACE=${QSFP_IFACE}|" \
            -e "s|^SPARK_PORT=.*|SPARK_PORT=${SPARK_PORT}|" \
            "${SPARK_TOOLS_DIR}/config/cluster.env.example" > "$CLUSTER_ENV_DEST"
        info "Created: $CLUSTER_ENV_DEST"
        info "  SPARK_MODE=${SPARK_MODE}  SPARK_ENGINE=${SPARK_ENGINE}"
        info "  SPARK_PRIMARY_HOST=monad  SPARK_SECONDARY_HOST=dyad"
        info "  SPARK_QSFP_IFACE=${QSFP_IFACE}"
    else
        info "[dry-run] Would write $CLUSTER_ENV_DEST"
        info "  SPARK_MODE=${SPARK_MODE}  SPARK_ENGINE=${SPARK_ENGINE}"
        info "  SPARK_QSFP_IFACE=${QSFP_IFACE}"
    fi
fi

#-----------------------------------------------------------------------------
# [4/8] Write model.env
#-----------------------------------------------------------------------------
_step "4/8" "Writing model.env"

MODEL_ENV_DEST="${SPARK_CONFIG_DIR}/model.env"

if [[ -f "$MODEL_ENV_DEST" && "$FORCE" != true ]]; then
    info "Already exists: $MODEL_ENV_DEST"
    info "(use --force to overwrite)"
else
    run cp "${SPARK_TOOLS_DIR}/config/model.env.example" "$MODEL_ENV_DEST"
    info "Created: $MODEL_ENV_DEST"
    info "Default model: nvidia/Qwen3-235B-A22B-FP4 (edit to change)"
fi

#-----------------------------------------------------------------------------
# [5/8] Install Bin Scripts → /usr/local/bin
#-----------------------------------------------------------------------------
_step "5/8" "Installing bin scripts → /usr/local/bin"

for cmd_path in "${SPARK_TOOLS_DIR}"/bin/spark-*; do
    cmd_name="$(basename "$cmd_path")"
    target="/usr/local/bin/${cmd_name}"
    if [[ "$DRY_RUN" == false ]]; then
        sudo ln -sf "$cmd_path" "$target"
        sudo chmod +x "$cmd_path"
        info "$cmd_name → $target"
    else
        info "[dry-run] ln -sf $cmd_path $target"
    fi
done

#-----------------------------------------------------------------------------
# [6/8] Install Systemd Service Units
#-----------------------------------------------------------------------------
_step "6/8" "Installing systemd service units  (mode: ${SPARK_MODE}, role: ${NODE_TYPE})"
echo ""

case "$SPARK_MODE" in
    ray)
        if [[ "$NODE_TYPE" == "head" ]]; then
            info "Ray head node services (monad):"
            # Ray cluster head — starts the Ray runtime.
            install_service "spark-ray-head.service.template"
            # vLLM server — waits for Ray to be ready, then launches model serving.
            install_service "spark-ray-vllm.service.template"
        else
            info "Ray worker node service (dyad):"
            # Ray worker — joins the cluster started by spark-ray-head on monad.
            install_service "spark-ray-worker.service.template"
        fi
        ;;

    swarm)
        if [[ "$NODE_TYPE" == "head" ]]; then
            info "Docker Swarm stack service (monad — swarm manager):"
            # Swarm stack — deploys the multi-node compose stack from monad.
            install_service "spark-swarm-stack.service.template"
            info ""
            info "Note: dyad must join the swarm before starting this service."
            info "  On monad: docker swarm init --advertise-addr ${HEAD_IP:-<HEAD_IP>}"
            info "  On dyad:  docker swarm join --token <token> ${HEAD_IP:-<HEAD_IP>}:2377"
        else
            info "Swarm worker (dyad): no service unit needed."
            info "  dyad participates by running 'docker swarm join' on the swarm overlay."
            info "  Run the join command printed by 'docker swarm init' on monad."
        fi
        ;;
esac

# Auth proxy service — head only, installed unconditionally but not auto-enabled.
# Enable separately with: sudo systemctl enable --now spark-proxy
if [[ "$NODE_TYPE" == "head" ]]; then
    echo ""
    info "Auth proxy service (optional, head only):"
    install_service "spark-proxy.service.template"
    info "  (installed but not enabled — see post-install steps to activate)"
fi

#-----------------------------------------------------------------------------
# [7/8] System Tuning
#-----------------------------------------------------------------------------
_step "7/8" "System tuning"

# --- vm.swappiness=1 ---
# GB10 GPUs use unified memory architecture (UMA): CPU and GPU share the same
# physical RAM pool. Under memory pressure, the kernel may swap out pages that
# the GPU is actively using, causing severe thrashing. Setting swappiness=1
# virtually disables swapping while retaining it as an emergency safety net.
CURRENT_SWAPPINESS="$(cat /proc/sys/vm/swappiness 2>/dev/null || echo '?')"
info "vm.swappiness: ${CURRENT_SWAPPINESS} → 1  (UMA/GB10 tuning)"

if [[ "$DRY_RUN" == false ]]; then
    sudo sysctl -q vm.swappiness=1
    if grep -q '^vm.swappiness' /etc/sysctl.conf 2>/dev/null; then
        sudo sed -i 's/^vm.swappiness=.*/vm.swappiness=1/' /etc/sysctl.conf
    else
        echo "vm.swappiness=1" | sudo tee -a /etc/sysctl.conf > /dev/null
    fi
    info "vm.swappiness=1  (runtime + persistent via /etc/sysctl.conf)"
else
    info "[dry-run] sysctl vm.swappiness=1"
    info "[dry-run] /etc/sysctl.conf: vm.swappiness=1"
fi

# --- Transparent Huge Pages: madvise ---
# madvise lets applications opt-in to THP rather than using it globally.
# LLM workloads (particularly PyTorch allocators) perform better with madvise
# than with the default "always" policy, which can cause GC latency spikes.
THP_PATH="/sys/kernel/mm/transparent_hugepage/enabled"
if [[ -f "$THP_PATH" ]]; then
    THP_CURRENT="$(grep -oP '\[\K[^\]]+' "$THP_PATH" 2>/dev/null || echo '?')"
    info "transparent_hugepage: ${THP_CURRENT} → madvise"
    if [[ "$DRY_RUN" == false ]]; then
        echo madvise | sudo tee "$THP_PATH" > /dev/null
        info "transparent_hugepage=madvise  (runtime only — does not persist across reboots)"
    else
        info "[dry-run] echo madvise > $THP_PATH"
    fi
else
    info "transparent_hugepage: sysfs path not found, skipping"
fi

#-----------------------------------------------------------------------------
# [8/8] /etc/hosts + Systemd Reload
#-----------------------------------------------------------------------------
_step "8/8" "Updating /etc/hosts and reloading systemd"
echo ""

# Add monad / dyad to /etc/hosts if not already present.
# This ensures Ray, NCCL, and Docker can resolve node names without external DNS.
add_hosts_entry() {
    local ip="$1"
    local hostname="$2"
    if [[ -z "$ip" ]]; then
        warn "/etc/hosts: skipping $hostname (IP unknown — set manually)"
        return
    fi
    if grep -qE "^[0-9].*[[:space:]]${hostname}([[:space:]]|$)" /etc/hosts 2>/dev/null; then
        info "/etc/hosts: '$hostname' already present (skipping)"
    else
        info "/etc/hosts: adding '$ip  $hostname'"
        if [[ "$DRY_RUN" == false ]]; then
            echo "$ip  $hostname" | sudo tee -a /etc/hosts > /dev/null
        fi
    fi
}

add_hosts_entry "$HEAD_IP"   "monad"
add_hosts_entry "$WORKER_IP" "dyad"

echo ""

# Reload systemd so the newly installed unit files are recognized.
info "Reloading systemd daemon..."
if [[ "$DRY_RUN" == false ]]; then
    sudo systemctl daemon-reload
    info "systemd daemon reloaded."
else
    info "[dry-run] systemctl daemon-reload"
fi

#=============================================================================
# Installation Complete — Post-Install Instructions
#=============================================================================
echo ""
echo "=============================================="
_green "  Installation complete!"
echo "=============================================="
echo ""

if [[ "$NODE_TYPE" == "head" ]]; then
    _bold "Next steps — HEAD node (monad):"
    echo ""
    echo "  1. Review and edit your config files:"
    echo "       \$EDITOR $CLUSTER_ENV_DEST"
    echo "       \$EDITOR $MODEL_ENV_DEST"
    echo ""

    if [[ "$SPARK_MODE" == "ray" ]]; then
        echo "  2. Enable and start Ray services:"
        echo "       sudo systemctl enable spark-ray-head spark-ray-vllm"
        echo "       sudo systemctl start  spark-ray-head"
        echo "       # spark-ray-vllm waits for Ray to be ready, then starts vLLM:"
        echo "       sudo systemctl start  spark-ray-vllm"
        echo ""
        echo "     On dyad (after its install.sh run):"
        echo "       sudo systemctl enable --now spark-ray-worker"
        echo ""
        echo "  3. Monitor startup:"
        echo "       journalctl -u spark-ray-head -f"
        echo "       journalctl -u spark-ray-vllm  -f"
        echo "       docker exec spark-ray ray status"
        echo ""
        echo "  4. Verify the API:"
        echo "       curl http://localhost:${SPARK_PORT}/health"
        echo "       curl http://localhost:${SPARK_PORT}/v1/models"
        echo ""
    else
        echo "  2. Initialize Docker Swarm (if not already done):"
        echo "       docker swarm init --advertise-addr ${HEAD_IP:-<HEAD_IP>}"
        echo "       # Copy the 'docker swarm join' token printed above, then run it on dyad."
        echo ""
        echo "  3. Create ~/docker-compose.yml for the TRT-LLM stack, then:"
        echo "       sudo systemctl enable --now spark-swarm-stack"
        echo ""
        echo "  4. Monitor startup:"
        echo "       journalctl -u spark-swarm-stack -f"
        echo "       docker stack ps trtllm-multinode"
        echo ""
    fi

    echo "  Optional — auth proxy (bearer-token protection for the API):"
    echo "       cd $SPARK_TOOLS_DIR/proxy"
    echo "       python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    echo "       mkdir -p ~/.config/vllm-proxy"
    echo "       openssl rand -hex 32 > ~/.config/vllm-proxy/secret.env"
    echo "       sudo systemctl enable --now spark-proxy"
    echo ""
    echo "  Optional — install login banner:"
    echo "       spark-install-extras --motd"
    echo ""

else
    _bold "Next steps — WORKER node (dyad):"
    echo ""

    if [[ "$SPARK_MODE" == "ray" ]]; then
        echo "  1. Ensure monad is in /etc/hosts (the installer added it if HEAD_IP was known):"
        echo "       grep monad /etc/hosts"
        echo ""
        echo "  2. Enable and start the Ray worker:"
        echo "       sudo systemctl enable --now spark-ray-worker"
        echo ""
        echo "  3. Monitor:"
        echo "       journalctl -u spark-ray-worker -f"
        echo "       # Verify from monad: docker exec spark-ray ray status"
        echo ""
    else
        echo "  1. Get the swarm join token from monad:"
        echo "       ssh monad docker swarm join-token worker"
        echo ""
        echo "  2. Join the swarm (run the command printed above):"
        echo "       docker swarm join --token <SWARM_TOKEN> ${HEAD_IP:-<HEAD_IP>}:2377"
        echo ""
        echo "  3. Verify from monad:"
        echo "       docker node ls"
        echo ""
    fi
fi

_bold "Cluster management commands (now in PATH):"
echo ""
echo "   spark-status      — cluster health check"
echo "   spark-mode        — show / switch orchestration mode (swarm|ray)"
echo "   spark-serve       — start inference server"
echo "   spark-stop        — stop inference server"
echo "   spark-set-model   — change active model"
echo "   spark-reset       — full cluster restart"
echo "   spark-bench       — run inference benchmark"
echo ""
_bold "Config files:"
echo ""
echo "   $CLUSTER_ENV_DEST"
echo "   $MODEL_ENV_DEST"
echo ""
if [[ "$DRY_RUN" == true ]]; then
    _yellow "[DRY RUN complete — no changes were made]"
    echo ""
fi
