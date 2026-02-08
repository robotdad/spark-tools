#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: sudo spark-harden [--check|--apply]"
    echo ""
    echo "Harden DGX Spark for inference workloads."
    echo ""
    echo "Actions:"
    echo "  --check    Show current state without changing anything (default)"
    echo "  --apply    Apply hardening changes"
    echo ""
    echo "Hardening includes:"
    echo "  1. Disable swap (unified memory + swap = system freeze on OOM)"
    echo "  2. Protect sshd from OOM killer (keep SSH access during OOM)"
    echo "  3. Set vm.overcommit_memory=1 (prevent allocation failures)"
    echo ""
    echo "WARNING: These changes affect the whole system. Review --check output first."
    exit 0
fi

MODE="${1:---check}"

# Color helpers
_green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
_red()   { printf '\033[0;31m%s\033[0m\n' "$*"; }
_yellow(){ printf '\033[0;33m%s\033[0m\n' "$*"; }

echo "=== DGX Spark System Hardening ==="
echo "Mode: ${MODE}"
echo ""

# 1. Swap
echo "--- Swap ---"
SWAP_ACTIVE=$(swapon --show --noheadings 2>/dev/null | wc -l)
if [[ "$SWAP_ACTIVE" -gt 0 ]]; then
    _red "  Swap is ACTIVE ($SWAP_ACTIVE device(s))"
    swapon --show
    if [[ "$MODE" == "--apply" ]]; then
        echo "  Disabling swap..."
        swapoff -a
        # Comment out swap entries in fstab
        sed -i '/\sswap\s/s/^/#/' /etc/fstab 2>/dev/null || true
        _green "  Swap disabled and fstab updated."
    else
        echo "  Run with --apply to disable."
    fi
else
    _green "  Swap is disabled. Good."
fi
echo ""

# 2. SSH OOM protection
echo "--- SSHD OOM Protection ---"
SSHD_PID=$(pgrep -x sshd | head -1 || true)
if [[ -n "$SSHD_PID" ]]; then
    OOM_SCORE=$(cat "/proc/$SSHD_PID/oom_score_adj" 2>/dev/null || echo "unknown")
    if [[ "$OOM_SCORE" == "-1000" ]]; then
        _green "  sshd (PID $SSHD_PID) is OOM-protected (oom_score_adj=$OOM_SCORE)."
    else
        _yellow "  sshd (PID $SSHD_PID) is NOT OOM-protected (oom_score_adj=$OOM_SCORE)."
        if [[ "$MODE" == "--apply" ]]; then
            # Protect current sshd process
            echo -1000 > "/proc/$SSHD_PID/oom_score_adj"
            # Make it persistent via systemd override
            mkdir -p /etc/systemd/system/ssh.service.d
            cat > /etc/systemd/system/ssh.service.d/oom-protect.conf <<'EOF'
[Service]
OOMScoreAdjust=-1000
EOF
            systemctl daemon-reload
            _green "  sshd OOM-protected (current + persistent via systemd override)."
        else
            echo "  Run with --apply to protect."
        fi
    fi
else
    _yellow "  sshd process not found."
fi
echo ""

# 3. vm.overcommit_memory
echo "--- Memory Overcommit ---"
OVERCOMMIT=$(cat /proc/sys/vm/overcommit_memory 2>/dev/null || echo "unknown")
if [[ "$OVERCOMMIT" == "1" ]]; then
    _green "  vm.overcommit_memory=1 (always overcommit). Good for inference."
else
    _yellow "  vm.overcommit_memory=$OVERCOMMIT (default heuristic)."
    if [[ "$MODE" == "--apply" ]]; then
        sysctl -w vm.overcommit_memory=1
        # Make persistent
        grep -q "vm.overcommit_memory" /etc/sysctl.d/99-spark.conf 2>/dev/null || \
            echo "vm.overcommit_memory=1" >> /etc/sysctl.d/99-spark.conf
        _green "  Set vm.overcommit_memory=1 (persistent)."
    else
        echo "  Run with --apply to change."
    fi
fi
echo ""

# Summary
echo "=== Summary ==="
if [[ "$MODE" == "--check" ]]; then
    echo "This was a dry run. To apply changes:"
    echo "  sudo spark-harden --apply"
else
    echo "Hardening applied. Reboot to verify persistence."
fi
