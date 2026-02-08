#!/usr/bin/env bash
set -euo pipefail

SECRET_DIR="$HOME/.config/vllm-proxy"
SECRET_FILE="$SECRET_DIR/secret.env"

echo "=== Rotating vLLM Proxy Secret ==="

# Create directory with secure permissions
mkdir -p "$SECRET_DIR"
chmod 700 "$SECRET_DIR"

# Generate new secret
NEW_SECRET=$(openssl rand -hex 32)

# Write secret file
echo "VLLM_PROXY_SECRET=${NEW_SECRET}" > "$SECRET_FILE"
chmod 600 "$SECRET_FILE"

echo "New secret written to: $SECRET_FILE"
echo ""
echo "Bearer token: ${NEW_SECRET}"
echo ""

# Restart proxy if it's running as a systemd service
if systemctl is-active spark-proxy.service &>/dev/null; then
    echo "Restarting proxy service..."
    sudo systemctl restart spark-proxy.service
    echo "Proxy restarted with new secret."
else
    echo "Proxy service not running. Start it manually or via systemd."
fi
