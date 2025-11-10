#!/usr/bin/env bash
set -euo pipefail

# Try the more specific service name first
CID="$(docker ps -q -f name=trtllm-multinode_trtllm)"

if [[ -z "${CID}" ]]; then
  # Fallback to just the stack prefix
  CID="$(docker ps -q -f name=trtllm-multinode || true)"
fi

if [[ -z "${CID}" ]]; then
  echo "Could not find TRT-LLM multi-node container (name contains 'trtllm-multinode')." >&2
  exit 1
fi

echo "${CID}"
