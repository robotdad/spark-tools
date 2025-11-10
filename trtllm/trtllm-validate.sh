#!/usr/bin/env bash
set -euo pipefail

# Host naming:
# - PRIMARY_HOST: this node (where you run the script), or override via SPARK_PRIMARY_HOST
# - SECONDARY_HOST: the other Spark node, override via SPARK_SECONDARY_HOST (defaults to 'dyad')
PRIMARY_HOST="${SPARK_PRIMARY_HOST:-$(hostname)}"
SECONDARY_HOST="${SPARK_SECONDARY_HOST:-dyad}"
echo "Primary host:   ${PRIMARY_HOST}"
echo "Secondary host: ${SECONDARY_HOST}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL [PORT]" >&2
  echo "  MODEL - Hugging Face model id (e.g. nvidia/Qwen3-14B-FP4)" >&2
  echo "  PORT  - HTTP port (default: 8355)" >&2
  exit 1
fi

MODEL="$1"
PORT="${2:-8355}"

echo "=== TRT-LLM validation ==="
echo "Model: $MODEL"
echo "Port:  $PORT"
echo

# ---------------------------------------------------------
# 1. Check that the TRT-LLM container exists (sanity check)
# ---------------------------------------------------------
if ! CID="$("${SCRIPT_DIR}/trtllm-container.sh" 2>/dev/null)"; then
  echo "ERROR: Could not determine TRT-LLM container ID on ${PRIMARY_HOST}." >&2
  exit 1
fi

if [[ -z "$CID" ]]; then
  echo "ERROR: No TRT-LLM container found on ${PRIMARY_HOST}." >&2
  exit 1
fi

echo "Local container on ${PRIMARY_HOST}: $CID"
echo

# ---------------------------------------------------------
# 2. Show GPU memory usage on PRIMARY_HOST and SECONDARY_HOST
#    Use full nvidia-smi so you can see per-process memory.
# ---------------------------------------------------------
echo "=== GPU memory usage (host view) ==="
echo

echo "--- ${PRIMARY_HOST} ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "nvidia-smi failed on ${PRIMARY_HOST}."
else
  echo "nvidia-smi not found on ${PRIMARY_HOST}."
fi

echo
echo "--- ${SECONDARY_HOST} ---"
if ssh "${SECONDARY_HOST}" 'command -v nvidia-smi >/dev/null 2>&1'; then
  ssh "${SECONDARY_HOST}" 'nvidia-smi' || echo "nvidia-smi failed on secondary host."
else
  echo "nvidia-smi not found or ssh to ${SECONDARY_HOST} failed."
fi

echo

# ---------------------------------------------------------
# 3. Send a simple health-check request from the HOST
#    (same style as your working curl)
# ---------------------------------------------------------
REQUEST_JSON=$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [
    { "role": "user", "content": "health check from trtllm-validate.sh" }
  ],
  "max_tokens": 16
}
EOF
)

echo "=== HTTP endpoint check ==="
echo "Checking http://localhost:$PORT/v1/chat/completions..."
echo

HTTP_ERR_FILE="$(mktemp)"

set +e
HTTP_RESPONSE=$(curl -sS -m 30 \
  -w "\nHTTP_CODE:%{http_code}\n" \
  -H "Content-Type: application/json" \
  -d "$REQUEST_JSON" \
  "http://localhost:$PORT/v1/chat/completions" \
  2>"$HTTP_ERR_FILE")
CURL_STATUS=$?
set -e

if [[ $CURL_STATUS -ne 0 ]]; then
  echo "❌ HTTP request failed (curl exit code: $CURL_STATUS)"
  echo "Error output:"
  cat "$HTTP_ERR_FILE"
  rm -f "$HTTP_ERR_FILE"
  exit 1
fi
rm -f "$HTTP_ERR_FILE"

# Split body vs HTTP code
HTTP_CODE="$(echo "$HTTP_RESPONSE" | awk -F: '/^HTTP_CODE:/ {print $2}' | tr -d ' \r')"
RESPONSE_BODY="$(echo "$HTTP_RESPONSE" | sed '/^HTTP_CODE:/d')"

if [[ -z "$HTTP_CODE" ]]; then
  HTTP_CODE="unknown"
fi

# Treat only 2xx as success; API should return 200
if [[ ! "$HTTP_CODE" =~ ^2[0-9][0-9]$ ]]; then
  echo "❌ Server responded with non-success HTTP code: $HTTP_CODE"
  echo "Response body:"
  echo "$RESPONSE_BODY"
  exit 1
fi

# Look for "choices" and show a small snippet
if echo "$RESPONSE_BODY" | grep -q '"choices"'; then
  ONE_LINE=$(echo "$RESPONSE_BODY" | tr '\n' ' ')
  SNIPPET=$(echo "$ONE_LINE" \
    | sed -n 's/.*"content":[[:space:]]*"\([^"]*\)".*/\1/p' \
    | head -c 160)

  echo "✅ Endpoint healthy (HTTP $HTTP_CODE)"
  if [[ -n "$SNIPPET" ]]; then
    echo "Sample content: \"$SNIPPET\"..."
  else
    echo "Response contained 'choices', but could not extract a content snippet."
  fi
else
  echo "⚠️  Server responded with HTTP $HTTP_CODE but no 'choices' field found."
  echo "Raw response:"
  echo "$RESPONSE_BODY"
fi

echo
echo "Validation complete."