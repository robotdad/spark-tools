#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MODEL_NAME PORT [RUNS] [MAX_TOKENS]" >&2
  exit 1
fi

MODEL="$1"
PORT="$2"
RUNS="${3:-3}"
MAX_TOKENS="${4:-256}"

ENDPOINT="http://localhost:${PORT}/v1/chat/completions"

echo "=== TRT-LLM benchmark ==="
echo "Model:      $MODEL"
echo "Port:       $PORT"
echo "Runs:       $RUNS"
echo "Max tokens: $MAX_TOKENS"
echo

total_time=0
total_tokens=0
effective_runs=0

for i in $(seq 1 "$RUNS"); do
  echo "Run ${i}/${RUNS} ..."

  start_ts="$(date +%s.%N)"

  RESP="$(curl -sS "$ENDPOINT" \
    -H "Content-Type: application/json" \
    -d "$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [
    {
      "role": "user",
      "content": "Benchmark run $i for $MODEL on DGX Spark. Please reply with a short sentence."
    }
  ],
  "max_tokens": $MAX_TOKENS,
  "stream": false
}
EOF
)" 2>/dev/null || true)"

  end_ts="$(date +%s.%N)"

  if [[ -z "${RESP}" ]]; then
    echo "  ❌ Empty or failed response from server"
    echo
    continue
  fi

  # Latency in seconds (as float) via awk
  latency="$(awk -v s="$start_ts" -v e="$end_ts" 'BEGIN { print e - s }')"

  # Extract completion_tokens from JSON using grep/sed
  tokens="$(printf '%s\n' "$RESP" \
    | grep -o '"completion_tokens":[0-9]\+' \
    | head -n1 \
    | sed 's/[^0-9]//g')"

  if [[ -z "${tokens}" ]]; then
    tokens=0
  fi

  # Tokens/sec, avoid divide-by-zero
  if [[ "$tokens" -gt 0 ]]; then
    tok_per_s="$(awk -v t="$tokens" -v l="$latency" 'BEGIN { if (l > 0) print t / l; else print 0 }')"
  else
    tok_per_s="0.0"
  fi

  printf "  Latency: %.3fs\n" "$latency"
  echo   "  Tokens:  $tokens"
  printf "  Tok/s:   %.2f\n" "$tok_per_s"
  echo

  # Accumulate totals (also with awk for floats)
  total_time="$(awk -v tt="$total_time" -v l="$latency" 'BEGIN { print tt + l }')"
  total_tokens=$(( total_tokens + tokens ))
  effective_runs=$(( effective_runs + 1 ))
done

echo "=== Summary ==="
echo "Effective runs: $effective_runs"
printf "Total time:     %.3f s\n" "$total_time"
echo "Total tokens:   $total_tokens"

if [[ "$effective_runs" -gt 0 ]]; then
  avg_lat="$(awk -v tt="$total_time" -v n="$effective_runs" 'BEGIN { if (n > 0) print tt / n; else print 0 }')"

  if [[ "$total_tokens" -gt 0 && "$(echo "$total_time > 0" | bc 2>/dev/null || echo 0)" -eq 1 ]]; then
    avg_tokps="$(awk -v t="$total_tokens" -v tt="$total_time" 'BEGIN { if (tt > 0) print t / tt; else print 0 }')"
  else
    avg_tokps="0.0"
  fi

  printf "Avg latency:    %.3f s\n" "$avg_lat"
  printf "Avg tokens/s:   %.2f\n" "$avg_tokps"
else
  echo "No successful runs."
fi
