#!/usr/bin/env bash
# PD router — run on 107 or jump host; points at P@109 + D@107
set -euo pipefail
source "$HOME/pd/activate-sglang.sh"
PREFILL_URL="${PREFILL_URL:-http://10.230.0.109:30000}"
DECODE_URL="${DECODE_URL:-http://10.230.0.107:30001}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_DIR="$HOME/pd/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/pd-router-${TS}.log"
echo "ROUTER P=$PREFILL_URL D=$DECODE_URL :$PORT LOG=$LOG" | tee "$LOG"
# Prefer sglang_router if present
if python -c "import sglang_router" 2>/dev/null; then
  exec python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "$PREFILL_URL" \
    --decode "$DECODE_URL" \
    --host "$HOST" \
    --port "$PORT" \
    >>"$LOG" 2>&1
fi
# Fallback: sglang.srt.entrypoints might not have router; fail clearly
echo "sglang_router not installed; uv pip install sglang-router" | tee -a "$LOG"
exit 1
