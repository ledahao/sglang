#!/usr/bin/env bash
set -euo pipefail
source "$HOME/pd/activate-sglang.sh"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.93}"
MAX_RUNNING="${MAX_RUNNING:-4}"
CTX="${CTX:-4096}"
LOG_DIR="$HOME/pd/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)

if [[ -d /data/models/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 ]]; then
  MODEL=/data/models/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4
  ROLE=prefill-host-109
elif [[ -d /data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4 ]]; then
  MODEL=/data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4
  ROLE=decode-host-107
else
  echo "model path not found" >&2
  exit 1
fi

LOG="$LOG_DIR/smoke-single-${ROLE}-${TS}.log"
echo "MODEL=$MODEL ROLE=$ROLE PORT=$PORT MEM=$MEM_FRACTION MAX=$MAX_RUNNING CTX=$CTX LOG=$LOG" | tee "$LOG"

# Ensure host-table path stays off (OOM'd 247GB box with 2x ~50GB tables).
unset SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE || true
export SGLANG_ENABLE_JIT_DEEPGEMM=1

NUMA=( )
if command -v numactl >/dev/null 2>&1; then
  NUMA=(numactl --interleave=all)
fi

"${NUMA[@]}" sglang serve \
  --model-path "$MODEL" \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --mem-fraction-static "$MEM_FRACTION" \
  --max-running-requests "$MAX_RUNNING" \
  --context-length "$CTX" \
  --attention-backend dsv4 \
  --moe-runner-backend flashinfer_cutlass \
  --enable-decoder-swa-bounded-replay \
  --cuda-graph-max-bs-decode 16 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --host 0.0.0.0 \
  --port "$PORT" \
  >>"$LOG" 2>&1
