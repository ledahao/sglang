#!/usr/bin/env bash
# Decode (D) on 107 — rail0 RoCE
# Speculative (DSpark) blocked on SM120 until FlashInfer ships TRTLLM batched-GEMM cubins.
# Enable with ENABLE_SPEC=1 TRANSFER_BACKEND=mooncake when cubins exist.
set -euo pipefail
source "$HOME/pd/activate-sglang.sh"

# Edit and commit /home/aaron/pd/sglang (branch host-v0523-sm120-engram,
# origin https://github.com/ledahao/sglang.git). The next start of this
# script imports that worktree. A server that is already running keeps the
# site-packages files it mapped at startup; this script does not rewrite them.
export PYTHONPATH="/home/aaron/pd/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
# Source checkouts cargo-build Rust extensions unless this is never.
# never loads the wheel *.so copied into the worktree (gitignored).
export SGLANG_RUST_BUILD_MODE="${SGLANG_RUST_BUILD_MODE:-never}"
# Do not run while prefill/decode is up: pip deletes site-packages/sglang
# and can change imports in the live process. After that process is stopped,
# run once if the wheel copy of the Python package should be retired:
#   "$HOME/pd/.venv-sglang/bin/python" -m pip install --no-deps --no-build-isolation -e "$HOME/pd/sglang/python"
MODEL="${MODEL:-/data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4}"
PORT="${PORT:-30001}"
MEM_FRACTION="${MEM_FRACTION:-0.72}"
MAX_RUNNING="${MAX_RUNNING:-64}"
CTX="${CTX:-1M}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-4096}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-64}"
SCHEDULE_CONSERVATIVENESS="${SCHEDULE_CONSERVATIVENESS:-0.3}"
NUM_CONTINUOUS_DECODE_STEPS="${NUM_CONTINUOUS_DECODE_STEPS:-1}"
SWA_FULL_RATIO="${SWA_FULL_RATIO:-0.1}"
IB_DEV="${IB_DEV:-mlx5_0}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-mooncake}"
ENABLE_SPEC="${ENABLE_SPEC:-1}"
DISABLE_DECODE_CUDA_GRAPH="${DISABLE_DECODE_CUDA_GRAPH:-0}"
SPEC_ALGO="${SPEC_ALGO:-DSPARK}"
SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-$MODEL}"
SPEC_DSPARK_BLOCK="${SPEC_DSPARK_BLOCK:-3}"
ATTN_DP_SIZE="${ATTN_DP_SIZE:-8}"
LOG_DIR="$HOME/pd/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/pd-decode-107-${TS}.log"
unset SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE || true
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
# expandable_segments breaks nvidia-peermem ibv_reg_mr (EFAULT); force off for RDMA.
if [[ "$TRANSFER_BACKEND" == "mooncake" || "$TRANSFER_BACKEND" == "nixl" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
else
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi
export MOONCAKE_DEVICE="${MOONCAKE_DEVICE:-$IB_DEV}"
export MC_GID_INDEX="${MC_GID_INDEX:-3}"
export MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:-192.168.100.107}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-192.168.100.107}"
export HOST_IP="${HOST_IP:-$SGLANG_HOST_IP}"

if [[ "$TRANSFER_BACKEND" == "mooncake_tcp" ]]; then
  export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
  export MC_FORCE_TCP="${MC_FORCE_TCP:-1}"
elif [[ "$TRANSFER_BACKEND" == "mooncake" ]]; then
  # Open NVIDIA KMD: DMA-BUF MR path (no nvidia-peermem needed).
  # Legacy ibv_reg_mr without peermem → ENOMEM on ~1GB KV regs.
  unset MC_FORCE_TCP || true
  export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}"
  export WITH_NVIDIA_PEERMEM="${WITH_NVIDIA_PEERMEM:-1}"
  export MC_RDMA_DATA_DIRECT="${MC_RDMA_DATA_DIRECT:-0}"
  # 109 GPUs have only 256MiB BAR1; chunk GPU MRs for peermem/GPUDirect.
  export MC_MAX_MR_SIZE="${MC_MAX_MR_SIZE:-134217728}"  # 128 MiB
elif [[ "$TRANSFER_BACKEND" == "nixl" ]]; then
  unset MC_FORCE_TCP || true
  unset MOONCAKE_PROTOCOL || true
else
  unset MC_FORCE_TCP || true
  export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-rdma}"
fi

SPEC_ARGS=()
if [[ "$ENABLE_SPEC" == "1" ]]; then
  SPEC_ARGS+=(
    --speculative-algorithm "$SPEC_ALGO"
    --speculative-draft-model-path "$SPEC_DRAFT_MODEL"
    --speculative-attention-mode decode
  )
  if [[ "$SPEC_ALGO" == "DSPARK" ]]; then
    SPEC_ARGS+=(--speculative-dspark-block-size "$SPEC_DSPARK_BLOCK")
  fi
fi

GRAPH_ARGS=(--cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS")
if [[ "$DISABLE_DECODE_CUDA_GRAPH" == "1" ]]; then
  GRAPH_ARGS+=(--cuda-graph-backend-decode=disabled)
fi

echo "D MODEL=$MODEL PORT=$PORT CTX=$CTX MEM=$MEM_FRACTION CG_BS=$CUDA_GRAPH_MAX_BS CONSERV=$SCHEDULE_CONSERVATIVENESS XFER=$TRANSFER_BACKEND SPEC=$ENABLE_SPEC/$SPEC_ALGO ATTN_DP=$ATTN_DP_SIZE IB=$IB_DEV LOG=$LOG" | tee "$LOG"
exec sglang serve \
  --model-path "$MODEL" \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --attn-dp-size "$ATTN_DP_SIZE" \
  --enable-dp-lm-head \
  --enable-decoder-swa-bounded-replay \
  --mem-fraction-static "$MEM_FRACTION" \
  --max-running-requests "$MAX_RUNNING" \
  --context-length "$CTX" \
  --chunked-prefill-size "$CHUNKED_PREFILL" \
  --swa-full-tokens-ratio "$SWA_FULL_RATIO" \
  --attention-backend dsv4 \
  --moe-runner-backend flashinfer_cutlass \
  "${GRAPH_ARGS[@]}" \
  --schedule-conservativeness "$SCHEDULE_CONSERVATIVENESS" \
  --num-continuous-decode-steps "$NUM_CONTINUOUS_DECODE_STEPS" \
  "${SPEC_ARGS[@]}" \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
  --disaggregation-ib-device "$IB_DEV" \
  --host 0.0.0.0 \
  --port "$PORT" \
  >>"$LOG" 2>&1
