#!/usr/bin/env bash
# Load DeepSeek-V4.1-Flash NVFP4 on s109 and serve encoder-only prefill.
# Matches the 2026-11-18 process: attn-dp 1, mem 0.82, chunk 4096,
# DSpark block 3, mooncake RDMA on rocep37s0f0. Layers 21-39 are not built.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate-sglang.sh"

export PYTHONPATH="/home/aaron/pd/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_RUST_BUILD_MODE=never
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSV4_ENCODER_ONLY_PREFILL=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_DEVICE="${MOONCAKE_DEVICE:-rocep37s0f0}"
export MC_GID_INDEX=3
export MC_RDMA_DATA_DIRECT=0
export MC_MAX_MR_SIZE=134217728
export WITH_NVIDIA_PEERMEM=1
export MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:-192.168.100.109}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-192.168.100.109}"
export HOST_IP="${HOST_IP:-$SGLANG_HOST_IP}"
unset SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE || true

LOG_DIR="${LOG_DIR:-$HOME/pd/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pd-prefill-109-block3-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG"
exec "$HOME/pd/.venv-sglang/bin/python" "$HOME/pd/.venv-sglang/bin/sglang" serve \
  --model-path /data/models/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --attn-dp-size 1 \
  --enable-decoder-swa-bounded-replay \
  --mem-fraction-static 0.82 \
  --max-running-requests 64 \
  --context-length 1M \
  --chunked-prefill-size 4096 \
  --dsv4-prefill-backend flashmla_sparse \
  --schedule-conservativeness 0.3 \
  --swa-full-tokens-ratio 0.1 \
  --attention-backend dsv4 \
  --moe-runner-backend flashinfer_cutlass \
  --cuda-graph-max-bs-decode 64 \
  --disable-prefill-cuda-graph \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /data/models/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 \
  --speculative-attention-mode decode \
  --speculative-dspark-block-size 3 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device rocep37s0f0 \
  --host 0.0.0.0 \
  --port 30000 \
  >"$LOG" 2>&1
