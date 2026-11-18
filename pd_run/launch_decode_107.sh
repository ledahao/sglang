#!/usr/bin/env bash
# Load DeepSeek-V4.1-Flash NVFP4 on gpu107 and serve decode.
# Matches the 2026-11-18 process: attn-dp 1, mem 0.82, chunk 8192,
# DSpark block 3, PCIe IPC allreduce, mooncake RDMA on mlx5_0.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/activate-sglang.sh"

export PYTHONPATH="/home/aaron/pd/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_RUST_BUILD_MODE=never
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS=0
export SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_DEVICE="${MOONCAKE_DEVICE:-mlx5_0}"
export MC_GID_INDEX=3
export MC_RDMA_DATA_DIRECT=0
export MC_MAX_MR_SIZE=134217728
export WITH_NVIDIA_PEERMEM=1
export MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:-192.168.100.107}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-192.168.100.107}"
export HOST_IP="${HOST_IP:-$SGLANG_HOST_IP}"
unset SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE || true

LOG_DIR="${LOG_DIR:-$HOME/pd/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pd-decode-107-pcieipc-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG"
exec "$HOME/pd/.venv-sglang/bin/python" "$HOME/pd/.venv-sglang/bin/sglang" serve \
  --model-path /data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4 \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --attn-dp-size 1 \
  --enable-decoder-swa-bounded-replay \
  --mem-fraction-static 0.82 \
  --max-running-requests 64 \
  --context-length 1M \
  --chunked-prefill-size 8192 \
  --swa-full-tokens-ratio 0.1 \
  --attention-backend dsv4 \
  --moe-runner-backend flashinfer_cutlass \
  --cuda-graph-max-bs-decode 64 \
  --schedule-conservativeness 0.3 \
  --num-continuous-decode-steps 1 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4 \
  --speculative-attention-mode decode \
  --speculative-dspark-block-size 3 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --host 0.0.0.0 \
  --port 30001 \
  >"$LOG" 2>&1
