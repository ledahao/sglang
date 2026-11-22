#!/bin/bash
# Decode entry for the 2026-11-22 record-22 deployment.
# Sources decode-env.sh next to this file, then the same serve command
# /tmp/start_decode_pcie.sh used when the 04:53 process started.
set -a
source "$(dirname "$0")/decode-env.sh"
set +a
LOG=/home/aaron/pd/logs/pd-decode-107-pcieipc-$(date +%Y%m%d-%H%M%S).log
echo "$LOG" > /tmp/decode-pcieipc.logpath
exec /home/aaron/pd/.venv-sglang/bin/python /home/aaron/pd/.venv-sglang/bin/sglang serve \
  --model-path /data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4 \
  --trust-remote-code --tp-size 8 --ep-size 8 --attn-dp-size 1 \
  --enable-decoder-swa-bounded-replay --mem-fraction-static 0.82 \
  --max-running-requests 64 --context-length 1M --chunked-prefill-size 8192 \
  --swa-full-tokens-ratio 0.1 --attention-backend dsv4 \
  --moe-runner-backend flashinfer_cutlass --cuda-graph-max-bs-decode 64 \
  --schedule-conservativeness 0.3 --num-continuous-decode-steps 1 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4 \
  --speculative-attention-mode decode --speculative-dspark-block-size 3 \
  --reasoning-parser auto --tool-call-parser auto \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 --host 0.0.0.0 --port 30001 \
  > "$LOG" 2>&1
