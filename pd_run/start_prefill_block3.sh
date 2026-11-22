#!/bin/bash
# Prefill entry for the 2026-11-22 record-22 deployment.
# Same serve command as /tmp/start_prefill_block3.sh. Login-session
# variables from that file are omitted.
set -a
export SHELL=/bin/bash
export MOONCAKE_PROTOCOL=rdma
export SGLANG_DSV4_ENCODER_ONLY_PREFILL=1
export MC_MAX_MR_SIZE=134217728
export SGLANG_HOST_IP=192.168.100.109
export CHUNKED_PREFILL=4096
export HOME=/home/aaron
export MC_RDMA_DATA_DIRECT=0
export ATTN_DP_SIZE=1
export VIRTUAL_ENV=/home/aaron/pd/.venv-sglang
export SPEC_DSPARK_BLOCK=3
export WITH_NVIDIA_PEERMEM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTHONPATH=/home/aaron/pd/sglang/python
export MOONCAKE_DEVICE=rocep37s0f0
export SGLANG_RUST_BUILD_MODE=never
export MC_GID_INDEX=3
export MOONCAKE_LOCAL_HOSTNAME=192.168.100.109
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/aaron/pd/.venv-sglang/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
export XDG_RUNTIME_DIR=/run/user/1003
export PATH=/home/aaron/pd/.venv-sglang/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export ENABLE_SPEC=1
export MEM_FRACTION=0.82
export HOST_IP=192.168.100.109
set +a
LOG=/home/aaron/pd/logs/pd-prefill-109-block3-$(date +%Y%m%d-%H%M%S).log
echo "$LOG" > /tmp/prefill-block3.logpath
exec /home/aaron/pd/.venv-sglang/bin/python /home/aaron/pd/.venv-sglang/bin/sglang serve \
  --model-path /data/models/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 \
  --trust-remote-code --tp-size 8 --ep-size 8 --attn-dp-size 1 \
  --enable-decoder-swa-bounded-replay --mem-fraction-static 0.82 \
  --max-running-requests 64 --context-length 1M --chunked-prefill-size 4096 \
  --dsv4-prefill-backend flashmla_sparse --schedule-conservativeness 0.3 \
  --swa-full-tokens-ratio 0.1 --attention-backend dsv4 \
  --moe-runner-backend flashinfer_cutlass --cuda-graph-max-bs-decode 64 \
  --disable-prefill-cuda-graph \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /data/models/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 \
  --speculative-attention-mode decode --speculative-dspark-block-size 3 \
  --reasoning-parser auto --tool-call-parser auto \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 --disaggregation-ib-device rocep37s0f0 \
  --host 0.0.0.0 --port 30000 \
  > "$LOG" 2>&1
