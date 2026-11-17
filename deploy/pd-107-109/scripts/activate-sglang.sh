#!/usr/bin/env bash
# shellcheck disable=SC1091
source "$HOME/pd/.venv-sglang/bin/activate"
# mooncake-transfer-engine wheels link libcudart.so.12
_CU12_LIB="$HOME/pd/.venv-sglang/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
if [[ -d "$_CU12_LIB" ]]; then
  export LD_LIBRARY_PATH="$_CU12_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# system CUDA 13 runtime (torch/sglang JIT)
if [[ -d /usr/local/cuda/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
echo "SGLang env active: $(python -c 'import sglang,torch; print(sglang.__version__, torch.__version__)')"
