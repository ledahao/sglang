#!/usr/bin/env bash
# 在 107 上跑 PD 适配版 ardelva bench。
# 用法：
#   ~/pd/bench/run_pd_bench.sh --smoke
#   ~/pd/bench/run_pd_bench.sh --out ~/pd/bench/runs/mat1
#   ~/pd/bench/run_pd_bench.sh --matrix '3000:1000:0:1,8,32,64;60000:1000:95:1,8,32'
set -euo pipefail
ROOT="${HOME}/pd"
# shellcheck disable=SC1091
source "${ROOT}/activate-sglang.sh"

has_out=0
for arg in "$@"; do
  if [[ "$arg" == "--out" || "$arg" == --out=* ]]; then
    has_out=1
    break
  fi
done
extra=()
if [[ "$has_out" -eq 0 ]]; then
  extra=(--out "${ROOT}/bench/runs/$(date +%Y%m%d-%H%M%S)")
fi
exec python "${ROOT}/bench/ardelva_bench_pd.py" "${extra[@]}" "$@"
