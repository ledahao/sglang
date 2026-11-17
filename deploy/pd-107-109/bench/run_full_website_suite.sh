#!/usr/bin/env bash
# Full ardelva-style suite on the current PD router.
# Random matrix, real text, think-length output, multi-turn, and the 40-question check.
# Concurrency goes to 64. stop-ttft is off so long prompts are not dropped.
set -u
set -o pipefail
ROOT="${HOME}/pd"
source "${ROOT}/activate-sglang.sh"
export BENCH_RANGE_RATIO=0
OUT="${ROOT}/bench/runs/website-suite-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/suite.log"
PY=python
echo "OUT=$OUT" | tee "$LOG"

wait_up() {
  local i
  for i in $(seq 1 90); do
    if curl -sf -m 5 http://127.0.0.1:30001/health >/dev/null \
      && curl -sf -m 5 http://10.230.0.109:30000/health >/dev/null \
      && curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null; then
      echo "healthy $(date '+%F %T')" | tee -a "$LOG"
      return 0
    fi
    echo "waiting for PD health $i $(date '+%T')" | tee -a "$LOG"
    sleep 20
  done
  echo "UNHEALTHY $(date '+%F %T')" | tee -a "$LOG"
  return 1
}

run_bench() {
  local name="$1"
  shift
  echo "== $name $(date '+%F %T')" | tee -a "$LOG"
  if ! wait_up; then
    echo "FAIL $name unhealthy" | tee -a "$LOG"
    return 0
  fi
  if ! "$PY" "${ROOT}/bench/ardelva_bench_pd.py" \
      --out "$OUT/$name" \
      --stop-ttft 0 \
      --min-requests 32 \
      --req-per-conc 4 \
      "$@" 2>&1 | tee -a "$LOG"; then
    echo "FAIL $name" | tee -a "$LOG"
  fi
}

{
  echo "== accuracy $(date '+%F %T')"
  "$PY" "${ROOT}/bench/run_accuracy_pd.py" --out "$OUT" || echo "FAIL accuracy"
  echo "== build realtext $(date '+%F %T')"
  "$PY" "${ROOT}/bench/build_realtext_pd.py" || echo "FAIL realtext-build"
} | tee -a "$LOG"

run_bench random-3k-hit0 --matrix '3000:1000:0:1,8,32,64' --warmup 8
run_bench random-3k-cold --matrix '3000:1000:cold:1,8,32,64' --warmup 8
run_bench random-60k-hit95 --matrix '60000:1000:95:1,8,32,64' --warmup 1
run_bench random-100k-hit0 --matrix '100000:1000:0:1,8,32,64' --warmup 1
run_bench random-100k-cold --matrix '100000:1000:cold:1,8,32,64' --warmup 1

for len in 3k 10k 32k; do
  run_bench "realtext-${len}" \
    --dataset "custom:${ROOT}/bench/data/realtext/realtext_${len}.jsonl" \
    --conc 1,8,32,64 \
    --output-len 1000 \
    --extra "--sharegpt-output-len 1000" \
    --warmup 1
done

run_bench think-10k-4k \
  --dataset "custom:${ROOT}/bench/data/realtext/realtext_10k.jsonl" \
  --conc 1,8,32,64 \
  --output-len 4000 \
  --extra "--sharegpt-output-len 4000" \
  --warmup 1

echo "== multiturn $(date '+%F %T')" | tee -a "$LOG"
if wait_up; then
  "$PY" "${ROOT}/bench/run_multiturn_pd.py" --out "$OUT/multiturn" --conc 1,8,32,64 \
    2>&1 | tee -a "$LOG" || echo "FAIL multiturn" | tee -a "$LOG"
fi

"$PY" - << PY | tee -a "$LOG"
import csv
from pathlib import Path
out = Path("$OUT")
rows = []
for p in sorted(out.glob("*/ardelva_points.csv")):
    with p.open() as f:
        for row in csv.DictReader(f):
            row["stage"] = p.parent.name
            rows.append(row)
mt = out / "multiturn/multiturn_points.json"
if mt.exists():
    import json
    for row in json.loads(mt.read_text()):
        row["stage"] = "multiturn"
        rows.append(row)
cols = []
for row in rows:
    for k in row:
        if k not in cols:
            cols.append(k)
dest = out / "website_suite.csv"
with dest.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"merged {len(rows)} rows -> {dest}")
PY
echo "SUITE_DONE $OUT" | tee -a "$LOG"
