#!/usr/bin/env python3
"""Website-style multi-turn agent load: 60K start, +5K each turn, 5 turns.

Uses the local vLLM multi-turn client (vLLM import stubbed) against the PD router.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

TOOL = Path.home() / "pd/bench/multi_turn/benchmark_serving_multi_turn.py"
MODEL = "/data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4"
BASE = "http://127.0.0.1:8000"


def mt_spec():
    start, step, rounds, out = 60000, 5000, 5, 500
    common = min(15000, start // 4)
    user = step - out
    prefix = start - max(common, user)
    return dict(
        start=start,
        step=step,
        rounds=rounds,
        out=out,
        common=common,
        user=user,
        prefix=prefix,
        label=f"multiturn:{start}+{step}x{rounds}",
    )


def mt_nominal_hit(sp):
    inp = [sp["start"] + k * sp["step"] for k in range(sp["rounds"])]
    hit = [sp["common"]] + [x - sp["user"] for x in inp[1:]]
    return round(100 * sum(hit) / sum(inp), 1)


def corpus(path: Path, tokens: int, seed: str) -> None:
    rnd = random.Random(seed)
    syl = [
        "ka", "lo", "mi", "ten", "ra", "sul", "ve", "dor", "an", "qui",
        "pe", "zo", "ri", "mon", "tal", "es", "vin", "cor", "ul", "ba",
    ]
    words = ["".join(rnd.choice(syl) for _ in range(rnd.randint(1, 3))) for _ in range(4000)]
    written = 0
    with path.open("w") as f:
        while written < tokens * 9:
            line = " ".join(rnd.choice(words) for _ in range(40)) + ".\n"
            f.write(line)
            written += len(line)


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def one(sp, c: int, outdir: Path) -> dict | None:
    nconv = max(8, 2 * c)
    corpus_path = outdir / f"mt_corpus_c{c}.txt"
    corpus(
        corpus_path,
        sp["common"] + nconv * (sp["prefix"] + 50) + sp["rounds"] * 2 * sp["user"] + 20000,
        f"{time.time_ns()}-{c}",
    )
    conf = outdir / f"mt_c{c}.json"
    d = lambda v: {"distribution": "constant", "value": v}
    conf.write_text(
        json.dumps(
            {
                "filetype": "generate_conversations",
                "num_conversations": nconv,
                "text_files": [str(corpus_path)],
                "prompt_input": {
                    "num_turns": d(2 * sp["rounds"]),
                    "common_prefix_num_tokens": d(sp["common"]),
                    "prefix_num_tokens": d(sp["prefix"]),
                    "num_tokens": d(sp["user"]),
                },
                "prompt_output": {"num_tokens": d(sp["out"])},
            },
            indent=2,
        )
    )
    stats = outdir / f"mt_c{c}_stats.json"
    cmd = [
        sys.executable,
        str(TOOL),
        "-m",
        MODEL,
        "--served-model-name",
        MODEL,
        "-u",
        BASE,
        "-i",
        str(conf),
        "-p",
        str(c),
        "-k",
        str(c),
        "--seed",
        "1",
        "--limit-min-tokens",
        str(sp["out"]),
        "--limit-max-tokens",
        str(sp["out"]),
        "--request-timeout-sec",
        "3600",
        "--stats-json-output",
        str(stats),
    ]
    print(f"  测量 {sp['label']} C={c} 会话={nconv}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(TOOL.parent))
    if proc.returncode != 0 or not stats.exists():
        print(proc.stdout[-2000:], proc.stderr[-3000:], sep="\n", file=sys.stderr)
        print(f"  ✗ 多轮 C={c} 失败 rc={proc.returncode}", flush=True)
        return None
    reqs = [r for r in json.loads(stats.read_text()) if r.get("ttft_ms", -1) > 0]
    if not reqs:
        print(f"  ✗ 多轮 C={c} 没有成功请求", flush=True)
        return None
    span = (
        max(r["start_time_ms"] + r["latency_ms"] for r in reqs)
        - min(r["start_time_ms"] for r in reqs)
    ) / 1000
    outs = sum(r["output_num_tokens"] for r in reqs)
    ins = sum(r["input_num_tokens"] for r in reqs)
    tpot = [
        (r["latency_ms"] - r["ttft_ms"]) / max(1, r["output_num_tokens"] - 1) for r in reqs
    ]
    ttft = [r["ttft_ms"] for r in reqs]
    first = [r["ttft_ms"] for r in reqs if r["input_num_turns"] <= 1]
    later = [r["ttft_ms"] for r in reqs if r["input_num_turns"] > 1]
    tpot_p50 = pct(tpot, 50) or 0
    row = {
        "isl": round(ins / len(reqs)),
        "osl": round(outs / len(reqs)),
        "hit": mt_nominal_hit(sp),
        "c": c,
        "nreq": len(reqs),
        "out_tps": round(outs / span, 2) if span else "",
        "ttft_p50": round(pct(ttft, 50), 2),
        "tpot_p50": round(tpot_p50, 3),
        "ttft_p99": round(pct(ttft, 99), 2),
        "tpot_p99": round(pct(tpot, 99), 3),
        "total_tps": round((ins + outs) / span, 2) if span else "",
        "success": 100.0,
        "accept": "",
        "ttft_t1": round(statistics.median(first), 1) if first else "",
        "ttft_tn": round(statistics.median(later), 1) if later else "",
        "ds": sp["label"],
        "in_tps": round(ins / span, 1) if span else "",
        "after_ttft_tps": round(1000.0 / tpot_p50, 2) if tpot_p50 else "",
        "ttft_p95": round(pct(ttft, 95), 2),
        "tpot_p95": round(pct(tpot, 95), 3),
        "active_decode_est": round((outs / span) * tpot_p50 / 1000.0, 2) if span and tpot_p50 else "",
    }
    print(
        f"    → out {row['out_tps']} tok/s · in {row['in_tps']} tok/s · "
        f"首字后单流 {row['after_ttft_tps']} · 平均输入 {row['isl']} · "
        f"TTFT 首轮 {row['ttft_t1']} / 后续 {row['ttft_tn']} ms · "
        f"TTFT p50/p95 {row['ttft_p50']}/{row['ttft_p95']} · "
        f"TPOT p50/p95 {row['tpot_p50']}/{row['tpot_p95']} · "
        f"用时 {time.time() - t0:.0f}s",
        flush=True,
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", default="1,8,32,64")
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    sp = mt_spec()
    rows = []
    print(f"== 多轮 {sp['label']} 理论命中 {mt_nominal_hit(sp)}%", flush=True)
    for c in [int(x) for x in a.conc.split(",") if x.strip()]:
        row = one(sp, c, outdir)
        if row:
            rows.append(row)
            (outdir / "multiturn_points.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=1)
            )
    if not rows:
        raise SystemExit("多轮没有成功的点")


if __name__ == "__main__":
    main()
