#!/usr/bin/env python3
"""ardelva-qc-v1 accuracy check against the PD router. Same 40 items as the site script."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

QC_SET = "ardelva-qc-v1"
QC_TAIL = (
    "\n\nThink as needed, then give only the final answer on the last line "
    "in the form: ANSWER: <answer>"
)


def qc_items():
    rnd = random.Random(20261103)
    items = []
    for _ in range(12):
        x, y, z = rnd.randint(1000, 9999), rnd.randint(12, 98), rnd.randint(100, 9999)
        items.append(("num", f"Compute {x} * {y} + {z}.", str(x * y + z)))
    for _ in range(8):
        b, n, out = rnd.randint(40, 400), rnd.randint(6, 48), rnd.randint(50, 900)
        items.append(
            (
                "num",
                f"A warehouse has {b} boxes and each box holds {n} parts. "
                f"{out} parts are shipped out. How many parts remain in the warehouse?",
                str(b * n - out),
            )
        )
    for _ in range(4):
        xs = [rnd.randint(1, 30) for _ in range(8)]
        items.append(
            (
                "num",
                "What does this Python code print?\n"
                f"```python\nxs = {xs}\nprint(sum(x * x for x in xs if x % 2 == 0))\n```",
                str(sum(x * x for x in xs if x % 2 == 0)),
            )
        )
    for _ in range(4):
        n = rnd.randint(8, 20)
        r = 1
        for i in range(1, n + 1):
            r = r * i % 1000
        items.append(
            (
                "num",
                "What does this Python code print?\n"
                f"```python\nr = 1\nfor i in range(1, {n} + 1):\n"
                f"    r = r * i % 1000\nprint(r)\n```",
                str(r),
            )
        )
    for _ in range(6):
        xs = rnd.sample(range(-60, 200), 9)
        items.append(
            (
                "list",
                "Sort these integers in ascending order and give them comma-separated: "
                + ", ".join(map(str, xs))
                + ".",
                ",".join(map(str, sorted(xs))),
            )
        )
    syl = ["ra", "ten", "mo", "lis", "ver", "qua", "dor", "sel"]
    for _ in range(6):
        w = "".join(rnd.choice(syl) for _ in range(rnd.randint(3, 5)))
        ch = rnd.choice(sorted(set(w)))
        items.append(
            (
                "num",
                f'How many times does the letter "{ch}" appear in the string "{w}"?',
                str(w.count(ch)),
            )
        )
    return items


def qc_parse(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    found = re.findall(r"ANSWER:\s*(.+)", text)
    return found[-1].strip().strip("`*. ") if found else None


def qc_norm(kind, value):
    if value is None:
        return None
    if kind == "num":
        found = re.findall(r"-?\d[\d,]*", value)
        return found[-1].replace(",", "") if found else value
    if kind == "list":
        return ",".join(re.findall(r"-?\d+", value))
    return value


def qc_ask(base_url, model, prompt, max_tokens):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        method="POST",
        headers={"content-type": "application/json"},
    )
    if os.environ.get("OPENAI_API_KEY"):
        req.add_header("authorization", "Bearer " + os.environ["OPENAI_API_KEY"])
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            msg = json.loads(resp.read().decode())["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
    except Exception as exc:
        return f"ERROR {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--model",
        default="/data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4",
    )
    ap.add_argument("--par", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    items = qc_items()
    print(
        f"== 精度检查 {QC_SET}：{len(items)} 题，temperature 0，{a.par} 路并行",
        flush=True,
    )
    t0 = time.time()
    with ThreadPoolExecutor(a.par) as ex:
        texts = list(
            ex.map(lambda it: qc_ask(a.base_url, a.model, it[1] + QC_TAIL, a.max_tokens), items)
        )
    answers = [qc_norm(k, qc_parse(t)) for (k, _, _), t in zip(items, texts)]
    ok = [ans == qc_norm(k, gold) for (k, _, gold), ans in zip(items, answers)]
    res = {
        "acc_set": QC_SET,
        "acc_n": len(items),
        "acc_pct": round(100 * sum(ok) / len(items), 1),
        "model": a.model,
        "date": date.today().isoformat(),
    }
    wrong = [i + 1 for i, good in enumerate(ok) if not good]
    (outdir / "ardelva_accuracy.json").write_text(
        json.dumps(
            {
                **res,
                "detail": [
                    {"q": q[:160], "gold": g, "got": ans, "ok": good, "raw": t[:500]}
                    for (_, q, g), ans, good, t in zip(items, answers, ok, texts)
                ],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(
        f"    → 答对 {res['acc_pct']}%"
        + (f"；错题 {wrong}" if wrong else "")
        + f" · 用时 {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
