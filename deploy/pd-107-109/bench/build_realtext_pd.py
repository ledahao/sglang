#!/usr/bin/env python3
"""Pack the local ShareGPT mini corpus into fixed-length realtext jsonl files.

Each prompt starts with a unique document id so the runs stay cold, then fills
the rest with real ShareGPT text up to the target token count.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

TOKENIZER = "/data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4"
SRC = Path.home() / "pd/bench/data/ShareGPT_local_mini.json"
OUT = Path.home() / "pd/bench/data/realtext"
N = 256
LENGTHS = {"3k": 3000, "10k": 10000, "32k": 32000}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    data = json.loads(SRC.read_text())
    pieces = []
    for item in data:
        for turn in item.get("conversations") or []:
            text = (turn.get("value") or turn.get("content") or "").strip()
            if text:
                pieces.append(text)
    ids = tok.encode("\n\n".join(pieces), add_special_tokens=False)
    if len(ids) < 1000:
        raise SystemExit(f"corpus too small: {len(ids)} tokens")
    tiled = ids * ((max(LENGTHS.values()) // len(ids)) + 2)
    OUT.mkdir(parents=True, exist_ok=True)
    stride = max(1, len(ids) // N)
    for name, ntok in LENGTHS.items():
        path = OUT / f"realtext_{name}.jsonl"
        lines = []
        for i in range(N):
            header = f"Document {i:04d}.\n"
            hid = tok.encode(header, add_special_tokens=False)
            body = tiled[(i * stride) % len(ids) : (i * stride) % len(ids) + ntok]
            full = (hid + body)[:ntok]
            text = tok.decode(full)
            lines.append(
                json.dumps(
                    {
                        "conversations": [
                            {"from": "human", "value": text},
                            {"from": "gpt", "value": "ok"},
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        path.write_text("\n".join(lines) + "\n")
        print(f"{name}: {N} prompts, target {ntok} tokens, corpus {len(ids)} -> {path}")


if __name__ == "__main__":
    main()
