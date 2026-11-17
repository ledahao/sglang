#!/usr/bin/env python3
"""Post website-suite-20261117-091436 to https://ardelva.ai as a verified record.

The site distinguishes official vs community by the verify key, not a body
flag. Header x-ardelva-key (env ARDELVA_KEY, else the mode-600 file
.ardelva-official.key) makes the stored record verified=true. The UI calls
that "Ardelva measured · verified". Re-running does not post a second copy:
the engine string contains the run id, and a record with that engine is left
in place.

Reads the suite CSV. Does not restart services or touch the running server
beyond reading /proc.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RUN_ID = "website-suite-20261117-091436"
SUITE = Path("/home/aaron/pd/bench/runs") / RUN_ID
CSV_PATH = SUITE / "website_suite.csv"
ACC_PATH = SUITE / "ardelva_accuracy.json"
KEY_FILE = Path("/home/aaron/pd/bench/.ardelva-official.key")
RECEIPT = SUITE / "ardelva_post.json"
CONFIG_OUT = SUITE / "ardelva_run_config.json"
SITE = "https://ardelva.ai"
UA = "Mozilla/5.0 (compatible; ardelva_bench/1.3; +https://ardelva.ai/tests/bench/)"

# Frozen identity of this suite. Changing it would publish a second record.
ENGINE = "SGLang 0.5.23 853ee5a6 site-pkg website-suite-20261117-091436"
TOOL = (
    "run_full_website_suite.sh + ardelva_bench_pd.py"
    " + run_multiturn_pd.py + run_accuracy_pd.py"
)
SUBMITTER = "Ardelva Lab"
DATE = "2026-11-17"

# Planner keys() for this topology. keycheck accepted this triple.
# P/D are 8-GPU dual-socket boxes, so effTpLink rewrites tp to pcie5-xgmi.
# side=gpu: host Engram table was unset and hicache was not attached.
MODEL_KEY = "deepseek/v4.1-flash-1017@nvfp4"
DEPLOY_KEY = (
    "split=pd-asym|P=tp8[L0-19]:prefill|D=tp8[L0-39]:decode"
    "|side=gpu|kv=fp8@hbm|spec=dspark"
)
HW_KEY = (
    "P=rtxpro5000-72g\u00d78(8/node;host:256G;tp:pcie5-xgmi;pp:pcie5-p2p;net:eth-200g)"
    "|D=rtxpro6000d-84g\u00d78(8/node;host:256G;tp:pcie5-xgmi;pp:pcie5-p2p;net:eth-200g)"
    "|P\u2192D=eth-200g"
)

CSV_COLS = [
    "isl", "osl", "hit", "c", "nreq", "out_tps", "ttft_p50", "tpot_p50",
    "ttft_p99", "tpot_p99", "total_tps", "success", "accept", "ttft_t1",
    "ttft_tn", "ds", "rate", "slo_ttft", "slo_tpot", "slo_pct", "good_tps",
    "goodput",
]
PREFILL_HOST = "192.168.100.109"  # gpu107 does not resolve the name s109
PREFILL_PID = 377491
DECODE_PID = 1067594

# Live flags this suite actually ran. A mismatch aborts the first post.
EXPECT_P = {
    "mem": "0.82",
    "chunk": "32768",
    "ib": "rocep37s0f0",
    "backend": "flashmla_sparse",
    "block": "3",
    "attn_dp": "8",
    "mode": "prefill",
}
EXPECT_D = {
    "mem": "0.82",
    "chunk": "8192",
    "ib": "mlx5_0",
    "block": "3",
    "attn_dp": "8",
    "mode": "decode",
}


def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def load_key() -> str:
    env = os.environ.get("ARDELVA_KEY", "").strip()
    if env:
        return env
    if not KEY_FILE.is_file():
        die(f"set ARDELVA_KEY or create {KEY_FILE} (mode 600)")
    mode = KEY_FILE.stat().st_mode & 0o777
    if mode != 0o600:
        die(f"{KEY_FILE} mode is {mode:o}, want 600")
    key = KEY_FILE.read_text().strip()
    if not key:
        die(f"{KEY_FILE} is empty")
    return key


def request(method: str, path: str, payload=None, key: str | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"user-agent": UA}
    if data is not None:
        headers["content-type"] = "application/json"
    if key:
        headers["x-ardelva-key"] = key
    req = urllib.request.Request(SITE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode()
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw[:500]}
        return exc.code, parsed


def ssh_json(host: str, code: str) -> dict:
    proc = subprocess.run(
        [
            "ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
            host, "python3", "-",
        ],
        input=code.encode(),
        capture_output=True,
        timeout=40,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")[-400:]
        die(f"ssh {host} failed: {err}")
    return json.loads(proc.stdout.decode())


CAPTURE = r"""
import json, os, re
prefer = int(os.environ["PREFER_PID"])
mode = os.environ["WANT_MODE"]
secret = ("KEY", "TOKEN", "SECRET", "PASS", "AUTH", "COOKIE")

def cmdline(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""

def serves(cmd):
    return "sglang" in cmd and "serve" in cmd and f"--disaggregation-mode {mode}" in cmd

pid = prefer if serves(cmdline(prefer)) else None
if pid is None:
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        cmd = cmdline(name)
        if serves(cmd):
            pid = int(name)
            break
out = {"pid": pid, "cmdline": "", "env": {}, "site_packages": False, "worktree": False, "version": ""}
if pid:
    out["cmdline"] = cmdline(pid).strip()
    raw = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
    keep = ("SGLANG_", "MEM_FRACTION", "CHUNKED_PREFILL", "SPEC_DSPARK_BLOCK",
            "ATTN_DP_SIZE", "TRANSFER_BACKEND", "MOONCAKE_", "MC_", "ENABLE_SPEC",
            "PYTORCH_CUDA_ALLOC_CONF", "WITH_NVIDIA_PEERMEM")
    for item in raw:
        if not item or b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        ks = k.decode()
        if any(s in ks.upper() for s in secret):
            continue
        if ks.startswith(keep) or ks in keep:
            out["env"][ks] = v.decode(errors="replace")
    try:
        maps = open(f"/proc/{pid}/maps", errors="replace").read()
    except OSError:
        maps = ""
    out["site_packages"] = "site-packages/sglang" in maps
    out["worktree"] = "/pd/sglang/python" in maps
    for meta in __import__("pathlib").Path("/home/aaron/pd/.venv-sglang/lib/python3.12/site-packages").glob("sglang-*.dist-info/METADATA"):
        for line in meta.read_text(errors="replace").splitlines():
            if line.startswith("Version:"):
                out["version"] = line.split(":", 1)[1].strip()
                break
print(json.dumps(out))
"""


def capture(host: str | None, prefer: int, mode: str) -> dict:
    code = f"import os\nos.environ['PREFER_PID']={prefer!s}\nos.environ['WANT_MODE']={mode!r}\n" + CAPTURE
    # The snippet above sets env inside the remote interpreter via the wrapper.
    wrapped = (
        "import os\n"
        f"os.environ['PREFER_PID'] = '{prefer}'\n"
        f"os.environ['WANT_MODE'] = '{mode}'\n"
        + CAPTURE
    )
    if host is None:
        proc = subprocess.run(
            [sys.executable, "-"],
            input=wrapped.encode(),
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0:
            die(proc.stderr.decode(errors="replace")[-400:])
        return json.loads(proc.stdout.decode())
    return ssh_json(host, wrapped)


def flag(cmd: str, name: str) -> str:
    m = re.search(rf"--{name}(?:=|\s+)(\S+)", cmd)
    return m.group(1) if m else ""


def script_default(text: str, var: str) -> str:
    m = re.search(rf'{var}="\${{{var}:-([^"]+)}}"', text)
    return m.group(1) if m else ""


def read_text(host: str | None, path: str) -> str:
    if host is None:
        return Path(path).read_text(errors="replace")
    proc = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", host, "cat", path],
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        die(f"cannot read {host}:{path}")
    return proc.stdout.decode(errors="replace")


def check_role(label: str, got: dict, expect: dict) -> dict:
    if not got.get("pid"):
        die(f"{label}: sglang {expect['mode']} process not found")
    cmd = got["cmdline"]
    parsed = {
        "pid": got["pid"],
        "mem": flag(cmd, "mem-fraction-static"),
        "chunk": flag(cmd, "chunked-prefill-size"),
        "ib": flag(cmd, "disaggregation-ib-device"),
        "block": flag(cmd, "speculative-dspark-block-size"),
        "attn_dp": flag(cmd, "attn-dp-size"),
        "tp": flag(cmd, "tp-size"),
        "ep": flag(cmd, "ep-size"),
        "mode": flag(cmd, "disaggregation-mode"),
        "backend": flag(cmd, "dsv4-prefill-backend"),
        "model": flag(cmd, "model-path"),
        "version": got.get("version") or "",
        "site_packages": bool(got.get("site_packages")),
        "worktree": bool(got.get("worktree")),
        "encoder_only_env": got.get("env", {}).get("SGLANG_DSV4_ENCODER_ONLY_PREFILL", ""),
        "mem_env": got.get("env", {}).get("MEM_FRACTION", ""),
        "chunk_env": got.get("env", {}).get("CHUNKED_PREFILL", ""),
    }
    for key, val in expect.items():
        if key == "mode":
            have = parsed["mode"]
        elif key == "backend":
            have = parsed["backend"]
        else:
            have = parsed[key]
        if have != val:
            die(f"{label} {key} is {have!r}, suite expects {val!r}; not posting")
    if not parsed["site_packages"] or parsed["worktree"]:
        die(f"{label}: import path is not site-packages (worktree={parsed['worktree']})")
    if parsed["version"] != "0.5.23":
        die(f"{label}: site-packages version {parsed['version']!r}, expect 0.5.23")
    return parsed


def notes_for(prefill_script: str, decode_script: str) -> str:
    p_mem = script_default(prefill_script, "MEM_FRACTION")
    p_ck = script_default(prefill_script, "CHUNKED_PREFILL")
    d_mem = script_default(decode_script, "MEM_FRACTION")
    d_ck = script_default(decode_script, "CHUNKED_PREFILL")
    if (p_mem, p_ck) != ("0.82", "32768"):
        die(f"prefill script defaults {p_mem}/{p_ck}, expect 0.82/32768")
    if (d_mem, d_ck) != ("0.72", "4096"):
        die(f"decode script defaults {d_mem}/{d_ck}, expect 0.72/4096")
    # 496 chars. The form maxlength is 500; do not grow this without recounting.
    note = (
        "P mem.82 ck32768/eff4096 FlashMLA-sparse rocep37s0f0; L0-19+L20 KV. "
        "D mem.82 ck8192/eff1024 blk3 mlx5_0; script-now .72/4096, proc .82/8192. "
        "EP=attnDP=8 Mooncake Cmax64 :8000 c.3->.09. site-pkg 0.5.23 != wt 853ee5a6. "
        "Engram host off. NVFP4/FP8. "
        "Caveat: 3k/100k early-stop, out~half of 1000. "
        "60k out=1000 but prompt=1 tok so in tok/s invalid; 95%&60000 are targets; C1 n=16. "
        "MT hit82% formula, success100 scripted; C1 8x5, C8 79/80, C32 r5 36/64 n259, "
        "C64 89/128 r5 40 n320; no ITL. "
        "100k not full."
    )
    if len(note) > 500:
        die(f"notes length {len(note)} exceeds 500")
    return note


def load_rows() -> list[dict]:
    rows = []
    with CSV_PATH.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key in CSV_COLS:
                val = (raw.get(key) or "").strip()
                if val == "":
                    continue
                if key == "ds":
                    row[key] = val
                elif "." in val or "e" in val.lower():
                    row[key] = float(val)
                else:
                    row[key] = int(val)
            if row:
                rows.append(row)
    if len(rows) != 40:
        die(f"expected 40 rows, got {len(rows)}")
    return rows


def load_acc() -> dict:
    acc = json.loads(ACC_PATH.read_text())
    out = {k: acc[k] for k in ("acc_set", "acc_n", "acc_pct") if k in acc}
    if out.get("acc_pct") != 100.0 or out.get("acc_n") != 40:
        die(f"unexpected accuracy payload: {out}")
    return out


def public_url() -> str:
    q = urllib.parse.urlencode(
        {"m": MODEL_KEY, "d": DEPLOY_KEY, "h": HW_KEY}
    )
    return f"{SITE}/tests/run/?{q}"


def existing_records() -> list[dict]:
    q = urllib.parse.urlencode({"m": MODEL_KEY, "d": DEPLOY_KEY, "h": HW_KEY})
    status, body = request("GET", "/api/results?" + q)
    if status != 200:
        die(f"GET /api/results failed: {status} {body}")
    return body.get("results") or []


def ours(records: list[dict]) -> list[dict]:
    return [r for r in records if (r.get("data") or {}).get("engine") == ENGINE]


def scenario_summary(rows: list[dict]) -> str:
    buckets: dict[str, list] = {}
    for row in rows:
        buckets.setdefault(row.get("ds", ""), []).append(row["c"])
    parts = [f"{name} C{','.join(str(int(c)) for c in cs)}" for name, cs in buckets.items()]
    return "; ".join(parts)


def main() -> None:
    dry = "--dry-run" in sys.argv
    if len(ENGINE) > 80 or len(TOOL) > 120 or len(SUBMITTER) > 60:
        die("engine/tool/submitter exceed the form limits")

    pre = check_role("prefill", capture(PREFILL_HOST, PREFILL_PID, "prefill"), EXPECT_P)
    dec = check_role("decode", capture(None, DECODE_PID, "decode"), EXPECT_D)
    p_script = read_text(PREFILL_HOST, "/home/aaron/pd/scripts/launch_pd_prefill_109.sh")
    d_script = read_text(None, "/home/aaron/pd/scripts/launch_pd_decode_107.sh")
    note = notes_for(p_script, d_script)
    rows = load_rows()
    acc = load_acc()
    if pre["encoder_only_env"] != "1":
        die("prefill SGLANG_DSV4_ENCODER_ONLY_PREFILL is not 1")

    config = {
        "run_id": RUN_ID,
        "model_key": MODEL_KEY,
        "deploy_key": DEPLOY_KEY,
        "hw_key": HW_KEY,
        "engine": ENGINE,
        "tool": TOOL,
        "notes": note,
        "prefill": pre,
        "decode": dec,
        "prefill_script_defaults": {
            "MEM_FRACTION": script_default(p_script, "MEM_FRACTION"),
            "CHUNKED_PREFILL": script_default(p_script, "CHUNKED_PREFILL"),
        },
        "decode_script_defaults": {
            "MEM_FRACTION": script_default(d_script, "MEM_FRACTION"),
            "CHUNKED_PREFILL": script_default(d_script, "CHUNKED_PREFILL"),
        },
        "router": "http://127.0.0.1:8000",
        "official_field": "verified",
        "official_how": "request header x-ardelva-key; no body flag",
    }
    CONFIG_OUT.write_text(json.dumps(config, indent=1) + "\n")

    status, checked = request(
        "GET",
        "/api/keycheck?" + urllib.parse.urlencode(
            {"m": MODEL_KEY, "d": DEPLOY_KEY, "h": HW_KEY}
        ),
    )
    if status != 200 or not checked.get("ok"):
        die(f"keycheck failed: {status} {checked}")

    prior = ours(existing_records())
    if prior:
        rec = prior[0]
        print(f"already published id {rec.get('id')} verified {rec.get('verified')}")
        print(public_url())
        print(f"rows_on_record {len((rec.get('data') or {}).get('points') or [])}")
        return

    payload = {
        "keys": {"modelKey": MODEL_KEY, "deployKey": DEPLOY_KEY, "hwKey": HW_KEY},
        "data": {
            "date": DATE,
            "submitter": SUBMITTER,
            "engine": ENGINE,
            "tool": TOOL,
            "spec": "on",
            "notes": note,
            "points": rows,
            **acc,
        },
    }
    if dry:
        print(f"dry-run rows {len(rows)} notes {len(note)} engine {len(ENGINE)}")
        print(scenario_summary(rows))
        print(public_url())
        return

    key = load_key()
    status, body = request("POST", "/api/results", payload, key=key)
    print(f"HTTP {status}")
    if status not in (200, 201) or not body.get("id"):
        print(json.dumps({k: v for k, v in body.items() if k != "points"}, ensure_ascii=False)[:800])
        die("post was not accepted", 2)
    rec_id = body.get("id")
    print(f"id {rec_id}")
    print(f"verified {body.get('verified')}")
    if body.get("merged"):
        print(f"merged replaced {body.get('replaced')} added {body.get('added')} points {body.get('points')}")
    print(public_url())

    after = ours(existing_records())
    if len(after) != 1:
        die(f"expected 1 record with this engine, found {len(after)}")
    got = after[0]
    pts = (got.get("data") or {}).get("points") or []
    if not got.get("verified"):
        die(f"record {got.get('id')} is not verified")
    if len(pts) != 40:
        die(f"record has {len(pts)} points, want 40")
    data = got.get("data") or {}
    if data.get("acc_pct") != 100 or data.get("acc_n") != 40:
        die(f"accuracy missing on record: {data.get('acc_pct')} / {data.get('acc_n')}")
    print(f"confirmed points {len(pts)} acc {data.get('acc_set')} {data.get('acc_pct')}")
    print(scenario_summary(pts))
    RECEIPT.write_text(json.dumps({
        "http": status,
        "id": rec_id,
        "verified": body.get("verified"),
        "url": public_url(),
        "points": len(pts),
        "official_field": "verified",
    }, indent=1) + "\n")


if __name__ == "__main__":
    main()
