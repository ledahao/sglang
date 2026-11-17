#!/usr/bin/env python3
"""Sample GPU, HBM occupancy, RoCE counters, and SGLang scheduler lines.

One row per GPU and one row per RoCE port every interval. SGLang rows are the
lines appended to the service log during that interval, reduced per rank.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from pathlib import Path

GPU_FIELDS = [
    "ts",
    "host",
    "gpu",
    "util_gpu",
    "util_mem",
    "mem_used_mib",
    "mem_total_mib",
    "power_w",
    "sm_clock_mhz",
    "mem_clock_mhz",
]
ROCE_FIELDS = [
    "ts",
    "host",
    "dev",
    "rcv_data",
    "xmit_data",
    "rcv_pkts",
    "xmit_pkts",
    "xmit_discards",
    "rcv_errors",
    "xmit_wait",
    "out_of_buffer",
    "ecn_marked",
    "cnp_sent",
    "cnp_handled",
    "rnr_nak",
    "ack_timeout",
    "retries",
]
SGL_FIELDS = [
    "ts",
    "host",
    "rank",
    "kind",
    "batches",
    "new_tokens",
    "running",
    "queue",
    "gen_tps",
    "input_tps",
    "accept_len",
    "accept_rate",
    "full_usage",
    "retracted",
    "transfer",
    "cuda_graph",
]


def nvidia_rows() -> list[list[str]]:
    query = (
        "index,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "power.draw,clocks.sm,clocks.mem"
    )
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        rows.append([part.strip() for part in line.split(",")])
    return rows


def read_int(path: Path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return ""


def ib_row(dev: str) -> dict:
    base = Path(f"/sys/class/infiniband/{dev}/ports/1")
    names = {
        "rcv_data": base / "counters/port_rcv_data",
        "xmit_data": base / "counters/port_xmit_data",
        "rcv_pkts": base / "counters/port_rcv_packets",
        "xmit_pkts": base / "counters/port_xmit_packets",
        "xmit_discards": base / "counters/port_xmit_discards",
        "rcv_errors": base / "counters/port_rcv_errors",
        "xmit_wait": base / "counters/port_xmit_wait",
        "out_of_buffer": base / "hw_counters/out_of_buffer",
        "ecn_marked": base / "hw_counters/np_ecn_marked_roce_packets",
        "cnp_sent": base / "hw_counters/np_cnp_sent",
        "cnp_handled": base / "hw_counters/rp_cnp_handled",
        "rnr_nak": base / "hw_counters/rnr_nak_retry_err",
        "ack_timeout": base / "hw_counters/local_ack_timeout_err",
        "retries": base / "hw_counters/req_transport_retries_exceeded",
    }
    return {key: read_int(path) for key, path in names.items()}


def grab(pattern: str, line: str, kind=float):
    match = re.search(pattern, line)
    if not match:
        return ""
    try:
        return kind(match.group(1))
    except ValueError:
        return ""


def parse_line(line: str):
    rank = re.search(r"\b(DP\d+)\b", line)
    if "Prefill batch" in line:
        kind = "prefill"
    elif "Decode batch" in line:
        kind = "decode"
    else:
        return None
    if not rank:
        return None
    return {
        "rank": rank.group(1),
        "kind": kind,
        "new_tokens": grab(r"#new-token:\s*([0-9]+)", line, int) or 0,
        "running": grab(r"#running-req:\s*([0-9]+)", line, int),
        "queue": grab(r"#queue-req:\s*([0-9]+)", line, int),
        "gen_tps": grab(r"gen throughput \(token/s\):\s*([0-9.]+)", line),
        "input_tps": grab(r"input throughput \(token/s\):\s*([0-9.]+)", line),
        "accept_len": grab(r"accept len:\s*([0-9.]+)", line),
        "accept_rate": grab(r"accept rate:\s*([0-9.]+)", line),
        "full_usage": grab(r"full token usage:\s*([0-9.]+)", line),
        "retracted": grab(r"#retracted-req:\s*([0-9]+)", line, int),
        "transfer": grab(r"#transfer-req:\s*([0-9]+)", line, int),
        "cuda_graph": (
            "True" if "cuda graph: True" in line else "False" if "cuda graph:" in line else ""
        ),
    }


def open_writer(path: Path, fields: list[str]):
    new = not path.exists()
    handle = path.open("a", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if new:
        writer.writeheader()
    return handle, writer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--ib", nargs="+", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--stop", required=True)
    parser.add_argument("--suite-log", default="")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu_f, gpu_w = open_writer(out / "gpu.csv", GPU_FIELDS)
    roce_f, roce_w = open_writer(out / "roce.csv", ROCE_FIELDS)
    sgl_f, sgl_w = open_writer(out / "sglang.csv", SGL_FIELDS)
    def resolve_log() -> Path:
        if "*" in args.log:
            matches = sorted(Path("/").glob(args.log.lstrip("/")), key=lambda p: p.stat().st_mtime)
            return matches[-1] if matches else Path(args.log)
        return Path(args.log)

    log_path = resolve_log()
    offset = log_path.stat().st_size if log_path.exists() else 0
    stop = Path(args.stop)
    suite = Path(args.suite_log) if args.suite_log else None

    while not stop.exists():
        if suite and suite.exists() and "SUITE_DONE" in suite.read_text(errors="replace")[-4000:]:
            break
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            for row in nvidia_rows():
                gpu_w.writerow(
                    {
                        "ts": ts,
                        "host": args.host,
                        "gpu": row[0],
                        "util_gpu": row[1],
                        "util_mem": row[2],
                        "mem_used_mib": row[3],
                        "mem_total_mib": row[4],
                        "power_w": row[5],
                        "sm_clock_mhz": row[6] if len(row) > 6 else "",
                        "mem_clock_mhz": row[7] if len(row) > 7 else "",
                    }
                )
            gpu_f.flush()
        except Exception as exc:
            sgl_w.writerow(
                {"ts": ts, "host": args.host, "rank": "gpu-error", "kind": str(exc)[:80]}
            )
        for dev in args.ib:
            sample = ib_row(dev)
            sample.update({"ts": ts, "host": args.host, "dev": dev})
            roce_w.writerow(sample)
        roce_f.flush()

        chunk = ""
        latest = resolve_log()
        if latest != log_path:
            log_path = latest
            offset = log_path.stat().st_size if log_path.exists() else 0
        if log_path.exists():
            size = log_path.stat().st_size
            if size < offset:
                offset = 0
            with log_path.open("r", errors="replace") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
        folded: dict[tuple[str, str], dict] = {}
        for line in chunk.splitlines():
            item = parse_line(line)
            if not item:
                continue
            key = (item["kind"], item["rank"])
            slot = folded.get(key)
            if slot is None:
                slot = {**item, "batches": 0, "new_tokens": 0}
                folded[key] = slot
            slot["batches"] += 1
            slot["new_tokens"] += int(item["new_tokens"] or 0)
            for field in (
                "running",
                "queue",
                "gen_tps",
                "input_tps",
                "accept_len",
                "accept_rate",
                "full_usage",
                "retracted",
                "transfer",
                "cuda_graph",
            ):
                if item[field] != "":
                    slot[field] = item[field]
        for slot in folded.values():
            slot.update({"ts": ts, "host": args.host})
            sgl_w.writerow(slot)
        if not folded:
            sgl_w.writerow({"ts": ts, "host": args.host, "rank": "-", "kind": "idle", "batches": 0})
        sgl_f.flush()
        time.sleep(args.interval)

    for handle in (gpu_f, roce_f, sgl_f):
        handle.close()


if __name__ == "__main__":
    main()
