"""Default-off CUDA graph instantiate topology for decode.

Armed only when SGLANG_GRAPH_DEPS=1. Uses torch's
register_graph_instantiate_hook and CUDAGraph.get_graph_data, the same
calls already run on this host. No CUPTI subscription and no stream
synchronize.

CUDAGraph.replay has no per-launch id. replay_launch_id stays null.
A replay is counted only after /tmp/sglang-graph-deps-serving exists,
so capture replays are not serving replays. The hook removes itself
after two serving replays of the largest stored graph.
"""

from __future__ import annotations

import json
import os
import threading
import warnings

_ENV = "SGLANG_GRAPH_DEPS"
_DIR = "/tmp/sglang-graph-deps-r0"
_SERVING = "/tmp/sglang-graph-deps-serving"
_DISARM = "/tmp/sglang-graph-deps-disarm"
_MAX_GRAPHS = 48
_MAX_NODES = 500000

_lock = threading.Lock()
_armed = False
_rank_known = None
_inst_handle = None
_replay_handle = None
_graphs: list[dict] = []
_by_obj: dict[int, int] = {}
_fires: dict[int, int] = {}
_largest: tuple[int, int] | None = None
_NOTE = (
    "CUDAGraph.replay exposes no per-launch id; replay_launch_id is null. "
    "fires counts serving-replay hook entries for one exec id, after "
    "the serving file exists. It is not a launch id."
)


def _tp_rank():
    try:
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )

        return int(get_tensor_model_parallel_rank())
    except Exception:
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return None


def _device():
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except Exception:
        return None
    return None


def _write(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)


def _index() -> None:
    os.makedirs(_DIR, exist_ok=True)
    _write(
        os.path.join(_DIR, "index.json"),
        {
            "replay_launch_id": None,
            "note": _NOTE,
            "largest_exec": _largest[0] if _largest else None,
            "graphs": _graphs,
        },
    )


def _replays(removed: bool) -> None:
    os.makedirs(_DIR, exist_ok=True)
    _write(
        os.path.join(_DIR, "replays.json"),
        {
            "replay_launch_id": None,
            "note": _NOTE,
            "largest_exec": _largest[0] if _largest else None,
            "fires": {str(k): v for k, v in _fires.items()},
            "hook_removed": removed,
        },
    )


def _error(text: str) -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
        with open(os.path.join(_DIR, "error.txt"), "a", encoding="utf-8") as fh:
            fh.write(text[:2000] + "\n")
    except Exception:
        pass


def _drop_handles() -> None:
    global _inst_handle, _replay_handle
    with _lock:
        inst, replay = _inst_handle, _replay_handle
        _inst_handle = None
        _replay_handle = None
        _replays(True)
    for handle in (replay, inst):
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass


def _on_instantiate(graph) -> None:
    global _largest, _rank_known
    try:
        rank = _tp_rank()
        _rank_known = rank
        if rank not in (None, 0):
            return
        if rank is None:
            _error("rank unknown; skipped get_graph_data")
            return
        device = _device()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = graph.get_graph_data()
        nested = "; ".join(str(item.message) for item in caught) or None
        raw_nodes = data.get("nodes") or []
        exec_id = data.get("exec_graph_id")
        rec = {
            "rank": rank,
            "device": device,
            "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "exec_graph_id": exec_id,
            "capture_graph_id": getattr(graph, "_capture_graph_id", None),
            "stream": None,
            "n": len(raw_nodes),
            "event_ptr_gaps": 0,
            "nested_opaque": nested,
            "opaque": None,
        }
        nodes = []
        if exec_id is None:
            rec["opaque"] = "exec_graph_id missing"
        elif len(raw_nodes) > _MAX_NODES:
            rec["opaque"] = f"node cap {len(raw_nodes)}"
        else:
            ids = []
            for node in raw_nodes:
                tools_id = node.get("tools_id")
                ids.append(tools_id)
            gaps = 0
            for node, tools_id in zip(raw_nodes, ids):
                preds = []
                for idx in node.get("dependencies") or []:
                    if isinstance(idx, int) and 0 <= idx < len(ids) and ids[idx] is not None:
                        preds.append(ids[idx])
                    else:
                        preds.append(None)
                        gaps += 1
                kind = node.get("node_type")
                eptr = node.get("event_ptr") or 0
                if kind in ("event_record", "wait_event") and not eptr:
                    gaps += 1
                if tools_id is None:
                    gaps += 1
                item = {
                    "id": tools_id,
                    "nid": node.get("node_id"),
                    "ty": kind,
                    "pred": preds,
                }
                nodes.append(item)
            rec["event_ptr_gaps"] = gaps
        with _lock:
            if len(_graphs) >= _MAX_GRAPHS:
                smallest = min(range(len(_graphs)), key=lambda i: _graphs[i]["n"])
                if rec["n"] <= _graphs[smallest]["n"]:
                    return
                old = _graphs.pop(smallest)
                old_exec = old.get("exec_graph_id")
                for obj, exe in list(_by_obj.items()):
                    if exe == old_exec:
                        _by_obj.pop(obj, None)
                if old_exec is not None:
                    try:
                        os.remove(os.path.join(_DIR, f"g-{old_exec}.json"))
                    except OSError:
                        pass
            os.makedirs(_DIR, exist_ok=True)
            if exec_id is not None and nodes:
                _write(
                    os.path.join(_DIR, f"g-{exec_id}.json"),
                    {"exec_graph_id": exec_id, "nodes": nodes},
                )
            _graphs.append(rec)
            if exec_id is not None:
                _by_obj[id(graph)] = exec_id
                if _largest is None or rec["n"] > _largest[1]:
                    _largest = (exec_id, rec["n"])
            _index()
    except Exception as exc:
        _error(f"{type(exc).__name__}: {exc}")


def _on_replay(graph) -> None:
    global _replay_handle, _inst_handle
    try:
        if _rank_known != 0:
            return
        if os.path.exists(_DISARM):
            _drop_handles()
            return
        if not os.path.exists(_SERVING):
            return
        exe = _by_obj.get(id(graph))
        if exe is None:
            return
        with _lock:
            _fires[exe] = _fires.get(exe, 0) + 1
            stop = (
                _largest is not None
                and exe == _largest[0]
                and _fires[exe] >= 2
            )
            if stop or _fires[exe] <= 2:
                _replays(False)
            if stop:
                inst, replay = _inst_handle, _replay_handle
                _inst_handle = None
                _replay_handle = None
            else:
                inst = replay = None
        if replay is not None or inst is not None:
            _replays(True)
            for handle in (replay, inst):
                if handle is not None:
                    try:
                        handle.remove()
                    except Exception:
                        pass
    except Exception as exc:
        _error(f"replay {type(exc).__name__}: {exc}")


def arm_graph_deps_diag() -> None:
    """Register the instantiate hook. No-op unless SGLANG_GRAPH_DEPS=1."""
    global _armed, _inst_handle, _replay_handle
    if os.environ.get(_ENV, "0") != "1" or _armed:
        return
    try:
        from torch.cuda.graphs import (
            register_graph_instantiate_hook,
            register_graph_replay_start_hook,
        )

        _armed = True
        _inst_handle = register_graph_instantiate_hook(_on_instantiate)
        _replay_handle = register_graph_replay_start_hook(_on_replay)
        rank = _tp_rank()
        if rank not in (None, 0):
            return
        os.makedirs(_DIR, exist_ok=True)
        _write(
            os.path.join(_DIR, "armed.json"),
            {
                "armed": True,
                "pid": os.getpid(),
                "rank": rank,
                "device": _device(),
            },
        )
    except Exception as exc:
        _error(f"arm {type(exc).__name__}: {exc}")
