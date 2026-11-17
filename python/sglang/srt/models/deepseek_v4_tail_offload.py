"""Stage DeepSeek-V4.1 decoder-tail weights from disk on prefill.

Host RAM cannot hold every tail shard pinned (about 200 GiB). Each tail
layer is written to disk after load, then copied to GPU only for its own
forward. Decode never sets the env var, so this stays a no-op there.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import shutil
from contextlib import contextmanager
from typing import Callable, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_ENV = "SGLANG_DSV4_PREFILL_OFFLOAD_FROM_LAYER"
_DIR_ENV = "SGLANG_DSV4_PREFILL_OFFLOAD_DIR"
_Setter = Callable[[torch.Tensor], None]


def tail_offload_from() -> int | None:
    raw = os.environ.get(_ENV, "").strip()
    if not raw:
        return None
    value = int(raw)
    return value if value >= 0 else None


def _assign_param_data(param: nn.Parameter, tensor: torch.Tensor) -> None:
    param.data = tensor


def _iter_targets(
    layer: nn.Module,
) -> Tuple[List[Tuple[torch.Tensor, _Setter]], List[Tuple[_Setter, int]]]:
    items: List[Tuple[torch.Tensor, _Setter]] = []
    aliases: List[Tuple[_Setter, int]] = []
    seen: dict[int, int] = {}

    def add(tensor: torch.Tensor, setter: _Setter) -> None:
        if not torch.is_tensor(tensor) or tensor.numel() < 1024:
            return
        if tensor.device.type == "meta":
            return
        key = id(tensor)
        if key in seen:
            aliases.append((setter, seen[key]))
            return
        seen[key] = len(items)
        items.append((tensor, setter))

    for param in layer.parameters(recurse=True):
        add(param.data, lambda tensor, param=param: _assign_param_data(param, tensor))
    for sub in layer.modules():
        for key, value in list(vars(sub).items()):
            if isinstance(value, nn.Parameter) or not torch.is_tensor(value):
                continue
            add(value, lambda tensor, sub=sub, key=key: setattr(sub, key, tensor))
    return items, aliases


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def _write_tensor(tensor: torch.Tensor, path: str) -> torch.Tensor:
    """Copy to a resizable CPU tensor and write those bytes. Do not call numpy()."""
    contig = tensor.detach().to("cpu").contiguous()
    nbytes = contig.numel() * contig.element_size()
    chunk = 32 * 1024 * 1024
    with open(path, "wb") as handle:
        offset = 0
        while offset < nbytes:
            size = min(chunk, nbytes - offset)
            data = (ctypes.c_char * size).from_address(contig.data_ptr() + offset)
            handle.write(data)
            offset += size
    return contig


def _read_tensor(spec: dict) -> torch.Tensor:
    dtype = _dtype_from_name(spec["dtype"])
    numel = 1
    for dim in spec["shape"]:
        numel *= dim
    raw = torch.from_file(
        spec["path"], shared=False, size=numel * dtype.itemsize, dtype=torch.uint8
    )
    return raw.view(dtype).reshape(spec["shape"])


def _release(tensor: torch.Tensor) -> None:
    tensor.untyped_storage().resize_(0)


def _cleanup_stale(base: str) -> None:
    os.makedirs(base, exist_ok=True)
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        try:
            pid = int(name)
        except ValueError:
            continue
        if not os.path.exists(f"/proc/{pid}"):
            shutil.rmtree(path, ignore_errors=True)


def offload_prefill_tail(model: nn.Module) -> None:
    """Write tail-layer tensors to disk and drop their RAM/GPU storage."""
    start = tail_offload_from()
    if start is None:
        return
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        logger.warning("Prefill tail offload skipped: model has no layers")
        return
    base = os.environ.get(_DIR_ENV, "/home/aaron/pd/dsv4-prefill-tail")
    _cleanup_stale(base)
    dest = os.path.join(base, str(os.getpid()))
    os.makedirs(dest, exist_ok=True)
    nbytes = 0
    ntensors = 0
    nlayers = 0
    protected: set[int] = set()
    for layer in layers:
        layer_id = getattr(layer, "layer_id", None)
        if layer_id is None or layer_id >= start:
            continue
        items, _aliases = _iter_targets(layer)
        for tensor, _setter in items:
            if tensor.device.type == "cuda":
                protected.add(tensor.untyped_storage().data_ptr())
    skipped = 0
    for layer in layers:
        layer_id = getattr(layer, "layer_id", None)
        if layer_id is None or layer_id < start:
            continue
        items, aliases = _iter_targets(layer)
        specs = []
        holders: dict[int, torch.Tensor] = {}
        for index, (tensor, setter) in enumerate(items):
            if tensor.device.type != "cuda" or tensor.untyped_storage().nbytes() == 0:
                continue
            if tensor.untyped_storage().data_ptr() in protected:
                skipped += 1
                continue
            path = os.path.join(dest, f"layer{layer_id}-{index}.bin")
            device_index = tensor.device.index
            cpu = _write_tensor(tensor, path)
            shape = tuple(cpu.shape)
            dtype = cpu.dtype
            del cpu
            holder = torch.empty(shape, dtype=dtype, device="cpu")
            _release(holder)
            setter(holder)
            holders[index] = holder
            items[index] = (holder, setter)
            specs.append(
                {
                    "path": path,
                    "shape": list(shape),
                    "dtype": _dtype_name(dtype),
                    "device": device_index,
                }
            )
            nbytes += os.path.getsize(path)
            ntensors += 1
        if not specs:
            continue
        for setter, index in aliases:
            holder = holders.get(index)
            if holder is not None:
                setter(holder)
        layer._tail_offload_specs = specs
        nlayers += 1
    gc.collect()
    torch.cuda.empty_cache()
    message = (
        f"Prefill decoder-tail offload: layers>={start}, {nlayers} layers, "
        f"{ntensors} tensors, {nbytes / (1024 ** 3):.2f} GiB on disk at {dest}, "
        f"shared_kept={skipped}"
    )
    logger.info(message)
    print(message, flush=True)


@contextmanager
def stage_offloaded_layer(layer: nn.Module):
    """Copy one offloaded layer to GPU for its forward, then drop it."""
    start = tail_offload_from()
    layer_id = getattr(layer, "layer_id", None)
    if os.environ.get("SGLANG_DSV4_PREFILL_OFFLOAD_DEBUG") == "1" and layer_id is not None:
        print(f"enter layer {layer_id}", flush=True)
        torch.cuda.synchronize()
    specs = getattr(layer, "_tail_offload_specs", None)
    if start is None or layer_id is None or layer_id < start or not specs:
        yield
        return
    print(f"stage layer {layer_id} tensors {len(specs)}", flush=True)
    torch.cuda.synchronize()
    items, aliases = _iter_targets(layer)
    pending = [
        (index, setter)
        for index, (tensor, setter) in enumerate(items)
        if tensor.untyped_storage().nbytes() == 0
    ]
    if len(pending) != len(specs):
        raise RuntimeError(
            f"layer {layer_id} has {len(pending)} offloaded tensors, expected {len(specs)}"
        )
    staged: dict[int, torch.Tensor] = {}
    host_copies: list[torch.Tensor] = []
    try:
        for spec, (index, setter) in zip(specs, pending):
            # Keep the file-backed tensor alive until the H2D copy finishes.
            # A non-blocking copy whose source is freed reports illegal address.
            device = torch.device("cuda", spec["device"])
            host = _read_tensor(spec)
            gpu = host.to(device, non_blocking=False)
            host_copies.append(host)
            setter(gpu)
            staged[index] = gpu
        for setter, index in aliases:
            gpu = staged.get(index)
            if gpu is not None:
                setter(gpu)
        torch.cuda.current_stream().synchronize()
        yield
    finally:
        # The forward just queued kernels that still read these weights.
        torch.cuda.synchronize()
        restored: dict[int, torch.Tensor] = {}
        for spec, (index, setter) in zip(specs, pending):
            holder = torch.empty(
                spec["shape"], dtype=_dtype_from_name(spec["dtype"]), device="cpu"
            )
            _release(holder)
            setter(holder)
            restored[index] = holder
        for setter, index in aliases:
            holder = restored.get(index)
            if holder is not None:
                setter(holder)
        del staged
        del host_copies
