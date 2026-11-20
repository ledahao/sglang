from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)
_span_choice_logged: set[str] = set()


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "n", "off"}

DSA_DENSE = "dense"
DSA_SPARSE = "sparse"


class AttentionGraphVariants(Protocol):
    # Capture order is significant when variants share a graph memory pool.
    capture_labels: ClassVar[tuple[str, ...]]

    def select(self, forward_batch: ForwardBatch) -> str:
        """Select one of capture_labels for the batch."""
        ...


@dataclass(frozen=True)
class DsaGraphVariants:
    index_topk: int
    # Dense comes first: the sparse capture peak subsumes its shared-pool storage.
    capture_labels: ClassVar[tuple[str, ...]] = (DSA_DENSE, DSA_SPARSE)

    def select(self, forward_batch: ForwardBatch) -> str:
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
            # Plain decode maintains this host mirror without a D2H sync.
            max_kv_len = int(seq_lens_cpu.max().item())
        elif forward_batch.seq_lens is not None and forward_batch.seq_lens.numel() > 0:
            # Fallback: a single scalar reduction d2h (cheap, per-step).
            max_kv_len = int(forward_batch.seq_lens.max().item())
        else:
            # No length info: be safe and use the correct-for-all sparse graph.
            return DSA_SPARSE
        return DSA_DENSE if max_kv_len <= self.index_topk else DSA_SPARSE


def create_attention_graph_variants(hf_config) -> Optional[AttentionGraphVariants]:
    from sglang.srt.configs.model_config import (
        get_dsa_index_kpool,
        get_dsa_index_topk,
        is_deepseek_dsa,
    )
    from sglang.srt.utils import is_hip

    if is_hip() and is_deepseek_dsa(hf_config):
        # KPool has no dense-skip path: both variants run the full indexer.
        # Capture it once, keeping the regular DSA dense/sparse split intact.
        if get_dsa_index_kpool(hf_config) > 1:
            return None

        index_topk = get_dsa_index_topk(hf_config)
        logger.info(
            "[dense-decode] DSA dual-graph enabled: capturing "
            "dense (k-only) + sparse (full indexer) decode graphs; "
            "dispatch on max_kv_len vs index_topk=%d.",
            index_topk,
        )
        return DsaGraphVariants(index_topk)
    return None


DSV41_CANDIDATE_FILTERED = "candidate_filtered"
DSV41_CANDIDATE_UNFILTERED = "candidate_unfiltered"
# Filtered graphs score only this many tokens. Ascending, each above the
# candidate span. A longer request stays on the page-table filtered graph.
DSV41_FILTERED_SPAN_BUCKETS: tuple[tuple[str, int], ...] = (
    ("candidate_filtered_32k", 32 * 1024),
    ("candidate_filtered_64k", 64 * 1024),
    ("candidate_filtered_128k", 128 * 1024),
    ("candidate_filtered_192k", 192 * 1024),
    ("candidate_filtered_256k", 256 * 1024),
)
# SGLANG_OPT_DSV41_TARGET_METADATA_EARLY serves these target verify variants.
# Filtered span buckets install the same metadata as candidate_filtered, so they
# share that early path instead of the in-graph upgrade.
DSV41_EARLY_VERIFY_VARIANTS = frozenset(
    {DSV41_CANDIDATE_UNFILTERED, DSV41_CANDIDATE_FILTERED}
    | {label for label, _ in DSV41_FILTERED_SPAN_BUCKETS}
)


def captured_token_span(variant: Optional[str], indexer) -> Optional[int]:
    """Token ceiling baked into a candidate graph.

    None keeps the page-table width. The ceiling is the variant contract, never
    the length of whatever request happens to be on the GPU during capture.
    """
    if variant == DSV41_CANDIDATE_UNFILTERED:
        return indexer.candidate_topk_blocks * indexer.candidate_block_size
    if variant == "candidate_c2_all":
        return indexer.index_topk * 2
    for label, limit in DSV41_FILTERED_SPAN_BUCKETS:
        if variant == label:
            return limit
    return None


@dataclass(frozen=True)
class Dsv41CandidateGraphVariants:
    """Candidate-indexer graphs keyed by the batch's longest request.

    Variants at or below the candidate span may skip low-ratio scoring or
    candidate filtering. Wider filtered buckets keep that filtering and only
    shorten the scored columns to their own ceiling.
    """

    # (label, max_seq_len it serves), ascending; the last label is the fallback.
    graph_limits: tuple[tuple[str, int], ...]
    capture_labels: tuple[str, ...]
    verify_extra_tokens: int = 0

    def select(self, forward_batch: ForwardBatch) -> str:
        device_len = _cpu_seq_len_max(forward_batch)
        bound = None
        if self.verify_extra_tokens:
            # Lifetime bound, including acceptance still in flight.
            bound = getattr(
                getattr(forward_batch, "spec_info", None),
                "candidate_max_seq_len_upper_bound",
                None,
            )
        if device_len is None:
            max_seq_len = bound
        elif bound is None:
            max_seq_len = device_len
        else:
            # A short CPU mirror must not hide a request that can outgrow
            # the captured span, and a short budget must not hide a longer KV.
            max_seq_len = max(device_len, int(bound))
        variant = self.select_for_max_seq_len(max_seq_len)
        if variant not in _span_choice_logged:
            _span_choice_logged.add(variant)
            logger.info(
                "candidate graph select variant=%s max_seq_len=%s "
                "device_len=%s bound=%s",
                variant,
                max_seq_len,
                device_len,
                bound,
            )
        return variant

    def select_for_max_seq_len(self, max_seq_len: Optional[int]) -> str:
        """The variant for a batch whose longest request (before this step's verify
        tokens) is ``max_seq_len``; None, when unknown, takes the fallback."""
        if max_seq_len is not None:
            max_seq_len += self.verify_extra_tokens
            for variant, limit in self.graph_limits:
                if max_seq_len <= limit:
                    return variant
        return DSV41_CANDIDATE_FILTERED



def _cpu_seq_len_max(forward_batch: ForwardBatch) -> Optional[int]:
    """Max live CPU length, or None when the batch has no host mirror."""
    lengths = getattr(forward_batch, "seq_lens_cpu", None)
    if lengths is None or not hasattr(lengths, "numel"):
        return None
    device = getattr(lengths, "device", None)
    if device is None or device.type != "cpu" or lengths.numel() == 0:
        return None
    n = int(lengths.numel())
    bs = getattr(forward_batch, "batch_size", None)
    if isinstance(bs, int) and 0 < bs < n:
        lengths = lengths[:bs]
    return int(lengths.max())


def create_dsv41_candidate_graph_variants(
    model_runner, capture_forward_mode, captured_req_width: int = 0
) -> Optional[Dsv41CandidateGraphVariants]:
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.utils import is_gfx95_supported, is_hip

    text_config = model_runner.model_config.hf_text_config
    dspark_target_verify = (
        capture_forward_mode == ForwardMode.TARGET_VERIFY
        and model_runner.spec_algorithm.is_dspark()
        and not model_runner.is_draft_worker
        and captured_req_width > 0
    )
    if not (
        (capture_forward_mode == ForwardMode.DECODE or dspark_target_verify)
        and model_runner.device == "cuda"
        and (
            is_gfx95_supported()
            if is_hip()
            else torch.cuda.get_device_capability(model_runner.gpu_id)[0] >= 10
        )
        and getattr(text_config, "model_type", None) == "deepseek_v41"
        and getattr(text_config, "candidate_source_layer_id", -1) >= 0
    ):
        return None
    span = text_config.candidate_topk_blocks * text_config.candidate_block_size
    if span <= 0:
        return None
    ratios = set(text_config.compress_ratios) & {1, 2}
    topk = text_config.index_topk
    variants = []
    # Verify needs per-query causal top-k, so it always keeps candidate filtering.
    if topk > 0 and ratios and not dspark_target_verify:
        variants.append(("candidate_all", topk * min(ratios)))
        if ratios == {1, 2}:
            variants.append(("candidate_c2_all", topk * 2))
    variants.append((DSV41_CANDIDATE_UNFILTERED, span))
    graph_limits = []
    for variant, limit in variants:
        graph_limits.append((variant, min(limit, span)))
        if limit >= span:
            break
    # Above the candidate span the indexer still filters. Each extra graph
    # scores only its own ceiling, and select() refuses a ceiling that cannot
    # hold the batch. The final candidate_filtered label keeps the page table.
    buckets_on = _env_enabled("SGLANG_DSV41_SPAN_BUCKETS", True)
    if buckets_on:
        for label, limit in DSV41_FILTERED_SPAN_BUCKETS:
            if limit > span:
                graph_limits.append((label, limit))
    short_cols = _env_enabled("SGLANG_DSV41_SHORT_COLS", True)
    logger.info(
        "Candidate indexer graph limits: %s; page-table filtering above %s. "
        "span_buckets=%s short_cols=%s",
        graph_limits,
        graph_limits[-1][1],
        int(buckets_on),
        int(short_cols),
    )
    return Dsv41CandidateGraphVariants(
        graph_limits=tuple(graph_limits),
        capture_labels=tuple(v for v, _ in graph_limits) + (DSV41_CANDIDATE_FILTERED,),
        # The verify backend adds this width to committed CPU lengths.
        verify_extra_tokens=captured_req_width if dspark_target_verify else 0,
    )
