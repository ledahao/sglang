# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""One-line PD stage logs joined later on bootstrap room.

Durations are emitted only when this process already has both timestamps.
Prefill forward ends at prefill-finished, before the KV poll. The publish-done
and decode-ready events are points; the bench subtracts their wall clocks.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def _emit_rank() -> bool:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    for name in ("attn_tp_rank", "pp_rank", "attn_cp_rank"):
        if int(getattr(parallel, name, 0) or 0) != 0:
            return False
    return True


def log_pd_stage(req, peer: str, stage: str, ms, ts: float | None = None) -> None:
    if not _emit_rank():
        return
    from sglang.srt.observability.req_time_stats import convert_time_to_realtime

    if ts is None:
        ts = time.perf_counter()
    t_ms = int(round(convert_time_to_realtime(ts) * 1000.0))
    ms_text = "-" if ms is None else f"{float(ms):.2f}"
    rid = getattr(req, "rid", "-")
    room = getattr(req, "bootstrap_room", None)
    room_text = "-" if room is None else str(room)
    origin = getattr(req, "origin_input_ids", None)
    input_len = len(origin) if origin is not None else 0
    if peer == "prefill":
        cached = getattr(req, "prefix_len", 0) or 0
    else:
        cached = getattr(req, "cached_tokens", 0) or 0
    logger.info(
        "PDSTAGE rid=%s room=%s peer=%s stage=%s ms=%s t_ms=%s input_len=%s cached=%s",
        rid,
        room_text,
        peer,
        stage,
        ms_text,
        t_ms,
        input_len,
        int(cached),
    )


def maybe_log_first_token(req) -> None:
    """Decode-ready to the first non-negative output id, including tail replay."""
    if getattr(req, "_pd_first_token_logged", False):
        return
    stats = getattr(req, "time_stats", None)
    if stats is None or stats.wait_queue_entry_time <= 0.0:
        return
    ids = list(getattr(req, "output_ids", []) or [])
    if not any(int(token) >= 0 for token in ids):
        return
    now = time.perf_counter()
    log_pd_stage(
        req,
        "decode",
        "first_token",
        (now - stats.wait_queue_entry_time) * 1000.0,
        now,
    )
    req._pd_first_token_logged = True
