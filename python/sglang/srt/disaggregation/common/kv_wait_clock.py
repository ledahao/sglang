"""Pure decisions for the early-allocation KV wait clock.

The 300s transfer clock starts at the last-chunk notice, not at metadata
publish. A separate prefill-hold budget covers the gap before that notice.
These helpers keep a duplicate or out-of-order notice from moving the
transfer stamp, and they let a cancel arm deferred page reclaim after
metadata was published even though the transfer clock has not started.
"""

WAITING_FOR_INPUT = 2

# Measured cold prefill forward for 997k tokens was 564s. This budget is
# three times that, and it is not SGLANG_DISAGGREGATION_WAITING_TIMEOUT.
PREFILL_HOLD_BUDGET_S = 1800.0


def accept_transfer_notice(*, room_status, already_stamped: bool) -> bool:
    """True only for the first notice while the room is still waiting.

    A repeat does not move the stamp. A notice after Failed or Success, or
    before the room is waiting, does not arm the transfer clock.
    """
    return room_status == WAITING_FOR_INPUT and not already_stamped


def should_arm_deferred_release(
    *,
    enabled: bool,
    force_arm: bool,
    transfer_clock_started: bool,
    metadata_published: bool,
) -> bool:
    """Arm drain-ack tracking when a cancel can race an in-flight write.

    Early allocation publishes metadata, and therefore may already be
    receiving KV, before the transfer clock starts. A cancel in that window
    still has to arm reclaim. A receiver that has not published metadata
    stays unarmed.
    """
    return bool(enabled) and (
        force_arm or transfer_clock_started or metadata_published
    )


def transfer_clock_start(metadata_init_time, notice_stamp):
    """The transfer deadline anchor. None until a notice arrives on early."""
    if metadata_init_time is not None:
        return metadata_init_time
    return notice_stamp


def waiting_deadline(
    *,
    init_time,
    hold_start,
    now: float,
    transfer_timeout: float,
    hold_budget: float,
):
    """Which clock expired, or None.

    ``transfer`` uses the existing transfer timeout and only runs after the
    last-chunk notice (or a non-early metadata publish) has set init_time.
    ``hold`` uses hold_budget and only runs while init_time is still None.
    A 300s transfer timeout cannot expire the prefill hold.
    """
    if init_time is not None:
        if now - init_time >= transfer_timeout:
            return "transfer"
        return None
    if hold_start is None:
        return None
    if now - hold_start >= hold_budget:
        return "hold"
    return None
