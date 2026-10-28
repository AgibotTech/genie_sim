# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Blocking readback at Newton lifecycle boundaries, never on an ordinary tick."""

from __future__ import annotations

from typing import Any


def flush_host_mirrors(engine: Any) -> None:
    """Join pending GPU work and fill both slots with the current completed state.

    Async mirrors normally expose the previous tick. A paused observation must
    expose the final tick instead, including when no further step will occur.
    This does not run FK, rewrite state, or touch solver carry-over.
    """
    import warp as wp

    wp.synchronize_device()
    if getattr(engine, "_pq_mirror", None) is not None:
        engine._update_record_snapshot_mirror()
    state = engine._state_0
    for mirror_name, source_name in (
        ("_jq_mirror", "joint_q"),
        ("_jqd_mirror", "joint_qd"),
        ("_bq_mirror", "body_q"),
        ("_pq_mirror", "particle_q"),
    ):
        source = getattr(state, source_name, None)
        mirror = getattr(engine, mirror_name, None)
        if mirror is not None and source is not None:
            stream = wp.get_stream(source.device)
            mirror.refresh(source, stream)
            mirror.refresh(source, stream)
    wp.synchronize_device()


def pause_at_tick_boundary(engine: Any) -> None:
    """Finish readback and optional rendered observation before reporting paused.

    The run loop calls this on its physics thread before setting lifecycle
    ``inactive``. Failures propagate; a failed capture cannot acknowledge pause.
    A repeated pause refreshes observations without integrating another tick.
    """
    flush_host_mirrors(engine)
    capture = getattr(engine, "_pause_observation_callback", None)
    if capture is not None:
        capture()
    was_paused = engine._paused
    engine._paused = True
    if not was_paused:
        engine._logger.info("[newton-standalone] paused — completed state readback; step() advances nothing")
