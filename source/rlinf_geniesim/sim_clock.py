# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Global sim clock: one monotonic int64 tick, owned by the handler, threaded
through poses -> render node -> rendered frame -> obs so a step's rgb/state/reward
all carry the same tick. Decision: the tick NEVER resets (episode boundaries are
per-env `done` + a separate counter). The tick rides on the /tf_render stamp
(atomic with the poses it labels); this module owns the (sec,nanosec) codec."""
from __future__ import annotations


class SimClock:
    """Single authoritative tick. advance() pre-increments and returns the new
    tick; there is deliberately no reset()."""

    def __init__(self, start: int = 0) -> None:
        self._t = int(start)

    def advance(self) -> int:
        self._t += 1
        return self._t

    def current(self) -> int:
        return self._t


_SEC_MASK = 0x7FFFFFFF  # low 31 bits -> ROS stamp.sec (int32-safe)
_NSEC_MASK = 0xFFFFFFFF  # 32 bits    -> ROS stamp.nanosec (uint32)


def encode_tick_to_stamp(tick: int) -> tuple[int, int]:
    """Pack a 63-bit tick into a ROS (sec, nanosec) stamp. Realistic ticks
    (< 2**31) land entirely in sec, so the wire stays human-readable."""
    t = int(tick)
    return (t & _SEC_MASK, (t >> 31) & _NSEC_MASK)


def decode_stamp_to_tick(sec: int, nanosec: int) -> int:
    return (int(sec) & _SEC_MASK) | (int(nanosec) << 31)


__all__ = ["SimClock", "encode_tick_to_stamp", "decode_stamp_to_tick"]
