"""Pure helpers for cuRobo trajectory timing.

The command-controller boundary resolves user slowdown and physics timestep
alignment once. MotionGen receives that resolved value and must consume it
without applying another timestep multiplier.
"""

from __future__ import annotations


REFERENCE_PHYSICS_DT = 1.0 / 60.0


def resolve_slowdown_factor(
    user_factor: float,
    physics_dt: float,
    reference_dt: float = REFERENCE_PHYSICS_DT,
) -> float:
    """Keep trajectory duration consistent with the historical 60 Hz path."""

    factor = max(1.0, float(user_factor or 1.0))
    if physics_dt <= 0:
        return factor
    timestep_alignment = max(1.0, float(reference_dt) / float(physics_dt))
    return factor * timestep_alignment


def consume_resolved_slowdown_factor(resolved_factor: float) -> float:
    """Return the final factor already resolved by ``CommandController``."""

    return float(resolved_factor or 1.0)
