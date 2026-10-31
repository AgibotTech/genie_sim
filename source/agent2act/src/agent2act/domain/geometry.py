# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Local interpolation only; no collision or reachability claims."""

import math


def quaternion_angle(a: list[float], b: list[float]) -> float:
    return 2 * math.acos(min(1.0, abs(sum(x * y for x, y in zip(a, b)))))


def quaternion_speed_bound(a: list[float], b: list[float]) -> float:
    """Bound radians per unit interpolation fraction, including short-arc NLERP."""
    dot = min(1.0, abs(sum(x * y for x, y in zip(a, b))))
    angle = 2 * math.acos(dot)
    # The normalized-linear branch of slerp is slightly faster at its midpoint
    # than constant-angular-rate SLERP. Its exact maximum is 4*tan(angle/4).
    return 4 * math.tan(angle / 4) if dot > 0.9995 else angle


def slerp(a: list[float], b: list[float], fraction: float) -> list[float]:
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b, dot = [-v for v in b], -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        result = [x + fraction * (y - x) for x, y in zip(a, b)]
    else:
        theta = math.acos(dot)
        w0 = math.sin((1 - fraction) * theta) / math.sin(theta)
        w1 = math.sin(fraction * theta) / math.sin(theta)
        result = [w0 * x + w1 * y for x, y in zip(a, b)]
    norm = math.sqrt(sum(v * v for v in result))
    return [v / norm for v in result]


def smoothstep(t: float) -> float:
    return t * t * t * (10 + t * (-15 + 6 * t))


def smooth_cruise(t: float) -> float:
    """Unit-distance C2 progress with 20% acceleration/deceleration ramps.

    Cubic smoothstep ramps around a constant cruise; normalized peak velocity 1.25, zero endpoint
    velocity and acceleration. A timing shape, not a dynamics guarantee."""
    ramp = 0.2
    if t < ramp:
        u = t / ramp
        return ramp / (1 - ramp) * (u**3 - 0.5 * u**4)
    if t > 1 - ramp:
        u = (1 - t) / ramp
        return 1 - ramp / (1 - ramp) * (u**3 - 0.5 * u**4)
    return (t - ramp / 2) / (1 - ramp)


def interpolate_pose(start: list[float], end: list[float], fraction: float) -> list[float]:
    return [a + fraction * (b - a) for a, b in zip(start[:3], end[:3])] + slerp(start[3:], end[3:], fraction)
