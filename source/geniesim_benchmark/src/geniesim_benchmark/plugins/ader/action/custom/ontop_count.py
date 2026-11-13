# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""One scored milestone for a same-observation count of objects on a surface."""

import math
from collections.abc import Sequence

from geniesim_benchmark.plugins.ader.action.common_actions import (
    ActionBase,
    ActionEvent,
    EvaluateAction,
)


def ontop_evidence(
    active_bounds: Sequence[Sequence[float]],
    surface_bounds: Sequence[Sequence[float]],
    threshold: float = 0.5,
) -> dict[str, float | bool]:
    """Use Ontop's existing 2 cm / active-largest-face predicate, without latching."""
    bounds = []
    for lower, upper in (active_bounds, surface_bounds):
        lo, hi = tuple(float(x) for x in lower), tuple(float(x) for x in upper)
        if len(lo) != 3 or len(hi) != 3 or not all(math.isfinite(x) for x in lo + hi):
            raise ValueError("OnTopCount needs finite three-dimensional AABBs")
        if any(a >= b for a, b in zip(lo, hi)):
            raise ValueError("OnTopCount needs positive AABB dimensions")
        bounds.append((lo, hi))
    (a, b), (c, d) = bounds
    dx, dy, dz = (b[i] - a[i] for i in range(3))
    denominator = max(dx * dy, dx * dz, dy * dz)
    intersection = max(0.0, min(b[0], d[0]) - max(a[0], c[0])) * max(
        0.0, min(b[1], d[1]) - max(a[1], c[1])
    )
    gap = a[2] - d[2]
    ratio = intersection / denominator
    return {
        "z_gap": gap,
        "area_ratio": ratio,
        "passed": abs(gap) <= 0.02 and ratio >= threshold,
    }


class OnTopCount(EvaluateAction):
    """Latch one score when enough distinct objects satisfy Ontop in one update.

    Unlike a WaitAll over latched Ontop children, earlier placements that have
    fallen off do not accumulate toward the later all-loaded milestone.
    """

    def __init__(
        self,
        env,
        objects: list[str],
        surface: str,
        minimum: int,
        threshold: float = 0.5,
    ):
        super().__init__(env)
        if (
            not isinstance(objects, list)
            or not objects
            or any(not isinstance(x, str) or not x for x in objects)
            or len(set(objects)) != len(objects)
        ):
            raise ValueError("OnTopCount requires distinct nonempty object names")
        if not isinstance(surface, str) or not surface or surface in objects:
            raise ValueError("OnTopCount requires a separate reference surface")
        if type(minimum) is not int or not 1 <= minimum <= len(objects):
            raise ValueError(
                "OnTopCount.minimum must be between one and the object count"
            )
        if (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or not 0 < threshold <= 1
        ):
            raise ValueError("OnTopCount.threshold must be finite and in (0, 1]")
        self.objects = tuple(objects)
        self.surface = surface
        self.minimum = minimum
        self.threshold = threshold
        self._done_flag = False

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        surface = self.get_obj_aabb_new(self.surface)
        observations = [
            {
                "object": obj,
                **ontop_evidence(self.get_obj_aabb_new(obj), surface, self.threshold),
            }
            for obj in self.objects
        ]
        matched = [x["object"] for x in observations if x["passed"]]
        self.progress_info["ONTOP_COUNT"] = {
            "minimum": self.minimum,
            "count": len(matched),
            "matched_objects": matched,
            "surface": self.surface,
            "observations": observations,
        }
        self._done_flag = len(matched) >= self.minimum
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        if event == ActionEvent.FINISHED:
            self.progress_info.update(SCORE=1, STATUS="SUCCESS")
