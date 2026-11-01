"""Small helpers for resolving CuRobo goal-pose inputs."""

from __future__ import annotations


def select_additional_goal_base(
    target_position,
    target_orientation,
    current_position,
    current_orientation,
    *,
    from_current_pose: bool,
):
    """Select the pose to which an additional tool's offset is applied."""
    if from_current_pose:
        return current_position, current_orientation
    return target_position, target_orientation
