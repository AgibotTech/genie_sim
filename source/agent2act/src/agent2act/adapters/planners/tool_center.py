# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Translate the model's fixed tool reference without changing execution coordinates."""

import math
from copy import deepcopy
from typing import Any

from agent2act.domain.models import ContractError, Decision, normalized_pose, vector

from .geometry_tools import rotate_vector

CENTER_INSTRUCTIONS = (
    "All model-facing EEF/end_pose, waypoint poses and Cartesian feedback refer to the fixed "
    "nominal gripper center (TCP), with axes parallel to the wrist. Output center targets directly; "
    "Agent2Act automatically converts them to wrist commands. Do not apply a wrist offset yourself. "
    "This reference is not a measured contact point or object center. The motion backend may interpolate "
    "joints rather than a Cartesian line; the wrist and center paths need not be straight."
)


class ToolCenter:
    def __init__(self, offsets: dict[str, list[float]]) -> None:
        if not isinstance(offsets, dict) or set(offsets) != {"left", "right"}:
            raise ContractError("gripper_center_offset_m requires left and right")
        self.offsets = {arm: vector(value, 3, "center offset") for arm, value in offsets.items()}

    def convert(self, value: list[float], arm: str, *, inverse: bool = False) -> list[float]:
        value = normalized_pose(value)
        offset = rotate_vector(value[3:], self.offsets[arm])
        sign = -1 if inverse else 1
        return [p + sign * d for p, d in zip(value[:3], offset)] + value[3:]

    def end_pose(self, source: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(source)
        for arms in result.values():
            for arm in self.offsets:
                key = arm + "_arm"
                if key in arms:
                    entry = arms[key]
                    converted = self.convert(entry["position"] + entry["orientation"], arm)
                    entry.update(position=converted[:3], orientation=converted[3:])
        return result

    def measured_motion(self, source: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(source)
        for arm, metrics in result.items():
            if "previous_pose" not in metrics or "observed_pose" not in metrics:
                metrics.pop("translation_delta_m", None)
                continue
            for key in ("previous_pose", "observed_pose"):
                metrics[key] = self.convert(metrics[key], arm)
            metrics["translation_delta_m"] = [
                a - b for a, b in zip(metrics["observed_pose"][:3], metrics["previous_pose"][:3])
            ]
        return result

    def tracking(self, metrics: dict[str, Any], arm: str) -> dict[str, Any]:
        """Re-derive endpoint tracking at the TCP; never relabel wrist deltas as TCP."""
        converted: dict[str, Any] = {
            key: self.convert(metrics[key], arm)
            for key in ("previous_pose", "observed_pose", "issued_pose")
            if key in metrics
        }
        if "rotation_error_rad" in metrics:
            converted["rotation_error_rad"] = metrics["rotation_error_rad"]
        if "issued_pose" in converted:
            remaining = [b - a for a, b in zip(converted["observed_pose"][:3], converted["issued_pose"][:3])]
            converted.update(remaining_delta_m=remaining, translation_error_m=math.hypot(*remaining))
            if "previous_pose" in converted:
                before = converted["previous_pose"][:3]
                commanded = [b - a for a, b in zip(before, converted["issued_pose"][:3])]
                observed = [b - a for a, b in zip(before, converted["observed_pose"][:3])]
                distance = math.hypot(*commanded)
                along = (
                    sum(a * b for a, b in zip(observed, commanded)) / distance if distance > 1e-4 else None
                )
                converted.update(
                    commanded_delta_m=commanded,
                    observed_delta_m=observed,
                    commanded_displacement_m=distance,
                    observed_displacement_m=math.hypot(*observed),
                    along_track_displacement_m=along,
                    along_track_fraction=along / distance if along is not None else None,
                    cross_track_error_m=math.hypot(
                        *[a - along * b / distance for a, b in zip(observed, commanded)]
                    )
                    if along is not None
                    else None,
                )
        return converted

    def context(self, source: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(source)
        feedback = result.get("tracking_feedback", {})
        if feedback.get("kind") == "EEF_ABS" or feedback.get("cartesian_target_source") == "issued_joint_fk":
            endpoint = feedback.get("generated_endpoint", {})
            for arm in self.offsets:
                key = arm + "_arm"
                if key in endpoint:
                    endpoint[key] = self.convert(endpoint[key], arm)
        for arm in self.offsets:
            metrics = feedback.get(arm)
            if not metrics or "observed_pose" not in metrics:
                continue
            feedback[arm] = self.tracking(metrics, arm)
        if feedback:
            feedback["measured_eef_motion"] = self.measured_motion(feedback.get("measured_eef_motion", {}))
            feedback.pop("summary", None)
            feedback["meaning"] = (
                "Fixed gripper-center tracking only; issued commands do not prove playback, contact or success."
            )
        for event in result.get("recent_events", []):
            endpoint = event.get("issued_endpoint", {})
            if event.get("kind") == "EEF_ABS":
                for arm in self.offsets:
                    key = arm + "_arm"
                    if key in endpoint:
                        endpoint[key] = self.convert(endpoint[key], arm)
            effect = event.get("observed_effect", {})
            if "measured_eef_motion" in effect:
                effect["measured_eef_motion"] = self.measured_motion(effect["measured_eef_motion"])
            if "tracking" in effect:
                effect["tracking"] = {arm: self.tracking(m, arm) for arm, m in effect["tracking"].items()}
        return result

    def decision(self, decision: Decision) -> Decision:
        result = deepcopy(decision)
        if result.motion is not None:
            field = "candidates" if "candidates" in result.motion else "waypoints"
            waypoints = result.motion.get(field)
            if not isinstance(waypoints, list) or not waypoints:
                raise ContractError("center motion requires nonempty waypoints")
            for waypoint in waypoints:
                if not isinstance(waypoint, dict):
                    raise ContractError("center waypoint must be an object")
                for arm in self.offsets:
                    key = arm + "_pose"
                    if key in waypoint:
                        waypoint[key] = self.convert(waypoint[key], arm, inverse=True)
        return result
