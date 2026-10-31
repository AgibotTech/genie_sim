# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Compile Cartesian intent to a validated joint path using local robot geometry."""

import math
from contextlib import ExitStack
from dataclasses import asdict, replace
from typing import Any

from agent2act.adapters.kinematics.numerical import NumericalIK, pose_error
from agent2act.adapters.kinematics.urdf import UrdfKinematics
from agent2act.adapters.telemetry import record_artifact
from agent2act.domain.actions import body_start, gripper_start
from agent2act.domain.geometry import quaternion_angle, smooth_cruise, smoothstep
from agent2act.domain.models import (
    ActionChunk,
    ContractError,
    Decision,
    Observation,
    RobotSpec,
    normalized_pose,
    number,
    vector,
)
from agent2act.domain.ports import ActionExecutor
from agent2act.domain.timing import MotionTiming
from agent2act.domain.validation import ActionValidator


class UrdfJointMotion:
    mode = "motion"

    def __init__(
        self, spec: RobotSpec, timing: MotionTiming, options: dict[str, Any], cartesian: ActionExecutor
    ) -> None:
        self.closed = False
        with ExitStack() as rollback:
            self.cartesian = cartesian
            rollback.callback(self.cartesian.close)
            options = dict(options)
            waist = options.pop("waist_yaw_joint", None)
            # Optional acceptance of a stalled IK search close to the target.
            acceptance = {}
            for key, limit in (("accept_translation_m", 0.01), ("accept_rotation_rad", 0.05)):
                value = options.pop(key, None)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= limit
                ):
                    raise ContractError(f"{key} must be a positive number no larger than {limit}")
                acceptance[key] = None if value is None else float(value)
            if (acceptance["accept_translation_m"] is None) != (acceptance["accept_rotation_rad"] is None):
                raise ContractError("accept_translation_m and accept_rotation_rad must be set together")
            if waist is not None and (not isinstance(waist, str) or waist not in spec.body_joints):
                raise ContractError("waist_yaw_joint must name one configured robot body joint")
            self.waist_index = spec.body_joints.index(waist) if waist is not None else None
            speed = options.pop("waist_speed_rad_s", None)
            if speed is not None:
                speed = number(speed, "waist_speed_rad_s")
                if self.waist_index is None:
                    raise ContractError("waist_speed_rad_s requires waist_yaw_joint")
                if not 0 < speed <= spec.body_speed_limits[self.waist_index]:
                    raise ContractError(
                        "waist_speed_rad_s must be positive and within the robot body speed limit"
                    )
            self.waist_speed_rad_s = speed
            # Extra hold at the end of a waist-turning chunk so a slow torso drive can
            # settle before the next observation; later chunks hold the measured waist.
            settle = options.pop("waist_settle_s", 0.0)
            if isinstance(settle, bool) or not isinstance(settle, (int, float)) or not 0 <= settle <= 10:
                raise ContractError("waist_settle_s must be a number in [0, 10]")
            if settle and waist is None:
                raise ContractError("waist_settle_s requires waist_yaw_joint")
            self.waist_settle_s = float(settle)
            self.fk = UrdfKinematics(spec, options)
            rollback.callback(self.fk.close)
            names = spec.left_joints + spec.right_joints
            bounds, velocities = self.fk.joint_constraints(names)
            if spec.joint_limits is not None:
                bounds = [[max(a, c), min(b, d)] for (a, b), (c, d) in zip(bounds, spec.joint_limits)]
            if any(a >= b for a, b in bounds):
                raise ContractError("configured and URDF joint intervals do not intersect")
            self.spec = replace(spec, joint_limits=bounds)
            self.speeds = [min(spec.max_joint_speed, v) * timing.speed_fraction for v in velocities]
            self.ik = NumericalIK(self.fk, bounds, **acceptance, waist_control=waist is not None)
            self.timing = timing
            rollback.pop_all()

    def describe(self) -> dict[str, Any]:
        result = self._describe()
        if self.waist_index is not None:
            low, high = self.spec.body_limits[self.waist_index]
            result["waist_control"] = {
                "waypoint_field": "waist_yaw",
                "joint": self.spec.body_joints[self.waist_index],
                "state_index": self.waist_index,
                "state": "states.waist_joint_states",
                "units": "absolute radians",
                "limits_rad": [low, high],
                "speed_rad_s": (
                    self.waist_speed_rad_s
                    if self.waist_speed_rad_s is not None
                    else self.spec.body_speed_limits[self.waist_index]
                )
                * self.timing.speed_fraction,
                "settle_hold_s": self.waist_settle_s,
                "positive_direction": "counterclockwise seen from above (turn left); negative turns right",
                "meaning": "Rotating the waist rotates the torso, both arms, and the head camera about the "
                "vertical axis. base_link does not rotate. A decision containing waist_yaw is compiled in "
                "joint space: an omitted arm keeps its joint angles and rides with the torso; a given pose is "
                "a base_link target reached at the new waist angle, not along a straight line.",
            }
        return result

    def _describe(self) -> dict[str, Any]:
        return {
            "mode": "motion",
            "input": "Cartesian waypoint endpoints; solve IK once per commanded arm and waypoint, then interpolate joints",
            "compiler": {
                "backend": "urdf_joint",
                "endpoint_ik": True,
                "frame": self.fk.frame,
                "operating_timing": asdict(self.timing),
                "speed_limits_enforced": self.spec.enforce_speed_limits,
                "joint_limits": self.spec.joint_limits,
                "joint_speed_limits": self.speeds,
                "ik_acceptance": {
                    "accept_translation_m": self.ik.accept_translation_m,
                    "accept_rotation_rad": self.ik.accept_rotation_rad,
                },
                "measured_start_recovery": "A slightly out-of-bounds measurement may receive one explicit in-bounds first command within existing IK pose accuracy, with speed checks when enabled. Measurements and joint bounds are never changed; all commands remain validated.",
                "collision_checking": False,
                "path_interpolation": "joint_space",
                "meaning": "Endpoint IK with bounded joints; speed limits extend timing only when speed_limits_enforced is true. Otherwise moving targets without duration_s use operating_timing.default_steps. Unreachable waypoint endpoints are rejected; Cartesian straight paths are not required. No collision or dynamics simulation.",
            },
        }

    def reset(self) -> None:
        self.fk.reset()
        self.cartesian.reset()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.cartesian.close()
            finally:
                self.fk.close()

    def _bounded_start(self, observation: Observation) -> tuple[list[float], dict[str, Any] | None]:
        """Propose one admissible command without rewriting the measured starting state."""
        assert self.spec.joint_limits is not None
        bounded = [
            min(high, max(low, q)) for q, (low, high) in zip(observation.joints, self.spec.joint_limits)
        ]
        if bounded == observation.joints:
            return bounded, None
        projected = list(bounded)
        before = self.fk.poses(observation.joints, observation.body)
        # Projection supplies only a feasible seed. Redundant, unsaturated joints
        # may restore the original FK pose without widening any interval or tolerance.
        try:
            for side in ("left", "right"):
                bounded = self.ik.solve(before[side], bounded, observation.body, side)
        except ContractError as exc:
            raise ContractError(
                "local IK start is outside joint bounds; no pose-preserving bounded seed found"
            ) from exc
        if self.spec.enforce_speed_limits and any(
            abs(b - a) > speed * self.spec.dt_s + 1e-9
            for a, b, speed in zip(observation.joints, bounded, self.speeds)
        ):
            raise ContractError("local IK start is outside joint bounds beyond one admissible command")
        after = self.fk.poses(bounded, observation.body)
        changes = {}
        for side in ("left", "right"):
            translation = math.dist(before[side][:3], after[side][:3])
            rotation = math.hypot(*pose_error(before[side], after[side])[3:])
            dt = self.spec.dt_s * self.timing.speed_fraction
            if (
                translation > self.ik.position_tolerance_m
                or rotation > self.ik.rotation_tolerance_rad
                or (
                    self.spec.enforce_speed_limits
                    and (
                        translation > self.spec.max_eef_speed * dt + 1e-9
                        or rotation > self.spec.max_eef_angular_speed * dt + 1e-9
                    )
                )
            ):
                raise ContractError("local IK start is outside joint bounds beyond pose-preserving recovery")
            changes[side] = {"translation_m": translation, "rotation_rad": rotation}
        return bounded, {
            "observed_joints": list(observation.joints),
            "projected_seed": projected,
            "first_command": bounded,
            "eef_changes": changes,
            "meaning": "One explicit in-bounds command within existing IK pose accuracy, with speed checks when enabled; original measurement is unchanged. Not observed recovery or contact preservation.",
        }

    def generate(self, observation: Observation, decision: Decision) -> ActionChunk:
        if self.closed:
            raise RuntimeError("motion compiler is closed")
        if not self.spec.supports_eef:
            raise ContractError("motion requires EEF-capable robot")
        motion = decision.motion
        if not isinstance(motion, dict) or set(motion) != {"frame", "waypoints"}:
            raise ContractError("motion requires exactly frame and waypoints")
        if motion["frame"] != self.fk.frame:
            raise ContractError("local IK requires the configured frame")
        waypoints = motion["waypoints"]
        if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= self.spec.max_chunk_steps:
            raise ContractError("motion needs a bounded, nonempty waypoint list")
        errors = self.fk.check_observation(observation)
        q, recovery = self._bounded_start(observation)
        grips, metadata = gripper_start(observation, self.spec)
        body = body_start(observation, self.spec)
        dt = self.spec.dt_s
        fraction = self.timing.speed_fraction
        body_speeds = [speed * fraction for speed in self.spec.body_speed_limits]
        if self.waist_index is not None and self.waist_speed_rad_s is not None:
            body_speeds[self.waist_index] = self.waist_speed_rad_s * fraction
        result = ActionChunk("JOINT_ABS", self.fk.frame, dt, [], [], [], dict(metadata))
        n = len(self.spec.left_joints)
        path: dict[str, list[list[float]]] = {"left": [], "right": []}

        def append(row_q: list[float], row_body: list[float], row_grips: list[float]) -> None:
            poses = self.fk.poses(row_q, row_body)
            for side in path:
                path[side].append(poses[side])
            result.left_arm.append(row_q[:n])
            result.right_arm.append(row_q[n:])
            result.grippers.append(list(row_grips))
            if self.spec.body_joints:
                result.body.append(list(row_body))

        if recovery is not None:
            append(q, body, grips)
            result.metadata["joint_bound_recovery"] = recovery
        progress, peak = {
            "quintic": (smoothstep, 1.875),
            "smooth_cruise": (smooth_cruise, 1.25),
        }[self.timing.profile]
        for waypoint in waypoints:
            if not isinstance(waypoint, dict) or set(waypoint) - {
                "duration_s",
                "left_pose",
                "right_pose",
                "grippers",
                "waist_yaw",
            }:
                raise ContractError("unknown waypoint fields; unsupported constraints cannot be ignored")
            target_body = list(body)
            if "waist_yaw" in waypoint:
                if self.waist_index is None:
                    raise ContractError(
                        "unknown waypoint fields: waist_yaw requires configured waist control"
                    )
                low, high = self.spec.body_limits[self.waist_index]
                yaw = number(waypoint["waist_yaw"], "waypoint waist_yaw")
                if not low <= yaw <= high:
                    raise ContractError(
                        f"waist_yaw {yaw} rad is outside the joint limits [{low}, {high}] rad"
                    )
                target_body[self.waist_index] = yaw
            target_grips = (
                vector(waypoint["grippers"], 2, "waypoint grippers")
                if "grippers" in waypoint
                else list(grips)
            )
            target_q = list(q)
            for side in ("left", "right"):
                if side + "_pose" in waypoint:
                    target = normalized_pose(waypoint[side + "_pose"], side + "_pose")
                    target_q = self.ik.solve(target, target_q, target_body, side)
            duration = number(waypoint.get("duration_s", dt), "waypoint duration_s")
            if duration <= 0 or duration / dt > self.spec.max_chunk_steps:
                raise ContractError("waypoint duration out of range")
            moving = target_q != q or target_body != body
            minimum = max(
                self.timing.min_duration_s if moving else 0,
                self.timing.min_gripper_duration_s if target_grips != grips else 0,
            )
            # Size steps for the selected profile's peak/average speed ratio.
            steps = max(1, math.ceil(duration / dt - 1e-9), math.ceil(minimum / dt - 1e-9))
            if (
                not self.spec.enforce_speed_limits
                and "duration_s" not in waypoint
                and (moving or target_grips != grips)
            ):
                steps = max(steps, self.timing.default_steps)
            limits = (
                (q, target_q, self.speeds),
                (body, target_body, body_speeds),
                (grips, target_grips, [self.spec.max_gripper_speed * fraction] * 2),
            )
            for start, end, speeds in limits if self.spec.enforce_speed_limits else ():
                for a, b, speed in zip(start, end, speeds):
                    steps = max(steps, math.ceil(peak * abs(b - a) / (speed * dt) - 1e-9))
            while True:
                if result.steps + steps > self.spec.max_chunk_steps:
                    raise ContractError("local IK motion exceeds max_chunk_steps after joint timing")
                rows = []
                # End-effector speed limits the arms' own joint motion only: FK at the starting
                # body pose, so carrying an arm with the torso is paced by body_speed_limits alone.
                previous = self.fk.poses(q, body)
                valid = True
                for step in range(1, steps + 1):
                    f = progress(step / steps)
                    row = (
                        [a + (b - a) * f for a, b in zip(q, target_q)],
                        [a + (b - a) * f for a, b in zip(body, target_body)],
                        [a + (b - a) * f for a, b in zip(grips, target_grips)],
                    )
                    current = self.fk.poses(row[0], body)
                    for side in ("left", "right"):
                        old, new = previous[side], current[side]
                        if self.spec.enforce_speed_limits and (
                            math.dist(old[:3], new[:3]) > self.spec.max_eef_speed * fraction * dt + 1e-9
                            or quaternion_angle(old[3:], new[3:])
                            > self.spec.max_eef_angular_speed * fraction * dt + 1e-9
                        ):
                            valid = False
                    previous = current
                    rows.append(row)
                if valid:
                    break
                steps *= 2
            for row in rows:
                append(*row)
            q, body, grips = target_q, target_body, target_grips
        motion_steps = result.steps
        hold_s = self.timing.target_hold_s
        if body != body_start(observation, self.spec):
            hold_s = max(hold_s, self.waist_settle_s)
        hold_steps = math.ceil(hold_s / dt - 1e-9)
        if motion_steps + hold_steps > self.spec.max_chunk_steps:
            raise ContractError("motion exceeds robot max_chunk_steps after target hold")
        for _ in range(hold_steps):
            append(q, body, grips)
        result.metadata.update(
            compiler="urdf_joint",
            motion_steps=motion_steps,
            target_hold_steps=hold_steps,
            timing_profile=self.timing.profile + "_joint_space",
            speed_limits_enforced=self.spec.enforce_speed_limits,
            cartesian_path=path,
            local_ik={
                "source_steps": len(waypoints),
                "joint_steps": result.steps,
                "urdf_sha256": self.fk.source_hash,
                "current_fk_errors": errors,
                "generated_endpoint_fk": self.fk.poses(q, body),
                "meaning": "Waypoint endpoints solved by IK; intervening commands interpolate joints, without Cartesian path IK. Full generated path before runtime prefix selection, not observed endpoint arrival or object success. No collision or dynamics check.",
            },
        )
        if self.waist_index is not None:
            result.metadata["waist_motion"] = {
                "joint": self.spec.body_joints[self.waist_index],
                "start_rad": body_start(observation, self.spec)[self.waist_index],
                "measured_start_rad": observation.body[self.waist_index],
                "target_rad": body[self.waist_index],
                "meaning": "Commanded waist yaw; actual rotation must be checked in the next observation.",
            }
        ActionValidator(self.spec).validate(result, observation)
        record_artifact("motion-local-ik", {"action": asdict(result), "waypoints": waypoints})
        return result
