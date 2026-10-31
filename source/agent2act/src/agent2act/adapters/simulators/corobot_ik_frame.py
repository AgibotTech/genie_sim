# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Reframe CoRobot EEF commands for receivers that assume the IK frame.

Uses paired measured-EEF / model-FK poses, never a guessed base offset; both arms must agree on
the same rigid transform each request."""

import math
from copy import deepcopy
from typing import Any

from agent2act.adapters.kinematics.urdf import UrdfKinematics, compose
from agent2act.adapters.telemetry import record_artifact
from agent2act.domain.geometry import quaternion_angle
from agent2act.domain.models import ActionChunk, ContractError, Observation, RobotSpec

from .corobot import CoRobotCodec, encode_result


def inverse_pose(pose: list[float]) -> list[float]:
    rotation = [-v for v in pose[3:6]] + [pose[6]]
    return compose([0.0, 0.0, 0.0] + rotation, [-v for v in pose[:3]] + [0.0, 0.0, 0.0, 1.0])


class CoRobotIKFrameCodec(CoRobotCodec):
    def __init__(self, spec: RobotSpec, options: dict[str, Any]) -> None:
        if set(options) != {"path", "left_link", "right_link"}:
            raise ContractError("corobot_ik_frame requires path, left_link and right_link")
        super().__init__(spec, {})
        self.fk = UrdfKinematics(spec, {**options, "frame": "arm_base_link"})

    def _transform(self, observation: Observation) -> list[float]:
        predicted = self.fk.poses(observation.joints, observation.body)
        transforms = {
            side: compose(predicted[side], inverse_pose(observation.eef(side, "base_link")))
            for side in ("left", "right")
        }
        left, right = transforms["left"], transforms["right"]
        translation = math.dist(left[:3], right[:3])
        rotation = quaternion_angle(left[3:], right[3:])
        if translation > 0.001 or rotation > 0.005:
            raise ContractError(
                f"CoRobot frame mismatch between arms: {translation:.6f} m, {rotation:.6f} rad; "
                "check IK URDF, end links and observation joint order"
            )
        return left

    def decode(self, request: dict) -> Observation:
        observation = super().decode(request)
        # Fail before any model call or runtime action when the frame cannot be established.
        self._transform(observation)
        return observation

    def encode(
        self, chunk: ActionChunk, observation: Observation, *, history: dict, need_depth: bool
    ) -> dict:
        if chunk.kind != "EEF_ABS":
            return super().encode(chunk, observation, history=history, need_depth=need_depth)
        if chunk.frame != "base_link":
            raise ContractError("corobot_ik_frame requires internal EEF actions in base_link")
        if chunk.body and any(row != observation.body for row in chunk.body):
            raise ContractError("corobot_ik_frame cannot reframe EEF commands with moving body targets")
        transform = self._transform(observation)
        wire = deepcopy(chunk)
        wire.left_arm = [compose(transform, pose) for pose in chunk.left_arm]
        wire.right_arm = [compose(transform, pose) for pose in chunk.right_arm]
        wire.frame = "arm_base_link"
        wire.metadata["wire_frame_conversion"] = {
            "source_frame": chunk.frame,
            "target_frame": wire.frame,
            "transform_xyzw": transform,
            "urdf_sha256": self.fk.source_hash,
        }
        record_artifact("wire-frame-conversion", wire.metadata["wire_frame_conversion"])
        return encode_result(wire, history=history, need_depth=need_depth)

    def reset(self) -> None:
        self.fk.reset()

    def close(self) -> None:
        self.fk.close()
