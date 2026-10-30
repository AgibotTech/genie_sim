# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Prepare one wrist-pose-preserving, IK-compatible pose for startup and reset."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import yaml

# RelaxedIKSQP::solve in the shipped MC uses this bound, independently of the
# trajectory filter's limits. Leave 0.02 rad for the first small tracked motion.
WRIST_LIMIT = np.pi / 2 - 0.3
RESET_WRIST_LIMIT = WRIST_LIMIT - 0.02
MODEL_DIR = Path(__file__).resolve().parents[1] / "app/configuration/robot/G2/model"


class ArmResetModel:
    """Seven arm joints and the fixed transforms up to the MC end frame."""

    def __init__(self, urdf_path, side):
        if side not in ("left", "right"):
            raise ValueError(f"Unknown arm side: {side}")
        root = ET.parse(urdf_path).getroot()
        by_child = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
        frame = f"arm_{side[0]}_end_link"
        chain = []
        seen = set()
        while frame != "arm_base_link":
            if frame in seen or frame not in by_child:
                raise ValueError(f"No arm_base_link chain for {side} arm in {urdf_path}")
            seen.add(frame)
            joint = by_child[frame]
            chain.append(joint)
            frame = joint.find("parent").get("link")

        model = yaml.safe_load((MODEL_DIR / f"{side}_arm.yaml").read_text())
        settings = {joint["name"]: joint for group in model["joints"] for joint in group}
        self.transforms = []
        self.names = []
        bounds = []
        for joint in reversed(chain):
            origin = joint.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                transform[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            axis = None
            if joint.get("type") != "fixed":
                name = joint.get("name")
                if joint.get("type") != "revolute" or name not in settings:
                    raise ValueError(f"Unsupported MC arm joint: {name}")
                self.names.append(name)
                axis_element = joint.find("axis")
                axis = np.fromstring(axis_element.get("xyz") if axis_element is not None else "1 0 0", sep=" ")
                axis = axis / np.linalg.norm(axis)
                limit = joint.find("limit")
                # Match prepare_mc's URDF limits and IKObjective's thresholds.
                margin = float(settings[name]["position_threshold"])
                bounds.append((float(limit.get("lower")) + margin, float(limit.get("upper")) - margin))
            self.transforms.append((transform, axis))
        expected = [f"idx{2 if side == 'left' else 6}{i}_arm_{side[0]}_joint{i}" for i in range(1, 8)]
        if self.names != expected:
            raise ValueError(f"Unexpected {side} arm joint order: {self.names}")
        self.bounds = np.asarray(bounds)

    def fk(self, joints):
        result = np.eye(4)
        index = 0
        for origin, axis in self.transforms:
            result = result @ origin
            if axis is not None:
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(axis * joints[index]).as_matrix()
                result = result @ rotation
                index += 1
        return result

    @staticmethod
    def wrist_slack(joints):
        return RESET_WRIST_LIMIT - np.abs([joints[5] + joints[6], joints[5] - joints[6]])

    def prepare(self, joints):
        original = np.asarray(joints, dtype=float)
        if original.shape != (7,) or not np.all(np.isfinite(original)):
            raise ValueError("Arm reset requires seven finite joint positions")
        lower, upper = self.bounds.T
        if np.all(original >= lower) and np.all(original <= upper) and np.all(self.wrist_slack(original) >= 0):
            return original.tolist()
        target = self.fk(original)

        def pose_error(q):
            actual = self.fk(q)
            return np.r_[
                actual[:3, 3] - target[:3, 3],
                Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec(),
            ]

        result = minimize(
            lambda q: np.sum((q - original) ** 2),
            np.clip(original, lower, upper),
            jac=lambda q: 2 * (q - original),
            method="SLSQP",
            bounds=self.bounds,
            constraints=[{"type": "eq", "fun": pose_error}, {"type": "ineq", "fun": self.wrist_slack}],
            options={"ftol": 1e-12, "maxiter": 200},
        )
        q = result.x
        error = pose_error(q)
        if (
            not result.success
            or not np.all(np.isfinite(q))
            or np.linalg.norm(error[:3]) > 1e-7
            or np.linalg.norm(error[3:]) > 1e-7
            or np.any(q < lower - 1e-8)
            or np.any(q > upper + 1e-8)
            or np.any(self.wrist_slack(q) < -1e-8)
        ):
            raise ValueError(f"Cannot prepare an MC-compatible reset without moving the wrist: {result.message}")
        return q.tolist()


def prepare_arm_reset(urdf_path, joints):
    if len(joints) != 14:
        raise ValueError("Dual-arm reset requires 14 joint positions")
    return ArmResetModel(urdf_path, "left").prepare(joints[:7]) + ArmResetModel(urdf_path, "right").prepare(joints[7:])
