# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""URDF tree FK for selected robot chains, with an observation consistency check on every preview. No physics or IK."""

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent2act.adapters.telemetry import record_artifact
from agent2act.domain.command_bounds import wrist_envelope
from agent2act.domain.geometry import quaternion_angle
from agent2act.domain.models import ActionChunk, ContractError, Observation, RobotSpec, number, vector


def multiply(a: list[float], b: list[float]) -> list[float]:
    x, y, z, w = a
    u, v, t, s = b
    return [
        w * u + x * s + y * t - z * v,
        w * v - x * t + y * s + z * u,
        w * t + x * v - y * u + z * s,
        w * s - x * u - y * v - z * t,
    ]


def compose(a: list[float], b: list[float]) -> list[float]:
    rotated = multiply(multiply(a[3:], b[:3] + [0]), [-v for v in a[3:6]] + [a[6]])
    q = multiply(a[3:], b[3:])
    norm = math.hypot(*q)
    return [a[i] + rotated[i] for i in range(3)] + [v / norm for v in q]


def axis_rotation(axis: list[float], angle: float) -> list[float]:
    norm = math.hypot(*axis)
    if norm < 1e-12:
        raise ContractError("URDF moving joint has zero axis")
    return [v / norm * math.sin(angle / 2) for v in axis] + [math.cos(angle / 2)]


class UrdfKinematics:
    def __init__(self, spec: RobotSpec, options: dict[str, Any]) -> None:
        allowed = {
            "path",
            "frame",
            "left_link",
            "right_link",
            "translation_tolerance_m",
            "rotation_tolerance_rad",
            "subtree_path",
            "subtree_root",
        }
        if set(options) - allowed or not {"path", "frame", "left_link", "right_link"} <= set(options):
            raise ContractError("URDF kinematics requires path, frame and left/right links")
        self.spec = spec
        self.frame = options["frame"]
        if self.frame not in ("base_link", "arm_base_link"):
            raise ContractError("invalid kinematics frame")
        self.translation_tolerance = number(options.get("translation_tolerance_m", 0.005), "FK tolerance")
        self.rotation_tolerance = number(options.get("rotation_tolerance_rad", 0.02), "FK rotation tolerance")
        if not 0 < self.translation_tolerance <= 0.02 or not 0 < self.rotation_tolerance <= 0.1:
            raise ContractError("FK consistency tolerances out of range")
        source = Path(options["path"]).read_bytes()
        self.source_hash = hashlib.sha256(source).hexdigest()
        root = ET.fromstring(source)
        parents: dict[str, ET.Element] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is None or child.attrib["link"] in parents:
                raise ContractError("invalid URDF child tree")
            parents[child.attrib["link"]] = joint
        if ("subtree_path" in options) != ("subtree_root" in options):
            raise ContractError("subtree_path and subtree_root must be provided together")
        if "subtree_path" in options:
            subtree_source = Path(options["subtree_path"]).read_bytes()
            subtree = ET.fromstring(subtree_source)
            replacements = {}
            for j in subtree.findall("joint"):
                child = j.find("child")
                if child is None or child.attrib["link"] in replacements:
                    raise ContractError("invalid replacement joint tree")
                replacements[child.attrib["link"]] = j
            self.source_hash = hashlib.sha256(source + b"\n" + subtree_source).hexdigest()
            for side in ("left", "right"):
                link = options[side + "_link"]
                visited = set()
                while link != options["subtree_root"]:
                    if link in visited or link not in replacements:
                        raise ContractError("invalid replacement subtree")
                    visited.add(link)
                    joint = replacements[link]
                    parents[link] = joint
                    parent = joint.find("parent")
                    if parent is None:
                        raise ContractError("replacement joint has no parent")
                    link = parent.attrib["link"]
        self._joint_elements = {j.attrib["name"]: j for j in parents.values()}
        self.chains: dict[str, list[tuple[str, str, list[float], list[float]]]] = {}
        known = set(spec.left_joints + spec.right_joints + spec.body_joints)
        for side in ("left", "right"):
            link = options[side + "_link"]
            chain = []
            seen = set()
            while link != self.frame:
                if link in seen or link not in parents:
                    raise ContractError("URDF target is not a descendant of frame")
                seen.add(link)
                joint = parents[link]
                name, kind = joint.attrib["name"], joint.attrib["type"]
                if (
                    kind not in ("fixed", "revolute", "continuous", "prismatic")
                    or joint.find("mimic") is not None
                ):
                    raise ContractError("unsupported URDF joint on target chain")
                if kind != "fixed" and name not in known:
                    raise ContractError(f"URDF joint lacks measured state: {name}")
                origin = joint.find("origin")
                xyz = self._values(origin, "xyz", "0 0 0")
                rpy = self._values(origin, "rpy", "0 0 0")
                q = multiply(
                    multiply(axis_rotation([0, 0, 1], rpy[2]), axis_rotation([0, 1, 0], rpy[1])),
                    axis_rotation([1, 0, 0], rpy[0]),
                )
                chain.append((name, kind, xyz + q, self._values(joint.find("axis"), "xyz", "1 0 0")))
                parent = joint.find("parent")
                if parent is None:
                    raise ContractError("URDF joint has no parent")
                link = parent.attrib["link"]
            self.chains[side] = list(reversed(chain))
        self.closed = False

    @staticmethod
    def _values(element: ET.Element | None, key: str, default: str) -> list[float]:
        raw = default if element is None else element.get(key, default)
        return vector([float(v) for v in raw.split()], 3, "URDF origin/axis")

    def poses(self, joints: list[float], body: list[float]) -> dict[str, list[float]]:
        if self.closed:
            raise RuntimeError("kinematics is closed")
        names = self.spec.left_joints + self.spec.right_joints + self.spec.body_joints
        values = dict(zip(names, vector(joints + body, len(names), "FK joint values")))
        result = {}
        for side, chain in self.chains.items():
            transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            for name, kind, origin, axis in chain:
                transform = compose(transform, origin)
                if kind in ("revolute", "continuous"):
                    transform = compose(transform, [0.0, 0.0, 0.0] + axis_rotation(axis, values[name]))
                elif kind == "prismatic":
                    transform = compose(transform, [v * values[name] for v in axis] + [0.0, 0.0, 0.0, 1.0])
            result[side] = transform
        return result

    def joint_constraints(self, names: list[str]) -> tuple[list[list[float]], list[float]]:
        bounds, speeds = [], []
        for name in names:
            joint = self._joint_elements.get(name)
            limit = joint.find("limit") if joint is not None else None
            if limit is None or not {"lower", "upper", "velocity"} <= set(limit.attrib):
                raise ContractError(f"finite URDF position and velocity limits required: {name}")
            low, high, speed = [
                number(float(limit.attrib[k]), "URDF limit") for k in ("lower", "upper", "velocity")
            ]
            if low >= high or speed <= 0:
                raise ContractError(f"invalid URDF limits: {name}")
            bounds.append([low, high])
            speeds.append(speed)
        return bounds, speeds

    def check_observation(self, observation: Observation) -> dict[str, Any]:
        current = self.poses(observation.joints, observation.body)
        errors = {}
        for side, predicted in current.items():
            measured = observation.eef(side, self.frame)
            translation = math.dist(measured[:3], predicted[:3])
            rotation = quaternion_angle(measured[3:], predicted[3:])
            errors[side] = {"translation_m": translation, "rotation_rad": rotation}
            if translation > self.translation_tolerance or rotation > self.rotation_tolerance:
                raise ContractError(
                    f"FK observation mismatch: {side}, translation={translation:.6f} m, rotation={rotation:.6f} rad"
                )
        return errors

    def preview(self, observation: Observation, chunk: ActionChunk, steps: int) -> dict[str, Any]:
        if chunk.kind != "JOINT_ABS" or chunk.frame != self.frame:
            raise ContractError("FK preview requires joint actions in the configured frame")
        errors = self.check_observation(observation)
        # Include execution boundary and uniform samples; retain the full gripper sequence.
        indices = {0, chunk.steps - 1, min(steps, chunk.steps) - 1}
        indices.update(round(i * (chunk.steps - 1) / 9) for i in range(10))
        all_poses = [
            self.poses(
                chunk.left_arm[i] + chunk.right_arm[i], chunk.body[i] if chunk.body else observation.body
            )
            for i in range(chunk.steps)
        ]
        trajectory = []
        for i in sorted(indices):
            body = chunk.body[i] if chunk.body else observation.body
            trajectory.append(
                {
                    "step": i + 1,
                    "time_s": (i + 1) * chunk.dt_s,
                    **all_poses[i],
                    "grippers": chunk.grippers[i],
                    "body": body,
                }
            )
        result = {
            "frame": self.frame,
            "pose_format": "xyz metres, quaternion xyzw",
            "urdf_sha256": self.source_hash,
            "current_fk_errors": errors,
            "trajectory": trajectory,
            "command_envelope": wrist_envelope(
                observation,
                self.frame,
                [row["left"] for row in all_poses],
                [row["right"] for row in all_poses],
            ),
            "executable_command_envelope": wrist_envelope(
                observation,
                self.frame,
                [row["left"] for row in all_poses[:steps]],
                [row["right"] for row in all_poses[:steps]],
            ),
            "gripper_trajectory": chunk.grippers,
            "meaning": "Robot-only FK, not contact, object trajectory, collision checking or task success.",
        }
        artifact = "bounded-command-preview" if "command_bounds" in chunk.metadata else "policy-proposal"
        record_artifact(artifact, {"action": asdict(chunk), "preview": result})
        return result

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True
