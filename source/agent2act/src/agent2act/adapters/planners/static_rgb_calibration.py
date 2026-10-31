# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Authored G2 RGB calibration with observation-time wrist transforms."""

import hashlib
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any

from agent2act.adapters.kinematics.urdf import axis_rotation, compose, multiply
from agent2act.domain.models import ContractError, Observation, number, vector

from .geometry_tools import rotate_vector


class StaticRgbCalibration:
    """Validate robot-asset calibration; never claim renderer-frame validation."""

    def __init__(self, entries: dict[str, Any]) -> None:
        if not isinstance(entries, dict) or set(entries) != {"head", "hand_left", "hand_right"}:
            raise ContractError("RGB calibration requires head, hand_left and hand_right")
        self.entries: dict[str, dict[str, Any]] = {}
        for camera, entry in entries.items():
            mount = "head_link3" if camera == "head" else f"arm_{camera.removeprefix('hand_')[0]}_end_link"
            expected = {"intrinsics", "camera_to_mount", "mount_frame"}
            if camera == "head":
                expected.add("kinematics_urdf")
            if not isinstance(entry, dict) or set(entry) != expected:
                raise ContractError(f"{camera} requires intrinsics, camera_to_mount and mount_frame")
            if entry["mount_frame"] != mount:
                raise ContractError(f"{camera} camera mount frame must be {mount}")
            k = entry["intrinsics"]
            keys = {"width", "height", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"}
            if not isinstance(k, dict) or set(k) != keys:
                raise ContractError(f"{camera} requires complete OpenCV pinhole intrinsics")
            for key in keys:
                number(k[key], f"{camera}.{key}")
            if any(type(k[key]) is not int or not 2 <= k[key] <= 8192 for key in ("width", "height")):
                raise ContractError(f"{camera} invalid calibration dimensions")
            if min(k["fx"], k["fy"]) <= 0:
                raise ContractError(f"{camera} focal lengths must be positive")
            matrix = entry["camera_to_mount"]
            if (
                not isinstance(matrix, list)
                or len(matrix) != 4
                or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
            ):
                raise ContractError(f"{camera} camera_to_mount must be 4x4")
            matrix = [[number(value, f"{camera}.camera_to_mount") for value in row] for row in matrix]
            if any(abs(a - b) > 1e-7 for a, b in zip(matrix[3], [0, 0, 0, 1])):
                raise ContractError(f"{camera} camera_to_mount must be homogeneous")
            columns = [[matrix[row][col] for row in range(3)] for col in range(3)]
            if any(
                abs(sum(a * b for a, b in zip(columns[i], columns[j])) - (1 if i == j else 0)) > 1e-5
                for i in range(3)
                for j in range(3)
            ):
                raise ContractError(f"{camera} camera_to_mount must have an orthonormal rotation")
            cross = [
                columns[0][1] * columns[1][2] - columns[0][2] * columns[1][1],
                columns[0][2] * columns[1][0] - columns[0][0] * columns[1][2],
                columns[0][0] * columns[1][1] - columns[0][1] * columns[1][0],
            ]
            if abs(sum(a * b for a, b in zip(cross, columns[2])) - 1) > 1e-5:
                raise ContractError(f"{camera} camera_to_mount must be right-handed")
            self.entries[camera] = deepcopy(entry)
        self._head_chain, self._head_urdf_sha256 = self._load_head_chain(
            self.entries["head"]["kinematics_urdf"]
        )

    @staticmethod
    def _load_head_chain(path: Any) -> tuple[list[tuple[str, list[float], list[float]]], str]:
        if not isinstance(path, str) or not path.strip():
            raise ContractError("head kinematics_urdf must be a path")
        try:
            source = Path(path).read_bytes()
            root = ET.fromstring(source)
        except (OSError, ET.ParseError) as exc:
            raise ContractError("head kinematics_urdf is unreadable or invalid") from exc
        parents = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None:
                if child.get("link") in parents:
                    raise ContractError("head URDF has duplicate child links")
                parents[child.get("link")] = joint
        expected_names = [f"idx0{i}_body_joint{i}" for i in range(1, 6)] + [
            f"idx1{i}_head_joint{i}" for i in range(1, 4)
        ]
        joints = []
        link = "head_link3"
        seen = set()
        while link != "base_link":
            if link in seen or link not in parents:
                raise ContractError("head_link3 is not connected to base_link in URDF")
            seen.add(link)
            joint = parents[link]
            if joint.get("type") != "revolute" or joint.find("mimic") is not None:
                raise ContractError("head chain must use measured revolute joints")
            origin = joint.find("origin")
            axis = joint.find("axis")
            parent = joint.find("parent")
            name = joint.get("name")
            if origin is None or axis is None or parent is None or name is None:
                raise ContractError("head URDF joint lacks origin, axis or parent")
            try:
                xyz = vector([float(v) for v in origin.get("xyz", "0 0 0").split()], 3, "head origin")
                rpy = vector([float(v) for v in origin.get("rpy", "0 0 0").split()], 3, "head rpy")
                joint_axis = vector([float(v) for v in axis.get("xyz", "").split()], 3, "head axis")
            except ValueError as exc:
                raise ContractError("head URDF has invalid numeric joint fields") from exc
            q = multiply(
                multiply(axis_rotation([0, 0, 1], rpy[2]), axis_rotation([0, 1, 0], rpy[1])),
                axis_rotation([1, 0, 0], rpy[0]),
            )
            joints.append(
                (
                    name,
                    xyz + q,
                    joint_axis,
                )
            )
            next_link = parent.get("link")
            if next_link is None:
                raise ContractError("head URDF joint lacks parent link")
            link = next_link
        joints.reverse()
        if [name for name, _, _ in joints] != expected_names:
            raise ContractError("head URDF joint order differs from measured state order")
        return joints, hashlib.sha256(source).hexdigest()

    def head_pose(self, state_groups: dict[str, Any]) -> list[float]:
        waist = vector(state_groups.get("waist_joint_states"), 5, "waist_joint_states")
        head = vector(state_groups.get("head_joint_states"), 3, "head_joint_states")
        values = waist + head
        transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        for (name, origin, axis), angle in zip(self._head_chain, values):
            transform = compose(transform, origin)
            transform = compose(transform, [0.0, 0.0, 0.0] + axis_rotation(axis, angle))
        return transform

    @staticmethod
    def _optical_to_base(entry: dict[str, Any], mount_pose: list[float]) -> list[list[float]]:
        authored = entry["camera_to_mount"]
        axis_sign = [1, -1, -1]
        columns = [
            rotate_vector(mount_pose[3:], [authored[row][col] * axis_sign[col] for row in range(3)])
            for col in range(3)
        ]
        origin = rotate_vector(mount_pose[3:], [authored[row][3] for row in range(3)])
        return [
            [columns[col][row] for col in range(3)] + [mount_pose[row] + origin[row]] for row in range(3)
        ] + [[0.0, 0.0, 0.0, 1.0]]

    def for_observation(self, observation: Observation) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for camera, entry in self.entries.items():
            metadata: dict[str, Any] = {
                "source": "authored static robot USD; effective renderer calibration unverified",
                "model": "opencv_pinhole",
                "intrinsics": deepcopy(entry["intrinsics"]),
                "intrinsics_size": [entry["intrinsics"]["width"], entry["intrinsics"]["height"]],
                "mount_frame": entry["mount_frame"],
                "camera_to_mount_usd": deepcopy(entry["camera_to_mount"]),
                "camera_to_mount_axes": "+x right, +y up, +z backward; column vectors; metres",
                "metric_projection_ready": False,
                "validation": "No same-frame render-pose or image reprojection check; do not assert metric object position from this metadata alone.",
            }
            if camera != "head":
                arm = camera.removeprefix("hand_")
                try:
                    wrist = observation.eef(arm, "base_link")
                except ContractError:
                    metadata["base_extrinsic_reason"] = "current wrist pose in base_link unavailable"
                else:
                    metadata["optical_to_base_link"] = self._optical_to_base(entry, wrist)
                    metadata["optical_axes"] = "+x right, +y down, +z forward"
                    metadata["extrinsic_basis"] = (
                        "current observed wrist pose in base_link × static camera-to-wrist"
                    )
            else:
                try:
                    head_pose = self.head_pose(observation.state_groups)
                except ContractError:
                    metadata["base_extrinsic_reason"] = "measured head/waist joint states unavailable"
                else:
                    metadata["optical_to_base_link"] = self._optical_to_base(entry, head_pose)
                    metadata["optical_axes"] = "+x right, +y down, +z forward"
                    metadata["extrinsic_basis"] = (
                        "current measured waist/head joints through configured URDF × static camera-to-head"
                    )
                    metadata["head_kinematics_urdf_sha256"] = self._head_urdf_sha256
            result[camera] = metadata
        return result

    @staticmethod
    def image_metadata(
        metadata: dict[str, Any], source_size: list[int], encoded_size: list[int]
    ) -> dict[str, Any]:
        result = deepcopy(metadata)
        width, height = result["intrinsics_size"]
        if not math.isclose(source_size[0] / source_size[1], width / height, rel_tol=0.002):
            return {"available": False, "reason": "RGB aspect ratio disagrees with authored calibration"}
        scale_x, scale_y = source_size[0] / width, source_size[1] / height
        k = result["intrinsics"]
        result["source_intrinsics"] = {
            **{
                key: k[key] * (scale_x if key in ("fx", "cx") else scale_y)
                for key in ("fx", "fy", "cx", "cy")
            },
            "width": source_size[0],
            "height": source_size[1],
            "distortion": [k[key] for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")],
        }
        result["available"] = True
        result["rgb_source_size"] = source_size
        result["rgb_encoded_size"] = encoded_size
        result["encoded_to_source_pixel"] = {
            "scale_x": source_size[0] / encoded_size[0],
            "scale_y": source_size[1] / encoded_size[1],
            "formula": "source_u=(encoded_u+0.5)*scale_x-0.5; source_v=(encoded_v+0.5)*scale_y-0.5",
        }
        return result
