# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""IK/FK through the G2op ik_solver wheel, with strict pose acceptance.

Model readers describe joint frames and bounds; the wheel performs all arm
FK and IK. Importing this module does not initialize a simulator or the SDK.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET
import tempfile
import threading
import warnings

import yaml

import numpy as np
from scipy.spatial.transform import Rotation as R

from geniesim_benchmark.utils.name_utils import ROBOT_CONFIGS, robot_type_mapping


class IKConvergenceError(RuntimeError):
    """Local pose search failed; this does not establish global infeasibility."""


def _transform(xyz, rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = xyz
    return matrix


def _vector(values, length, label):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(f"{label} requires {length} finite values")
    return values


class HardIKFKSolver:
    """Thread-safe dual-arm SDK IK/FK, using measured-state seeds.

    FK poses use xyz+wxyz; EEF action rows use left/right xyz+rpy plus two
    legacy gripper scalars. The policy retains the full effector channels.
    """

    position_tolerance_m = 1e-5
    rotation_tolerance_rad = 1e-4

    def __init__(
        self, arm_init_joint_position=None, head_init_position=None, waist_init_position=None, robot_cfg="G1_omnipicker"
    ):
        self._lock = threading.RLock()
        self._solvers = None
        self.robot_cfg = robot_type_mapping("G2_omnipicker" if robot_cfg == "G2" else robot_cfg)
        cfg = ROBOT_CONFIGS[self.robot_cfg]
        cfg_dir = Path(__file__).resolve().parents[1] / "app/robot_cfg"
        robot = json.loads((cfg_dir / f"{self.robot_cfg}.json").read_text())["robot"]
        self.urdf_path = cfg_dir / robot["urdf_name"].lstrip("/")
        self.frame = "base_link" if self.robot_cfg == "arx" else "arm_base_link"
        self.arm_dof = len(cfg["left_arm_joints"])
        self.joint_names = cfg["left_arm_joints"] + cfg["right_arm_joints"]
        elements = ET.parse(self.urdf_path).getroot().findall("joint")
        parents = {j.find("child").get("link"): j for j in elements}
        joints = {j.get("name"): j for j in elements}
        limits = []
        for name in self.joint_names:
            limit = joints[name].find("limit")
            bounds = _vector([float(limit.get(k)) for k in ("lower", "upper")], 2, name)
            if bounds[0] >= bounds[1]:
                raise ValueError(f"Invalid URDF limits: {name}")
            limits.append(bounds)
        self.lower, self.upper = np.array(limits).T
        self._chains = {}
        for side, suffix in (("left", "fl"), ("right", "fr")):
            end = f"arm_{suffix}_end_link" if self.robot_cfg == "agilex" else f"arm_{side[0]}_end_link"
            chain, seen = [], set()
            link = end
            arm_names = cfg[f"{side}_arm_joints"]
            while link != self.frame:
                if link in seen or link not in parents:
                    raise ValueError(f"URDF {end} is not a descendant of {self.frame}")
                seen.add(link)
                joint = parents[link]
                name, kind = joint.get("name"), joint.get("type")
                if kind not in ("fixed", "revolute", "continuous", "prismatic") or joint.find("mimic") is not None:
                    raise ValueError(f"Unsupported IK chain joint: {name}")
                if kind != "fixed" and name not in arm_names:
                    raise ValueError(f"IK chain requires unmeasured joint: {name}")
                origin = joint.find("origin")
                xyz = self._attribute(origin, "xyz", "0 0 0")
                rpy = self._attribute(origin, "rpy", "0 0 0")
                axis = self._attribute(joint.find("axis"), "xyz", "1 0 0")
                if kind != "fixed" and np.linalg.norm(axis) == 0:
                    raise ValueError(f"Zero URDF joint axis: {name}")
                axis = axis / np.linalg.norm(axis) if kind != "fixed" else axis
                index = arm_names.index(name) if kind != "fixed" else None
                chain.append((index, kind, _transform(xyz, R.from_euler("xyz", rpy).as_matrix()), axis))
                link = joint.find("parent").get("link")
            self._chains[side] = list(reversed(chain))

    def base_transform(self, positions):
        """arm_base_link -> base_link from measured torso joints, never fixed offsets."""
        if self.frame == "base_link":
            return np.eye(4)
        root = ET.parse(self.urdf_path).getroot()
        parents = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
        link, chain, seen = self.frame, [], set()
        while link != "base_link":
            if link in seen or link not in parents:
                raise ValueError("IK frame is not a descendant of base_link")
            seen.add(link)
            joint = parents[link]
            kind, name = joint.get("type"), joint.get("name")
            origin = joint.find("origin")
            matrix = _transform(
                self._attribute(origin, "xyz", "0 0 0"),
                R.from_euler("xyz", self._attribute(origin, "rpy", "0 0 0")).as_matrix(),
            )
            if kind != "fixed":
                value = _vector([positions[name]], 1, name)[0]
                axis = self._attribute(joint.find("axis"), "xyz", "1 0 0")
                axis = axis / np.linalg.norm(axis)
                if kind in ("revolute", "continuous"):
                    matrix = matrix @ _transform([0, 0, 0], R.from_rotvec(axis * value).as_matrix())
                elif kind == "prismatic":
                    matrix = matrix @ _transform(axis * value, np.eye(3))
                else:
                    raise ValueError(f"Unsupported torso joint: {name}")
            chain.append(matrix)
            link = joint.find("parent").get("link")
        result = np.eye(4)
        for matrix in reversed(chain):
            result = result @ matrix
        return result

    @staticmethod
    def _attribute(element, key, default):
        raw = default if element is None else element.get(key, default)
        return _vector([float(v) for v in raw.split()], 3, f"URDF {key}")

    def _ensure_sdk(self):
        if self._solvers is not None:
            return
        try:
            import ik_solver
        except ImportError as exc:
            raise RuntimeError("IK/FK requires the bundled ik_solver wheel matching this Python version") from exc

        # Export only the independent arm chains. USD joint frames and native
        # models can differ from the checked-in URDF; exporting their actual
        # geometry keeps SDK FK consistent with the loaded robot.
        robot = ET.Element("robot", name="benchmark_ik")
        ET.SubElement(robot, "link", name=self.frame)
        tips = {}
        for side, start in (("left", 0), ("right", self.arm_dof)):
            parent = self.frame
            for number, (index, kind, origin, axis) in enumerate(self._chains[side]):
                child = f"ik_{side}_link_{number}"
                name = self.joint_names[start + index] if index is not None else f"ik_{side}_fixed_{number}"
                joint = ET.SubElement(robot, "joint", name=name, type=kind)
                ET.SubElement(robot, "link", name=child)
                ET.SubElement(joint, "parent", link=parent)
                ET.SubElement(joint, "child", link=child)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Gimbal lock detected")
                    rpy = R.from_matrix(origin[:3, :3]).as_euler("xyz")
                ET.SubElement(joint, "origin", xyz=" ".join(map(str, origin[:3, 3])), rpy=" ".join(map(str, rpy)))
                if index is not None:
                    ET.SubElement(joint, "axis", xyz=" ".join(map(str, axis)))
                    ET.SubElement(
                        joint,
                        "limit",
                        lower=str(self.lower[start + index]),
                        upper=str(self.upper[start + index]),
                        effort="0",
                        velocity="0",
                    )
                parent = child
            tips[side] = parent
        config = {
            "limit_way": "urdf",
            "optimizer": {
                "initial_step": 1e-4,
                "xtol_rel": 1e-10,
                "ftol_rel": 1e-12,
                "maxeval": 2000,
                "maxtime": 0.1,
                "gradient_eps": 1e-6,
            },
            "objectives": {
                "cartesian_space": [
                    {
                        "name": "end_effector",
                        "position": {"weight": 1.0, "use_groove": True},
                        "orientation": {"weight": 1.0, "use_groove": True},
                    }
                ],
                "joint_space": {"weights": {f"joint_{i + 1}": 1.0 for i in range(self.arm_dof)}},
            },
            "active_joints": {
                f"{side}_arm": self.joint_names[start : start + self.arm_dof]
                for side, start in (("left", 0), ("right", self.arm_dof))
            },
            "target_links": {f"{side}_arm": {"base": self.frame, "target": [tips[side]]} for side in ("left", "right")},
        }
        solvers = {}
        # The wheel parses URDF/YAML during construction and owns its models.
        # No generated model or configuration files need to persist afterwards.
        with tempfile.TemporaryDirectory(prefix="geniesim-ik-") as directory:
            urdf, settings = Path(directory) / "robot.urdf", Path(directory) / "solver.yaml"
            ET.ElementTree(robot).write(urdf, encoding="unicode")
            settings.write_text(yaml.safe_dump(config, sort_keys=False))
            for side, part in (("left", ik_solver.RobotPart.LEFT_ARM), ("right", ik_solver.RobotPart.RIGHT_ARM)):
                solvers[side] = ik_solver.Solver(part, str(urdf), str(settings))
                solvers[side].set_debug_mode(False)
        self._solvers = solvers

    def _fk(self, side, q):
        with self._lock:
            self._ensure_sdk()
            return np.asarray(self._solvers[side].compute_fk(np.asarray(q)), dtype=float).reshape(4, 4)

    def compute_eef(self, arm_joint_states):
        joints = _vector(arm_joint_states, 2 * self.arm_dof, "FK joints")
        poses = {}
        for side, q in (("left", joints[: self.arm_dof]), ("right", joints[self.arm_dof :])):
            matrix = self._fk(side, q)
            quat = R.from_matrix(matrix[:3, :3]).as_quat()
            poses[side] = np.r_[matrix[:3, 3], quat[[3, 0, 1, 2]]].tolist()
        return poses

    def _solve(self, side, target, seed, lower, upper):
        target_rotation = R.from_euler("xyz", target[3:])

        def error(q):
            current = self._fk(side, q)
            rotation = target_rotation * R.from_matrix(current[:3, :3]).inv()
            return np.r_[target[:3] - current[:3, 3], rotation.as_rotvec()]

        self._ensure_sdk()
        sdk = self._solvers[side]
        seed = np.clip(seed, lower, upper)
        generator = np.random.default_rng(0)
        best_error = error(seed)
        if (
            np.linalg.norm(best_error[:3]) <= self.position_tolerance_m
            and np.linalg.norm(best_error[3:]) <= self.rotation_tolerance_rad
        ):
            return seed
        for attempt in range(4):
            initial = seed if attempt == 0 else np.clip(seed + generator.uniform(-0.6, 0.6, seed.size), lower, upper)
            sdk.sync_target_with_joints(initial)
            sdk.update_target_mat(target[:3], target_rotation.as_matrix())
            candidate = _vector(sdk.solve(), self.arm_dof, "SDK IK result")
            residual = error(candidate)
            position = np.linalg.norm(residual[:3])
            rotation = np.linalg.norm(residual[3:])
            if (
                np.all(candidate >= lower)
                and np.all(candidate <= upper)
                and position <= self.position_tolerance_m
                and rotation <= self.rotation_tolerance_rad
            ):
                return candidate
            if np.dot(residual, residual) < np.dot(best_error, best_error):
                best_error = residual
        raise IKConvergenceError(
            f"Hard IK did not converge for {side} in {self.frame}: "
            f"position={np.linalg.norm(best_error[:3]):.6g} m, "
            f"rotation={np.linalg.norm(best_error[3:]):.6g} rad; no command returned"
        )

    def eef_actions_to_joint(self, eef_actions, arm_joint_states, head_init_position=None, max_ik_iters=None):
        joints = _vector(arm_joint_states, 2 * self.arm_dof, "IK seed").copy()
        with self._lock:
            result = []
            for row in eef_actions:
                row = _vector(row, 14, "EEF target")
                solved = []
                for side, start, target in (("left", 0, row[:6]), ("right", self.arm_dof, row[6:12])):
                    end = start + self.arm_dof
                    solved.extend(
                        self._solve(side, target, joints[start:end], self.lower[start:end], self.upper[start:end])
                    )
                joints = np.array(solved)
                result.append(solved + row[12:].tolist())
        return result
