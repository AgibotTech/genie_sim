# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R


def _wxyz_to_xyzw(q):
    return [q[1], q[2], q[3], q[0]]


def _xyzw_to_wxyz(q):
    return [q[3], q[0], q[1], q[2]]


def _resolve_urdf_path(robot_cfg):
    """Same URDF the IK solver loads (IK-SDK ships next to ikfk_utils)."""
    import geniesim_benchmark

    urdf_name = "G2_NO_GRIPPER.urdf" if "G2" in robot_cfg else "G1_NO_GRIPPER.urdf"
    return str(Path(geniesim_benchmark.__file__).resolve().parent / "utils" / "IK-SDK" / urdf_name)


def base_to_arm_base_transform(urdf_path, body_joint_q, base="base_link", tip="arm_base_link"):
    """FK the URDF joint chain base_link -> arm_base_link at the given body
    joint values (name -> angle; missing names default to 0). This stands in
    for the reference teleop's live TF lookup of arm_base_link in base_link --
    valid here because the RLinf teleop holds the waist at its init pose."""
    root = ET.parse(urdf_path).getroot()
    joint_by_child = {j.find("child").attrib["link"]: j for j in root.findall("joint")}
    chain, link = [], tip
    while link != base:
        j = joint_by_child.get(link)
        if j is None:
            raise ValueError(f"no joint chain {base} -> {tip} in {urdf_path}")
        chain.append(j)
        link = j.find("parent").attrib["link"]
    T = np.eye(4)
    for j in reversed(chain):
        origin = j.find("origin")
        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
        To = np.eye(4)
        To[:3, :3] = R.from_euler("xyz", rpy).as_matrix()  # URDF rpy = extrinsic xyz
        To[:3, 3] = xyz
        T = T @ To
        if j.attrib.get("type") == "revolute":
            axis = np.array([float(v) for v in j.find("axis").attrib["xyz"].split()])
            q = float(body_joint_q.get(j.attrib["name"], 0.0))
            Tq = np.eye(4)
            Tq[:3, :3] = R.from_rotvec(axis * q).as_matrix()
            T = T @ Tq
    return T


class TeleopIKBridge:
    """Bridges Pico teleop Cartesian pose deltas to arm joint targets via
    per-hand engage-gated IK/FK, replicating the reference teleop pipeline
    (geniesim_teleop ros_nodes.parse_delta_pose + engage-edge snapshot)."""

    def __init__(
        self, arm_init_q14, head_init3, waist_init5, robot_cfg="G2_omnipicker", body_joint_q=None, joint_names=None
    ):
        from geniesim_benchmark.utils.ikfk_utils import IKFKSolver

        self._solver = IKFKSolver(list(arm_init_q14), list(head_init3), list(waist_init5), robot_cfg=robot_cfg)
        # Gripper masters are resolved from the SCENE's joint names, not from
        # ``robot_cfg``: the caller may not know which robot the server loaded,
        # and the schema always does. Omitted, this keeps the historical
        # omnipicker span so nothing changes for existing callers.
        self._grip_names = (
            resolve_grip_names(joint_names)
            if joint_names
            else ("idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1")
        )
        self._head_init = list(head_init3)
        self._ref = {}  # hand -> [xyz, quat_wxyz(4)] frozen at engage edge
        # base_link <-> arm_base_link (waist is held constant, so compute once)
        self._T_base_arm = base_to_arm_base_transform(_resolve_urdf_path(robot_cfg), dict(body_joint_q or {}))
        self._T_arm_base = np.linalg.inv(self._T_base_arm)

    def current_eef(self, arm_q14):
        """Forward-kinematics: arm_q14 (left7+right7 joints) -> per-hand EE pose [xyz, quat_wxyz] in arm_base frame."""
        return self._solver.compute_eef(list(arm_q14))

    def on_engage(self, hand, arm_q14):
        """Engage rising edge: freeze `hand`'s current EE pose as the reference
        the per-tick delta is applied against (the reference teleop snapshots
        parts[...]["pose"] via update_without_judge only on this edge -- the
        Pico "delta" is the controller's clutch-relative offset, so the target
        is engage_EE + offset, NOT an accumulating per-tick increment)."""
        self._ref[hand] = list(self.current_eef(arm_q14)[hand])

    def _target_xyzrpy(self, hand, arm_q14, delta_xyzxyzw, engaged):
        """Strict replica of ros_nodes.parse_delta_pose: take the engage-frozen
        reference EE, move it to base_link, add the base_link-frame delta
        (pos add; rot pre-compose), and transform the result back to
        arm_base_link for the IK solver. arm_q14 is only used to hold the
        live pose while disengaged."""
        from geniesim_benchmark.utils.ikfk_utils import xyzquat_to_xyzrpy

        if not engaged or hand not in self._ref:
            return xyzquat_to_xyzrpy(np.array(self.current_eef(arm_q14)[hand]))
        ref = self._ref[hand]
        T_arm_ee = np.eye(4)
        T_arm_ee[:3, :3] = R.from_quat(_wxyz_to_xyzw(ref[3:])).as_matrix()
        T_arm_ee[:3, 3] = ref[:3]
        T_base_ee = self._T_base_arm @ T_arm_ee
        ee_rot = R.from_matrix(T_base_ee[:3, :3])
        delta_rot = R.from_quat(delta_xyzxyzw[3:7])
        new_rot = ee_rot * (ee_rot.inv() * delta_rot * ee_rot)
        T_base_ee_new = np.eye(4)
        T_base_ee_new[:3, :3] = new_rot.as_matrix()
        T_base_ee_new[:3, 3] = T_base_ee[:3, 3] + np.asarray(delta_xyzxyzw[:3])
        T_arm_ee_new = self._T_arm_base @ T_base_ee_new
        new_wxyz = _xyzw_to_wxyz(R.from_matrix(T_arm_ee_new[:3, :3]).as_quat())
        return xyzquat_to_xyzrpy(np.concatenate([T_arm_ee_new[:3, 3], new_wxyz]))

    def deltas_to_joints(self, arm_q14, frame, max_ik_iters=8):
        """Map engaged Cartesian deltas to joint commands; hold the seed when both hands are released."""
        lo, ro = self._grip_names
        if not frame["l_on"] and not frame["r_on"]:
            return list(arm_q14) + [grip_cmd(lo, frame["l_grip"]), grip_cmd(ro, frame["r_grip"])]
        l = self._target_xyzrpy("left", arm_q14, frame["left"], frame["l_on"])
        r = self._target_xyzrpy("right", arm_q14, frame["right"], frame["r_on"])
        action = [list(l) + list(r) + [grip_cmd(lo, frame["l_grip"]), grip_cmd(ro, frame["r_grip"])]]
        return self._solver.eef_actions_to_joint(action, list(arm_q14), self._head_init, max_ik_iters=max_ik_iters)[0]


# Raw joint radians from open to closed; benchmark gripper actions use a different scale.
_GRIP_SPAN = {
    "idx41": (0.65, 0.0),
    "idx81": (0.65, 0.0),  # omnipicker
    "idx32": (0.0, -0.91),
    "idx72": (0.0, -0.91),  # 90d
}
_GRIP_SPAN_DEFAULT = (0.65, 0.0)


def grip_span(master_name):
    """[open, closed] raw-radian travel for this master joint."""
    return _GRIP_SPAN.get(str(master_name or "").split("_")[0], _GRIP_SPAN_DEFAULT)


def grip_cmd(master_name, grip):
    """Trigger value -> raw joint command.

    ``grip`` is 1.0 released / 0.0 squeezed (``pico_signal`` computes it as
    ``1 - indexTrig``), so released maps to open and squeezed to closed.
    """
    o, c = grip_span(master_name)
    return c + (o - c) * float(grip)


def resolve_grip_names(joint_names):
    """Return left/right gripper master names from the server schema, or None for absent joints."""
    g = [n for n in joint_names if n.endswith("outer_joint1")]
    lo = next((n for n in g if "_l_" in n), None)
    ro = next((n for n in g if "_r_" in n), None)
    return lo, ro


def build_full_action(
    joint_names, arm16, init_full, left_names=None, right_names=None, l_grip_name=None, r_grip_name=None
):
    """Place a 16-dim [l7, r7, l_grip, r_grip] arm action into the sim's full action
    vector by joint name, holding every other joint (waist/head/etc.) at its init value."""
    import numpy as np

    if l_grip_name is None or r_grip_name is None:
        _lo, _ro = resolve_grip_names(joint_names)
        l_grip_name = l_grip_name or _lo
        r_grip_name = r_grip_name or _ro
    if left_names is None or right_names is None:
        from geniesim_teleop.config.robot_interface import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES

        left_names = left_names or LEFT_ARM_JOINT_NAMES
        right_names = right_names or RIGHT_ARM_JOINT_NAMES
    idx = {n: i for i, n in enumerate(joint_names)}
    out = np.array(init_full, dtype=np.float32).copy()
    for k, n in enumerate(left_names):
        if n in idx:
            out[idx[n]] = arm16[k]
    for k, n in enumerate(right_names):
        if n in idx:
            out[idx[n]] = arm16[7 + k]
    if l_grip_name and l_grip_name in idx:
        out[idx[l_grip_name]] = arm16[14]
    if r_grip_name and r_grip_name in idx:
        out[idx[r_grip_name]] = arm16[15]
    return out
