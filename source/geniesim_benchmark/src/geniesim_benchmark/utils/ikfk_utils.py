# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import threading

import numpy as np
from scipy.spatial.transform import Rotation as R

from geniesim_benchmark.utils.hard_ik import HardIKFKSolver

# Preserve the public constructor name used by conversion and teleop tools.
IKFKSolver = HardIKFKSolver

# scipy >=1.14 added `scalar_first=` to from_quat/as_quat, but older scipy
# bundled with some Isaac Sim images doesn't have it. Stay version-agnostic
# by reordering the wxyz <-> xyzw layouts ourselves.
_WXYZ_TO_XYZW = [1, 2, 3, 0]
_XYZW_TO_WXYZ = [3, 0, 1, 2]


def _from_quat_wxyz(q):
    return R.from_quat(np.asarray(q)[_WXYZ_TO_XYZW])


def _as_quat_wxyz(rot):
    return rot.as_quat()[_XYZW_TO_WXYZ]


# Cache by embodiment and actual model; compatibility backend names all share it.
_IKFK_SOLVER_CACHE: dict = {}
_IKFK_SOLVER_CACHE_LOCK = threading.Lock()


def ik_frame(robot_cfg: str) -> str:
    """The link the IK/FK solver works in for robot_cfg."""
    from geniesim_benchmark.utils.name_utils import robot_type_mapping

    return "base_link" if robot_type_mapping(robot_cfg) == "arx" else "arm_base_link"


def arm_base_transform_from_poses(base_pos, base_rot, arm_pos, arm_rot):
    """4x4 transform mapping arm_base_link-frame points into base_link.

    `base_rot`/`arm_rot` are wxyz quaternions, as returned by
    `APICore._get_link_world_pose`.
    """
    base_pos = np.asarray(base_pos, dtype=np.float64)
    arm_pos = np.asarray(arm_pos, dtype=np.float64)
    base_rot_wxyz = np.asarray(base_rot, dtype=np.float64)
    arm_rot_wxyz = np.asarray(arm_rot, dtype=np.float64)

    R_base = R.from_quat([base_rot_wxyz[1], base_rot_wxyz[2], base_rot_wxyz[3], base_rot_wxyz[0]]).as_matrix()
    R_arm = R.from_quat([arm_rot_wxyz[1], arm_rot_wxyz[2], arm_rot_wxyz[3], arm_rot_wxyz[0]]).as_matrix()

    T_base = np.eye(4)
    T_base[:3, :3] = R_base
    T_base[:3, 3] = base_pos

    T_arm = np.eye(4)
    T_arm[:3, :3] = R_arm
    T_arm[:3, 3] = arm_pos

    return np.linalg.inv(T_base) @ T_arm


def xyzquat_to_xyzrpy(xyzquat):
    xyz = xyzquat[:3]
    rpy = _from_quat_wxyz(xyzquat[3:]).as_euler("xyz", degrees=False)
    xyzrpy = np.concatenate([xyz, rpy])
    return xyzrpy


def xyzrpy_to_xyzquat(xyzrpy):
    xyz = xyzrpy[:3]
    quat = R.from_euler("xyz", xyzrpy[3:]).as_quat()
    xyzquat = np.concatenate([xyz, quat])
    return xyzquat


def xyzrpy2mat(xyzrpy):
    rot = R.from_euler("xyz", xyzrpy[3:6]).as_matrix()
    mat = np.eye(4)
    mat[0:3, 0:3] = rot
    mat[0:3, 3] = xyzrpy[0:3]
    return mat


def mat2xyzrpy(mat):
    rpy = R.from_matrix(mat[0:3, 0:3]).as_euler("xyz", degrees=False)
    xyz = mat[0:3, 3]
    xyzrpy = np.concatenate([xyz, rpy])
    return xyzrpy


def get_shared_ikfk_solver(
    arm_init_joint_position,
    head_init_position,
    waist_init_position,
    robot_cfg: str = "G1_omnipicker",
    ik_backend: str = "auto",
    robot_usd_path=None,
):
    """One ik_solver IK/FK implementation for every embodiment.

    Backend names are accepted for compatibility and never select another
    solver or model. Loaded runtime geometry is used whenever supplied.
    """
    from geniesim_benchmark.utils.name_utils import robot_type_mapping

    robot_cfg = robot_type_mapping("G2_omnipicker" if robot_cfg == "G2" else robot_cfg)
    if ik_backend not in ("auto", "ik_solver", "hard", "ik_sdk", "curobo"):
        raise ValueError(f"Unsupported IK backend: {ik_backend}")
    cache_key = (robot_cfg, str(robot_usd_path) if robot_usd_path else None)
    with _IKFK_SOLVER_CACHE_LOCK:
        solver = _IKFK_SOLVER_CACHE.get(cache_key)
        if solver is None:
            if robot_usd_path is not None:
                from geniesim_benchmark.utils.usd_ik import UsdIKFKSolver

                solver = UsdIKFKSolver(robot_cfg, robot_usd_path)
            else:
                solver = HardIKFKSolver(robot_cfg=robot_cfg)
            _IKFK_SOLVER_CACHE[cache_key] = solver
        return solver


def clear_shared_ikfk_solver_cache() -> None:
    """Drop the process-wide solver cache."""
    with _IKFK_SOLVER_CACHE_LOCK:
        _IKFK_SOLVER_CACHE.clear()
