# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import threading
from typing import Optional

import ik_solver
import numpy as np
from scipy.spatial.transform import Rotation as R

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()

# scipy >=1.14 added `scalar_first=` to from_quat/as_quat, but older scipy
# bundled with some Isaac Sim images doesn't have it. Stay version-agnostic
# by reordering the wxyz <-> xyzw layouts ourselves.
_WXYZ_TO_XYZW = [1, 2, 3, 0]
_XYZW_TO_WXYZ = [3, 0, 1, 2]


def _from_quat_wxyz(q):
    return R.from_quat(np.asarray(q)[_WXYZ_TO_XYZW])


def _as_quat_wxyz(rot):
    return rot.as_quat()[_XYZW_TO_WXYZ]


current_dir = os.path.dirname(os.path.abspath(__file__))

# Process-wide IKFKSolver cache, keyed by robot_cfg. Public methods are
# stateless or re-sync per call, so sharing across vec env workers is safe.
_IKFK_SOLVER_CACHE: dict = {}
_IKFK_SOLVER_CACHE_LOCK = threading.Lock()

# Per-embodiment IK/FK model + solver config, and the per-arm DOF count.
#
# `urdf` points at the runtime robot URDF for embodiments that reuse it
# directly (arx, agilex), or at the stripped IK-SDK copy for G1/G2. `dof` is
# the joint count of ONE arm: the solver slices arm_joint_states as
# [:dof] / [dof:2*dof], so a 6-DOF arm and a 7-DOF arm share one code path.
#
# `frame` records which link the solver's FK/IK works in, purely so callers
# can pin arm_base_transform to identity for embodiments that solve directly
# in base_link (arx has no arm_base_link link; agilex's sits 0.15m up the
# chassis and is ignored by the runtime frame convention for these two).
_IK_SDK_DIR = os.path.join(current_dir, "IK-SDK")
_ROBOT_CFG_DIR = os.path.normpath(os.path.join(current_dir, "..", "app", "robot_cfg"))

_IK_SPECS = {
    "arx": {
        "urdf": os.path.join(_ROBOT_CFG_DIR, "arx", "arx.urdf"),
        "config": os.path.join(_IK_SDK_DIR, "arx_solver.yaml"),
        "dof": 6,
        "frame": "base_link",
    },
    "agilex": {
        "urdf": os.path.join(_ROBOT_CFG_DIR, "agilex", "agilex.urdf"),
        "config": os.path.join(_IK_SDK_DIR, "agilex_solver.yaml"),
        "dof": 6,
        "frame": "arm_base_link",
    },
    "G2": {
        "urdf": os.path.join(_IK_SDK_DIR, "G2_NO_GRIPPER.urdf"),
        "config": os.path.join(_IK_SDK_DIR, "g2_solver.yaml"),
        "dof": 7,
        "frame": "arm_base_link",
    },
    "G1": {
        "urdf": os.path.join(_IK_SDK_DIR, "G1_NO_GRIPPER.urdf"),
        "config": os.path.join(_IK_SDK_DIR, "g1_solver.yaml"),
        "dof": 7,
        "frame": "arm_base_link",
    },
}


def resolve_ik_spec(robot_cfg: str) -> dict:
    """IK model spec for robot_cfg.

    Matching is by substring to stay consistent with the rest of the
    codebase's robot_cfg handling. Anything that isn't arx/agilex/G2 falls
    back to the G1 spec, matching this module's pre-existing behaviour.
    """
    if "arx" in robot_cfg:
        return _IK_SPECS["arx"]
    if "agilex" in robot_cfg:
        return _IK_SPECS["agilex"]
    if "G2" in robot_cfg:
        return _IK_SPECS["G2"]
    return _IK_SPECS["G1"]


def ik_frame(robot_cfg: str) -> str:
    """The link the IK/FK solver works in for robot_cfg."""
    return resolve_ik_spec(robot_cfg)["frame"]


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


class _StubIKFKSolver:
    """No-op IK/FK solver for robots without IK config (e.g. lift2, split_aloha).

    `compute_eef` returns zero-pose so abs_joint observation pipelines
    that include "eef" stay shape-stable. `eef_actions_to_joint` raises —
    EEF-mode control is genuinely unsupported.
    """

    def __init__(self, robot_cfg: str):
        self.robot_cfg = robot_cfg
        logger.warning(
            f"IK/FK solver not configured for robot_cfg='{robot_cfg}'; "
            "using stub. EEF observations will be zero; abs_pose control will fail."
        )

    def compute_eef(self, arm_joint_states):
        zero = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        return {"left": list(zero), "right": list(zero)}

    def eef_actions_to_joint(self, eef_actions, arm_joint_states, head_init_position, max_ik_iters=3):
        raise NotImplementedError(
            f"EEF/IK control is not configured for robot_cfg='{self.robot_cfg}'; use abs_joint mode."
        )


class IKFKSolver:
    """IK/FK for one robot config.

    Thread-safety: ``get_shared_ikfk_solver`` hands the SAME instance to every
    ``CoRobotPolicy`` in the process, and both public methods below mutate the
    underlying ``ik_solver.Solver`` objects (``sync_target_with_joints`` /
    ``update_target_quat`` / ``solve``). Parallel evaluation now calls
    ``policy.act()`` for its N env slots concurrently (see
    ``pi05_bridge/parallel_runner.py``), so two slots can land in
    ``eef_actions_to_joint`` at the same time — interleaved target writes
    would hand one slot the other slot's IK result, silently and with
    perfectly plausible-looking joints. ``_lock`` serializes the whole
    mutating region; the cost is negligible (the EEF_ABS path is only taken
    when the served policy returns ``kind == "EEF_ABS"`` at all).
    """

    def __init__(self, arm_init_joint_position, head_init_position, waist_init_position, robot_cfg="G1_omnipicker"):
        self._lock = threading.Lock()
        spec = resolve_ik_spec(robot_cfg)
        urdf_path = spec["urdf"]
        config_path = spec["config"]
        # Per-arm joint count: every slice below derives from this instead of
        # assuming 7, so 6-DOF arms (arx, agilex) use the same code path.
        self.arm_dof = spec["dof"]
        self.frame = spec["frame"]
        self.robot_cfg = robot_cfg

        self.left_solver = ik_solver.Solver(
            part=ik_solver.RobotPart.LEFT_ARM, urdf_path=urdf_path, config_path=config_path
        )
        self.right_solver = ik_solver.Solver(
            part=ik_solver.RobotPart.RIGHT_ARM, urdf_path=urdf_path, config_path=config_path
        )
        self.left_solver.set_debug_mode(False)
        self.right_solver.set_debug_mode(False)
        self._sync_targets(arm_init_joint_position)

    def _sync_targets(self, arm_joint_states):
        """Seed both solvers' IK targets from the current joint state.

        Tolerates a short/empty state (some tasks have no init pose
        configured) by padding with zeros, matching the old behaviour where a
        short list simply produced short slices.
        """
        dof = self.arm_dof
        joints = np.zeros(2 * dof, dtype=np.float64)
        given = np.asarray(arm_joint_states, dtype=np.float64).ravel()[: 2 * dof]
        joints[: given.size] = given
        self.left_solver.sync_target_with_joints(joints[:dof])
        self.right_solver.sync_target_with_joints(joints[dof : 2 * dof])

    def quaternion_to_euler(self, w, x, y, z):
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_Cosp = 1 - 2 * (y**2 + z**2)
        yaw = np.arctan2(siny_cosp, cosy_Cosp)

        return np.stack([roll, pitch, yaw], axis=-1)

    def compute_eef(self, arm_joint_states):
        """Compute left/right EEF poses via FK from current arm joint states.

        Returns dict {"left": [x,y,z,qw,qx,qy,qz], "right": [x,y,z,qw,qx,qy,qz]}
        in this solver's frame (`self.frame`): arm_base_link for G1/G2,
        base_link for arx and agilex.
        """
        dof = self.arm_dof
        left_joints = np.asarray(arm_joint_states[:dof], dtype=np.float32)
        right_joints = np.asarray(arm_joint_states[dof : 2 * dof], dtype=np.float32)

        with self._lock:
            left_mat = np.asarray(self.left_solver.compute_fk(left_joints), dtype=np.float64).reshape(4, 4)
            right_mat = np.asarray(self.right_solver.compute_fk(right_joints), dtype=np.float64).reshape(4, 4)

        left_pos = left_mat[:3, 3]
        left_quat = _as_quat_wxyz(R.from_matrix(left_mat[:3, :3]))
        right_pos = right_mat[:3, 3]
        right_quat = _as_quat_wxyz(R.from_matrix(right_mat[:3, :3]))

        return {
            "left": np.concatenate([left_pos, left_quat]).tolist(),
            "right": np.concatenate([right_pos, right_quat]).tolist(),
        }

    def eef_actions_to_joint(self, eef_actions, arm_joint_states, head_init_position, max_ik_iters=3):
        """Solve IK for a chunk of EEF targets.

        Each entry of `eef_actions` is [left_xyzrpy(6), right_xyzrpy(6),
        left_gripper(1), right_gripper(1)] — a fixed 14-wide layout
        independent of arm DOF, since a pose is 6 numbers either way.
        Returned rows are [left_joints(dof), right_joints(dof), left_gripper,
        right_gripper].
        """
        joint_actions = []
        # Whole loop under the lock, not per-iteration: the solver carries the
        # synced arm state ACROSS iterations, so releasing between them would
        # let another thread re-sync mid-sequence (see the class docstring).
        with self._lock:
            self._sync_targets(arm_joint_states)

            for _, action in enumerate(eef_actions):
                eefrot_left_cur = np.array(action[:6], dtype=np.float32)
                eefrot_right_cur = np.array(action[6:12], dtype=np.float32)

                target_pos_left = xyzrpy_to_xyzquat(eefrot_left_cur)
                target_pos_right = xyzrpy_to_xyzquat(eefrot_right_cur)

                self.left_solver.update_target_quat(target_pos=target_pos_left[:3], target_quat=target_pos_left[3:])
                self.right_solver.update_target_quat(target_pos=target_pos_right[:3], target_quat=target_pos_right[3:])

                for _ in range(max_ik_iters):
                    left_joints = self.left_solver.solve()
                    right_joints = self.right_solver.solve()

                l_gripper = action[12:13] if type(action) == list else action[12:13].tolist()
                r_gripper = action[13:14] if type(action) == list else action[13:14].tolist()
                joint_actions.append(left_joints.tolist() + right_joints.tolist() + l_gripper + r_gripper)

        return joint_actions


def get_shared_ikfk_solver(
    arm_init_joint_position,
    head_init_position,
    waist_init_position,
    robot_cfg: str = "G1_omnipicker",
    ik_backend: str = "ik_sdk",
) -> Optional[IKFKSolver]:
    """Process-wide shared IKFKSolver for the given robot_cfg.

    arx and agilex now have a real IK-SDK config (see _IK_SPECS) and go
    through the same construction path as G1/G2. Falls back to None if
    construction fails for any robot_cfg, rather than killing the caller —
    JOINT_ABS replay should still work even without EEF_ABS.

    ``ik_backend`` selects the solver implementation:
    - ``"ik_sdk"`` (default): CPU-based NLopt relaxed IK via ``ik_solver``.
    - ``"curobo"``: GPU-accelerated IK via cuRobo v0.7.6.  Falls back to
      ``ik_sdk`` if cuRobo is not installed or init fails.
    """
    cache_key = f"{robot_cfg}_{ik_backend}"
    with _IKFK_SOLVER_CACHE_LOCK:
        solver = _IKFK_SOLVER_CACHE.get(cache_key)
        if solver is None:
            if "lift2" in robot_cfg or "split_aloha" in robot_cfg:
                solver = _StubIKFKSolver(robot_cfg=robot_cfg)
            elif ik_backend == "curobo":
                try:
                    from geniesim_benchmark.utils.curobo_ikfk_utils import CuroboIKFKSolver

                    solver = CuroboIKFKSolver(
                        arm_init_joint_position,
                        head_init_position,
                        waist_init_position,
                        robot_cfg=robot_cfg,
                    )
                except Exception as e:
                    logger.warning(f"cuRobo IK init failed for robot_cfg='{robot_cfg}': {e}; " "falling back to ik_sdk")
                    try:
                        solver = IKFKSolver(
                            arm_init_joint_position,
                            head_init_position,
                            waist_init_position,
                            robot_cfg=robot_cfg,
                        )
                    except Exception as e2:
                        logger.error(f"Fallback IKFKSolver also failed: {e2}")
                        return None
            else:
                try:
                    solver = IKFKSolver(
                        arm_init_joint_position,
                        head_init_position,
                        waist_init_position,
                        robot_cfg=robot_cfg,
                    )
                except Exception as e:
                    logger.error(f"Failed to build IKFKSolver for robot_cfg='{robot_cfg}': {e}")
                    return None
            _IKFK_SOLVER_CACHE[cache_key] = solver
        return solver


def clear_shared_ikfk_solver_cache() -> None:
    """Drop the process-wide solver cache."""
    with _IKFK_SOLVER_CACHE_LOCK:
        _IKFK_SOLVER_CACHE.clear()
