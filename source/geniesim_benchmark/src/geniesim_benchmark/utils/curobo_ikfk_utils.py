# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""cuRobo v0.7.6 based IK/FK solver — drop-in replacement for IKFKSolver.

Requires ``pip install nvidia-curobo`` (v0.7.6) and a CUDA GPU.  When cuRobo
is not installed the module raises ``ImportError`` at import time; callers
(``get_shared_ikfk_solver``) catch this and fall back to the CPU ik_solver.
"""

import os
import threading
from typing import Optional, List

import numpy as np
from scipy.spatial.transform import Rotation as R

from geniesim_benchmark.plugins.logger import Logger
from geniesim_benchmark.utils.ikfk_utils import (
    IKFKSolver,
    resolve_ik_spec,
    xyzrpy_to_xyzquat,
)

import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen

logger = Logger()

_XYZW_TO_WXYZ = [3, 0, 1, 2]

_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curobo_configs")
_IK_SDK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IK-SDK")
_ROBOT_CFG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "robot_cfg")
)

_CUROBO_SPECS = {
    "arx": {
        "yaml": os.path.join(_CFG_DIR, "arx_ik.yml"),
        "urdf": os.path.join(_ROBOT_CFG_DIR, "arx", "arx.urdf"),
        "dof": 6,
        "frame": "base_link",
        "left_ee": "arm_l_end_link",
        "right_ee": "arm_r_end_link",
        "left_joint_names": [
            "arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
            "arm_l_joint4", "arm_l_joint5", "arm_l_joint6",
        ],
        "right_joint_names": [
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3",
            "arm_r_joint4", "arm_r_joint5", "arm_r_joint6",
        ],
    },
    "agilex": {
        "yaml": os.path.join(_CFG_DIR, "agilex_ik.yml"),
        "urdf": os.path.join(_ROBOT_CFG_DIR, "agilex", "agilex.urdf"),
        "dof": 6,
        "frame": "arm_base_link",
        "left_ee": "arm_fl_end_link",
        "right_ee": "arm_fr_end_link",
        "left_joint_names": [
            "arm_fl_joint1", "arm_fl_joint2", "arm_fl_joint3",
            "arm_fl_joint4", "arm_fl_joint5", "arm_fl_joint6",
        ],
        "right_joint_names": [
            "arm_fr_joint1", "arm_fr_joint2", "arm_fr_joint3",
            "arm_fr_joint4", "arm_fr_joint5", "arm_fr_joint6",
        ],
    },
    "G2": {
        "yaml": os.path.join(_CFG_DIR, "G2_ik.yml"),
        "urdf": os.path.join(_IK_SDK_DIR, "G2_NO_GRIPPER.urdf"),
        "dof": 7,
        "frame": "arm_base_link",
        "left_ee": "arm_l_end_link",
        "right_ee": "arm_r_end_link",
        "left_joint_names": [
            "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
            "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
        ],
        "right_joint_names": [
            "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
            "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
        ],
    },
    "G1": {
        "yaml": os.path.join(_CFG_DIR, "G1_ik.yml"),
        "urdf": os.path.join(_IK_SDK_DIR, "G1_NO_GRIPPER.urdf"),
        "dof": 7,
        "frame": "arm_base_link",
        "left_ee": "arm_left_link7",
        "right_ee": "arm_right_link7",
        "left_joint_names": [
            "idx05_left_arm_joint1", "idx06_left_arm_joint2", "idx07_left_arm_joint3",
            "idx08_left_arm_joint4", "idx09_left_arm_joint5", "idx10_left_arm_joint6",
            "idx11_left_arm_joint7",
        ],
        "right_joint_names": [
            "idx12_right_arm_joint1", "idx13_right_arm_joint2", "idx14_right_arm_joint3",
            "idx15_right_arm_joint4", "idx16_right_arm_joint5", "idx17_right_arm_joint6",
            "idx18_right_arm_joint7",
        ],
    },
}


def _resolve_curobo_spec(robot_cfg: str) -> dict:
    if "arx" in robot_cfg:
        return _CUROBO_SPECS["arx"]
    if "agilex" in robot_cfg:
        return _CUROBO_SPECS["agilex"]
    if "G2" in robot_cfg:
        return _CUROBO_SPECS["G2"]
    return _CUROBO_SPECS["G1"]


class CuroboIKFKSolver:
    """cuRobo v0.7.6 IK/FK solver with the same interface as IKFKSolver.

    Falls back to the CPU-based ik_solver when IK fails.

    cuRobo orders joints by URDF tree traversal which may differ from our
    convention (left arm first, then right arm). This class handles the
    mapping transparently: inputs/outputs always use the [left..., right...]
    convention matching IKFKSolver.
    """

    def __init__(
        self,
        arm_init_joint_position,
        head_init_position,
        waist_init_position,
        robot_cfg: str = "G1_omnipicker",
    ):
        self._lock = threading.Lock()
        spec = _resolve_curobo_spec(robot_cfg)
        self.arm_dof = spec["dof"]
        self.frame = spec["frame"]
        self.robot_cfg = robot_cfg
        self._left_ee = spec["left_ee"]
        self._right_ee = spec["right_ee"]
        self._left_joint_names: List[str] = spec["left_joint_names"]
        self._right_joint_names: List[str] = spec["right_joint_names"]

        yaml_path = spec["yaml"]
        urdf_path = spec["urdf"]

        cfg = load_yaml(yaml_path)
        if "robot_cfg" in cfg:
            self._robot_cfg_dict = cfg["robot_cfg"]
        else:
            self._robot_cfg_dict = cfg
        self._robot_cfg_dict["kinematics"]["urdf_path"] = urdf_path

        self._tensor_args = TensorDeviceType()

        fk_cfg = CudaRobotModelConfig.from_data_dict(
            self._robot_cfg_dict, self._tensor_args
        )
        self._cuda_model = CudaRobotModel(fk_cfg)

        self._curobo_joint_names: List[str] = list(
            self._cuda_model.kinematics_config.joint_names
        )
        self._n_curobo_joints = len(self._curobo_joint_names)

        self._left_indices_in_curobo = [
            self._curobo_joint_names.index(n) for n in self._left_joint_names
        ]
        self._right_indices_in_curobo = [
            self._curobo_joint_names.index(n) for n in self._right_joint_names
        ]

        ee_link = self._robot_cfg_dict["kinematics"]["ee_link"]
        link_names = self._robot_cfg_dict["kinematics"].get("link_names") or []
        self._ee_link = ee_link
        self._link_names = link_names

        self._left_ee_idx: Optional[int] = None
        self._right_ee_idx: Optional[int] = None
        fk_link_names = self._cuda_model.kinematics_config.link_names
        for i, name in enumerate(fk_link_names):
            if name == self._left_ee:
                self._left_ee_idx = i
            if name == self._right_ee:
                self._right_ee_idx = i

        self._motion_gen: Optional[MotionGen] = None

        self._fallback_solver = IKFKSolver(
            arm_init_joint_position, head_init_position,
            waist_init_position, robot_cfg,
        )

        logger.info(
            f"CuroboIKFKSolver initialized for robot_cfg='{robot_cfg}' "
            f"(dof={self.arm_dof}, frame={self.frame}, "
            f"curobo_joint_order={self._curobo_joint_names})"
        )

    def _get_motion_gen(self) -> MotionGen:
        """Lazy-init MotionGen (expensive GPU warmup)."""
        if self._motion_gen is not None:
            return self._motion_gen
        mg_config = MotionGen.load_from_robot_config(
            robot_cfg=self._robot_cfg_dict,
            world_model=None,
            tensor_args=self._tensor_args,
            collision_checker_type=None,
            use_cuda_graph=False,
            num_ik_seeds=32,
            num_batch_ik_seeds=32,
            num_trajopt_seeds=4,
            num_graph_seeds=1,
            interpolation_dt=0.01,
            self_collision_check=False,
        )
        self._motion_gen = MotionGen(mg_config)
        self._motion_gen.warmup(parallel_finetune=False, batch=1)
        logger.info("CuroboIKFKSolver: MotionGen warmed up")
        return self._motion_gen

    def _our_to_curobo(self, arm_joint_states) -> np.ndarray:
        """Convert [left_arm..., right_arm...] to cuRobo joint order."""
        dof = self.arm_dof
        our = np.zeros(2 * dof, dtype=np.float64)
        given = np.asarray(arm_joint_states, dtype=np.float64).ravel()[: 2 * dof]
        our[: given.size] = given

        left_vals = our[:dof]
        right_vals = our[dof: 2 * dof]

        curobo_joints = np.zeros(self._n_curobo_joints, dtype=np.float32)
        for i, idx in enumerate(self._left_indices_in_curobo):
            curobo_joints[idx] = left_vals[i]
        for i, idx in enumerate(self._right_indices_in_curobo):
            curobo_joints[idx] = right_vals[i]
        return curobo_joints

    def _curobo_to_our(self, curobo_joints) -> list:
        """Convert cuRobo joint order back to [left_arm..., right_arm...]."""
        arr = np.asarray(curobo_joints, dtype=np.float64).ravel()
        left = [float(arr[i]) for i in self._left_indices_in_curobo]
        right = [float(arr[i]) for i in self._right_indices_in_curobo]
        return left + right

    def compute_eef(self, arm_joint_states):
        curobo_joints = self._our_to_curobo(arm_joint_states)
        pos_tensor = self._tensor_args.to_device(curobo_joints.reshape(1, -1))
        js = JointState.from_position(pos_tensor, joint_names=self._curobo_joint_names)

        with self._lock:
            result = self._cuda_model.compute_kinematics(js)

        positions = result.links_position[0].detach().cpu().numpy()
        quaternions = result.links_quaternion[0].detach().cpu().numpy()

        def _extract(idx):
            if idx is None:
                return [0.0] * 7
            pos = positions[idx]
            quat_wxyz = quaternions[idx]
            return [
                float(pos[0]), float(pos[1]), float(pos[2]),
                float(quat_wxyz[0]), float(quat_wxyz[1]),
                float(quat_wxyz[2]), float(quat_wxyz[3]),
            ]

        return {
            "left": _extract(self._left_ee_idx),
            "right": _extract(self._right_ee_idx),
        }

    def eef_actions_to_joint(
        self, eef_actions, arm_joint_states, head_init_position, max_ik_iters=3
    ):
        joint_actions = []
        for action in eef_actions:
            result = self._solve_single_ik(action, arm_joint_states)
            if result is not None:
                joint_actions.append(result)
            else:
                logger.warning("CuroboIKFKSolver IK failed, falling back to ik_solver")
                fb = self._fallback_solver.eef_actions_to_joint(
                    [action], arm_joint_states, head_init_position, max_ik_iters
                )
                joint_actions.extend(fb)
        return joint_actions

    def _solve_single_ik(self, action, arm_joint_states):
        """Solve IK for a single EEF action using cuRobo.

        Input action layout: [left_xyzrpy(6), right_xyzrpy(6), l_grip(1), r_grip(1)]
        Returns: [left_joints..., right_joints..., l_grip, r_grip] or None.
        """
        dof = self.arm_dof
        try:
            mg = self._get_motion_gen()
        except Exception as e:
            logger.error(f"CuroboIKFKSolver: MotionGen init failed: {e}")
            return None

        eefrot_left = np.array(action[:6], dtype=np.float64)
        eefrot_right = np.array(action[6:12], dtype=np.float64)

        target_right_xyzquat = xyzrpy_to_xyzquat(eefrot_right)
        target_left_xyzquat = xyzrpy_to_xyzquat(eefrot_left)

        right_pos = target_right_xyzquat[:3]
        right_quat_xyzw = target_right_xyzquat[3:]
        right_quat_wxyz = right_quat_xyzw[_XYZW_TO_WXYZ]

        left_pos = target_left_xyzquat[:3]
        left_quat_xyzw = target_left_xyzquat[3:]
        left_quat_wxyz = left_quat_xyzw[_XYZW_TO_WXYZ]

        goal_pose = Pose(
            position=self._tensor_args.to_device(
                right_pos.reshape(1, 3).astype(np.float32)
            ),
            quaternion=self._tensor_args.to_device(
                right_quat_wxyz.reshape(1, 4).astype(np.float32)
            ),
        )

        link_poses = None
        if self._link_names and len(self._link_names) > 1:
            left_pose = Pose(
                position=self._tensor_args.to_device(
                    left_pos.reshape(1, 3).astype(np.float32)
                ),
                quaternion=self._tensor_args.to_device(
                    left_quat_wxyz.reshape(1, 4).astype(np.float32)
                ),
            )
            link_poses = {self._left_ee: left_pose}

        seed_q = self._our_to_curobo(arm_joint_states)
        seed_tensor = self._tensor_args.to_device(
            seed_q.reshape(1, 1, -1)
        )

        try:
            with self._lock:
                ik_result = mg.ik_solver.solve_single(
                    goal_pose, link_poses=link_poses,
                    q_sample=seed_tensor,
                )
        except Exception as e:
            logger.error(f"CuroboIKFKSolver IK solve error: {e}")
            return None

        if not bool(ik_result.success.any().item()):
            return None

        # js_solution includes locked joints (e.g. waist); extract by name
        ik_joint_names = ik_result.js_solution.joint_names
        ik_positions = ik_result.js_solution.position[0][0].detach().cpu().numpy()
        ik_map = dict(zip(ik_joint_names, ik_positions.tolist()))

        left_joints = [ik_map.get(n, 0.0) for n in self._left_joint_names]
        right_joints = [ik_map.get(n, 0.0) for n in self._right_joint_names]

        l_gripper = list(action[12:13]) if len(action) > 12 else [0.0]
        r_gripper = list(action[13:14]) if len(action) > 13 else [0.0]

        return left_joints + right_joints + l_gripper + r_gripper
