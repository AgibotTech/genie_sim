# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Shared hard IK using the loaded native model's FK and limits."""

import threading

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from geniesim_benchmark.utils.hard_ik import HardIKFKSolver, _transform


class NativeIKFKSolver(HardIKFKSolver):
    def __init__(self, model, data, arm_q, arm_bounds):
        self._lock = threading.RLock()
        self._solvers = None
        self.arm_dof = 7
        self.frame = "arm_base_link"
        self.robot_cfg = "G2_crsB_wuji"
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = data.qpos
        self.arm_q = np.asarray(arm_q)
        self.lower, self.upper = np.asarray(arm_bounds).T.copy()
        self._ends = {side: model.body(f"arm_{letter}_end_link").id for side, letter in (("left", "l"), ("right", "r"))}
        self._base = model.body("base_link").id
        self._arm_base = model.body(self.frame).id
        mujoco.mj_kinematics(model, self.data)

        self.joint_names = [model.joint(int(np.flatnonzero(model.jnt_qposadr == q)[0])).name for q in self.arm_q]
        self._chains = {}
        for side, start in (("left", 0), ("right", 7)):
            body, blocks = self._ends[side], []
            while body != self._arm_base:
                if body == 0:
                    raise ValueError("Native end link is not a descendant of arm_base_link")
                origin = _transform(model.body_pos[body], R.from_quat(model.body_quat[body][[1, 2, 3, 0]]).as_matrix())
                joint_ids = np.flatnonzero(model.jnt_bodyid == body)
                if len(joint_ids) > 1:
                    raise ValueError("Native IK requires at most one joint per arm body")
                if len(joint_ids):
                    joint = int(joint_ids[0])
                    qindex = int(model.jnt_qposadr[joint])
                    matches = np.flatnonzero(self.arm_q[start : start + 7] == qindex)
                    if len(matches) != 1:
                        raise ValueError("Native arm chain contains an uncontrolled joint")
                    index = int(matches[0])
                    axis, pivot, ref = model.jnt_axis[joint], model.jnt_pos[joint], model.qpos0[qindex]
                    if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                        origin = origin @ _transform(pivot, R.from_rotvec(-axis * ref).as_matrix())
                        block = [
                            (index, "revolute", origin, axis),
                            (None, "fixed", _transform(-pivot, np.eye(3)), np.zeros(3)),
                        ]
                    elif model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE:
                        block = [(index, "prismatic", origin @ _transform(-axis * ref, np.eye(3)), axis)]
                    else:
                        raise ValueError("Unsupported native arm joint type")
                else:
                    block = [(None, "fixed", origin, np.zeros(3))]
                blocks.append(block)
                body = int(model.body_parentid[body])
            self._chains[side] = [entry for block in reversed(blocks) for entry in block]

    def base_transform(self, positions=None):
        rotation = self.data.xmat[self._base].reshape(3, 3)
        return _transform(
            rotation.T @ (self.data.xpos[self._arm_base] - self.data.xpos[self._base]),
            rotation.T @ self.data.xmat[self._arm_base].reshape(3, 3),
        )
