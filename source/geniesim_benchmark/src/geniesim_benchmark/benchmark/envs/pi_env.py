# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from .dummy_env import DummyEnv

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()

from geniesim_benchmark.benchmark.tasks.llm_task import LLMTask
from geniesim_benchmark.utils.name_utils import *

# Isaac reports link orientations as wxyz; scipy's from_quat takes xyzw. Older
# scipy (bundled with some Isaac Sim images) lacks the `scalar_first=` kwarg,
# so reorder explicitly instead.
_WXYZ_TO_XYZW = [1, 2, 3, 0]


class PiEnv(DummyEnv):
    def __init__(self, api_core, task_file: str, init_task_config, need_setup=True):
        super().__init__(api_core, task_file, init_task_config, need_setup)
        # Joint groups the server commands but this robot_cfg can't drive;
        # warned about once each rather than every step.
        self._unusable_joint_groups = set()
        self.load_task_setup()

    def load_task_setup(self):
        self.task = LLMTask(self)

    def get_observation(self, fetch_images=True):
        # C: single physics-loop round-trip for image + depth + joint_state
        # + base/arm link poses, vs serial run_on_physics_loop waits before.
        # B: when fetch_images is False (chunk replay between inferences) the
        # policy won't consume images, so we skip them.
        bundle = self.data_courier.get_obs_bundle(
            fetch_images=fetch_images,
            fetch_depth=getattr(self, "need_depth", True),
            link_names=("base_link", "arm_base_link"),
        )
        images = bundle.get("images", {})
        depth = bundle.get("depth", {})
        full_joint_states = bundle["joint_state"]

        def joint_values(names):
            return [full_joint_states[name] for name in names]

        waist_joints = self.cfg["waist_joints"]
        gripper_values = joint_values(self.cfg["gripper_joints"])

        # Keyed states; policies flatten as left_arm + right_arm + left/right
        # gripper + waist + head (same order as the legacy flat list).
        states = {
            "left_arm": joint_values(self.cfg["left_arm_joints"]),
            "right_arm": joint_values(self.cfg["right_arm_joints"]),
            "left_gripper": gripper_values[:1],
            "right_gripper": gripper_values[1:2],
            "waist": joint_values(waist_joints),
            "head": joint_values(self.cfg["obs_extra_joints"]),
        }
        self.cur_arm = states["left_arm"] + states["right_arm"]

        obs = {"images": images, "states": states, "depth": depth}
        obs["eef"] = self.ikfk_solver.compute_eef(self.cur_arm)
        # FK EEF above is in arm_base_link; this transform lets the policy also
        # express it in (and accept EEF actions in) base_link.
        link_poses = bundle.get("link_poses") or {}
        if "base_link" in link_poses and "arm_base_link" in link_poses:
            try:
                obs["arm_base_transform"] = self._arm_base_transform_from_poses(link_poses)
            except Exception as e:
                logger.warning(f"arm_base_transform unavailable: {e}")
        return obs

    @staticmethod
    def _arm_base_transform_from_poses(link_poses):
        """Build the 4x4 arm_base_link -> base_link homogeneous transform.

        `link_poses` values are (position, quaternion_wxyz) world poses.
        """
        base_pos, base_rot = link_poses["base_link"]
        arm_pos, arm_rot = link_poses["arm_base_link"]

        base_pos = np.asarray(base_pos, dtype=np.float64)
        arm_pos = np.asarray(arm_pos, dtype=np.float64)
        base_rot_wxyz = np.asarray(base_rot, dtype=np.float64)
        arm_rot_wxyz = np.asarray(arm_rot, dtype=np.float64)

        R_base = R.from_quat(base_rot_wxyz[_WXYZ_TO_XYZW]).as_matrix()
        R_arm = R.from_quat(arm_rot_wxyz[_WXYZ_TO_XYZW]).as_matrix()

        T_base = np.eye(4)
        T_base[:3, :3] = R_base
        T_base[:3, 3] = base_pos

        T_arm = np.eye(4)
        T_arm[:3, :3] = R_arm
        T_arm[:3, 3] = arm_pos

        return np.linalg.inv(T_base) @ T_arm

    def reset(self):
        self._followed_objects = set()
        self._picked_objects = set()
        self.last_update_time = time.time()
        self.has_done = False
        self.task.reset(self)
        self.robot_joint_indices = self.api_core.get_robot_joint_indices()

        init_gripper = list(self.cfg.get("init_gripper_open", [0.0, 0.0]))

        # Apply all initial joint targets in one physics-loop round-trip.
        # is_trajectory=False is a direct snap-to-target, so a single apply
        # is enough — replaying it (as the previous nested loop did) is a
        # no-op for the controller and just burns physics ticks.
        self.api_core.set_joint_positions_batched(
            [
                (self.init_arm, [self.robot_joint_indices[v] for v in self.cfg["arm_joints"]], False),
                (self.init_waist, [self.robot_joint_indices[v] for v in self.cfg["waist_joints"]], False),
                (self.init_head, [self.robot_joint_indices[v] for v in self.cfg["head_joints"]], False),
                (init_gripper, [self.robot_joint_indices[v] for v in self.cfg["gripper_joints"]], False),
            ]
        )

        # Single polling loop with early exit on convergence; 1s budget.
        eps = 1e-2
        init_arm_arr = np.array(self.init_arm)
        init_waist_arr = np.array(self.init_waist)
        init_gripper_arr = np.array(init_gripper)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            time.sleep(0.02)
            js = self.data_courier.get_joint_state_dict()
            if not js:
                continue
            arm = np.array([js[n] for n in self.cfg["arm_joints"]])
            waist = np.array([js[n] for n in self.cfg["waist_joints"]])
            gripper = np.array([js[n] for n in self.cfg["gripper_joints"]])
            arm_ok = np.max(np.abs(arm - init_arm_arr)) < eps if arm.size else True
            waist_ok = np.max(np.abs(waist - init_waist_arr)) < eps if waist.size else True
            gripper_ok = np.max(np.abs(gripper - init_gripper_arr)) < eps if gripper.size else True
            if arm_ok and waist_ok and gripper_ok:
                break
        logger.info("Finish reset robot...")

        self.api_core.reset_env()

        try:
            obs = self.get_observation()
        except Exception as e:
            logger.warning("get_observation failed after reset (%s) — retrying once", e)
            time.sleep(1.0)
            self.api_core.reset_env()
            obs = self.get_observation()
        logger.info("Finish reset env...")
        return obs

    def step(self, action):
        if action is None:
            self.has_done = True
            return self.get_observation(), self.has_done, False, self.task.task_progress

        self.current_step += 1
        need_update = False
        if self.current_step != 1 and self.current_step % 30 == 0:
            self.task.step(self)
            self.action_update()
            need_update = True

        # fmt: off
        arm = action.get("arm")
        gripper = action.get("gripper")
        waist = action.get("waist")
        head = action.get("head")

        batch = []
        if arm is not None:
            batch.append(([float(v) for v in arm], [self.robot_joint_indices[v] for v in self.cfg["arm_joints"]], True))
        if gripper is not None:
            batch.append(([float(v) for v in gripper], [self.robot_joint_indices[v] for v in self.cfg["gripper_joints"]], True))
        if waist is not None:
            self._append_joint_target(batch, waist, self.cfg["waist_joints"], "waist")
        if head is not None:
            self._append_joint_target(batch, head, self.cfg["head_joints"], "head")
        if batch:
            self.api_core.set_joint_positions_batched(batch)
        # fmt: on

        # Only wait for arm convergence on the chunk's last step (the obs fed to
        # the next inference), not on history-capture render steps.
        if getattr(self, "settle_due", self.need_infer) and arm is not None:
            self._wait_arm_settled([float(v) for v in arm])

        next_obs = self.get_observation(fetch_images=self.need_infer or self.has_done)
        return next_obs, self.has_done, need_update, self.task.task_progress

    def _append_joint_target(self, batch, values, joint_names, label):
        """Queue a joint-position target, pairing values with joints 1:1.

        The action's width is whatever the inference server chose to send, so
        clamp to the shorter of the two rather than handing the controller a
        values/indices mismatch. A robot whose config declares no joints for
        this group simply can't follow it — say so once instead of dropping the
        command silently every step.
        """
        n = min(len(values), len(joint_names))
        if n == 0:
            if label not in self._unusable_joint_groups:
                self._unusable_joint_groups.add(label)
                logger.warning(
                    f"Server sent a {label} action but robot_cfg declares no {label} joints; "
                    f"leaving {label} uncontrolled"
                )
            return
        batch.append(([float(v) for v in values[:n]], [self.robot_joint_indices[v] for v in joint_names[:n]], True))

    def _wait_arm_settled(self, target_arm):
        """Poll until arm joints converge to target (or timeout). Mirrors the
        convergence loop in reset(); physics free-runs so the arm keeps moving
        between polls."""
        eps = self.cfg.get("infer_settle_eps", 1e-2)
        timeout = self.cfg.get("infer_settle_timeout", 1.0)
        names = self.cfg["arm_joints"]
        target = np.array(target_arm)
        deadline = time.time() + timeout
        while time.time() < deadline:
            js = self.data_courier.get_joint_state_dict()
            if js:
                cur = np.array([js[n] for n in names])
                if cur.size and np.max(np.abs(cur - target)) < eps:
                    return
            time.sleep(0.01)
        logger.warning(f"arm did not settle within {timeout:.2f}s before observation")
