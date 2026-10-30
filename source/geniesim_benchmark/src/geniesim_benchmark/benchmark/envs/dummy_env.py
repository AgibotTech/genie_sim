# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .base_env import BaseEnv
from geniesim_benchmark.plugins.logger import Logger
from geniesim_benchmark.benchmark.tasks.demo_task import DemoTask
from geniesim_benchmark.benchmark.tasks.dummy_task import DummyTask
from geniesim_benchmark.utils.name_utils import *

logger = Logger()  # Create singleton instance


class DummyEnv(BaseEnv):
    def __init__(
        self,
        api_core,
        task_file: str,
        init_task_config,
        need_setup=True,
        ader_instance=0,
    ):
        super().__init__(
            api_core,
            task_file,
            init_task_config,
            need_setup,
            ader_instance,
        )
        self._load_teleop_initial_states()

    def _load_teleop_initial_states(self):
        import os

        config = os.environ.get("GENIESIM_TELEOP_CONFIG")
        self.teleop_initial_states = type(self) is DummyEnv and bool(config)
        if not self.teleop_initial_states:
            return
        from geniesim_teleop.config.task_config import load_teleop_task, robot_family

        task = load_teleop_task(config)
        if robot_family(task.robot_cfg) != self.robot_cfg or task.sub_task_name != self.sub_task_name:
            raise ValueError("Simulator and Pico must use the same teleop robot and sub-task")
        state = task.mc_initial_states()
        self.init_arm = state["init_arm"]
        self.init_waist = state["body_state"]
        self.init_head = state["head_state"]
        self.init_gripper = state["init_hand"]
        logger.info("Using shared IK-compatible task pose for teleop startup and reset")

    def load_task_setup(self):
        if "task_name" not in self.task_info:
            self.task = DummyTask(self)
        else:
            try:
                self.task = DemoTask(self)
            except ImportError:
                raise Exception("bddl is not available.")

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self.current_episode += 1
        self.current_step = 0

    def get_observation(self):
        return None

    def step(self, actions):
        observaion = None
        self.current_step += 1
        need_update = False
        if self.current_step != 1 and self.current_step % 30 == 0:
            observaion = self.get_observation()
            # self.task.step(self)
            # self.action_update()
            need_update = True

        return observaion, self.has_done, need_update, self.task.task_progress

    def start_recording(self, camera_prim_list, fps, extra_prim_paths, record_topic_list):
        self.api_core.start_recording(
            camera_prim_list=camera_prim_list,
            fps=fps,
            extra_prim_paths=extra_prim_paths,
            record_topic_list=record_topic_list,
        )

    def stop_recording(self):
        self.api_core.stop_recording()

    def reset(self):
        self._followed_objects = set()
        self._picked_objects = set()
        init_gripper = self.init_gripper_target()
        if self.api_core.robot_name == "G2_crsB_90d":
            init_gripper = [0.0] * len(self.cfg["gripper_joints"])
        self.robot_joint_indices = self.api_core.get_robot_joint_indices()
        self.api_core.prepare_teleop_grippers()

        free_base = bool(self.init_task_config.get("robot", {}).get("free_base", False))
        if free_base or self.teleop_initial_states:
            if free_base:
                self.api_core.set_robot_base_anchored(True)
                self.api_core.prepare_teleop_chassis()
            positions = list(self.init_arm) + list(self.init_waist) + list(self.init_head) + list(init_gripper)
            names = (
                self.cfg["arm_joints"] + self.cfg["waist_joints"] + self.cfg["head_joints"] + self.cfg["gripper_joints"]
            )
            indices = [self.robot_joint_indices[name] for name in names]
            if free_base:
                self.api_core.release_teleop_chassis(positions, indices)
            else:
                self.api_core.set_teleop_initial_pose(positions, indices)
            return

        # fmt: off
        self.api_core.set_joint_positions(self.init_arm, joint_indices=[self.robot_joint_indices[v] for v in self.cfg["arm_joints"]], is_trajectory=False)
        self.api_core.set_joint_positions(self.init_waist, joint_indices=[self.robot_joint_indices[v] for v in self.cfg["waist_joints"]], is_trajectory=False)
        self.api_core.set_joint_positions(self.init_head, joint_indices=[self.robot_joint_indices[v] for v in self.cfg["head_joints"]], is_trajectory=False)
        self.api_core.set_joint_positions(init_gripper, joint_indices=[self.robot_joint_indices[v] for v in self.cfg["gripper_joints"]], is_trajectory=False)
        # fmt: on
