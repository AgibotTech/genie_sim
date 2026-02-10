# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .base_env import BaseEnv
from geniesim.plugins.logger import Logger
from geniesim.benchmark.tasks.demo_task import DemoTask
from geniesim.benchmark.tasks.dummy_task import DummyTask
from geniesim.utils.name_utils import *

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

        if self.data_courier.enable_ros:
            self.data_courier.sim_ros_node.publish_image()

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
        init_gripper = [1 - v for v in self.init_gripper]
        self.robot_joint_indices = self.api_core.get_robot_joint_indices()
        if self.robot_cfg == "G1_omnipicker":
            self.api_core.set_joint_positions(
                self.init_arm,
                joint_indices=[self.robot_joint_indices[v] for v in G1_DUAL_ARM_JOINT_NAMES],
                is_trajectory=False,
            )
            self.api_core.set_joint_positions(
                self.init_waist,
                joint_indices=[self.robot_joint_indices[v] for v in G1_WAIST_JOINT_NAMES],
                is_trajectory=False,
            )
            self.api_core.set_joint_positions(
                self.init_head,
                joint_indices=[self.robot_joint_indices[v] for v in G1_HEAD_JOINT_NAMES],
                is_trajectory=False,
            )
            self.api_core.set_joint_positions(
                init_gripper,
                joint_indices=[self.robot_joint_indices[v] for v in OMNIPICKER_AJ_NAMES],
                is_trajectory=False,
            )
        elif self.robot_cfg == "G2_omnipicker":
            self.api_core.set_joint_positions(
                self.init_arm,
                joint_indices=[self.robot_joint_indices[v] for v in G2_DUAL_ARM_JOINT_NAMES],
                is_trajectory=False,
            )
            self.api_core.set_joint_positions(
                self.init_waist,
                joint_indices=[self.robot_joint_indices[v] for v in G2_WAIST_JOINT_NAMES],
                is_trajectory=False,
            )
            self.api_core.set_joint_positions(
                self.init_head,
                joint_indices=[self.robot_joint_indices[v] for v in G2_HEAD_JOINT_NAMES],
                is_trajectory=False,
            )
            self.api_core.set_joint_positions(
                init_gripper,
                joint_indices=[self.robot_joint_indices[v] for v in OMNIPICKER_AJ_NAMES],
                is_trajectory=False,
            )
