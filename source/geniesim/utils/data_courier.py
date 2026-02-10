# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import time

from geniesim.utils.ros_nodes.pi_node import PIROSNode
from geniesim.app.controllers.api_core import APICore


class DataCourier:
    def __init__(self, api_core: APICore, enable_ros: bool, node_name, hz=30):

        self.api_core = api_core
        self.enable_ros = enable_ros
        if enable_ros:
            if node_name in ["pi", ""]:
                self.sim_ros_node = PIROSNode(robot_name="G1_omnipicker")
                # Spin in main loop for unified processing
                self.api_core.benchmark_ros_node = self.sim_ros_node
                # Set sub_task_name if available (even if empty string, we'll publish it)
                if hasattr(api_core, "sub_task_name"):
                    # Publish sub_task_name immediately, even if empty
                    # This ensures auto_record_and_extract.py receives it
                    self.sim_ros_node.set_sub_task_name(api_core.sub_task_name or "")
            else:
                # To be implemented
                pass

        self._next = time.time()
        self.hz = hz
        self.robot_cfg = None

    def set_robot_cfg(self, robot_cfg):
        self.robot_cfg = robot_cfg

    def loop_ok(self):
        if self.enable_ros:
            import rclpy

            return rclpy.ok()

        return True

    def sleep(self):
        if self.enable_ros:
            self.sim_ros_node.loop_rate.sleep()
            return

        now = time.time()
        to_sleep = self._next - now
        if to_sleep > 0:
            time.sleep(to_sleep)
        else:
            pass
        self._next += 1 / self.hz

    def pub_static_info_msg(self, msg: str):
        if not self.enable_ros:
            return

        self.sim_ros_node.pub_static_info_msg(msg)

    def pub_dynamic_info_msg(self, msg: str):
        if not self.enable_ros:
            return

        self.sim_ros_node.pub_dynamic_info_msg(msg)

    def sim_time(self):
        if not self.enable_ros:
            return time.time_ns() * 1e-9

        return self.sim_ros_node.get_clock().now().nanoseconds * 1e-9

    def get_joint_state_dict(self):
        return self.api_core.get_joint_state_dict()

    def get_observation_image(self):
        if self.robot_cfg == "G1_omnipicker":
            return self.api_core.get_observation_image(
                {"head": "head_camera", "left_hand": "left_camera", "right_hand": "right_camera"}
            )
        elif self.robot_cfg == "G2_omnipicker":
            return self.api_core.get_observation_image(
                {
                    "head": "head_front_camera",
                    "left_hand": "left_camera",
                    "right_hand": "right_camera",
                }
            )
        else:
            raise ValueError(f"Invalid robot cfg: {self.robot_cfg}")

    def set_joint_state(self, name, position):
        if self.enable_ros:
            self.sim_ros_node.set_joint_state(name, position)
            return

    def get_instruction(self):
        """Gets the content of the /sim/instruction message"""
        if not self.enable_ros:
            return None
        return self.sim_ros_node.get_instruction()

    def get_reset(self):
        """Gets the content of the /sim/reset message"""
        if not self.enable_ros:
            return None
        return self.sim_ros_node.get_reset()

    def get_infer_start(self):
        """Gets the content of the /sim/infer_start message"""
        if not self.enable_ros:
            return None
        return self.sim_ros_node.get_infer_start()

    def get_shuffle(self):
        """Gets the content of the /sim/shuffle message"""
        if not self.enable_ros:
            return None
        return self.sim_ros_node.get_shuffle()

    def get_teleop_recording(self):
        return self.sim_ros_node.get_teleop_recording()
