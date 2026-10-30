# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
import threading

import rclpy
from rclpy.executors import ExternalShutdownException

from geniesim_teleop.utils.ros_nodes import SimNode


class RosUtils(object):
    def __init__(self, robot_name, node_name="teleop_node", arm_only=False):
        self.robot_name = robot_name
        self.node_name = node_name
        self.arm_only = arm_only
        self.sim_ros_node = None
        self.spin_thread = None
        self._spin_error = None

    def start_ros_node(self):
        rclpy.init()
        self.sim_ros_node = SimNode(
            robot_name=self.robot_name,
            node_name=self.node_name,
            arm_only=self.arm_only,
        )
        self.stop_spin_thread_flag = False
        self.spin_thread = threading.Thread(target=self.spin_ros_node, daemon=True)
        self.spin_thread.start()

    def spin_ros_node(self):
        try:
            while not self.stop_spin_thread_flag and rclpy.ok():
                rclpy.spin_once(self.sim_ros_node, timeout_sec=0.01)
        except ExternalShutdownException:
            # ROS signal handling can shut down while spin_once is waiting.
            pass
        except Exception as exc:
            self._spin_error = exc

    def check_health(self):
        if self._spin_error is not None:
            raise RuntimeError("Teleop ROS executor failed") from self._spin_error

    def stop_ros_node(self):
        if self.spin_thread is not None and self.spin_thread.is_alive():
            self.stop_spin_thread_flag = True
            self.spin_thread.join(timeout=5.0)
            if self.spin_thread.is_alive():
                raise RuntimeError("Teleop ROS executor did not stop")
        if self.sim_ros_node is not None:
            self.sim_ros_node.destroy_node()
            self.sim_ros_node = None
        if rclpy.ok():
            rclpy.shutdown()

    def set_joint_state(self, names, positions):
        self.sim_ros_node.set_joint_state(names, positions)

    def set_joint_velocity(self, names, velocities):
        self.sim_ros_node.set_joint_velocity(names, velocities)
