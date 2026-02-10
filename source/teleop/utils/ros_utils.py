# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
import rclpy
from utils.ros_nodes import SimNode
import threading


class RosUtils(object):
    def __init__(self, robot_name, node_name="teleop_node"):
        self.robot_name = robot_name
        self.node_name = node_name

    def start_ros_node(self):
        rclpy.init()
        self.sim_ros_node = SimNode(robot_name=self.robot_name, node_name=self.node_name)
        self.stop_spin_thread_flag = False
        self.spin_thread = threading.Thread(target=self.spin_ros_node, daemon=True)
        self.spin_thread.start()
        self.lock = threading.Lock()

    def spin_ros_node(self):
        while not self.stop_spin_thread_flag:
            rclpy.spin_once(self.sim_ros_node, timeout_sec=0.01)

    def stop_ros_node(self):
        if self.spin_thread.is_alive():
            self.stop_spin_thread_flag = True
            self.spin_thread.join(timeout=1.0)
        self.sim_ros_node.destroy_node()
        rclpy.shutdown()

    def get_joint_state(self):
        js = self.sim_ros_node.get_joint_state()
        return js.position, js.name

    def set_joint_state(self, names, positions):
        self.sim_ros_node.set_joint_state(names, positions)

    def set_joint_position_and_velocity(self, names, positions, velocities):
        self.sim_ros_node.set_joint_position_and_velocity(names, positions, velocities)
