# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys
import threading

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from common.base_utils.logger import logger

ROOT_DIR = os.getenv("SIM_REPO_ROOT")
UTIL_DIR = os.path.join(str(ROOT_DIR), "base_utils")
sys.path.append(UTIL_DIR)
logger.info(f"UTIL_DIR: {UTIL_DIR}")

from common.base_utils.name_utils import (
    G1_JOINT_NAMES,
    G2_JOINT_NAMES,
    OMNIPICKER_AJ_NAMES,
)

QOS_PROFILE_LATEST = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

QOS_PROFILE_VOLATILE = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class SimNode(Node):
    def __init__(self, robot_name="G1_120s", node_name="sim_ros_node"):
        super().__init__(
            node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.init_pose = None
        self.robot_name = robot_name
        self.config_robot()
        self.bridge = CvBridge()
        self.pub_joint_command = self.create_publisher(
            JointState,
            "/joint_command",
            QOS_PROFILE_LATEST,
        )
        self.sub_js = self.create_subscription(
            JointState,
            "/joint_states",
            self.callback_joint_state,
            1,
        )
        self.sub_tf = self.create_subscription(
            TFMessage,
            "/tf",
            self.callback_tf,
            QOS_PROFILE_VOLATILE,
        )

        self.lock_js = threading.Lock()
        self.lock_tf = threading.Lock()
        self.js_msg = JointState()
        self.tf_msg = TFMessage()

        self.loop_rate = self.create_rate(30.0)

    def config_robot(self):
        if "G1" in self.robot_name:
            self.robot_id = "G1"
        elif "G2" in self.robot_name:
            self.robot_id = "G2"
        else:
            raise Exception(f"Invalid robot name {self.robot_name}")

        if self.robot_id == "G1":
            self.joint_names = G1_JOINT_NAMES
        elif self.robot_id == "G2":
            self.joint_names = G2_JOINT_NAMES
        self.config_eef()

    def config_eef(self):
        if "omnipicker" in self.robot_name:
            self.joint_names += OMNIPICKER_AJ_NAMES
        else:
            raise Exception(f"Invalid eef {self.robot_name}")

    def callback_joint_state(self, msg):
        with self.lock_js:
            self.js_msg = msg

    def callback_tf(self, msg):
        with self.lock_tf:
            self.tf_msg = msg

    def get_joint_state(self):
        with self.lock_js:
            return self.js_msg

    def set_joint_state(self, name, position):
        cmd_msg = JointState()
        cmd_msg.position = position
        cmd_msg.name = name
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_joint_command.publish(cmd_msg)
