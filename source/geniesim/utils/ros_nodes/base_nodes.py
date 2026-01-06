# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from rclpy.node import Node
from rclpy.parameter import Parameter

from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage

from collections import deque
import threading
from geniesim.utils.name_utils import *
from cv_bridge import CvBridge

import numpy as np
import time

QOS_PROFILE_LATEST = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

QOS_PROFILE_VOLATILE = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)


def image_to_numpy(img_msg):
    # Zero-copy data reading
    img_np = np.frombuffer(img_msg.data, dtype=np.uint8)

    encoding = img_msg.encoding.lower()  # Convert to lowercase

    if encoding == "mono8":
        img_np = img_np.reshape((img_msg.height, img_msg.width))
    elif encoding == "rgb8":
        img_np = img_np.reshape((img_msg.height, img_msg.width, 3))
    elif encoding == "bgr8":
        img_np = img_np.reshape((img_msg.height, img_msg.width, 3))
    elif encoding == "rgba8":
        img_np = img_np.reshape((img_msg.height, img_msg.width, 4))
        img_np = img_np[:, :, :3]  # Remove alpha → RGB
    elif encoding == "bgra8":
        img_np = img_np.reshape((img_msg.height, img_msg.width, 4))
    else:
        raise ValueError(f"Unsupported encoding: {img_msg.encoding}")

    return img_np


class SimNode(Node):
    def __init__(self, robot_name="G1_120s", node_name="sim_ros_node"):
        super().__init__(
            node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.init_pose = None
        self.robot_name = robot_name
        self.config_robot()
        self.img2np = image_to_numpy
        self.bridge = CvBridge()
        self.pub_joint_command = self.create_publisher(
            JointState,
            "/joint_command",
            1,
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
        self.publisher_playback = self.create_publisher(Bool, "/sim/playback_flag", 1)
        self.publisher_onreset = self.create_publisher(Bool, "/sim/reset_flag", 1)
        self.lock_js = threading.Lock()
        self.lock_tf = threading.Lock()
        self.js_msg = JointState()
        self.tf_msg = TFMessage()
        self.joint_states = {}

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

    def get_base_pose(self):
        with self.lock_tf:
            for t in self.tf_msg.transforms:
                if "world" == t.header.frame_id and "Cube" == t.child_frame_id:
                    position = np.array(
                        [
                            t.transform.translation.x,
                            t.transform.translation.y,
                            t.transform.translation.z,
                        ]
                    )
                    rotation = np.array(
                        [
                            t.transform.rotation.w,
                            t.transform.rotation.x,
                            t.transform.rotation.y,
                            t.transform.rotation.z,
                        ]
                    )
                    return (position, rotation)

    def pub_playback(self, val):
        msg = Bool()
        msg.data = val
        self.publisher_playback.publish(msg)

    def pub_reset(self, val):
        msg = Bool()
        msg.data = val
        self.publisher_onreset.publish(msg)
