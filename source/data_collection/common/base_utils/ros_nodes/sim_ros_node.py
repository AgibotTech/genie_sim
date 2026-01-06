# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys

import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

from common.base_utils.logger import logger

ROOT_DIR = os.getenv("SIM_REPO_ROOT")
UTIL_DIR = os.path.join(str(ROOT_DIR), "base_utils")
sys.path.append(UTIL_DIR)
logger.info(f"UTIL_DIR: {UTIL_DIR}")

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


class SimPubRosNode(Node):
    def __init__(
        self,
        type,
        topic,
        get_msg_callback,
        frame_id="base_link",
        robot_name="G1_120s",
        node_name="sim_ros_node",
        step_size=1,
    ):
        super().__init__(
            node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self._robot_name = robot_name
        self._publisher = self.create_publisher(type, topic, 1)
        self._topic = topic
        self._type = type
        self._step_size = step_size
        self._cur_step = 0
        self._get_msg_callback = get_msg_callback
        self._sec = 0
        self._nanosec = 0
        self._frame_id = frame_id
        self._header = Header(frame_id=frame_id)
        logger.info(f"Created node: {self.get_name()}, topic: {self._topic}, type: {self._type}")

    def handle_msg(self, msg):
        if not isinstance(msg, self.type):
            raise Exception(
                f"Invalid msg type {type(msg)}, node: {self.get_name()}, topic: {self._topic}, expected: {self._type}"
            )
        return msg

    def tick(self, current_time):
        self._sec = int(current_time)
        self._nanosec = int((current_time - self._sec) * 1e9)

        self._header.stamp.sec = self._sec
        self._header.stamp.nanosec = self._nanosec

        self._cur_step += 1
        if self._cur_step >= self._step_size:
            try:
                raw_msg = self._get_msg_callback()
                msg = self.handle_msg(raw_msg)
                if msg is not None:
                    self._publisher.publish(msg)
                    self._cur_step = 0
            except Exception as e:
                logger.error(
                    f"Error publishing message: {e}, node: :{self.get_name()}, topic: {self._topic}"
                )


class ImagePubRosNode(SimPubRosNode):
    def __init__(
        self,
        topic,
        get_msg_callback,
        frame_id,
        robot_name="G1_120s",
        node_name="image_pub_node",
        step_size=1,
    ):
        super().__init__(Image, topic, get_msg_callback, frame_id, robot_name, node_name, step_size)
        self._bridge = CvBridge()

    def handle_msg(self, msg):
        msg = msg.astype(np.uint8)
        msg = np.ascontiguousarray(msg)
        img = self._bridge.cv2_to_imgmsg(
            msg,
            encoding="rgb8",
        )
        img.header = self._header
        img.header.frame_id = self._frame_id
        img.header.stamp = self._header.stamp
        return img


class JointStatePubRosNode(SimPubRosNode):
    def __init__(
        self,
        topic,
        get_msg_callback,
        frame_id,
        robot_name="G1_120s",
        node_name="joint_state_pub_ros_node",
        step_size=1,
    ):
        super().__init__(
            JointState,
            topic,
            get_msg_callback,
            frame_id,
            robot_name,
            node_name,
            step_size,
        )

    def handle_msg(self, msg):
        js = JointState()
        js.header = self._header
        js.position = msg.get("position", np.array([])).tolist()
        js.velocity = msg.get("velocity", np.array([])).tolist()
        js.effort = msg.get("effort", np.array([])).tolist()
        js.name = msg.get("joint_names", [])
        return js
