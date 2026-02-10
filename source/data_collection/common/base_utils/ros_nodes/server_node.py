# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geometry_msgs.msg import TransformStamped
from rclpy.constants import S_TO_NS
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from common.base_utils.ros_nodes.base_nodes import Node, Parameter


class ServerNode(Node):
    def __init__(self, robot_name="G1_120s", node_name="server_ros_node"):
        super().__init__(
            node_name=node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        self.robot_name = robot_name

        self.sec, self.nanosec = 0, 0

        self.pub_clock = self.create_publisher(Clock, "/clock", 1)
        self.publisher_soft_state = self.create_publisher(TFMessage, "/soft_state", 1)
        self.publisher_soft_code = self.create_publisher(TFMessage, "/soft_code", 1)

    def publish_clock(self, time_in_s):
        self.sec = int(time_in_s)
        self.nanosec = int((time_in_s - self.sec) * S_TO_NS)
        msg = Clock()
        msg.clock.sec = self.sec
        msg.clock.nanosec = self.nanosec
        self.pub_clock.publish(msg)

    def publish_soft_state(self, object_name, position, rotation):
        transform = TransformStamped()
        transform.header.stamp.sec = self.sec
        transform.header.stamp.nanosec = self.nanosec
        transform.header.frame_id = object_name

        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]

        transform.transform.rotation.w = rotation[0]
        transform.transform.rotation.x = rotation[1]
        transform.transform.rotation.y = rotation[2]
        transform.transform.rotation.z = rotation[3]

        msg = TFMessage()
        msg.transforms = [transform]
        self.publisher_soft_state.publish(msg)

    def publish_soft_code(self, object_name, points):
        transforms = []
        for index, point in enumerate(points):
            transform = TransformStamped()
            transform.header.stamp.sec = self.sec
            transform.header.stamp.nanosec = self.nanosec
            transform.header.frame_id = object_name
            transform.child_frame_id = f"{object_name}_point_{index}"

            transform.transform.translation.x = point[0]
            transform.transform.translation.y = point[1]
            transform.transform.translation.z = point[2]

            transform.transform.rotation.w = 1.0
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transforms.append(transform)

        msg = TFMessage()
        msg.transforms = transforms
        self.publisher_soft_code.publish(msg)
