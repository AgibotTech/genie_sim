# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from rclpy.constants import S_TO_NS
from rosgraph_msgs.msg import Clock

from .base_nodes import *


class ServerNode(Node):
    def __init__(self, robot_name="G1_120s", node_name="server_ros_node"):
        super().__init__(
            node_name=node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        self.robot_name = robot_name

        self.sec, self.nanosec = 0, 0
        self.subscriber_playback = self.create_subscription(
            Bool,
            "/sim/playback_flag",
            self.callback_playback,
            1,
        )
        self.subscriber_reset = self.create_subscription(
            Bool,
            "/sim/reset_flag",
            self.callback_reset,
            1,
        )
        self.subscriber_teleop_recording = self.create_subscription(
            Bool, "/sim/is_recording", self.callback_recording, 1
        )
        self.pub_clock = self.create_publisher(Clock, "/clock", 1)
        self.playback_msg = False
        self.reset_msg = False
        self.recording_msg = False
        self.playback_lock = threading.Lock()
        self.reset_lock = threading.Lock()
        self.recording_lock = threading.Lock()

    def publish_clock(self, time_in_s):
        self.sec = int(time_in_s)
        self.nanosec = int((time_in_s - self.sec) * S_TO_NS)
        msg = Clock()
        msg.clock.sec = self.sec
        msg.clock.nanosec = self.nanosec
        self.pub_clock.publish(msg)

    def callback_playback(self, msg):
        self.playback_msg = msg.data

    def callback_recording(self, msg):
        self.recording_msg = msg.data

    def callback_reset(self, msg):
        self.reset_msg = msg.data

    def get_playback_state(self):
        with self.playback_lock:
            return self.playback_msg

    def get_reset(self):
        with self.reset_lock:
            return self.reset_msg

    def get_teleop_recording(self):
        with self.recording_lock:
            return self.recording_msg
