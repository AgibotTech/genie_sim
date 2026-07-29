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
        # Multi-episode recording: autoteleop.sh publishes on this topic when the
        # operator confirms an episode with y/n. The simulator then finalizes the
        # current ros2 bag gracefully (instead of being killed) and re-arms for the
        # next record-button press.
        self.subscriber_stop_episode = self.create_subscription(
            Bool, "/sim/stop_episode", self.callback_stop_episode, 1
        )
        self.pub_clock = self.create_publisher(Clock, "/clock", 1)
        self.playback_msg = False
        self.reset_msg = False
        self.recording_msg = False
        self.stop_episode_msg = False
        self.playback_lock = threading.Lock()
        self.reset_lock = threading.Lock()
        self.recording_lock = threading.Lock()
        self.stop_episode_lock = threading.Lock()

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

    # ---- Multi-episode recording -------------------------------------------
    def callback_stop_episode(self, msg):
        # Latch only; api_core clears it via clear_stop_episode() once handled,
        # so the signal is processed exactly once.
        if msg.data:
            with self.stop_episode_lock:
                self.stop_episode_msg = True

    def get_stop_episode(self):
        with self.stop_episode_lock:
            return self.stop_episode_msg

    def clear_stop_episode(self):
        with self.stop_episode_lock:
            self.stop_episode_msg = False

    def reset_recording_state(self):
        # After an episode ends, drop the cached "record button pressed" state.
        # Without this, re-arming wait_recording would immediately start the next
        # episode from the stale True value instead of waiting for a new press.
        with self.recording_lock:
            self.recording_msg = False
