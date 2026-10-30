# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Merge Wuji finger commands with MC arm commands."""

import argparse
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState

from geniesim_teleop.bridge import BridgeSender, QOS_PROFILE_SENSOR
from geniesim_teleop.wuji.contracts import HandCommands, load_task


class WujiBridge(BridgeSender):
    def __init__(self, args):
        load_task(args.config)
        self.hands = HandCommands(args.model_urdf)
        super().__init__(args, node_name="wuji_command_bridge")
        for subscription in (self.sub_gripper_cmd, self.sub_chassis_cmd, self.sub_chassis_twist):
            self.destroy_subscription(subscription)
        self.sub_hand = self.create_subscription(JointState, args.hand_topic, self.on_hand, QOS_PROFILE_SENSOR)

    def on_hand(self, msg):
        if self.playback_flag:
            return
        try:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if stamp and not -0.1 <= time.time() - stamp < self.command_mux.timeout_s:
                raise ValueError("Hand command timestamp is stale or not in host wall time")
            commands = self.hands.split(msg.name, msg.position)
            for side, (names, positions) in commands.items():
                self.command_mux.update(f"wuji_{side}", names, positions, [], [], time.monotonic())
        except ValueError as exc:
            self.get_logger().warning(f"Rejected Wuji frame: {exc}", throttle_duration_sec=2.0)

    def _callback_joint_cmd(self, msg):
        if set(msg.name).intersection(self.hands.limits):
            self.get_logger().error("MC attempted to command Wuji fingers", throttle_duration_sec=2.0)
            return
        super()._callback_joint_cmd(msg)

    def _callback_playback(self, msg):
        if msg.data != self.playback_flag:
            for side in self.hands.sides:
                self.command_mux.sources.pop(f"wuji_{side}", None)
        super()._callback_playback(msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-urdf", required=True)
    parser.add_argument("--hand-topic", default="/teleop/wuji/hand_command")
    args = parser.parse_args()
    # Let KeyboardInterrupt unwind spin before destroying the context. The ROS
    # default SIGINT handler can invalidate a wait set during supervisor cleanup.
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = WujiBridge(args)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
