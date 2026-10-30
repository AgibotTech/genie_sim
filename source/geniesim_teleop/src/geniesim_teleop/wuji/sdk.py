# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Publish fresh glove frames using the teleop-owned Wuji SDK adapter."""

import argparse
import time

from geniesim_teleop.wuji.contracts import load_task
from geniesim_teleop.wuji.hand_bridge import WujiSdkBridge, sdk_joint_command


class FreshWujiSdkBridge(WujiSdkBridge):
    def __init__(self, **kwargs):
        self.received = {}
        super().__init__(**kwargs)

    def _on_timer(self):
        fresh = {}
        now = time.monotonic()
        for side, source in self._inputs.items():
            try:
                qpos = source.poll_latest()
            except Exception as exc:
                self.node.get_logger().warning(f"{side} glove frame rejected: {exc}", throttle_duration_sec=2.0)
                qpos = None
            if qpos is not None:
                fresh[side] = qpos
                self._latest[side] = qpos
                self.received[side] = now
            elif now - self.received.get(side, 0.0) > 0.3:
                self.node.get_logger().warning(
                    f"No fresh {side} glove data; holding the last hand target", throttle_duration_sec=2.0
                )
        if not fresh or any(side not in self._latest for side in self._inputs):
            return
        if not self._started:
            if any(now - self.received[side] > 0.3 for side in self._inputs):
                return
            fresh = self._latest.copy()
            self._started = True
        names, positions = sdk_joint_command(fresh)
        message = self._joint_state_type()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name, message.position = names, positions
        self.publisher.publish(message)


def main():
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.signals import SignalHandlerOptions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--left-sn", default="")
    parser.add_argument("--right-sn", default="")
    parser.add_argument("--sdk-user", default="")
    parser.add_argument("--output-topic", default="/teleop/wuji/hand_command")
    args = parser.parse_args()
    if not (args.left_sn or args.right_sn):
        parser.error("Provide --left-sn and/or --right-sn")
    load_task(args.config)
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    bridge = None
    try:
        bridge = FreshWujiSdkBridge(
            left_sn=args.left_sn,
            right_sn=args.right_sn,
            sdk_user=args.sdk_user,
            output_topic=args.output_topic,
            rate_hz=100.0,
        )
        rclpy.spin(bridge.node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if bridge is not None:
                try:
                    bridge.close()
                finally:
                    bridge.node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
