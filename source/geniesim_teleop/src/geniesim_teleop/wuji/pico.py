# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pico arm control and B-to-record for the separate Wuji session."""

import argparse
from rclpy.signals import uninstall_signal_handlers
from geniesim_teleop.teleop import TeleOp
from geniesim_teleop.utils.ros_utils import RosUtils
from geniesim_teleop.wuji.contracts import load_task


class WujiRosUtils(RosUtils):
    def start_ros_node(self):
        super().start_ros_node()
        # Preserve Python's SIGINT handling so TeleOp.run closes the spin thread
        # before shutting down its context, including while waiting for feedback.
        uninstall_signal_handlers()


class WujiPico(TeleOp):
    def __init__(self, args):
        super().__init__(args)
        self.ros_utils = WujiRosUtils(self.robot_name, node_name="wuji_pico", arm_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host_ip", required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    load_task(args.config)
    args.interface, args.robot_cfg, args.device_type, args.arm_only = "", None, "pico", True
    teleop = WujiPico(args)
    try:
        teleop.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
