#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import math
import os
import threading
import time
from copy import deepcopy

import numpy as np
import rclpy
from sensor_msgs.msg import JointState

from geniesim_teleop.config.chassis import ChassisControl
from geniesim_teleop.config.gripper import gripper_control
from geniesim_teleop.config.robot_interface import (
    BODY_JOINT_NAMES,
    HEAD_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)
from geniesim_teleop.config.task_config import load_teleop_task, robot_family
from geniesim_teleop.devices.pico_device import ControllerPoseClutch, PicoDevice
from geniesim_teleop.utils.logger import Logger
from geniesim_teleop.utils.network import add_network_arguments, select_host_ip
from geniesim_teleop.utils.ros_nodes import QOS_PROFILE_SENSOR
from geniesim_teleop.utils.ros_utils import RosUtils

logger = Logger()


class TeleOp(object):
    def __init__(self, args):
        self.args = args
        self.port = args.port
        self.reset_flg = False
        self.task_context = load_teleop_task(args.config, args.robot_cfg)
        self.chassis_control = ChassisControl.from_task(self.task_context.config_path)
        self.initial_states = self.task_context.mc_initial_states()
        self.robot_cfg = self.task_context.robot_cfg
        logger.info(f"Task {self.task_context.task_name}/{self.task_context.sub_task_name}: {self.robot_cfg}")
        self.last_eef_pub = [None, None]
        self.last_on = [False, False]
        self.controller_clutch = [ControllerPoseClutch(), ControllerPoseClutch()]
        self.waist_clutch = ControllerPoseClutch()
        self.waist_anchor = None
        self.last_eef_control_on = [0.0, 0.0]
        self.ee_pub = [None, None]
        self.waist_pub = None
        self.waist_update = False
        self.current_mode = "realtime"
        self.is_recording = False
        self.host_ip = select_host_ip(args.interface, args.host_ip)
        logger.info(f"Pico network address: {self.host_ip}:{self.port}")
        self.setup_robot()
        self.count = 0
        self.pub_mc_count = 0
        self.waist_control_count = 0

        self.ros_utils = RosUtils(self.robot_name, arm_only=args.arm_only)
        self.device_type = args.device_type
        self.setup_device()

        self.robot_init_body_state = None
        self.robot_init_head_state = None
        self.robot_init_arm = None
        self.robot_init_hand = None

    def _extract_pose_from_joint_state(self, state):
        names = list(state.name)
        positions = list(state.position)

        def get_positions(joint_names):
            out = []
            for n in joint_names:
                if n in names and names.index(n) < len(positions):
                    out.append(positions[names.index(n)])
                else:
                    return None
            return out

        return (
            get_positions(BODY_JOINT_NAMES),
            get_positions(HEAD_JOINT_NAMES),
            get_positions(LEFT_ARM_JOINT_NAMES),
            get_positions(RIGHT_ARM_JOINT_NAMES),
        )

    def setup_robot(self):
        self.robot_name = self.robot_cfg.removesuffix(".json")
        self.robot_eef = robot_family(self.robot_cfg)
        self.gripper_control = None if self.args.arm_only else gripper_control(self.robot_cfg)

    def setup_device(self):
        if self.device_type == "pico":
            self.device = PicoDevice(self.host_ip, self.port, self.robot_cfg, self.chassis_control.input_timeout_s)
        else:
            raise ValueError(f"Unsupported device_type {self.device_type}")

    def reset_command(self):
        self.command_ = np.array([0.0, 0.0, 0.0])
        self.rotation_command = np.array([0.0, 0.0, 0.0])
        self.robot_command = np.array([0.0, 0.0, 0.0])
        self.robot_rotation_command = np.array([0.0, 0.0, 0.0])
        self.waist_command = np.array([0.0, 0.0])
        self.head_command = np.array([0.0, 0.0])

    def initialize(self):
        init_marker = False
        while rclpy.ok() and not init_marker:
            self.ros_utils.check_health()
            init_marker = self.ros_utils.sim_ros_node.initialize()
            time.sleep(0.1)
            print("waiting for initialize")
        if not init_marker:
            return False
        print("initialize success")

        self.reset_command()
        self.current_step = 0
        self.eval_interval = 30

        self.device.initialize()
        self.joint_cmd = {}

        self._load_robot_init_states()
        return self._initialize_mc()

    def _load_robot_init_states(self):
        robot_state = self.initial_states
        self.robot_init_body_state = deepcopy(robot_state.get("body_state"))
        self.robot_init_head_state = deepcopy(robot_state.get("head_state"))
        self.robot_init_arm = deepcopy(robot_state.get("init_arm"))
        self.robot_init_hand = deepcopy(robot_state.get("init_hand"))
        logger.info(f"Loaded task initial states for {self.robot_cfg}")

    def _publish_initial_pose(self):
        left = list(self.robot_init_arm[:7])
        right = list(self.robot_init_arm[7:])
        body = deepcopy(self.robot_init_body_state) if not self.args.arm_only else None
        head = deepcopy(self.robot_init_head_state) if not self.args.arm_only else None
        self.ros_utils.sim_ros_node.pub_robot_pose(body, head, left, right)

    def _initialize_mc(self):
        """Send the task reset before accepting tracking input, including Wuji."""
        node = self.ros_utils.sim_ros_node
        ready = threading.Event()
        arm_names = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
        target = dict(zip(arm_names, self.robot_init_arm))

        def on_command(msg):
            if len(msg.position) != len(msg.name):
                return
            positions = dict(zip(msg.name, msg.position))
            if all(
                name in positions and math.isfinite(positions[name]) and abs(positions[name] - target[name]) <= 0.02
                for name in arm_names
            ):
                ready.set()

        subscription = node.create_subscription(JointState, "/joint_command", on_command, QOS_PROFILE_SENSOR)
        deadline = time.monotonic() + 15.0
        sent = False
        try:
            while rclpy.ok() and time.monotonic() < deadline:
                self.ros_utils.check_health()
                _, _, left, right = self._extract_pose_from_joint_state(node.get_joint_state())
                if left is not None and right is not None and all(math.isfinite(q) for q in left + right):
                    if (
                        sent
                        and ready.is_set()
                        and all(
                            abs(actual - expected) <= 0.02
                            for actual, expected in zip(left + right, self.robot_init_arm)
                        )
                    ):
                        logger.info("Task initial arm pose reached; Pico tracking enabled")
                        return True
                    # MC needs feedback and its first joint target before it
                    # publishes. Use exactly the same task pose as Pico X.
                    self._publish_initial_pose()
                    sent = True
                time.sleep(0.2)
            if not rclpy.ok():
                return False
            missing = (
                "complete arm feedback" if not sent else "task arm targets on /joint_command and matching arm feedback"
            )
            raise RuntimeError(f"Teleop startup timed out waiting for {missing}; inspect MC and bridge output")
        finally:
            node.destroy_subscription(subscription)

    def parse_arm_control(self):
        if self.current_mode == "playback":
            return
        xyzxyzw_l = self.input.get("left")
        xyzxyzw_r = self.input.get("right")
        on_l = bool(self.input.get("l_on"))
        # A takes ownership of the right controller until released. The
        # arm clutch disengages too, so resuming the arm starts at zero delta.
        waist_on = bool(self.input.get("r_a")) and not self.args.arm_only
        on_r = bool(self.input.get("r_on")) and not waist_on

        rising_l = not self.last_on[0] and on_l
        rising_r = not self.last_on[1] and on_r
        if rising_l or rising_r:
            if self.ros_utils.sim_ros_node.update_without_judge(self.last_eef_pub) is False:
                return

        # Re-anchor the controllers against the held MC targets, not
        # lagging measured wrists. Both active hands must use the refreshed
        # arm-base frame without changing either arm's commanded pose.
        reanchor = rising_l or rising_r
        delta_l = self.controller_clutch[0].update(
            xyzxyzw_l,
            on_l,
            reanchor=reanchor and on_l,
        )
        delta_r = self.controller_clutch[1].update(
            xyzxyzw_r,
            on_r,
            reanchor=reanchor and on_r,
        )
        self.last_on = [on_l, on_r]
        if delta_l is not None:
            pose = self.ros_utils.sim_ros_node.parse_delta_pose(delta_l, "left")
            if pose != None:
                self.ee_pub[0] = pose
        if delta_r is not None:
            pose = self.ros_utils.sim_ros_node.parse_delta_pose(delta_r, "right")
            if pose != None:
                self.ee_pub[1] = pose

    def parse_eef_control(self):
        cmd_l, cmd_r = self.input.get("l_eef"), self.input.get("r_eef")
        if cmd_l is None or cmd_r is None:
            return
        if self.gripper_control is None:
            return
        targets = self.gripper_control.targets(cmd_l, cmd_r)
        names = list(self.gripper_control.joint_names)
        if self.gripper_control.mode == "velocity":
            self.ros_utils.set_joint_velocity(names, targets)
        else:
            self.ros_utils.set_joint_state(names, targets)
        self.last_eef_control_on = targets

    def parse_waist_control(self):
        active = bool(self.input.get("r_a"))
        if not active:
            self.waist_clutch.update(None, False)
            self.waist_anchor = None
            return
        controller_pose = self.input.get("right")
        if controller_pose is None:
            return
        node = self.ros_utils.sim_ros_node
        if self.waist_anchor is None:
            # Keep a separate anchor: engaging either arm refreshes node.parts.
            self.waist_anchor = node.get_link_transform_from_base_link("arm_base_link")
            if self.waist_anchor is None:
                return
        delta = self.waist_clutch.update(controller_pose, True)
        self.waist_pub = node.parse_waist_delta_pose(delta, self.waist_anchor)

    def parse_body_control(self):
        # Publish neutral too: wheel velocity commands persist until replaced.
        self.ros_utils.sim_ros_node.pub_chassis_twist(*self.chassis_control.velocity(self.input))

    def stop_motion(self):
        if self.args.arm_only:
            return
        self.ros_utils.sim_ros_node.pub_chassis_twist(0.0, 0.0, 0.0)
        if self.gripper_control is not None and self.gripper_control.mode == "velocity":
            self.ros_utils.set_joint_velocity(list(self.gripper_control.joint_names), [0.0, 0.0])

    def send_command(self):
        if any(pose is not None for pose in self.ee_pub) or self.waist_pub is not None:
            self.ros_utils.sim_ros_node.pub_mc(self.ee_pub, self.waist_pub)
            for i, pose in enumerate(self.ee_pub):
                if pose is not None:
                    # MC keeps this arm-base-relative target after grip release.
                    self.last_eef_pub[i] = deepcopy(pose)

    def reset_tracking(self):
        self.last_eef_pub = [None, None]
        for clutch in self.controller_clutch:
            clutch.update(None, False)
        self.last_on = [False, False]
        self.waist_clutch.update(None, False)
        self.waist_anchor = None
        self.waist_pub = None
        self.ee_pub = [None, None]

    def on_playback(self):
        if self.device.extra_l():
            print("Pub sig")
            self.ros_utils.sim_ros_node.pub_playback(True)
            self.current_mode = "playback"
            state = self.ros_utils.sim_ros_node.get_joint_state()
            body_position, head_position, left_arm_position, right_arm_position = self._extract_pose_from_joint_state(
                state
            )
            self.ros_utils.sim_ros_node.pub_robot_pose(
                body_position, head_position, left_arm_position, right_arm_position
            )
        else:
            self.ros_utils.sim_ros_node.pub_playback(False)
            if self.current_mode == "playback":
                self.current_mode = "realtime"

    def is_start_recording(self):
        if self.is_recording == True:
            return
        if self.device.extra_r():
            self.ros_utils.sim_ros_node.pub_recording(True)
            self.is_recording = True

    def reset_whole_robot(self):
        if not self.input.get("l_x"):
            return False
        self.reset_tracking()
        self.current_mode = "realtime"
        if not self.args.arm_only:
            self.ros_utils.sim_ros_node.pub_playback(False)
        self._publish_initial_pose()
        return True

    def process_input(self):
        if not self.input:
            self.stop_motion()
            self.reset_tracking()
            return
        self.is_start_recording()
        self.ee_pub = [None, None]
        # Reset has priority over A and arm grips; no pose target can overwrite it.
        if self.reset_whole_robot():
            self.stop_motion()
            return
        if not self.args.arm_only:
            self.on_playback()
            if self.current_mode == "playback":
                self.stop_motion()
                self.reset_tracking()
                return
        self.parse_arm_control()
        if not self.args.arm_only:
            self.parse_waist_control()
            self.parse_eef_control()
            self.parse_body_control()
        self.send_command()

    def run(self):
        try:
            self.ros_utils.start_ros_node()
            self._run_loop()
        finally:
            try:
                if rclpy.ok() and self.ros_utils.sim_ros_node is not None:
                    self.stop_motion()
            finally:
                try:
                    self.device.close()
                finally:
                    self.ros_utils.stop_ros_node()

    def _run_loop(self):
        if not self.initialize():
            return

        target_hz = 30.0
        target_period = 1.0 / target_hz
        while rclpy.ok():
            self.ros_utils.check_health()
            loop_start = time.time()

            self.input = self.device.update()
            self.process_input()

            elapsed = time.time() - loop_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--client_host", type=str, default="localhost:50051", help="The client")
    add_network_arguments(parser)
    parser.add_argument("--port", type=int, default=8080, help="Set vr port")
    parser.add_argument("--config", default=os.environ.get("GENIESIM_TELEOP_CONFIG"), help="Task YAML (default: benchmark config/teleop.yaml)")
    parser.add_argument("--robot_cfg", default=None, help="Optional assertion: must match the robot selected by the task")
    parser.add_argument("--device_type", type=str, default="pico", help="Set device type")
    parser.add_argument(
        "--arm_only",
        action="store_true",
        help="Only publish arm commands; leave hand and whole-body joints to external controllers",
    )
    # fmt: on
    args = parser.parse_args()

    try:
        teleop = TeleOp(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    try:
        teleop.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
