#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import rclpy
import numpy as np

# from utils.ik_utils import IKSolver
from utils.logger import Logger
from utils.ros_utils import RosUtils
from utils.name_utils import *
from devices.pico_device import PicoDevice
from pynput import keyboard
from geometry_msgs.msg import Pose

import os, sys, argparse, math
from copy import deepcopy
import json
import time, subprocess, shutil, signal
import yaml

from config.robot_interface import *

logger = Logger()


class TeleOp(object):
    def __init__(self, args, ik_version="0.4.3"):
        self.args = args
        self.port = args.port
        self.reset_flg = False
        self.switch_flg = False
        self.robot_cfg = args.robot_cfg
        self.last_eef_pub = [None, None]
        self.last_on = [False, False]
        self.last_eef_control_on = [0.0, 0.0]
        self.ee_pub = [None, None]
        self.waist_pub = None
        self.waist_update = False
        self.update_wait_num = [0, 0]
        self.iskeyboard = False
        self.keyboard_pos = [None, None]
        self.ik_version = ik_version
        self.current_mode = "realtime"
        self.is_recording = False
        self.host_ip = self.get_local_ip()
        self.test_num = 0
        self.process_pid = []
        self.waist_angle = 0
        if self.host_ip == None:
            print("======>Can't get host_ip!!!! Please enter the ip address manually !!!")
            self.host_ip = args.host_ip
        self.setup_robot()
        self.count = 0
        self.pub_mc_count = 0
        self.waist_control_count = 0

        self.ros_utils = RosUtils(self.robot_name)
        self.device_type = args.device_type
        self.setup_device()

        self.robot_init_body_state = None
        self.robot_init_head_state = None
        self.robot_init_arm = None
        self.robot_init_hand = None

    def get_local_ip(self):
        try:
            # same as bash command
            result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=True)
            # get first IP (same as awk '{print $1}')
            ip = result.stdout.strip().split()[0]
            return ip
        except (subprocess.CalledProcessError, IndexError, FileNotFoundError) as e:
            return None

    def _extract_pose_from_joint_state(self, state):
        names = list(state.name)
        positions = list(state.position)

        def get_positions(joint_names):
            out = []
            for n in joint_names:
                if n in names:
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
        self.robot_eef = "omnipicker"
        self.robot_name = self.robot_cfg.split(".")[0]
        if "omnipicker" in self.robot_cfg:
            self.robot_eef = "omnipicker"
        elif "120s" in self.robot_cfg:
            self.robot_eef = "120s"
        else:
            raise ValueError(f"Invalid robot_cfg {self.robot_cfg}")

    def setup_device(self):
        if self.device_type == "pico":
            self.device = PicoDevice(self.host_ip, self.port, self.robot_cfg)
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
        # To be optimized
        init_marker = False
        while not init_marker:
            init_marker = self.ros_utils.sim_ros_node.initialize()
            time.sleep(0.1)
            print("waiting for initialize")
        print("initialize success")

        self.reset_command()
        self.current_step = 0
        self.eval_interval = 30

        self.device.initialize()
        self.joint_cmd = {}

        self._load_robot_init_states()

    def _load_robot_init_states(self):
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _source_dir = os.path.normpath(os.path.join(_this_dir, ".."))
        teleop_yaml_path = os.path.join(_source_dir, "geniesim", "config", "teleop.yaml")
        if not os.path.isfile(teleop_yaml_path):
            logger.warning(f"teleop.yaml not found: {teleop_yaml_path}, skip loading robot init states")
            return
        with open(teleop_yaml_path, "r", encoding="utf-8") as f:
            teleop_cfg = yaml.safe_load(f)
        sub_task_name = (teleop_cfg or {}).get("benchmark", {}).get("sub_task_name")
        if not sub_task_name:
            logger.warning("benchmark.sub_task_name not found in teleop.yaml, skip loading robot init states")
            return
        if _source_dir not in sys.path:
            sys.path.insert(0, _source_dir)
        try:
            from geniesim.benchmark.config.robot_init_states import TASK_INFO_DICT
        except Exception as e:
            logger.warning(f"failed to import TASK_INFO_DICT: {e}, skip loading robot init states")
            return
        task_states = TASK_INFO_DICT.get(sub_task_name)
        if not task_states:
            logger.warning(f"sub_task_name '{sub_task_name}' not in TASK_INFO_DICT, skip loading robot init states")
            return
        robot_state = task_states.get("G2_omnipicker")
        if not robot_state:
            logger.warning(
                f"G2_omnipicker not found for sub_task_name '{sub_task_name}', skip loading robot init states"
            )
            return
        self.robot_init_body_state = deepcopy(robot_state.get("body_state"))
        self.robot_init_head_state = deepcopy(robot_state.get("head_state"))
        self.robot_init_arm = deepcopy(robot_state.get("init_arm"))
        self.robot_init_hand = deepcopy(robot_state.get("init_hand"))
        self.waist_yaw = self.robot_init_body_state[0]
        self.waist_pitch = self.robot_init_body_state[2]
        logger.info(f"loaded robot init states for sub_task_name='{sub_task_name}' (G2_omnipicker)")

    def parse_arm_control(self):
        if self.current_mode == "playback":
            return
        if self.iskeyboard:
            self.ee_pub = self.ros_utils.sim_ros_node.parse_keyboard_pose(self.keyboard_pos)
            self.iskeyboard = False
        else:
            xyzxyzw_l = self.input.get("left")
            xyzxyzw_r = self.input.get("right")
            on_l = self.input.get("l_on")
            on_r = self.input.get("r_on")
            extra_r = self.input.get("r_b")

            if self.last_on[0] == False and on_l == True:
                self.ros_utils.sim_ros_node.update_without_judge()
            if self.last_on[1] == False and on_r == True:
                self.ros_utils.sim_ros_node.update_without_judge()
            self.last_on = [on_l, on_r]
            if xyzxyzw_l and on_l:
                pose = self.ros_utils.sim_ros_node.parse_delta_pose(xyzxyzw_l, "left")
                if pose != None:
                    self.ee_pub[0] = pose
            if xyzxyzw_r and on_r and (not extra_r):
                pose = self.ros_utils.sim_ros_node.parse_delta_pose(xyzxyzw_r, "right")
                if pose != None:
                    self.ee_pub[1] = pose

    def parse_eef_control(self):
        cmd_l, cmd_r = self.input.get("l_eef"), self.input.get("r_eef")
        if cmd_l is None or cmd_r is None:
            return
        output_l = 0.65 * cmd_l
        output_r = 0.65 * cmd_r
        self.ros_utils.set_joint_state(
            ["idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1"],
            [output_l, output_r],
        )
        self.last_eef_control_on = [output_l, output_r]

    def parse_waist_control(self):
        if not (self.input.get("r_axisX") or self.input.get("r_axisY")):
            return
        print(f"waist control: {self.input.get('r_axisX')} {self.input.get('r_axisY')}")
        self.waist_control_count += 1
        if self.waist_control_count < 10:
            return
        self.waist_control_count = 0
        waist_angle_limit = 1.67
        self.waist_yaw -= self.input.get("r_axisX") * 0.1
        self.waist_pitch += self.input.get("r_axisY") * 0.1
        print(f"=====>waist control: {self.waist_yaw} {self.waist_pitch}")
        self.waist_yaw = max(-waist_angle_limit, min(waist_angle_limit, self.waist_yaw))
        self.waist_pitch = max(-waist_angle_limit, min(waist_angle_limit, self.waist_pitch))
        self.ros_utils.sim_ros_node.pub_waist_pose(self.robot_init_body_state, self.waist_yaw, self.waist_pitch)

    def parse_body_control(self):
        if not (self.input.get("l_axisX") or self.input.get("l_axisY")):
            return
        name = [
            "idx111_chassis_lwheel_front_joint1",
            "idx112_chassis_lwheel_front_joint2",
            "idx131_chassis_rwheel_front_joint1",
            "idx132_chassis_rwheel_front_joint2",
            "idx121_chassis_lwheel_rear_joint1",
            "idx122_chassis_lwheel_rear_joint2",
            "idx141_chassis_rwheel_rear_joint1",
            "idx142_chassis_rwheel_rear_joint2",
        ]
        position = [0.0] * 8
        velocity = [0.0] * 8
        wheel_yaw = self.input.get("l_axisX")
        wheel_velocity = self.input.get("l_axisY")
        position[0] = -wheel_yaw
        position[2] = -wheel_yaw
        position[4] = -wheel_yaw
        position[6] = -wheel_yaw
        velocity[1] = 2.0 * math.pi * wheel_velocity
        velocity[3] = 2.0 * math.pi * wheel_velocity
        velocity[5] = 2.0 * math.pi * wheel_velocity
        velocity[7] = 2.0 * math.pi * wheel_velocity
        self.ros_utils.set_joint_position_and_velocity(name, position, velocity)

    def send_command(self):
        if self.ee_pub[0] != None or self.ee_pub[1] != None:
            self.ros_utils.sim_ros_node.pub_mc(self.ee_pub)

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

    def sub_keyboard_event(self):
        self.pressed_keys = set()

        def on_press(key):
            try:
                if key.char == "d":
                    self.waist_angle -= 0.01
                    self.iskeyboard = True
                elif key.char == "a":
                    self.waist_angle += 0.01
                    self.iskeyboard = True
            except AttributeError:
                pass

        def on_release(key):
            self.pressed_keys.discard(key)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

    def reset_whole_robot(self):
        left_reset = self.input.get("l_x")
        right_reset = False
        body_reset = self.input.get("r_a")

        left_arm_position = None
        right_arm_position = None
        body_position = None
        head_position = None

        if left_reset and self.robot_init_arm:
            left_arm_position = list(self.robot_init_arm[:7])
            right_arm_position = list(self.robot_init_arm[7:])
        # if right_reset and self.robot_init_arm:
        #     right_arm_position = list(self.robot_init_arm[7:])
        if body_reset:
            if self.robot_init_body_state is not None:
                body_position = list(reversed(self.robot_init_body_state))
            if self.robot_init_head_state is not None:
                head_position = list(self.robot_init_head_state)

        self.ros_utils.sim_ros_node.pub_robot_pose(body_position, head_position, left_arm_position, right_arm_position)

    def run(self):
        self.ros_utils.start_ros_node()
        self.initialize()
        # self.sub_keyboard_event()

        target_hz = 30.0
        target_period = 1.0 / target_hz
        while rclpy.ok():
            loop_start = time.time()

            self.input = self.device.update()
            # logger.info(f"Input {self.input}")
            if self.input:
                self.is_start_recording()
                self.ee_pub = [None, None]
                self.parse_arm_control()
                self.parse_waist_control()
                self.send_command()
                self.parse_eef_control()
                self.parse_body_control()
                self.reset_whole_robot()
                self.on_playback()

            elapsed = time.time() - loop_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        for process_pid in self.process_pid:
            os.kill(process_pid, signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--client_host", type=str, default="localhost:50051", help="The client")
    parser.add_argument("--host_ip", type=str, default="", help="Set vr host ip")
    parser.add_argument("--port", type=int, default=8080, help="Set vr port")
    parser.add_argument("--robot_cfg", type=str, default="G2_omnipicker.json", help="Set robot config")
    parser.add_argument("--device_type", type=str, default="pico", help="Set device type")
    # fmt: on
    args = parser.parse_args()

    teleop = TeleOp(args)
    try:
        teleop.run()
    except KeyboardInterrupt:
        teleop.ros_utils.stop_ros_node()
        pass


if __name__ == "__main__":
    main()
