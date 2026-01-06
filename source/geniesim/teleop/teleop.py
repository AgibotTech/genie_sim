# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import os, sys
import json
import numpy as np
import glob
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import geniesim.utils.system_utils as system_utils
from geniesim.benchmark.envs.dummy_env import DummyEnv
from geniesim.robot.genie_robot import IsaacSimRpcRobot
from geniesim.layout.task_generate import TaskGenerator
from vr_server import VRServer
from scipy.spatial.transform import Rotation as R
import math
from pynput import keyboard
import threading

from geniesim.plugins.logger import Logger
from geniesim.plugins.output_system.eval_utils import *
from geniesim.utils.ros_nodes.base_nodes import SimNode
from geniesim.utils.name_utils import *
from copy import deepcopy
import rclpy
import time
from datetime import datetime, timezone, timedelta
import signal

logger = Logger()


def handle_exit(signum, frame):
    raise KeyboardInterrupt(f"Receive signal {signum}, start cleaning...")


signal.signal(signal.SIGINT, handle_exit)  # Ctrl+C


def get_local_time():
    tz_utc8 = timezone(timedelta(hours=8))
    utc_now = datetime.now(tz_utc8)
    return str(utc_now)


def scale_quat_to_euler(q, s):
    rotation = R.from_quat(q)
    rpy = rotation.as_euler("xzy", degrees=False)
    r, p, y = rpy
    if r >= 0:
        r_trans = math.pi - r
    else:
        r_trans = -math.pi - r
    p_trans = -p
    y_trans = y
    rpy_trans = np.array([p_trans, -r_trans, y_trans])
    rpy_scaled = rpy_trans * s

    return rpy_scaled


class TeleOp(object):
    def __init__(self, args):
        if args.task_name != "":
            self.task_name = args.task_name
        else:
            raise ValueError("Invalid task_name or task_id")
        self.args = args
        self.task_config = None
        self.mode = args.mode
        self.record = args.record
        self.fps = args.fps
        self.episodes_per_instance = 1
        self.host_ip = args.host_ip
        self.port = args.port
        if self.mode == "pico":
            self.init_pico_control()
            self.init_keyboard_control()
        elif self.mode == "keyboard":
            self.init_keyboard_control()
        else:
            raise ValueError("Invalid mode")
        self.reset_flg = False
        self.switch_flg = False
        self.single_evaluate_ret = deepcopy(EVAL_TEMPLATE)
        self.eval_out_dir = os.path.join(system_utils.teleop_root_path(), "output")
        self.operator = args.operator
        self.robot_cfg = args.robot_cfg
        self.robot_generation = "G1"
        if "G2" in self.robot_cfg:
            self.robot_generation = "G2"
        self.robot_eef = "120s"
        if "omnipicker" in self.robot_cfg:
            self.robot_eef = "omnipicker"
        elif "120s" in self.robot_cfg:
            self.robot_eef = "120s"
        self.current_mode = "realtime"

    def init_pico_control(self):
        self.vr_server = VRServer(self.host_ip, self.port)

    def init_keyboard_control(self):
        self.reset_command()
        self.gripper_command = "close"
        self.current_gripper_state = "close"
        self.current_arm_type = "right"
        default_cmd = {
            "pos": np.array([0.0, 0.0, 0.0]),
            "rot": np.array([0.0, 0.0, 0.0]),
        }
        self.last_command = [default_cmd, default_cmd]

    def reset_command(self):
        self.command_ = np.array([0.0, 0.0, 0.0])
        self.rotation_command = np.array([0.0, 0.0, 0.0])
        self.robot_command = np.array([0.0, 0.0, 0.0])
        self.robot_rotation_command = np.array([0.0, 0.0, 0.0])
        self.waist_command = np.array([0.0, 0.0])
        self.head_command = np.array([0.0, 0.0])
        self.control_mode = "pico"

    def sub_keyboard_event(self):
        self.pressed_keys = set()

        def on_press(key):
            try:
                if key.char == "w":
                    self.command_ += np.array([0.01, 0.0, 0.0])
                elif key.char == "s":
                    self.command_ += np.array([-0.01, 0.0, 0.0])
                elif key.char == "a":
                    self.command_ += np.array([0.0, 0.01, 0.0])
                elif key.char == "d":
                    self.command_ += np.array([0.0, -0.01, 0.0])
                elif key.char == "q":
                    self.command_ += np.array([0.0, 0.0, 0.01])
                elif key.char == "e":
                    self.command_ += np.array([0.0, 0.0, -0.01])
                elif key.char == "j":
                    self.rotation_command += np.array([-0.02, 0.0, 0.0])
                elif key.char == "l":
                    self.rotation_command += np.array([0.02, 0.0, 0.0])
                elif key.char == "i":
                    self.rotation_command += np.array([0.0, 0.02, 0.0])
                elif key.char == "k":
                    self.rotation_command += np.array([0.0, -0.02, 0.0])
                elif key.char == "u":
                    self.rotation_command += np.array([0.0, 0.0, 0.02])
                elif key.char == "o":
                    self.rotation_command += np.array([0.0, 0.0, -0.02])
                elif key.char == "c":
                    if keyboard.Key.ctrl in self.pressed_keys:
                        self.gripper_command = "open"
                    else:
                        self.gripper_command = "close"
                elif key.char == "r":
                    self.reset_flg = True
                elif key.char == "b":
                    if self.control_mode == "pico":
                        self.control_mode = "keyboard"
                    else:
                        self.control_mode = "pico"

            except AttributeError:
                self.pressed_keys.add(key)
                if key == keyboard.Key.up:
                    if keyboard.Key.shift in self.pressed_keys:
                        self.head_command += np.array([0.0, -0.01])
                    elif keyboard.Key.ctrl in self.pressed_keys:
                        self.waist_command += np.array([0.01, 0.0])
                    else:
                        self.robot_command += np.array([0.02, 0.0, 0.0])
                elif key == keyboard.Key.down:
                    if keyboard.Key.shift in self.pressed_keys:
                        self.head_command += np.array([0.0, 0.01])
                    elif keyboard.Key.ctrl in self.pressed_keys:
                        self.waist_command += np.array([-0.01, 0.0])
                    else:
                        self.robot_command += np.array([-0.02, 0.0, 0.0])
                elif key == keyboard.Key.left:
                    if keyboard.Key.shift in self.pressed_keys:
                        self.head_command += np.array([0.01, 0.0])
                    elif keyboard.Key.ctrl in self.pressed_keys:
                        self.waist_command += np.array([0.0, 0.01])
                    else:
                        self.robot_rotation_command += np.array([0.0, 0.0, 0.02])
                elif key == keyboard.Key.right:
                    if keyboard.Key.shift in self.pressed_keys:
                        self.head_command += np.array([-0.01, 0.0])
                    elif keyboard.Key.ctrl in self.pressed_keys:
                        self.waist_command += np.array([0.0, -0.01])
                    else:
                        self.robot_rotation_command += np.array([0.0, 0.0, -0.02])
                elif key == keyboard.Key.tab and keyboard.Key.ctrl in self.pressed_keys:
                    if self.current_arm_type == "right":
                        self.current_arm_type = "left"
                        self.last_command[1] = {
                            "pos": self.command_,
                            "rot": self.rotation_command,
                        }
                    else:
                        self.current_arm_type = "right"
                        self.last_command[0] = {
                            "pos": self.command_,
                            "rot": self.rotation_command,
                        }
                    self.switch_flg = True

        def on_release(key):
            self.pressed_keys.discard(key)
            if key == keyboard.Key.up or key == keyboard.Key.down:
                self.robot_command = np.array([0.0, 0.0, 0.0])
            elif key == keyboard.Key.left or key == keyboard.Key.right:
                self.robot_rotation_command = np.array([0.0, 0.0, 0.0])

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

    def decode_arm(self, joint_pos, joint_name, precision=3):
        l_arm, r_arm, body, gripper, ori_pos, base_pos = {}, {}, {}, {}, {}, {}
        for idx, name in enumerate(joint_name):
            val = round(joint_pos[idx], precision)
            if name.startswith("idx2"):
                l_arm[name] = val
            elif name.startswith("idx6"):
                r_arm[name] = val
            elif "body" in name or "head" in name:
                body[name] = val
            elif name in OMNIPICKER_AJ_NAMES:
                gripper[name] = val
            if name in G1_CHASSIS:
                base_pos[name] = val
                continue
            ori_pos[name] = val
        return l_arm, r_arm, body, gripper, ori_pos, base_pos

    def parse_joint_pose(self):
        joint_pos, joint_name = self.get_joint_state()
        l_arm, r_arm, body, gripper, ori_pos, base_pos = self.decode_arm(joint_pos, joint_name)
        self.joint_info = {
            "l_arm": l_arm,
            "r_arm": r_arm,
            "body": body,
            "gripper": gripper,
            "ori_pos": ori_pos,
            "base_pos": base_pos,
        }

    def parse_pico_command(self, content):
        """
        Args:
            content (list(dict)): parse pico control command to local frame
        """
        [l_sig, r_sig] = content
        ret = {"l": {}, "r": {}}
        ret["l"]["position"] = np.array([l_sig["position"]["z"], -l_sig["position"]["x"], l_sig["position"]["y"]])
        ret["l"]["quaternion"] = np.array(
            [
                l_sig["rotation"]["w"],
                l_sig["rotation"]["x"],
                l_sig["rotation"]["y"],
                l_sig["rotation"]["z"],
            ]
        )
        ret["l"]["axisMode"] = "reset" if l_sig["axisClick"] == "true" else "move"
        ret["l"]["axisX"] = l_sig["axisX"]
        ret["l"]["axisY"] = l_sig["axisY"]
        ret["l"]["gripper"] = 1 - l_sig["indexTrig"]
        ret["l"]["On"] = l_sig["handTrig"] > 0.9
        ret["l"]["reset"] = l_sig["keyOne"] == "true"
        ret["l"]["against"] = l_sig["keyTwo"] == "true"  # also used as playback

        ret["r"]["position"] = np.array([r_sig["position"]["z"], -r_sig["position"]["x"], r_sig["position"]["y"]])
        print("XYZ", ret["r"]["position"])
        ret["r"]["quaternion"] = np.array(
            [
                r_sig["rotation"]["w"],
                r_sig["rotation"]["x"],
                r_sig["rotation"]["y"],
                r_sig["rotation"]["z"],
            ]
        )
        ret["r"]["axisX"] = r_sig["axisX"]
        ret["r"]["axisY"] = r_sig["axisY"]
        ret["r"]["axisMode"] = "head" if r_sig["axisClick"] == "true" else "waist"
        ret["r"]["gripper"] = 1 - r_sig["indexTrig"]
        ret["r"]["On"] = r_sig["handTrig"] > 0.9
        ret["r"]["reset"] = r_sig["keyOne"] == "true"
        ret["r"]["resetbh"] = r_sig["keyTwo"] == "true"

        self.pico_command = ret

    def parse_body_control(self):
        if self.pico_command["r"]["axisMode"] == "head":
            delta_angle = 2e-3
            thresh = 0.5
            head_yaw = self.pico_command["r"]["axisX"]
            head_pitch = self.pico_command["r"]["axisY"]

        if self.pico_command["r"]["axisMode"] == "waist":
            delta_angle = 1e-1
            THRESH = 0.5
            waist_lift_sig = self.pico_command["r"]["axisY"]
            waist_pitch_sig = self.pico_command["r"]["axisX"]
            enabled = False
            if self.robot_generation == "G1":
                waist_lift = self.joint_info["body"]["idx01_body_joint1"]
                waist_pitch = self.joint_info["body"]["idx02_body_joint2"]
                if waist_lift_sig > THRESH:
                    waist_lift += delta_angle
                    enabled = True
                if waist_lift_sig < -THRESH:
                    waist_lift -= delta_angle
                    enabled = True
                if waist_pitch_sig > THRESH:
                    waist_pitch += delta_angle
                    enabled = True
                if waist_pitch_sig < -THRESH:
                    waist_pitch -= delta_angle
                    enabled = True
                if enabled:
                    print("waist cmd", waist_lift)
                    self.joint_cmd.update(
                        {
                            "idx01_body_joint1": waist_lift,
                            "idx02_body_joint2": waist_pitch,
                        }
                    )
            elif self.robot_generation == "G2":
                waist_lift1 = self.joint_info["body"]["idx01_body_joint1"]
                waist_lift2 = self.joint_info["body"]["idx02_body_joint2"]
                waist_lift4 = self.joint_info["body"]["idx04_body_joint4"]
                if waist_lift_sig > THRESH:
                    waist_lift2 -= delta_angle
                    waist_lift4 -= delta_angle
                    enabled = True
                if waist_lift_sig < -THRESH:
                    waist_lift2 += delta_angle
                    waist_lift4 += delta_angle
                    enabled = True

                if enabled:
                    self.joint_cmd.update(
                        {
                            # "idx01_body_joint1": waist_lift1,
                            "idx02_body_joint2": waist_lift2,
                            "idx04_body_joint4": waist_lift4,
                        }
                    )

    def parse_arm_control(self, coef_pos, coef_quat):
        on_l, on_r = self.pico_command["l"]["On"], self.pico_command["r"]["On"]
        rescaled_pos_l = coef_pos * self.pico_command["l"]["position"]
        rescaled_pos_r = coef_pos * self.pico_command["r"]["position"]
        rescaled_rpy_l = scale_quat_to_euler(self.pico_command["l"]["quaternion"], coef_quat)
        rescaled_rpy_r = scale_quat_to_euler(self.pico_command["r"]["quaternion"], coef_quat)
        reset_l = self.pico_command["l"]["reset"]
        reset_r = self.pico_command["r"]["reset"]
        reset_bh = self.pico_command["r"]["resetbh"]

        jp_l = self.env.robot.get_joint_from_deltapos(xyz=rescaled_pos_l, rpy=rescaled_rpy_l, id="left", isOn=on_l)
        jp_r = self.env.robot.get_joint_from_deltapos(xyz=rescaled_pos_r, rpy=rescaled_rpy_r, id="right", isOn=on_r)

        for idx, val in enumerate(jp_l):
            self.joint_cmd[list(self.init_l_arm.keys())[idx]] = val

        for idx, val in enumerate(jp_r):
            self.joint_cmd[list(self.init_r_arm.keys())[idx]] = val

        if reset_l:
            logger.info("reset left arm...")
            self.env.robot.initialize_solver(self.init_l_arm, False)
            self.joint_cmd.update(self.init_l_arm)
        if reset_r:
            logger.info("reset right arm...")
            self.env.robot.initialize_solver(self.init_r_arm, True)
            self.joint_cmd.update(self.init_r_arm)
        if reset_bh:
            logger.info("reset body and head...")
            self.joint_cmd.update(self.init_body)

    def parse_gripper_control(self):
        cmd_l, cmd_r = (
            self.pico_command["l"]["gripper"],
            self.pico_command["r"]["gripper"],
        )
        self.joint_cmd.update({self.active_joint_name[0]: cmd_l, self.active_joint_name[1]: cmd_r})

    def apply_base_control(self):
        if self.current_mode == "playback":
            return
        x = self.pico_command["l"]["axisY"]
        y = -self.pico_command["l"]["axisX"]
        mode = self.pico_command["l"]["axisMode"]
        activated = False
        THRESHOLD = 0.8
        target_x, target_yaw = 0.0, 0.0
        if mode == "move":
            if x > THRESHOLD:
                target_x = 0.02
                activated = True
            if x < -THRESHOLD:
                target_x = -0.02
                activated = True
            if y < -THRESHOLD:
                target_yaw = -0.03
                activated = True
            if y > THRESHOLD:
                target_yaw = 0.03
                activated = True

            if activated:
                pos_diff = np.array([target_x, 0, 0])
                rot_diff = np.array([0, 0, target_yaw])
                target_pos, target_rpy = self.env.robot.update_odometry(pos_diff, rot_diff)
                if target_rpy[2] >= np.pi:
                    target_rpy[2] = np.pi - 0.01
                elif target_rpy[2] <= -np.pi:
                    target_rpy[2] = -np.pi + 0.01

                self.sim_ros_node.set_joint_state(name=[G1_CHASSIS[2]], position=[float(target_rpy[2])])
                time.sleep(0.1)
                self.sim_ros_node.set_joint_state(name=G1_CHASSIS[0:2], position=[float(v) for v in target_pos[0:2]])

        else:
            self.reset_robot_pose()

    def initialize(self):
        self.parse_joint_pose()
        self.init_pos = self.joint_info["ori_pos"]
        self.init_l_arm = self.joint_info["l_arm"]
        self.init_r_arm = self.joint_info["r_arm"]
        self.init_body = self.joint_info["body"]
        self.init_gripper = self.joint_info["gripper"]
        self.env.robot.initialize_solver(self.init_l_arm, False)
        self.env.robot.initialize_solver(self.init_r_arm, True)
        self.robot_init_pos, self.robot_init_rot = self.env.robot.get_init_pose()
        self.reset_command()
        self.current_step = 0
        self.eval_interval = 30
        self.active_joint_name = list(self.joint_info["gripper"].keys())
        assert len(self.active_joint_name) == 2, "Active joint dimension should be two"
        logger.info(f"Active joint name: {self.active_joint_name}")
        self.joint_cmd = deepcopy(self.init_pos)

    def reset_robot_pose(self):
        self.env.robot.set_base_pose(target_pos=self.robot_init_pos, target_rot=self.robot_init_rot)
        self.env.robot.reset_odometry()

    def update_eval_ret(self, task_progress):
        self.single_evaluate_ret["result"]["progress"] = task_progress

    def run_eval(self):
        if self.current_step != 0 and self.current_step % self.eval_interval == 0:
            self.env.action_update()
            self.update_eval_ret(self.env.task.task_progress)
        self.current_step += 1

    def shuffle_pose(self, obj_info):
        center = obj_info["center"]
        size = obj_info["size"]
        old_pos = obj_info["cur_pos"]
        old_quat = obj_info["cur_quat"]
        dx, dy, dz = [np.random.uniform(-v / 2, v / 2) for v in size]
        new_pos = [center[0] + dx, center[1] + dy, old_pos[2] + 0.01]
        while np.linalg.norm(new_pos - old_pos) < self.POS_DIFF_THRESHOLD:
            logger.info("shuffle pose diff too small, try another")
            dx, dy, dz = [np.random.uniform(-v / 2, v / 2) for v in size]
            new_pos = [center[0] + dx, center[1] + dy, old_pos[2]]
        return new_pos, old_quat

    def apply_against_collection(self):
        self.POS_DIFF_THRESHOLD = 0.1
        if self.pico_command["l"]["against"]:
            for obj_id in self.task_related_objects:
                obj_prim_path = f"/World/Objects/{obj_id}"
                rsp = self.env.robot.client.get_object_pose(obj_prim_path)
                obj_pos = np.array(
                    [
                        rsp.object_pose.position.x,
                        rsp.object_pose.position.y,
                        rsp.object_pose.position.z,
                    ]
                )
                obj_quat = np.array(
                    [
                        rsp.object_pose.rpy.rw,
                        rsp.object_pose.rpy.rx,
                        rsp.object_pose.rpy.ry,
                        rsp.object_pose.rpy.rz,
                    ]
                )
                self.task_related_objects[obj_id]["cur_pos"] = obj_pos
                self.task_related_objects[obj_id]["cur_quat"] = obj_quat

            for obj_id, val in self.task_related_objects.items():
                new_pos, new_quat = self.shuffle_pose(val)
                self.env.robot.set_object_pose(
                    prim_path=f"/World/Objects/{obj_id}",
                    target_pos=new_pos,
                    target_rot=new_quat,
                )
                val["cur_pos"] = new_pos
                val["cur_quat"] = new_quat
                time.sleep(0.2)

    def on_playback(self):
        if self.pico_command["l"]["against"]:
            print("Pub sig")
            self.sim_ros_node.pub_playback(True)
            self.current_mode = "playback"
        else:
            self.sim_ros_node.pub_playback(False)
            if self.current_mode == "playback":
                self.env.robot.initialize_solver(self.joint_info["l_arm"], False)
                self.env.robot.initialize_solver(self.joint_info["r_arm"], True)
                self.current_mode = "realtime"
                # base_x = self.joint_info["base_pos"]["base_linear_joint_x"]
                # base_y = self.joint_info["base_pos"]["base_linear_joint_y"]
                # base_yaw = self.joint_info["base_pos"]["base_angular_joint"]
                # self.env.robot.set_odometry(base_x, base_y, base_yaw)

    def run_pico_control(self, with_physics=True):
        now_t = self.sim_ros_node.get_clock().now().nanoseconds * 1e-9
        while now_t <= 3.0:
            print("wait server for %.3fs" % now_t)
            now_t = self.sim_ros_node.get_clock().now().nanoseconds * 1e-9
            time.sleep(0.1)
        self.initialize()
        self.sub_keyboard_event()
        coef_pos = 0.8
        coef_quat = 0.8
        self.env.do_eval_action()
        self.start_time = time.time()
        self.single_evaluate_ret["task_name"] = self.task_name
        self.single_evaluate_ret["robot_type"] = self.robot_cfg
        self.single_evaluate_ret["start_time"] = get_local_time()
        while rclpy.ok():
            pico_cmd = self.vr_server.on_update()
            if pico_cmd:
                logger.info(f"Control mode {self.control_mode}")
                self.parse_pico_command(pico_cmd)
                self.joint_cmd = {}
                if self.control_mode == "pico":
                    self.parse_arm_control(coef_pos, coef_quat)
                    self.parse_gripper_control()
                    self.parse_body_control()
                    self.apply_against_collection()
                    self.apply_base_control()
                else:
                    self.parse_arm_control_kbd()
                self.set_joint_state()
                self.on_playback()
                self.parse_joint_pose()
                self.run_eval()
                if self.env.has_done:
                    break
            else:
                logger.info("waiting for pico cmd")
            self.sim_ros_node.loop_rate.sleep()

    def parse_arm_control_kbd(self):
        jp = self.env.robot.get_joint_from_deltapos(
            xyz=self.command_,
            rpy=self.rotation_command,
            id="right",
            isOn=True,
        )
        for idx, val in enumerate(jp):
            self.joint_cmd[list(self.init_r_arm.keys())[idx]] = val

        self.joint_cmd.update({self.active_joint_name[1]: (0.0 if self.gripper_command == "close" else 1.0)})
        self.joint_cmd.update({self.active_joint_name[1]: (0.0 if self.gripper_command == "close" else 1.0)})

    def run_keyboard_control(self, with_physics=True):
        self.initialize()
        self.sub_keyboard_event()
        self.env.do_eval_action()
        while True:
            with self.lock:
                if self.reset_flg:
                    self.reset_command()
                    self.reset_robot_pose()
                    self.reset_flg = False
                if self.switch_flg:
                    idx = 0 if self.current_arm_type == "left" else 1
                    self.command_ = self.last_command[idx]["pos"]
                    self.rotation_command = self.last_command[idx]["rot"]
                    self.switch_flg = False
                if self.current_arm_type == "right":
                    jp = self.env.robot.get_joint_from_deltapos(
                        xyz=self.command_,
                        rpy=self.rotation_command,
                        id="right",
                        isOn=True,
                    )
                    for idx, val in enumerate(jp):
                        self.joint_cmd[list(self.init_r_arm.keys())[idx]] = val

                    self.joint_cmd.update(
                        {self.active_joint_name[1]: (0.0 if self.gripper_command == "close" else 1.0)}
                    )
                else:
                    jp = self.env.robot.get_joint_from_deltapos(
                        xyz=self.command_,
                        rpy=self.rotation_command,
                        id="left",
                        isOn=True,
                    )
                    for idx, val in enumerate(jp):
                        self.joint_cmd[list(self.init_l_arm.keys())[idx]] = val

                    self.joint_cmd.update(
                        {self.active_joint_name[0]: (0.0 if self.gripper_command == "close" else 1.0)}
                    )

                logger.info(
                    f"Base moving speed: {self.robot_command[0]:.2f}, yaw rate: {self.robot_rotation_command[2]:.2f}"
                )

                target_pos, target_rpy = self.env.robot.update_odometry(self.robot_command, self.robot_rotation_command)
                self.sim_ros_node.set_joint_state(name=[G1_CHASSIS[2]], position=[float(target_rpy[2])])
                time.sleep(0.1)
                self.sim_ros_node.set_joint_state(name=G1_CHASSIS[0:2], position=[float(v) for v in target_pos[0:2]])
                # self.init_pos[0:2] += self.waist_command
                # self.init_pos[2:4] += self.head_command
                self.set_joint_state()
                self.parse_joint_pose()
                self.run_eval()
                if self.env.has_done:
                    break

    def load_task_config(self, task):
        task_config_file = os.path.join(system_utils.teleop_root_path(), "tasks", task + ".json")
        logger.info(f"task config file {task_config_file}")
        if not os.path.exists(task_config_file):
            raise ValueError("Task config file not found: {}".format(task_config_file))
        with open(task_config_file) as f:
            self.task_config = json.load(f)
        robot_config_file = os.path.join(
            system_utils.app_root_path(),
            "robot_cfg/",
            self.robot_cfg,
        )
        if not os.path.exists(robot_config_file):
            raise ValueError("Robot config file not found: {}".format(robot_config_file))
        with open(robot_config_file, "r") as f:
            self.robot_cfg_content = json.load(f)

        self.robot_name = self.robot_cfg_content["robot"]["robot_name"]
        robot_init_pose_file = os.path.join(system_utils.benchmark_layout_path(), "robot_init_pose.json")
        with open(robot_init_pose_file, "r") as f:
            self.robot_init_pose = json.load(f)

    def spin_ros_node(self):
        while not self.stop_spin_thread_flag:
            rclpy.spin_once(self.sim_ros_node, timeout_sec=0.01)

    def start_ros_node(self):
        rclpy.init()
        self.sim_ros_node = SimNode(robot_name=self.robot_name, node_name="teleop_node")
        self.stop_spin_thread_flag = False
        self.spin_thread = threading.Thread(target=self.spin_ros_node, daemon=True)
        self.spin_thread.start()
        self.lock = threading.Lock()

    def stop_ros_node(self):
        if self.spin_thread.is_alive():
            self.stop_spin_thread_flag = True
            self.spin_thread.join(timeout=1.0)
        self.sim_ros_node.destroy_node()
        rclpy.shutdown()

    def get_joint_state(self):
        js = self.sim_ros_node.get_joint_state()
        return js.position, js.name

    def set_joint_state(self):
        names = []
        cmds = []
        for k, v in self.joint_cmd.items():
            names.append(k)
            cmds.append(v)
        self.sim_ros_node.set_joint_state(names, cmds)

    def config_camera(self):
        cam_prim_list = self.robot_cfg_content["camera"].keys()
        self.task_config["recording_setting"]["camera_list"] = cam_prim_list

    def collect_task_related_obj(self):
        self.task_related_objects = {}
        for obj in self.task_config["objects"]["task_related_objects"]:
            workspace_id = obj["workspace_id"]
            function_space_objects = self.task_config["scene"].get("function_space_objects", {})
            workspace_info = function_space_objects.get(workspace_id, None)
            if workspace_info is not None:
                center = workspace_info["position"]
                size = workspace_info["size"]
                self.task_related_objects[obj["object_id"]] = {
                    "cur_pos": None,
                    "cur_quat": None,
                    "center": center,
                    "size": size,
                }

    def run(self):
        self.load_task_config(self.task_name)
        self.task_config["robot"]["robot_cfg"] = self.robot_cfg
        self.start_ros_node()
        self.config_camera()
        self.collect_task_related_obj()

        # init robot and scene
        scene_info = self.task_config["scene"]
        workspace_id = scene_info["scene_id"].split("/")[-1]
        if workspace_id in scene_info["function_space_objects"]:
            robot_init_pose = self.task_config["robot"]["robot_init_pose"][workspace_id]
        else:
            robot_init_pose = self.task_config["robot"]["robot_init_pose"]
        self.task_config["specific_task_name"] = self.task_name

        logger.info(f"scene_usd {self.task_config['scene']['scene_usd']}, robot_cfg {self.robot_cfg}")
        # If self.task_config['scene']['scene_usd'] is a list, randomly select one using np.random.choice
        ader_instance = 0
        if isinstance(self.task_config["scene"]["scene_usd"], list):
            item = np.random.choice(self.task_config["scene"]["scene_usd"])
            if isinstance(item, dict):
                scene_usd, ader_instance = next(iter(item.items()))
            else:
                scene_usd = item
        else:
            scene_usd = self.task_config["scene"]["scene_usd"]
        logger.info(f"scene_usd {scene_usd}, robot_cfg {self.robot_cfg}")
        robot = IsaacSimRpcRobot(
            robot_cfg=self.robot_cfg,
            scene_usd=self.task_config["scene"]["scene_usd"],
            client_host=self.args.client_host,
            position=robot_init_pose["position"],
            rotation=robot_init_pose["quaternion"],
            gripper_control_type=1,
            ik_version="0.4.3",
        )

        # init state
        task_generator = TaskGenerator(self.task_config)
        task_folder = str(system_utils.teleop_root_path()) + "/saved_task/%s" % (self.task_config["task"])
        task_generator.generate_tasks(
            save_path=task_folder,
            task_num=self.episodes_per_instance,
            task_name=self.task_config["task"],
        )
        robot_position = task_generator.robot_init_pose["position"]
        robot_rotation = task_generator.robot_init_pose["quaternion"]
        self.task_config["robot"]["robot_init_pose"]["position"] = robot_position
        self.task_config["robot"]["robot_init_pose"]["quaternion"] = robot_rotation
        specific_task_files = glob.glob(task_folder + "/*.json")
        self.eval_result = []
        for episode_id in range(self.episodes_per_instance):
            episode_file_path = specific_task_files[episode_id]
            env = DummyEnv(robot, episode_file_path, self.task_config, ader_instance=ader_instance)
            self.env = env
            recording_objects_prim = self.task_config.get("recording_setting").get("objects_prim")
            robot_id = self.robot_cfg.split(".")[0]
            assert robot_id in self.robot_init_pose, f"Robot_cfg {self.robot_cfg} not defined in robot_init_pose.json"
            assert (
                self.task_name in self.robot_init_pose[robot_id]
            ), f"Task_name {self.task_name} with robot {self.robot_cfg} not defined in robot_init_pose.json"

            init_pose = self.robot_init_pose[robot_id][self.task_name]["init_arm_pose"]
            print("SET init pose", init_pose)
            robot.set_init_pose(init_pose)
            env.reset()

            if self.record:
                self.env.start_recording(
                    task_name=self.task_name,
                    camera_prim_list=[],
                    fps=self.fps,
                    extra_objects_prim=recording_objects_prim,
                )
            if self.mode == "pico":
                self.run_pico_control()
            elif self.mode == "keyboard":
                self.run_keyboard_control()
            else:
                raise ValueError("Invalid mode: {}".format(self.mode))

    def post_process(self):
        self.end_time = time.time()
        self.single_evaluate_ret["end_time"] = get_local_time()
        self.single_evaluate_ret["duration"] = str(self.end_time - self.start_time)
        self.single_evaluate_ret["operator"] = self.operator
        summarize_scores(self.single_evaluate_ret, self.task_name)
        self.eval_result.append(self.single_evaluate_ret)
        logger.info("Episode done...")
        if self.record:
            self.env.stop_recording(True)
        dump_eval_result(self.eval_out_dir, self.eval_result)
        logger.info("Episode done...")
        self.env.robot.client.Exit()
        self.stop_ros_node()


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--client_host", type=str, default="localhost:50051", help="The client")
    parser.add_argument("--fps", type=int, default=30, help="Set fps of the recording")
    parser.add_argument("--task_name", default="iros_stamp_the_seal", type=str, help="Selected task to run")
    parser.add_argument("--mode", type=str, default="pico", help="Choose teleop mode: pico or keyboard")
    parser.add_argument("--record", action="store_true", help="Enable data recording")
    parser.add_argument("--host_ip", type=str, default="localhost", help="Set vr host ip")
    parser.add_argument("--port", type=int, default=8080, help="Set vr port")
    parser.add_argument("--operator", type=str, default="jonasjin", help="teleop operator")
    parser.add_argument("--robot_cfg", type=str, default="G1_omnipicker.json", help="Set robot config json")
    # fmt: on
    args = parser.parse_args()
    task = TeleOp(args)
    try:
        task.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, teleop stop")
    finally:
        try:
            task.post_process()
        except Exception as e:
            logger.error("Error in post process: {}".format(e))


if __name__ == "__main__":
    main()
