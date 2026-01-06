# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import os
import time

import numpy as np

from client.robot.omni_robot import IsaacSimRpcRobot
from common.base_utils.logger import logger


class BaseAgent(object):
    def __init__(self, robot: IsaacSimRpcRobot) -> None:
        self.robot = robot
        self.reset()

    def reset(self):
        self.robot.reset()

    def add_object(self, object_info: dict):
        name = object_info["object_id"]
        usd_path = os.path.join(object_info["data_info_dir"], "Aligned.usd")
        position = np.array(object_info["position"])
        quaternion = np.array(object_info["quaternion"])
        if "scale" not in object_info:
            object_info["scale"] = 1.0
            size = object_info.get("size", np.array([]))
            if isinstance(size, list):
                size = np.array(size)
            if size.shape[0] == 0:
                size = np.array(100)

            if np.max(size) > 10:
                object_info["scale"] = 0.001

        add_particle = False
        particle_position = [0, 0, 0]
        particle_scale = [1, 1, 1]
        if "add_particle" in object_info:
            add_particle = True
            particle_position = object_info["add_particle"]["position"]
            particle_scale = object_info["add_particle"]["scale"]
        scale = np.array([object_info["scale"]] * 3)
        material = "general" if "material" not in object_info else object_info["material"]
        mass = 0.01 if "mass" not in object_info else object_info["mass"]
        static_friction = object_info.get("static_friction", 0.5)
        dynamic_friction = object_info.get("dynamic_friction", 0.5)
        model_type = object_info.get("model_type", "convexDecomposition")
        add_rigid_body = object_info.get("add_rigid_body", True)
        object_info.get("pose_part", "")
        prim_path = object_info.get("prim_path", "/World/Objects/%s" % name)
        self.robot.client.add_object(
            usd_path=usd_path,
            prim_path=prim_path,
            label_name=name,
            target_position=position,
            target_quaternion=quaternion,
            target_scale=scale,
            material=material,
            color=[1, 1, 1],
            mass=mass,
            add_particle=add_particle,
            particle_position=particle_position,
            particle_scale=particle_scale,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            model_type=model_type,
            add_rigid_body=add_rigid_body,
        )

    def generate_layout(self, task_file, init=True):
        with open(task_file, "r") as f:
            task_info = json.load(f)
        if init:
            for object_info in task_info["objects"]:
                self.add_object(object_info)
            time.sleep(1)

        self.task_info = task_info
        self.robot.target_object = task_info["target_object"]
        self.arm = task_info["arm"]
        return task_info

    def execute(self, commands):
        for command in commands:
            type = command["action"]
            content = command["content"]
            if type == "move" or type == "rotate":
                if not self.robot.move(content):
                    return False
                else:
                    self.last_pose = content["matrix"]
            elif type == "open_gripper":
                id = "left" if "gripper" not in content else content["gripper"]
                width = 0.1 if "width" not in content else content["width"]
                self.robot.open_gripper(id=id, width=width)

            elif type == "close_gripper":
                id = "left" if "gripper" not in content else content["gripper"]
                force = 50 if "force" not in content else content["force"]
                self.robot.close_gripper(id=id, force=force)
                if "attach_obj" in content:
                    self.robot.client.attach_obj(prim_paths=[content["attach_obj"]])

            else:
                Exception("Not implemented.")
        return True

    def start_recording(self, task_name, camera_prim_list):
        logger.info(camera_prim_list)
        self.robot.client.start_recording(
            task_name=task_name,
            fps=30,
            data_keys={
                "camera": {
                    "camera_prim_list": camera_prim_list,
                    "render_depth": False,
                    "render_semantic": False,
                },
                "joint_position": True,
                "gripper": True,
            },
        )

    def stop_recording(self, success):
        self.robot.client.stop_recording()
        self.robot.client.send_task_status(success)
