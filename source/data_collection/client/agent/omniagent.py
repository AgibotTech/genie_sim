# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import glob
import json
import os
import pickle
import re
import time

import numpy as np

from client.agent.base import BaseAgent
from client.layout.object import OmniObject
from client.planner.action.action_script import ActionScript
from client.planner.action.stage import PLACE_LIKE_ACTIONS, Action, Stage
from client.robot import Robot
from common.base_utils.logger import logger
from common.base_utils.transform_utils import (
    add_random_noise_to_pose,
    calculate_rotation_matrix,
    pose_difference,
    pose_from_position_quaternion,
    quaternion_rotate,
)

MAX_ATTEMPTIONS = 4


def load_task_solution(task_info):
    stages = task_info["stages"]

    stage_related_objs = []
    processed_stages = []
    for i, stage in enumerate(stages):
        stage_related_objs.append(stage.get("active", {}).get("object_id", None))
        stage_related_objs.append(stage.get("passive", {}).get("object_id", None))
        processed_stages.append(copy.deepcopy(stage))
    stage_related_objs = list(set(stage_related_objs))
    task_info["stages"] = processed_stages

    objects = {"gripper": OmniObject("gripper")}

    for obj_info in task_info["objects"]:
        obj_id = obj_info["object_id"]
        if "fix_pose" in obj_id:
            obj = OmniObject(obj_id)
            if "position" not in obj_info or "direction" not in obj_info:
                logger.error(f"Error: Missing position/direction in object {obj_id}")
                continue
            obj.set_pose(np.array(obj_info["position"]), np.array([0.001, 0.001, 0.001]))
            obj.elements = {
                "active": {},
                "passive": {
                    "place": {
                        "default": [
                            {
                                "xyz": np.array([0, 0, 0]),
                                "direction": np.array(obj_info["direction"]),
                            }
                        ]
                    }
                },
            }
            objects[obj_id] = obj
        else:
            obj_dir = obj_info["data_info_dir"]
            obj = OmniObject.from_obj_dir(
                obj_dir,
                obj_info=obj_info,
                object_id=obj_id,
                is_key_object=obj_id in stage_related_objs,
            )
            objects[obj_id] = obj

            if hasattr(obj, "part_ids"):
                if hasattr(obj, "part_joint_limits") and obj.part_joint_limits is not None:
                    obj_parts_joint_limits = obj.part_joint_limits
                for part_id in obj.part_ids:
                    id = obj_id + "/%s" % part_id
                    objects[id] = copy.deepcopy(obj)
                    objects[id].name = id
                    objects[id].prim_path += f"/{part_id}"
                    if part_id in obj_parts_joint_limits:
                        objects[id].part_joint_limit = obj_parts_joint_limits[part_id]
                if len(obj.part_ids):
                    del objects[obj_id]
    return objects


class DataCollectionAgent(BaseAgent):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.action_script = ActionScript()
        self.attached_obj_id = None

    def start_recording(
        self,
        task_name,
        camera_prim_list,
        fps,
        render_semantic=False,
        recording_setting={},
    ):
        self.robot.client.start_recording(
            task_name=task_name,
            fps=fps,
            data_keys={
                "camera": {
                    "camera_prim_list": camera_prim_list,
                    "render_depth": False,
                    "render_semantic": render_semantic,
                },
                "joint_position": True,
                "gripper": True,
                "additional_parameters": json.dumps(recording_setting),
            },
        )

    def generate_layout(self, task_file):
        self.task_file = task_file
        with open(task_file, "r") as f:
            task_info = json.load(f)

        self.articulated_objs = []
        set_poses = []
        for object_info in task_info["objects"]:
            if "fix_pose" in object_info["object_id"]:
                continue
            is_articulated = object_info.get("is_articulated", False)
            if is_articulated:
                self.articulated_objs.append(object_info["object_id"])
            object_info["material"] = "general"
            if object_info.get("scene_object", False):
                if "position" not in object_info or "quaternion" not in object_info:
                    logger.error(
                        "Scene object %s missing position or quaternion, skipping."
                        % object_info["object_id"]
                    )
                    continue
                object_pose = {}
                name = object_info["object_id"]
                object_pose["prim_path"] = object_info.get("prim_path", "/World/Objects/%s" % name)
                object_pose["position"] = np.array(object_info["position"])
                object_pose["rotation"] = np.array(object_info["quaternion"])
                set_poses.append(object_pose)
            else:
                self.add_object(object_info)
                time.sleep(0.1)
                self.add_object(object_info)
                time.sleep(0.2)
        if len(set_poses):
            self.robot.client.set_object_pose(set_poses, [])
        time.sleep(2)

        self.arm = task_info["arm"]

        task_related_objs = []
        for stage in task_info["stages"]:
            for type in ["active", "passive"]:
                obj_id = stage[type]["object_id"]
                if obj_id == "gripper" or obj_id in task_related_objs:
                    continue
                task_related_objs.append(obj_id)

        target_lookat_point = []
        for obj in task_info["objects"]:
            if obj["object_id"] not in task_related_objs or "position" not in obj:
                continue
            target_lookat_point.append(obj["position"])
        if len(target_lookat_point):
            target_lookat_point = np.mean(np.stack(target_lookat_point), axis=0)
            self.robot.client.set_target_point(target_lookat_point.tolist())

        """ Set material """
        material_infos = []
        if "object_with_material" in task_info:
            for key in task_info["object_with_material"]:
                material_infos += task_info["object_with_material"][key]
            if len(material_infos):
                self.robot.client.set_material(material_infos)
                time.sleep(0.3)

        """ Set light """
        light_infos = []
        if "lights" in task_info:
            for key in task_info["lights"]:
                light_infos += task_info["lights"][key]
            if len(light_infos):
                original_light_infos = copy.deepcopy(light_infos)
                for idx, info in enumerate(light_infos):
                    if "light_temperature" in info:
                        temperature_random_ratio = min(max(0.4, np.random.normal(1, 0.2)), 1.6)
                        info["light_temperature"] = (
                            original_light_infos[idx]["light_temperature"]
                            * temperature_random_ratio
                        )
                    if "light_intensity" in info:
                        intensity_random_ratio = max(0.1, np.random.normal(1, 0.3))
                        info["light_intensity"] = (
                            original_light_infos[idx]["light_intensity"] * intensity_random_ratio
                        )
                    if "rotation" in info:
                        info["rotation"] = original_light_infos[idx]["rotation"]
                    else:
                        info["rotation"] = quaternion_rotate(
                            np.array([1, 0.0, 0.0, 0.0]),
                            "z",
                            np.random.choice(np.arange(0, 180, 15)),
                        ).tolist()
                self.robot.client.set_light(light_infos)
                time.sleep(2)

        """ Set camera"""
        if "cameras" in task_info:
            for cam_id in task_info["cameras"]:
                cam_info = task_info["cameras"][cam_id]
                self.robot.client.add_camera(
                    cam_id,
                    cam_info["position"],
                    cam_info["quaternion"],
                    cam_info["width"],
                    cam_info["height"],
                    cam_info["focal_length"],
                    cam_info["horizontal_aperture"],
                    cam_info["vertical_aperture"],
                    cam_info["is_local"],
                )

    def update_objects(self, objects, arm="right"):
        # update gripper pose
        objects["gripper"].obj_pose = self.robot.get_ee_pose(ee_type="gripper", id=arm)

        # update object pose
        for obj_id in objects:
            if obj_id == "gripper":
                continue
            if "fix_pose" in obj_id:
                if len(objects[obj_id].obj_pose) == 3:
                    position = objects[obj_id].obj_pose
                    rotation_matrix = calculate_rotation_matrix(
                        objects[obj_id].direction, [0, 0, 1]
                    )
                    objects[obj_id].obj_pose = np.eye(4)
                    objects[obj_id].obj_pose[:3, 3] = position.flatten()
                    objects[obj_id].obj_pose[:3, :3] = rotation_matrix
                continue
            if "/" in obj_id:
                obj_id.split("/")[0]
                part_name = obj_id.split("/")[1]
                dof_joint = self.robot.client.get_part_dof_joint(
                    objects[obj_id].parent_prim_path, part_name
                )
                objects[obj_id].joint_position = dof_joint.joint_position
                objects[obj_id].joint_velocity = dof_joint.joint_velocity

            objects[obj_id].obj_pose = self.robot.get_prim_world_pose(objects[obj_id].pose_prim)
            if (
                hasattr(objects[obj_id], "info")
                and "simple_place" in objects[obj_id].info
                and objects[obj_id].info["simple_place"]
            ):
                down_direction_world = (
                    np.linalg.inv(objects[obj_id].obj_pose) @ np.array([0, 0, -1, 1])
                )[:3]
                down_direction_world = (
                    down_direction_world / np.linalg.norm(down_direction_world) * 0.08
                )
                objects[obj_id].elements["active"]["place"]["direction"] = down_direction_world

        return objects

    def check_task_file(self, task_file):
        with open(task_file, "r") as f:
            task_info = json.load(f)

        objs_dir = {}
        objs_interaction = {}
        for obj_info in task_info["objects"]:
            obj_id = obj_info["object_id"]
            if "fix_pose" in obj_id:
                continue
            objs_dir[obj_id] = obj_info["data_info_dir"]
            if "interaction" in obj_info:
                objs_interaction[obj_id] = obj_info["interaction"]
            else:
                object_dir_base_name = os.path.basename(os.path.normpath(objs_dir[obj_id]))
                interaction_label_file = os.path.join(
                    os.environ.get("SIM_ASSETS"),
                    "interaction",
                    object_dir_base_name,
                    "interaction.json",
                )
                assert os.path.exists(
                    interaction_label_file
                ), f"interaction.json not found in {interaction_label_file}"
                objs_interaction[obj_id] = json.load(open(interaction_label_file))["interaction"]

        for stage in task_info["stages"]:
            active_obj_id = stage["active"]["object_id"]
            passive_obj_id = stage["passive"]["object_id"]

            if active_obj_id != "gripper":
                if isinstance(active_obj_id, list):
                    for id in active_obj_id:
                        if id not in objs_dir:
                            logger.info("Active obj not in objs_dir: %s" % id)
                            return False
                elif isinstance(active_obj_id, dict):  # If it's a dictionary
                    for id in active_obj_id.values():  # Iterate over all values in the dictionary
                        if id not in objs_dir and "fix_pose" not in id:
                            logger.info("Active obj not in objs_dir: %s" % id)
                            return False
                else:
                    if active_obj_id not in objs_dir:
                        logger.info("Active obj not in objs_dir: %s" % active_obj_id)
                        return False

            if passive_obj_id != "gripper" and "fix_pose" not in passive_obj_id:
                if isinstance(passive_obj_id, list):
                    for id in passive_obj_id:
                        if id not in objs_dir:
                            logger.info("Passive obj not in objs_dir: %s" % id)
                            return False
                elif isinstance(passive_obj_id, dict):  # If it's a dictionary
                    for id in passive_obj_id.values():  # Iterate over all values in the dictionary
                        if id not in objs_dir and "fix_pose" not in id:
                            logger.info("Passive obj not in objs_dir: %s" % id)
                            return False
                else:
                    if passive_obj_id not in objs_dir:
                        logger.info("Passive obj not in objs_dir: %s" % passive_obj_id)
                        return False

            if stage["action"] in ["grasp", "pick"]:
                passive_obj_id = stage["passive"]["object_id"]
                if isinstance(passive_obj_id, list):
                    for id in passive_obj_id:
                        obj_dir = objs_dir[id]
                        object_dir_base_name = os.path.basename(os.path.normpath(obj_dir))
                        primitive = stage["passive"]["primitive"]
                        if primitive is None:
                            file = "grasp_pose/grasp_pose.pkl"
                        else:
                            file = objs_interaction[id]["passive"]["grasp"][primitive]
                            if isinstance(file, list):
                                file = file[0]
                        grasp_file = os.path.join(
                            os.environ.get("SIM_ASSETS"), "interaction", object_dir_base_name, file
                        )
                        if not os.path.exists(grasp_file):
                            logger.info("-- Grasp file not exist: %s" % grasp_file)
                            return False

                        _data = pickle.load(open(grasp_file, "rb"))
                        if len(_data["grasp_pose"]) == 0:
                            logger.info("-- Grasp file empty: %s" % grasp_file)
                            return False
                else:
                    obj_dir = objs_dir[passive_obj_id]
                    object_dir_base_name = os.path.basename(os.path.normpath(obj_dir))
                    primitive = stage["passive"]["primitive"]
                    if primitive is None:
                        file = "grasp_pose/grasp_pose.pkl"
                    else:
                        file = objs_interaction[passive_obj_id]["passive"]["grasp"][primitive]
                        if isinstance(file, list):
                            file = file[0]
                    grasp_file = os.path.join(
                        os.environ.get("SIM_ASSETS"), "interaction", object_dir_base_name, file
                    )
                    if not os.path.exists(grasp_file):
                        logger.info("-- Grasp file not exist: %s" % grasp_file)
                        return False

                    _data = pickle.load(open(grasp_file, "rb"))
                    if len(_data["grasp_pose"]) == 0:
                        logger.info("-- Grasp file empty: %s" % grasp_file)
                        return False
        return True

    # Load task after self.check_task_file succeeds
    def load_task(
        self,
        task_file,
        use_recording,
        camera_list,
        fps,
        render_semantic,
        workspaces,
        origin_task_info,
    ):
        logger.info(f"Start Task{task_file}")
        self.reset()
        self.attached_obj_id = None

        self.generate_layout(task_file)
        self.robot.open_gripper(id="right", detach=False)
        self.robot.open_gripper(id="left", detach=False)
        self.robot.client.detach_obj()
        time.sleep(2.0)

        self.robot.reset_pose = {
            "right": self.robot.get_ee_pose(ee_type="gripper", id="right"),
            "left": self.robot.get_ee_pose(ee_type="gripper", id="left"),
        }
        logger.info(f"Reset pose{self.robot.reset_pose}")

        task_info = json.load(open(task_file, "rb"))

        objects = load_task_solution(task_info)
        objects = self.update_objects(objects)

        # Modify arm parameters for pick and place stages
        for stage in task_info["stages"]:
            # Check if active_object_id and passive_object_id are lists
            if isinstance(stage["active"]["object_id"], list):
                rule = stage["active"].get("rule", "random_choice")
                if rule == "random_choice":
                    active_obj_id = np.random.choice(stage["active"]["object_id"])
                    stage["active"]["object_id"] = active_obj_id
                    logger.info(
                        f"Stage {stage['action']} randomly selected active object id: {active_obj_id}"
                    )
                else:
                    raise NotImplementedError(
                        f"Rule {rule} for active object selection is not implemented"
                    )
            if isinstance(stage["passive"]["object_id"], list):
                rule = stage["passive"].get("rule", "random_choice")
                if rule == "random_choice":
                    passive_obj_id = np.random.choice(stage["passive"]["object_id"])
                    stage["passive"]["object_id"] = passive_obj_id
                    logger.info(
                        f"Stage {stage['action']} randomly selected passive object id: {passive_obj_id}"
                    )
                else:
                    raise NotImplementedError(
                        f"Rule {rule} for passive object selection is not implemented"
                    )

            if stage["action"] == "pick" and stage["extra_params"]["arm"] == "auto":
                passive_obj_id = stage["passive"]["object_id"]

                # Find passive_obj
                passive_obj = None
                for obj in task_info["objects"]:
                    if obj["object_id"] == passive_obj_id:
                        passive_obj = obj
                        break
                object_position = passive_obj.get("position")
                object_quat = passive_obj.get("quaternion")
                object_pose = pose_from_position_quaternion(object_position, object_quat)
                left_gripper_pose = self.robot.get_ee_pose(ee_type="gripper", id="left")
                right_gripper_pose = self.robot.get_ee_pose(ee_type="gripper", id="right")

                # Calculate distance difference
                DIFF_THRESHOLD = 0.1
                left_diff, _ = pose_difference(left_gripper_pose, object_pose)
                right_diff, _ = pose_difference(right_gripper_pose, object_pose)
                if abs(left_diff - right_diff) <= DIFF_THRESHOLD:
                    # Randomly select left or right
                    arm = np.random.choice(["left", "right"])
                elif left_diff <= right_diff:
                    arm = "left"
                else:
                    arm = "right"
                logger.info("Auto select arm: %s" % arm)
                # Update arm parameter in stage
                stage["extra_params"]["arm"] = arm

            # Also update arm parameter in place stage to maintain consistency
            if (
                stage["action"] in PLACE_LIKE_ACTIONS or stage["action"] in ["rotate", "reset"]
            ) and stage["extra_params"]["arm"] == "auto":
                stage["extra_params"]["arm"] = arm

            # Only select based on arm when passive_obj_id is a dictionary
            if isinstance(stage["passive"]["object_id"], dict):
                stage["passive"]["object_id"] = stage["passive"]["object_id"][arm]

            if "checker" in stage:
                for checker in stage["checker"]:
                    if "params" not in checker:
                        continue
                    for key, value in checker["params"].items():
                        if isinstance(value, str):
                            if value == "gripper":
                                checker["params"][key] = stage["extra_params"]["arm"]
                            elif value in objects:
                                checker["params"][key] = objects[value].prim_path
        # Modify action_text and english_action_text in checker
        for stage in task_info["stages"]:
            if "action_description" in stage:
                if stage["extra_params"]["arm"] == "left":
                    chinese_arm = "左"
                    english_arm_upper = "Left"
                    english_arm_lower = "left"
                elif stage["extra_params"]["arm"] == "right":
                    chinese_arm = "右"
                    english_arm_upper = "Right"
                    english_arm_lower = "right"
                if "action_text" in stage["action_description"]:
                    stage["action_description"]["action_text"] = stage["action_description"][
                        "action_text"
                    ].replace("{左/右}", chinese_arm)
                if "english_action_text" in stage["action_description"]:
                    stage["action_description"]["english_action_text"] = stage[
                        "action_description"
                    ]["english_action_text"].replace("{Left/Right}", english_arm_upper)
                    stage["action_description"]["english_action_text"] = stage[
                        "action_description"
                    ]["english_action_text"].replace("{left/right}", english_arm_lower)
                if "action_text" in stage["action_description"]:
                    match = re.search(r"\{object:.*\}", stage["action_description"]["action_text"])
                    if match:
                        object_name = match.group(0).replace("{object:", "").replace("}", "")
                        for obj in task_info["objects"]:
                            if obj["object_id"] == object_name:
                                if "chinese_semantic_name" in obj:
                                    stage["action_description"]["action_text"] = stage[
                                        "action_description"
                                    ]["action_text"].replace(
                                        match.group(0), obj["chinese_semantic_name"]
                                    )
                                else:
                                    raise ValueError(
                                        f"Object {object_name} has no chinese_semantic_name"
                                    )
                                break
                    match = re.search(
                        r"\{position:.*\}", stage["action_description"]["action_text"]
                    )
                    if match:
                        object_name = match.group(0).replace("{position:", "").replace("}", "")
                        for obj in task_info["objects"]:
                            if obj["object_id"] == object_name:
                                if "chinese_position_semantic" in obj:
                                    stage["action_description"]["action_text"] = stage[
                                        "action_description"
                                    ]["action_text"].replace(
                                        match.group(0), obj["chinese_position_semantic"]
                                    )
                                else:
                                    raise ValueError(
                                        f"Object {object_name} has no chinese_position_semantic"
                                    )
                                break
                if "english_action_text" in stage["action_description"]:
                    match = re.search(
                        r"\{object:.*\}",
                        stage["action_description"]["english_action_text"],
                    )
                    if match:
                        object_name = match.group(0).replace("{object:", "").replace("}", "")
                        for obj in task_info["objects"]:
                            if obj["object_id"] == object_name:
                                if "english_semantic_name" in obj:
                                    stage["action_description"]["english_action_text"] = stage[
                                        "action_description"
                                    ]["english_action_text"].replace(
                                        match.group(0), obj["english_semantic_name"]
                                    )
                                else:
                                    raise ValueError(
                                        f"Object {object_name} has no english_semantic_name"
                                    )
                                break
                    match = re.search(
                        r"\{position:.*\}", stage["action_description"]["english_action_text"]
                    )
                    if match:
                        object_name = match.group(0).replace("{position:", "").replace("}", "")
                        for obj in task_info["objects"]:
                            if obj["object_id"] == object_name:
                                if "english_position_semantic" in obj:
                                    stage["action_description"]["english_action_text"] = stage[
                                        "action_description"
                                    ]["english_action_text"].replace(
                                        match.group(0), obj["english_position_semantic"]
                                    )
                                else:
                                    raise ValueError(
                                        f"Object {object_name} has no english_position_semantic"
                                    )
                                break
        # Modify english_task_name, task_name, init_scene_text in task_description
        if "task_description" in task_info:
            task_description = task_info["task_description"]
            if "english_task_name" in task_description:
                match = re.search(r"\{object:.*\}", task_description["english_task_name"])
                if match:
                    object_name = match.group(0).replace("{object:", "").replace("}", "")
                    for obj in task_info["objects"]:
                        if obj["object_id"] == object_name:
                            if "english_semantic_name" in obj:
                                task_description["english_task_name"] = task_description[
                                    "english_task_name"
                                ].replace(match.group(0), obj["english_semantic_name"])
                            else:
                                raise ValueError(
                                    f"Object {object_name} has no english_semantic_name"
                                )
                            break
                match = re.search(r"\{position:.*\}", task_description["english_task_name"])
                if match:
                    object_name = match.group(0).replace("{position:", "").replace("}", "")
                    for obj in task_info["objects"]:
                        if obj["object_id"] == object_name:
                            if "english_position_semantic" in obj:
                                task_description["english_task_name"] = task_description[
                                    "english_task_name"
                                ].replace(match.group(0), obj["english_position_semantic"])
                            else:
                                raise ValueError(
                                    f"Object {object_name} has no english_position_semantic"
                                )
                            break
            if "task_name" in task_description:
                match = re.search(r"\{object:.*\}", task_description["task_name"])
                if match:
                    object_name = match.group(0).replace("{object:", "").replace("}", "")
                    for obj in task_info["objects"]:
                        if obj["object_id"] == object_name:
                            if "chinese_semantic_name" in obj:
                                task_description["task_name"] = task_description[
                                    "task_name"
                                ].replace(match.group(0), obj["chinese_semantic_name"])
                            else:
                                raise ValueError(
                                    f"Object {object_name} has no chinese_semantic_name"
                                )
                            break
                match = re.search(r"\{position:.*\}", task_description["task_name"])
                if match:
                    object_name = match.group(0).replace("{position:", "").replace("}", "")
                    for obj in task_info["objects"]:
                        if obj["object_id"] == object_name:
                            if "chinese_position_semantic" in obj:
                                task_description["task_name"] = task_description[
                                    "task_name"
                                ].replace(match.group(0), obj["chinese_position_semantic"])
                            else:
                                raise ValueError(
                                    f"Object {object_name} has no chinese_position_semantic"
                                )
                            break
            if "init_scene_text" in task_description:
                match = re.search(r"\{object:.*\}", task_description["init_scene_text"])
                if match:
                    object_name = match.group(0).replace("{object:", "").replace("}", "")
                    for obj in task_info["objects"]:
                        if obj["object_id"] == object_name:
                            if "english_semantic_name" in obj:
                                task_description["init_scene_text"] = task_description[
                                    "init_scene_text"
                                ].replace(match.group(0), obj["english_semantic_name"])
                            else:
                                raise ValueError(
                                    f"Object {object_name} has no english_semantic_name"
                                )
                            break
                match = re.search(r"\{position:.*\}", task_description["init_scene_text"])
                if match:
                    object_name = match.group(0).replace("{position:", "").replace("}", "")
                    for obj in task_info["objects"]:
                        if obj["object_id"] == object_name:
                            if "english_position_semantic" in obj:
                                task_description["init_scene_text"] = task_description[
                                    "init_scene_text"
                                ].replace(match.group(0), obj["english_position_semantic"])
                            else:
                                raise ValueError(
                                    f"Object {object_name} has no english_position_semantic"
                                )
                            break

        self.action_script.initialize(task_info, objects)
        if use_recording:
            recording_setting = origin_task_info.get("recording_setting", {})
            self.start_recording(
                task_name="[%s]" % (os.path.basename(os.path.normpath(task_file)).split(".")[0]),
                camera_prim_list=camera_list,
                fps=fps,
                render_semantic=render_semantic,
                recording_setting=recording_setting,
            )

        return task_info, objects

    def step(
        self,
        action: Action,
        stage: Stage,
        objects,
        workspaces,
        stage_id,
        index,
        step_index,
    ):
        logger.info(
            f"==> [RUN] Stage {stage_id} Substage {index} Step {step_index}: {stage.action_type}"
        )
        step_success = False
        need_retry = True
        extra_params = stage.extra_params.copy()
        arm = extra_params.get("arm", "right")
        action_type = stage.action_type
        active_id = stage.active_obj_id
        passive_id = stage.passive_obj_id

        objects = self.update_objects(objects, arm=arm)

        (
            target_gripper_pose,
            motion_type,
            gripper_action,
            arm,
            action_description,
            action_extra_params,
        ) = stage.parse_action(action, objects)
        extra_params.update(action_extra_params)
        arm = extra_params.get("arm", "right")
        motion_run_ratio = 1.0
        remove_obstacles = False
        gripper_action_timing = {}
        error_data = {}
        if "error_data" in extra_params:
            error_data = extra_params["error_data"]
            error_params = error_data.get("params", {})
            motion_run_ratio = error_params.get("motion_run_ratio", 1.0)
            if error_data.get("type", None) == "Drop":
                timing = error_params.get("drop_timing", 0.2)
                gripper_action_timing = {
                    "state": "open",
                    "timing": timing,
                    "is_right": arm == "right",
                }
                gripper_action = "open"
            elif error_data.get("type", None) == "KeepClose":
                if action_type == "pick":
                    remove_obstacles = True

        self.robot.client.set_frame_state(
            action_type,
            step_index,
            active_id,
            passive_id,
            self.attached_obj_id is not None,
            arm=arm,
            target_pose=target_gripper_pose,
            action_description=action_description,
            error_description=error_data,
        )

        # execution action
        goal_offset = extra_params.get("goal_offset", [0, 0, 0, 1, 0, 0, 0])
        path_constraint = extra_params.get("path_constraint", [])
        from_current_pose = extra_params.get("from_current_pose", False)
        offset_and_constraint_in_goal_frame = extra_params.get(
            "offset_and_constraint_in_goal_frame", True
        )
        disable_collision_links = extra_params.get("disable_collision_links", [])
        if remove_obstacles:
            self.robot.client.remove_objs_from_obstacle([objects[stage.passive_obj_id].prim_path])

        # check if target_gripper_pose is reachable

        if target_gripper_pose is None:
            state = True
        else:
            if motion_type == "Simple":
                ik_success, _ = self.robot.solve_ik(
                    np.array([target_gripper_pose]),
                    ee_type="gripper",
                    arm=arm,
                    type="Simple",
                )
                if not ik_success[0]:
                    logger.info(
                        f"Substage {index} step_id {step_index}: target gripper pose not reachable, try add noise"
                    )
                    noisy_target_gripper_poses = []
                    for _ in range(10):
                        noisy_target_gripper_poses.append(
                            add_random_noise_to_pose(target_gripper_pose, rot_noise=10)
                        )
                    noisy_target_gripper_poses = np.array(noisy_target_gripper_poses)
                    ik_success, _ = self.robot.solve_ik(
                        noisy_target_gripper_poses,
                        ee_type="gripper",
                        arm=arm,
                        type="Simple",
                    )
                    noisy_target_gripper_poses = noisy_target_gripper_poses[ik_success]
                    if len(noisy_target_gripper_poses) > 0:
                        dists = np.linalg.norm(
                            noisy_target_gripper_poses[:, :3, 3] - target_gripper_pose[:3, 3],
                            axis=1,
                        )
                        min_index = np.argmin(dists)
                        target_gripper_pose = noisy_target_gripper_poses[min_index]
                        logger.info(
                            f"Substage {index} step_id {step_index}: find reachable gripper pose"
                        )
            state = self.robot.move_pose(
                target_gripper_pose,
                motion_type,
                arm=arm,
                block=True,
                goal_offset=goal_offset,
                path_constraint=path_constraint,
                offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
                disable_collision_links=disable_collision_links,
                motion_run_ratio=motion_run_ratio,
                gripper_action_timing=gripper_action_timing,
                from_current_pose=from_current_pose,
            )
        if not state:
            step_success = False
            need_retry = True
            return step_success, need_retry
        else:
            set_gripper_open = gripper_action == "open"
            set_gripper_close = gripper_action == "close"
            self.robot.client.set_frame_state(
                action_type,
                step_index,
                active_id,
                passive_id,
                self.attached_obj_id is not None,
                set_gripper_open,
                set_gripper_close,
                arm=arm,
                target_pose=target_gripper_pose,
                action_description=action_description,
                error_description=error_data,
            )
            self.robot.set_gripper_action(gripper_action, arm=arm)
            if set_gripper_open or set_gripper_close:
                time.sleep(1.0)

            self.robot.client.set_frame_state(
                action_type,
                step_index,
                active_id,
                passive_id,
                self.attached_obj_id is not None,
                arm=arm,
                target_pose=target_gripper_pose,
                action_description=action_description,
                error_description=error_data,
            )

            # check sub-stage completion
            objects["gripper"].obj_pose = self.robot.get_ee_pose(ee_type="gripper", id=arm)
            objects = self.update_objects(objects, arm=arm)

            step_success = stage.check_completion(objects, self.robot)
            self.robot.client.set_frame_state(
                action_type,
                step_index,
                active_id,
                passive_id,
                self.attached_obj_id is not None,
                arm=arm,
                target_pose=target_gripper_pose,
                action_description=action_description,
                error_description=error_data,
            )
            if not step_success:
                self.attached_obj_id = None
            else:
                logger.info(
                    f"action_stages: {stage_id}, antion: {action_type}, [stage {index} step_id {step_index}] success"
                )
                if not len(stage.active_action_sequence.actions) == step_index + 1:
                    logger.info("jump to next action")
                else:
                    logger.info("\n")

                if gripper_action == "close" and action_type == "pick":
                    self.attached_obj_id = stage.passive_obj_id
                elif gripper_action == "open":
                    self.attached_obj_id = None
                if self.attached_obj_id is not None:
                    arm = stage.extra_params.get("arm", "right")
                    if self.attached_obj_id.split("/")[0] not in self.articulated_objs:
                        self.robot.client.attach_obj(
                            prim_paths=[objects[self.attached_obj_id].prim_path],
                            is_right=arm == "right",
                        )
                self.robot.client.set_frame_state(
                    action_type,
                    step_index,
                    active_id,
                    passive_id,
                    self.attached_obj_id is not None,
                    arm=arm,
                    target_pose=target_gripper_pose,
                    action_description=action_description,
                    error_description=error_data,
                )
            objects = self.update_objects(objects, arm=arm)
        return step_success, need_retry

    def run(
        self,
        task_folder,
        camera_list,
        use_recording,
        workspaces,
        fps=10,
        render_semantic=False,
        origin_task_info={},
    ):
        tasks = glob.glob(task_folder + "/*.json")
        for index, task_file in enumerate(tasks):
            success = True
            if not self.check_task_file(task_file):
                logger.error(f"Task file {task_file} check failed, skip this task")
                continue
            task_info, objects = self.load_task(
                task_file,
                use_recording,
                camera_list,
                fps,
                render_semantic,
                workspaces,
                origin_task_info,
            )
            time.sleep(1)
            if "task_description" in task_info:
                self.robot.client.set_task_task_basic_info(task_info["task_description"])
            if "task_metric" in task_info:
                self.robot.client.set_task_metric(task_info["task_metric"])
            logger.info(f"Task {task_file} loaded")

            task_success = True
            for stage_id, stage in enumerate(self.action_script):
                if (
                    "action_text" in stage.action_description
                    and "english_action_text" in stage.action_description
                ):
                    logger.info(
                        f"RUNNING TASK: Stage {stage_id} action_text: {stage.action_description['action_text']}"
                    )
                    logger.info(
                        f"RUNNING TASK: Stage {stage_id} english_action_text: {stage.action_description['english_action_text']}"
                    )
                # Special handling for reset and place stages
                if stage.action_type in ["reset"]:
                    extra_params = stage.extra_params
                    arm = extra_params.get("arm", "right")
                    plan_type = extra_params.get("plan_type", "AvoidObs")
                    init_pose = self.robot.reset_pose[arm]
                    curr_pose = self.robot.get_ee_pose(ee_type="gripper", id=arm)
                    interp_pose = init_pose.copy()
                    interp_pose[:3, 3] = (
                        curr_pose[:3, 3] + (init_pose[:3, 3] - curr_pose[:3, 3]) * 0.25
                    )
                    task_success = self.robot.move_pose(
                        self.robot.reset_pose[arm],
                        type=plan_type,
                        arm=arm,
                        block=True,
                    )
                    if not task_success and plan_type == "AvoidObs":
                        logger.error(
                            f"Stage {stage_id} reset move to reset pose with AvoidObs failed, try Simple"
                        )
                        task_success = self.robot.move_pose(
                            self.robot.reset_pose[arm],
                            type="Simple",
                            arm=arm,
                            block=True,
                        )
                    continue
                # Initialize action sequence
                if not stage.initialize_action_sequence_buffer(objects, self.robot):
                    task_success = False
                    logger.warning(
                        f"Stage {stage_id} {stage.action_type} initialize action sequence buffer failed"
                    )
                    break
                attempt_count = 0
                stage_success = False
                try_next_sequence = True
                store_name = f"stage_{stage_id}"
                self.robot.client.store_current_state(store_name)
                logger.info(f"Store state {store_name}")
                while not stage_success and try_next_sequence and attempt_count < MAX_ATTEMPTIONS:
                    # Execution
                    stage_success = True
                    attempt_count += 1
                    action_sequence = stage.get_action_sequence()
                    if not action_sequence:
                        # no more action sequence to try, fail this stage
                        stage_success = False
                        try_next_sequence = False
                        logger.info(
                            f"Stage {stage_id} {stage.action_type} failed due to no more action sequence"
                        )
                        break
                    for action in action_sequence:
                        step_success, try_next_sequence = self.step(
                            action,
                            stage,
                            objects,
                            workspaces,
                            stage_id,
                            attempt_count,
                            action_sequence.get_step_index(),
                        )
                        if not step_success:
                            # for first step fail, try next sequence. Otherwise, fail this stage
                            if try_next_sequence:
                                logger.warning(
                                    f"Stage {stage_id} {stage.action_type} fail at first step, try next sequence"
                                )
                            else:
                                logger.warning(
                                    f"Stage {stage_id} {stage.action_type} fail at step {action_sequence.get_step_index()}, fail this stage"
                                )
                            stage_success = False
                            break
                    if (
                        not stage_success
                        and try_next_sequence
                        and not stage.action_sequence_buffer.empty()
                    ):
                        self.robot.client.playback(store_name)
                        logger.info(f"Playback state {store_name}")
                    elif stage.checker_config:
                        for check_config in stage.checker_config:
                            checker_status = self.robot.client.get_checker_status(check_config).msg
                            logger.info(
                                f"stage finish: {stage_id}, checker: {check_config}, status: {checker_status}"
                            )
                            if checker_status == "fail":
                                stage_success = False
                                try_next_sequence = True
                                if not stage.action_sequence_buffer.empty():
                                    self.robot.client.playback(store_name)
                                    logger.info(f"Playback state {store_name}")
                                break

                task_success = task_success and stage_success
                if not task_success:
                    logger.warning(f"Stage {stage_id} {stage.action_type} failed")
                    break

                time.sleep(1)

            if not task_success:
                success = False
                logger.warning(f"Task {task_file} failed")
            time.sleep(2)
            self.robot.client.stop_recording()
            # reset object pose
            for obj_id in objects:
                if obj_id == "gripper":
                    continue
                object_pose = {}
                object_pose["prim_path"] = objects[obj_id].prim_path
                object_pose["position"] = [99, 99, 0]
                object_pose["rotation"] = [0, 0, 0, 1]
                poses = []
                poses.append(object_pose)
                self.robot.client.set_object_pose(poses, [])

            step_id = -1
            fail_stage_step = [stage_id, step_id] if not success else [-1, -1]

            task_info.copy()
            self.robot.client.send_task_status(success, fail_stage_step)
            if success:
                logger.info(">>>>>>>>>>>>>>>>>>>>  TASK SUCCESS ! <<<<<<<<<<<<<<<<<<<<")

        return True
