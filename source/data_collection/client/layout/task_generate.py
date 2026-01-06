# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import json
import os
import random
import sys

import numpy as np

current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_directory)

from client.layout.layout_generate import LayoutGenerator
from client.layout.object import OmniObject
from client.layout.utils.layout_object import LayoutObject
from common.base_utils.logger import logger
from common.base_utils.transform_utils import euler2quat_wxyz, mat2quat_wxyz, quat2mat_wxyz


def list_to_dict(data: list):
    tmp = {}
    for i in range(len(data)):
        tmp[str(i)] = data[i]
    return tmp


class TaskGenerator:
    def __init__(self, task_template):
        self.data_root = os.environ.get("SIM_ASSETS")
        self.origin_task_template = task_template
        self.workspaces = {}
        self.origin_pose = np.eye(4)
        origin_position = [0, 0, 0]
        origin_quaternion = [1, 0, 0, 0]
        if "origin" in self.origin_task_template:
            origin_position = self.origin_task_template["origin"].get("position", [0, 0, 0])
            origin_quaternion = self.origin_task_template["origin"].get("quaternion", [1, 0, 0, 0])
        self.origin_pose[:3, :3] = quat2mat_wxyz(origin_quaternion)
        self.origin_pose[:3, 3] = origin_position
        self._parser_scene_and_robot()
        self._process_task_related_objects()
        self.workspaces_in_world_frame = self._get_workspace_in_world_frame()

    def _get_workspace_in_world_frame(self):
        workspace_in_world_frame = {}
        for workspace_id, workspace in self.workspaces.items():
            if "position" in workspace:
                local_position = np.array(workspace["position"])
                local_quaternion = np.array(workspace["quaternion"])
                workspace_matrix = np.eye(4)
                workspace_matrix[:3, :3] = quat2mat_wxyz(local_quaternion)
                workspace_matrix[:3, 3] = local_position
                workspace_matrix = self.origin_pose @ workspace_matrix
                workspace_in_world_frame[workspace_id] = {
                    "position": workspace_matrix[:3, 3].tolist(),
                    "quaternion": mat2quat_wxyz(workspace_matrix[:3, :3]).tolist(),
                }
            else:
                posed_world = []
                for pose in workspace["poses"]:
                    local_position = np.array(pose["position"])
                    local_quaternion = np.array(pose["quaternion"])
                    pose_matrix = np.eye(4)
                    pose_matrix[:3, :3] = quat2mat_wxyz(local_quaternion)
                    pose_matrix[:3, 3] = local_position
                    pose_matrix = self.origin_pose @ pose_matrix
                    pose_world = {
                        "position": pose_matrix[:3, 3].tolist(),
                        "quaternion": mat2quat_wxyz(pose_matrix[:3, :3]).tolist(),
                    }
                    posed_world.append(pose_world)
                workspace_in_world_frame[workspace_id] = {
                    "poses": posed_world,
                }
        return workspace_in_world_frame

    def _parser_scene_and_robot(self):
        ## Scene ##
        scene_info = self.origin_task_template["scene"]
        # if task_template["scene"]["scene_usd"] is a list, random choose one
        if isinstance(self.origin_task_template["scene"]["scene_usd"], list):
            scene_usd = random.choice(self.origin_task_template["scene"]["scene_usd"])
            self.origin_task_template["scene"]["scene_usd"] = scene_usd
        self.scene_usd = self.origin_task_template["scene"]["scene_usd"]

        ## Robot ##
        if "robot" not in self.origin_task_template:
            self.robot_arm = "right"
            self.robot_id = "G2"
        else:
            self.robot_arm = self.origin_task_template["robot"]["arm"]
            self.robot_id = self.origin_task_template["robot"]["robot_id"]
        ## Robot init pose ##
        # Retrieve scene information
        robot_init_workspace_id = scene_info["scene_id"].split("/")[-1]
        if "function_space_objects" in scene_info:
            self.workspaces = scene_info["function_space_objects"]
            if isinstance(self.workspaces, list):
                self.workspaces = list_to_dict(self.workspaces)
            if robot_init_workspace_id not in self.origin_task_template["robot"]["robot_init_pose"]:
                self.robot_init_pose = self.origin_task_template["robot"]["robot_init_pose"]
            else:
                self.robot_init_pose = self.origin_task_template["robot"]["robot_init_pose"][
                    robot_init_workspace_id
                ]
        else:
            self.robot_init_pose = self.origin_task_template["robot"]["robot_init_pose"]

        # random robot init pose
        if "random" in self.robot_init_pose:
            random_range = self.robot_init_pose["random"]
            delta_position = random_range.get("delta_position", [0, 0, 0])
            self.robot_init_pose = {
                "position": [
                    self.robot_init_pose["position"][0]
                    + np.random.uniform(-delta_position[0], delta_position[0]),
                    self.robot_init_pose["position"][1]
                    + np.random.uniform(-delta_position[1], delta_position[1]),
                    self.robot_init_pose["position"][2]
                    + np.random.uniform(-delta_position[2], delta_position[2]),
                ],
                "quaternion": self.robot_init_pose["quaternion"],
            }
            logger.info(f"Random robot init position{self.robot_init_pose}")
        robot_init_pose_mat = np.eye(4)
        robot_init_pose_mat[:3, :3] = quat2mat_wxyz(np.array(self.robot_init_pose["quaternion"]))
        robot_init_pose_mat[:3, 3] = np.array(self.robot_init_pose["position"])
        robot_init_pose_mat = self.origin_pose @ robot_init_pose_mat
        self.robot_init_pose["position"] = robot_init_pose_mat[:3, 3].tolist()
        self.robot_init_pose["quaternion"] = mat2quat_wxyz(robot_init_pose_mat[:3, :3]).tolist()
        logger.info(f"Robot init pose{self.robot_init_pose}")

    def _build_objects_and_infos(self, all_objs, task_template, key_obj_ids):
        """Build object information and object instances.

        Args:
            all_objs: List of all objects
            task_template: Task template

        Returns:
            tuple: (obj_infos, objects) Object information dictionary and object instance dictionary
        """
        obj_infos = {}
        objects = {}
        all_key_objs = [obj_id for ws_id in key_obj_ids for obj_id in key_obj_ids[ws_id]]
        for obj in all_objs:
            obj_id = obj["object_id"]
            if obj.get("scene_object", False):
                obj["data_info_dir"] = os.path.join(
                    task_template["scene"]["scene_info_dir"], obj["data_info_dir"]
                )
            if "fix_pose" in obj_id:
                info = dict()
                info["object_id"] = obj_id
                info["position"] = obj["position"]
                info["direction"] = obj["direction"]
                obj_infos[obj_id] = info
                objects[obj_id] = OmniObject(obj_id)
            else:
                obj_dir = os.path.join(self.data_root, obj["data_info_dir"])
                if "metadata" in obj:
                    logger.info(obj["metadata"])
                    info = obj["metadata"]["info"]
                    info["interaction"] = obj["metadata"]["interaction"]
                else:
                    info = json.load(open(obj_dir + "/object_parameters.json"))
                    # If obj has fields with the same name as info, overwrite the fields in info
                    for key in obj:
                        if key in info:
                            info[key] = obj[key]
                info["data_info_dir"] = obj_dir
                info["obj_path"] = obj_dir + "/Aligned.obj"
                info["object_id"] = obj_id
                if "prim_path" in obj:
                    info["prim_path"] = obj["prim_path"]
                if "workspace_id" in obj:
                    info["workspace_id"] = obj.get("workspace_id")
                if "sub_workspace" in obj:
                    info["sub_workspace"] = obj["sub_workspace"]
                if "workspace_relative_position" in obj:
                    info["workspace_relative_position"] = obj["workspace_relative_position"]
                if "workspace_relative_orientation" in obj:
                    info["workspace_relative_orientation"] = obj["workspace_relative_orientation"]
                if "add_particle" in obj:
                    info["add_particle"] = obj["add_particle"]
                if "extent" in obj:
                    info["extent"] = obj["extent"]
                if "english_semantic_name" in obj and "chinese_semantic_name" in obj:
                    if not isinstance(obj["english_semantic_name"], list):
                        obj["english_semantic_name"] = [obj["english_semantic_name"]]
                    if not isinstance(obj["chinese_semantic_name"], list):
                        obj["chinese_semantic_name"] = [obj["chinese_semantic_name"]]
                    semantic_num = min(
                        len(obj["english_semantic_name"]),
                        len(obj["chinese_semantic_name"]),
                    )
                    random_semantic_index = random.randint(0, semantic_num - 1)
                    info["english_semantic_name"] = obj["english_semantic_name"][
                        random_semantic_index
                    ]
                    info["chinese_semantic_name"] = obj["chinese_semantic_name"][
                        random_semantic_index
                    ]
                if obj.get("scene_object", False):
                    info["model_path"] = os.path.join(
                        task_template["scene"]["scene_info_dir"],
                        info.get("model_path", ""),
                    )
                    info["scene_object"] = True
                else:
                    info["scene_object"] = False
                obj_infos[obj_id] = info
                objects[obj_id] = LayoutObject(info, use_sdf=obj_id in all_key_objs)
            if "angle_constraint" in obj:
                objects[obj_id].angle_upper_limit = obj["angle_constraint"]["upper"] * 180 / np.pi
                objects[obj_id].angle_lower_limit = obj["angle_constraint"]["lower"] * 180 / np.pi
        return obj_infos, objects

    def _process_task_related_objects(self):
        # For task_related_objects, since they are allowed to use the same id, if different instances
        # have different corresponding assets, it will cause errors in subsequent runs.
        # Therefore, we need to determine the generalization result first.
        processed_task_objs = []
        for obj in self.origin_task_template["objects"]["task_related_objects"]:
            if "candidate_objects" in obj:
                # Randomly select one from candidates
                if not obj["candidate_objects"]:
                    raise ValueError(
                        f"candidate_objects is empty for object with object_id: {obj.get('object_id', 'unknown')}"
                    )

                # Check if external object_id exists
                if "object_id" not in obj:
                    raise ValueError("object_id is required when candidate_objects is specified")

                # Randomly select a candidate
                selected_candidate = random.choice(obj["candidate_objects"]).copy()

                # Overwrite candidate fields with external item fields
                # External fields have higher priority and will overwrite corresponding fields in candidate
                for key, value in obj.items():
                    if key != "candidate_objects":  # Skip the candidate_objects field itself
                        selected_candidate[key] = value

                processed_task_objs.append(selected_candidate)
            else:
                # No candidate_objects, use the original object directly
                processed_task_objs.append(obj)

        # Also update task_related_objects in task_template for subsequent use
        self.origin_task_template["objects"]["task_related_objects"] = processed_task_objs
        self.task_objs = self.origin_task_template["objects"]["task_related_objects"]

    def _recursive_sample_object(self, object_groups):
        sampled_objects = []
        for object_group in object_groups:
            # If object_group is a list, process recursively
            if "data_info_dir" in object_group:
                if object_group["data_info_dir"] in self._task_obj_data_info_dirs:
                    continue
                sampled_objects.append(object_group)
            elif "sample" in object_group:
                sample = object_group["sample"]
                if "workspace_id" not in object_group:
                    raise ValueError("workspace_id is required for object_group")
                if "num" not in sample and ("min_num" not in sample or "max_num" not in sample):
                    raise ValueError("num or min_num and max_num are required for sample")
                available_objects = object_group["available_objects"]
                group_extracted_objects = self._recursive_sample_object(available_objects)
                # group_extracted_objects is a list of list of objects [[obj1, obj2, obj3], [obj4, obj5, obj6]] or list of objects [obj1, obj2, obj3]
                max_repeat = sample.get("max_repeat", 1)
                repeated_group_extracted_objects = []
                for i in range(0, max_repeat):
                    repeated_group_extracted_objects += copy.deepcopy(group_extracted_objects)
                group_extracted_objects = repeated_group_extracted_objects

                if "num" in sample:
                    num = sample["num"]
                else:
                    max_num = min(sample["max_num"], len(group_extracted_objects))
                    min_num = min(sample["min_num"], max_num)
                    num = random.randint(min_num, max_num)
                # sample num objects from group_extracted_objects
                group_sampled_objects = random.sample(group_extracted_objects, num)
                if not len(group_sampled_objects):
                    continue
                # Flatten group_sampled_objects to list of objects [obj1, obj2, obj3, obj4, obj5, obj6]
                flat_sampled_objects = []
                for item in group_sampled_objects:
                    if isinstance(item, list):
                        flat_sampled_objects.extend(item)
                    else:
                        flat_sampled_objects.append(item)
                for key in object_group:
                    if key != "available_objects" and key != "sample":
                        for obj in flat_sampled_objects:
                            obj[key] = object_group[key]
                sampled_objects.append(flat_sampled_objects)
            else:
                raise ValueError("either data_info_dir or sample is required for object_group")
        return sampled_objects

    def _sample_objects(self, object_groups):
        grouped_sampled_objects = self._recursive_sample_object(object_groups)
        sampled_objects = []
        for grouped_sampled_object in grouped_sampled_objects:
            if isinstance(grouped_sampled_object, list):
                for sampled_object in grouped_sampled_object:
                    sampled_objects.append(sampled_object)
            else:
                sampled_objects.append(grouped_sampled_object)
        object_ids = {}
        for sampled_object in sampled_objects:
            object_id = sampled_object["object_id"]
            if object_id in object_ids:
                sampled_object["object_id"] = object_id + "_" + str(object_ids[object_id])
                object_ids[object_id] += 1
            else:
                object_ids[object_id] = 1
        return sampled_objects

    def _pre_process(self, task_template):
        # Load all objects  & constraints
        fix_objs = task_template["objects"].get("fix_objects", [])

        ## Process generalized objects
        # Process generalization in scene objects
        self._task_obj_data_info_dirs = {
            obj["data_info_dir"] for obj in self.task_objs if not obj.get("allow_duplicate", False)
        }
        selected_scene_objs = self._sample_objects(
            task_template["objects"].get("scene_objects", [])
        )
        attach_objs = self._sample_objects(task_template["objects"].get("attach_objects", []))

        # Task object collection
        all_objs = (
            task_template["objects"]["task_related_objects"]
            + fix_objs
            + attach_objs
            + selected_scene_objs
        )

        fix_obj_ids = [obj["object_id"] for obj in fix_objs]

        key_obj_ids, extra_obj_ids = {"0": []}, {"0": []}
        for obj in task_template["objects"]["task_related_objects"]:
            ws_id = obj.get("workspace_id", "0")
            if ws_id not in key_obj_ids:
                key_obj_ids[ws_id] = []
            key_obj_ids[ws_id].append(obj["object_id"])
        for obj in selected_scene_objs:
            ws_id = obj.get("workspace_id", "0")
            if ws_id not in key_obj_ids:
                key_obj_ids[ws_id] = []
            key_obj_ids[ws_id].append(obj["object_id"])

        obj_infos, objects = self._build_objects_and_infos(all_objs, task_template, key_obj_ids)

        fix_obj_infos = []
        for fix_obj in fix_objs:
            fix_obj["is_key"] = True
            fix_obj.update(obj_infos[fix_obj["object_id"]])
            fix_obj_infos.append(fix_obj)

        ## Task template ##
        task_instance = {
            "scene_usd": self.scene_usd,
            "arm": self.robot_arm,
            "task_name": task_template["task"],
            "task_description": task_template.get("task_description", {}),
            "task_metric": task_template.get("task_metric", {}),
            "robot_id": self.robot_id,
            "stages": task_template["stages"],
            "object_with_material": task_template.get("object_with_material", {}),
            "lights": task_template.get("lights", {}),
            "cameras": task_template.get("cameras", {}),
            "objects": [],
        }

        ## Layout ##
        layouts = {}
        for key in self.workspaces:
            ws, key_ids, extra_ids = (
                self.workspaces[key],
                key_obj_ids.get(key, []),
                extra_obj_ids.get(key, []),
            )
            layouts[key] = LayoutGenerator(
                ws,
                obj_infos,
                objects,
                key_ids,
                extra_ids,
                constraint=None,
                fix_obj_ids=fix_obj_ids,
            )
        return task_instance, layouts, obj_infos, attach_objs, fix_obj_infos

    def generate(self, output_file):
        task_instance, layouts, all_obj_infos, attach_objs, fix_obj_infos = self._pre_process(
            copy.deepcopy(self.origin_task_template)
        )
        task_instance["objects"] = []
        task_instance["objects"] += fix_obj_infos

        for key in layouts:
            obj_infos = layouts[key]()
            if obj_infos is None:
                return False
            task_instance["objects"] += obj_infos
        for object_info in all_obj_infos:
            if "fix_pose" in object_info:
                fix_pose_dict = all_obj_infos[object_info]
                task_instance["objects"].append(fix_pose_dict)
                break
        for obj in attach_objs:
            anchor_info = obj.get("anchor_info", None)
            if not anchor_info:
                continue
            attach_obj_info = all_obj_infos[obj["object_id"]]
            anchor_obj_id = anchor_info["anchor_object"]
            attach_obj_info["anchor_object"] = anchor_obj_id
            for obj_info in task_instance["objects"]:
                if obj_info["object_id"] == anchor_obj_id:
                    anchor_obj = obj_info
                    anchor_position = np.array(anchor_obj["position"])

                    anchor_quaternion = np.array(anchor_obj["quaternion"])
                    anchor_matrix = np.eye(4)
                    anchor_matrix[:3, :3] = quat2mat_wxyz(anchor_quaternion)
                    anchor_matrix[:3, 3] = anchor_position
                    relative_position = np.array(anchor_info.get("position", [0, 0, 0]))
                    if "quaternion" not in anchor_info:
                        random_euler = np.random.uniform(0, 2 * np.pi, 3)
                        relative_quaternion = euler2quat_wxyz(random_euler)
                    else:
                        relative_quaternion = np.array(anchor_info["quaternion"])
                    random_range = anchor_info.get("random_range", None)
                    if random_range is not None:
                        relative_position[0] += np.random.uniform(
                            -random_range[0] / 2, random_range[0] / 2
                        )
                        relative_position[1] += np.random.uniform(
                            -random_range[1] / 2, random_range[1] / 2
                        )
                        relative_position[2] += np.random.uniform(
                            -random_range[2] / 2, random_range[2] / 2
                        )
                    relative_matrix = np.eye(4)
                    relative_matrix[:3, :3] = quat2mat_wxyz(relative_quaternion)
                    relative_matrix[:3, 3] = relative_position
                    attach_obj_matrix = np.dot(anchor_matrix, relative_matrix)
                    attach_obj_info["position"] = (attach_obj_matrix[:3, 3]).tolist()
                    attach_obj_info["quaternion"] = mat2quat_wxyz(
                        attach_obj_matrix[:3, :3]
                    ).tolist()
                    attach_obj_info["is_key"] = True
                    task_instance["objects"].append(attach_obj_info)
                    break
        # convert objects to world frame
        for obj in task_instance["objects"]:
            obj_position = np.array(obj["position"])
            obj_quaternion = np.array(obj["quaternion"])
            obj_matrix = np.eye(4)
            obj_matrix[:3, :3] = quat2mat_wxyz(obj_quaternion)
            obj_matrix[:3, 3] = obj_position
            obj_matrix = self.origin_pose @ obj_matrix
            obj["position"] = obj_matrix[:3, 3].tolist()
            obj["quaternion"] = mat2quat_wxyz(obj_matrix[:3, :3]).tolist()
        logger.info("Saved task json to %s" % output_file)
        with open(output_file, "w") as f:
            json.dump(task_instance, f, indent=4)
        return True

    def generate_tasks(self, save_path, task_num, task_name):
        os.makedirs(save_path, exist_ok=True)
        import shutil

        shutil.rmtree(save_path, ignore_errors=True)
        os.makedirs(save_path, exist_ok=True)
        max_attempt = 5
        for i in range(task_num):
            output_file = os.path.join(save_path, f"{task_name}_%d.json" % (i))
            for attempt in range(max_attempt):
                if self.generate(output_file):
                    break
                else:
                    logger.error(
                        f"Attempt {attempt+1}/{max_attempt} failed for task {i}, retrying..."
                    )
