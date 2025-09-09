# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, sys
import json
import time
import numpy as np

from geniesim.utils.object import OmniObject
from .utils.object import LayoutObject
from .solver_2d.solver import LayoutSolver2D

from geniesim.robot.utils import get_quaternion_wxyz_from_rotation_matrix
import geniesim.utils.system_utils as system_utils

from geniesim.utils.logger import Logger

logger = Logger()  # Create singleton instance


def list_to_dict(data: list):
    tmp = {}
    for i in range(len(data)):
        tmp[str(i)] = data[i]
    return tmp


class LayoutGenerator:
    def __init__(
        self,
        workspace,
        obj_infos,
        objects,
        key_obj_ids,
        extra_obj_ids,
        constraint=None,
        fix_obj_ids=[],
    ):
        self.workspace = workspace
        self.objects = objects
        self.obj_infos = obj_infos

        self.key_obj_ids = key_obj_ids
        self.extra_obj_ids = extra_obj_ids
        self.fix_obj_ids = fix_obj_ids
        self.constraint = constraint

        if constraint is None:
            self.key_obj_ids_2d = self.key_obj_ids
            self.key_obj_ids_3d = []
        else:
            self.key_obj_ids_2d = [constraint["passive"]]
            self.key_obj_ids_3d = [constraint["active"]]
        self.constraint = constraint

        workspace_xyz, workspace_size = np.array(workspace["position"]), np.array(
            workspace["size"]
        )
        workspace_size = workspace_size * 1000
        # extra info about workspace

        self.solver_2d = LayoutSolver2D(
            workspace_xyz,
            workspace_size,
            objects,
            fix_obj_ids=fix_obj_ids,
            obj_infos=obj_infos,
        )

        self.succ_obj_ids = []

    def __call__(self):
        """Generate Layout"""
        # import pdb;pdb.set_trace()
        if len(self.key_obj_ids_2d) > 0:
            objs_succ = self.solver_2d(
                self.key_obj_ids_2d,
                self.succ_obj_ids,
                object_extent=30,
                start_with_edge=True,
                key_obj=True,
                initial_angle=0,
            )
            self.update_obj_info(objs_succ)
            logger.info("-- 2d layout done --")

        if len(self.extra_obj_ids) > 0:
            objs_succ = self.solver_2d(
                self.extra_obj_ids,
                self.succ_obj_ids,
                object_extent=30,
                start_with_edge=False,
                key_obj=False,
            )
            self.update_obj_info(objs_succ)
            logger.info("-- extra layout done --")

        """ Check completion """
        res_infos = []
        if len(self.key_obj_ids) > 0:
            for obj_id in self.key_obj_ids:
                if obj_id not in self.succ_obj_ids:
                    return None
                res_infos.append(self.obj_infos[obj_id])
            return res_infos
        elif len(self.extra_obj_ids) > 0:
            if len(self.succ_obj_ids) > 0:
                for obj_id in self.succ_obj_ids:
                    res_infos.append(self.obj_infos[obj_id])
            return res_infos
        else:
            return res_infos

    def update_obj_info(self, obj_ids):
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        for obj_id in obj_ids:
            pose = self.objects[obj_id].obj_pose
            xyz, quat = pose[:3, 3], get_quaternion_wxyz_from_rotation_matrix(
                pose[:3, :3]
            )
            self.obj_infos[obj_id]["position"] = (xyz / 1000).tolist()
            self.obj_infos[obj_id]["quaternion"] = quat.tolist()
            self.obj_infos[obj_id]["is_key"] = obj_id in self.key_obj_ids
            self.succ_obj_ids.append(obj_id)


class TaskGenerator:
    def __init__(self, task_template):
        self.data_root = str(system_utils.assets_path())
        self.init_info(task_template)

    def _load_json(self, relative_path):
        with open(os.path.join(self.data_root, relative_path), "r") as file:
            return json.load(file)

    def init_info(self, task_template):
        # Load all objects  & constraints
        self.fix_objs = task_template["objects"].get("fix_objects", [])
        all_objs = (
            task_template["objects"]["task_related_objects"]
            + task_template["objects"]["extra_objects"]
            + self.fix_objs
        )
        self.fix_obj_ids = [obj["object_id"] for obj in self.fix_objs]

        self.key_obj_ids, self.extra_obj_ids = {"0": []}, {"0": []}
        for obj in task_template["objects"]["task_related_objects"]:
            ws_id = obj.get("workspace_id", "0")
            if ws_id not in self.key_obj_ids:
                self.key_obj_ids[ws_id] = []
            self.key_obj_ids[ws_id].append(obj["object_id"])
        for obj in task_template["objects"]["extra_objects"]:
            ws_id = obj.get("workspace_id", "0")
            if ws_id not in self.extra_obj_ids:
                self.extra_obj_ids[ws_id] = []
            self.extra_obj_ids[ws_id].append(obj["object_id"])

        obj_infos = {}
        objects = {}
        all_key_objs = [
            obj_id for ws_id in self.key_obj_ids for obj_id in self.key_obj_ids[ws_id]
        ]
        for obj in all_objs:
            obj_id = obj["object_id"]
            if obj_id == "fix_pose":
                info = dict()
                info["object_id"] = obj_id
                info["position"] = obj["position"]
                info["direction"] = obj["direction"]
                obj_infos[obj_id] = info
                objects[obj_id] = OmniObject("fix_pose")
            else:
                obj_dir = os.path.join(self.data_root, obj["data_info_dir"])
                if "metadata" in obj:
                    info = obj["metadata"]["info"]
                    info["interaction"] = obj["metadata"]["interaction"]
                else:
                    info = json.load(open(obj_dir + "/object_parameters.json"))
                info["data_info_dir"] = obj_dir
                info["obj_path"] = obj_dir + "/Aligned.obj"
                info["object_id"] = obj_id
                if "extent" in obj:
                    info["extent"] = obj["extent"]
                obj_infos[obj_id] = info
                logger.info(f"obj_id {obj_id} all_key_objs {all_key_objs}")
                objects[obj_id] = LayoutObject(info, use_sdf=obj_id in all_key_objs)

        self.obj_infos, self.objects = obj_infos, objects

        self.fix_obj_infos = []
        for fix_obj in self.fix_objs:
            fix_obj["is_key"] = True
            fix_obj.update(obj_infos[fix_obj["object_id"]])
            self.fix_obj_infos.append(fix_obj)

        if "robot" not in task_template:
            arm = "right"
            robot_id = "G1"
        else:
            arm = task_template["robot"]["arm"]
            robot_id = task_template["robot"]["robot_id"]

        scene_info = task_template["scene"]
        self.scene_usd = task_template["scene"]["scene_usd"]
        self.task_template = {
            "scene_usd": self.scene_usd,
            "arm": arm,
            "task_name": task_template["task"],
            "robot_id": robot_id,
            "stages": task_template["stages"],
            "object_with_material": task_template.get("object_with_material", {}),
            "lights": task_template.get("lights", {}),
            "lights": task_template.get("lights", {}),
            "objects": [],
        }

        constraint = task_template.get("constraints")
        robot_init_workspace_id = scene_info["scene_id"].split("/")[-1]

        # Retrieve scene information
        self.scene_usd = scene_info["scene_usd"]
        if "function_space_objects" in scene_info:
            workspaces = scene_info["function_space_objects"]
            if robot_init_workspace_id not in task_template["robot"]["robot_init_pose"]:
                self.robot_init_pose = task_template["robot"]["robot_init_pose"]
            else:
                self.robot_init_pose = task_template["robot"]["robot_init_pose"][
                    robot_init_workspace_id
                ]
        else:
            scene_info = self._load_json(
                scene_info["scene_info_dir"] + "/scene_parameters.json"
            )
            workspaces = scene_info["function_space_objects"]
            # Normalize format
            if isinstance(scene_info["robot_init_pose"], list):
                scene_info["robot_init_pose"] = list_to_dict(
                    scene_info["robot_init_pose"]
                )
            self.robot_init_pose = scene_info["robot_init_pose"][
                robot_init_workspace_id
            ]
        self.robot_init_pose
        if isinstance(workspaces, list):
            workspaces = list_to_dict(workspaces)
            workspaces = {"0": workspaces[robot_init_workspace_id]}
        elif isinstance(workspaces, dict) and "position" in workspaces:
            workspaces = {"0": workspaces}
        self.layouts = {}

        for key in workspaces:
            ws, key_ids, extra_ids = (
                workspaces[key],
                self.key_obj_ids.get(key, []),
                self.extra_obj_ids.get(key, []),
            )
            self.layouts[key] = LayoutGenerator(
                ws,
                obj_infos,
                objects,
                key_ids,
                extra_ids,
                constraint=constraint,
                fix_obj_ids=self.fix_obj_ids,
            )

    def generate_tasks(self, save_path, task_num, task_name):
        os.makedirs(save_path, exist_ok=True)

        for i in range(task_num):
            output_file = os.path.join(save_path, f"{task_name}_%d.json" % (i))
            self.task_template["objects"] = []
            self.task_template["objects"] += self.fix_obj_infos

            flag_failed = False
            for key in self.layouts:
                obj_infos = self.layouts[key]()
                if not obj_infos:
                    if obj_infos is None:
                        flag_failed = True
                        break
                    continue
                self.task_template["objects"] += obj_infos

            if flag_failed:
                logger.error(f"Failed to place key object, skipping")
                continue

            logger.info("Saved task json to %s" % output_file)
            with open(output_file, "w") as f:
                json.dump(self.task_template, f, indent=4)
