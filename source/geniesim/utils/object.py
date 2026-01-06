# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from scipy.spatial.transform import Rotation as R
import numpy as np
import pickle
import os, json

from .transform_utils import transform_coordinates_3d
from grasp_nms import nms_grasp

from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance


class OmniObject:
    def __init__(
        self,
        name="obj",
        cam_info=None,
        type="Active",
        mask=None,
        pose=np.eye(4),
        size=np.array([0.001, 0.001, 0.001]),
    ):
        self.name = name
        self.cam_info = cam_info
        self.type = type
        self.obj_pose = pose
        self.obj_length = size
        self.info = {}

        self.xyz = np.array([0, 0, 0])
        self.direction = np.array([0, 0, 0.05])
        self.direction_proposals = None
        self.elements = {}
        if name == "gripper":
            self.elements["active"] = {
                "push": [
                    {
                        "part": "finger edge",
                        "task": "forward push",
                        "xyz": np.array([0, 0, 0]),
                        "direction": np.array([0, 0, 0.08]),
                    },
                    {
                        "part": "finger edge",
                        "task": "side push",
                        "xyz": np.array([0, 0, 0]),
                        "direction": np.array([1, 0, 0.3]),
                    },
                    {
                        "part": "finger edge",
                        "task": "side push",
                        "xyz": np.array([0, 0, 0]),
                        "direction": np.array([-1, 0, 0.3]),
                    },
                ],
                "click": {
                    "part": "finger edge",
                    "task": "click",
                    "xyz": np.array([0, 0, 0]),
                    "direction": np.array([0, 0, 1]),
                },
                "touch": {
                    "part": "finger edge",
                    "task": "touch",
                    "xyz": np.array([0, 0, 0]),
                    "direction": np.array([0, 0, 1]),
                },
                "pull": {
                    "part": "finger edge",
                    "task": "pull",
                    "xyz": np.array([0, 0, 0]),
                    "direction": np.array([0, 0, 0.08]),
                },
                "rotate": {
                    "part": "finger edge",
                    "task": "rotate",
                    "xyz": np.array([0, 0, 0]),
                    "direction": np.array([0, 0, 0.08]),
                },
            }

    @classmethod
    def from_obj_dir(cls, obj_dir, obj_info=None):
        if "interaction" in obj_info:
            obj_info = obj_info
            interaction_info = obj_info["interaction"]
            part_joint_limits_info = obj_info.get("part_joint_limits", None)

        else:

            obj_info_file = "%s/object_parameters.json" % obj_dir
            interaction_label_file = "%s/interaction.json" % obj_dir

            assert os.path.exists(obj_info_file), "object_parameters.json not found in %s" % obj_dir
            assert os.path.exists(interaction_label_file), "interaction.json not found in %s" % obj_dir

            obj_info = json.load(open(obj_info_file))
            interaction_data = json.load(open(interaction_label_file))
            interaction_info = interaction_data["interaction"]
            part_joint_limits_info = interaction_data.get("part_joint_limits", None)

        obj = cls(name=obj_info["object_id"], size=obj_info["size"])

        mesh_file = "%s/Aligned.obj" % obj_dir
        if os.path.exists(mesh_file):
            obj_info["mesh_file"] = mesh_file

        obj.part_joint_limits = part_joint_limits_info

        """ Load interaction labels """
        obj.part_ids = []
        for type in ["active", "passive"]:
            if type not in interaction_info:
                continue
            for action in interaction_info[type]:
                action_info = interaction_info[type][action]
                if action == "grasp" and type == "passive":
                    for grasp_part in action_info:
                        grasp_files = action_info[grasp_part]
                        grasp_data = {"grasp_pose": [], "width": []}
                        if isinstance(grasp_files, str):
                            grasp_files = [grasp_files]
                        for grasp_file in grasp_files:
                            grasp_file = "%s/%s" % (obj_dir, grasp_file)
                            if not os.path.exists(grasp_file):
                                continue
                            _data = pickle.load(open(grasp_file, "rb"))
                            _data["grasp_pose"] = np.array(_data["grasp_pose"])
                            _data["width"] = np.array(_data["width"])

                            if _data["grasp_pose"].shape[0] == 0:
                                continue
                            grasp_data["grasp_pose"].append(_data["grasp_pose"])
                            grasp_data["width"].append(_data["width"])
                        if len(grasp_data["grasp_pose"]) == 0:
                            continue

                        grasp_data["grasp_pose"] = np.concatenate(grasp_data["grasp_pose"])
                        grasp_data["width"] = np.concatenate(grasp_data["width"])

                        N_grasp = grasp_data["grasp_pose"].shape[0]

                        use_nms = True if N_grasp > 60 else False
                        if use_nms:
                            gripper_pose = grasp_data["grasp_pose"]
                            N = gripper_pose.shape[0]
                            width = grasp_data["width"][:N, np.newaxis]

                            rotation = gripper_pose[:, :3, :3].reshape(N, -1)
                            translation = gripper_pose[:, :3, 3]

                            height = 0.02 * np.ones_like(width)
                            depth = np.zeros_like(width)
                            score = np.ones_like(width) * 0.1
                            obj_id = -1 * np.ones_like(width)

                            grasp_group_array = np.concatenate(
                                [
                                    score,
                                    width,
                                    height,
                                    depth,
                                    rotation,
                                    translation,
                                    obj_id,
                                ],
                                axis=-1,
                            )

                            if True:
                                translation_thresh = 0.015
                                rotation_thresh = 25.0 / 180.0 * np.pi
                                grasp_group_array = nms_grasp(
                                    grasp_group_array,
                                    translation_thresh,
                                    rotation_thresh,
                                )

                            rotation = grasp_group_array[:, 4 : 4 + 9]
                            translation = grasp_group_array[:, 4 + 9 : 4 + 9 + 3]
                            width = grasp_group_array[:, 1]
                            grasp_data["grasp_pose"] = np.tile(np.eye(4), (grasp_group_array.shape[0], 1, 1))
                            grasp_data["grasp_pose"][:, :3, :3] = rotation.reshape(-1, 3, 3)
                            grasp_data["grasp_pose"][:, :3, 3] = translation
                            grasp_data["width"] = width

                            logger.info("Grasp num after NMS: %d/%d" % (grasp_data["grasp_pose"].shape[0], N_grasp))

                        action_info[grasp_part] = grasp_data

                    interaction_info[type][action] = action_info
                else:
                    for primitive in action_info:
                        for primitive_info in action_info[primitive]:
                            if "part_id" in primitive_info:
                                obj.part_ids.append(primitive_info["part_id"])
        obj.elements = interaction_info
        obj.info = obj_info
        obj.is_articulated = True if part_joint_limits_info is not None else False
        return obj

    def set_element(self, element):
        action = element["action"]
        self.elements[action] = element

    def set_mask(self, mask, roi=None):
        self.mask = mask
        self.roi = roi

    def set_pose(self, pose, length):
        self.obj_pose = pose
        self.obj_length = length

    def set_part(self, xyz=None, direction=None, direction_proposals=None, relative=True):
        if xyz is not None:
            if not isinstance(xyz, np.ndarray):
                xyz = np.array(xyz)
            if relative:
                xyz = xyz * self.obj_length / 2.0
            self.xyz = xyz

        if direction is not None:
            if not isinstance(direction, np.ndarray):
                direction = np.array(direction)
                direction = direction / np.linalg.norm(direction) * 0.08
            self.direction = direction

        if direction_proposals is not None:
            self.direction_proposals = direction_proposals

    def format_object(self, relative=True):
        xyz, direction = self.xyz, self.direction

        obj_type = self.type.lower()
        if obj_type == "active":
            xyz_start = xyz
            xyz_end = xyz_start + direction
        elif obj_type == "passive" or obj_type == "plane":
            xyz_end = xyz
            xyz_start = xyz_end - direction

        arrow_in_obj = np.array([xyz_start, xyz_end]).transpose(1, 0)
        arrow_in_world = transform_coordinates_3d(arrow_in_obj, self.obj_pose).transpose(1, 0)

        xyz_start_world, xyz_end_world = arrow_in_world

        direction_world = xyz_end_world - xyz_start_world
        direction_world = direction_world / np.linalg.norm(direction_world)

        part2obj = np.eye(4)
        part2obj[:3, 3] = xyz_start
        self.obj2part = np.linalg.inv(part2obj)

        object_world = {
            "pose": self.obj_pose,
            "length": self.obj_length,
            "xyz_start": xyz_start,
            "xyz_end": xyz_end,
            "xyz_start_world": xyz_start_world,
            "xyz_end_world": xyz_end_world,
            "direction": direction_world,
            "obj2part": self.obj2part,
        }
        return object_world
