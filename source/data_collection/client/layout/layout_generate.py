# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import random
from enum import Enum

import numpy as np

from client.layout.solver_2d.solver import LayoutSolver2D
from common.base_utils.logger import logger
from common.base_utils.transform_utils import (
    axis_to_quaternion,
    mat2quat_wxyz,
    pose2mat,
    quaternion_rotate,
)


class GeneratorType(Enum):
    SPACE = 1
    SAMPLE = 2


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
        attach_obj_ids=[],
    ):
        self.workspace = workspace
        self.objects = objects
        self.obj_infos = obj_infos

        self.key_obj_ids = key_obj_ids
        self.extra_obj_ids = extra_obj_ids
        self.fix_obj_ids = fix_obj_ids
        self.attach_obj_ids = attach_obj_ids
        self.constraint = constraint
        self.generator_type = GeneratorType.SPACE

        self.constraint = constraint

        if "poses" in workspace:
            self.generator_type = GeneratorType.SAMPLE
        else:
            workspace_xyz, workspace_size = np.array(workspace["position"]), np.array(workspace["size"])
            blocked_zone = None
            if "blocked_zone" in workspace:
                blocked_zone = np.array(workspace["blocked_zone"])
            workspace_size = workspace_size * 1000
            # extra info about workspace

            self.solver_2d = LayoutSolver2D(
                workspace_xyz,
                workspace_size,
                objects,
                fix_obj_ids=fix_obj_ids,
                obj_infos=obj_infos,
                blocked_zone=blocked_zone,
            )

        self.succ_obj_ids = []

    def sample_solver(self):
        """Randomly assign object positions from workspace poses.

        For each object belonging to this workspace, randomly select a pose from the workspace's poses list,
        then calculate the object's final position in the world coordinate system based on the object's
        workspace_relative_position and workspace_relative_orientation.

        Note: Each object is assigned a unique pose. An error will be raised if the number of poses is less than the number of objects.
        """
        objs_succ = []
        if "poses" not in self.workspace:
            raise ValueError("workspace must have 'poses' field for sample_solver")

        poses = self.workspace["poses"]
        if not poses:
            raise ValueError("workspace poses list is empty")

        # Get all object IDs belonging to this workspace
        all_obj_ids = []
        if isinstance(self.key_obj_ids, list):
            all_obj_ids.extend(self.key_obj_ids)
        elif isinstance(self.key_obj_ids, dict):
            # If it's a dictionary, get all values
            for obj_ids in self.key_obj_ids.values():
                all_obj_ids.extend(obj_ids)

        if isinstance(self.extra_obj_ids, list):
            all_obj_ids.extend(self.extra_obj_ids)
        elif isinstance(self.extra_obj_ids, dict):
            # If it's a dictionary, get all values
            for obj_ids in self.extra_obj_ids.values():
                all_obj_ids.extend(obj_ids)

        # Filter out object IDs not in obj_infos
        valid_obj_ids = [obj_id for obj_id in all_obj_ids if obj_id in self.obj_infos]

        # Check if there are enough poses
        if len(poses) < len(valid_obj_ids):
            raise ValueError(
                f"Not enough poses in workspace: {len(poses)} poses available, "
                f"but {len(valid_obj_ids)} objects need to be placed"
            )

        # Randomly assign poses, ensuring each object gets a different pose
        selected_poses = random.sample(poses, len(valid_obj_ids))

        # Assign a pose to each object
        for obj_id, selected_pose in zip(valid_obj_ids, selected_poses):
            obj_info = self.obj_infos[obj_id]
            if "chinese_position_semantic" in selected_pose:
                obj_info["chinese_position_semantic"] = selected_pose["chinese_position_semantic"]
            if "english_position_semantic" in selected_pose:
                obj_info["english_position_semantic"] = selected_pose["english_position_semantic"]
            up_axis = obj_info.get("upAxis", ["y"])[0]
            upside_down = False
            if "-" in up_axis:
                up_axis = up_axis[1:]
                upside_down = True
            # Check if object has workspace_relative_position and workspace_relative_orientation
            if "workspace_relative_position" not in obj_info:
                # If no relative position, use default value [0, 0, 0]
                rel_position = np.array([0.0, 0.0, 0.0])
            else:
                rel_position = np.array(obj_info["workspace_relative_position"])

            if "workspace_relative_orientation" not in obj_info or len(obj_info["workspace_relative_orientation"]) != 4:
                # If no relative orientation, use default value [0, 0, 0, 1] (w, x, y, z)
                rel_quaternion = axis_to_quaternion(up_axis, "z", upside_down)
            else:
                rel_quat = obj_info["workspace_relative_orientation"]
                rel_quaternion = np.array(rel_quat)

            ws_position = np.array(selected_pose["position"])
            ws_quaternion = np.array(selected_pose["quaternion"])
            if "random" in selected_pose:
                if "delta_position" in selected_pose["random"]:
                    delta_position = selected_pose["random"]["delta_position"]
                    ws_position += np.array(
                        [
                            np.random.uniform(-delta_position[0], delta_position[0]),
                            np.random.uniform(-delta_position[1], delta_position[1]),
                            np.random.uniform(-delta_position[2], delta_position[2]),
                        ]
                    )
                if "delta_angle" in selected_pose["random"]:
                    delta_angle = selected_pose["random"]["delta_angle"]
                    ws_quaternion = quaternion_rotate(
                        ws_quaternion,
                        "z",
                        np.random.uniform(-delta_angle, delta_angle) * 180 / np.pi,
                    )
            # Ensure quaternion is in (w, x, y, z) format
            if len(ws_quaternion) == 4:
                ws_quat_wxyz = ws_quaternion
            else:
                ws_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
            # Convert to (x, y, z, w) format for pose2mat
            ws_quat_xyzw = np.array([ws_quat_wxyz[1], ws_quat_wxyz[2], ws_quat_wxyz[3], ws_quat_wxyz[0]])
            rel_quat_xyzw = np.array(
                [
                    rel_quaternion[1],
                    rel_quaternion[2],
                    rel_quaternion[3],
                    rel_quaternion[0],
                ]
            )

            # Build transformation matrix for workspace pose
            ws_pose_mat = pose2mat((ws_position, ws_quat_xyzw))

            # Build transformation matrix for relative pose
            rel_pose_mat = pose2mat((rel_position, rel_quat_xyzw))

            # Calculate final world coordinate system pose: world_pose = workspace_pose @ relative_pose
            final_pose_mat = ws_pose_mat @ rel_pose_mat

            final_pose_mat[:3, 3] = final_pose_mat[:3, 3] * 1000

            # Set object pose
            if obj_id in self.objects:
                self.objects[obj_id].obj_pose = final_pose_mat
                objs_succ.append(obj_id)
        return objs_succ

    def __call__(self):
        """Generate Layout"""
        if self.generator_type == GeneratorType.SAMPLE:
            objs_succ = self.sample_solver()
            self.update_obj_info(objs_succ)
        else:
            if len(self.key_obj_ids) > 0:
                objs_succ = self.solver_2d(
                    self.key_obj_ids,
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
            xyz, quat = pose[:3, 3], mat2quat_wxyz(pose[:3, :3])
            self.obj_infos[obj_id]["position"] = (xyz / 1000).tolist()
            self.obj_infos[obj_id]["quaternion"] = quat.tolist()
            self.obj_infos[obj_id]["is_key"] = obj_id in self.key_obj_ids
            self.succ_obj_ids.append(obj_id)
