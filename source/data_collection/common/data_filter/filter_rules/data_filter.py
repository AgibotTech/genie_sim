# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import os
import random
from math import acos

import numpy as np

from common.base_utils.logger import logger
from common.data_filter.filter_rules import math_ as ma


class DataFilter:
    def __init__(self, config) -> None:
        self.config = config

    def get_ids(self, path):
        parts = path.split(os.sep)
        job_id = parts[-4]
        task_id = parts[-3]
        episode_id = parts[-1]
        return job_id, task_id, episode_id

    def filter_data(self, data_dir):
        self.data_dir = data_dir

        job_id, task_id, episode_id = self.get_ids(self.data_dir)
        collected_data_valid = True
        result_code = 0
        state_file = os.path.join(self.data_dir, "state.json")
        with open(state_file, "r", encoding="utf-8") as sf:
            self.state = json.load(sf)
            for filter_rule in self.config["filter_rules"]:
                rule_name = filter_rule.get("rule_name", None)
                # Get corresponding filter method (method name same as rule name)
                filter_method = getattr(self, rule_name, None)
                if filter_method is None:
                    raise ValueError(f"Filter method '{rule_name}' not found in DataFilter class")
                collected_data_valid = filter_method(**filter_rule.get("params", {}))

                logger.info(f"Folder name: {episode_id}, ", end="")  # Note: end parameter is ignored in logger
                if not collected_data_valid:
                    result_code = filter_rule.get("result_code", 1)
                    logger.info(f"Can't pass [{rule_name}] check")
                    status = rule_name
                    return collected_data_valid, result_code, status
                logger.info(f"Pass [{rule_name}] check")
            status = "success"
            return collected_data_valid, result_code, status

    """
    Whether object finally reaches target area

    objects: Names of objects to be placed
    target: Can be string or list, string represents name of container for placing objects, and gets last frame position through this name
        If list, length must be 3, representing target point coordinates in world coordinate system
    target_scope: Represents allowable range of target point, within this range, object is considered placed in target area
    """

    def is_object_reach_target(
        self, objects: list[str], target: str | list = None, target_scope: list[list[float]] = None
    ) -> bool:
        for obj in objects:
            last_frames = self.state["frames"][-1]
            object_matrix_last = last_frames["objects"][obj]["pose"]
            object_pos_last = np.array(object_matrix_last)[:3, 3]

            if isinstance(target, str):
                target_pos_last = np.array(last_frames["objects"][target]["pose"])[:3, 3]
            elif isinstance(target, list):
                if len(target) == 3:
                    target_pos_last = np.array(target)
                else:
                    raise ValueError("When input target as [List], the length must be 3!")
            else:
                raise ValueError("The type of target must be [List: len=3] or [str: target_obj_name]!")
            distance = object_pos_last - target_pos_last
            target_scope_array = np.array(target_scope)
            if target_scope_array.shape != (3, 2):
                raise ValueError("The shape of target_scope must be (3, 2)")
            is_reach_target = np.all((distance >= target_scope_array[:, 0]) & (distance <= target_scope_array[:, 1]))
            if not is_reach_target:
                return False
        return True

    """
    Whether interactive objects that need to remain fixed in scene have collisions (pose offset)
    objects is a Tuple variable containing two str variables,
    first represents object type, can be one of 'object','camera','gripper'
    second represents object name, if 'object' is scene object name, 'camera' is scene camera name, 'gripper' is left or right
    euler_threshold unit is degrees
    """

    def is_object_pose_similar2start(
        self,
        objects: list[(str, str)],
        pos_threshold: list[float] = [0.1, 0.1, 0.1],
        euler_threshold: list[float] = [5, 5, 5],
        check_exist: bool = True,
    ) -> bool:
        for obj_tuple in objects:
            first_frames = self.state["frames"][0]
            for frame in self.state["frames"][1:]:
                is_pose_similar = ma.check_pose_similar(
                    frame0=first_frames,
                    obj0=obj_tuple,
                    frame1=frame,
                    obj1=obj_tuple,
                    pos_threshold=pos_threshold,
                    euler_threshold=euler_threshold,
                    check_exist=check_exist,
                )
                if not is_pose_similar:
                    return False
        return True

    """
    Check if objects are within image range

    objects: Objects to be detected, pass as list, can detect multiple objects
    camera: Camera to check, name corresponds to state.json
    downsample_ratio: Downsampling ratio (< 1.0), downsample total frames to speed up
    refresh_rate: Actual frame rate used for simulation data collection, default 30
    out_view_allow_time: Allowable time (unit: s) for objects to leave camera view, if time exceeds out_view_allow_time will be judged as failure
    camear_z_reverse: Whether z-axis is reversed, in G1 robot, head view z-axis defaults to pointing backward, so default is True, adjust according to actual situation for other cases
    """

    def is_object_in_view(
        self,
        objects: list[str],
        camera: str = "head",
        downsample_ratio: float = 0.2,
        refresh_rate: int = 30,
        out_view_allow_time: float = 0.5,
        camear_z_reverse: bool = True,
    ) -> bool:
        # Extract world coordinates of object center (assuming object center is at local coordinate system origin)
        frames = self.state["frames"]
        frames_length = len(frames)
        downsample = int(frames_length * downsample_ratio)
        frames = random.sample(frames, downsample)

        intrinsic = self.state["cameras"][camera]["intrinsic"]
        fx, fy = intrinsic["fx"], intrinsic["fy"]
        ppx, ppy = intrinsic["ppx"], intrinsic["ppy"]

        out_view_max_frame_count = max(1, int(out_view_allow_time * refresh_rate * downsample_ratio))
        out_view_frame_count = np.zeros(len(objects), dtype=int)
        for frame in frames:
            extrinsic_matrix = np.array(frame["cameras"][camera]["pose"])
            for obj_index in range(len(objects)):
                obj_pos_world = np.array(frame["objects"][objects[obj_index]]["pose"])[
                    :3, 3
                ]  # Object world coordinates [x, y, z]
                obj_pos_world_h = np.append(obj_pos_world, 1.0)  # Homogeneous coordinates [x, y, z, 1]
                # Calculate object coordinates in camera coordinate system
                # World coordinates → Camera coordinates
                obj_pos_cam_h = np.linalg.inv(extrinsic_matrix) @ obj_pos_world_h
                obj_pos_cam = obj_pos_cam_h[:3]  # [X_c, Y_c, Z_c]
                # Check depth (normally Z_c must be positive)
                if camear_z_reverse:
                    if obj_pos_cam[2] >= 0:  # Object is behind camera
                        return False
                else:
                    if obj_pos_cam[2] <= 0:  # Object is behind camera
                        return False

                # Project to image plane
                X_c, Y_c, Z_c = obj_pos_cam
                u = (fx * X_c / Z_c) + ppx
                v = (fy * Y_c / Z_c) + ppy
                # Check if within image range
                if 0 <= u < intrinsic["width"] and 0 <= v < intrinsic["height"]:
                    out_view_frame_count[obj_index] = 0
                else:
                    out_view_frame_count[obj_index] += 1
                if np.any(out_view_frame_count >= out_view_max_frame_count):
                    logger.info(obj_pos_world)
                    return False
        return True

    """
    Check if a gripper is within camera view

    Same as: Check if objects are within image range
    """

    def is_gripper_in_view(
        self,
        gripper: str = "right",
        camera: str = "head",
        downsample_ratio: float = 0.2,
        refresh_rate: int = 30,
        out_view_allow_time: float = 0.5,
        camear_z_reverse: bool = True,
        zoom_factor: float = 1.05,
    ) -> bool:
        frames = self.state["frames"]
        frames_length = len(frames)
        downsample = int(frames_length * downsample_ratio)
        frames = random.sample(frames, downsample)

        intrinsic = self.state["cameras"][camera]["intrinsic"]
        fx, fy = float(intrinsic["fx"]), float(intrinsic["fy"])
        if "ppx" not in intrinsic:
            ppx = float(intrinsic["cx"])
        else:
            ppx = float(intrinsic["ppx"])
        if "ppy" not in intrinsic:
            ppy = float(intrinsic["cy"])
        else:
            ppy = float(intrinsic["ppy"])

        if "imageSize" in intrinsic:
            width, height = intrinsic["imageSize"].strip("()").split(",")
            width = float(width)
            height = float(height)
        else:
            width = float(intrinsic["width"])
            height = float(intrinsic["height"])

        out_view_max_frame_count = max(1, int(out_view_allow_time * refresh_rate * downsample_ratio))
        out_view_frame_count = 0
        for frame in frames:
            extrinsic_matrix = np.array(frame["cameras"][camera]["pose"])
            if "ee_center" in frame:
                gripper_pos = np.array(frame["ee_center"][gripper]["pose"])[:3, 3]
            else:
                gripper_pos = np.array(frame["ee"][gripper]["pose"])[:3, 3]
            gripper_pos_h = np.append(gripper_pos, 1.0)  # Homogeneous coordinates [x, y, z, 1]
            # Calculate object coordinates in camera coordinate system
            # World coordinates → Camera coordinates
            gripper_pos_cam_h = np.linalg.inv(extrinsic_matrix) @ gripper_pos_h
            gripper_pos_cam = gripper_pos_cam_h[:3]  # [X_c, Y_c, Z_c]
            # Check depth (normally Z_c must be positive)
            if camear_z_reverse:
                if gripper_pos_cam[2] >= 0:  # Object is behind camera
                    return False
            else:
                if gripper_pos_cam[2] <= 0:  # Object is behind camera
                    return False
            # Project to image plane
            X_c, Y_c, Z_c = gripper_pos_cam
            u = (fx * X_c / Z_c) / zoom_factor + ppx
            v = (fy * Y_c / Z_c) / zoom_factor + ppy
            # Check if within image range
            if 0 <= u < width and 0 <= v < height:
                out_view_frame_count = 0
            else:
                out_view_frame_count += 1
            if np.any(out_view_frame_count >= out_view_max_frame_count):
                return False
        return True

    """
    Detect if object's final pose is vertically upward
    objects: list, can specify multiple objects
    objects_up_axis: Upward axis for each object, can be string like 'y', or list of length 3,
        Note this upward direction references object's own coordinate system, e.g., in sim when object is normally placed, y-axis is up,
        then can write str as 'y', or list as [0,1,0]
    thresholds: Angle threshold between object's upward axis and world coordinate system z-axis direction, in radians
    """

    def is_object_end_pose_up(
        self, objects: list[str], objects_up_axis: list[str] | list[list], thresholds: list[float]
    ) -> bool:
        last_frames = self.state["frames"][-1]
        for i in range(len(objects)):
            obj = objects[i]
            up_vector_local = ma.get_corresponding_vector(objects_up_axis[i])
            obj_pose_last = np.array(last_frames["objects"][obj]["pose"])
            R = obj_pose_last[:3, :3]
            up_vector_world = R @ up_vector_local
            dot_product = np.dot(up_vector_world, np.array([0, 0, 1]))
            norm_world = np.linalg.norm(up_vector_world)
            norm_z = 1.0  # Z-axis magnitude is fixed at 1
            angle_rad = acos(dot_product / (norm_world * norm_z))

            threshold = thresholds[i]
            if angle_rad > threshold:
                return False
        return True

    """

    Determine if target object's final state position is higher than initial value by delta_z on z-axis, with tolerance error.

    """

    def is_object_end_higher_than_start(
        self, objects: list[str], delta_z: float = 0.08, tolerance: float = 0.03
    ) -> bool:

        frames = self.state["frames"]

        if not frames:
            return False

        for obj in objects:
            logger.info(f"obj:{obj}")
            if obj not in frames[0]["objects"]:
                continue  # Skip if object does not exist

            start_z = np.array(frames[0]["objects"][obj]["pose"])[2, 3]
            logger.info(f"start_z:{start_z}")
            end_z = np.array(frames[-1]["objects"][obj]["pose"])[2, 3]

            # Determine if y-axis displacement falls within given interval
            if not (delta_z - tolerance <= end_z - start_z):
                return False
        return True

    """
    Check if distance between specified objects is greater than threshold
    objects: List of objects to check
    min_distance: Minimum distance threshold (unit: meters)
    check_frame: Frame to check, can be "first" (first frame), "last" (last frame) or "all" (all frames)
    """

    def is_objects_distance_greater_than(
        self, objects: list[str], min_distance: float = 0.3, check_frame: str = "last"
    ) -> bool:
        if len(objects) < 2:
            return True  # No need to check if less than 2 objects

        frames = self.state["frames"]

        if check_frame == "first":
            frames_to_check = [frames[0]]
        elif check_frame == "last":
            frames_to_check = [frames[-1]]
        elif check_frame == "all":
            frames_to_check = frames
        else:
            raise ValueError("check_frame must be 'first', 'last' or 'all'")

        for frame in frames_to_check:
            # Get positions of all objects
            positions = []
            for obj in objects:
                if obj not in frame["objects"]:
                    return False  # If object does not exist
                pos = np.array(frame["objects"][obj]["pose"])[:3, 3]
                positions.append(pos)

            # Check distances between all object pairs
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance < min_distance:
                        return False
        return True

    """
    Check if object's final position is within specified region
    objects: List of objects to check
    region_center: Region center point coordinates (in world coordinate system)
    region_size: Region size (length, width, height)
    """

    def is_object_end_in_region(self, objects: list[str], region_center: list[float], region_size: list[float]) -> bool:
        last_frame = self.state["frames"][-1]
        region_min = np.array(region_center) - np.array(region_size) / 2
        region_max = np.array(region_center) + np.array(region_size) / 2

        for obj in objects:
            if obj not in last_frame["objects"]:
                return False  # If object does not exist
            obj_pos = np.array(last_frame["objects"][obj]["pose"])[:3, 3]
            if not np.all((obj_pos >= region_min) & (obj_pos <= region_max)):
                return False
        return True

    """
    Check if object has been grasped, and determine if there is unexpected drop
    """

    def is_object_grasped_by_gripper(
        self,
        objs: list[str],
        gripper: str = "right",
        active_gripper_joint: str = "idx81_gripper_r_outer_joint1",
        grasp_time_threshold: float = 1.0,
        check_unexpected_drop: bool = False,
        object_gripper_move_threshold: float = 0.002,
    ) -> bool:
        frames = self.state["frames"]
        for object_name in objs:
            if object_name in frames[0]["objects"]:
                obj = object_name
                break
        gripper_joint_index = frames[0]["robot"]["joints"]["joint_name"].index(active_gripper_joint)
        last_gripper_pose = frames[0]["ee"][gripper]["pose"]
        last_object_pose = frames[0]["objects"][obj]["pose"]
        last_gripper_joint_pos = frames[0]["robot"]["joints"]["joint_position"][gripper_joint_index]
        last_obj2gripper_pose = np.linalg.inv(last_gripper_pose) @ last_object_pose
        standstill_count = 0
        grasp_count = 0
        gripper_open_velocity_threshold = 0.02  # greater than this value means the gripper is opening
        # Check if gripper grasps object, and if gripper is opening when gripper and object separate
        for frame_index in range(len(frames) - 1):
            frame = frames[frame_index]
            gripper_pose = np.array(frame["ee"][gripper]["pose"])
            object_pose = np.array(frame["objects"][obj]["pose"])
            gripper_joint_pos = frame["robot"]["joints"]["joint_position"][gripper_joint_index]
            if not check_unexpected_drop:
                if grasp_count >= grasp_time_threshold * 30:  # 30 frames = 1 second
                    return True
            else:
                if (
                    grasp_count >= grasp_time_threshold * 30
                    and gripper_joint_pos - last_gripper_joint_pos > gripper_open_velocity_threshold
                ):
                    return True
            gripper_transmission = np.linalg.inv(last_gripper_pose) @ gripper_pose
            if np.linalg.norm(gripper_transmission[:3, 3]) > 0.002:
                standstill_count = 0
            else:
                standstill_count += 1
            obj2gripper_pose = np.linalg.inv(gripper_pose) @ object_pose
            relatve_transmission = np.linalg.inv(last_obj2gripper_pose) @ obj2gripper_pose
            if standstill_count < 15 and np.linalg.norm(relatve_transmission[:3, 3]) < object_gripper_move_threshold:
                # Relative pose change between gripper and object is small, consider object is grasped
                if abs(gripper_joint_pos - last_gripper_joint_pos) < gripper_open_velocity_threshold:
                    grasp_count += 1
            else:
                grasp_count = 0
            last_gripper_pose = gripper_pose
            last_object_pose = object_pose
            last_gripper_joint_pos = gripper_joint_pos
            last_obj2gripper_pose = obj2gripper_pose
        return False

    # Check final position relationship between object and target object, calculate object position in target object coordinate system, check if relative position is within range (used to check if object enters container)
    def is_object_relative_position_in_target(
        self, objects: list[str], target: str, relative_position_range: list[list[float]]
    ) -> bool:
        last_frame = self.state["frames"][-1]
        target_pose = np.array(last_frame["objects"][target]["pose"])
        target_pose_inv = np.linalg.inv(target_pose)

        for obj in objects:
            obj_pose = np.array(last_frame["objects"][obj]["pose"])
            relative_pose = target_pose_inv @ obj_pose
            relative_pos = relative_pose[:3, 3]

            for i in range(3):
                if not (relative_position_range[i][0] <= relative_pos[i] <= relative_position_range[i][1]):
                    return False
        return True
