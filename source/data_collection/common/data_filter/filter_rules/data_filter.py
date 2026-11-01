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

                logger.info(f"Folder name: {episode_id}")
                logger.info(f"rule_name: {rule_name}, result: {collected_data_valid}")
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
                raise ValueError(
                    "The type of target must be [List: len=3] or [str: target_obj_name]!"
                )
            distance = object_pos_last - target_pos_last
            target_scope_array = np.array(target_scope)
            if target_scope_array.shape != (3, 2):
                raise ValueError("The shape of target_scope must be (3, 2)")
            is_reach_target = np.all(
                (distance >= target_scope_array[:, 0]) & (distance <= target_scope_array[:, 1])
            )
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

        out_view_max_frame_count = max(
            1, int(out_view_allow_time * refresh_rate * downsample_ratio)
        )
        out_view_frame_count = np.zeros(len(objects), dtype=int)
        for frame in frames:
            extrinsic_matrix = np.array(frame["cameras"][camera]["pose"])
            for obj_index in range(len(objects)):
                obj_pos_world = np.array(frame["objects"][objects[obj_index]]["pose"])[
                    :3, 3
                ]  # Object world coordinates [x, y, z]
                obj_pos_world_h = np.append(
                    obj_pos_world, 1.0
                )  # Homogeneous coordinates [x, y, z, 1]
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
        # Preserve original frame IDs for failure diagnostics; keep the random.sample order unchanged.
        # Preserve out_view_frame_count semantics: N consecutive out-of-view samples, not trajectory frames.
        sampled = random.sample(list(enumerate(frames)), downsample)

        intrinsic = self.state["cameras"][camera]["intrinsic"]

        def _intr_float(*keys, default=None):
            for k in keys:
                v = intrinsic.get(k)
                if v is None or v == "None":
                    continue
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
            return default

        # Without valid intrinsics, projection is unavailable; treat the object as in view instead of crashing.
        fx = _intr_float("fx")
        fy = _intr_float("fy")
        if fx is None or fy is None:
            return True
        ppx = _intr_float("ppx", "cx")
        ppy = _intr_float("ppy", "cy")
        if ppx is None or ppy is None:
            return True

        if "imageSize" in intrinsic and intrinsic["imageSize"] not in (None, "None"):
            try:
                w_s, h_s = intrinsic["imageSize"].strip("()").split(",")
                width = float(w_s)
                height = float(h_s)
            except Exception:
                width = _intr_float("width")
                height = _intr_float("height")
        else:
            width = _intr_float("width")
            height = _intr_float("height")
        if width is None or height is None:
            return True

        out_view_max_frame_count = max(
            1, int(out_view_allow_time * refresh_rate * downsample_ratio)
        )
        out_view_frame_count = 0
        out_view_recent: list[tuple[int, float, float]] = []  # (frame_idx, u, v), retained for failure diagnostics
        for orig_idx, frame in sampled:
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
                    logger.info(
                        f"is_gripper_in_view FAIL[behind-camera]: gripper={gripper} "
                        f"camera={camera} frame_idx={orig_idx} "
                        f"Zc={gripper_pos_cam[2]:.4f} (>=0 with camear_z_reverse=True) "
                        f"gripper_world_pos={gripper_pos.tolist()} "
                        f"total_frames={frames_length} sampled={downsample}"
                    )
                    return False
            else:
                if gripper_pos_cam[2] <= 0:  # Object is behind camera
                    logger.info(
                        f"is_gripper_in_view FAIL[behind-camera]: gripper={gripper} "
                        f"camera={camera} frame_idx={orig_idx} "
                        f"Zc={gripper_pos_cam[2]:.4f} (<=0 with camear_z_reverse=False) "
                        f"gripper_world_pos={gripper_pos.tolist()} "
                        f"total_frames={frames_length} sampled={downsample}"
                    )
                    return False
            # Project to image plane
            X_c, Y_c, Z_c = gripper_pos_cam
            u = (fx * X_c / Z_c) / zoom_factor + ppx
            v = (fy * Y_c / Z_c) / zoom_factor + ppy
            # Check if within image range
            if 0 <= u < width and 0 <= v < height:
                out_view_frame_count = 0
                out_view_recent.clear()
            else:
                out_view_frame_count += 1
                out_view_recent.append((orig_idx, float(u), float(v)))
            if np.any(out_view_frame_count >= out_view_max_frame_count):
                run_str = ", ".join(
                    f"(idx={i},u={u_:.1f},v={v_:.1f})" for i, u_, v_ in out_view_recent
                )
                logger.info(
                    f"is_gripper_in_view FAIL[out-of-view-run]: gripper={gripper} "
                    f"camera={camera} image={int(width)}x{int(height)} "
                    f"zoom_factor={zoom_factor} "
                    f"consecutive_out={out_view_frame_count} (threshold={out_view_max_frame_count}) "
                    f"total_frames={frames_length} sampled={downsample}. "
                    f"Trailing out-of-view samples (consecutive samples, not trajectory frames): {run_str}"
                )
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
    Determine if target object's final z position does not exceed initial z by more than max_delta_z.
    Returns False if any object has end_z - start_z > max_delta_z (optionally plus tolerance).
    """

    def is_object_end_z_not_exceeding_limit(
        self, objects: list[str], max_delta_z: float = 0.08, tolerance: float = 0.0
    ) -> bool:

        frames = self.state["frames"]

        if not frames:
            return False

        for obj in objects:
            if obj not in frames[0]["objects"]:
                continue  # Skip if object does not exist

            start_z = np.array(frames[0]["objects"][obj]["pose"])[2, 3]
            end_z = np.array(frames[-1]["objects"][obj]["pose"])[2, 3]

            # Final z must not be higher than initial z by more than max_delta_z (+ tolerance)
            if end_z - start_z > max_delta_z + tolerance:
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

    def is_object_end_in_region(
        self, objects: list[str], region_center: list[float], region_size: list[float]
    ) -> bool:
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
        gripper_open_velocity_threshold = (
            0.02  # greater than this value means the gripper is opening
        )
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
            if (
                standstill_count < 15
                and np.linalg.norm(relatve_transmission[:3, 3]) < object_gripper_move_threshold
            ):
                # Relative pose change between gripper and object is small, consider object is grasped
                if (
                    abs(gripper_joint_pos - last_gripper_joint_pos)
                    < gripper_open_velocity_threshold
                ):
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
                if not (
                    relative_position_range[i][0]
                    <= relative_pos[i]
                    <= relative_position_range[i][1]
                ):
                    return False
        return True

    def is_crs_bound_valid_g2(self) -> bool:
        frames = self.state["frames"]
        def check_bound(joint6_pos, joint7_pos):
            if np.abs(joint6_pos + joint7_pos) > 1.92 or np.abs(joint6_pos - joint7_pos) > 1.92:
                return False
            return True
        invalid_frames = []
        for frame_index, frame in enumerate(frames):
            robot = frame["robot"]
            robot_joints = robot["joints"]
            robot_joint_names = robot_joints["joint_name"]
            robot_joint_positions = robot_joints["joint_position"]
            if "idx26_arm_l_joint6" not in robot_joint_names or "idx27_arm_l_joint7" not in robot_joint_names or "idx66_arm_r_joint6" not in robot_joint_names or "idx67_arm_r_joint7" not in robot_joint_names:
                continue
            joint6_l_idx = robot_joint_names.index("idx26_arm_l_joint6")
            joint7_l_idx = robot_joint_names.index("idx27_arm_l_joint7")
            joint6_r_idx = robot_joint_names.index("idx66_arm_r_joint6")
            joint7_r_idx = robot_joint_names.index("idx67_arm_r_joint7")
            joint6_l_pos = robot_joint_positions[joint6_l_idx]
            joint7_l_pos = robot_joint_positions[joint7_l_idx]
            joint6_r_pos = robot_joint_positions[joint6_r_idx]
            joint7_r_pos = robot_joint_positions[joint7_r_idx]
            if not check_bound(joint6_l_pos, joint7_l_pos):
                invalid_frames.append(frame_index)
            if not check_bound(joint6_r_pos, joint7_r_pos):
                invalid_frames.append(frame_index)
        if len(invalid_frames) > 0:
            logger.info(f"Invalid CRS bound frames: length: {len(invalid_frames)}, frames: {invalid_frames}")
            return False
        return True

    """
    Check if all body/head joints stay stable (do not move) throughout the whole episode.

    - By default, joints are selected by name substring matching: "_body_joint" and "_head_joint".
    - tolerance: maximum allowed absolute joint position change (unit follows state.json joint_position, typically radians).
    - check_exist: if True, missing joints will cause failure; if False, missing joints are skipped.
    - exclude_active_actions: frames whose frame_state.action is in this list are treated as
      intentional motion (e.g. body turn) and excluded from the stability check. The remaining
      frames are split into contiguous kept-segments and each segment is checked against its own
      first-frame baseline, so a change of posture across an excluded segment does not trigger
      failure. Pass [] to disable.
    - action_buffer_frames: extra frames excluded on each side of every excluded run, to absorb
      acceleration/deceleration near action boundaries.
    """

    def is_body_stable(
        self,
        tolerance: float = 0.01,
        joint_name_substrings: list[str] = ["_body_joint", "_head_joint"],
        joint_names: list[str] | None = None,
        check_exist: bool = True,
        print_max_diff: bool = True,
        exclude_active_actions: list[str] = ["turn"],
        action_buffer_frames: int = 5,
    ) -> bool:
        frames = self.state.get("frames", [])
        n_frames = len(frames)
        if n_frames < 2:
            return True

        first_robot_joints = frames[0].get("robot", {}).get("joints", {})
        first_names = first_robot_joints.get("joint_name", [])
        first_pos = first_robot_joints.get("joint_position", [])
        if not first_names or not first_pos:
            return False

        if joint_names is None:
            joint_names = [
                n for n in first_names if any(sub in n for sub in joint_name_substrings)
            ]

        if len(joint_names) == 0:
            logger.info("body_stable: no body/head joints matched, skip as pass")
            return True

        first_name_to_idx = {n: i for i, n in enumerate(first_names)}
        tracked: list[str] = []
        for name in joint_names:
            if name not in first_name_to_idx:
                if check_exist:
                    logger.info(f"body_stable: joint missing in first frame: {name}")
                    return False
                continue
            tracked.append(name)

        if len(tracked) == 0:
            return True

        # Build excluded-frame mask from per-frame frame_state.action, dilated by action_buffer_frames.
        excluded = [False] * n_frames
        if exclude_active_actions:
            buf = max(0, int(action_buffer_frames))
            for i in range(n_frames):
                action = (frames[i].get("frame_state") or {}).get("action")
                if action in exclude_active_actions:
                    lo = max(0, i - buf)
                    hi = min(n_frames - 1, i + buf)
                    for j in range(lo, hi + 1):
                        excluded[j] = True

        # Split into contiguous kept-segments; each segment is checked against its own base frame.
        segments: list[list[int]] = []
        cur: list[int] = []
        for i in range(n_frames):
            if excluded[i]:
                if cur:
                    segments.append(cur)
                    cur = []
            else:
                cur.append(i)
        if cur:
            segments.append(cur)

        if not segments:
            logger.info("body_stable: all frames excluded by exclude_active_actions, skip as pass")
            return True

        max_abs_diffs: dict[str, float] = {name: 0.0 for name in tracked}
        for seg in segments:
            if len(seg) < 2:
                continue
            base_idx = seg[0]
            base_joints = frames[base_idx].get("robot", {}).get("joints", {})
            base_names = base_joints.get("joint_name", [])
            base_pos = base_joints.get("joint_position", [])
            if not base_names or not base_pos:
                return False
            base_name_to_idx = {n: i for i, n in enumerate(base_names)}
            base_values: dict[str, float] = {}
            for name in tracked:
                if name not in base_name_to_idx:
                    if check_exist:
                        logger.info(
                            f"body_stable: joint missing at segment base frame {base_idx}: {name}"
                        )
                        return False
                    continue
                base_values[name] = float(base_pos[base_name_to_idx[name]])

            for frame_index in seg[1:]:
                robot_joints = frames[frame_index].get("robot", {}).get("joints", {})
                names = robot_joints.get("joint_name", [])
                pos = robot_joints.get("joint_position", [])
                if not names or not pos:
                    return False
                name_to_idx = {n: i for i, n in enumerate(names)}
                for name, base_value in base_values.items():
                    if name not in name_to_idx:
                        if check_exist:
                            logger.info(
                                f"body_stable: joint missing at frame {frame_index}: {name}"
                            )
                            return False
                        continue
                    cur_value = float(pos[name_to_idx[name]])
                    diff = abs(cur_value - base_value)
                    if diff > max_abs_diffs[name]:
                        max_abs_diffs[name] = diff

        if print_max_diff:
            seg_info = ", ".join([f"[{s[0]}..{s[-1]}]" for s in segments])
            logger.info(f"body_stable: kept segments (base=seg start): {seg_info}")
            sorted_items = sorted(max_abs_diffs.items(), key=lambda x: x[1], reverse=True)
            logger.info(
                "body_stable: max abs diffs per joint (segment-based), sorted desc:\n"
                + "\n".join([f"  {k}: {v:.6f}" for k, v in sorted_items])
            )

        return all(d <= tolerance for d in max_abs_diffs.values())
