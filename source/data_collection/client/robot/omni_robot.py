# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Send commands
# 001. Capture image
# 002. Move left/right hand to specified pose
# 003. Move all joints to specified angles
# 004. Get gripper pose
# 005. Get pose of any object
# 006. Add object
import json
import math
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

from client.robot import Robot
from client.robot.client import RpcClient
from common.base_utils.logger import logger
from common.base_utils.name_utils import is_g2_90d
from common.base_utils.noise_utils import add_noise_with_regex
from common.base_utils.transform_utils import (
    euler2quat,
    mat2euler,
    mat2quat_wxyz,
    quat2mat_wxyz,
    quat_xyzw_to_wxyz,
)


class IsaacSimRpcRobot(Robot):
    def __init__(
        self,
        robot_cfg="G2_omnipicker_fixed_dual.json",
        scene_usd="Default.usd",
        client_host="localhost:50051",
        position=[0, 0, 0],
        rotation=[1, 0, 0, 0],
        stand_type="cylinder",
        stand_size_x=0.1,
        stand_size_y=0.1,
        robot_init_arm_pose=None,
        robot_init_arm_pose_noise=None,
        body_init_poses=None,
    ):
        robot_urdf = robot_cfg.split(".")[0] + ".urdf"
        self.robot_cfg = robot_cfg
        if isinstance(robot_init_arm_pose, list):
            robot_joint_names = self._get_robot_joint_names()
            if len(robot_joint_names) != len(robot_init_arm_pose):
                raise ValueError("robot_init_arm_pose length does not match joint_names length")
            robot_init_arm_pose = {
                robot_joint_names[i]: robot_init_arm_pose[i] for i in range(len(robot_joint_names))
            }
        self.client = RpcClient(client_host, robot_urdf)
        # tmp: for fix joints, it must be initialized in the first step
        if robot_init_arm_pose_noise is not None:
            robot_init_arm_pose = add_noise_with_regex(robot_init_arm_pose, robot_init_arm_pose_noise)
            keys_to_remove = [k for k in robot_init_arm_pose_noise.keys() if "body_joint" in k]
            for k in keys_to_remove:
                logger.info(f"random body joint: {k}: {robot_init_arm_pose[k]}")
                del robot_init_arm_pose_noise[k]
        if body_init_poses is not None:
            self.body_init_poses = body_init_poses
            robot_init_arm_pose.update(body_init_poses)
        else:
            self.body_init_poses = {}
            for key in robot_init_arm_pose:
                if "body_joint" in key:
                    self.body_init_poses[key] = robot_init_arm_pose[key]
        self.client.init_robot(
            robot_cfg=robot_cfg,
            robot_usd="",
            scene_usd=scene_usd,
            init_position=position,
            init_rotation=rotation,
            stand_type=stand_type,
            stand_size_x=stand_size_x,
            stand_size_y=stand_size_y,
            robot_init_arm_pose=robot_init_arm_pose,
        )
        self.cam_info = None
        if "omnipicker" in robot_cfg or "Aloha" in robot_cfg or "aloha" in robot_cfg or is_g2_90d(robot_cfg):
            self.robot_gripper_2_grasp_gripper = np.array(
                [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
            )
        else:
            self.robot_gripper_2_grasp_gripper = np.array(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
            )
        self.robot_init_arm_pose = robot_init_arm_pose
        self.robot_init_arm_pose_noise = robot_init_arm_pose_noise

        # Read joint names from configuration file
        if "ARX" in robot_cfg:
            robot_names_key = "ARX"
        elif "Aloha" in robot_cfg or "aloha" in robot_cfg:
            robot_names_key = "Aloha"
        elif is_g2_90d(robot_cfg):
            robot_names_key = "G2_crsB_90d"
        elif "G1" in robot_cfg:
            robot_names_key = "G1"
        else:
            robot_names_key = "G2"
        config = self._load_robot_config()
        arm_joint_names = config[robot_names_key]["arm_joint_names"]

        # First 7 in arm_joint_names are left arm, last 7 are right arm
        num_joints_per_arm = len(arm_joint_names) // 2
        self.joint_names = {
            "left": arm_joint_names[:num_joints_per_arm],
            "right": arm_joint_names[num_joints_per_arm:],
        }

        self.init_position = position
        self.init_rotation = rotation
        self.setup()

    def _load_robot_config(self):
        """Load robot configuration file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Find project root directory
        project_root = current_dir
        while project_root != os.path.dirname(project_root):
            config_path = os.path.join(
                project_root, "config", "robot_cfg", "robot_joint_names.json"
            )
            if os.path.exists(config_path):
                break
            project_root = os.path.dirname(project_root)
        else:
            raise FileNotFoundError("Cannot find config/robot_cfg/robot_joint_names.json")

        with open(config_path, "r") as f:
            config = json.load(f)
        return config

    def _get_robot_joint_names(self):
        if "ARX" in self.robot_cfg:
            robot_names_key = "ARX"
        elif "Aloha" in self.robot_cfg or "aloha" in self.robot_cfg:
            robot_names_key = "Aloha"
        elif is_g2_90d(self.robot_cfg):
            robot_names_key = "G2_crsB_90d"
        elif "G1" in self.robot_cfg:
            robot_names_key = "G1"
        else:
            robot_names_key = "G2"
        config = self._load_robot_config()
        robot_joint_names = config[robot_names_key]["dof_order"]
        return robot_joint_names

    def set_init_pose(self, init_pose: dict, init_pose_noise: dict = None):
        if not init_pose:
            return
        if init_pose_noise is not None and init_pose_noise is not None:
            init_pose = add_noise_with_regex(init_pose, init_pose_noise)
        if not init_pose:
            return
        target_joint_position = []
        target_joint_names = []
        positions = list(init_pose.values())
        names = list(init_pose.keys())
        for idx, val in enumerate(positions):
            if val is None:
                continue
            if not np.isfinite(val):
                continue
            target_joint_position.append(val)
            target_joint_names.append(names[idx])
        self.client.set_joint_positions(
            target_joint_position,
            target_joint_names=target_joint_names,
            is_trajectory=False,
        )

    def reset(self):
        self.target_object = None
        self.client.reset()
        self.set_init_pose(self.robot_init_arm_pose, self.robot_init_arm_pose_noise)

    def setup(self):
        self.target_object = None

        # set robot init state

    def open_gripper(self, id="left", width=0.1, detach=True):
        is_Right = True if id == "right" else False
        if width is None:
            width = 0.1
        self.client.set_gripper_state(gripper_command="open", is_right=is_Right, opened_width=width)
        if detach:
            self.client.detach_obj()

    def close_gripper(self, id="left", force=50):
        is_Right = True if id == "right" else False
        self.client.set_gripper_state(gripper_command="close", is_right=is_Right, opened_width=0.00)

    def move_pose(
        self,
        target_pose,
        type,
        arm="right",
        goal_offset=[0, 0, 0, 1, 0, 0, 0],
        path_constraint=[],
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links=[],
        from_current_pose=False,
        **kwargs,
    ):
        motion_run_ratio = kwargs.get("motion_run_ratio", 1.0)
        gripper_action_timing = kwargs.get("gripper_action_timing", {})
        if type == "BimanualRigidAvoidObs":
            return self._move_bimanual_rigid(
                target_pose,
                arm=arm,
                goal_offset=goal_offset,
                path_constraint=path_constraint,
                offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
                disable_collision_links=disable_collision_links,
                motion_run_ratio=motion_run_ratio,
                max_consecutive_failures=kwargs.get("max_consecutive_failures", 1),
                from_current_pose=from_current_pose,
            )
        if type.lower() == "trajectory":
            content = {
                "type": "trajectory_4x4_pose",
                "data": target_pose,
                "goal_offset": goal_offset,
                "path_constraint": path_constraint,
                "offset_and_constraint_in_goal_frame": offset_and_constraint_in_goal_frame,
                "disable_collision_links": disable_collision_links,
                "from_current_pose": from_current_pose,
            }
        else:
            if type == "AvoidObs":
                type = "ObsAvoid"
            elif type == "Normal":
                type = "Simple"

            content = {
                "type": "matrix",
                "matrix": target_pose,
                "trajectory_type": type,
                "arm": arm,
                "goal_offset": goal_offset,
                "path_constraint": path_constraint,
                "offset_and_constraint_in_goal_frame": offset_and_constraint_in_goal_frame,
                "disable_collision_links": disable_collision_links,
                "motion_run_ratio": motion_run_ratio,
                "gripper_action_timing": gripper_action_timing,
                "from_current_pose": from_current_pose,
                "slowdown_factor": float(kwargs.get("slowdown_factor", 1.0) or 1.0),
            }
        return self.move(content)

    def _move_bimanual_rigid(
        self,
        target_pose,
        arm="right",
        goal_offset=[0, 0, 0, 1, 0, 0, 0],
        path_constraint=[],
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links=[],
        motion_run_ratio=1.0,
        max_consecutive_failures=1,
        from_current_pose=False,
        slowdown_factor=1,
    ):
        leader_arm = arm
        follower_arm = "left" if leader_arm == "right" else "right"
        if not isinstance(target_pose, np.ndarray):
            raise ValueError("BimanualRigidAvoidObs requires a 4x4 matrix target_pose")
        init_rotation_matrix = quat2mat_wxyz(self.init_rotation)
        translation_matrix = np.zeros((4, 4))
        translation_matrix[:3, :3] = init_rotation_matrix
        translation_matrix[:3, 3] = self.init_position
        translation_matrix[3, 3] = 1
        target_matrix_local = np.linalg.inv(translation_matrix) @ target_pose
        target_position_local = target_matrix_local[:3, 3]
        target_rotation_local = quat_xyzw_to_wxyz(
            euler2quat(mat2euler(target_matrix_local[:3, :3], order="zyx"), order="zyx")
        )
        command_config = {
            "command": "BimanualRigidMove",
            "leader_arm": leader_arm,
            "follower_arm": follower_arm,
            "target_position": target_position_local.tolist(),
            "target_rotation": target_rotation_local.tolist(),
            "goal_offset": goal_offset,
            "path_constraint": path_constraint if path_constraint else [],
            "offset_and_constraint_in_goal_frame": offset_and_constraint_in_goal_frame,
            "disable_collision_links": disable_collision_links,
            "motion_run_ratio": motion_run_ratio,
            "max_consecutive_failures": max_consecutive_failures,
            "from_current_pose": from_current_pose,
            "slowdown_factor": float(slowdown_factor),
        }
        result = self.client.command_any(command_config)
        return result.get("success", False)

    def dual_move_pose(self, left_pose=None, right_pose=None, **kwargs):
        """Simultaneously move both arms to their target poses (backend/trajectory mode).

        Args:
            left_pose:  4x4 np.ndarray for left arm target pose, or None to skip.
            right_pose: 4x4 np.ndarray for right arm target pose, or None to skip.
            kwargs: optional overrides — ee_interpolation (bool), distance_frame (float).

        Returns:
            dict with {"success": bool}
        """
        moves = []
        for arm_key, pose in [("left", left_pose), ("right", right_pose)]:
            if pose is None:
                continue
            if isinstance(pose, list):
                pose = np.array(pose)
            R, T = pose[:3, :3], pose[:3, 3]
            quat_wxyz = quat_xyzw_to_wxyz(euler2quat(mat2euler(R, order="zyx"), order="zyx"))
            moves.append(
                {
                    "arm": arm_key,
                    "target_position": list(T),
                    "target_rotation": list(quat_wxyz),
                    "ee_interpolation": kwargs.get("ee_interpolation", False),
                    "distance_frame": kwargs.get("distance_frame", 0.0008),
                }
            )
        if not moves:
            return {"success": True}
        return self.client.dual_moveto(moves)

    def dual_move_pose_curobo(self, left_pose=None, right_pose=None, **kwargs):
        """Simultaneously move both arms using CuRobo planners.

        Poses are in world frame (4x4 np.ndarray); internally converted to robot-local
        frame the same way the single-arm CuRobo path does it.

        Args:
            left_pose:  4x4 np.ndarray for left arm target pose, or None to skip.
            right_pose: 4x4 np.ndarray for right arm target pose, or None to skip.
            kwargs: optional overrides forwarded to each move entry —
                goal_offset, path_constraint, offset_and_constraint_in_goal_frame,
                disable_collision_links, from_current_pose.

        Returns:
            dict with {"success": bool}
        """
        init_rotation_matrix = quat2mat_wxyz(self.init_rotation)
        T_world_robot = np.eye(4)
        T_world_robot[:3, :3] = init_rotation_matrix
        T_world_robot[:3, 3] = self.init_position
        T_robot_world = np.linalg.inv(T_world_robot)

        moves = []
        for arm_key, pose in [("left", left_pose), ("right", right_pose)]:
            if pose is None:
                continue
            if isinstance(pose, list):
                pose = np.array(pose)
            target_matrix = T_robot_world @ pose
            R, T = target_matrix[:3, :3], target_matrix[:3, 3]
            target_rotation = quat_xyzw_to_wxyz(euler2quat(mat2euler(R, order="zyx"), order="zyx"))
            moves.append(
                {
                    "arm": arm_key,
                    "target_position": list(T),
                    "target_rotation": list(target_rotation),
                    "goal_offset": kwargs.get("goal_offset", [0, 0, 0, 1, 0, 0, 0]),
                    "path_constraint": kwargs.get("path_constraint", None),
                    "offset_and_constraint_in_goal_frame": kwargs.get(
                        "offset_and_constraint_in_goal_frame", True
                    ),
                    "disable_collision_links": kwargs.get("disable_collision_links", []),
                    "from_current_pose": kwargs.get("from_current_pose", False),
                    "slowdown_factor": float(kwargs.get("slowdown_factor", 1.0) or 1.0),
                }
            )
        if not moves:
            return {"success": True}
        return self.client.dual_moveto_curobo(moves)

    def set_gripper_action(self, action, arm="right"):
        assert arm in ["left", "right"]
        if action is None:
            return
        if action == "open":
            self.open_gripper(id=arm, width=0.1)
        elif action == "close":
            self.close_gripper(id=arm)

    def move(self, content):
        """
        type: str, 'matrix' or 'joint'
            'pose': np.array, 4x4
            'joint': np.array, 1xN
        """
        type = content["type"]
        arm_name = content.get("arm", "right")
        goal_offset = content.get("goal_offset", [0, 0, 0, 1, 0, 0, 0])
        path_constraint = content.get("path_constraint", [])
        offset_and_constraint_in_goal_frame = content.get(
            "offset_and_constraint_in_goal_frame", True
        )
        disable_collision_links = content.get("disable_collision_links", [])
        motion_run_ratio = content.get("motion_run_ratio", 1.0)
        gripper_action_timing = content.get("gripper_action_timing", {})
        from_current_pose = content.get("from_current_pose", False)
        slowdown_factor = float(content.get("slowdown_factor", 1.0) or 1.0)
        logger.info(f"robot: arm{arm_name}")
        if type == "matrix":
            if isinstance(content["matrix"], list):
                content["matrix"] = np.array(content["matrix"])
            R, T = content["matrix"][:3, :3], content["matrix"][:3, 3]
            quat_wxyz = quat_xyzw_to_wxyz(euler2quat(mat2euler(R, order="zyx"), order="zyx"))
            ee_interpolation = False
            target_position = T
            if "trajectory_type" in content and content["trajectory_type"] == "Simple":
                is_backend = True
                target_rotation = quat_wxyz
            elif "trajectory_type" in content and content["trajectory_type"] == "Straight":
                is_backend = True
                target_rotation = quat_wxyz
                ee_interpolation = True
            else:
                is_backend = False
                init_rotation_matrix = quat2mat_wxyz(self.init_rotation)
                translation_matrix = np.zeros((4, 4))
                translation_matrix[:3, :3] = init_rotation_matrix
                translation_matrix[:3, 3] = self.init_position
                translation_matrix[3, 3] = 1
                target_matrix = np.linalg.inv(translation_matrix) @ content["matrix"]
                target_rotation_matrix, target_position = (
                    target_matrix[:3, :3],
                    target_matrix[:3, 3],
                )
                target_rotation = quat_xyzw_to_wxyz(
                    euler2quat(mat2euler(target_rotation_matrix, order="zyx"), order="zyx")
                )

                logger.info(f"target_position is{target_position}")
                logger.info(f"target_rotation is{target_rotation}")

            state = (
                self.client.moveto(
                    target_position=target_position,
                    target_quaternion=target_rotation,
                    arm_name=arm_name,
                    is_backend=is_backend,
                    ee_interpolation=ee_interpolation,
                    goal_offset=goal_offset,
                    path_constraint=path_constraint,
                    offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
                    disable_collision_links=disable_collision_links,
                    motion_run_ratio=motion_run_ratio,
                    gripper_action_timing=gripper_action_timing,
                    from_current_pose=from_current_pose,
                    slowdown_factor=slowdown_factor,
                ).errmsg
                == "True"
            )
            logger.info(f"move! {T, quat_wxyz, state}")
            # Create rotation object
            rot = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            euler_angles = rot.as_euler("YZX", degrees=True)
            logger.info(
                f"Yaw: {euler_angles[0]}, Pitch: {euler_angles[1]}, Roll: {euler_angles[2]}"
            )

        elif type == "joint":
            state = self.client.set_joint_positions(content["position"])

        elif type.lower() == "trajectory_4x4_pose":
            waypoint_list = content["data"]

            traj = []
            for pose in waypoint_list:
                xyz = pose[:3, 3]
                quat_wxyz = mat2quat_wxyz(pose[:3, :3])
                pose = list(xyz), list(quat_wxyz)
                traj.append(pose)

            logger.info("Set Trajectory ! ")
            self.client.set_trajectory_list(traj, is_block=True)

            state = True

        else:
            raise NotImplementedError

        return state

    def get_prim_world_pose(self, prim_path, camera=False):
        rotation_x_180 = np.array(
            [[1.0, 0.0, 0.0, 0], [0.0, -1.0, 0.0, 0], [0.0, 0.0, -1.0, 0], [0, 0, 0, 1]]
        )
        response = self.client.get_object_pose(prim_path=prim_path)
        x, y, z = (
            response.object_pose.position.x,
            response.object_pose.position.y,
            response.object_pose.position.z,
        )
        quat_wxyz = np.array(
            [
                response.object_pose.rpy.rw,
                response.object_pose.rpy.rx,
                response.object_pose.rpy.ry,
                response.object_pose.rpy.rz,
            ]
        )
        rot_mat = quat2mat_wxyz(quat_wxyz)

        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = np.array([x, y, z])

        if camera:
            pose = pose @ rotation_x_180
        return pose

    def get_ee_pose(self, id="right", **kwargs):
        state = self.client.get_ee_pose(is_right=id == "right")
        xyz = np.array(
            [
                state.ee_pose.position.x,
                state.ee_pose.position.y,
                state.ee_pose.position.z,
            ]
        )
        quat = np.array(
            [
                state.ee_pose.rpy.rw,
                state.ee_pose.rpy.rx,
                state.ee_pose.rpy.ry,
                state.ee_pose.rpy.rz,
            ]
        )
        pose = np.eye(4)
        pose[:3, 3] = xyz
        pose[:3, :3] = quat2mat_wxyz(quat)
        return pose

    def solve_ik(self, poses, arm="right", type="Simple", output_link_pose=False, **kwargs):
        single_mode = len(poses.shape) == 2
        if single_mode:
            poses = poses[np.newaxis, ...]
        ObsAvoid = type == "ObsAvoid" or type == "AvoidObs"
        result = self.client.get_ik_status(
            target_poses=poses,
            is_right=arm == "right",
            ObsAvoid=ObsAvoid,
            output_link_pose=output_link_pose,
        )  # True: isaac,  False: curobo
        ik_result = []
        jacobian_score = []
        joint_positions = []
        joint_names = []
        link_poses = []
        for state in result:
            ik_result.append(state["status"])
            jacobian_score.append(state["Jacobian"])
            joint_positions.append(state["joint_positions"])
            joint_names.append(state["joint_names"])
            single_link_poses = state.get("link_poses", {})
            for link_name, link_pose in single_link_poses.items():
                single_link_poses[link_name] = [
                    np.array(
                        [
                            link_pose.position.x,
                            link_pose.position.y,
                            link_pose.position.z,
                        ]
                    ),
                    np.array(
                        [
                            link_pose.rpy.rw,
                            link_pose.rpy.rx,
                            link_pose.rpy.ry,
                            link_pose.rpy.rz,
                        ]
                    ),
                ]
            link_poses.append(single_link_poses)

        if single_mode:
            ik_result = ik_result[0]
            jacobian_score = jacobian_score[0]
            joint_positions = joint_positions[0]
            joint_names = joint_names[0]
        info = {
            "jacobian_score": np.array(jacobian_score),
            "joint_positions": np.array(joint_positions),
            "joint_names": np.array(joint_names),
            "link_poses": np.array(link_poses),
        }
        return np.array(ik_result), info
