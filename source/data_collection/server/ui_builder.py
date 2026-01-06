# This software contains source code provided by NVIDIA Corporation.
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import time
from typing import Dict, Optional

import numpy as np
import omni.kit.app
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.prims import (
    delete_prim,
    get_prim_at_path,
    get_prim_children,
    get_prim_object_type,
)
from isaacsim.core.utils.types import ArticulationAction

# Perception sensors
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.physics import _sensor

# ros2bridge
from pxr import Usd

from common.base_utils.logger import logger
from common.base_utils.transform_utils import mat2quat_wxyz
from server.controllers.kinematics_solver import KinematicsSolver
from server.controllers.ruckig_move import RuckigController
from server.motion_generator.motion_gen_reacher import CuroboMotion


class UIBuilder:
    def __init__(self, world: World, debug=False):
        self.debug = debug
        self._cs = _sensor.acquire_contact_sensor_interface()
        self._is = _sensor.acquire_imu_sensor_interface()
        self.frames = []
        self.wrapped_ui_elements = []
        self.collision_paths = []
        self._Joint_Info_Sliders = []
        self._Sensor_parent = None
        self._Camera_parent = None
        self.camera: Camera = None
        self.depth_camera: Camera = None
        self.articulation = None
        self.articulation_rmpflow = None
        self.right_articulation_rmpflow = None
        self._taskspace_trajectory_generator = None
        self._target = None
        self._right_target = None
        self._currentCamera = ""
        self._followingPos = np.array([0, 0, 0])
        self._followingOrientation = np.array([1, 0, 0, 0])
        self.my_world: World = world
        self.currentImg = None
        self.currentCamInfo = None
        self.curoboMotion: Dict[str, CuroboMotion] = {}
        self.current_curobo_motion: Optional[CuroboMotion] = None
        self.camera_prim_list = []
        self.camera_list = []
        self.rmp_move = False
        self.cmd_list = None
        self.reached = False
        self.cameras = []
        self.art_controllers = []

    def _init_solver(self, robot, enable_curobo, batch_num):
        self.enable_curobo = enable_curobo
        self.robot_name = robot.robot_name
        self.robot_prim_path = robot.robot_prim_path
        self.dof_nums = robot.dof_nums
        self.lock_joints = robot.lock_joints
        self.joint_delta_time = robot.joint_delta_time
        self.curobo_config_file = robot.curobo_config_file
        self.cameras = robot.cameras
        self.init_joint_position = robot.init_joint_position
        self.end_effector_prim_path = robot.end_effector_prim_path
        self.initialize_articulation(batch_num)
        self.arm_type = robot.arm_type
        self.end_effector_name = robot.end_effector_name
        self.active_arm_joints = robot.active_arm_joints

    def _init_kinematic_solver(self, robot):
        if robot.arm_type == "dual":
            self.kinematics_solver = {
                "left": KinematicsSolver(
                    robot_description_path=robot.robot_description_path["left"],
                    urdf_path=robot.urdf_name,
                    end_effector_name=robot.end_effector_name["left"],
                    articulation=self.articulation,
                ),
                "right": KinematicsSolver(
                    robot_description_path=robot.robot_description_path["right"],
                    urdf_path=robot.urdf_name,
                    end_effector_name=robot.end_effector_name["right"],
                    articulation=self.articulation,
                ),
            }
        else:
            self.kinematics_solver = KinematicsSolver(
                robot_description_path=robot.robot_description_path,
                urdf_path=robot.urdf_name,
                end_effector_name=robot.end_effector_name,
                articulation=self.articulation,
            )

    def on_physics_step(self, step):
        for curoboMotion in self.curoboMotion.values():
            curoboMotion.on_physics_step()

    def _on_capture_cam(self, isRGB, isDepth, isSemantic):
        if self._currentCamera:
            resolution = [640, 480]
            if self._currentCamera in self.cameras:
                resolution = self.cameras[self._currentCamera]
            _Camera = Camera(prim_path=self._currentCamera, resolution=resolution)
            _Camera.initialize()
            self.camera_list.append(_Camera)
            self.camera_prim_list.append(self._currentCamera)
            focal_length = _Camera.get_focal_length()
            horizontal_aperture = _Camera.get_horizontal_aperture()
            vertical_aperture = _Camera.get_vertical_aperture()
            width, height = _Camera.get_resolution()
            fx = width * focal_length / horizontal_aperture
            fy = height * focal_length / vertical_aperture
            ppx = width * 0.5
            ppy = height * 0.5
            self.currentCamInfo = {
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "ppx": ppx,
                "ppy": ppy,
            }
            self.currentImg = {}
            self.currentImg["camera_info"] = self.currentCamInfo
            self.currentImg["rgb"] = []
            self.currentImg["depth"] = []
            self.currentImg["semantic"] = ()

    def _Generate_following_position(self, isRight=True):
        curoboMotion = self.get_curobo_motion(isRight)
        if curoboMotion:
            curoboMotion.target = XFormPrim(
                "/World/target",
                position=self._followingPos,
                orientation=self._followingOrientation,
            )
            target_world = XFormPrim(
                f"{self.robot_prim_path}/target",
            )
            target_world.set_local_pose(
                translation=self._followingPos, orientation=self._followingOrientation
            )
            if self.arm_type == "dual":
                key = self.end_effector_name["left"]
                if isRight:
                    key = self.end_effector_name["right"]
                    curoboMotion.target_links[key] = XFormPrim(
                        "/World/target_" + key,
                        position=np.array(self._followingPos),
                        orientation=np.array(self._followingOrientation),
                    )
            self._target = None

    def _follow_target(
        self,
        isRight=True,
        goal_offset=[0, 0, 0, 1, 0, 0, 0],
        path_constraint=None,
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links=[],
        from_current_pose=False,
    ):
        self._Generate_following_position(isRight)
        self.set_locked_joint_positions(isRight)
        curoboMotion = self.get_curobo_motion(isRight)
        curoboMotion.caculate_ik_goal(
            goal_offset=goal_offset,
            path_constraint=path_constraint,
            offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
            disable_collision_links=disable_collision_links,
            from_current_pose=from_current_pose,
        )
        if self.arm_type == "dual":
            js_names = self.active_arm_joints["left"]
            if isRight:
                js_names = self.active_arm_joints["right"]
            curoboMotion.exclude_js(js_names)

    def _get_ee_pose(self, is_right, is_local=False):
        robot_base_translation, robot_base_orientation = self.articulation.get_world_pose()
        if is_local:
            robot_base_translation = np.array([0.0, 0.0, 0.0])
            robot_base_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        if self.arm_type == "dual":
            key = "left"
            if is_right:
                key = "right"
            self.kinematics_solver[key]._kinematics_solver.set_robot_base_pose(
                robot_base_translation, robot_base_orientation
            )
            return self.kinematics_solver[
                key
            ]._articulation_kinematics_solver.compute_end_effector_pose()
        self.kinematics_solver._kinematics_solver.set_robot_base_pose(
            robot_base_translation, robot_base_orientation
        )
        return self.kinematics_solver._articulation_kinematics_solver.compute_end_effector_pose()

    def _get_ik_status(self, target_position, target_orientation, isRight):
        robot_base_translation, robot_base_orientation = self.articulation.get_world_pose()
        if self.arm_type == "dual":
            key = "left"
            if isRight:
                key = "right"
            self.kinematics_solver[key]._kinematics_solver.set_robot_base_pose(
                robot_base_translation, robot_base_orientation
            )
            actions, success = self.kinematics_solver[
                key
            ]._articulation_kinematics_solver.compute_inverse_kinematics(
                target_position, target_orientation
            )
        else:
            self.kinematics_solver._kinematics_solver.set_robot_base_pose(
                robot_base_translation, robot_base_orientation
            )
            actions, success = (
                self.kinematics_solver._articulation_kinematics_solver.compute_inverse_kinematics(
                    target_position, target_orientation
                )
            )
        return success, actions

    def _limit_joint_positions(self, positions, joint_indices=None):
        if joint_indices is None:
            joint_indices = np.arange(len(positions))
        lowers = self.articulation.dof_properties["lower"][joint_indices]
        uppers = self.articulation.dof_properties["upper"][joint_indices]
        positions = np.clip(positions, lowers, uppers)
        return positions

    def _safe_set_joint_positions(self, positions, joint_indices=None):
        positions = self._limit_joint_positions(positions, joint_indices)
        self.articulation.set_joint_positions(positions, joint_indices=joint_indices)

    def _move_to(self, target_positions, joint_indices=None, is_trajectory=False, is_action=False):
        if not self.articulation:
            return
        self._currentLeftTask = 10
        self._currentRightTask = 10
        actions = ArticulationAction(joint_positions=target_positions)
        if not is_trajectory:
            if joint_indices is None:
                joint_indices = []
                positions = []
                for idx, joint_position in enumerate(target_positions):
                    if joint_position is not None:
                        positions.append(joint_position)
                        joint_indices.append(idx)
                self._safe_set_joint_positions(positions, joint_indices=joint_indices)
            else:
                self._safe_set_joint_positions(target_positions, joint_indices=joint_indices)
        else:
            self.reached = True
            self.articulation.apply_action(actions)

    def _trajectory_list_follow_target(
        self,
        target_position,
        target_orientation,
        is_right,
        ee_interpolation=False,
        distance_frame=0.01,
    ):
        current_position, rotation_matrix = self._get_ee_pose(is_right)
        XFormPrim("/ruckig", position=target_position, orientation=target_orientation)
        current_rotation = mat2quat_wxyz(rotation_matrix)
        if not self._get_ik_status(target_position, target_orientation, is_right)[0]:
            self.cmd_list = None
            self.reached = True
            logger.info("IK not success")
            return
        target_arm_positions = self._get_ik_status(target_position, target_orientation, is_right)[
            1
        ].joint_positions
        self.idx_list = self._get_ik_status(target_position, target_orientation, is_right)[
            1
        ].joint_indices
        current_arm_positions = self.articulation.get_joint_positions(joint_indices=self.idx_list)

        def lerp(start, end, t):
            return start + t * (end - start)

        def slerp(q0, q1, t):
            dot = np.dot(q0, q1)
            if dot < 0.0:
                q1 = -q1
                dot = -dot

            dot = min(max(dot, -1.0), 1.0)
            theta_0 = np.arccos(dot)
            theta = theta_0 * t
            sin_theta = np.sin(theta)
            sin_theta_0 = np.sin(theta_0)
            if sin_theta == 0:
                return q0
            s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0

            return s0 * q0 + s1 * q1

        self.cmd_list = []
        if ee_interpolation:
            distance = np.linalg.norm(target_position - current_position)
            joint_distance = 0
            step = (int)(distance / distance_frame)
            if step > 1:
                for i in range(step):
                    t = i / (step - 1)
                    position = lerp(current_position, target_position, t)
                    rotation = slerp(current_rotation, target_orientation, t)
                    issuccess, arm_position = self._get_ik_status(position, rotation, is_right)
                    joint_distance = np.linalg.norm(
                        arm_position.joint_positions - current_arm_positions
                    )
                    if joint_distance < 1:
                        self.cmd_list.append(arm_position.joint_positions)
                        current_arm_positions = arm_position.joint_positions
        else:
            cmd_list = self.ruckig_controller.caculate_trajectory(
                current_arm_positions, target_arm_positions
            )
            distance = np.linalg.norm(target_position - current_position)
            for position in cmd_list:
                joint_distance = np.linalg.norm(
                    np.array(current_arm_positions) - np.array(position)
                )
                current_arm_positions = position
                self.cmd_list.append(position)
        self.cmd_idx = 0
        self.reached = False
        self.time_index = 0
        if not self.cmd_list:
            self.reached = True

    def _on_every_frame_trajectory_list(self):
        if self.cmd_list and self.articulation:
            self.time_index += 1
            cmd_state = self.cmd_list[self.cmd_idx]
            art_action = ArticulationAction(joint_positions=cmd_state, joint_indices=self.idx_list)
            for robot in self.art_controllers:
                robot.apply_action(art_action)
            self.articulation.apply_action(art_action)
            self.cmd_idx += 1
            if self.cmd_idx >= len(self.cmd_list):
                self.cmd_idx = 0
                self.cmd_list = None
                self.reached = True

    def _on_init(self):
        self.articulation = None

    def initialize_articulation(self, batch_num=0):
        scene = self.my_world.scene
        if scene._scene_registry.name_exists(self.robot_name):
            self.articulation = scene.get_object(self.robot_name)
        else:
            self.articulation = Articulation(prim_path=self.robot_prim_path, name=self.robot_name)
            scene.add(self.articulation)
        self.articulation.initialize()
        self.ruckig_controller = RuckigController(self.dof_nums, self.joint_delta_time)
        robot_list = []
        for idx in range(batch_num):
            articulation = Articulation(
                prim_path=self.robot_prim_path + "_{}".format(idx),
                name=self.robot_name + "_{}".format(idx),
            )
            articulation.initialize()
            robot_list.append(articulation)
            self.art_controllers = [r.get_articulation_controller() for r in robot_list]
        if self.enable_curobo:
            for key, cfg in self.curobo_config_file.items():
                if not self.curoboMotion.get(key):
                    before = time.time()
                    curobo_motion = self._init_curobo(cfg, key)
                    after = time.time()
                    logger.info(f"init curobo motion for {key} takes {after - before}")
                    self.curoboMotion[key] = curobo_motion
                else:
                    if self.arm_type == "dual":
                        for effector_key, value in self.end_effector_name.items():
                            if effector_key == "left":
                                current_position, rotation_matrix = self._get_ee_pose(
                                    False, is_local=True
                                )
                            else:

                                current_position, rotation_matrix = self._get_ee_pose(
                                    True, is_local=True
                                )
                            self.curoboMotion.get(key).init_ee_pose[value] = {
                                "position": current_position,
                                "orientation": mat2quat_wxyz(rotation_matrix),
                            }
                        self.curoboMotion.get(key).reset_link()
                    self.curoboMotion.get(key).reset()

    def _init_curobo(self, curobo_config, name):
        if self.articulation:
            curoboMotion = CuroboMotion(
                name,
                self.articulation,
                self.my_world,
                curobo_config,
                self.robot_prim_path,
                self.art_controllers,
                step=32,
                debug=self.debug,
            )
            curoboMotion.set_obstacles()
            return curoboMotion
        return None

    def get_curobo_motion(self, is_right=True):
        return self.curoboMotion.get("right" if is_right else "left")

    def attach_objs(self, prim_path_list, is_right=True):
        result = False
        curoboMotion = self.get_curobo_motion(is_right)
        if curoboMotion:
            logger.info("Attach!!!!")

            link_name = "attached_object"
            position, rotation_matrix = self._get_ee_pose(is_right, is_local=True)
            rotation = mat2quat_wxyz(rotation_matrix)
            if self.arm_type == "dual" and not is_right:
                link_name = "left_attached_object"
            result = curoboMotion.attach_obj(prim_path_list, link_name, position, rotation)
            curoboMotion.view_debug_world()
        return result

    def detach_objs(self):
        if self.articulation:
            for curoboMotion in self.curoboMotion.values():
                curoboMotion.detach_obj()
                logger.info("Detach!!!!!")
                curoboMotion.view_debug_world()

    def remove_objects_from_world(self, prim_paths):
        if self.articulation:
            for curoboMotion in self.curoboMotion.values():
                curoboMotion.remove_objects_from_world(prim_paths)

    def set_locked_joint_positions(self, is_right=True):
        if not self.lock_joints:
            return
        articulation = self.articulation
        joint_positions = articulation.get_joint_positions()
        articulation.dof_names
        curoboMotion = self.get_curobo_motion(is_right)
        ids = {}
        for idx in range(len(articulation.dof_names)):
            name = articulation.dof_names[idx]
            if name in curoboMotion.lock_js_names:
                ids[name] = float(joint_positions[idx])
        curoboMotion.update_lock_joints(ids)

    def _on_reset(self):
        async def _on_rest_async():
            await omni.kit.app.get_app().next_update_async()
            self.initialize_articulation()
            self.rmp_move = False

        asyncio.ensure_future(_on_rest_async())
        return

    def _find_all_objects_of_type(self, obj_type):
        items = []
        stage = omni.usd.get_context().get_stage()
        if stage:
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/")):
                path = str(prim.GetPath())
                type = get_prim_object_type(path)
                if type == obj_type:
                    items.append(path)
        return items

    def _initialize_object_articulations(self):
        articulations = self._find_all_objects_of_type("articulation")
        for art in articulations:
            articulation = Articulation(art)
            if articulation:
                articulation.initialize()

    def set_articulation_state(self, state: bool):
        articulations = self._find_all_objects_of_type("articulation")
        for art in articulations:
            _prim = get_prim_at_path(art)
            if art not in self.robot_prim_path:
                _prim.GetAttribute("physxArticulation:articulationEnabled").Set(state)

    def remove_objects(self):
        parent_prim = get_prim_at_path("/World/Objects")
        if parent_prim.IsValid():
            prims = get_prim_children(parent_prim)
            for prim in prims:
                delete_prim(prim.GetPath())

    def remove_graph(self, prim_paths):
        for prim in prim_paths:
            replicator_prim = get_prim_at_path(prim)
            if replicator_prim.IsValid():
                delete_prim(prim)
