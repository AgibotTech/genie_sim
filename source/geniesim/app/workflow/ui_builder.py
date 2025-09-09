# This software contains source code provided by NVIDIA Corporation.
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import omni
import asyncio, os

from geniesim.utils.logger import Logger

logger = Logger()  # Create singleton instance

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.prims import (
    get_prim_at_path,
    get_prim_children,
    delete_prim,
    get_prim_object_type,
)
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import (
    get_all_camera_objects,
)  # from omni.isaac.sensor import Camera

# import omni.kit.commands
# import omni.syntheticdata
import numpy as np
from geniesim.app.controllers.kinematics_solver import Kinematics_Solver
from geniesim.app.controllers.ruckig_move import Ruckig_Controller
from pxr import Usd

import omni.replicator.core as rep

from pprint import pprint


class UIBuilder:
    def __init__(self, world: World):
        self.depth_camera: Camera = None
        self.articulation = None
        self.articulation_rmpflow = None
        self._target = None
        self._currentCamera = ""
        self._followingPos = np.array([0, 0, 0])
        self._followingOrientation = np.array([1, 0, 0, 0])
        self.my_world: World = world
        self.currentImg = None
        self.currentCamInfo = None
        self.curoboMotion = None
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
        self.rep_registry_list, self.sensor_cam_list = self._init_cameras(robot)
        self.arm_type = robot.arm_type
        self.end_effector_name = robot.end_effector_name
        self.active_arm_joints = robot.active_arm_joints
        if robot.arm_type == "dual":
            self.kinematics_solver = {
                "left": Kinematics_Solver(
                    robot_description_path=robot.robot_description_path["left"],
                    urdf_path=robot.urdf_name,
                    end_effector_name=robot.end_effector_name["left"],
                    articulation=self.articulation,
                ),
                "right": Kinematics_Solver(
                    robot_description_path=robot.robot_description_path["right"],
                    urdf_path=robot.urdf_name,
                    end_effector_name=robot.end_effector_name["right"],
                    articulation=self.articulation,
                ),
            }
        else:
            self.kinematics_solver = Kinematics_Solver(
                robot_description_path=robot.robot_description_path,
                urdf_path=robot.urdf_name,
                end_effector_name=robot.end_effector_name,
                articulation=self.articulation,
            )

    def _init_cameras(self, robot):
        rep_registry_list = {}
        sensor_cam_list = {}

        all_cameras = get_all_camera_objects()
        for sensor_cam in all_cameras:
            if sensor_cam.prim_path not in robot.cameras:
                continue

            render_rp = rep.create.render_product(
                str(sensor_cam.prim_path), robot.cameras[sensor_cam.prim_path]
            )
            sensor_cam._render_product_path = render_rp.path

            rep_registry = {}
            for name in ["rgb", "distance_to_image_plane"]:
                # create annotator
                rep_annotator = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
                rep_annotator.attach(render_rp)
                # add to registry
                rep_registry[name] = rep_annotator
            rep_registry_list[sensor_cam.prim_path] = rep_registry
            sensor_cam_list[sensor_cam.prim_path] = sensor_cam
        return rep_registry_list, sensor_cam_list

    def on_physics_step(self, step):
        if self.curoboMotion:
            self.curoboMotion.on_physics_step()

    def _on_capture_cam(self, isRGB, isDepth, isSemantic):
        if self._currentCamera:
            resolution = [640, 480]
            if self._currentCamera in self.cameras:
                resolution = self.cameras[self._currentCamera]

            ref_cam = self.sensor_cam_list[self._currentCamera]
            render_rp = self.rep_registry_list[self._currentCamera]

            focal_length = ref_cam.get_focal_length()
            horizontal_aperture = ref_cam.get_horizontal_aperture()
            vertical_aperture = ref_cam.get_vertical_aperture()
            width, height = ref_cam.get_resolution()
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
            self.currentImg = {
                "camera_info": None,
                "rgb": None,
                "depth": None,
                "semantic": None,
            }
            self.currentImg["camera_info"] = self.currentCamInfo

            if isRGB:
                self.currentImg["rgb"] = render_rp["rgb"].get_data()
            if isDepth:
                self.currentImg["depth"] = render_rp[
                    "distance_to_image_plane"
                ].get_data()
            if isSemantic:
                semantics = render_rp["semantic_segmentation"].get_data()
                masks = semantics["data"]
                ids = {}
                for idx in semantics["info"]["idToLabels"]:
                    name = semantics["info"]["idToLabels"][idx]["class"]
                    ids[name] = int(idx)
                self.currentImg["semantic"] = masks, ids

    def _on_capture_cam_attach_detach(self, isRGB, isDepth, isSemantic):
        if self._currentCamera:
            resolution = [640, 480]
            if self._currentCamera in self.cameras:
                resolution = self.cameras[self._currentCamera]

            all_cameras = get_all_camera_objects()
            spec_cams = [
                obj for obj in all_cameras if obj.prim_path == self._currentCamera
            ]
            if len(spec_cams) <= 0:
                return

            ref_cam = spec_cams[0]
            render_rp = rep.create.render_product(str(ref_cam.prim_path), resolution)
            ref_cam._render_product_path = render_rp.path

            focal_length = ref_cam.get_focal_length()
            horizontal_aperture = ref_cam.get_horizontal_aperture()
            vertical_aperture = ref_cam.get_vertical_aperture()
            width, height = ref_cam.get_resolution()
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
            self.currentImg = {
                "camera_info": None,
                "rgb": None,
                "depth": None,
                "semantic": None,
            }
            self.currentImg["camera_info"] = self.currentCamInfo

            if isRGB:
                self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                self.rgb_annotator.attach(render_rp)
            if isDepth:
                self.depth_annotator = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_image_plane"
                )
                self.depth_annotator.attach(render_rp)
            if isSemantic:
                self.semantics_annotator = rep.AnnotatorRegistry.get_annotator(
                    "semantic_segmentation"
                )
                self.semantics_annotator.attach(render_rp)

            for _ in range(3):  # temp attach to grab data
                self.my_world.step(render=True)
                omni.kit.app.get_app().update()  # force Omniverse update

            if isRGB:
                self.currentImg["rgb"] = self.rgb_annotator.get_data()
                self.rgb_annotator.detach()
            if isDepth:
                self.currentImg["depth"] = self.depth_annotator.get_data()
                self.depth_annotator.detach()
            if isSemantic:
                semantics = self.semantics_annotator.get_data()
                masks = semantics["data"]
                ids = {}
                for idx in semantics["info"]["idToLabels"]:
                    name = semantics["info"]["idToLabels"][idx]["class"]
                    ids[name] = int(idx)
                self.currentImg["semantic"] = masks, ids
                self.semantics_annotator.detach()

    def _Generate_following_position(self, isRight=True):
        if self.curoboMotion:
            self.curoboMotion.target = SingleXFormPrim(
                "/World/target",
                position=self._followingPos,
                orientation=self._followingOrientation,
            )
            if self.arm_type == "dual":
                key = self.end_effector_name["left"]
                if isRight:
                    key = self.end_effector_name["right"]
                self.curoboMotion.target_links[key] = SingleXFormPrim(
                    "/World/target_" + key,
                    position=np.array(self._followingPos),
                    orientation=np.array(self._followingOrientation),
                )
            self._target = None

    def _Generate_rmp_position(self):
        self._target = SingleXFormPrim(
            "/World/rmp_target",
            position=self._followingPos,
            orientation=self._followingOrientation,
        )

    def _follow_target(self, isRight=True):
        self._Generate_following_position(isRight)
        self.curoboMotion.caculate_ik_goal()
        if self.arm_type == "dual":
            js_names = self.active_arm_joints["left"]
            if isRight:
                js_names = self.active_arm_joints["right"]
            self.curoboMotion.exclude_js(js_names)

    def _rmp_follow_target(self):
        self._Generate_rmp_position()
        self._trajectory_list_follow_target(
            self._followingPos, self._followingOrientation, True
        )

    def _get_ee_pose(self, is_right, is_local=False):
        robot_base_translation, robot_base_orientation = (
            self.articulation.get_world_pose()
        )
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
        return (
            self.kinematics_solver._articulation_kinematics_solver.compute_end_effector_pose()
        )

    def _get_ik_status(self, target_position, target_orientation, isRight):
        robot_base_translation, robot_base_orientation = (
            self.articulation.get_world_pose()
        )
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

    def _move_to(self, target_positions, target_joint_indices, is_trajectory=False):
        if not self.articulation:
            return
        self._currentLeftTask = 10
        self._currentRightTask = 10
        actions = ArticulationAction(
            joint_positions=target_positions, joint_indices=target_joint_indices
        )
        if not is_trajectory:
            self.articulation.set_joint_positions(
                target_positions, joint_indices=target_joint_indices
            )
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
        def rotation_matrix_to_quaternion(R):
            assert R.shape == (3, 3)

            trace = np.trace(R)
            if trace > 0:
                S = np.sqrt(trace + 1.0) * 2  # S=4*qw
                qw = 0.25 * S
                qx = (R[2, 1] - R[1, 2]) / S
                qy = (R[0, 2] - R[2, 0]) / S
                qz = (R[1, 0] - R[0, 1]) / S
            elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S

            return np.array([qw, qx, qy, qz])

        current_position, rotation_matrix = self._get_ee_pose(is_right)

        current_rotation = rotation_matrix_to_quaternion(rotation_matrix)
        if not self._get_ik_status(target_position, target_orientation, is_right)[0]:
            self.cmd_list = None
            self.reached = True
            return
        target_arm_positions = self._get_ik_status(
            target_position, target_orientation, is_right
        )[1].joint_positions
        self.idx_list = self._get_ik_status(
            target_position, target_orientation, is_right
        )[1].joint_indices
        current_arm_positions = self.articulation.get_joint_positions(
            joint_indices=self.idx_list
        )

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
            pre_js = None
            joint_distance = 0
            step = (int)(distance / distance_frame)
            if step > 1:
                for i in range(step):
                    t = i / (step - 1)
                    position = lerp(current_position, target_position, t)
                    rotation = slerp(current_rotation, target_orientation, t)
                    issuccess, arm_position = self._get_ik_status(
                        position, rotation, is_right
                    )
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
            art_action = ArticulationAction(
                joint_positions=cmd_state, joint_indices=self.idx_list
            )
            for robot in self.art_controllers:
                robot.apply_action(art_action)
            self.articulation.apply_action(art_action)
            self.cmd_idx += 1
            if self.cmd_idx >= len(self.cmd_list):
                self.cmd_idx = 0
                self.cmd_list = None
                self.reached = True

    def _on_time_head_plan(self, target_rotation):
        if self.articulation:
            current_position = self.articulation.get_joint_positions()[1]
            art_action = ArticulationAction(
                joint_positions=[target_rotation - current_position], joint_indices=[3]
            )
            self.articulation.apply_action(art_action)

    def _on_time_trajectory(self, position, rotation):
        if self.articulation:
            succ, result = self._get_ik_status(position, rotation, True)
            if succ:
                target_arm_positions = result.joint_positions
                idx_list = result.joint_indices
                art_action = ArticulationAction(
                    joint_positions=target_arm_positions, joint_indices=idx_list
                )
                current_joint_position = self.articulation.get_joint_positions(
                    joint_indices=idx_list
                )
                distance = np.linalg.norm(target_arm_positions - current_joint_position)
                if distance < 1:
                    self.articulation.apply_action(art_action)

    def _caculate_multiple_ik(self, trajectory_list):
        current_positions = self.articulation.get_joint_positions()
        current_joint_positions = []
        self.idx_list = self._get_ik_status(
            np.array([0, 0, 0]), np.array([0, 0, 1, 0]), True
        )[1].joint_indices
        for idx in self.idx_list:
            current_joint_positions.append(current_positions[idx])
        for point in trajectory_list:
            position, rotation = point
            succ, result = self._get_ik_status(position, rotation, True)
            distance = np.linalg.norm(result.joint_positions - current_joint_positions)
            logger.info(distance)
            if distance > 1:
                return False
            current_joint_positions = result.joint_positions
        return True

    def _teleport_robot_to_position(self, articulation_action):
        initial_positions = np.zeros(self._articulation.num_dof)
        initial_positions[articulation_action.joint_indices] = (
            articulation_action.joint_positions
        )

        self._articulation.set_joint_positions(initial_positions)
        self._articulation.set_joint_velocities(np.zeros_like(initial_positions))

    # update
    def _on_every_frame(self):
        if self.articulation is None or not self.rmp_move:
            return
        actions = None
        if self.articulation_rmpflow is None or self._target is None:
            return

        target_pos, target_orientation = self._target.get_world_pose()
        self.articulation_rmpflow.rmpflow.update_world()
        self.articulation_rmpflow.rmpflow.set_end_effector_target(
            target_pos, target_orientation
        )
        if (
            self.articulation_rmpflow._articulation_motion_policy._active_joints_view.get_joint_positions()
            is None
        ):
            self.articulation_rmpflow = None
            return
        actions = (
            self.articulation_rmpflow._articulation_motion_policy.get_next_articulation_action()
        )
        self.articulation.apply_action(actions)

    def _on_init(self):
        self.articulation = None

    def initialize_articulation(self, batch_num=0):
        scene = self.my_world.scene
        if scene._scene_registry.name_exists(self.robot_name):
            self.articulation = scene.get_object(self.robot_name)
        else:
            self.articulation = SingleArticulation(
                prim_path=self.robot_prim_path, name=self.robot_name
            )
            scene.add(self.articulation)
        self.articulation.initialize()

        self.ruckig_controller = Ruckig_Controller(self.dof_nums, self.joint_delta_time)
        robot_list = []
        for idx in range(batch_num):
            articulation = SingleArticulation(
                prim_path=self.robot_prim_path + "_{}".format(idx),
                name=self.robot_name + "_{}".format(idx),
            )
            articulation.initialize()
            robot_list.append(articulation)
            self.art_controllers = [r.get_articulation_controller() for r in robot_list]

        if self.enable_curobo:
            if not self.curoboMotion:
                self._init_curobo()
            else:
                if self.arm_type == "dual":

                    def rotation_matrix_to_quaternion(R):
                        assert R.shape == (3, 3)

                        trace = np.trace(R)
                        if trace > 0:
                            S = np.sqrt(trace + 1.0) * 2  # S=4*qw
                            qw = 0.25 * S
                            qx = (R[2, 1] - R[1, 2]) / S
                            qy = (R[0, 2] - R[2, 0]) / S
                            qz = (R[1, 0] - R[0, 1]) / S
                        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
                            qw = (R[2, 1] - R[1, 2]) / S
                            qx = 0.25 * S
                            qy = (R[0, 1] + R[1, 0]) / S
                            qz = (R[0, 2] + R[2, 0]) / S
                        elif R[1, 1] > R[2, 2]:
                            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
                            qw = (R[0, 2] - R[2, 0]) / S
                            qx = (R[0, 1] + R[1, 0]) / S
                            qy = 0.25 * S
                            qz = (R[1, 2] + R[2, 1]) / S
                        else:
                            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
                            qw = (R[1, 0] - R[0, 1]) / S
                            qx = (R[0, 2] + R[2, 0]) / S
                            qy = (R[1, 2] + R[2, 1]) / S
                            qz = 0.25 * S

                        return np.array([qw, qx, qy, qz])

                    for key, value in self.end_effector_name.items():
                        if key == "left":
                            current_position, rotation_matrix = self._get_ee_pose(
                                False, is_local=True
                            )
                        else:

                            current_position, rotation_matrix = self._get_ee_pose(
                                True, is_local=True
                            )
                        self.curoboMotion.init_ee_pose[value] = {
                            "position": current_position,
                            "orientation": rotation_matrix_to_quaternion(
                                rotation_matrix
                            ),
                        }
                    self.curoboMotion.reset_link()
                self.curoboMotion.reset()
                self._get_ee_pose

    def _init_curobo(self):
        if self.articulation:
            from geniesim.app.utils.motion_gen_reacher import CuroboMotion

            self.curoboMotion = CuroboMotion(
                self.articulation,
                self.my_world,
                self.curobo_config_file,
                self.robot_prim_path,
                self.art_controllers,
                step=100,
            )
            self.curoboMotion.set_obstacles()

    def attach_objs(self, prim_path_list, is_right=True):
        result = False
        if self.curoboMotion:
            self.set_locked_joint_positions()
            logger.info("Attach!!!!")

            def rotation_matrix_to_quaternion(R):
                assert R.shape == (3, 3)

                trace = np.trace(R)
                if trace > 0:
                    S = np.sqrt(trace + 1.0) * 2  # S=4*qw
                    qw = 0.25 * S
                    qx = (R[2, 1] - R[1, 2]) / S
                    qy = (R[0, 2] - R[2, 0]) / S
                    qz = (R[1, 0] - R[0, 1]) / S
                elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                    S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
                    qw = (R[2, 1] - R[1, 2]) / S
                    qx = 0.25 * S
                    qy = (R[0, 1] + R[1, 0]) / S
                    qz = (R[0, 2] + R[2, 0]) / S
                elif R[1, 1] > R[2, 2]:
                    S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
                    qw = (R[0, 2] - R[2, 0]) / S
                    qx = (R[0, 1] + R[1, 0]) / S
                    qy = 0.25 * S
                    qz = (R[1, 2] + R[2, 1]) / S
                else:
                    S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
                    qw = (R[1, 0] - R[0, 1]) / S
                    qx = (R[0, 2] + R[2, 0]) / S
                    qy = (R[1, 2] + R[2, 1]) / S
                    qz = 0.25 * S

                return np.array([qw, qx, qy, qz])

            link_name = "attached_object"
            position, rotation_matrix = self._get_ee_pose(is_right, is_local=True)
            rotation = rotation_matrix_to_quaternion(rotation_matrix)
            if self.arm_type == "dual" and is_right == False:
                link_name = "left_attached_object"
            result = self.curoboMotion.attach_obj(
                prim_path_list, link_name, position, rotation
            )
        return result

    def detach_objs(self):
        if self.curoboMotion and self.articulation:
            self.set_locked_joint_positions()
            self.curoboMotion.detach_obj()
            logger.info("Detach!!!!!")

    def set_locked_joint_positions(self):
        if not self.lock_joints:
            return
        articulation = self.articulation
        joint_positions = articulation.get_joint_positions()
        dof_names = articulation.dof_names
        ids = {}
        for idx in range(len(articulation.dof_names)):
            name = articulation.dof_names[idx]
            if name in self.lock_joints:
                ids[name] = float(joint_positions[idx])
        self.curoboMotion.update_lock_joints(ids)

    def _on_reset(self):
        async def _on_rest_async():
            await omni.kit.app.get_app().next_update_async()
            self.remove_objects()
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
            articulation = SingleArticulation(art)
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
