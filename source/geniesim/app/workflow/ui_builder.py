# This software contains source code provided by NVIDIA Corporation.
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
from pxr import Usd
import omni

from isaacsim.core.api import World
from isaacsim.core.utils.prims import (
    get_prim_at_path,
    get_prim_children,
    delete_prim,
    get_prim_object_type,
)
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from geniesim.app.controllers.kinematics_solver import Kinematics_Solver
from geniesim.app.controllers.ruckig_move import Ruckig_Controller
from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance


class UIBuilder:
    def __init__(self, world: World):
        self.articulation = None
        self._currentCamera = ""
        self.my_world: World = world
        self.currentImg = None
        self.currentCamInfo = None
        self.curoboMotion = None
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
        self.cameras = robot.cameras
        self.init_joint_position = robot.init_joint_position
        self.end_effector_prim_path = robot.end_effector_prim_path
        self.initialize_articulation(batch_num)
        self.arm_type = robot.arm_type
        self.end_effector_name = robot.end_effector_name
        self.active_arm_joints = robot.active_arm_joints
        if robot.robot_description_path:
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
                self.currentImg["depth"] = render_rp["distance_to_image_plane"].get_data()
            if isSemantic:
                semantics = render_rp["semantic_segmentation"].get_data()
                masks = semantics["data"]
                ids = {}
                for idx in semantics["info"]["idToLabels"]:
                    name = semantics["info"]["idToLabels"][idx]["class"]
                    ids[name] = int(idx)
                self.currentImg["semantic"] = masks, ids

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
            return self.kinematics_solver[key]._articulation_kinematics_solver.compute_end_effector_pose()
        self.kinematics_solver._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
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
            actions, success = self.kinematics_solver[key]._articulation_kinematics_solver.compute_inverse_kinematics(
                target_position, target_orientation
            )
        else:
            self.kinematics_solver._kinematics_solver.set_robot_base_pose(
                robot_base_translation, robot_base_orientation
            )
            actions, success = self.kinematics_solver._articulation_kinematics_solver.compute_inverse_kinematics(
                target_position, target_orientation
            )
        return success, actions

    def _move_to(self, target_positions, target_joint_indices, is_trajectory=False):
        if not self.articulation:
            return
        self._currentLeftTask = 10
        self._currentRightTask = 10
        actions = ArticulationAction(joint_positions=target_positions, joint_indices=target_joint_indices)
        if not is_trajectory:
            self.articulation.set_joint_positions(target_positions, joint_indices=target_joint_indices)
        else:
            self.reached = True
            self.articulation.apply_action(actions)

    def initialize_articulation(self, batch_num=0):
        scene = self.my_world.scene
        if scene._scene_registry.name_exists(self.robot_name):
            self.articulation = scene.get_object(self.robot_name)
        else:
            self.articulation = SingleArticulation(prim_path=self.robot_prim_path, name=self.robot_name)
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
            # articulation.set_solver_position_iteration_count(4)
            # articulation.set_solver_velocity_iteration_count(1)
            articulation.set_sleep_threshold(0.01)
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
                            current_position, rotation_matrix = self._get_ee_pose(False, is_local=True)
                        else:

                            current_position, rotation_matrix = self._get_ee_pose(True, is_local=True)
                        self.curoboMotion.init_ee_pose[value] = {
                            "position": current_position,
                            "orientation": rotation_matrix_to_quaternion(rotation_matrix),
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
            result = self.curoboMotion.attach_obj(prim_path_list, link_name, position, rotation)
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
        self.remove_objects()
        self.initialize_articulation()

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

    def remove_objects(self):
        parent_prim = get_prim_at_path("/World/Objects")
        if parent_prim.IsValid():
            prims = get_prim_children(parent_prim)
            for prim in prims:
                delete_prim(prim.GetPath())
