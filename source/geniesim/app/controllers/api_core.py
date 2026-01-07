# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
from typing import Tuple
import numpy as np
import threading
import queue
import json
from pathlib import Path
import asyncio
import subprocess
import signal
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdPhysics, PhysxSchema
import rclpy

import omni
import omni.usd
import omni.kit.commands
import omni.graph.core as og
import omni.replicator.core as rep
from omni.physx.scripts import utils, physicsUtils, particleUtils
from omni.kit.viewport.utility import get_active_viewport_and_window

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.api.materials import PhysicsMaterial, OmniPBR, OmniGlass
from isaacsim.core.prims import SingleXFormPrim, SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path, get_prim_object_type, delete_prim
from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.xforms import get_world_pose

from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance
from geniesim.app.utils import RobotCfg
from geniesim.app.utils import material_changer, Light
from geniesim.app.utils.utils import (
    get_rotation_matrix_from_quaternion,
    get_quaternion_from_euler,
    matrix_to_euler_angles,
    rotation_matrix_to_quaternion,
)
from geniesim.utils import system_utils
from geniesim.utils.usd_utils import *
from geniesim.utils.ros_nodes.server_node import *
from geniesim.config.params import Config
from geniesim.app.ros_publisher.base import USDBase
from geniesim.app.ros_publisher.robot_interface import RobotInterface
from geniesim.app.workflow.ui_builder import UIBuilder


class APICore:
    def __init__(self, ui_builder: UIBuilder, config: Config):
        context = rclpy.get_default_context()
        if not context.ok():
            rclpy.init()

        self.task_queue_on_render_loop = queue.Queue()
        self.task_queue_on_physics_loop = queue.Queue()
        self.benchmark_ros_node = None
        self.exit = False
        self.ui_builder: UIBuilder = ui_builder
        self.data = None
        self.Command = 0
        self.data_to_send = None
        self.gripper_state = ""
        self.condition = threading.Condition()
        self.result_queue = queue.Queue()
        self.target_position = np.array([0, 0, 0])
        self.target_rotation = np.array([0, 0, 0])
        self.recording_started = False
        self.task_name = None
        self.cameras = {}
        self.step_server = 0
        self.path_to_save_record = None
        self.object_prims = {
            "object_prims": [],
            "articulated_object_prims": [],
        }
        self.usd_objects = {}
        self.articulat_objects = {}
        self.trajectory_list = None
        self.trajectory_index = 0
        self.trajectory_reached = False
        self.target_joints_pose = []
        self.graph_path = []
        self.camera_graph_path = []
        self.loop_count = 0
        self.record_process = []
        self.target_point = None
        self.debug_view = {}
        self.gripper_cmd_r = None
        self.light_config = []
        self._lock = threading.Lock()
        self._physics_info = {}
        self._history_info = deque(maxlen=1000)
        self._on_play_back = False
        self._current_mode = "realtime"
        self._stage = omni.usd.get_context().get_stage()

        # app config
        self.enable_physics = not config.app.disable_physics
        self.enable_curobo = config.app.enable_curobo
        self.reset_fallen = config.app.reset_fallen
        self.rendering_step = config.app.rendering_step
        self.enable_ros = config.app.enable_ros
        self.record_images = config.app.record_img
        self.record_video = config.app.record_video
        self.data_convert = config.app.data_convert
        self.enable_playback = config.app.enable_playback

        # layout config
        self.seed = config.layout.seed
        self.autogen_ratio = config.layout.autogen_ratio
        self.num_obj = config.layout.num_obj

        # task config
        self.task_name = config.benchmark.task_name
        self.sub_task_name = config.benchmark.sub_task_name

        # robot data
        self.sensor_base = USDBase()
        self.robot_interface: RobotInterface = RobotInterface()
        if not self.enable_ros:
            self.robot_interface.disable_ros_pub()
        self.ros_node_initialized = False
        self.playback_timerange = []
        self.playback_start = 0
        self.playback_end = 0
        self.add_object_flag = False
        self.reset_flag = False
        self.init_frame_info = {}
        self.sensor_base_initialized = False
        self.robot_initialized = False
        self.index = 0

        # robot ros
        if not rclpy.get_default_context().ok():
            rclpy.init()

    def run_on_render_loop(self, func, *args, **kwargs):
        done = threading.Event()
        result = {}

        def wrapper():
            try:
                result["value"] = func(*args, **kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                done.set()

        self.task_queue_on_render_loop.put(wrapper)  # Put into queue, wait for main thread to process
        done.wait()

        if "error" in result:
            raise result["error"]
        return result.get("value")

    def run_on_physics_loop(self, func, *args, **kwargs):
        done = threading.Event()
        result = {}

        def wrapper():
            try:
                result["value"] = func(*args, **kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                done.set()

        self.task_queue_on_physics_loop.put(wrapper)  # Put into queue, wait for main thread to process
        done.wait()

        if "error" in result:
            raise result["error"]
        return result.get("value")

    def render_step(self):
        try:
            task = self.task_queue_on_render_loop.get_nowait()
        except queue.Empty:
            return
        task()

    def physics_step(self):
        try:
            task = self.task_queue_on_physics_loop.get_nowait()
        except queue.Empty:
            return
        task()

    ######################===================== New API BEGIN ===================================
    def init_robot_cfg(self, robot_cfg, scene_usd, init_position, init_rotation, sub_task_name=""):
        self.run_on_render_loop(self._init_robot_cfg, robot_cfg, scene_usd, init_position, init_rotation, sub_task_name)

    def get_obj_world_pose_matrix(self, prim_path, camera=False):
        rotation_x_180 = np.array([[1.0, 0.0, 0.0, 0], [0.0, -1.0, 0.0, 0], [0.0, 0.0, -1.0, 0], [0, 0, 0, 1]])
        position, rotation = self.get_obj_world_pose(prim_path)
        x, y, z = position
        rw, rx, ry, rz = rotation
        quat_wxyz = np.array(
            [
                rw,
                rx,
                ry,
                rz,
            ]
        )
        rot_mat = get_rotation_matrix_from_quaternion(quat_wxyz)

        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = np.array([x, y, z])

        if camera:
            pose = pose @ rotation_x_180
        return pose

    def get_obj_world_pose(self, prim_path):
        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            return (0, 0, 0), (0, 0, 0, 0)

        if prim_path == "robot":
            position, rotation = self.usd_objects["robot"].get_world_pose()
        else:
            # Check if there's an 'entity' child prim
            entity_path = prim_path + "/entity"
            entity_prim = stage.GetPrimAtPath(entity_path)
            if entity_prim.IsValid():
                # Use entity layer for position and rotation
                position, rotation = get_world_pose(entity_path)
            else:
                position, rotation = get_world_pose(prim_path)
        return position, rotation

    def get_obj_aabb(self, prim_path):
        cache = create_bbox_cache()
        aabb = compute_aabb(cache, prim_path=prim_path)
        return aabb

    def get_obj_joint(self, prim_path):
        if prim_path not in self.articulat_objects.keys():
            self.articulat_objects[prim_path] = SingleArticulation(prim_path)
        self.articulat_objects[prim_path].initialize()
        dof_names = self.articulat_objects[prim_path].dof_names
        positions = self.articulat_objects[prim_path].get_joint_positions()
        velocities = self.articulat_objects[prim_path].get_joint_velocities()
        return {
            "joint_names": dof_names,
            "joint_positions": positions,
            "joint_velocities": velocities,
        }

    def get_releated_objs(self):
        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath("/World/Objects")
        if not root_prim.IsValid():
            return []

        prims = [str(child.GetPath()) for child in root_prim.GetChildren()]
        return prims

    def set_joint_positions(self, target_pose, joint_indices, is_trajectory):
        self.run_on_physics_loop(self._set_joint_positions, target_pose, joint_indices, is_trajectory)

    def add_usd_obj(
        self,
        usd_path,
        prim_path,
        label_name,
        position,
        rotation,
        scale,
        object_color,
        object_material,
        object_mass,
        add_particle,
        particle_position,
        particle_scale,
        particle_color,
        object_com,
        model_type,
        static_friction,
        dynamic_friction,
    ):
        self.run_on_render_loop(
            self._add_usd_object,
            usd_path,
            prim_path,
            label_name,
            position,
            rotation,
            scale,
            object_color,
            object_material,
            object_mass,
            add_particle,
            particle_position,
            particle_scale,
            particle_color,
            object_com,
            model_type,
            static_friction,
            dynamic_friction,
        )

    def add_object(self, usd_path, prim_path, translation, rotation, mass=0.2):
        """
        Simply add an object with mass (mainly used for auto-generating layout scenes)

        - usd_path: Absolute path to USD asset
        - prim_path: Prim path to place in scene (e.g., /World/Objects/cup_01)
        - translation: Position in world coordinates [x, y, z]
        - rotation: XYZ Euler angles (degrees)
        - mass: Object mass (kg)
        """
        self.run_on_render_loop(self._add_object, usd_path, prim_path, translation, rotation, mass)

    def set_light(
        self,
        light_type,
        light_prim,
        light_temperature,
        light_intensity,
        light_rotation,
        light_texture,
    ):
        self.run_on_render_loop(
            self._set_light,
            light_type,
            light_prim,
            light_temperature,
            light_intensity,
            light_rotation,
            light_texture,
        )

    def reset(self):
        self.run_on_render_loop(self._on_reset)

    def stop(self):
        self.exit = True

    def post_process(self):
        pass

    def start_recording(self, camera_prim_list, fps, extra_prim_paths, record_topic_list):
        self.run_on_render_loop(
            self._start_recording,
            camera_prim_list,
            fps,
            extra_prim_paths,
            record_topic_list,
        )

    def stop_recording(self):
        self.run_on_render_loop(self._stop_recording)

    def collect_init_physics(self):
        self.run_on_render_loop(self._collect_init_physics)

    def reset_env(self):
        self.run_on_render_loop(self._reset_env)

    def shuffle_scene(self):
        """Randomly adjust x and y positions of rigid body objects in the scene"""
        self.run_on_render_loop(self._shuffle_scene)

    def get_joint_state_dict(self):
        return self.run_on_physics_loop(self._get_joint_state_dict)

    def get_observation_image(self, dir):
        return self.run_on_physics_loop(self._get_observation_image, dir)

    def count_visible_meshes(self, prim_path: str):
        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            return 0

        count = 0
        for prim in Usd.PrimRange(root_prim):
            if prim.GetTypeName() == "Mesh":
                geom = UsdGeom.Imageable(prim)
                visibility = geom.GetVisibilityAttr().Get()
                if visibility == "inherited":
                    count += 1
        return count

    def get_trigger_action(self, prim_path: str):
        return str(og.Controller.attribute(prim_path).get())

    ######################===================== New API END ======================================

    ######################===================== Private Methods BEGIN ===================================

    def _collect_init_physics(self):
        robot_articulation = self._get_articulation()
        self._physics_info = {}
        collect_physics(self._physics_info)
        if robot_articulation:
            self.init_frame_info = store_init_physics(robot_articulation, self._physics_info)

    def _get_joint_state_dict(self):
        return self.robot_interface.get_joint_state_dict()

    def _get_observation_image(self, dir):
        return self.robot_interface.get_observation_image(dir)

    def _play(self):
        self.ui_builder.my_world.play()
        self._init_robot(self.robot_cfg, self.enable_curobo)

        self.frame_status = []

    def _init_robot(self, robot: RobotCfg, enable_curobo):
        self.robot_name = robot.robot_name
        self.robot_prim_path = robot.robot_prim_path
        self.end_effector_prim_path = robot.end_effector_prim_path
        self.end_effector_name = robot.end_effector_name

        self.finger_names = robot.finger_names
        self.gripper_names = [robot.left_gripper_name, robot.right_gripper_name]
        self.gripper_control_joint = robot.gripper_control_joint
        self.opened_positions = robot.opened_positions
        self.closed_velocities = robot.closed_velocities
        self.cameras = robot.cameras
        self.is_single_gripper = robot.is_single
        self.gripper_type = robot.gripper_type
        self.gripper_max_force = robot.gripper_max_force
        self.init_joint_position = robot.init_joint_position
        self.ui_builder._init_solver(robot, enable_curobo, 0)
        self.past_position = [0, 0, 0]
        self.past_rotation = [1, 0, 0, 0]

        self.robot_interface.register_joint_state(self._get_articulation())
        self.robot_interface.register_robot_tf(self._stage, self.robot_prim_path)
        if robot.perception:
            self.robot_interface.register_perception(self._stage, self.robot_prim_path)
        # cams
        for camera in self.robot_cfg.cameras:
            self.robot_interface.register_camera(camera, self.robot_cfg.cameras[camera], 1)
        if self.enable_ros and not self.ros_node_initialized:
            self.server_ros_node = ServerNode(robot_name=self.robot_name)
            # joint_states
            logger.info(f"sensor_ros.publish_joint {self.robot_prim_path} {self.robot_name}")
            self.sensor_base.publish_joint(robot_prim=self.robot_prim_path)
            self.ros_node_initialized = True

    def _init_robot_cfg(
        self,
        robot_cfg,
        scene_usd,
        init_position=[0, 0, 0],
        init_rotation=[1, 0, 0, 0],
        sub_usd_path="",
    ):
        ws_prim = get_prim_at_path("/Workspace")
        if sub_usd_path != "":
            b_return = False
            if ws_prim.IsValid():
                delete_prim(ws_prim.GetPath())
                b_return = True
            add_reference_to_stage(
                sub_usd_path,
                "/Workspace",
            )
            if b_return:
                return
        self.robot_cfg = RobotCfg(str(system_utils.app_root_path()) + "/robot_cfg/" + robot_cfg)
        robot_usd_path = str(system_utils.assets_path()) + "/" + self.robot_cfg.robot_usd
        scene_usd_path = str(system_utils.assets_path()) + "/" + scene_usd
        add_reference_to_stage(robot_usd_path, self.robot_cfg.robot_prim_path)
        add_reference_to_stage(scene_usd_path, "/World")
        self.usd_objects["robot"] = SingleXFormPrim(
            prim_path=self.robot_cfg.robot_prim_path,
            position=init_position,
            orientation=init_rotation,
        )
        self.robot_init_position = init_position
        self.robot_init_rotation = init_rotation
        self.scene_usd = scene_usd
        self.scene_glb = os.path.join(os.path.dirname(scene_usd), "compressed_simplified.glb")
        if "multispace" in scene_usd:
            self.scene_name = scene_usd.split("/")[-3] + "/" + scene_usd.split("/")[-2]
        else:
            self.scene_name = scene_usd.split("/")[-2]
        self.robot_name = self.robot_cfg.robot_name
        self.material_changer = material_changer()
        # physics_scene settings
        self.scene = UsdPhysics.Scene.Define(self._stage, Sdf.Path("/physicsScene"))
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        self.scene.CreateGravityMagnitudeAttr().Set(9.81)
        physics_scene = PhysxSchema.PhysxSceneAPI.Get(self._stage, "/physicsScene")
        physics_scene.CreateGpuMaxRigidContactCountAttr(8388608)
        physics_scene.CreateGpuMaxRigidPatchCountAttr(163840)
        physics_scene.CreateGpuFoundLostPairsCapacityAttr(2097152)
        physics_scene.CreateGpuFoundLostAggregatePairsCapacityAttr(33554432)
        physics_scene.CreateGpuTotalAggregatePairsCapacityAttr(2097152)

        with rep.get.prims(path_pattern=self.robot_cfg.robot_prim_path, prim_types=["Xform"]):
            rep.modify.semantics([("class", "robot")])

        viewport, window = get_active_viewport_and_window()
        # Set camera based on robot type
        viewport.set_active_camera("/G1/head_link2/Head_Camera")
        if "G2" in self.robot_name:
            viewport.set_active_camera("/G2/head_link3/head_front_Camera")
        time.sleep(1)
        self._play()

    def _set_joint_positions(self, target_pose, target_joint_indices, is_trajectory):
        if not len(self.target_joints_pose):
            for idx, value in enumerate(self.robot_interface.get_joint_state()):
                if idx in target_joint_indices:
                    self.target_joints_pose.append(value)
        diff = np.asarray(self.target_joints_pose) - np.asarray(target_pose)
        if np.linalg.norm(diff) != 0:
            self.target_joints_pose = target_pose
            self._joint_moveto(
                target_pose,
                target_joint_indices=target_joint_indices,
                is_trajectory=is_trajectory,
            )
        if not is_trajectory:
            self.target_joints_pose = []
        else:
            if self.ui_builder.reached:
                self.target_joints_pose = []

    # Add objects
    def _add_usd_object(
        self,
        usd_path: str,
        prim_path: str,
        label_name: str,
        position,
        rotation,
        scale,
        object_color,
        object_material,
        object_mass,
        add_particle=False,
        particle_position=[0, 0, 0],
        particle_scale=[0.1, 0.1, 0.1],
        particle_color=[1, 1, 1],
        object_com=[0, 0, 0],
        model_type="convexDecomposition",
        static_friction=1.0,
        dynamic_friction=1.0,
    ):
        usd_path = os.path.join(system_utils.assets_path(), usd_path)
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        if add_particle:
            particle_pos = [
                position[0] + particle_position[0],
                position[1] + particle_position[1],
                position[2] + particle_position[2],
            ]
            self._add_particle(particle_pos, particle_scale)
        usd_object = SingleXFormPrim(prim_path=prim_path, position=position, orientation=rotation, scale=scale)
        type = get_prim_object_type(prim_path)
        items = []
        if self._stage:
            for prim in Usd.PrimRange(self._stage.GetPrimAtPath(prim_path)):
                path = str(prim.GetPath())
                prim = get_prim_at_path(path)
                if prim.IsA(UsdGeom.Mesh):
                    items.append(path)
        object_rep = rep.get.prims(path_pattern=prim_path, prim_types=["Xform"])

        with object_rep:
            rep.modify.semantics([("class", label_name)])
        if type == "articulation":
            self.ui_builder.my_world.play()
            articulation = SingleArticulation(prim_path)
            articulation.initialize()
            self.articulat_objects[prim_path] = articulation
            self.usd_objects[prim_path] = usd_object
        else:
            self.usd_objects[prim_path] = usd_object
            self.object_prims["object_prims"].append(prim_path)
            for _prim in items:
                geometry_prim = SingleGeometryPrim(prim_path=_prim)
                obj_physics_prim_path = f"{_prim}/object_physics"
                geometry_prim.apply_physics_material(
                    PhysicsMaterial(
                        prim_path=obj_physics_prim_path,
                        static_friction=static_friction,
                        dynamic_friction=dynamic_friction,
                        restitution=0.1,
                    )
                )
                # set friction combine mode to max to enable stable grasp
                obj_physics_prim = self._stage.GetPrimAtPath(obj_physics_prim_path)
                physx_material_api = PhysxSchema.PhysxMaterialAPI(obj_physics_prim)
                if physx_material_api is not None:
                    fric_combine_mode = physx_material_api.GetFrictionCombineModeAttr().Get()
                    if fric_combine_mode == None:
                        physx_material_api.CreateFrictionCombineModeAttr().Set("max")
                    elif fric_combine_mode != "max":
                        physx_material_api.GetFrictionCombineModeAttr().Set("max")

                if object_material != "general":
                    if object_material == "Glass":
                        material_prim = "/World/G1_video/Looks_01/OmniGlass"
                        material = OmniGlass(prim_path=material_prim)
                        geometry_prim.apply_visual_material(material)
                    elif object_material not in self.materials:
                        material_prim = prim_path + "/Looks/DefaultMaterial"
                        material = OmniPBR(
                            prim_path=material_prim,
                            color=object_color,
                        )
                        material.set_metallic_constant(1)
                        material.set_reflection_roughness(0.4)
                        geometry_prim.apply_visual_material(material)
                    else:
                        Material = self.materials[object_material]
                        prim = self._.GetPrimAtPath(_prim)
                        UsdShade.MaterialBindingAPI(prim).Bind(Material)

            prim = self._stage.GetPrimAtPath(prim_path)
            if model_type != "None":
                utils.setRigidBody(prim, model_type, False)
            rigid_prim = SingleRigidPrim(prim_path=prim_path)
            # Get Physics API
            physics_api = UsdPhysics.MassAPI.Apply(rigid_prim.prim)
            physics_api.CreateMassAttr().Set(object_mass)

    def _add_object(self, usd_path, prim_path, translation, rotation=[90, 0, 0], mass=0.01):
        # Validate parameters
        if not usd_path:
            raise ValueError(f"usd_path cannot be empty or None: {repr(usd_path)}")
        if not isinstance(usd_path, str) or not usd_path.strip():
            raise ValueError(f"usd_path must be a non-empty string: {repr(usd_path)}")

        if not prim_path:
            raise ValueError(f"prim_path cannot be empty or None: {repr(prim_path)}")
        if not isinstance(prim_path, str) or not prim_path.strip():
            raise ValueError(f"prim_path must be a non-empty string: {repr(prim_path)}")
        if not prim_path.startswith("/"):
            raise ValueError(f"prim_path must be an absolute path (starting with '/'): {repr(prim_path)}")

        # Log the actual parameters passed
        logger.info(f"_add_object call: usd_path={repr(usd_path)}, prim_path={repr(prim_path)}, mass={mass}")

        stage = omni.usd.get_context().get_stage()
        # Add obj
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        # Set transform
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Prim '{prim_path}' invalid")

        # Convert to Xformable type and set transform
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()  # Clear existing transforms
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform.AddRotateXYZOp().Set(Gf.Vec3d(*rotation))

        # Set rigid body and mass to ensure the object participates in physics simulation
        try:
            # Set the prim as a rigid body (using the same default collision type as add_usd_obj)
            utils.setRigidBody(prim, "convexDecomposition", False)

            rigid_prim = SingleRigidPrim(prim_path=prim_path)
            physics_api = UsdPhysics.MassAPI.Apply(rigid_prim.prim)
            physics_api.CreateMassAttr().Set(float(mass))
        except Exception as e:
            logger.error(f"Failed to set mass for prim {prim_path}: {e}")

    def _set_light(
        self,
        light_type,
        light_prim,
        light_temperature,
        light_intensity,
        light_rotation,
        light_texture,
    ):
        light = Light(
            light_type=light_type,
            prim_path=light_prim,
            stage=self._stage,
            intensity=light_intensity,
            color=light_temperature,
            orientation=light_rotation,
            texture_file=light_texture,
        )
        light.initialize()

    def _on_reset(self):
        logger.info("api_core reset.")
        self.ui_builder._on_reset()
        self.usd_objects = {}
        self.target_position = [0, 0, 0]
        self.articulat_objects = {}
        self.frame_status = []

    def _start_recording(self, camera_prim_list, fps, extra_prim_paths, record_topic_list):
        pass

    def _stop_recording(self):
        pass

    def _reset_env(self):
        robot_articulation = self._get_articulation()
        if robot_articulation:
            reset_one_frame(robot_articulation, self.init_frame_info)
            self.reset_flag = False

    def _shuffle_scene(self):
        """Traverse objects with rigidbody in the scene and randomly adjust x and y positions (±0.1 range)"""
        stage = get_current_stage()
        if not stage:
            logger.warning("Stage not available for shuffle")
            return

        # Get all rigid body objects
        rigidbody_prims = []
        # Define path prefixes to exclude
        excluded_paths = [
            self.robot_prim_path if hasattr(self, "robot_prim_path") else "/G1",
            "/OmniverseKit_Persp",
            "/OmniverseKit_Front",
            "/OmniverseKit_Top",
            "/OmniverseKit_Right",
            "/Render",
            "/Environment",
            "/Workspace/Camera",
        ]

        for prim in stage.Traverse():
            prim_path = str(prim.GetPrimPath())

            # Exclude system paths and robot
            should_exclude = False
            for excluded_path in excluded_paths:
                if prim_path.startswith(excluded_path):
                    should_exclude = True
                    break

            if should_exclude:
                continue

            # Check if it has RigidBodyAPI (rigid body object)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                # Further exclude cameras, lights, etc.
                prim_type = prim.GetTypeName()
                if prim_type not in [
                    "Camera",
                    "DistantLight",
                    "DomeLight",
                    "SphereLight",
                    "RectLight",
                ]:
                    rigidbody_prims.append(prim_path)

        logger.info(f"Found {len(rigidbody_prims)} rigidbody objects to shuffle")

        # Randomize position for each rigid body object
        for prim_path in rigidbody_prims:
            try:
                # Get current position
                current_position, current_rotation = get_world_pose(prim_path)

                # Randomly adjust x and y within ±0.1 range
                x_offset = np.random.uniform(-0.1, 0.1)
                y_offset = np.random.uniform(-0.1, 0.1)

                new_position = np.array(
                    [
                        current_position[0] + x_offset,
                        current_position[1] + y_offset,
                        current_position[2],  # z position remains unchanged
                    ]
                )

                # Update object position
                if prim_path in self.usd_objects:
                    # If object is in usd_objects, use its interface
                    self.usd_objects[prim_path].set_world_pose(position=new_position, orientation=current_rotation)
                    logger.info(f"Shuffled {prim_path}: offset=({x_offset:.3f}, {y_offset:.3f})")
                else:
                    # Otherwise use SingleXFormPrim interface to set position
                    try:
                        # Create or get XFormPrim object
                        xform_prim = SingleXFormPrim(prim_path=prim_path)
                        xform_prim.set_world_pose(position=new_position, orientation=current_rotation)
                        logger.info(f"Shuffled {prim_path}: offset=({x_offset:.3f}, {y_offset:.3f})")
                    except Exception as e:
                        logger.warning(f"Failed to use SingleXFormPrim for {prim_path}: {e}, trying alternative method")
                        # Fallback: use physics utilities to set position
                        prim = stage.GetPrimAtPath(prim_path)
                        if prim.IsValid():
                            physicsUtils.set_or_add_translate_op(prim, Gf.Vec3f(*new_position))
                            logger.info(f"Shuffled {prim_path}: offset=({x_offset:.3f}, {y_offset:.3f})")
            except Exception as e:
                logger.error(f"Failed to shuffle {prim_path}: {e}")

        logger.info("Scene shuffle completed")

    ######################===================== Private Methods END ===================================

    def process_recording_path(self):
        recording_path = os.path.join(system_utils.recording_output_path(), self.task_name)
        if os.path.isdir(recording_path):
            folder_index = 1
            while os.path.isdir(os.path.join(recording_path, str(folder_index))):
                folder_index += 1
            recording_path = os.path.join(recording_path, str(folder_index))
        self.path_to_save_record = recording_path

    def process_camera_info_list(self):
        self.camera_info_list = {}
        for prim in self.camera_prim_list:
            image = self._capture_camera(
                prim_path=prim,
                isRGB=False,
                isDepth=False,
                isSemantic=False,
                isGN=False,
            )
            prim_name = prim.split("/")[-1]
            if "G1" in self.robot_name:
                if "Fisheye_Camera" in prim_name:
                    prim_name = "head_right_fisheye"
                elif "Fisheye_Camera" in prim_name:
                    prim_name = "head_left_fisheye"
                elif "Fisheye_Back" in prim_name:
                    prim_name = "back_left_fisheye"
                elif "Fisheye_Back" in prim_name:
                    prim_name = "back_right_fisheye"
                elif "head" in prim_name:
                    prim_name = "head"
                elif "right" in prim_name:
                    prim_name = "hand_right"
                elif "left" in prim_name:
                    prim_name = "hand_left"
                elif "top" in prim_name:
                    prim_name = "head_front_fisheye"
            self.camera_info_list[prim_name] = {
                "intrinsic": image["camera_info"],
                "output": {
                    "rgb": "camera/" + "{frame_num}/" + f"{prim_name}.jpg",
                    "video": f"{prim_name}.mp4",
                },
            }
            if "fisheye" not in prim_name:
                self.camera_info_list[prim_name]["output"]["depth"] = (
                    "camera/" + "{frame_num}/" + f"{prim_name}_depth.png"
                )

            if self.data["render_semantic"]:
                self.camera_info_list[prim_name]["output"]["semantic"] = (
                    "camera/" + "{frame_num}/" + f"{prim_name}_semantic.png"
                )

    def _init_gripper_contact_end(self):
        if "omnipicker" in self.robot_cfg.robot_usd:
            self.gripper_contact_ends = [
                "/G1/gripper_r_inner_link4",
                "/G1/gripper_r_outer_link4",
            ]
        elif "120s" in self.robot_cfg.robot_usd:
            self.gripper_contact_ends = [
                "/G1/gripper_r_inner_link5",
                "/G1/gripper_r_outer_link5",
            ]
        else:
            raise ("Undefined robot")

    def on_ros_tick(self, step_size):
        if not self.enable_ros:
            return

        if self.ros_node_initialized:
            rclpy.spin_once(self.server_ros_node, timeout_sec=0)
            rclpy.spin_once(self.robot_interface, timeout_sec=0)
            rclpy.spin_once(self.benchmark_ros_node, timeout_sec=0)

            # main sim clock source here
            self.robot_interface.tick(
                self.ui_builder.my_world.current_time,
                self.ui_builder.my_world.current_time_step_index,
            )
            self.server_ros_node.publish_clock(self.ui_builder.my_world.current_time)
            self._on_play_back = self.server_ros_node.get_playback_state()

    def on_playback(self):
        robot_articulation = self._get_articulation()

        if robot_articulation:
            # playback
            if not self._physics_info or self.add_object_flag:
                collect_physics(self._physics_info)
            if self._on_play_back and self._current_mode == "realtime":
                disable_physics(self._physics_info)
                self._current_mode = "playback"

            if self._current_mode == "playback" and not self._on_play_back:
                restore_physics(self._physics_info)
                # Who knows ehy?
                disable_physics(self._physics_info)
                restore_physics(self._physics_info)
                self._current_mode = "realtime"
                self.playback_end = self.ui_builder.my_world.current_time
                self.playback_timerange.append([self.playback_start, self.playback_end])

            if self._current_mode == "playback":
                logger.info("In playback mode")
                playback_timestamp = playback_once(robot_articulation, self._history_info)
                # udpate timestamp
                if playback_timestamp > 0:
                    self.playback_start = playback_timestamp
            else:
                # store history
                store_history_physics(
                    robot_articulation,
                    self._physics_info,
                    self._history_info,
                    self.ui_builder.my_world.current_time,
                )
            return self._current_mode == "playback"
        return False

    # 1. Photo capturing function, prim path of Input camera in isaac side scene and whether to use Gaussian Noise, return
    def _capture_camera(self, prim_path: str, isRGB, isDepth, isSemantic, isGN: bool):
        self.ui_builder._currentCamera = prim_path
        self.ui_builder._on_capture_cam(isRGB, isDepth, isSemantic)
        currentImage = self.ui_builder.currentImg
        return currentImage

    def _get_articulation(self):
        return self.ui_builder.articulation

    # 3. The whole body joints move to the specified angle, Input:np.array([None])*28
    def _joint_moveto(self, joint_position, is_trajectory, target_joint_indices):
        self._get_articulation()
        self.ui_builder._move_to(joint_position, target_joint_indices, is_trajectory)

    def _set_object_joint(self, prim_path, target_positions):
        self.articulat_objects[prim_path].initialize()
        self.articulat_objects[prim_path].set_joint_positions(target_positions)

    def get_particle_pt_num_inbbox(self, prim_path, bbox_3d):
        stage = get_current_stage()
        point_set_prim = stage.GetPrimAtPath(prim_path)
        points = UsdGeom.Points(point_set_prim).GetPointsAttr().Get()
        points_position = np.array(points)

        # Determine whether the point set is in a bounding box
        def points_in_bbox(points, bbox):
            # Define bounding box
            xmin, ymin, zmin, xmax, ymax, zmax = bbox
            # Use boolean index to determine whether the point is in the bounding box
            inside = np.all((points >= [xmin, ymin, zmin]) & (points <= [xmax, ymax, zmax]), axis=1)
            return points[inside]

        points_in_bbox_3d = points_in_bbox(points_position, bbox_3d)
        return len(points_in_bbox_3d)

    def _add_particle(self, position, size):
        stage = get_current_stage()
        particle_system_path = Sdf.Path("/World/Objects/part/particleSystem")
        if stage.GetPrimAtPath(particle_system_path):
            return

        # create a scene with gravity and up axis:
        scene = self.scene
        Particle_Contact_Offset = 0.004
        Sample_Volume = 1
        particle_system = particleUtils.add_physx_particle_system(
            stage,
            particle_system_path,
            particle_system_enabled=True,
            simulation_owner=scene.GetPath(),
            particle_contact_offset=Particle_Contact_Offset,
            max_velocity=0.3,
        )
        # create particle material and assign it to the system:
        particle_material_path = Sdf.Path("/World/Objects/part/particleMaterial")
        particleUtils.add_pbd_particle_material(
            stage,
            particle_material_path,
            friction=0.0,
            density=1.0,
            viscosity=0.0091,
            cohesion=0.01,
            surface_tension=0.0074,
            drag=0.0,
            lift=0.0,
        )  # Set the viscosity.

        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath(particle_system_path), particle_material_path
        )
        cube_mesh_path = Sdf.Path("/World/Objects/part/Cube")
        cube_resolution = (
            20  # resolution can be low because we'll sample the surface / volume only irrespective of the vertex count
        )
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type="Cylinder",
            u_patches=cube_resolution,
            v_patches=cube_resolution,
            prim_path=cube_mesh_path,
        )
        cube_mesh = UsdGeom.Mesh.Get(stage, cube_mesh_path)
        physicsUtils.set_or_add_translate_op(cube_mesh, Gf.Vec3f(position[0], position[1], position[2]))
        physicsUtils.set_or_add_scale_op(cube_mesh, Gf.Vec3f(size[0], size[1], size[2]))
        particle_points_path = Sdf.Path("/World/Objects/part/sampledParticles")
        points = UsdGeom.Points.Define(stage, particle_points_path)
        point_prim = stage.GetPrimAtPath(particle_points_path)
        visibility_attribute = point_prim.GetAttribute("visibility")
        if visibility_attribute is not None:
            visibility_attribute.Set("invisible")
        geometry_prim = SingleGeometryPrim(prim_path="/World/Objects/part/particleSystem")
        material_prim = "/World/Looks_01/OmniGlass"
        material = OmniGlass(prim_path=material_prim, color=np.array([0.645, 0.271, 0.075]))
        geometry_prim.apply_visual_material(material)
        particle_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(points.GetPrim())
        PhysxSchema.PhysxParticleAPI(particle_set_api).CreateParticleSystemRel().SetTargets([particle_system_path])
        fluid_rest_offset = 0.99 * 0.6 * Particle_Contact_Offset
        particle_sampler_distance = 2.0 * fluid_rest_offset
        sampling_api = PhysxSchema.PhysxParticleSamplingAPI.Apply(cube_mesh.GetPrim())
        sampling_api.CreateParticlesRel().AddTarget(particle_points_path)
        sampling_api.CreateSamplingDistanceAttr().Set(particle_sampler_distance)
        sampling_api.CreateMaxSamplesAttr().Set(5e5)
        sampling_api.CreateVolumeAttr().Set(Sample_Volume)
        particleUtils.add_physx_particle_isosurface(stage, particle_system_path, enabled=True)
        self.ui_builder.my_world.stop()
        self._play()

    def _set_object_material(self, prim_path, material_name, material_path, label_name=None):
        items = []
        logger.info(label_name)
        if label_name:
            object_rep = rep.get.prims(path_pattern=prim_path, prim_types=["Xform"])
            with object_rep:
                rep.modify.semantics([("class", label_name)])
        if not self._stage:
            return
        if "Glass" in material_name or "glass" in material_name:
            material_prim = "/World/Materials/OmniGlass"
            material = OmniGlass(prim_path=material_prim)
            for prim in Usd.PrimRange(self._stage.GetPrimAtPath(prim_path)):
                path = str(prim.GetPath())
                prim = get_prim_at_path(path)
                if prim.IsA(UsdGeom.Mesh) or prim.GetTypeName() in "GeomSubset":
                    geometry_prim = SingleGeometryPrim(prim_path=path)
                    geometry_prim.apply_visual_material(material)

        else:
            material = self.material_changer.assign_material(material_path, material_name)
            for prim in Usd.PrimRange(self._stage.GetPrimAtPath(prim_path)):
                path = str(prim.GetPath())
                prim = get_prim_at_path(path)
                if prim.IsA(UsdGeom.Mesh) or prim.GetTypeName() in "GeomSubset":
                    UsdShade.MaterialBindingAPI(prim).Bind(material)

    def _find_all_objects_of_type(self, obj_type):
        items = []
        if self._stage:
            for prim in Usd.PrimRange(self._stage.GetPrimAtPath("/")):
                path = str(prim.GetPath())
                type = get_prim_object_type(path)
                if type == obj_type:
                    items.append(path)
        return items

    def _get_ik_status(self, target_position, target_rotation, isRight, ObsAvoid=False):
        SingleXFormPrim("/ik_pos", position=target_position)
        joint_positions = {}
        if not ObsAvoid:
            is_success, joint_state = self.ui_builder._get_ik_status(target_position, target_rotation, isRight)
            joint_names = []
            all_names = self.ui_builder.articulation.dof_names
            for i, idx in enumerate(joint_state.joint_indices):
                joint_positions[all_names[idx]] = joint_state.joint_positions[i]
        else:
            init_rotation_matrix = get_rotation_matrix_from_quaternion(self.robot_init_rotation)
            translation_matrix = np.zeros((4, 4))
            translation_matrix[:3, :3] = init_rotation_matrix
            translation_matrix[:3, 3] = self.robot_init_position
            translation_matrix[3, 3] = 1
            target_rotation_world = get_rotation_matrix_from_quaternion(target_rotation)
            target_matrix_world = np.zeros((4, 4))
            target_matrix_world[:3, :3] = target_rotation_world
            target_matrix_world[:3, 3] = target_position
            target_matrix_world[3, 3] = 1
            target_matrix = np.linalg.inv(translation_matrix) @ target_matrix_world
            target_rotation_matrix, target_position_local = (
                target_matrix[:3, :3],
                target_matrix[:3, 3],
            )
            target_rotation_local = get_quaternion_from_euler(
                matrix_to_euler_angles(target_rotation_matrix), order="ZYX"
            )
            if isinstance(self.end_effector_name, dict):
                end_effector_name = self.end_effector_name["left"]
                if isRight:
                    end_effector_name = self.end_effector_name["right"]
            else:
                end_effector_name = self.end_effector_name
            is_success, joint_state = self.ui_builder.curoboMotion.solve_batch_ik(
                target_position_local, target_rotation_local, end_effector_name
            )
            for i, name in enumerate(joint_state.joint_names):
                joint_positions[name] = joint_state.position[0][0].cpu().tolist()[i]

        return is_success, joint_positions
