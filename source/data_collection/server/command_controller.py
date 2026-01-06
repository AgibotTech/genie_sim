import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Tuple

import numpy as np
import omni
import omni.replicator.core as rep
import omni.timeline
import rclpy
import yaml
from isaacsim.core.api.materials import OmniGlass, OmniPBR, PhysicsMaterial
from isaacsim.core.api.objects import cuboid, cylinder
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim
from isaacsim.core.prims import SingleRigidPrim as RigidPrim
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.prims import get_prim_at_path, get_prim_object_type
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from omni.kit.viewport.utility import get_active_viewport_and_window
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from omni.physx.scripts import utils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from common.base_utils.logger import logger
from common.base_utils.ros_nodes.server_node import ServerNode
from common.base_utils.transform_utils import mat2quat_wxyz, quat2mat_wxyz
from common.data_filter.runtime_checker import CheckerStatus, create_checker
from server.command_enum import Command, command_value_to_string
from server.controllers.parallel_gripper import ParallelGripper
from server.material_changer import Light, material_changer
from server.robot import RobotCfg
from server.ros_publisher.base import USDBase
from server.ui_builder import UIBuilder
from server.utils import batch_matrices_to_quaternions_scipy_w_first

MAX_EXTRACT_PROCESS_NUM = 2


def find_joints(prim):
    joint_paths = []
    # Check if current Prim is a joint
    if prim.IsA(UsdPhysics.Joint):
        joint_paths.append(prim.GetPath().pathString)
    # Recursively traverse child Prims
    for child_prim in prim.GetChildren():
        joint_paths.extend(find_joints(child_prim))
    return joint_paths


class CommandController:
    def __init__(
        self,
        ui_builder: UIBuilder,
        enable_physics=False,
        enable_curobo=False,
        publish_ros=False,
        rendering_step=60,
        debug=False,
    ):
        self.sim_assets_root = os.environ.get("SIM_ASSETS")
        self.ui_builder = ui_builder
        self.debug = debug
        self.data = None
        self.Command = 0
        self.data_to_send = None
        self.gripper_L = None
        self.gripper_R = None
        self.gripper_state_L = ""
        self.gripper_state_R = ""
        self.gripper_state = ""
        self.condition = threading.Condition()
        self.result_queue = queue.Queue()
        self.target_position = np.array([0, 0, 0])
        self.target_rotation = np.array([0, 0, 0])
        self.target_joints_pose = None
        self.task_name = None
        self.cameras = {}
        self.step = 0
        self.path_to_save = None
        self.exit = False
        self.object_asset_dict = {}
        self.usd_objects = {}
        self.articulat_objects = {}
        self.rigid_bodies = {}
        self.enable_physics = enable_physics
        self.enable_curobo = enable_curobo
        self.trajectory_list = None
        self.trajectory_index = 0
        self.trajectory_reached = False
        self.target_joints_pose = []
        self.graph_path = []
        self.camera_graph_path = []
        self.loop_count = 0
        self.publish_ros = publish_ros
        self.rendering_step = rendering_step
        self.process = []
        self.extract_process = []
        self.target_point = None
        self.debug_view = {}
        self.timeline = omni.timeline.get_timeline_interface()
        self.light_config = []
        self.attached_joints = {}
        self.write_semantic = False
        self.motion_run_ratio = 1.0
        self.gripper_action_timing = None
        self.object_code_dict = {}
        self.task_description = {
            "task_name": "",
            "english_task_name": "",
            "init_scene_text": "",
        }
        self.task_metric = {}
        self.sensor_base = USDBase()
        self.ros_node_initialized = False
        self.ros_step = 0  # control pub hz
        self.scene_usd = ""
        self.playback_timerange = []
        self.ros_publishers = []
        self.dof_names = []
        self.playback_frames = {}
        self.playback_waited_frame_num = 0
        self.attach_states = {}
        self.camera_info_list = {}
        self.fps = 60
        self.cur_runtime_checker = None
        # Timing statistics related
        self.timing_stats = {}  # Store total time for each function {function_name: total_time}
        self.timing_lock = threading.Lock()  # For thread-safe timing statistics
        # ros
        if publish_ros:
            rclpy.init()

    def _timing_context(self, function_name: str):
        """Timing context manager for counting function execution time"""

        class TimingContext:
            def __init__(self, controller, name):
                self.controller = controller
                self.enable = True
                self.name = name
                self.start_time = None

            def __enter__(self):
                if self.enable:
                    self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.enable:
                    elapsed_time = time.time() - self.start_time
                    with self.controller.timing_lock:
                        if self.name not in self.controller.timing_stats:
                            self.controller.timing_stats[self.name] = {
                                "total_time": 0.0,
                                "call_count": 0,
                                "avg_time": 0.0,
                            }
                        self.controller.timing_stats[self.name]["total_time"] += elapsed_time
                        self.controller.timing_stats[self.name]["call_count"] += 1
                        self.controller.timing_stats[self.name]["avg_time"] = (
                            self.controller.timing_stats[self.name]["total_time"]
                            / self.controller.timing_stats[self.name]["call_count"]
                        )
                return False

        return TimingContext(self, function_name)

    def get_timing_stats(self):
        """Get timing statistics information"""
        with self.timing_lock:
            return self.timing_stats.copy()

    def print_timing_stats(self):
        """Print timing statistics information"""
        stats = self.get_timing_stats()
        if not stats:
            logger.info("No timing statistics available")
            return
        logger.info("=" * 80)
        logger.info("Timing Statistics Report")
        logger.info("=" * 80)
        # Sort by total time
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total_time"], reverse=True)
        for func_name, data in sorted_stats:
            logger.info(f"{func_name}:")
            logger.info(f"  Total time: {data['total_time']:.4f} seconds")
            logger.info(f"  Call count: {data['call_count']}")
            logger.info(f"  Average time: {data['avg_time']:.4f} seconds")
        logger.info("=" * 80)

    def reset_timing_stats(self):
        """Reset timing statistics"""
        with self.timing_lock:
            self.timing_stats.clear()

    def _init_robot_cfg(
        self,
        robot_cfg,
        scene_usd,
        is_mocap=False,
        batch_num=0,
        init_position=[0, 0, 0],
        init_rotation=[1, 0, 0, 0],
        stand_type="cylinder",
        size_x=0.1,
        size_y=0.1,
        init_joint_position=[],
        init_joint_names=[],
    ):
        with self._timing_context("_init_robot_cfg"):
            self.scene_usd = scene_usd
            current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            robot_config_dir = os.path.join(current_directory, "config/robot_cfg")
            robot = RobotCfg(os.path.join(robot_config_dir, robot_cfg))
            # create robot first to use robot dof names
            self.robot_usd_path = os.path.join(self.sim_assets_root, robot.robot_usd)
            self.scene_usd_path = os.path.join(self.sim_assets_root, scene_usd)
            get_prim_at_path("/World")
            self.batch_num = batch_num
            if "World" not in robot.robot_prim_path:
                add_reference_to_stage(self.robot_usd_path, robot.robot_prim_path)
            else:
                add_reference_to_stage(self.robot_usd_path, "/World")
            add_reference_to_stage(self.scene_usd_path, "/World")
            self.usd_objects["robot"] = XFormPrim(
                prim_path=robot.robot_prim_path,
                position=init_position,
                orientation=init_rotation,
            )
            # overwrite init joint position
            if len(init_joint_position) != len(init_joint_names):
                raise ValueError("robot init joint position and names length not match")
            if init_position[2] > 0:
                cube_position = [
                    init_position[0],
                    init_position[1],
                    init_position[2] / 2,
                ]
                cube_scale = [size_x, size_y, init_position[2]]
                if stand_type == "cylinder":
                    cylinder.VisualCylinder(
                        prim_path="/base_cube",
                        position=cube_position,
                        orientation=(1, 0, 0, 0),
                        scale=cube_scale,
                        color=np.array([1, 1, 1]),
                    )
                else:
                    cuboid.VisualCuboid(
                        prim_path="/base_cube",
                        position=cube_position,
                        orientation=(1, 0, 0, 0),
                        scale=cube_scale,
                        color=np.array([1, 1, 1]),
                    )
            self.robot_init_position = init_position
            self.robot_init_rotation = init_rotation
            if "multispace" in scene_usd:
                self.scene_name = scene_usd.split("/")[-3] + "/" + scene_usd.split("/")[-2]
            else:
                self.scene_name = scene_usd.split("/")[-2]
            for idx in range(batch_num):
                if "World" not in robot.robot_prim_path:
                    prim_path = robot.robot_prim_path + "_{}".format(idx)
                    add_reference_to_stage(self.robot_usd_path, prim_path)
                    XFormPrim(prim_path=prim_path, position=[0, 2 * idx + 1, 0])
                else:
                    add_reference_to_stage(self.robot_usd_path, "/World_{}".format(idx))
                add_reference_to_stage(self.scene_usd_path, "/World_{}".format(idx))
                XFormPrim(prim_path="/World_{}".format(idx), position=[0, 2 * idx + 1, 0])
            self.material_changer = material_changer()
            camera_state = ViewportCameraState("/OmniverseKit_Persp")
            camera_state.set_position_world(
                Gf.Vec3d(1.9634841037804776, 0.9488467163528935, 2.1182000480154555),
                True,
            )
            camera_state.set_target_world(
                Gf.Vec3d(init_position[0], init_position[1], init_position[2]), True
            )
            stage = omni.usd.get_context().get_stage()
            self.scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
            self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            self.scene.CreateGravityMagnitudeAttr().Set(9.81)
            robot_rep = rep.get.prims(path_pattern=robot.robot_prim_path, prim_types=["Xform"])
            viewport, window = get_active_viewport_and_window()
            if "G1" in robot.robot_name:
                viewport.set_active_camera("/G1/head_link2/Head_Camera")
            elif "G2" in robot.robot_name:
                viewport.set_active_camera("/G2/head_link3/head_front_Camera")
            with robot_rep:
                rep.modify.semantics([("class", "robot")])
            self.robot_cfg = robot
            self._play()

            # init kinematic solver
            articulation = self._initialize_articulation()
            self.dof_names = articulation.dof_names
            # overwrite fixed joints in robot description
            joint_indices_mapping = {}
            joint_names = []
            if "G1" in robot.robot_name:
                joint_names = [
                    "idx01_body_joint1",
                    "idx02_body_joint2",
                    "idx11_head_joint1",
                    "idx12_head_joint2",
                ]
            elif "G2" in robot.robot_name:
                joint_names = [
                    "idx01_body_joint1",
                    "idx02_body_joint2",
                    "idx03_body_joint3",
                    "idx04_body_joint4",
                    "idx05_body_joint5",
                    "idx11_head_joint1",
                    "idx12_head_joint2",
                    "idx13_head_joint3",
                ]
            joint_indices_mapping = {
                joint_name: articulation.get_dof_index(joint_name) for joint_name in joint_names
            }
            for idx, joint_name in enumerate(init_joint_names):
                joint_index = articulation.get_dof_index(joint_name)
                if joint_index < len(robot.init_joint_position):
                    robot.init_joint_position[joint_index] = init_joint_position[idx]

            def create_temp_robot_description(robot_description_path):
                with open(robot_config_dir + robot_description_path, "r") as file:
                    robot_description = yaml.safe_load(file)
                    for cspace_rule in robot_description["cspace_to_urdf_rules"]:
                        if (
                            cspace_rule["name"] in joint_indices_mapping
                            and cspace_rule["rule"] == "fixed"
                        ):
                            joint_index = joint_indices_mapping[cspace_rule["name"]]
                            if joint_index < len(robot.init_joint_position):
                                cspace_rule["value"] = robot.init_joint_position[joint_index]
                temp_robot_description_path = robot_description_path.replace(".yaml", "_tmp.yaml")
                with open(robot_config_dir + temp_robot_description_path, "w") as file:
                    yaml.dump(robot_description, file, default_flow_style=False)
                return temp_robot_description_path

            if robot.arm_type == "dual":
                left_description_path = robot.robot_description_path["left"]
                right_description_path = robot.robot_description_path["right"]
                robot.robot_description_path = {
                    "left": create_temp_robot_description(left_description_path),
                    "right": create_temp_robot_description(right_description_path),
                }
            else:
                robot.robot_description_path = create_temp_robot_description(
                    robot.robot_description_path
                )
            self.robot_cfg = robot
            self.ui_builder._init_kinematic_solver(self.robot_cfg)
            self.init_joint_position = robot.init_joint_position
            self.ui_builder.init_joint_position = robot.init_joint_position
            self.goal_position, self.goal_rotation = self._get_ee_pose(True)

    def _reset_scene_material(self):
        stage = omni.usd.get_context().get_stage()
        # material randomizer
        scene_usd_dir = os.path.dirname(self.scene_usd_path)
        if os.path.exists(scene_usd_dir + "/material_config.json"):
            np.random.seed(None)
            with open(scene_usd_dir + "/material_config.json", "r") as f:
                material_config = json.load(f)
            scene_usd_filename = os.path.basename(os.path.normpath(self.scene_usd_path))
            scene_material_config = material_config.get(scene_usd_filename, {})
            for random_config in scene_material_config.get("random_materials", []):
                prim_paths = random_config.get("prim_paths", [])
                material_buffer = random_config.get("material_buffer", [])
                # Select len(prim_paths) materials from material_buffer for replacement
                selected_materials = np.random.choice(
                    material_buffer, size=len(prim_paths), replace=False
                )
                for idx, prim_path in enumerate(prim_paths):
                    # check if prim_path is a valid prim path
                    if not stage.GetPrimAtPath(prim_path).IsValid():
                        continue
                    mesh_prim = stage.GetPrimAtPath(prim_path)
                    material_path = selected_materials[idx]
                    material = UsdShade.Material.Get(stage, Sdf.Path(material_path))
                    UsdShade.MaterialBindingAPI(mesh_prim).UnbindAllBindings()
                    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

    def _play(self):
        self.ui_builder.my_world.play()
        self._init_robot(self.robot_cfg, False, self.enable_curobo)
        self.frame_status = []

    def _init_robot(self, robot: RobotCfg, is_mocap, enable_curobo):
        with self._timing_context("_init_robot"):
            self.robot_name = robot.robot_name
            self.robot_prim_path = robot.robot_prim_path
            self.end_effector_prim_path = robot.end_effector_prim_path
            self.end_effector_center_prim_path = robot.end_effector_center_prim_path
            self.arm_base_prim_path = robot.arm_base_prim_path
            self.end_effector_name = robot.end_effector_name
            self.finger_names = robot.finger_names
            self.gripper_names = [robot.left_gripper_name, robot.right_gripper_name]
            self.gripper_controll_joint = robot.gripper_controll_joint
            self.opened_positions = robot.opened_positions
            self.closed_velocities = robot.closed_velocities
            self.closed_positions = robot.closed_positions
            self.cameras = robot.cameras
            self.is_single_gripper = robot.is_single
            self.gripper_type = robot.gripper_type
            self.gripper_max_force = robot.gripper_max_force
            self.init_joint_position = robot.init_joint_position
            self.ui_builder._init_solver(robot, enable_curobo, self.batch_num)
            self._get_observation()

    def on_physics_step(self):
        with self._timing_context("on_physics_step"):
            with self._timing_context("on_physics_step:ui_builder"):
                self.ui_builder._on_every_frame_trajectory_list()
                # curobo step
                for key, curobo_motion in self.ui_builder.curoboMotion.items():
                    additional_action = None
                    if self.gripper_action_timing is not None:
                        state = self.gripper_action_timing.get("state", None)
                        timing = self.gripper_action_timing.get("timing", None)
                        is_right = self.gripper_action_timing.get("is_right", True)
                        if is_right and key == "right" or (not is_right and key == "left"):
                            if state is not None and timing is not None:
                                if (
                                    curobo_motion.cmd_idx
                                    >= len(curobo_motion.cmd_plan.position) * timing
                                ):
                                    additional_action = self._get_gripper_action(state, is_right)
                    curobo_motion.on_physics_step(self.motion_run_ratio, additional_action)

            self.on_command_step()
            with self._timing_context("on_physics_step:publish_ros"):
                if self.publish_ros:
                    for ros_publisher_node in self.ros_publishers:
                        ros_publisher_node.tick(self.ui_builder.my_world.current_time)
                    return

    def get_camera_intrinsic_info(self, prim_path):
        camera_info = [
            "omni:lensdistortion:model",
            "omni:lensdistortion:opencvPinhole:cx",
            "omni:lensdistortion:opencvPinhole:cy",
            "omni:lensdistortion:opencvPinhole:fx",
            "omni:lensdistortion:opencvPinhole:fy",
            "omni:lensdistortion:opencvPinhole:imageSize",
            "omni:lensdistortion:opencvPinhole:k1",
            "omni:lensdistortion:opencvPinhole:k2",
            "omni:lensdistortion:opencvPinhole:k3",
            "omni:lensdistortion:opencvPinhole:p1",
            "omni:lensdistortion:opencvPinhole:p2",
        ]
        stage = omni.usd.get_context().get_stage()
        camera_intrinsic_info = {}
        camera_prim = stage.GetPrimAtPath(prim_path)
        for info in camera_info:
            value = camera_prim.GetAttribute(info).Get()
            info = info.split(":")[-1]
            camera_intrinsic_info[info] = str(value)
        return camera_intrinsic_info

    def store_current_state(self):
        state_info = {}
        state_info["rigid_body"] = {}
        state_info["articulation_objects"] = {}
        state_info["articulation"] = {}
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return {}
        for prim_path, rigid_body in self.rigid_bodies.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            state_info["rigid_body"][prim_path] = rigid_body.get_current_dynamic_state()
        for prim_path, articulation_obj in self.articulat_objects.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            state_info["articulation_objects"][prim_path] = {}
            state_info["articulation_objects"][prim_path][
                "joint_positions"
            ] = articulation_obj.get_joint_positions()
            state_info["articulation_objects"][prim_path][
                "joint_velocities"
            ] = articulation_obj.get_joint_velocities()
        articulation = self._initialize_articulation()
        state_info["articulation"]["joint_positions"] = articulation.get_joint_positions()
        state_info["articulation"]["joint_velocities"] = articulation.get_joint_velocities()

        # curobo
        state_info["attach_states"] = self.attach_states.copy()

        # timestamp
        state_info["timestamp"] = self.ui_builder.my_world.current_time

        # other local tmporary variable
        state_info["target_joints_pose"] = self.target_joints_pose.copy()
        state_info["target_position"] = self.target_position.copy()
        state_info["target_rotation"] = self.target_rotation.copy()
        state_info["gripper_state"] = self.gripper_state
        state_info["gripper_state_L"] = self.gripper_state_L
        state_info["gripper_state_R"] = self.gripper_state_R
        state_info["gripper_action_timing"] = (
            self.gripper_action_timing.copy() if self.gripper_action_timing is not None else None
        )
        state_info["ros_step"] = self.ros_step
        state_info["target_point"] = self.target_point.copy()
        return state_info

    def restore_state(self, state_info):
        MAX_PLAYBACK_WAITED_FRAME_NUM = 10
        self.playback_waited_frame_num += 1
        if not state_info:
            return False
        if (
            self.playback_waited_frame_num == 1
            or self.playback_waited_frame_num >= MAX_PLAYBACK_WAITED_FRAME_NUM
        ):
            # isaac related
            stage = omni.usd.get_context().get_stage()
            if not stage:
                logger.info("stage not found")
                return False
            for prim_path, dynamic_state in state_info["rigid_body"].items():
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    logger.info(f"prim is not valid {prim_path}")
                    return False
                rigidbody = self.rigid_bodies.get(prim_path, None)
                if not rigidbody:
                    logger.info(f"rigidbody is not valid {prim_path}")
                    return False
                rigidbody.set_world_pose(dynamic_state.position, dynamic_state.orientation)
                rigidbody.set_linear_velocity(dynamic_state.linear_velocity)
                rigidbody.set_angular_velocity(dynamic_state.angular_velocity)
            for prim_path, articulation_obj_info in state_info["articulation_objects"].items():
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    logger.info(f"prim is not valid {prim_path}")
                    return False
                articulation_obj = self.articulat_objects.get(prim_path, None)
                if not articulation_obj:
                    logger.info(f"articulation_obj is not valid {prim_path}")
                    return False
                articulation_obj.set_joint_positions(articulation_obj_info["joint_positions"])
                articulation_obj.set_joint_velocities(articulation_obj_info["joint_velocities"])

            articulation = self._initialize_articulation()
            articulation.set_joint_positions(state_info["articulation"]["joint_positions"])
            articulation.set_joint_velocities(state_info["articulation"]["joint_velocities"])
            # other local variables
            self.target_joints_pose = state_info["target_joints_pose"]
            self.target_position = state_info["target_position"]
            self.target_rotation = state_info["target_rotation"]
            self.gripper_state = state_info["gripper_state"]
            self._reset_stiffness()
            self._set_gripper_state(state_info["gripper_state_R"], True, 0.8)
            self._set_gripper_state(state_info["gripper_state_L"], False, 0.8)
            self.ros_step = state_info["ros_step"]
            self.target_point = state_info["target_point"]
            # curobo related
            if state_info["attach_states"] and state_info["attach_states"] != self.attach_states:
                if self.playback_waited_frame_num == 1:
                    self.ui_builder.detach_objs()
                else:
                    # add current not attached
                    right_prims = [
                        path
                        for path, is_right in state_info["attach_states"].items()
                        if is_right and path not in self.attach_states
                    ]
                    left_prims = [
                        path
                        for path, is_right in state_info["attach_states"].items()
                        if not is_right and path not in self.attach_states
                    ]
                    if right_prims:
                        self.ui_builder.attach_objs(right_prims, True)
                    if left_prims:
                        self.ui_builder.attach_objs(left_prims, False)
                    self.attach_states = state_info["attach_states"]
        if self.playback_waited_frame_num >= MAX_PLAYBACK_WAITED_FRAME_NUM:
            # timestamp
            start_timstamp = state_info["timestamp"]
            end_timstamp = self.ui_builder.my_world.current_time
            self.playback_timerange.append([start_timstamp, end_timstamp])
            self.playback_waited_frame_num = 0
            logger.info("playback finish")
        return True

    def command_thread(self):
        while True:
            self.on_command_step()

    # Command handler methods - extracted from on_command_step for better organization
    def handle_linear_move(self):
        """Handle Command 2: LinearMove"""
        self.data_to_send = None
        target_position = self.data["target_position"]
        target_rotation = self.data["target_rotation"]
        is_backend = self.data["is_backend"]
        goal_offset = self.data.get("goal_offset", [0, 0, 0, 1, 0, 0, 0])
        path_constraint = self.data.get("path_constraint", None)
        offset_and_constraint_in_goal_frame = self.data.get(
            "offset_and_constraint_in_goal_frame", True
        )
        disable_collision_links = self.data.get("disable_collision_links", [])
        from_current_pose = self.data.get("from_current_pose", False)
        is_Right = False
        if self.data["isArmRight"]:
            is_Right = True
        if not is_backend:
            self.ui_builder.rmp_flow = False
            if (
                np.linalg.norm(self.target_position - target_position) != 0.0
                or np.linalg.norm(self.target_rotation - target_rotation) != 0.0
                or self.ui_builder.get_curobo_motion(is_Right).success is False
            ):
                self.motion_run_ratio = self.data.get("motion_run_ratio", 1.0)
                self.gripper_action_timing = self.data.get("gripper_action_timing", None)
                self.target_position = target_position
                self.target_rotation = target_rotation
                self._hand_moveto(
                    position=target_position,
                    rotation=target_rotation,
                    isRight=is_Right,
                    goal_offset=goal_offset,
                    path_constraint=path_constraint,
                    offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
                    disable_collision_links=disable_collision_links,
                    from_current_pose=from_current_pose,
                )
            if self.ui_builder.get_curobo_motion(is_Right).reached:
                self.data_to_send = self.ui_builder.get_curobo_motion(is_Right).success
                self.motion_run_ratio = 1.0
                self.gripper_action_timing = None
        else:
            if (
                np.linalg.norm(self.target_position - target_position) != 0.0
                or np.linalg.norm(self.target_rotation - target_rotation) != 0.0
            ):
                self.target_position = target_position
                self.target_rotation = target_rotation
                self.arm_move_rmp(
                    position=target_position,
                    rotation=target_rotation,
                    ee_interpolation=self.data["ee_interpolation"],
                    distance_frame=self.data["distance_frame"],
                    is_right=is_Right,
                )
            if self.ui_builder.reached:
                self.data_to_send = True

    def handle_set_joint_position(self):
        """Handle Command 3: SetJointPosition"""
        target_joints_pose = self.data["target_joints_position"]
        target_joint_names = self.data["target_joint_names"]
        articulation = self._initialize_articulation()
        target_joint_indices = [articulation.get_dof_index(name) for name in target_joint_names]
        is_trajectory = self.data["is_trajectory"]
        if not len(self.target_joints_pose):
            for idx, value in enumerate(list(self._get_joint_positions().values())):
                if idx in target_joint_indices:
                    self.target_joints_pose.append(value)
        if np.linalg.norm(np.array(self.target_joints_pose) - np.array(target_joints_pose)) != 0:
            self.target_joints_pose = target_joints_pose
            self._joint_moveto(
                target_joints_pose, target_joint_indices, is_trajectory=is_trajectory
            )
        if not is_trajectory:
            self.data_to_send = "move joints"
            self.target_joints_pose = []
        else:
            if self.ui_builder.reached:
                self.data_to_send = "move_joints"
                self.target_joints_pose = []

    def handle_get_object_pose(self):
        """Handle Command 5: GetObjectPose"""
        prim_path = self.data["object_prim_path"]
        self.data_to_send = self._get_object_pose(prim_path)

    def handle_add_object(self):
        """Handle Command 6: AddObject"""
        usd_path = self.data["usd_object_path"]
        prim_path = self.data["usd_object_prim_path"]
        label_name = self.data["usd_label_name"]
        position = self.data["usd_object_position"]
        rotation = self.data["usd_object_rotation"]
        scale = self.data["usd_object_scale"]
        object_color = self.data["object_color"]
        object_material = self.data["object_material"]
        object_mass = self.data["object_mass"]
        static_friction = self.data["static_friction"]
        dynamic_friction = self.data["dynamic_friction"]
        add_rigid_body = self.data["add_rigid_body"]
        model_type = self.data["model_type"]
        self._add_usd_object(
            usd_path=usd_path,
            prim_path=prim_path,
            label_name=label_name,
            position=position,
            rotation=rotation,
            scale=scale,
            object_color=object_color,
            object_material=object_material,
            object_mass=object_mass,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            add_rigid_body=add_rigid_body,
            model_type=model_type,
        )
        self.data_to_send = "object added"

    def handle_get_joint_position(self):
        """Handle Command 8: GetJointPosition"""
        self.data_to_send = self._get_joint_positions()

    def handle_set_gripper_state(self):
        """Handle Command 9: SetGripperState"""
        state = self.data["gripper_state"]
        isRight = self.data["is_gripper_right"]
        width = self.data["opened_width"]
        if self.gripper_state != state:
            self._set_gripper_state(state=state, isRight=isRight, width=width)
            self.gripper_state = state
        if isRight:
            is_reached = self.gripper_R.is_reached
        else:
            is_reached = self.gripper_L.is_reached
        if is_reached:
            self.gripper_state = ""
            self.data_to_send = "gripper moving"

    def get_camera_prim_name(self, prim_path):
        prim_name = prim_path.split("/")[-1]
        if "G1" in self.robot_name:
            if "head" in prim_name.lower():
                prim_name = "head"
            elif "right" in prim_name.lower():
                prim_name = "hand_right"
            elif "left" in prim_name.lower():
                prim_name = "hand_left"
            elif "top" in prim_name.lower():
                prim_name = "head_front_fisheye"
        if "G2" in self.robot_name:
            if "head_front" in prim_name.lower():
                prim_name = "head"
            elif "head_right" in prim_name.lower():
                prim_name = "head_stereo_right"
            elif "head_left" in prim_name.lower():
                prim_name = "head_stereo_left"
            elif "left_camera" in prim_name.lower() and "head" not in prim_name.lower():
                prim_name = "hand_right"
            elif "right_camera" in prim_name.lower() and "head" not in prim_name.lower():
                prim_name = "hand_left"
        return prim_name

    def handle_get_observation(self):
        """Handle Command 11: GetObservation / StartRecording / StopRecording"""
        if self.data["startRecording"]:
            with self._timing_context("start_recording"):
                self.task_name = self.data["task_name"]
                self.fps = self.data["fps"]
                current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                root_path = current_directory + "/recording_data/"
                recording_path = root_path + self.task_name
                if os.path.isdir(recording_path):
                    folder_index = 1
                    while os.path.isdir(recording_path + str(folder_index)):
                        folder_index += 1
                    recording_path = recording_path + str(folder_index)
                self.path_to_save = recording_path
                self.camera_info_list = {}
                tf_target = []
                for prim_path in self.data["camera_prim_list"]:
                    image = self._capture_camera(
                        prim_path=prim_path,
                        isRGB=False,
                        isDepth=False,
                        isSemantic=False,
                        isGN=False,
                    )
                    try:
                        camera_info = self.get_camera_intrinsic_info(prim_path)
                    except Exception as e:
                        logger.error(f"Failed to get camera intrinsic info: {e}")
                        camera_info = image["camera_info"]
                    prim_name = self.get_camera_prim_name(prim_path)
                    self.camera_info_list[prim_name] = {
                        "intrinsic": camera_info,
                        "output": {
                            "rgb": "camera/" + "{frame_num}/" + f"{prim_name}_color.jpg",
                            "video": f"{prim_name}_color.mp4",
                        },
                    }
                    if "fisheye" not in prim_name:
                        self.camera_info_list[prim_name]["output"]["depth"] = (
                            "camera/" + "{frame_num}/" + f"{prim_name}_depth.png"
                        )

                    if self.data["render_semantic"]:
                        self.write_semantic = True
                        self.camera_info_list[prim_name]["output"]["semantic"] = (
                            "camera/" + "{frame_num}/" + f"{prim_name}_semantic.png"
                        )
                    tf_target.append(prim_path)

                if self.publish_ros:
                    ros_cmd_distro = os.getenv("ROS_CMD_DISTRO", "humble")
                    exclude_args = "--exclude-regex" if ros_cmd_distro != "humble" else "--exclude"
                    command_str = f"""
                        unset PYTHONPATH
                        unset LD_LIBRARY_PATH
                        source /opt/ros/{ros_cmd_distro}/setup.bash
                        ros2 bag record -o {recording_path} {exclude_args} '.*_rgb(?!_)' -a
                        """
                    logger.info("publish_ros command: " + command_str)
                    process = subprocess.Popen(
                        command_str,
                        shell=True,
                        executable="/bin/bash",
                        preexec_fn=os.setsid,
                    )
                    self.process.append(process)
                    frequency = (int)(
                        1 / (self.ui_builder.my_world.get_rendering_dt() * self.data["fps"])
                    )  # this is actually step_size in the condition of 60 fps
                    logger.info(
                        f"frequency: {frequency}, fps: {self.data['fps']}, rendering_dt: {self.ui_builder.my_world.get_rendering_dt()}"
                    )
                    additional_cam_parameters = self.data.get("additional_cam_parameters", "")
                    if additional_cam_parameters:
                        additional_cam_parameters = json.loads(additional_cam_parameters)
                    noised_probability = additional_cam_parameters.get("noised_probability", 0.2)
                    logger.info(f"noised_probability{noised_probability}")
                    noise_parameters = additional_cam_parameters.get("noise_parameters", {})
                    noised_camera_indices = additional_cam_parameters.get(
                        "noised_camera_indices",
                        np.arange(len(self.data["camera_prim_list"])),
                    )
                    noised_camera_prim_list = [
                        self.data["camera_prim_list"][i] for i in noised_camera_indices
                    ]
                    logger.info(f"noised_camera_indices{noised_camera_indices}")
                    logger.info(f"noised_camera_prim_list{noised_camera_prim_list}")
                    self.sensor_base._init_sensor(self.loop_count)
                    for prim in self.object_asset_dict.keys():
                        if prim not in tf_target:
                            tf_target.append(prim)
                    for prim in self.end_effector_prim_path.values():
                        if prim not in tf_target:
                            tf_target.append(prim)
                    for prim in self.end_effector_center_prim_path.values():
                        if prim not in tf_target:
                            tf_target.append(prim)
                    tf_target.append(self.robot_prim_path)
                    if self.arm_base_prim_path not in tf_target:
                        tf_target.append(self.arm_base_prim_path)
                    delta_time = 1 / (2 * self.rendering_step)
                    logger.info(f"tf_target{tf_target}")
                    self.sensor_base.publish_tf(
                        robot_prim=self.robot_prim_path,
                        targets=tf_target,
                        approx_freq=2,
                        delta_time=delta_time,
                    )

                    for prim in self.articulat_objects.keys():
                        self.sensor_base.publish_joint(
                            robot_prim=prim,
                            approx_freq=2,
                            delta_time=delta_time,
                            topic_name=prim,
                        )
                    self.fps = self.data["fps"]
                    self.graph_path = [
                        "/World/RobotTFActionGraph",
                        "/World/RobotJointActionGraph",
                        "/ClockActionGraph",
                    ]

                    if not self.camera_graph_path:
                        for camera in self.data["camera_prim_list"]:

                            camera_param = {
                                "path": camera,
                                "frequency": frequency,
                                "resolution": {
                                    "width": self.cameras[camera][0],
                                    "height": self.cameras[camera][1],
                                },
                                "publish": [
                                    "rgb:/" + camera.split("/")[-1] + "_rgb",
                                    "depth:/" + camera.split("/")[-1],
                                ],
                            }

                            if "Fisheye" in camera or "Top" in camera:
                                camera_param["publish"] = ["rgb:/" + camera.split("/")[-1] + "_rgb"]
                            elif self.data["render_semantic"]:
                                camera_param["publish"] = [
                                    "rgb:/" + camera.split("/")[-1] + "_rgb",
                                    "depth:/" + camera.split("/")[-1],
                                    "semantic:/" + camera.split("/")[-1] + "_semantic",
                                ]
                            else:
                                camera_param["publish"] = [
                                    "rgb:/" + camera.split("/")[-1] + "_rgb",
                                    "depth:/" + camera.split("/")[-1],
                                ]
                            if camera in noised_camera_prim_list:
                                camera_param["noised"] = np.random.uniform() < noised_probability
                                camera_param["noise_parameters"] = noise_parameters
                            else:
                                camera_param["noised"] = False
                            if "head_right_Camera" in camera or "head_left_Camera" in camera:
                                camera_param["publish"].remove("depth:/" + camera.split("/")[-1])
                            self.camera_info_list[self.get_camera_prim_name(camera)]["noised"] = (
                                camera_param["noised"]
                            )
                            camera_graph, ros_nodes = self.sensor_base._init_camera(camera_param)
                            self.ros_publishers += ros_nodes
                        self.camera_graph_path.append(self.data["camera_prim_list"])
                        # joint action
                        articulation_action_node = self.sensor_base.publish_articulation_action(
                            robot=self.robot,
                            step_size=1,
                        )
                        self.ros_publishers.append(articulation_action_node)

                    for camera in self.data["camera_prim_list"]:
                        logger.info(f"republish camera{camera}")
                        topic_name = "/" + camera.split("/")[-1] + "_rgb"
                        compressed_name = topic_name + "_compressed"
                        ros_cmd_distro = os.getenv("ROS_CMD_DISTRO", "humble")
                        extra_args = (
                            "--remap _out_transport:=compressed"
                            if ros_cmd_distro != "humble"
                            else ""
                        )
                        command_str = f"""
                        unset PYTHONPATH
                        unset LD_LIBRARY_PATH
                        source /opt/ros/{ros_cmd_distro}/setup.bash
                        ros2 run image_transport republish raw compressed {extra_args} --ros-args --remap /in:={topic_name} --remap /out:={compressed_name}
                        """
                        logger.info(command_str)
                        subpro = subprocess.Popen(
                            command_str,
                            shell=True,
                            executable="/bin/bash",
                            preexec_fn=os.setsid,
                        )
                        self.process.append(subpro)
                    if not self.ros_node_initialized:
                        self.server_ros_node = ServerNode(robot_name=self.robot_name)
                        self.ros_node_initialized = True
                    self.data_to_send = "Start"
                else:
                    raise ValueError("publish ros is not enabled")
        elif self.data["stopRecording"]:
            with self._timing_context("stop_recording"):
                if self.publish_ros:
                    for process in self.process:
                        try:
                            if process.poll() is None:  # Check if process is still running
                                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                                logger.info(f"Sent SIGINT to process group {process.pid}")
                        except ProcessLookupError:
                            logger.info(f"Process {process.pid} has exited")
                        except Exception as e:
                            logger.info(f"Failed to send signal to process {process.pid}: {e}")
                    for process in self.process:
                        try:
                            if process.poll() is None:
                                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                                process.wait(timeout=5)  # Wait for rosbag to exit completely
                        except Exception as e:
                            logger.info(f"Failed to force terminate process {process.pid}: {e}")
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    self.ui_builder.remove_graph(self.graph_path)

                    self.process = []
                self.data_to_send = "Stopped"
        else:
            raise ValueError("Invalid command: GetObservation is not supported")

    def handle_reset(self):
        """Handle Command 12: Reset"""
        self._on_reset()
        self.loop_count += 1
        self.data_to_send = "reset"

    def handle_attach_obj(self):
        """Handle Command 13: AttachObj"""
        obj_prims = self.data["obj_prim_paths"]
        is_right = self.data["is_right"]
        items = []
        stage = omni.usd.get_context().get_stage()
        if stage:
            for prim_path in obj_prims:
                for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                    path = str(prim.GetPath())
                    prim = get_prim_at_path(path)
                    if prim.IsA(UsdGeom.Mesh):
                        items.append(path)
        self.ui_builder.attach_objs(items, is_right)
        attach_states = {item: is_right for item in items}
        self.attach_states.update(attach_states)
        self.data_to_send = "attaching"

    def handle_detach_obj(self):
        """Handle Command 14: DetachObj"""
        self.ui_builder.detach_objs()
        self.attach_states = {}
        self.data_to_send = "detaching"

    def handle_task_status(self):
        """Handle Command 16: TaskStatus"""
        isSuccess = self.data["isSuccess"]
        config = self.task_metric
        if self.task_name is not None:
            if isSuccess is True:
                task_info = {
                    "bag_file": self.path_to_save,
                    "output_dir": self.path_to_save,
                    "robot_init_position": self.robot_init_position.tolist(),
                    "robot_init_rotation": self.robot_init_rotation.tolist(),
                    "camera_info": self.camera_info_list,
                    "scene_name": self.scene_name,
                    "scene_usd": self.scene_usd,
                    "object_names": {
                        "object_prims": list(self.object_asset_dict.keys()),
                        "articulated_object_prims": list(self.articulat_objects.keys()),
                    },
                    "fps": self.fps,
                    "robot_name": self.robot_name,
                    "frame_status": self.frame_status,
                    "light_config": self.light_config,
                    "gripper_names": self.gripper_names,
                    "with_img": True,
                    "with_video": True,
                    "playback_timerange": self.playback_timerange,
                    "end_effector_prim_path": self.end_effector_prim_path,
                    "end_effector_center_prim_path": self.end_effector_center_prim_path,
                    "arm_base_prim_path": self.arm_base_prim_path,
                    "task_name": self.task_name,
                    "fail_stage_step": self.data["failStep"],
                    "object_asset_dict": self.object_asset_dict,
                }
                task_info_path = self.path_to_save + "/recording_info.json"
                with open(task_info_path, "w") as f:
                    json.dump(task_info, f, indent=4)
                with open(
                    self.path_to_save + "/frame_state.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(self.frame_status, f, indent=4)
                metric_config_path = self.path_to_save + "/metric_config.json"
                with open(metric_config_path, "w") as f:
                    json.dump(config, f, indent=4)

                total_time = 0
                clean_once = True
                while len(self.extract_process) > MAX_EXTRACT_PROCESS_NUM or clean_once:
                    new_process = []
                    clean_once = False
                    for p, log_file in self.extract_process:
                        if p.poll():
                            new_process.append((p, log_file))
                        else:
                            log_file.close()
                    self.extract_process = new_process
                    time.sleep(0.1)
                    total_time += 0.1
                    if total_time > 120:
                        process, log_file = self.extract_process[0]
                        os.killpg(os.getpgid(process.pid), signal.SIGINT)
                        log_file.close()
                        logger.info("Extract process waiting timeout 120s, kill it")

                log_file = open(self.path_to_save + "/extract.log", "w")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                extract_sub_process = subprocess.Popen(
                    [
                        sys.executable,
                        f"{current_dir}/recording/extract_and_convert_data.py",
                        "--path_to_save",
                        self.path_to_save,
                        "--task_info_path",
                        task_info_path,
                        "--metric_config_path",
                        metric_config_path,
                    ],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                )
                logger.info("Extract process started")
                self.extract_process.append((extract_sub_process, log_file))
            else:
                # remove folder if exist
                if os.path.exists(self.path_to_save):
                    shutil.rmtree(self.path_to_save)

        self.object_asset_dict = {}
        self.data_to_send = str(isSuccess)

    def handle_exit(self):
        """Handle Command 17: Exit"""
        # wait for extract process to finish
        for process, log_file in self.extract_process:
            try:
                if process.poll() is None:
                    process.wait(timeout=300)
                log_file.close()
            except subprocess.TimeoutExpired:
                logger.info("Exit:Extract process waiting timeout 300s, kill it")
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
        self.exit = self.data["exit"]
        self.data_to_send = "exit"

    def handle_get_ee_pose(self):
        """Handle Command 18: GetEEPose"""
        is_right = self.data["isRight"]
        self.data_to_send = self._get_ee_pose(is_right)

    def handle_get_ik_status(self):
        """Handle Command 19: GetIKStatus"""
        target_poses = np.array(self.data["target_poses"])
        ik_result = []
        is_Right = self.data["isRight"]
        ObsAvoid = self.data["ObsAvoid"]
        output_link_pose = self.data["output_link_pose"]
        ik_result = self._get_ik_status(target_poses, is_Right, ObsAvoid, output_link_pose)
        self.data_to_send = ik_result

    def handle_init_robot(self):
        """Handle Command 21: InitRobot"""
        robot_cfg_file = self.data["robot_cfg_file"]
        scene_usd_path = self.data["scene_usd_path"]
        self._init_robot_cfg(
            robot_cfg=robot_cfg_file,
            scene_usd=scene_usd_path,
            init_position=self.data["robot_position"],
            init_rotation=self.data["robot_rotation"],
            stand_type=self.data["stand_type"],
            size_x=self.data["stand_size_x"],
            size_y=self.data["stand_size_y"],
            init_joint_position=self.data["init_joint_position"],
            init_joint_names=self.data["init_joint_names"],
        )
        self.data_to_send = "success"

    def handle_add_camera(self):
        """Handle Command 22: AddCamera"""
        camera_prim = self.data["camera_prim"]
        camera_position = self.data["camera_position"]
        camera_rotation = self.data["camera_rotation"]
        self._add_camera(
            camera_prim=camera_prim,
            camera_position=camera_position,
            camera_rotation=camera_rotation,
            width=self.data["width"],
            height=self.data["height"],
            focal_length=self.data["focus_length"],
            horizontal_aperture=self.data["horizontal_aperture"],
            vertical_aperture=self.data["vertical_aperture"],
            is_local=self.data["is_local"],
        )
        self.data_to_send = "success"

    def handle_set_object_pose(self):
        """Handle Command 24: SetObjectPose"""
        self._set_object_pose(
            self.data["object_poses"],
            self.data["joint_position"],
            self.data["object_joints"],
        )
        self.data_to_send = "success"

    def handle_set_target_point(self):
        """Handle Command 27: SetTargetPoint"""
        self.target_point = self.data["target_position"]
        self.data_to_send = "success"

    def handle_set_frame_state(self):
        """Handle Command 28: SetFrameState"""
        time_stamp = self.timeline.get_current_time()
        frame_state = json.loads(self.data["frame_state"])
        if "task_description" in frame_state:
            self.task_description = frame_state["task_description"]
        else:
            self.frame_status.append(
                {
                    "time_stamp": time_stamp,
                    "frame_state": frame_state,
                    "task_description": self.task_description,
                }
            )
        self.data_to_send = "success"

    def handle_set_material(self):
        """Handle Command 29: SetMaterial"""
        for material_info in self.data:
            self._set_object_material(
                material_info["object_prim"],
                material_info["material_name"],
                material_info["material_path"],
                material_info["label_name"],
            )
        self.data_to_send = "success"

    def handle_set_light(self):
        """Handle Command 30: SetLight"""
        for light in self.data:
            self._set_light(
                light_type=light["light_type"],
                light_prim=light["light_prim"],
                light_temperature=light["light_temperature"],
                light_intensity=light["light_intensity"],
                light_rotation=light["light_rotation"],
                light_texture=light["light_texture"],
            )
        self.light_config = self.data
        self.data_to_send = "success"

    def handle_get_part_dof_joint(self):
        """Handle Command 32: GetPartDofJoint"""
        self.data_to_send = self._get_dof_joint_for_part(
            self.data["object_prim_path"], self.data["part_name"]
        )

    def handle_remove_objs_from_obstacle(self):
        """Handle Command 52: RemoveObjsFromObstacle"""
        obj_prims = self.data["obj_prim_paths"]
        items = []
        stage = omni.usd.get_context().get_stage()
        if stage:
            for prim_path in obj_prims:
                for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                    path = str(prim.GetPath())
                    prim = get_prim_at_path(path)
                    if prim.IsA(UsdGeom.Mesh):
                        items.append(path)
        self.ui_builder.remove_objects_from_world(items)
        self.data_to_send = f"{obj_prims} removed from obstacle"

    def handle_set_task_metric(self):
        """Handle Command 53: SetTaskMetric"""
        self.timeline.get_current_time()
        metric = json.loads(self.data["task_metric"])
        if "task_metric" in metric:
            self.task_metric = metric["task_metric"]
        self.data_to_send = "success"

    def handle_store_current_state(self):
        """Handle Command 54: StoreCurrentState"""
        self.playback_frames.update({self.data["playback_id"]: self.store_current_state()})
        self.data_to_send = "success"

    def handle_playback(self):
        """Handle Command 55: Playback"""
        success = False
        if self.data["playback_id"] in self.playback_frames:
            success = self.restore_state(self.playback_frames[self.data["playback_id"]])
        if not success:
            self.data_to_send = "playback failed"
        elif self.playback_waited_frame_num == 0:
            self.data_to_send = "success"
        else:
            # wait for some frames after restore state for rendering
            self.data_to_send = None

    def handle_get_checker_status(self):
        """Handle Command 56: GetCheckerStatus"""
        if self.cur_runtime_checker is None:
            checker_config = json.loads(self.data["checker"])
            parameters = checker_config.get("params", {})
            parameters["command_controller"] = self
            parameters["checker_name"] = checker_config["checker_name"]
            self.cur_runtime_checker = create_checker(**parameters)
        self.cur_runtime_checker.check()
        if self.cur_runtime_checker.status == CheckerStatus.PASS:
            self.data_to_send = "success"
            self.cur_runtime_checker = None
        elif (
            self.cur_runtime_checker.status == CheckerStatus.FAIL
            or self.cur_runtime_checker.status == CheckerStatus.ERROR
        ):
            self.data_to_send = "fail"
            self.cur_runtime_checker = None
        return

    # update
    def on_command_step(self):
        with self._timing_context("on_command_step"):
            if not self.data or not self.Command:
                return
            else:
                with self._timing_context(
                    f"rpc_server.step_command_{command_value_to_string[self.Command]}"
                ):
                    if self.Command == Command.LINEAR_MOVE:
                        self.handle_linear_move()
                    elif self.Command == Command.SET_JOINT_POSITION:
                        self.handle_set_joint_position()
                    elif self.Command == Command.GET_OBJECT_POSE:
                        self.handle_get_object_pose()
                    elif self.Command == Command.ADD_OBJECT:
                        self.handle_add_object()
                    elif self.Command == Command.GET_JOINT_POSITION:
                        self.handle_get_joint_position()
                    elif self.Command == Command.SET_GRIPPER_STATE:
                        self.handle_set_gripper_state()
                    elif self.Command == Command.GET_OBSERVATION:
                        self.handle_get_observation()
                    elif self.Command == Command.RESET:
                        self.handle_reset()
                    elif self.Command == Command.ATTACH_OBJ:
                        self.handle_attach_obj()
                    elif self.Command == Command.DETACH_OBJ:
                        self.handle_detach_obj()
                    elif self.Command == Command.TASK_STATUS:
                        self.handle_task_status()
                    elif self.Command == Command.EXIT:
                        self.handle_exit()
                    elif self.Command == Command.GET_EE_POSE:
                        self.handle_get_ee_pose()
                    elif self.Command == Command.GET_IK_STATUS:
                        self.handle_get_ik_status()
                    elif self.Command == Command.INIT_ROBOT:
                        self.handle_init_robot()
                    elif self.Command == Command.ADD_CAMERA:
                        self.handle_add_camera()
                    elif self.Command == Command.SET_OBJECT_POSE:
                        self.handle_set_object_pose()
                    elif self.Command == Command.SET_TARGET_POINT:
                        self.handle_set_target_point()
                    elif self.Command == Command.SET_FRAME_STATE:
                        self.handle_set_frame_state()
                    elif self.Command == Command.SET_MATERIAL:
                        self.handle_set_material()
                    elif self.Command == Command.SET_LIGHT:
                        self.handle_set_light()
                    elif self.Command == Command.GET_PART_DOF_JOINT:
                        self.handle_get_part_dof_joint()
                    elif self.Command == Command.REMOVE_OBJS_FROM_OBSTACLE:
                        self.handle_remove_objs_from_obstacle()
                    elif self.Command == Command.SET_TASK_METRIC:
                        self.handle_set_task_metric()
                    elif self.Command == Command.STORE_CURRENT_STATE:
                        self.handle_store_current_state()
                    elif self.Command == Command.PLAYBACK:
                        self.handle_playback()
                    elif self.Command == Command.GET_CHECKER_STATUS:
                        self.handle_get_checker_status()
                    else:
                        raise ValueError(f"Invalid command: {self.Command}")

        if self.Command:
            with self.condition:
                self.condition.notify_all()

    def _generate_materials(self):
        self.materials = {}
        material_infos = {}
        path = os.path.dirname(__file__) + "/material_infos.json"
        with open(path, "r") as f:
            material_infos = json.load(f)

        for mat in material_infos:
            material = self.material_changer.assign_material(
                material_infos[mat]["material_path"], mat
            )
            self.materials[mat] = material

    def _get_observation(self):
        for camera in self.cameras:
            self._capture_camera(
                prim_path=camera, isRGB=True, isDepth=True, isSemantic=True, isGN=False
            )

    def _on_reset(self):
        self._reset_stiffness()
        self.ui_builder._on_reset()
        self.target_position = [0, 0, 0]
        self._reset_scene_material()
        self._get_observation()
        self.frame_status = []
        self.playback_frames = {}
        self.playback_timerange = []
        self.playback_waited_frame_num = 0

    def _on_blocking_thread(self, data, Command):
        self.data = data
        self.Command = Command
        with self.condition:
            while self.data_to_send is None:
                self.condition.wait()
            result = self.data_to_send
            self.data_to_send = None
            self.Command = 0
            self.result_queue.put(result)

    def blocking_start_server(self, data, Command):
        self._on_blocking_thread(data, Command)
        if not self.result_queue.empty():
            result = self.result_queue.get()
            return result

    # 1. Camera capture function, Input: camera prim path in isaac scene and whether to use Gaussian Noise, return
    def _capture_camera(self, prim_path: str, isRGB, isDepth, isSemantic, isGN: bool):
        self.ui_builder._currentCamera = prim_path
        self.ui_builder._on_capture_cam(isRGB, isDepth, isSemantic)
        currentImage = self.ui_builder.currentImg
        return currentImage

    # 2. Move left and right hands to specified pose, position(x,y,z), rotation(x,y,z) angles
    def _hand_moveto(
        self,
        position,
        rotation,
        isRight=True,
        goal_offset=[0, 0, 0, 1, 0, 0, 0],
        path_constraint=None,
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links=[],
        from_current_pose=False,
    ):
        self.ui_builder._followingPos = position
        self.ui_builder._followingOrientation = rotation
        self._initialize_articulation()
        self.ui_builder._follow_target(
            isRight=isRight,
            goal_offset=goal_offset,
            path_constraint=path_constraint,
            offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
            disable_collision_links=disable_collision_links,
            from_current_pose=from_current_pose,
        )

    def _reset_stiffness(self):
        self._init_grippers()
        self.gripper_L.reset_stiffness()
        self.gripper_R.reset_stiffness()

    def arm_move_rmp(self, position, rotation, ee_interpolation, distance_frame, is_right=True):
        self.ui_builder._followingPos = position
        self.ui_builder._followingOrientation = rotation
        self.ui_builder._trajectory_list_follow_target(
            position, rotation, is_right, ee_interpolation, distance_frame
        )

    def _initialize_articulation(self):
        return self.ui_builder.articulation

    # 3. Move all joints to specified angles, Input: np.array([None])*28
    def _joint_moveto(self, joint_position, joint_indices=None, is_trajectory=False):
        self._initialize_articulation()
        self.ui_builder._move_to(
            joint_position, joint_indices=joint_indices, is_trajectory=is_trajectory
        )

    def _add_camera(
        self,
        camera_prim,
        camera_position,
        camera_rotation,
        width=640,
        height=480,
        focal_length=18.14756,
        horizontal_aperture=20.955,
        vertical_aperture=15.2908,
        is_local=False,
    ):
        camera = Camera(prim_path=camera_prim, resolution=[width, height])
        camera.initialize()
        self._get_observation()
        self._capture_camera(
            prim_path=camera_prim, isRGB=True, isDepth=True, isSemantic=True, isGN=False
        )
        if is_local:
            camera.set_local_pose(
                translation=camera_position,
                orientation=camera_rotation,
                camera_axes="usd",
            )
        else:
            camera.set_world_pose(
                position=camera_position, orientation=camera_rotation, camera_axes="usd"
            )
        self.cameras[camera_prim] = [width, height]
        self.ui_builder.cameras[camera_prim] = [width, height]
        _prim = get_prim_at_path(camera_prim)
        _prim.GetAttribute("focalLength").Set(focal_length)
        _prim.GetAttribute("horizontalAperture").Set(horizontal_aperture)
        _prim.GetAttribute("verticalAperture").Set(vertical_aperture)
        _prim.GetAttribute("clippingRange").Set((0.01, 100000))

    def _get_dof_joint_for_part(self, prim_path, part_name):
        part_names = [part_name]
        self.articulat_objects[prim_path].initialize()
        dof_names = self.articulat_objects[prim_path].dof_names
        positions = self.articulat_objects[prim_path].get_joint_positions()
        velocities = self.articulat_objects[prim_path].get_joint_velocities()
        stage = omni.usd.get_context().get_stage()
        joint_prim_paths = []
        if stage:
            joint_prim_paths = find_joints(stage.GetPrimAtPath(prim_path))
        idx = None
        if stage:
            while idx is None and len(joint_prim_paths) > 0:
                traveled_joint_prim_paths = []
                for joint_prim in joint_prim_paths:
                    joint_api = UsdPhysics.Joint(stage.GetPrimAtPath(joint_prim))
                    body0_rel_targets = joint_api.GetBody0Rel().GetTargets()
                    body1_rel_targets = joint_api.GetBody1Rel().GetTargets()
                    rel_targets = body0_rel_targets + body1_rel_targets
                    for target in rel_targets:
                        if any([name in target.pathString for name in part_names]):
                            if joint_prim.split("/")[-1] in dof_names:
                                idx = dof_names.index(joint_prim.split("/")[-1])
                            else:
                                traveled_joint_prim_paths.append(joint_prim)
                                part_names = [
                                    rel_target.pathString.split("/")[-1]
                                    for rel_target in rel_targets
                                ]
                            break
                    if len(traveled_joint_prim_paths) or idx is not None:
                        break
                for joint_prim in traveled_joint_prim_paths:
                    # remove the joint prime, which is already traversed
                    joint_prim_paths.remove(joint_prim)
                if not len(traveled_joint_prim_paths):
                    # no related joints found
                    break
        if idx is not None:
            dof_name = dof_names[idx]
            position = positions[idx]
            velocity = velocities[idx]
        else:
            dof_name = ""
            position = 0.0
            velocity = 0.0
        return {
            "joint_name": dof_name,
            "joint_position": position,
            "joint_velocity": velocity,
        }

    def _set_object_joint(self, prim_path, target_positions):
        self.articulat_objects[prim_path].initialize()
        self.articulat_objects[prim_path].set_joint_positions(target_positions)

    # Add object
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
        static_friction=0.5,
        dynamic_friction=0.5,
        model_type="convexDecomposition",
        add_rigid_body=True,
    ):
        usd_path = os.path.join(self.sim_assets_root, usd_path)
        self.object_asset_dict[prim_path] = usd_path
        already_in_stage = False
        stage = omni.usd.get_context().get_stage()
        if stage:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                already_in_stage = True
        if not already_in_stage:
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            usd_object = XFormPrim(
                prim_path=prim_path,
                position=position,
                orientation=rotation,
                scale=scale,
            )
            stage = omni.usd.get_context().get_stage()
            type = get_prim_object_type(prim_path)
            items = []
            if stage:
                for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                    path = str(prim.GetPath())
                    prim = get_prim_at_path(path)
                    if prim.IsA(UsdGeom.Mesh):
                        items.append(path)
            object_rep = rep.get.prims(path_pattern=prim_path, prim_types=["Xform"])

            with object_rep:
                rep.modify.semantics([("class", label_name)])
            if type == "articulation":
                self.ui_builder.my_world.play()
                for path in items:
                    if not self.enable_physics:
                        collisionAPI = UsdPhysics.CollisionAPI.Get(stage, path)
                        if collisionAPI:
                            collisionAPI.GetCollisionEnabledAttr().Set(False)
                articulation = Articulation(prim_path)
                articulation.initialize()
                self.articulat_objects[prim_path] = articulation
                self.usd_objects[prim_path] = usd_object
            else:
                self.usd_objects[prim_path] = usd_object
                for _prim in items:
                    geometry_prim = GeometryPrim(prim_path=_prim, reset_xform_properties=False)
                    obj_physics_prim_path = f"{_prim}/object_physics"
                    geometry_prim.apply_physics_material(
                        PhysicsMaterial(
                            prim_path=obj_physics_prim_path,
                            static_friction=static_friction,
                            dynamic_friction=dynamic_friction,
                            restitution=None,
                        )
                    )
                    # set friction combine mode to max to enable stable grasp
                    obj_physics_prim = stage.GetPrimAtPath(obj_physics_prim_path)
                    physx_material_api = PhysxSchema.PhysxMaterialAPI(obj_physics_prim)
                    if physx_material_api is not None:
                        fric_combine_mode = physx_material_api.GetFrictionCombineModeAttr().Get()
                        if fric_combine_mode is None:
                            physx_material_api.CreateFrictionCombineModeAttr().Set("max")
                        elif fric_combine_mode != "max":
                            physx_material_api.GetFrictionCombineModeAttr().Set("max")

                    if object_material != "general":
                        if object_material not in self.materials:
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
                            prim = stage.GetPrimAtPath(_prim)
                            UsdShade.MaterialBindingAPI(prim).Bind(Material)
                if self.enable_physics and add_rigid_body:
                    prim = stage.GetPrimAtPath(prim_path)
                    utils.setRigidBody(prim, model_type, False)
                    logger.info(f" add rigid body to  {prim_path}")
                    rigid_prim = RigidPrim(prim_path=prim_path, mass=object_mass)
                    # rigid_prim.set_mass(10) // deprecated
                    # Get Physics API
                    physics_api = UsdPhysics.MassAPI.Apply(rigid_prim.prim)
                    physics_api.CreateMassAttr().Set(object_mass)
                    self.rigid_bodies[prim_path] = rigid_prim
                    rigid_prim.initialize()
                    # Set center of gravity offset (unit: meters)
                    # physics_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(object_com[0], object_com[1],object_com[2]))
            for curobo_motion in self.ui_builder.curoboMotion.values():
                if curobo_motion:
                    prim_usd_path = usd_path
                    if not (np.array(scale) == 1.0).all():
                        prim_usd_path = None
                        logger.info("scale is not 1.0, so simplified mesh cache cannot be used")
                    curobo_motion.add_obstacle_from_prim_path(prim_path, prim_usd_path)
                    break
        else:
            usd_object = XFormPrim(
                prim_path=prim_path,
                position=position,
                orientation=rotation,
                scale=scale,
            )

    def _set_object_pose(self, object_poses, joint_position, object_joints=None, action=False):
        for pose in object_poses:
            if pose["prim_path"] in self.usd_objects:
                object = self.usd_objects[pose["prim_path"]]
                object.set_world_pose(pose["position"], pose["rotation"])
            else:
                stage = omni.usd.get_context().get_stage()
                if not stage:
                    return
                prim = stage.GetPrimAtPath(pose["prim_path"])
                if not prim.IsValid():
                    continue
                translate_attr = prim.GetAttribute("xformOp:translate")
                if translate_attr.IsValid() and len(pose["position"]) == 3:
                    translate_type = type(translate_attr.Get())
                    translate_attr.Set(translate_type(*pose["position"]))
                rotation_data = pose["rotation"]
                orient_attr = prim.GetAttribute("xformOp:orient")
                if orient_attr.IsValid() and len(rotation_data) == 4:
                    quat_type = type(orient_attr.Get())
                    orient_attr.Set(quat_type(*rotation_data))
        if len(joint_position):
            self.ui_builder._move_to(joint_position, is_action=action)
        if object_joints is not None:
            for joint in object_joints:
                self._set_object_joint(
                    prim_path=joint["prim_path"], target_positions=joint["object_joint"]
                )

    def _set_object_material(self, prim_path, material_name, material_path, label_name=None):
        stage = omni.usd.get_context().get_stage()
        logger.info(label_name)
        if label_name:
            object_rep = rep.get.prims(path_pattern=prim_path, prim_types=["Xform"])
            with object_rep:
                rep.modify.semantics([("class", label_name)])
        if not stage:
            return
        if "Glass" in material_name or "glass" in material_name:
            material_prim = "/World/Materials/OmniGlass"
            material = OmniGlass(prim_path=material_prim)
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                path = str(prim.GetPath())
                prim = get_prim_at_path(path)
                if prim.IsA(UsdGeom.Mesh) or prim.GetTypeName() in "GeomSubset":
                    geometry_prim = GeometryPrim(prim_path=path)
                    geometry_prim.apply_visual_material(material)

        else:
            material = self.material_changer.assign_material(material_path, material_name)
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                path = str(prim.GetPath())
                prim = get_prim_at_path(path)
                if prim.IsA(UsdGeom.Mesh) or prim.GetTypeName() in "GeomSubset":
                    UsdShade.MaterialBindingAPI(prim).Bind(material)

    def _set_light(
        self,
        light_type,
        light_prim,
        light_temperature,
        light_intensity,
        light_rotation,
        light_texture,
    ):
        stage = omni.usd.get_context().get_stage()
        light = Light(
            light_type=light_type,
            prim_path=light_prim,
            stage=stage,
            intensity=light_intensity,
            color=light_temperature,
            orientation=light_rotation,
            texture_file=light_texture,
        )
        light.initialize()

    def _get_joint_positions(self):
        self._initialize_articulation()
        articulation = self.ui_builder.articulation
        joint_positions = articulation.get_joint_positions()
        ids = {}
        for idx in range(len(articulation.dof_names)):
            name = articulation.dof_names[idx]
            ids[name] = float(joint_positions[idx])
        return ids

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

    def _init_grippers(self):
        robot = self._initialize_articulation()
        end_effector_prim_path = self.end_effector_prim_path["left"]
        right_end_effector_prim_path = self.end_effector_prim_path["right"]
        self.gripper_L = ParallelGripper(
            end_effector_prim_path=end_effector_prim_path,
            joint_prim_names=self.finger_names["left"],
            joint_closed_velocities=self.closed_velocities["left"],
            joint_closed_positions=self.closed_positions["left"],
            joint_opened_positions=self.opened_positions["left"],
            joint_controll_prim=self.gripper_controll_joint["left"],
            gripper_type=self.gripper_type,
            gripper_max_force=self.gripper_max_force,
            robot_name=self.robot_name,
        )
        self.gripper_L.initialize(
            articulation_apply_action_func=robot.apply_action,
            get_joint_positions_func=robot.get_joint_positions,
            set_joint_positions_func=robot.set_joint_positions,
            dof_names=robot.dof_names,
        )
        self.gripper_R = ParallelGripper(
            end_effector_prim_path=right_end_effector_prim_path,
            joint_prim_names=self.finger_names["right"],
            joint_closed_velocities=self.closed_velocities["right"],
            joint_closed_positions=self.closed_positions["right"],
            joint_opened_positions=self.opened_positions["right"],
            joint_controll_prim=self.gripper_controll_joint["right"],
            gripper_type=self.gripper_type,
            gripper_max_force=self.gripper_max_force,
            robot_name=self.robot_name,
        )
        self.gripper_R.initialize(
            articulation_apply_action_func=robot.apply_action,
            get_joint_positions_func=robot.get_joint_positions,
            set_joint_positions_func=robot.set_joint_positions,
            dof_names=robot.dof_names,
        )
        return robot

    def _get_gripper_action(self, state: str, isRight: bool):
        if isRight:
            self.gripper_state_R = state
            action = self.gripper_R.forward(action=self.gripper_state_R)
            return action
        else:
            self.gripper_state_L = state
            action = self.gripper_L.forward(action=self.gripper_state_L)
            return action

    def _set_gripper_state(self, state: str, isRight: bool, width):
        self.robot = self._init_grippers()
        action = self._get_gripper_action(state, isRight)
        self.robot.apply_action(action)

    # Get pose of any object, Input: prim_path
    def _get_object_pose(self, object_prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
        for value in self.articulat_objects.values():
            value.initialize()
        if object_prim_path == "robot":
            position, rotation = self.usd_objects["robot"].get_world_pose()
        else:
            target_object = XFormPrim(prim_path=object_prim_path)
            position, rotation = target_object.get_world_pose()
        for value in self.articulat_objects.values():
            value.initialize()
        return position, rotation

    def _get_ee_pose(self, is_right: bool) -> Tuple[np.ndarray, np.ndarray]:
        position, rotation_matrix = self.ui_builder._get_ee_pose(is_right)
        rotation = mat2quat_wxyz(rotation_matrix)
        return position, rotation

    def _get_ik_status(self, target_poses, isRight, ObsAvoid=False, output_link_pose=False):
        joint_positions = {}
        if not ObsAvoid:
            time00 = time.time()
            results = []
            for i in range(target_poses.shape[0]):
                target_position = target_poses[i, :3, 3]
                target_rotation = mat2quat_wxyz(target_poses[i, :3, :3])
                is_success, joint_state = self.ui_builder._get_ik_status(
                    target_position, target_rotation, isRight
                )
                all_names = self.ui_builder.articulation.dof_names
                for i, idx in enumerate(joint_state.joint_indices):
                    joint_positions[all_names[idx]] = joint_state.joint_positions[i]
                results.append((is_success, joint_positions))
            time01 = time.time()
            logger.info(f"ik lula time cost{time01 - time00}")
        else:
            time0 = time.time()
            init_rotation_matrix = quat2mat_wxyz(self.robot_init_rotation)
            robot_translation_matrix = np.zeros((4, 4))
            robot_translation_matrix[:3, :3] = init_rotation_matrix
            robot_translation_matrix[:3, 3] = self.robot_init_position
            robot_translation_matrix[3, 3] = 1
            target_poses_local = (
                np.linalg.inv(robot_translation_matrix[np.newaxis, ...]) @ target_poses
            )
            target_rotations_local = batch_matrices_to_quaternions_scipy_w_first(target_poses_local)
            target_positions_local = target_poses_local[:, :3, 3]
            if isinstance(self.end_effector_name, dict):
                end_effector_name = self.end_effector_name["left"]
                if isRight:
                    end_effector_name = self.end_effector_name["right"]
            else:
                end_effector_name = self.end_effector_name
            time.time()
            curoboMotion = self.ui_builder.get_curobo_motion(isRight)
            self.ui_builder.set_locked_joint_positions(isRight)
            time.time()
            results = curoboMotion.solve_batch_ik(
                target_positions_local,
                target_rotations_local,
                end_effector_name,
                output_link_pose=output_link_pose,
            )
            time_3 = time.time()
            logger.info(f"ik curobo time cost{time_3 - time0}")
        return results
