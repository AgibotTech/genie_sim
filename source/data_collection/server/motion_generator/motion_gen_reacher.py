# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import re
import time
from typing import Optional

import carb
import numpy as np
import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper, get_prim_world_pose
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenPlanConfig, PoseCostMetric
from isaacsim.core.api import World
from isaacsim.core.api.objects import sphere
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.types import ArticulationAction

from common.base_utils.logger import logger
from server.motion_generator.mesh_utils import get_mesh_attrs, simplify_obstacles_from_stage
from server.motion_generator.path_filters import (
    filter_paths_by_position_error,
    filter_paths_by_rotation_error,
    sort_by_difference_js,
)

# USD imports (using try-except for optional dependency)
try:
    from pxr import UsdGeom, UsdPhysics
except ImportError:
    # These will be imported dynamically when needed
    pass


CUROBO_BATCH_SIZE = 20
MAX_MESH_FACES = 1000  # Maximum face count limit


class AgibotUsdHelper(UsdHelper):
    """Subclass of UsdHelper that uses the instance xform cache and the
    local get_mesh_attrs implementation for mesh extraction.

    This lets us reuse UsdHelper's transform cache (`self._xform_cache`) while
    customizing mesh handling (triangulation / fallback logic) in the local
    `get_mesh_attrs` function.
    """

    def get_obstacles_from_stage(
        self,
        only_paths: Optional[list] = None,
        ignore_paths: Optional[list] = None,
        only_substring: Optional[list] = None,
        ignore_substring: Optional[list] = None,
        reference_prim_path: Optional[str] = None,
        timecode: float = 0,
    ) -> WorldConfig:
        obstacles = {
            "cuboid": None,
            "sphere": None,
            "mesh": None,
            "cylinder": None,
            "capsule": None,
        }

        r_T_w = None
        # use the instance xform cache
        try:
            self._xform_cache.Clear()
            self._xform_cache.SetTime(timecode)
        except Exception:
            # fallback: create local cache
            try:
                self._xform_cache = UsdGeom.XformCache(timecode)
            except Exception:
                pass

        if reference_prim_path is not None:
            reference_prim = self.stage.GetPrimAtPath(reference_prim_path)
            r_T_w, _ = get_prim_world_pose(self._xform_cache, reference_prim, inverse=True)

        # iterate stage prims (use Traverse for full traversal)
        all_items = self.stage.Traverse()
        for prim in all_items:
            prim_path = str(prim.GetPath())

            # only/ignore path filters
            if only_paths is not None and not any([prim_path.startswith(k) for k in only_paths]):
                continue
            if ignore_paths is not None and any([prim_path.startswith(k) for k in ignore_paths]):
                continue
            if only_substring is not None and not any([k in prim_path for k in only_substring]):
                continue
            if ignore_substring is not None and any([k in prim_path for k in ignore_substring]):
                continue

            # Optionally check for collision enabled attribute
            try:
                collisionAPI = UsdPhysics.CollisionAPI.Get(self.stage, prim_path)
                if collisionAPI and not collisionAPI.GetCollisionEnabledAttr().Get():
                    # skip prims that explicitly disable collision
                    continue
            except Exception:
                # if we can't query collision API, proceed normally
                pass

            try:
                if prim.IsA(UsdGeom.Mesh):
                    if obstacles["mesh"] is None:
                        obstacles["mesh"] = []
                    # use the local get_mesh_attrs (triangulating wrapper)
                    m_data = get_mesh_attrs(prim, cache=self._xform_cache, transform=r_T_w)
                    if m_data is not None:
                        obstacles["mesh"].append(m_data)
            except Exception as e:
                logger.error(f"Error extracting prim {prim_path}: {e}")
                continue

        world_model = WorldConfig(**obstacles)
        return world_model


class CuroboMotion:
    world_coll_checker = None
    cached_obstacle_info = {}
    curobo_kinematics = None
    curobo_kinematics_robot_cfg = {}

    def reset(self):
        self.motion_gen.clear_world_cache()

    def reset_link(self):
        self.link_names = self.motion_gen.kinematics.link_names
        self.ee_link_name = self.motion_gen.kinematics.ee_link
        for i in self.link_names:
            self.target_links[i] = XFormPrim(
                "/World/target_" + i,
                position=self.init_ee_pose[i]["position"],
                orientation=self.init_ee_pose[i]["orientation"],
            )

    def _get_curobo_kinematics(self):
        if (
            CuroboMotion.curobo_kinematics is None
            and hasattr(self, "robot_cfg")
            and self.robot_cfg is not None
        ):
            CuroboMotion.curobo_kinematics_robot_cfg = robot_cfg = copy.deepcopy(self.robot_cfg)
            if robot_cfg["kinematics"].get("link_names", None) is None:
                robot_cfg["kinematics"]["link_names"] = []
            for link_name in robot_cfg["kinematics"]["collision_link_names"]:
                if "arm" in link_name:
                    robot_cfg["kinematics"]["link_names"].append(link_name)
            cuda_robot_model_config = CudaRobotModelConfig.from_data_dict(
                data_dict=robot_cfg, tensor_args=self.tensor_args
            )
            CuroboMotion.curobo_kinematics = CudaRobotModel(cuda_robot_model_config)
        return CuroboMotion.curobo_kinematics

    def _extract_cached_obstacles(self, need_reset_cache=True):
        """
        Extract and cache all geometry information during initialization, only update poses later
        """
        logger.info("Extracting and caching obstacle geometries...")

        # Initialize ignore list
        self.ignore_substring_list = [
            self.robot_prim_path,
            "/World/target",
            "/World/Xform_01",
            "/World/GroundPlane",
            "/World/Environment_01",
            "/curobo",
            "/World/GroundPlane_01",
            "/World/Meshes",
            "/World/Root/Meshes",
            "/World/Objects/part",
            "/base_cube",
            "virtual_fixed_joint",
            "/World/background",
            "/World/Background",
        ]
        if need_reset_cache:
            # Get initial obstacle configuration (without pose transformation)
            initial_obstacles = self.usd_help.get_obstacles_from_stage(
                only_paths=None,
                ignore_substring=self.ignore_substring_list,
                reference_prim_path=None,
                timecode=0,
            )

            # Simplify geometry
            time0 = time.time()
            simplified_obstacles = simplify_obstacles_from_stage(
                initial_obstacles, max_faces=MAX_MESH_FACES
            )
            time1 = time.time()
            logger.info(f"extract simplified geometry time: {time1 - time0} seconds")

            # Cache obstacle geometry information and corresponding prim paths
            CuroboMotion.cached_obstacle_info = {}

            # Cache each type of geometry
            if simplified_obstacles.mesh:
                for mesh in simplified_obstacles.mesh:
                    prim_path = mesh.name
                    CuroboMotion.cached_obstacle_info[prim_path] = {
                        "type": "mesh",
                        "geometry": mesh,
                        "original_pose": mesh.pose,  # Save original pose as reference
                    }

            logger.info(
                f"Cache completed, extracted {len(CuroboMotion.cached_obstacle_info)} obstacle geometries"
            )

        # Get robot reference coordinate system transformation
        if self.robot_prim_path:
            reference_prim = self.usd_help.stage.GetPrimAtPath(self.robot_prim_path)
            self.robot_transform_cache = self.usd_help._xform_cache
            self.robot_transform_cache.Clear()
            self.robot_transform_cache.SetTime(0)
            self.robot_reference_prim = reference_prim
        else:
            self.robot_transform_cache = None
            self.robot_reference_prim = None

    def add_obstacle_from_prim_path(self, prim_path, usd_path):
        """
        Manually add obstacles under specified prim path to cache
        Use get_obstacles_from_stage's only_paths feature to extract only obstacles under specific path

        Args:
            prim_path (str): Prim path of the object to add

        Returns:
            int: Number of newly added obstacles
        """
        try:
            logger.info(f"Adding obstacles under prim path {prim_path}...")

            # Use only_paths parameter to extract only obstacles under specified path
            new_obstacles = self.usd_help.get_obstacles_from_stage(
                only_paths=[prim_path],
                ignore_substring=self.ignore_substring_list,
                reference_prim_path=None,
                timecode=0,
            )

            # Simplify geometry
            time0 = time.time()
            simplified_obstacles = simplify_obstacles_from_stage(new_obstacles, usd_path=usd_path)
            time1 = time.time()
            logger.info(f"Simplified geometry time: {time1 - time0:.4f} seconds")
            new_obstacles_count = 0

            # Check each type of obstacle
            obstacle_types = [
                ("mesh", simplified_obstacles.mesh),
            ]

            for obstacle_type, obstacles_list in obstacle_types:
                if obstacles_list:
                    for obstacle in obstacles_list:
                        obstacle_prim_path = obstacle.name

                        # Add to cache (update if already exists)
                        if obstacle_prim_path not in CuroboMotion.cached_obstacle_info:
                            new_obstacles_count += 1
                            logger.info(
                                f"New obstacle: {obstacle_prim_path} (type: {obstacle_type})"
                            )
                        else:
                            logger.info(
                                f"Updated existing obstacle: {obstacle_prim_path} (type: {obstacle_type})"
                            )

                        CuroboMotion.cached_obstacle_info[obstacle_prim_path] = {
                            "type": obstacle_type,
                            "geometry": obstacle,
                            "original_pose": obstacle.pose,
                        }

            if new_obstacles_count > 0:
                logger.info(f"Successfully added {new_obstacles_count} new obstacles to cache")
            else:
                logger.info(f"No new obstacles found under path {prim_path}")

            logger.info(f"Current total cache count: {len(CuroboMotion.cached_obstacle_info)}")
            return new_obstacles_count

        except Exception as e:
            logger.info(f"Error adding obstacles from prim path {prim_path}: {e}")
            return 0

    def add_obstacles_from_prim_paths(self, prim_paths):
        """
        Batch add obstacles under multiple prim paths to cache

        Args:
            prim_paths (list): List of prim paths of objects to add

        Returns:
            int: Total number of newly added obstacles
        """
        total_added = 0
        for prim_path in prim_paths:
            added_count = self.add_obstacle_from_prim_path(prim_path)
            total_added += added_count

        logger.info(f"Batch addition completed, total {total_added} new obstacles added")
        return total_added

    def get_cached_obstacles_info(self):
        """
        Get currently cached obstacle information

        Returns:
            dict: Cached obstacle information, format: {prim_path: {'type': str, 'geometry': obj}}
        """
        info = {}
        for prim_path, obstacle_info in CuroboMotion.cached_obstacle_info.items():
            info[prim_path] = {
                "type": obstacle_info["type"],
                "geometry_name": obstacle_info["geometry"].name,
            }
        return info

    def refresh_obstacle_cache(self):
        """
        Force re-extract and cache obstacle geometry information
        Called when new obstacles are added or removed in the scene
        """
        logger.info("Re-extracting obstacle cache...")
        self._extract_cached_obstacles()

    def __init__(
        self,
        name: str,
        robot: Articulation,
        world: World,
        robot_cfg,
        robot_prim_path,
        robot_list,
        step=100,
        debug=False,
    ):
        self.name = name
        self.debug = debug
        self.usd_help = AgibotUsdHelper()
        self.target_pose = None
        self.target_orientation = None
        self.past_pose = None
        self.past_orientation = None
        self.robot_list = robot_list
        tensor_args = TensorDeviceType()
        self.robot_prim_path = robot_prim_path
        n_obstacle_cuboids = 30
        self.init_ee_pose = {}
        n_obstacle_mesh = 30

        robot_cfg_path = get_robot_configs_path()
        self.robot_cfg = load_yaml(join_path(robot_cfg_path, robot_cfg))["robot_cfg"]
        self.robot_cfg["kinematics"]["extra_collision_spheres"] = {
            "attached_object": 30,
            "left_attached_object": 30,
        }
        self.lock_joints = self.robot_cfg["kinematics"]["lock_joints"]
        self.lock_js_names = []
        if self.lock_joints:
            for key in self.lock_joints:
                self.lock_js_names.append(key)
        j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]
        self.collision_link_names = self.robot_cfg["kinematics"]["collision_link_names"]

        self.world_cfg = WorldConfig()
        motion_gen_config = MotionGen.load_from_robot_config(
            robot_cfg=self.robot_cfg,
            world_model=self.world_cfg,
            tensor_args=tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            num_trajopt_seeds=4,
            num_graph_seeds=4,
            num_ik_seeds=32,
            num_batch_ik_seeds=32,
            interpolation_dt=0.01,
            interpolation_steps=5000,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            optimize_dt=True,
            trajopt_dt=None,
            trajopt_tsteps=step,
            num_trajopt_noisy_seeds=1,
            num_batch_trajopt_seeds=1,
            collision_activation_distance=0.01,
            world_coll_checker=CuroboMotion.world_coll_checker,
        )

        self.tensor_args = tensor_args
        self.motion_gen = MotionGen(motion_gen_config)
        if CuroboMotion.world_coll_checker is None:
            CuroboMotion.world_coll_checker = self.motion_gen.world_coll_checker
        self.motion_gen.warmup(parallel_finetune=True, batch=CUROBO_BATCH_SIZE)
        self.world_model = self.motion_gen.world_collision
        self.plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_opt=True,
            need_graph_success=True,
            enable_graph_attempt=5,
            max_attempts=40,
            enable_finetune_trajopt=True,
            parallel_finetune=True,
            time_dilation_factor=1.0,
            ik_fail_return=5,
            success_ratio=0.5,
        )
        self.target = XFormPrim(
            "/World/target",
            position=np.array([0.5, 0, 0.5]),
            orientation=np.array([0, 1, 0, 0]),
        )
        self.cmd_plan = None
        self.cmd_plans = []
        self.cmd_idx = 0
        self.num_targets = 0
        self.past_cmd = None
        self.pose_metic = None
        self.robot = robot
        self.robot._articulation_view.initialize()
        self.idx_list = [self.robot.get_dof_index(x) for x in j_names]
        self.robot.set_joint_positions(default_config, self.idx_list)
        self.robot._articulation_view.set_max_efforts(
            values=np.array([5000 for i in range(len(self.idx_list))]),
            joint_indices=self.idx_list,
        )
        self.reached = False
        self.success = False
        self.saved_poses = []
        self.my_world = World(stage_units_in_meters=1.0)
        stage = self.my_world.stage
        self.usd_help.load_stage(stage)
        self.time_index = 0
        self.spheres = None
        self.obstacle_spheres = None

        # Maintain list of objects attached to robot
        self.attached_objects = []

        # First extract and cache obstacle geometry information
        self._extract_cached_obstacles(need_reset_cache=CuroboMotion.cached_obstacle_info == {})

        # Then set obstacles (will use cache now)
        self.set_obstacles()
        self.link_names = self.motion_gen.kinematics.link_names
        self.ee_link_name = self.motion_gen.kinematics.ee_link
        kin_state = self.motion_gen.kinematics.get_state(
            self.motion_gen.get_retract_config().view(1, -1)
        )
        link_retract_pose = kin_state.link_pose
        self.target_links = {}
        for i in self.link_names:
            k_pose = np.ravel(link_retract_pose[i].to_list())
            self.target_links[i] = XFormPrim(
                "/World/target_" + i,
                position=np.array(k_pose[:3]),
                orientation=np.array(k_pose[3:]),
            )
        self.lock_joint_states = None

    def set_obstacles(self):
        """
        Quickly update obstacle poses using cached geometry information
        """
        import time

        start_time = time.time()

        # Use cached geometry to quickly update poses
        updated_obstacles, has_update = self._update_obstacle_poses_fast()
        if has_update:
            # Create WorldConfig
            world_config = WorldConfig(**updated_obstacles)

            # Convert to collision detection world
            obstacle = world_config.get_collision_check_world()
            self.world_cfg = obstacle  # NOTE this world_config only affects visualization
            self.motion_gen.world_coll_checker.load_collision_model(
                obstacle, fix_cache_reference=self.motion_gen.use_cuda_graph
            )

        self.motion_gen.graph_planner.reset_buffer()

        elapsed_time = time.time() - start_time
        logger.info(
            f"Fast obstacle update completed, time: {elapsed_time:.4f}s, needs update: {has_update}"
        )

    def _update_obstacle_poses_fast(self):
        """
        Core method for quickly updating obstacle poses
        """
        try:
            pass

            import torch
            from curobo.types.math import Pose
            from curobo.util.usd_helper import get_prim_world_pose
        except ImportError as e:
            logger.info(f"Import error: {e}")
            raise e

        # Get current transformation of robot reference coordinate system
        r_T_w = None
        if self.robot_reference_prim and self.robot_transform_cache:
            self.robot_transform_cache.Clear()
            self.robot_transform_cache.SetTime(0)  # Can be changed to current time
            r_T_w, _ = get_prim_world_pose(
                self.robot_transform_cache, self.robot_reference_prim, inverse=True
            )

        # Prepare updated obstacle list
        updated_obstacles = {
            "cuboid": [],
            "sphere": [],
            "mesh": [],
            "cylinder": [],
            "capsule": [],
        }

        has_update = False
        # Iterate through cached obstacle information, update each pose
        for prim_path, obstacle_info in CuroboMotion.cached_obstacle_info.items():
            try:
                # Skip objects attached to robot
                is_attached = False
                for attached_path in self.attached_objects:
                    if prim_path.startswith(attached_path):
                        is_attached = True
                        break
                if is_attached:
                    continue
                # Get current prim
                prim = self.usd_help.stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue

                # Get current world pose
                current_mat, _ = get_prim_world_pose(self.robot_transform_cache, prim)

                # Apply robot coordinate system transformation
                if r_T_w is not None:
                    current_mat = r_T_w @ current_mat

                # Convert to pose
                tensor_mat = torch.as_tensor(current_mat, device=torch.device("cuda", 0))
                updated_pose = Pose.from_matrix(tensor_mat).tolist()

                # Calculate distance of object relative to robot
                object_position = np.array(updated_pose[:3])
                distance_to_robot = np.linalg.norm(object_position)

                # Skip obstacle if distance is greater than 10 meters
                if distance_to_robot > 10.0:
                    continue

                # Create updated obstacle copy
                geometry = obstacle_info["geometry"]
                if not has_update and geometry.pose != updated_pose:
                    has_update = True
                geometry.pose = updated_pose

                # Add to corresponding type list
                obstacle_type = obstacle_info["type"]
                updated_obstacles[obstacle_type].append(geometry)

            except Exception as e:
                logger.info(f"Error updating obstacle {prim_path} pose: {e}")
                continue
        return updated_obstacles, has_update

    def visualize_spheres(
        self,
        sph_list,
        spheres_buffer,
        prim_prefix="/curobo/robot_sphere_",
        color=np.array([0, 0.8, 0.2]),
    ):
        robot_prim_path = self.robot.prim_path
        if spheres_buffer is None:
            spheres_buffer = []
            for si, s in enumerate(sph_list[0]):
                sp = sphere.VisualSphere(
                    prim_path=robot_prim_path + prim_prefix + str(si),
                    radius=float(s.radius),
                    color=color,
                )
                sp.set_local_pose(
                    translation=np.array([s.position[0], s.position[1], s.position[2]])
                )
                spheres_buffer.append(sp)
        else:
            if len(spheres_buffer) < len(sph_list[0]):
                for si in range(len(spheres_buffer), len(sph_list[0])):
                    sp = sphere.VisualSphere(
                        prim_path=robot_prim_path + prim_prefix + str(si),
                        radius=0.01,
                        color=color,
                    )
                    spheres_buffer.append(sp)
            for si, s in enumerate(sph_list[0]):
                if not np.isnan(s.position[0]):
                    spheres_buffer[si].set_local_pose(
                        translation=np.array([s.position[0], s.position[1], s.position[2]])
                    )
                    spheres_buffer[si].set_radius(float(s.radius))

    def visualize_obstacles(self):
        sph_list = []
        for obs in self.world_cfg.objects:
            sph = obs.get_bounding_spheres(
                300,
                surface_sphere_radius=0.001,
                pre_transform_pose=None,
                tensor_args=self.tensor_args,
            )
            sph_list += sph
        self.visualize_spheres(
            [sph_list],
            self.obstacle_spheres,
            prim_prefix="/curobo/obstacle_sphere_",
            color=np.array([1.0, 0.25, 0.25]),
        )

    def visualize_robot_spheres(self):
        sim_js = self.robot.get_joints_state()
        sim_js_names = self.robot.dof_names
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities),
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js.acceleration *= 0.0
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        sph_list = self.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
        self.visualize_spheres(sph_list, self.spheres, prim_prefix="/curobo/robot_sphere_")

    def kinematic_forward(self, joint_states, output_link_names=None):
        # joint_positions should [batch_size, dof]
        t1 = time.time()
        result = self._get_curobo_kinematics().compute_kinematics(joint_states)
        links_position = result.links_position
        links_quaternion = result.links_quaternion
        link_names = result.link_names
        output = []
        for i in range(links_position.shape[0]):
            tmp_output = {}
            for j, link_name in enumerate(link_names):
                if (
                    output_link_names is None
                    or not len(output_link_names)
                    or link_name in output_link_names
                ):
                    tmp_output[link_name] = [
                        links_position[i][j].cpu().numpy().tolist(),
                        links_quaternion[i][j].cpu().numpy().tolist(),
                    ]
            output.append(tmp_output)
        t2 = time.time()
        logger.info(f"kinematic forward time{t2 - t1}")
        return output

    def solve_batch_ik(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        active_ee_name: str,
        output_link_pose=False,
    ):
        carb.log_warn("len positions is {}".format(len(positions)))
        t1 = time.time()
        results = []
        link_poses = None
        batch_size = CUROBO_BATCH_SIZE
        # Split first dimension of positions and rotations by batch_size, if last group is insufficient, pad by repeating
        pos_num = positions.shape[0]
        num_splits = (pos_num + batch_size - 1) // batch_size
        remainder = pos_num % batch_size
        if remainder != 0:
            padding = batch_size - remainder
            positions = np.concatenate([positions, np.tile(positions[-1:], (padding, 1))], axis=0)
            rotations = np.concatenate([rotations, np.tile(rotations[-1:], (padding, 1))], axis=0)
        position_batched = np.array_split(positions, num_splits)
        rotation_batched = np.array_split(rotations, num_splits)
        if len(self.link_names) > 1:
            link_poses = {}
            for i in self.target_links.keys():
                c_p, c_rot = self.target_links[i].get_world_pose()
                link_poses[i] = Pose(
                    position=self.tensor_args.to_device(np.tile(c_p, (batch_size, 1))),
                    quaternion=self.tensor_args.to_device(np.tile(c_rot, (batch_size, 1))),
                    batch=batch_size,
                )
        ee_c_p, ee_c_rot = self.target_links[self.ee_link_name].get_world_pose()
        goal_pose = Pose(
            position=self.tensor_args.to_device(np.tile(ee_c_p, (batch_size, 1))),
            quaternion=self.tensor_args.to_device(np.tile(ee_c_rot, (batch_size, 1))),
            batch=batch_size,
        )
        if output_link_pose:
            self.update_curobo_kinematics_lock_joints(self.lock_joint_states)
        for i in range(num_splits):
            pos_batch = position_batched[i]
            rot_batch = rotation_batched[i]

            if link_poses and active_ee_name in link_poses:
                link_poses[active_ee_name] = Pose(
                    position=self.tensor_args.to_device(pos_batch),
                    quaternion=self.tensor_args.to_device(rot_batch),
                    batch=batch_size,
                )
            if active_ee_name == self.ee_link_name:
                goal_pose = Pose(
                    position=self.tensor_args.to_device(pos_batch),
                    quaternion=self.tensor_args.to_device(rot_batch),
                    batch=batch_size,
                )
            t00 = time.time()
            result = self.motion_gen.ik_solver.solve_batch(goal_pose, link_poses=link_poses)
            t11 = time.time()
            carb.log_warn("ik batch {} time is {}".format(i, t11 - t00))
            if output_link_pose:
                js = result.js_solution.get_ordered_joint_state(
                    self._get_curobo_kinematics().kinematics_config.joint_names
                )
                js = js.squeeze(1)
                ik_link_poses = self.kinematic_forward(js)
            for k in range(result.success.shape[0]):
                joint_positions = {}
                for j, name in enumerate(result.js_solution.joint_names):
                    joint_positions[name] = result.js_solution.position[k][0].cpu().tolist()[j]
                if output_link_pose:
                    results.append([result.success[k], joint_positions, ik_link_poses[k]])
                else:
                    results.append((result.success[k], joint_positions))
        t2 = time.time()
        carb.log_warn("ik time is {}".format(t2 - t1))
        return results[:pos_num]

    def update_lock_joints(self, locked_joints):
        if (
            self.lock_joint_states is None
            or np.abs(
                np.array(list(self.lock_joint_states.values()))
                - np.array(list(locked_joints.values()))
            ).max()
            > 1e-3
        ):
            before = time.time()
            self.motion_gen.update_locked_joints(locked_joints, self.robot_cfg)
            self.lock_joint_states = locked_joints
            after = time.time()
            carb.log_warn("update lock joints time is {}".format(after - before))
        else:
            carb.log_warn("lock joints is the same, no need to update")

    def update_curobo_kinematics_lock_joints(self, locked_joints):
        before = time.time()
        if CuroboMotion.curobo_kinematics is not None and locked_joints is not None:
            if (
                CuroboMotion.curobo_kinematics_robot_cfg["kinematics"]["lock_joints"]
                != locked_joints
            ):
                logger.info("update kinematics lock joints")
                CuroboMotion.curobo_kinematics_robot_cfg["kinematics"][
                    "lock_joints"
                ] = locked_joints
                robot_cfg = RobotConfig.from_dict(
                    CuroboMotion.curobo_kinematics_robot_cfg, self.tensor_args
                )
                CuroboMotion.curobo_kinematics.update_kinematics_config(
                    robot_cfg.kinematics.kinematics_config
                )
        after = time.time()
        carb.log_warn("update curobo kinematics lock joints time is {}".format(after - before))

    def view_debug_world(self):
        if self.debug:
            self.visualize_robot_spheres()
            self.visualize_obstacles()

    def caculate_ik_goal(
        self,
        goal_offset=[0, 0, 0, 1, 0, 0, 0],
        path_constraint=None,
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links: list[str] = [],
        from_current_pose=False,
    ):
        """
        goal_offset: Offset pose from the goal pose. Reference frame is the goal
                    pose frame if constraint_in_goal_frame is True, otherwise the
                    reference frame is the robot base frame.
        path_constraint:  Path constraint for the approach to goal pose and
                    goal to retract path. This is a list of 6 values, where each value is a weight
                    for each Cartesian dimension. The first three are for orientation and the last
                    three are for position. If None, no path constraint is applied. e.g. [0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
        constraint_in_goal_frame:If True, the goal offset is in the
                    goal pose frame. If False, the goal offset is in the robot base frame.
                    Also applies to path_constraint.
        disable_collision_links: Name of links to disable collision with the world.
        from_current_pose: If True, the goal pose is the current ee pose + offset.
        """
        t0 = time.time()
        self.reached = False
        if from_current_pose and not goal_offset:
            self.reached = True
            self.success = False
            carb.log_warn("from_current_pose is True, but goal_offset is None, return")
            return
        cube_position, cube_orientation = self.target.get_world_pose()
        if self.past_pose is None:
            self.past_pose = cube_position
        if self.target_pose is None:
            self.target_pose = cube_position
        if self.target_orientation is None:
            self.target_orientation = cube_orientation
        if self.past_orientation is None:
            self.past_orientation = cube_orientation
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        sim_js_names = []
        lock_idx = []
        sim_js_positions = []
        sim_js_velocities = []
        for idx, name in enumerate(js_names):
            if name not in self.lock_js_names:
                sim_js_names.append(name)
            else:
                lock_idx.append(idx)
        for idx, position in enumerate(sim_js.positions):
            if idx not in lock_idx:
                sim_js_positions.append(position)
        for idx, velocity in enumerate(sim_js.velocities):
            if idx not in lock_idx:
                sim_js_velocities.append(velocity)
        sim_js_positions = np.array(sim_js_positions)[np.newaxis, :]
        sim_js_velocities = np.array(sim_js_velocities)[np.newaxis, :]
        cu_js = JointState(
            position=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))),
            velocity=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1)))
            * 0.0,
            acceleration=self.tensor_args.to_device(
                np.tile(sim_js_velocities, (CUROBO_BATCH_SIZE, 1))
            )
            * 0.0,
            jerk=self.tensor_args.to_device(np.tile(sim_js_velocities, (CUROBO_BATCH_SIZE, 1)))
            * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        start_time = time.time()
        if from_current_pose:
            start_pose = self.motion_gen.compute_kinematics(cu_js).ee_pose.clone()
            cube_position = start_pose.position.squeeze()[0].cpu().numpy()
            cube_orientation = start_pose.quaternion.squeeze()[0].cpu().numpy()

        ee_translation_goal = cube_position
        ee_orientation_teleop_goal = cube_orientation
        if goal_offset is not None:
            offset = Pose.from_list(goal_offset)
            goal_pose_list = np.concatenate(
                [ee_translation_goal, ee_orientation_teleop_goal]
            ).tolist()
            goal_pose = Pose.from_list(goal_pose_list)
            if offset_and_constraint_in_goal_frame:
                offset_goal_pose = goal_pose.clone().multiply(offset)
            else:
                offset_goal_pose = offset.clone().multiply(goal_pose.clone())
            ee_translation_goal = offset_goal_pose.position.squeeze().cpu().numpy()
            ee_orientation_teleop_goal = offset_goal_pose.quaternion.squeeze().cpu().numpy()
        ik_goal = Pose(
            position=self.tensor_args.to_device(
                np.tile(ee_translation_goal, (CUROBO_BATCH_SIZE, 1))
            ),
            quaternion=self.tensor_args.to_device(
                np.tile(ee_orientation_teleop_goal, (CUROBO_BATCH_SIZE, 1))
            ),
            batch=CUROBO_BATCH_SIZE,
        )
        if path_constraint is not None and len(path_constraint) == 6:
            hold_pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.tensor_args.to_device(path_constraint),
                project_to_goal_frame=offset_and_constraint_in_goal_frame,
            )
            self.plan_config.pose_cost_metric = hold_pose_cost_metric
        else:
            logger.info("no valid path constraint provided")
            self.plan_config.pose_cost_metric = self.pose_metic
        link_poses = None
        if self.plan_config.pose_cost_metric:
            update_res = self.motion_gen.update_pose_cost_metric(
                self.plan_config.pose_cost_metric, cu_js, ik_goal
            )
            if not update_res:
                self.reached = True
                self.success = False
                carb.log_warn("update pose cost metric failed")
                return
        disable_collision_links = list(
            filter(
                lambda link: any(re.match(pattern, link) for pattern in disable_collision_links),
                self.collision_link_names,
            )
        )
        self.motion_gen.toggle_link_collision(disable_collision_links, False)
        try:
            result = self.motion_gen.plan_batch(
                cu_js,
                ik_goal,
                self.plan_config.clone(),
                link_poses=link_poses,
            )
            if result.success.any():
                self.reached = False
                self.success = True
                carb.log_warn("end_time is{}".format(time.time() - start_time))
                self.num_targets += 1
                paths = result.get_successful_paths()
                position_filter_res = filter_paths_by_position_error(
                    paths, result.position_error[result.success]
                )
                rotation_filter_res = filter_paths_by_rotation_error(
                    paths, result.rotation_error[result.success]
                )
                filtered_paths = []
                for i in range(len(paths)):
                    if position_filter_res[i] and rotation_filter_res[i]:
                        filtered_paths.append(paths[i])
                if len(filtered_paths) == 0:
                    filtered_paths = paths
                sorted_indices = sort_by_difference_js(
                    filtered_paths,
                    weights=self.tensor_args.to_device(
                        [
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            3.0,
                            3.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            3.0,
                            3.0,
                            1.0,
                        ]
                    ),
                )
                self.cmd_plan = paths[sorted_indices[0]]
                self.cmd_plan = self.motion_gen.get_full_js(self.cmd_plan)
                logger.info(len(self.cmd_plan))
                self.idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in self.cmd_plan.joint_names:
                        self.idx_list.append(self.robot.get_dof_index(x))
                        common_js_names.append(x)
                self.cmd_plan = self.cmd_plan.get_ordered_joint_state(common_js_names)
                self.cmd_idx = 0
                carb.log_warn("success")
            else:
                self.reached = True
                self.success = False
                carb.log_warn("plan did not converge to a solution: {}".format(str(result.status)))
        except Exception as e:
            self.reached = True
            self.success = False
            carb.log_warn("plan got an exception: {}".format(str(e)))
        self.motion_gen.toggle_link_collision(disable_collision_links, True)
        self.target_pose = cube_position
        self.target_orientation = cube_orientation
        self.past_pose = cube_position
        self.past_orientation = cube_orientation
        t1 = time.time()
        carb.log_warn("total time is {}".format(t1 - t0))

    def exclude_js(self, joint_names):
        if self.cmd_plan:
            positions = []
            velocities = []
            self.idx_list = []
            for name in joint_names:
                self.idx_list.append(self.robot.get_dof_index(name))
            for index, pos in enumerate(self.cmd_plan.position):
                position = []
                velocity = []
                for name in joint_names:
                    idx = self.cmd_plan.joint_names.index(name)
                    position.append(pos.cpu().numpy()[idx])
                    velocity.append(self.cmd_plan.velocity.cpu().numpy()[index][idx])
                positions.append(position)
                velocities.append(velocity)
            self.cmd_plan = JointState(
                position=self.tensor_args.to_device(positions),
                velocity=self.tensor_args.to_device(velocities) * 0.0,
                acceleration=self.tensor_args.to_device(velocities) * 0.0,
                jerk=self.tensor_args.to_device(velocities) * 0.0,
                joint_names=joint_names,
            )

    def detach_obj(self):
        self.motion_gen.detach_object_from_robot("left_attached_object")
        self.motion_gen.detach_object_from_robot()
        # Clear attached objects list
        self.attached_objects.clear()
        self.set_obstacles()

    def remove_objects_from_world(self, prim_paths):
        for x in prim_paths:
            obs = self.motion_gen.world_model.get_obstacle(x)
            if not obs:
                continue
            self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=x)
            self.motion_gen.world_model.remove_obstacle(x)

    def attach_obj(
        self,
        prim_paths,
        link_name="attached_object",
        ee_position=[0, 0, 0],
        ee_rotation=[1, 0, 0, 0],
    ):
        self.motion_gen.detach_object_from_robot()
        attach_result = False
        self.set_obstacles()
        logger.info(f"object_names={prim_paths}")
        ee_pose = Pose(
            position=self.tensor_args.to_device(ee_position),
            quaternion=self.tensor_args.to_device(ee_rotation),
        )
        attach_result = self.attach_objects_to_robot(
            object_names=prim_paths,
            link_name=link_name,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            surface_sphere_radius=0.005,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.005, 1, 0, 0, 0], self.tensor_args),
            remove_obstacles_from_world_config=True,
            ee_pose=ee_pose,
        )
        logger.info(f"attach_result={attach_result}")

        return attach_result

    def attach_objects_to_robot(
        self,
        object_names,
        surface_sphere_radius: float = 0.001,
        link_name: str = "attached_object",
        sphere_fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        world_objects_pose_offset=None,
        remove_obstacles_from_world_config: bool = False,
        ee_pose=None,
    ) -> bool:
        if world_objects_pose_offset is not None:
            ee_pose = world_objects_pose_offset.inverse().multiply(ee_pose)
        ee_pose = ee_pose.inverse()  # ee_T_w to multiply all objects later
        max_spheres = self.motion_gen.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(
            link_name
        )
        if len(object_names) == 0:
            return
        n_spheres = int(max_spheres / len(object_names))
        sphere_tensor = torch.zeros((max_spheres, 4))
        sphere_tensor[:, 3] = -10.0
        sph_list = []
        if n_spheres == 0:
            return False
        for i, object_name in enumerate(object_names):
            obs = self.motion_gen.world_model.get_obstacle(object_name)
            if not obs:
                continue
            sph = obs.get_bounding_spheres(
                n_spheres,
                surface_sphere_radius,
                pre_transform_pose=ee_pose,
                tensor_args=self.tensor_args,
                fit_type=sphere_fit_type,
                voxelize_method=voxelize_method,
            )
            sph_list += [s.position + [s.radius] for s in sph]
            self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=object_name)
            if remove_obstacles_from_world_config:
                self.motion_gen.world_model.remove_obstacle(object_name)
                if object_name not in self.attached_objects:
                    self.attached_objects.append(object_name)
        spheres = self.tensor_args.to_device(torch.as_tensor(sph_list))
        if not spheres.shape[0]:
            carb.log_warn("No spheres found for the given objects.")
            return False

        if spheres.shape[0] > max_spheres:
            spheres = spheres[: spheres.shape[0]]
        sphere_tensor[: spheres.shape[0], :] = spheres.contiguous()

        self.motion_gen.attach_spheres_to_robot(sphere_tensor=sphere_tensor, link_name=link_name)

        return True

    def get_articulation_action_without_lock_joints(self, cmd_state):
        tmp_idx_list = []
        for i in range(len(self.idx_list)):
            if cmd_state.joint_names[i] not in self.robot_cfg["kinematics"]["lock_joints"]:
                tmp_idx_list.append(i)

        art_action = ArticulationAction(
            cmd_state.position.cpu().numpy()[tmp_idx_list],
            cmd_state.velocity.cpu().numpy()[tmp_idx_list],
            joint_indices=np.array(self.idx_list)[tmp_idx_list],
        )
        return art_action

    def on_physics_step(self, run_ratio=1.0, additional_action: ArticulationAction = None):
        self.time_index += 1
        if run_ratio <= 0.0 or run_ratio > 1.0:
            carb.log_warn("run_ratio should be in the range (0, 1], setting to 1.0")
            run_ratio = 1.0

        if self.cmd_plan is not None:
            cmd_state = self.cmd_plan[self.cmd_idx]
            self.past_cmd = cmd_state.clone()
            art_action = self.get_articulation_action_without_lock_joints(cmd_state)
            # merge art_action with additional_action if provided
            if additional_action is not None and additional_action.joint_positions is not None:
                for idx in range(len(additional_action.joint_positions)):
                    if additional_action.joint_positions[idx] is None:
                        continue
                    if idx in art_action.joint_indices:
                        art_idx = art_action.joint_indices.index(idx)
                        art_action.joint_positions[art_idx] = additional_action.joint_positions[idx]
                        art_action.joint_velocities[art_idx] = additional_action.joint_velocities[
                            idx
                        ]
                    else:
                        art_action.joint_indices = np.append(art_action.joint_indices, idx)
                        art_action.joint_positions = np.append(
                            art_action.joint_positions,
                            additional_action.joint_positions[idx],
                        )
                        art_action.joint_velocities = np.append(
                            art_action.joint_velocities,
                            additional_action.joint_velocities[idx],
                        )
            self.robot.apply_action(art_action)
            self.cmd_idx += 1
            if self.cmd_idx >= len(self.cmd_plan.position) * run_ratio:
                self.cmd_idx = 0
                self.cmd_plan = None
                self.past_cmd = None
                self.reached = True
                logger.info(f"Reached{self.reached}")
