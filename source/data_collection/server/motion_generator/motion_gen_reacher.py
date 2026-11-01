# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import os
import yaml

from server.motion_generator.goal_pose_utils import select_additional_goal_base
# cuRobo v2 imports (0.8+). v1's MotionGen / PoseCostMetric / MotionGenPlanConfig /
# Use the cuRobo v2 kinematics, device, scene, and planner APIs.
# Ported from the v2 Isaac Sim motion-planning examples.
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene  # v2 alias for old WorldConfig / SceneCfg
from curobo.types import GoalToolPose, JointState, Pose
from curobo._src.robot.kinematics.kinematics_state import KinematicsState  # only for type hints

# ---------------------------------------------------------------------
# Normalize obstacle get_idx errors for cuRobo fallback dispatch.
# SceneData.enable_obstacle tries cuboid, mesh, and voxel stores in sequence
# and catches RuntimeError to proceed to the next store.
# Cuboid lookup uses list.index and raises
# a native ValueError when the name is missing.
# Mesh lookup catches ValueError, then invokes log_and_raise,
# which also defaults to ValueError.
# Voxel lookup has the same behavior.
# Without normalization, missing cuboid or mesh names stop the fallback
# and break attachment, detachment, or obstacle toggles.
# Wrap all three lookups to raise RuntimeError
# until upstream unifies their missing-name errors.
_curobo_patch_targets = [
    ("curobo._src.geom.data.data_cuboid", "CuboidData", "Cuboid"),
    ("curobo._src.geom.data.data_mesh", "MeshData", "Mesh"),
]
try:
    from curobo._src.geom.data import data_voxel as _voxel_mod  # noqa: F401
    _curobo_patch_targets.append(
        ("curobo._src.geom.data.data_voxel", "VoxelData", "Voxel")
    )
except Exception:
    pass

for _mod_path, _cls_name, _label in _curobo_patch_targets:
    try:
        import importlib as _il
        _mod = _il.import_module(_mod_path)
        _cls = getattr(_mod, _cls_name)
        _orig = _cls.get_idx

        def _make_patched(_orig=_orig, _label=_label):
            def _patched(self, name, env_idx=0):
                try:
                    return _orig(self, name, env_idx)
                except ValueError as _ve:
                    raise RuntimeError(
                        f"{_label} with name '{name}' not found in environment {env_idx}"
                    ) from _ve
            return _patched

        _cls.get_idx = _make_patched()
    except Exception as _e:  # noqa: BLE001
        import logging as _lg
        _lg.getLogger(__name__).warning(f"[curobo] {_cls_name} get_idx patch failed: {_e}")
# ---------------------------------------------------------------------


# DeviceCfg replaces TensorDeviceType while retaining device conversion helpers.
from curobo._src.types.device_cfg import DeviceCfg as TensorDeviceType
# v2 splits the earlier USD helper module:
# UsdSceneParser replaces UsdHelper with compatible stage/cache functionality,
#     load_stage, get_obstacles_from_stage);
# while get_prim_world_pose lives in usd_util.
from curobo._src.util.usd_scene_parser import UsdSceneParser as UsdHelper
from curobo._src.util.usd_util import get_prim_world_pose
from curobo._src.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
# isaacsim imports are latency-critical for Isaac Sim boot ordering; keep as top-level
# but wrap in try/except so non-sim environments (data_collect_v2) can still import the
# module for static inspection and linting. Runtime call sites still assume these succeeded.
try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import sphere
    from isaacsim.core.prims import SingleArticulation as Articulation
    from isaacsim.core.prims import SingleXFormPrim as XFormPrim
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:  # pragma: no cover — non-sim env
    World = None
    sphere = None
    Articulation = None
    XFormPrim = None
    ArticulationAction = None

from common.base_utils.logger import logger
from sim_backend import cuda_torch_device, physics
from server.motion_generator.mesh_utils import (
    get_mesh_attrs,
    simplify_obstacles_from_stage,
)
from server.motion_generator.path_filters import (
    filter_paths_by_position_error,
    filter_paths_by_rotation_error,
    filter_paths_by_crs_bound,
    filter_ik_result_by_crs_bound,
    sort_by_difference_js,
)
from server.motion_generator.trajectory_timing import (
    consume_resolved_slowdown_factor,
)

# USD imports (using try-except for optional dependency)
try:
    from pxr import UsdGeom, UsdPhysics
except ImportError:
    # These will be imported dynamically when needed
    pass


CUROBO_BATCH_SIZE = 20
MAX_MESH_FACES = 1000  # Maximum face count limit
DATA_COLLECTION_ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

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
    ) -> Scene:
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
            r_T_w, _ = get_prim_world_pose(
                self._xform_cache, reference_prim, inverse=True
            )

        # iterate stage prims (use Traverse for full traversal)
        all_items = self.stage.Traverse()
        for prim in all_items:
            prim_path = str(prim.GetPath())

            # only/ignore path filters
            if only_paths is not None and not any(
                [prim_path.startswith(k) for k in only_paths]
            ):
                continue
            if ignore_paths is not None and any(
                [prim_path.startswith(k) for k in ignore_paths]
            ):
                continue
            if only_substring is not None and not any(
                [k in prim_path for k in only_substring]
            ):
                continue
            if ignore_substring is not None and any(
                [k in prim_path for k in ignore_substring]
            ):
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
                    m_data = get_mesh_attrs(
                        prim, cache=self._xform_cache, transform=r_T_w
                    )
                    if m_data is not None:
                        obstacles["mesh"].append(m_data)
            except Exception as e:
                logger.error(f"Error extracting prim {prim_path}: {e}")
                continue

        world_model = Scene(**obstacles)
        return world_model


class _PlanConfigShim:
    """Placeholder for v1's ``MotionGenPlanConfig`` — v2 has no equivalent class.

    Legacy code touches ``self.plan_config.pose_cost_metric`` / ``.clone()`` /
    ``.max_attempts`` etc.; we keep this minimal shim so those accesses don't
    KeyError while the migration is in progress. New code paths should read the
    corresponding attributes on ``CuroboMotion`` directly (``max_attempts``,
    ``success_ratio``, ``enable_graph_attempt``) and treat ``pose_cost_metric``
    as a marker to build a fresh ``ToolPoseCriteria`` per plan.
    """

    def __init__(self, max_attempts: int, success_ratio: float, enable_graph_attempt: int):
        self.max_attempts = max_attempts
        self.success_ratio = success_ratio
        self.enable_graph_attempt = enable_graph_attempt
        self.pose_cost_metric = None  # v1 legacy attribute, unused in v2 dispatch

    def clone(self) -> "_PlanConfigShim":
        c = _PlanConfigShim(self.max_attempts, self.success_ratio, self.enable_graph_attempt)
        c.pose_cost_metric = self.pose_cost_metric
        return c


class CuroboMotion:
    world_coll_checker = None
    cached_obstacle_info = {}
    curobo_kinematics = None
    curobo_kinematics_robot_cfg = {}

    def reset(self):
        # v1 `clear_world_cache()` + `reset(reset_seed=False)` → v2 `clear_scene_cache()`
        # Reset planner seed state separately from scene collision caches.
        # v2 provides clear_scene_cache and reset_seed as distinct operations.
        self.planner.clear_scene_cache()
        # Preserve the previous reset behavior by clearing optimizer seeds
        # on each simulation reset.
        if hasattr(self.planner, "reset_seed"):
            self.planner.reset_seed()

    def reset_link(self):
        # v2 queries configured tool_frames instead of legacy link_names/ee_link.
        # Keep a copied link_names list for ui_builder set differences.
        self.link_names = list(self.tool_frames)
        self.ee_link_name = self.tool_frames[0]
        for i in self.link_names:
            self.target_links[i] = XFormPrim(
                "/World/target_" + i,
                position=self.init_ee_pose[i]["position"],
                orientation=self.init_ee_pose[i]["orientation"],
            )

    def _get_curobo_kinematics(self):
        """Return the class-level Kinematics instance for FK-only link pose queries.
        Keep it independent of planner kinematics so FK requests do not disturb CUDA
        graph state. Refresh lock joints through update_curobo_kinematics_lock_joints."""
        if (
            CuroboMotion.curobo_kinematics is None
            and hasattr(self, "robot_cfg")
            and self.robot_cfg is not None
        ):
            CuroboMotion.curobo_kinematics_robot_cfg = robot_cfg = copy.deepcopy(
                self.robot_cfg
            )
            # Add queried arm collision links to v2 tool_frames
            # as extra FK query frames, replacing the earlier link_names configuration
            # while preserving the kinematic_forward link outputs.
            existing_tools = list(robot_cfg["kinematics"].get("tool_frames") or [])
            for link_name in robot_cfg["kinematics"].get("collision_link_names", []):
                if "arm" in link_name and link_name not in existing_tools:
                    existing_tools.append(link_name)
            robot_cfg["kinematics"]["tool_frames"] = existing_tools
            kin_cfg = KinematicsCfg.from_data_dict(
                {"kinematics": robot_cfg["kinematics"]}
            )
            CuroboMotion.curobo_kinematics = Kinematics(kin_cfg)
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
            "/default_ground_plane",
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
            if self.simplify_mesh:
                simplified_obstacles = simplify_obstacles_from_stage(
                    initial_obstacles, max_faces=MAX_MESH_FACES
                )
            else:
                # Use original meshes when simplification is disabled, preserving small-scene geometry.
                simplified_obstacles = initial_obstacles
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
            simplified_obstacles = simplify_obstacles_from_stage(
                new_obstacles, usd_path=usd_path
            )
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
                logger.info(
                    f"Successfully added {new_obstacles_count} new obstacles to cache"
                )
            else:
                logger.info(f"No new obstacles found under path {prim_path}")

            logger.info(
                f"Current total cache count: {len(CuroboMotion.cached_obstacle_info)}"
            )
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

        logger.info(
            f"Batch addition completed, total {total_added} new obstacles added"
        )
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
        self.set_obstacles()

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
        simplify_mesh=True,
    ):
        self.name = name
        self.debug = debug
        self.simplify_mesh = simplify_mesh
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
        self.filter_crs_bound = True

        robot_cfg_path = get_robot_configs_path()
        # v2 does not consume the old task-specific cost-tuning directories.
        # The former MotionGen loader read cuRobo content/configs/task;
        # MotionPlannerCfg.create uses robot YAML and solver keyword arguments.
        # Custom costs should update solver_cfg here instead of copying task directories.

        # Load v2 robot YAML from the repository configuration first.
        # Fall back to the cuRobo asset directory for default lookup compatibility.
        repo_v2_yaml = os.path.join(
            DATA_COLLECTION_ROOT_DIR, "config/curobo/v2/configs/robot", robot_cfg
        )
        if os.path.isfile(repo_v2_yaml):
            cfg_full_path = repo_v2_yaml
        else:
            cfg_full_path = join_path(get_robot_configs_path(), robot_cfg)
        loaded_yaml = load_yaml(cfg_full_path)
        if "robot_cfg" in loaded_yaml:
            self.robot_cfg = loaded_yaml["robot_cfg"]
        else:
            self.robot_cfg = loaded_yaml
        # Retain the YAML path for locked-joint kinematics rebuilding.
        self._cfg_full_path = cfg_full_path

        # Reserve sphere slots for right and left attached-object links.
        #
        # Preserve extra_collision_spheres supplied by the YAML.
        # Overwriting a larger configured capacity with the default 30 would reduce
        # the number of fitted spheres available for large meshes
        # and make attachment fail when the fitted count exceeds capacity.
        # Only supply 30 slots when the YAML does not specify a value.
        _existing_ecs = self.robot_cfg["kinematics"].get("extra_collision_spheres") or {}
        _min_ecs = {"attached_object": 30, "left_attached_object": 30}
        _merged_ecs = dict(_min_ecs)
        for k, v in _existing_ecs.items():
            # Integer values denote slot counts; dictionaries contain explicit sphere definitions.
            if isinstance(v, int):
                _merged_ecs[k] = max(_merged_ecs.get(k, 0), v)
            else:
                _merged_ecs[k] = v
        self.robot_cfg["kinematics"]["extra_collision_spheres"] = _merged_ecs
        self.lock_joints = self.robot_cfg["kinematics"].get("lock_joints") or {}
        self.lock_js_names = list(self.lock_joints.keys())
        cspace_cfg = self.robot_cfg["kinematics"]["cspace"]
        j_names = cspace_cfg["joint_names"]
        # v2 uses default_joint_position; accept legacy retract_config for compatibility.
        default_config = cspace_cfg.get("default_joint_position") or cspace_cfg.get(
            "retract_config"
        )
        self.collision_link_names = self.robot_cfg["kinematics"].get(
            "collision_link_names", []
        )

        # v2 tool_frames replaces ee_link/link_names; require it explicitly
        # to reject incompatible legacy YAML shapes.
        tool_frames = self.robot_cfg["kinematics"].get("tool_frames")
        if not tool_frames:
            raise ValueError(
                f"v2 curobo requires `tool_frames` in {cfg_full_path}; "
                "found none. Convert this YAML (drop ee_link/link_names, "
                "add tool_frames: [...]) before loading."
            )
        self.tool_frames: List[str] = list(tool_frames)

        # Snapshot current simulation joint values before building the planner.
        # Override locked body/head/gripper joints in the input configuration
        # so fixed transforms are baked from actual simulation state.
        # This avoids relying on in-place kinematics updates after construction
        # and isolates IK failures caused by unsynchronized locked joints.
        #
        # Use the robot argument; self.robot is assigned later in construction.
        robot_cfg_for_planner: Dict = copy.deepcopy(self.robot_cfg)
        try:
            _sim_dof_names = physics.get_dof_names(robot)
            _sim_positions = physics.get_joint_positions(robot, as_torch=False)
            _sim_positions = np.asarray(_sim_positions).reshape(-1)
            _lock_snapshot = {}
            for jn, default_val in (
                robot_cfg_for_planner["kinematics"].get("lock_joints") or {}
            ).items():
                if jn in _sim_dof_names:
                    idx = _sim_dof_names.index(jn)
                    _lock_snapshot[jn] = float(_sim_positions[idx])
                else:
                    _lock_snapshot[jn] = float(default_val)
            if _lock_snapshot:
                robot_cfg_for_planner["kinematics"]["lock_joints"] = _lock_snapshot
                logger.info(
                    f"[CuroboMotion.__init__] snapshot lock_joints from sim: "
                    f"{ {k: round(v, 4) for k, v in _lock_snapshot.items() if abs(v) > 1e-4} }"
                )
                # Update the cached robot configuration with synchronized locked joints.
                self.robot_cfg["kinematics"]["lock_joints"] = dict(_lock_snapshot)
                self.lock_joints = self.robot_cfg["kinematics"]["lock_joints"]
                self.lock_js_names = list(self.lock_joints.keys())
                # Prepopulate lock_joint_states so subsequent identical lock updates
                # are no-ops rather than unnecessary YAML reloads and rebuilds.
                # Body/head locked joints normally remain stable during picking;
                # rebuild only when their values actually change.
                self.lock_joint_states = dict(_lock_snapshot)
        except Exception as _e:
            logger.warning(
                f"[CuroboMotion.__init__] failed to snapshot sim lock_joints: {_e}; "
                "planner will use YAML default lock_joints values."
            )

        # Start with an empty scene; set_obstacles fills its mesh cache later.
        self.world_cfg = Scene()
        # CUROBO_DISABLE_OBSTACLES=1 is a diagnostic switch to keep the scene empty.
        # If IK then succeeds, investigate collision geometry or sphere-fit constraints.
        self._disable_scene_obstacles = os.environ.get("CUROBO_DISABLE_OBSTACLES", "0") == "1"
        if self._disable_scene_obstacles:
            logger.warning(
                "[CuroboMotion] CUROBO_DISABLE_OBSTACLES=1: set_obstacles/mesh cache are all no-ops,"
                "planner will see EMPTY world. Diagnostic mode only!"
            )

        # Use BatchMotionPlanner to generate batch_size candidates per planning pass.
        # Single-arm reach also uses this path:
        # repeat the same goal CUROBO_BATCH_SIZE times to obtain multiple candidates.
        #
        # A scene model or collision cache is needed to create the scene checker.
        # Provide an empty Scene with 30 cuboid and 30 mesh cache slots
        # as the starting collision model.
        # set_obstacles later loads the real scene geometry
        # into the scene collision checker.
        planner_cfg = MotionPlannerCfg.create(
            robot=robot_cfg_for_planner,
            scene_model=self.world_cfg,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            self_collision_check=True,
            use_cuda_graph=True,
            optimizer_collision_activation_distance=0.01,
            max_batch_size=CUROBO_BATCH_SIZE,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
        )
        self.tensor_args = tensor_args
        self.planner: BatchMotionPlanner = BatchMotionPlanner(planner_cfg)

        # Cache the scene_collision_checker at class level for reuse.
        if CuroboMotion.world_coll_checker is None:
            CuroboMotion.world_coll_checker = self.planner.scene_collision_checker
        # Disable graph warmup to reduce startup GPU memory use.
        # PRM graph capture and trajectory seed pools can consume substantial memory.
        # This matters particularly for dual-arm robot configurations.
        # The first plan can still initialize graph state lazily.
        # One warmup iteration is enough for this setup.
        self.planner.warmup(enable_graph=True, num_warmup_iterations=5)
        # Keep world_model as an alias of scene_model for visualization compatibility.
        self.world_model = self.planner.scene_collision_checker.scene_model

        # v2 has no MotionGenPlanConfig; store the relevant options as attributes.
        # ToolPoseCriteria manages pose weights separately.
        # Keep the plan_config shim for max_attempts and success_ratio reads.
        self.max_attempts = 40
        self.success_ratio = 1.0
        self.enable_graph_attempt = 5
        self.plan_config = _PlanConfigShim(
            max_attempts=self.max_attempts,
            success_ratio=self.success_ratio,
            enable_graph_attempt=self.enable_graph_attempt,
        )
        self.target = XFormPrim(
            "/World/target",
            position=np.array([0.5, 0, 0.5]),
            orientation=np.array([0, 1, 0, 0]),
        )
        self.cmd_plan = None
        self.cmd_plans = []
        self.cmd_idx = 0
        # Slowdown factors are at least one. Resample the selected position path
        # when the factor exceeds one, increasing its length proportionally.
        # Consuming one point per physics step then increases the simulated duration.
        # Reset the factor after planning; see _resample_cmd_plan for batch-retiming limits.
        self.pending_slowdown_factor = 1.0
        self.num_targets = 0
        self.past_cmd = None
        self.pose_metic = None
        self.robot = robot
        physics.initialize_view(self.robot)
        self.idx_list = [self.robot.get_dof_index(x) for x in j_names]
        physics.set_joint_positions(self.robot, default_config, self.idx_list)
        physics.set_max_efforts(
            self.robot,
            np.array([5000 for i in range(len(self.idx_list))]),
            joint_indices=self.idx_list,
        )
        self.reached = False
        self.success = False
        # After the command path ends, wait for PD settling before reporting reached.
        # Any of these conditions completes settling:
        #   target: maximum joint error below eps_pos.
        #   stall: joint movement below eps_stall for N consecutive steps.
        #   timeout: settling duration exceeds max_delay_time.
        # GPU synchronization and hybrid action application can require extra steps
        # before completion checks observe the final EE pose.
        self._reach_settle_eps_pos = 1e-3          # Joint arrival threshold in radians (about 0.057 degrees).
        self._reach_settle_eps_stall = 1e-4        # Per-step movement threshold in radians for stall detection.
        self._reach_settle_stall_window = 5        # Consecutive stalled steps required for completion.
        self._reach_settle_max_delay_s = 0.5       # Settling timeout in seconds.
        self._reach_settle_active = False
        self._reach_settle_counter = 0             # Settling step count.
        self._reach_settle_stall_counter = 0       # Current consecutive stalled-step count.
        self._reach_settle_target = None           # Last commanded joint positions as a numpy array.
        self._reach_settle_dof_indices = None      # Corresponding DOF indices as a numpy array.
        self._reach_settle_prev_cur = None         # Previous joint positions as a numpy array.
        # ui_builder refreshes the active arm side for v2 CRS filtering.
        # A shared dual-arm planner is selected by both arms in turn;
        # use this flag instead of a hardcoded right-gripper link name.
        self._active_is_right: Optional[bool] = None
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
        self._extract_cached_obstacles(
            need_reset_cache=CuroboMotion.cached_obstacle_info == {}
        )

        # Then set obstacles (will use cache now)
        self.set_obstacles()
        # Keep all collision query links for ui_builder follower-frame selection.
        # Default ee_link_name to tool_frames[0] for caller compatibility.
        self.link_names = list(self.tool_frames)
        self.ee_link_name = self.tool_frames[0]
        # Compute tool FK at the planner default joint state
        # and create /World/target_<link> IK markers.
        default_js = self.planner.default_joint_state.position.view(1, -1)
        default_js_state = JointState.from_position(
            default_js, joint_names=self.planner.joint_names
        )
        kin_state: KinematicsState = self.planner.compute_kinematics(default_js_state)
        self.target_links: Dict[str, XFormPrim] = {}
        for tool in self.tool_frames:
            tp = kin_state.tool_poses.get_link_pose(tool)
            pos = tp.position.view(-1, 3)[0].detach().cpu().numpy().astype(np.float64)
            quat = tp.quaternion.view(-1, 4)[0].detach().cpu().numpy().astype(np.float64)
            self.target_links[tool] = XFormPrim(
                "/World/target_" + tool,
                position=pos,
                orientation=quat,
            )
        self.lock_joint_states = None

    def set_obstacles(self):
        """Refresh curobo's collision world from cached obstacle geometry."""
        if getattr(self, "_disable_scene_obstacles", False):
            # Diagnostic empty-scene mode isolates obstacle-related planning failures.
            # Production should retain scene obstacles.
            return
        start_time = time.time()

        updated_obstacles, has_update = self._update_obstacle_poses_fast()

        # v2 has no fix_cache_reference or legacy _env_mesh_names introspection.
        # After reset clears scene caches, reload the complete SceneCfg.
        # Reload for nonempty obstacles on pose changes and on the first update
        # even when obstacle poses match the previous cached poses.
        obstacles_present = any(v for v in updated_obstacles.values())
        if not has_update and obstacles_present and self._world_coll_needs_refill():
            has_update = True

        if has_update:
            scene_cfg = Scene(**updated_obstacles)
            # Keep collision-only fields in the SceneCfg returned by this helper;
            # its obstacle list uses the v2 typed collision configuration.
            scene_cfg = scene_cfg.get_collision_check_world()
            self.world_cfg = scene_cfg  # NOTE: only affects visualization / debug
            self.planner.scene_collision_checker.load_collision_model(scene_cfg)

        if self.planner.graph_planner is not None:
            self.planner.graph_planner.reset_buffer()
        logger.info(
            f"Fast obstacle update completed, time: {time.time() - start_time:.4f}s, "
            f"needs update: {has_update}"
        )

    def _world_coll_needs_refill(self) -> bool:
        """Check obstacle counts to detect an empty collision checker requiring reload."""
        try:
            existing = self.planner.scene_collision_checker.get_obstacle_names(env_idx=0)
        except Exception:
            return True
        return len(existing) == 0

    def _resample_cmd_plan(self, cmd_plan, factor: float):
        """Linearly resample the selected command path to slow it by factor.
        Batch-result retiming supports only single results, so resample after selecting
        one path. Interpolate position and available derivative arrays; divide velocity,
        acceleration, and jerk by factor, factor squared, and factor cubed respectively.
        Keep joint names and other metadata unchanged."""
        if abs(factor - 1.0) < 1e-6 or factor <= 0:
            return cmd_plan
        try:
            import torch  # noqa: PLC0415
        except Exception:
            return cmd_plan
        try:
            pos = cmd_plan.position
            T = pos.shape[0]
            new_T = max(2, int(round(T * factor)))

            def _interp(t, scale=1.0):
                if t is None:
                    return None
                return (
                    torch.nn.functional.interpolate(
                        t.T.unsqueeze(0).float(),
                        size=new_T,
                        mode="linear",
                        align_corners=True,
                    ).squeeze(0).T.to(t.dtype)
                    * scale
                )

            # Resample position, velocity, acceleration, and jerk together.
            # JointState indexing accesses each array, so mismatched lengths can
            # cause out-of-bounds reads. Scale velocity by 1/factor, acceleration by
            # 1/factor squared, and jerk by 1/factor cubed.
            cmd_plan.position = _interp(pos)
            f = float(factor)
            vel = getattr(cmd_plan, "velocity", None)
            if vel is not None:
                cmd_plan.velocity = _interp(vel, scale=1.0 / f)
            acc = getattr(cmd_plan, "acceleration", None)
            if acc is not None:
                cmd_plan.acceleration = _interp(acc, scale=1.0 / (f * f))
            jerk = getattr(cmd_plan, "jerk", None)
            if jerk is not None:
                cmd_plan.jerk = _interp(jerk, scale=1.0 / (f * f * f))
            logger.info(
                f"[_resample_cmd_plan] factor={factor:.3f} T={T} → new_T={new_T}"
            )
        except Exception as exc:
            logger.warning(f"[_resample_cmd_plan] failed: {exc}")
        return cmd_plan

    def _get_target_link_pose(self, link_name: str):
        """Read the current world pose of /World/target_<link> without creating a view.
        Use physics.get_prim_world_pose; nonphysical markers fall back to USD,
        matching the marker write path and avoiding repeated view handle reuse."""
        return physics.get_prim_world_pose(f"/World/target_{link_name}")

    # ------------------------------------------------------------------
    # v1 → v2 planning helpers
    # ------------------------------------------------------------------

    def _tile_np_pose(self, pos_np: np.ndarray, quat_np: np.ndarray, batch: int) -> Pose:
        """Repeat a numpy xyz/wxyz pose across the requested batch and construct a device Pose."""
        pos = self.tensor_args.to_device(
            np.tile(np.asarray(pos_np, dtype=np.float64).reshape(1, 3), (batch, 1))
        )
        quat = self.tensor_args.to_device(
            np.tile(np.asarray(quat_np, dtype=np.float64).reshape(1, 4), (batch, 1))
        )
        return Pose(position=pos, quaternion=quat)

    def _tile_torch_pose(self, pose: Pose, batch: int) -> Pose:
        """Repeat a single-element v2 Pose across the requested batch."""
        pos = pose.position.view(-1, 3)[:1].expand(batch, 3).contiguous()
        quat = pose.quaternion.view(-1, 4)[:1].expand(batch, 4).contiguous()
        return Pose(position=pos, quaternion=quat)

    def _build_batch_goal(
        self,
        active_tool: str,
        target_pos_np: np.ndarray,
        target_quat_np: np.ndarray,
        cu_js: JointState,
        batch: Optional[int] = None,
        hold_link_poses: Optional[Dict[str, Pose]] = None,
    ) -> GoalToolPose:
        # None uses the current module-level CUROBO_BATCH_SIZE.
        # Do not bind the constant as a function default, since defaults are evaluated
        # when the function is defined. A later startup batch-size override
        # would otherwise leave the old goal batch size in use
        # and mismatch the planner max_batch_size.
        if batch is None:
            batch = CUROBO_BATCH_SIZE
        """Build a batched GoalToolPose with active targets and other-tool holds.
        Repeat the active target to batch size. For other tools, prefer explicit
        hold_link_poses and otherwise use current FK from cu_js.
        Single-tool configurations include only the active target.
        """
        goal_dict: Dict[str, Pose] = {
            active_tool: self._tile_np_pose(target_pos_np, target_quat_np, batch)
        }
        other_tools = [t for t in self.planner.tool_frames if t != active_tool]
        if other_tools:
            # Compute FK once from cu_js[:1] instead of for the whole batch.
            fk_input = cu_js if cu_js.position.shape[0] == 1 else cu_js[0:1]
            kin_state = self.planner.compute_kinematics(fk_input)
            for t in other_tools:
                if hold_link_poses is not None and t in hold_link_poses:
                    goal_dict[t] = self._tile_torch_pose(hold_link_poses[t], batch)
                else:
                    tp = kin_state.tool_poses.get_link_pose(t)
                    goal_dict[t] = self._tile_torch_pose(tp, batch)
        return GoalToolPose.from_poses(
            goal_dict,
            ordered_tool_frames=self.planner.tool_frames,
            num_goalset=1,
        )

    def _build_tool_pose_criteria(
        self,
        active_tool: str,
        path_constraint,
        project_to_goal: bool,
        additional_constraints: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, ToolPoseCriteria]:
        """Build independent v2 ToolPoseCriteria for each tool.
        Convert incoming [rotation, position] weights to v2 [position, rotation] order.
        Apply hold weights to the active tool; other tools use default 6D criteria
        unless additional_constraints supplies their path and goal-frame options."""

        def _make_criteria(_pc, _project):
            if _pc is None or len(_pc) != 6:
                return ToolPoseCriteria()
            rot_w = list(_pc[:3])
            pos_w = list(_pc[3:])
            non_terminal = pos_w + rot_w  # v1[rot,pos] → v2[pos,rot]
            return ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1.0] * 6,
                non_terminal_pose_axes_weight_factor=non_terminal,
                project_distance_to_goal=bool(_project),
            )

        criteria: Dict[str, ToolPoseCriteria] = {
            t: ToolPoseCriteria() for t in self.planner.tool_frames
        }
        criteria[active_tool] = _make_criteria(path_constraint, project_to_goal)
        if additional_constraints:
            for _tool, _cfg in additional_constraints.items():
                if _tool not in criteria:
                    continue  # Ignore unrelated tools.
                criteria[_tool] = _make_criteria(
                    _cfg.get("path_constraint"),
                    _cfg.get("offset_and_constraint_in_goal_frame", True),
                )
        return criteria

    def _extract_candidate_paths(self, result) -> tuple:
        """Extract one successful seed per batch from a TrajOptSolverResult.
        Choose the successful seed with the smallest position error. Return JointState
        paths with corresponding position and rotation error tensors for filtering."""
        if result is None or result.js_solution is None:
            return [], torch.empty(0), torch.empty(0)
        js = result.js_solution
        # js.position shape: (batch, num_seeds, horizon, dof)
        # Accept (batch, horizon, dof) when num_seeds is one.
        pos_full = js.position
        vel_full = getattr(js, "velocity", None)
        acc_full = getattr(js, "acceleration", None)
        jerk_full = getattr(js, "jerk", None)
        dt_full = getattr(js, "dt", None)
        if pos_full.ndim == 3:
            pos_full = pos_full.unsqueeze(1)
            if vel_full is not None and vel_full.ndim == 3:
                vel_full = vel_full.unsqueeze(1)
            if acc_full is not None and acc_full.ndim == 3:
                acc_full = acc_full.unsqueeze(1)
            if jerk_full is not None and jerk_full.ndim == 3:
                jerk_full = jerk_full.unsqueeze(1)
        batch = pos_full.shape[0]
        succ = result.success  # (batch, num_seeds)
        if succ.ndim == 1:
            succ = succ.unsqueeze(-1)
        pos_err = result.position_error
        rot_err = result.rotation_error
        paths = []
        pe_list = []
        re_list = []
        for bi in range(batch):
            s_mask = succ[bi]
            if not bool(s_mask.any().item()):
                continue
            # among successful seeds, pick min position_error
            candidates = pos_err[bi].clone()
            candidates[~s_mask] = float("inf")
            best_seed = int(torch.argmin(candidates).item())
            traj = JointState(
                position=pos_full[bi, best_seed],
                velocity=vel_full[bi, best_seed] if vel_full is not None else None,
                acceleration=acc_full[bi, best_seed] if acc_full is not None else None,
                jerk=jerk_full[bi, best_seed] if jerk_full is not None else None,
                joint_names=list(js.joint_names) if js.joint_names is not None else None,
            )
            if dt_full is not None:
                try:
                    traj.dt = dt_full[bi] if dt_full.ndim >= 1 else dt_full
                except Exception:
                    pass
            paths.append(traj)
            pe_list.append(pos_err[bi, best_seed].item())
            re_list.append(rot_err[bi, best_seed].item())
        return paths, torch.tensor(pe_list), torch.tensor(re_list)

    def _make_pose_reader(self):
        """Return a (cache, prim, inverse=False) -> (matrix, scale) pose reader.
        Torch uses physics.get_prim_world_pose for current Fabric poses and USD for
        static scale. Numpy uses the cuRobo USD reader synchronized with CPU physics."""
        from sim_backend import backend as _bk  # noqa: PLC0415

        if _bk.tensors_as() != "torch":
            return get_prim_world_pose

        from pxr import Gf  # noqa: PLC0415

        tensor_args = TensorDeviceType()

        def _reader(cache, prim, inverse: bool = False):
            usd_world_xf = cache.GetLocalToWorldTransform(prim)
            scale = list(
                Gf.Vec3d(*(v.GetLength() for v in usd_world_xf.ExtractRotationMatrix()))
            )
            pos, quat_wxyz = physics.get_prim_world_pose(str(prim.GetPath()))
            pose = Pose.from_list(
                [float(pos[0]), float(pos[1]), float(pos[2]),
                 float(quat_wxyz[0]), float(quat_wxyz[1]),
                 float(quat_wxyz[2]), float(quat_wxyz[3])],
                tensor_args,
            )
            if inverse:
                pose = pose.inverse()
            t_mat = pose.get_matrix().view(4, 4).cpu().numpy()
            return t_mat, scale

        return _reader

    def _update_obstacle_poses_fast(self):
        """Refresh cached obstacle poses in robot coordinates.
        Return updated obstacles and a pose-change flag. The flag alone cannot determine
        whether the checker needs reloading; also check _world_coll_needs_refill."""
        _read_world_pose = self._make_pose_reader()

        r_T_w = None
        if self.robot_reference_prim and self.robot_transform_cache:
            self.robot_transform_cache.Clear()
            self.robot_transform_cache.SetTime(0)
            r_T_w, _ = _read_world_pose(
                self.robot_transform_cache, self.robot_reference_prim, inverse=True
            )

        updated_obstacles = {
            "cuboid": [],
            "sphere": [],
            "mesh": [],
            "cylinder": [],
            "capsule": [],
        }

        has_update = False
        for prim_path, obstacle_info in CuroboMotion.cached_obstacle_info.items():
            if any(prim_path.startswith(p) for p in self.attached_objects):
                continue
            prim = self.usd_help.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue

            try:
                current_mat, _ = _read_world_pose(self.robot_transform_cache, prim)
            except Exception as e:
                logger.info(f"Error reading obstacle {prim_path} pose: {e}")
                continue

            if r_T_w is not None:
                current_mat = r_T_w @ current_mat

            tensor_mat = torch.as_tensor(current_mat, device=cuda_torch_device())
            updated_pose = Pose.from_matrix(tensor_mat).tolist()

            # Skip obstacles more than 10 meters from the robot.
            if np.linalg.norm(np.array(updated_pose[:3])) > 10.0:
                continue

            geometry = obstacle_info["geometry"]
            if geometry.pose != updated_pose:
                has_update = True
                geometry.pose = updated_pose

            updated_obstacles[obstacle_info["type"]].append(geometry)

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
                        translation=np.array(
                            [s.position[0], s.position[1], s.position[2]]
                        )
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
        sim_js = physics.get_joints_state(self.robot, as_torch=False)
        sim_js_names = physics.get_dof_names(self.robot)
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities),
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js.acceleration *= 0.0
        cu_js = cu_js.reorder(self.planner.kinematics.joint_names)
        sph_list = self.planner.kinematics.get_robot_as_spheres(cu_js.position)
        self.visualize_spheres(
            sph_list, self.spheres, prim_prefix="/curobo/robot_sphere_"
        )

    def kinematic_forward(self, joint_states, output_link_names=None):
        """Compute FK and collect tool poses from v2 KinematicsState.
        Return one link-name-to-[position, quaternion] dictionary per batch row."""
        # joint_positions should [batch_size, dof]
        t1 = time.time()
        kin = self._get_curobo_kinematics()
        if not isinstance(joint_states, JointState):
            joint_states = JointState.from_position(
                joint_states, joint_names=kin.joint_names,
            )
        result: KinematicsState = kin.compute_kinematics(joint_states)
        tp = result.tool_poses
        # Tool positions are [B, H, L, 3]; use the first horizon element.
        # Select horizon zero to obtain [B, L, 3].
        pos = tp.position
        quat = tp.quaternion
        if pos.ndim == 4:
            pos = pos[:, 0]  # [B, L, 3]
            quat = quat[:, 0]  # [B, L, 4]
        link_names = tp.tool_frames
        output = []
        for i in range(pos.shape[0]):
            tmp_output = {}
            for j, link_name in enumerate(link_names):
                if (
                    output_link_names is None
                    or not len(output_link_names)
                    or link_name in output_link_names
                ):
                    tmp_output[link_name] = [
                        pos[i][j].cpu().numpy().tolist(),
                        quat[i][j].cpu().numpy().tolist(),
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
        """Solve active-tool candidates in CUROBO_BATCH_SIZE chunks.
        Supply other-tool FK placeholders with disabled criteria so they do not
        constrain single-arm IK. Reset criteria to defaults after solving.
        Return success and joint mappings, plus link poses when requested."""
        logger.info("len positions is {}".format(len(positions)))
        t1 = time.time()
        results = []
        batch_size = CUROBO_BATCH_SIZE
        # Split first dimension by batch_size, pad last group by repeating
        pos_num = positions.shape[0]
        num_splits = (pos_num + batch_size - 1) // batch_size
        remainder = pos_num % batch_size
        if remainder != 0:
            padding = batch_size - remainder
            positions = np.concatenate(
                [positions, np.tile(positions[-1:], (padding, 1))], axis=0
            )
            rotations = np.concatenate(
                [rotations, np.tile(rotations[-1:], (padding, 1))], axis=0
            )
        position_batched = np.array_split(positions, num_splits)
        rotation_batched = np.array_split(rotations, num_splits)

        # Compute other-tool FK poses for the combined goal dictionary.
        # These are required placeholders during single-active-tool IK.
        other_tools = [t for t in self.planner.tool_frames if t != active_ee_name]
        hold_pose_map: Dict[str, Pose] = {}
        if other_tools:
            try:
                sim_js = physics.get_joints_state(self.robot, as_torch=False)
                sim_js_names = physics.get_dof_names(self.robot)
                cu_js_current = JointState(
                    position=self.tensor_args.to_device(np.asarray(sim_js.positions)),
                    velocity=self.tensor_args.to_device(np.asarray(sim_js.velocities)) * 0.0,
                    acceleration=self.tensor_args.to_device(np.asarray(sim_js.velocities)) * 0.0,
                    jerk=self.tensor_args.to_device(np.asarray(sim_js.velocities)) * 0.0,
                    joint_names=sim_js_names,
                ).reorder(self.planner.kinematics.joint_names)
                # FK requires position shape (1, dof).
                if cu_js_current.position.ndim == 1:
                    cu_js_current = JointState.from_position(
                        cu_js_current.position.unsqueeze(0),
                        joint_names=cu_js_current.joint_names,
                    )
                fk = self.planner.compute_kinematics(cu_js_current)
                for t in other_tools:
                    hold_pose_map[t] = fk.tool_poses.get_link_pose(t)
            except Exception as _fe:
                logger.warning(f"failed to build hold-tool FK: {_fe}")

        if output_link_pose:
            self.update_curobo_kinematics_lock_joints(self.lock_joint_states)

        ik_solver = self.planner.ik_solver
        # Disable criteria for other tools during single-arm IK.
        # Exact goals for both arms can make the active solve unnecessarily difficult
        # compared with the former single-arm API.
        # GoalToolPose still requires a pose for every tool frame;
        # keep all required entries in the dictionary.
        # Disabled criteria set all weights to zero, leaving these poses unconstrained.
        # Prepare default criteria to apply after solving
        # so temporary IK settings do not affect later planning.
        prev_criteria = None
        try:
            prev_criteria = {
                t: ToolPoseCriteria() for t in self.planner.tool_frames
            }
        except Exception:
            prev_criteria = None
        ik_criteria = {t: ToolPoseCriteria.disabled() for t in self.planner.tool_frames}
        ik_criteria[active_ee_name] = ToolPoseCriteria()
        self.planner.ik_solver.update_tool_pose_criteria(ik_criteria)
        try:
            for i in range(num_splits):
                pos_batch = position_batched[i]
                rot_batch = rotation_batched[i]
                # Active tool gets targets; other tools get batched FK placeholders.
                # Disabled criteria means IK need not satisfy placeholder poses.
                goal_dict: Dict[str, Pose] = {
                    active_ee_name: Pose(
                        position=self.tensor_args.to_device(pos_batch),
                        quaternion=self.tensor_args.to_device(rot_batch),
                    ),
                }
                for t, hp in hold_pose_map.items():
                    goal_dict[t] = self._tile_torch_pose(hp, batch_size)
                # Use a zero placeholder if FK did not provide a required tool.
                for t in self.planner.tool_frames:
                    if t not in goal_dict:
                        zero_pos = self.tensor_args.to_device(np.zeros((batch_size, 3)))
                        zero_quat = self.tensor_args.to_device(
                            np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (batch_size, 1))
                        )
                        goal_dict[t] = Pose(position=zero_pos, quaternion=zero_quat)
                goal_tool_poses = GoalToolPose.from_poses(
                    goal_dict,
                    ordered_tool_frames=self.planner.tool_frames,
                    num_goalset=1,
                )
                t00 = time.time()
                result = ik_solver.solve_pose(goal_tool_poses, return_seeds=1)
                # v2 IKSolverResult: .success (batch, return_seeds), .js_solution (position:
                # Solution is (batch, return_seeds, dof); select [k, 0] for one seed.
                # Do not retry by dropping other link poses:
                # GoalToolPose requires all configured tool frames.
                # Try once and return success=False if the batch solve fails.
                t11 = time.time()
                logger.info("ik batch {} time is {}".format(i, t11 - t00))

                if output_link_pose:
                    js = result.js_solution.reorder(
                        self._get_curobo_kinematics().kinematics_config.joint_names
                    )
                    # (batch, return_seeds=1, dof) → (batch, dof)
                    js = js.squeeze(1)
                    ik_link_poses = self.kinematic_forward(js)
                success_1d = result.success[:, 0] if result.success.ndim == 2 else result.success
                for k in range(success_1d.shape[0]):
                    joint_positions = {}
                    for j, name in enumerate(result.js_solution.joint_names):
                        joint_positions[name] = (
                            result.js_solution.position[k][0].cpu().tolist()[j]
                        )
                    if output_link_pose:
                        results.append(
                            [success_1d[k], joint_positions, ik_link_poses[k]]
                        )
                    else:
                        results.append((success_1d[k], joint_positions))
        finally:
            # Restore default 6D criteria for every tool
            # before later plan_pose calls constrain both tools.
            if prev_criteria is not None:
                try:
                    self.planner.ik_solver.update_tool_pose_criteria(prev_criteria)
                except Exception as _e:
                    logger.warning(f"restore ik criteria failed: {_e}")
        t2 = time.time()
        logger.info("ik time is {}".format(t2 - t1))
        return results[:pos_num]

    def _attachment_managers(self):
        """Return both IK and trajectory attachment managers for synchronized updates.
        The planner property exposes the trajectory manager, while IK has an independent
        SolverCore manager. Fan attach/detach operations out so both see the same spheres."""
        managers = []
        for solver_name in ("ik_solver", "trajopt_solver"):
            solver = getattr(self.planner, solver_name, None)
            if solver is None:
                continue
            core = getattr(solver, "core", None)
            if core is None:
                continue
            am = getattr(core, "attachment_manager", None)
            if am is not None:
                managers.append(am)
        return managers

    def update_lock_joints(self, locked_joints):
        """Rebuild KinematicsCfg so new locked values enter baked fixed transforms.
        Copy the in-memory robot configuration to retain injected collision spheres,
        replace lock_joints, then update planner kinematics with matching tensor shapes.
        Changing lock_jointstate alone does not update baked FK transforms."""
        if (
            self.lock_joint_states is not None
            and np.abs(
                np.array(list(self.lock_joint_states.values()))
                - np.array(list(locked_joints.values()))
            ).max()
            <= 1e-3
        ):
            logger.info("lock joints is the same, no need to update")
            return
        before = time.time()
        # Copy the in-memory config including injected extra collision spheres
        # and replace locked joints to preserve initial tensor shapes.
        # Reloading disk YAML would omit injected sphere slots
        # and make kinematics copy operations fail on shape mismatch.
        raw_cfg = copy.deepcopy(self.robot_cfg)
        raw_cfg["kinematics"]["lock_joints"] = dict(locked_joints)
        new_kin_cfg = KinematicsCfg.from_data_dict({"kinematics": raw_cfg["kinematics"]})
        self.planner.kinematics.update_kinematics_config(new_kin_cfg.kinematics_config)
        # Update the cached configuration used by other methods.
        self.robot_cfg["kinematics"]["lock_joints"] = dict(locked_joints)
        self.lock_joints = self.robot_cfg["kinematics"]["lock_joints"]
        self.lock_joint_states = dict(locked_joints)
        after = time.time()
        logger.info("update lock joints time is {}".format(after - before))

    def update_curobo_kinematics_lock_joints(self, locked_joints):
        """Rebuild the class-level FK-only Kinematics config with updated locked joints."""
        before = time.time()
        if CuroboMotion.curobo_kinematics is not None and locked_joints is not None:
            if (
                CuroboMotion.curobo_kinematics_robot_cfg["kinematics"].get("lock_joints")
                != locked_joints
            ):
                logger.info("update kinematics lock joints")
                CuroboMotion.curobo_kinematics_robot_cfg["kinematics"][
                    "lock_joints"
                ] = dict(locked_joints)
                new_kin_cfg = KinematicsCfg.from_data_dict(
                    {"kinematics": CuroboMotion.curobo_kinematics_robot_cfg["kinematics"]}
                )
                CuroboMotion.curobo_kinematics.update_kinematics_config(
                    new_kin_cfg.kinematics_config
                )
        after = time.time()
        logger.info(
            "update curobo kinematics lock joints time is {}".format(after - before)
        )

    def view_debug_world(self):
        if self.debug:
            self.visualize_robot_spheres()
            self.visualize_obstacles()

    def get_ee_pose(self, tool: Optional[str] = None):
        """Return robot-local tool FK; default to the first configured tool when omitted."""
        active_tool = tool or self.tool_frames[0]
        cu_js, _ = self._get_current_joint_state()
        # cu_js is tiled across the batch; compute FK
        # and select the first row.
        kin_state = self.planner.compute_kinematics(cu_js)
        tp = kin_state.tool_poses.get_link_pose(active_tool)
        start_position = tp.position.view(-1, 3)[0].detach().cpu().numpy()
        start_orientation = tp.quaternion.view(-1, 4)[0].detach().cpu().numpy()
        return start_position, start_orientation

    def calculate_ik_goal(
        self,
        goal_offset=[0, 0, 0, 1, 0, 0, 0],
        path_constraint=None,
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links: list[str] = [],
        from_current_pose=False,
        target_position=None,
        target_orientation=None,
        active_tool: Optional[str] = None,
        additional_goals: Optional[Dict[str, Dict]] = None,
    ):
        """Plan an active-tool goal with optional offsets and path constraints.
        goal_offset uses the goal frame when enabled, otherwise the robot-base frame.
        path_constraint is [rx, ry, rz, x, y, z]; convert to v2 position/rotation order.
        None adds no path constraints. The goal-frame option also applies to constraints.
        disable_collision_links excludes links from world collision checks.
        from_current_pose uses current EE plus offset as the goal.
        Explicit robot-local target_position/target_orientation (wxyz) take precedence
        over target-marker reads, avoiding stale marker poses on GPU physics.
        active_tool defaults to the configured first tool; other tools hold current FK
        unless additional goals specify their target poses."""
        t0 = time.time()
        self.reached = False
        active_tool = active_tool or self.ee_link_name or self.tool_frames[0]
        if from_current_pose and not goal_offset:
            self.reached = True
            self.success = False
            logger.info("from_current_pose is True, but goal_offset is None, return")
            return
        if target_position is not None and target_orientation is not None:
            # Explicit targets bypass marker-prim pose reads.
            # Without the wrapper patch, GPU markers can return stale
            # Fabric/USD values or the current EE pose,
            # causing a stationary plan after the first step.
            cube_position = np.asarray(target_position, dtype=np.float64).reshape(3)
            cube_orientation = np.asarray(target_orientation, dtype=np.float64).reshape(4)
        else:
            cube_position, cube_orientation = self.target.get_world_pose()
        if self.past_pose is None:
            self.past_pose = cube_position
        if self.target_pose is None:
            self.target_pose = cube_position
        if self.target_orientation is None:
            self.target_orientation = cube_orientation
        if self.past_orientation is None:
            self.past_orientation = cube_orientation
        cu_js, sim_js_names = self._get_current_joint_state()
        start_time = time.time()
        if from_current_pose:
            # Read the selected v2 tool pose instead of a single legacy ee_pose.
            kin_state = self.planner.compute_kinematics(cu_js[0:1])
            tp = kin_state.tool_poses.get_link_pose(active_tool)
            cube_position = tp.position.view(-1, 3)[0].detach().cpu().numpy()
            cube_orientation = tp.quaternion.view(-1, 4)[0].detach().cpu().numpy()

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
            ee_orientation_teleop_goal = (
                offset_goal_pose.quaternion.squeeze().cpu().numpy()
            )

        if (
            path_constraint is not None
            and len(path_constraint) == 6
            and path_constraint[-3] > 0
            and path_constraint[-2] > 0
            and path_constraint[-1] > 0
        ):
            logger.info(
                "path constraint allows no position change, use ee_pose as goal position"
            )
            kin_state = self.planner.compute_kinematics(cu_js[0:1])
            tp = kin_state.tool_poses.get_link_pose(active_tool)
            ee_translation_goal = tp.position.view(-1, 3)[0].detach().cpu().numpy()

        # Build GoalToolPose with the active target and current-FK holds.
        # _build_batch_goal includes other-tool holds directly
        # in one plan_pose call.
        #
        # additional_goals maps tool frames to independent targets
        # instead of current-FK holds. Entries may contain:
        #   target_position: required xyz.
        #   target_orientation: required wxyz.
        #   goal_offset: optional per-tool pose offset.
        #   path_constraint: optional per-tool six-dimensional constraints.
        #   offset_and_constraint_in_goal_frame: optional frame selection.
        # Dual motion uses one call to avoid overwriting shared planner state.
        # Each arm can carry independent constraints for two-arm manipulation.
        hold_link_poses = None
        additional_constraints: Dict[str, Dict] = {}
        if additional_goals:
            from curobo.types import Pose as _CuroboPose
            hold_link_poses = {}
            for _tool, _cfg in additional_goals.items():
                _pos_np = np.asarray(
                    _cfg["target_position"], dtype=np.float64
                ).reshape(3)
                _quat_np = np.asarray(
                    _cfg["target_orientation"], dtype=np.float64
                ).reshape(4)
                # Relative dual-arm moves must resolve every tool from its own
                # current FK pose. The client target is only a reference pose
                # and may already contain the requested offset.
                if _cfg.get("from_current_pose", False):
                    _kin_state = self.planner.compute_kinematics(cu_js[0:1])
                    _tool_pose = _kin_state.tool_poses.get_link_pose(_tool)
                    _current_pos_np = (
                        _tool_pose.position.view(-1, 3)[0].detach().cpu().numpy()
                    )
                    _current_quat_np = (
                        _tool_pose.quaternion.view(-1, 4)[0].detach().cpu().numpy()
                    )
                else:
                    _current_pos_np = None
                    _current_quat_np = None
                _pos_np, _quat_np = select_additional_goal_base(
                    _pos_np,
                    _quat_np,
                    _current_pos_np,
                    _current_quat_np,
                    from_current_pose=_cfg.get("from_current_pose", False),
                )
                # Apply this tool goal offset independently.
                _tool_offset = _cfg.get("goal_offset")
                _tool_use_goal_frame = _cfg.get(
                    "offset_and_constraint_in_goal_frame", True
                )
                if _tool_offset is not None:
                    _base = _CuroboPose.from_list(
                        list(_pos_np) + list(_quat_np)
                    )
                    _off = _CuroboPose.from_list(list(_tool_offset))
                    _resolved = (
                        _base.clone().multiply(_off)
                        if _tool_use_goal_frame
                        else _off.clone().multiply(_base)
                    )
                    _pos_np = _resolved.position.squeeze().cpu().numpy()
                    _quat_np = _resolved.quaternion.squeeze().cpu().numpy()
                hold_link_poses[_tool] = _CuroboPose(
                    position=self.tensor_args.to_device(_pos_np.reshape(1, 3)),
                    quaternion=self.tensor_args.to_device(_quat_np.reshape(1, 4)),
                )
                # Build this tool criteria from its path constraint.
                _tool_pc = _cfg.get("path_constraint")
                if _tool_pc is not None:
                    additional_constraints[_tool] = {
                        "path_constraint": _tool_pc,
                        "offset_and_constraint_in_goal_frame": _tool_use_goal_frame,
                    }
        goal_tool_poses = self._build_batch_goal(
            active_tool, ee_translation_goal, ee_orientation_teleop_goal, cu_js,
            hold_link_poses=hold_link_poses,
        )
        # ToolPoseCriteria replaces PoseCostMetric; convert axis order in the helper.
        criteria = self._build_tool_pose_criteria(
            active_tool, path_constraint, offset_and_constraint_in_goal_frame,
            additional_constraints=additional_constraints or None,
        )
        self.planner.update_tool_pose_criteria(criteria)
        # Retain a shim marker indicating that criteria are configured.
        self.plan_config.pose_cost_metric = criteria if path_constraint else None

        disable_collision_links = list(
            filter(
                lambda link: any(
                    re.match(pattern, link) for pattern in disable_collision_links
                ),
                self.collision_link_names,
            )
        )
        self.planner.disable_link_collision(disable_collision_links)
        try:
            # Batch plan_pose runs IK and trajectory optimization for candidates.
            # max_attempts controls retries; success_ratio permits early completion.
            result = self.planner.plan_pose(
                goal_tool_poses,
                cu_js,
                max_attempts=self.max_attempts,
                success_ratio=self.success_ratio,
                enable_graph_attempt=self.enable_graph_attempt,
                finetune_attempts=0,
                initial_iters=25,
            )
            batch_success = (
                result is not None
                and result.success is not None
                and bool(result.success.any().item())
            )
            if not batch_success:
                self.reached = True
                self.success = False
                status = "no result" if result is None else "no batch success"
                logger.info(f"plan did not converge to a solution: {status}")
                return
            self.reached = False
            self.success = True
            logger.info("end_time is{}".format(time.time() - start_time))
            self.num_targets += 1
            # Extract candidate paths and errors using the helper.
            paths, pos_err_per_path, rot_err_per_path = self._extract_candidate_paths(result)
            if not paths:
                self.reached = True
                self.success = False
                logger.info("plan_pose: no successful candidate paths after extraction")
                return
            position_filter_res = filter_paths_by_position_error(paths, pos_err_per_path)
            rotation_filter_res = filter_paths_by_rotation_error(paths, rot_err_per_path)
            crs_bound_filter_res = [True] * len(paths)
            if "G2" in self.robot_prim_path and self.filter_crs_bound:
                crs_bound_filter_res = filter_paths_by_crs_bound(
                    paths,
                    is_right_arm=(
                        self._active_is_right
                        if self._active_is_right is not None
                        else active_tool == (self.ee_link_name or "")
                    ),
                )
            filtered_paths = []
            for i in range(len(paths)):
                if position_filter_res[i] and rotation_filter_res[i] and crs_bound_filter_res[i]:
                    filtered_paths.append(paths[i])
            logger.info(f"paths length: length: {len(paths)} -> filtered_paths length: {len(filtered_paths)}")
            logger.info(f"position_filter_res: {len([x for x in position_filter_res if x])}")
            logger.info(f"rotation_filter_res: {len([x for x in rotation_filter_res if x])}")
            logger.info(f"crs_bound_filter_res: {len([x for x in crs_bound_filter_res if x])}")
            if len(filtered_paths) == 0:
                filtered_paths = paths
            sorted_indices = sort_by_difference_js(
                filtered_paths,
                weights=self.tensor_args.to_device(
                    [
                        1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0,
                    ]
                ) if filtered_paths[0].position.shape[-1] == 14 else None,
            )
            self.cmd_plan = filtered_paths[sorted_indices[0]]
            # Legacy results needed explicit full-joint expansion.
            # v2 optimization already expands the solution
            # through get_full_dof_from_solution.
            # Result joint names already contain active and locked joints;
            # another get_full_js call would duplicate locked names
            # → `augment_joint_state(cmd_plan, all_names, lock_state)`,cmd_plan
            # and trigger the lock-joint overlap error.
            # Do not expand the selected path again.
            logger.info(len(self.cmd_plan))
            self.idx_list = []
            common_js_names = []
            for x in sim_js_names:
                if x in self.cmd_plan.joint_names:
                    self.idx_list.append(self.robot.get_dof_index(x))
                    common_js_names.append(x)
            self.cmd_plan = self.cmd_plan.reorder(common_js_names)
            # The controller has already combined user slowdown
            # with reference-timestep alignment.
            # Consume the result without scaling by physics_dt again.
            _total = consume_resolved_slowdown_factor(
                getattr(self, "pending_slowdown_factor", 1.0)
            )
            if abs(_total - 1.0) > 1e-3:
                self.cmd_plan = self._resample_cmd_plan(self.cmd_plan, _total)
            self.pending_slowdown_factor = 1.0
            self.cmd_idx = 0
            logger.info("success")
        except Exception as e:
            self.reached = True
            self.success = False
            logger.info("plan got an exception: {}".format(str(e)))
        self.planner.enable_link_collision(disable_collision_links)
        self.target_pose = cube_position
        self.target_orientation = cube_orientation
        self.past_pose = cube_position
        self.past_orientation = cube_orientation
        t1 = time.time()
        logger.info("total time is {}".format(t1 - t0))

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
                    if hasattr(self.cmd_plan, "velocity") and self.cmd_plan.velocity is not None:
                        velocity.append(self.cmd_plan.velocity.cpu().numpy()[index][idx])
                    else:
                        velocity.append(0.0)
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
        """Detach managed links and re-enable their world obstacles; ignore links never attached."""
        for link_name in ("attached_object", "left_attached_object"):
            for am in self._attachment_managers():
                try:
                    am.detach(link_name=link_name)
                except Exception as e:
                    logger.debug(f"detach {link_name} on {am!r} skipped: {e}")
        # Clear attached objects list
        self.attached_objects.clear()
        self.set_obstacles()

    def remove_objects_from_world(self, prim_paths):
        checker = self.planner.scene_collision_checker
        scene_model = checker.scene_model
        for x in prim_paths:
            obs = scene_model.get_obstacle(x)
            if not obs:
                continue
            # SceneCfg removal does not disable the GPU collision slot. Update
            # it directly; reloading the world would re-enable held obstacles.
            if checker.check_obstacle_exists(x):
                checker.enable_obstacle(x, enable=False)
            scene_model.remove_obstacle(x)
            for mesh in scene_model.mesh:
                if mesh.name == x:
                    scene_model.mesh.remove(mesh)
                    break
            CuroboMotion.cached_obstacle_info.pop(x, None)

    def attach_obj(
        self,
        prim_paths,
        link_name="attached_object",
        ee_position=[0, 0, 0],
        ee_rotation=[1, 0, 0, 0],
    ):
        """Detach previous objects, refresh the scene, then attach objects from the scene.
        The manager fits spheres, attaches them, and disables their world obstacles.
        Current joint-state FK determines tool pose; legacy ee_position/ee_rotation
        arguments remain for caller compatibility and are not used internally."""
        self.detach_obj()
        attach_result = False
        self.set_obstacles()
        logger.info(f"object_names={prim_paths}")
        # Retain unused parameters for caller compatibility.
        _ = (ee_position, ee_rotation)
        try:
            cu_js, _ = self._get_current_joint_state()
            # attach_from_scene requires (num_envs, dof); use the first
            # row of tiled cu_js for the single environment.
            single_js = cu_js[0:1]

            # Filter prim_paths to those actually registered in curobo's runtime
            # Filter using actual scene-collision entries rather than only SceneCfg meshes.
            # Cache capacity or obstacle type mismatches can omit configured names.
            # Otherwise enable_obstacle can reach a missing-name lookup
            # and fail in MeshData.get_idx
            # before attachment completes.
            registered = set()
            try:
                registered = set(
                    self.planner.scene_collision_checker.get_obstacle_names(env_idx=0) or []
                )
            except Exception:
                pass
            filtered = [p for p in prim_paths if p in registered]
            missing = [p for p in prim_paths if p not in registered]
            if missing:
                logger.warning(
                    f"attach_obj: {len(missing)} obstacle(s) not in curobo runtime scene "
                    f"(collision_cache full / name mismatch / filtered by update_world), skipping: "
                    f"{missing}"
                )
            if not filtered:
                logger.error("attach_obj: no attachable obstacles left after filter; skipping")
                return False

            # Automatic sphere-count estimation during VOXEL fitting
            # can exceed the configured attached-link capacity
            # and raise a sphere-slot error.
            # `Fitted X spheres but link 'attached_object' only has N sphere slots`.
            # Read the actual kinematics slot count
            # and leave two spare slots when requesting fitted spheres.
            try:
                kparams = self.planner.kinematics.kinematics_config
                _slot_cnt = int(
                    kparams.get_sphere_index_from_link_name(link_name).shape[0]
                )
                _num_spheres_cap = max(1, _slot_cnt - 2)
            except Exception as _e:
                logger.warning(
                    f"attach_obj: failed to query sphere slots for '{link_name}' "
                    f"({_e!r}); falling back to None (may raise on fit overflow)"
                )
                _num_spheres_cap = None

            for am in self._attachment_managers():
                am.attach_from_scene(
                    joint_states=single_js,
                    obstacle_names=filtered,
                    link_name=link_name,
                    surface_radius=0.005,
                    num_spheres=_num_spheres_cap,
                    # MORPHIT sampling can fail for small meshes such as apples.
                    # `morphit_sphere_fit: attempt N returned too few spheres` ×10 ,
                    # An empty fitted sphere set leaves attachment without collision geometry
                    # while the world obstacle can block later IK.
                    # VOXEL uses a bounding grid and SDF filtering without iterative fitting.
                    sphere_fit_type=SphereFitType.VOXEL,
                )
            for pp in filtered:
                if pp not in self.attached_objects:
                    self.attached_objects.append(pp)
            attach_result = True
        except Exception as e:
            logger.error(f"attach_obj failed: {e}")
            attach_result = False
        logger.info(f"attach_result={attach_result}")
        return attach_result

    def attach_objects_to_robot(
        self,
        object_names,
        surface_sphere_radius: float = 0.001,
        link_name: str = "attached_object",
        # As in attach_obj, MORPHIT can return no spheres for small objects.
        # The remaining world obstacle can obstruct
        # subsequent grasp IK; default to VOXEL fitting.
        sphere_fit_type: SphereFitType = SphereFitType.VOXEL,
        voxelize_method: str = "ray",
        world_objects_pose_offset=None,
        remove_obstacles_from_world_config: bool = False,
        ee_pose=None,
    ) -> bool:
        """Compatibility wrapper forwarding attachment to attach_from_scene.
        The manager handles sphere fitting and world-obstacle disabling. Legacy
        voxelize_method, ee_pose, and world_objects_pose_offset remain in the signature."""
        _ = (voxelize_method, ee_pose, world_objects_pose_offset)
        if not object_names:
            return False
        try:
            cu_js, _ = self._get_current_joint_state()
            single_js = cu_js[0:1]
            for am in self._attachment_managers():
                am.attach_from_scene(
                    joint_states=single_js,
                    obstacle_names=list(object_names),
                    link_name=link_name,
                    surface_radius=surface_sphere_radius,
                    sphere_fit_type=sphere_fit_type,
                )
            if remove_obstacles_from_world_config:
                scene_model = self.planner.scene_collision_checker.scene_model
                for object_name in object_names:
                    if scene_model.get_obstacle(object_name):
                        scene_model.remove_obstacle(object_name)
                    for mesh in list(scene_model.mesh or []):
                        if mesh.name == object_name:
                            scene_model.mesh.remove(mesh)
                            break
                    if object_name not in self.attached_objects:
                        self.attached_objects.append(object_name)
            return True
        except Exception as e:
            logger.error(f"attach_objects_to_robot failed: {e}")
            return False

    def get_articulation_action_without_lock_joints(self, cmd_state):
        tmp_idx_list = []
        for i in range(len(self.idx_list)):
            if (
                cmd_state.joint_names[i]
                not in self.robot_cfg["kinematics"]["lock_joints"]
            ):
                tmp_idx_list.append(i)
        if hasattr(cmd_state, "velocity") and cmd_state.velocity is not None:
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy()[tmp_idx_list],
                cmd_state.velocity.cpu().numpy()[tmp_idx_list],
                joint_indices=np.array(self.idx_list)[tmp_idx_list],
            )
        else:
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy()[tmp_idx_list],
                joint_indices=np.array(self.idx_list)[tmp_idx_list],
            )
        return art_action

    def follow_trajectory(
        self,
        trajectory_positions,
        trajectory_orientations,
        active_ee_name,
        solvable_threshold=0.5,
        max_consecutive_failures=3,
    ):
        """Build a joint trajectory by batch IK along end-effector waypoints in wxyz order.
        Repeat each target and seed across the batch. Start from current ordered joints;
        use each successful solution as the next seed and retain the seed after failures.
        Stop after max_consecutive_failures (at least one). Overall success requires
        the last solvable waypoint fraction (last_success_index + 1) / n_input to exceed
        solvable_threshold; individual waypoints may fail."""
        time_start = time.time()
        ik_calls = 0
        ik_time_total = 0.0
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be >= 1")

        sim_js = physics.get_joints_state(self.robot, as_torch=False)
        js_names = physics.get_dof_names(self.robot)
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
        cu_js = JointState(
            position=self.tensor_args.to_device(
                np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))
            ),
            joint_names=sim_js_names,
        )
        cu_js = cu_js.reorder(self.planner.kinematics.joint_names)

        ik_solver = self.planner.ik_solver
        # Single-arm trajectory following must constrain only the active tool.
        # GoalToolPose still requires entries for every configured tool frame,
        # but treating the inactive arm's placeholder pose as a hard 6D goal
        # makes even the current active-arm pose unsolvable on dual-arm G2.
        ik_criteria = {
            t: ToolPoseCriteria.disabled() for t in self.planner.tool_frames
        }
        ik_criteria[active_ee_name] = ToolPoseCriteria()
        ik_solver.update_tool_pose_criteria(ik_criteria)

        def restore_tool_pose_criteria():
            ik_solver.update_tool_pose_criteria(
                {t: ToolPoseCriteria() for t in self.planner.tool_frames}
            )

        cur_js = cu_js.reorder(ik_solver.joint_names)
        cur_q = cur_js.position.view(CUROBO_BATCH_SIZE, -1)
        n_seeds = ik_solver.config.num_seeds

        pos_list = []
        consecutive_fails = 0

        traj_pts = list(zip(trajectory_positions, trajectory_orientations))
        n_input = len(traj_pts)
        if n_input == 0:
            restore_tool_pose_criteria()
            self.cmd_plan = None
            self.cmd_idx = 0
            self.reached = True
            self.success = False
            logger.info("follow_trajectory: empty trajectory")
            return

        last_success_index = -1
        # Hold other tools at their target-marker world poses
        # while building the waypoint joint path.
        # For a single-tool robot, only the active goal is needed.
        hold_pose_map: Dict[str, Pose] = {}
        if len(self.planner.tool_frames) > 1:
            for t in self.planner.tool_frames:
                if t == active_ee_name:
                    continue
                try:
                    c_p, c_rot = self._get_target_link_pose(t)
                    hold_pose_map[t] = Pose(
                        position=self.tensor_args.to_device(np.asarray(c_p).reshape(1, 3)),
                        quaternion=self.tensor_args.to_device(np.asarray(c_rot).reshape(1, 4)),
                    )
                except Exception:
                    pass
        for idx, (pos_np, quat_np) in enumerate(traj_pts):
            pos_a = np.asarray(pos_np, dtype=np.float64).ravel()
            quat_a = np.asarray(quat_np, dtype=np.float64).ravel()
            # Construct GoalToolPose with active and held tools.
            # Infer the batch from tensors tiled to CUROBO_BATCH_SIZE.
            active_pose = Pose(
                position=self.tensor_args.to_device(
                    np.tile(pos_a.reshape(1, -1), (CUROBO_BATCH_SIZE, 1))
                ),
                quaternion=self.tensor_args.to_device(
                    np.tile(quat_a.reshape(1, -1), (CUROBO_BATCH_SIZE, 1))
                ),
            )
            goal_dict: Dict[str, Pose] = {active_ee_name: active_pose}
            for t, hp in hold_pose_map.items():
                goal_dict[t] = self._tile_torch_pose(hp, CUROBO_BATCH_SIZE)
            goal_tool_poses_with_hold = GoalToolPose.from_poses(
                goal_dict,
                ordered_tool_frames=self.planner.tool_frames,
                num_goalset=1,
            )
            # v1 `retract_config` (batch, dof) → v2 `current_state` JointState(batch, dof)
            current_state = JointState.from_position(
                cur_q, joint_names=ik_solver.joint_names,
            )
            t_ik = time.time()
            # v2 solve_pose uses current_state for seeds instead of retract_config.
            ik_res = ik_solver.solve_pose(
                goal_tool_poses_with_hold,
                current_state=current_state,
                return_seeds=10,
            )
            ik_time_total += time.time() - t_ik
            ik_calls += 1
            crs_bound_filter_res = (
                filter_ik_result_by_crs_bound(
                    ik_res.js_solution,
                    is_right_arm=(
                        self._active_is_right
                        if self._active_is_right is not None
                        else active_ee_name == (self.ee_link_name or "")
                    ),
                )
                if "G2" in self.robot_prim_path and self.filter_crs_bound
                else [True] * len(ik_res.js_solution)
            )
            raw_success_mask = np.array(ik_res.success.cpu())
            success_mask = raw_success_mask & np.array(crs_bound_filter_res)
            # Do not retry by dropping holds; GoalToolPose requires
            # all tool frames, so try once with the complete goal dictionary.
            if not success_mask.any():
                logger.info(
                    "follow_trajectory: waypoint=%s raw_ik_success=%s crs_bound_success=%s target_pos=%s target_quat=%s",
                    idx,
                    int(np.asarray(ik_res.success.cpu()).sum()),
                    int(np.asarray(crs_bound_filter_res).sum()),
                    pos_a.tolist(),
                    quat_a.tolist(),
                )
                consecutive_fails += 1
                if consecutive_fails >= max_consecutive_failures:
                    logger.info(
                        "follow_trajectory: stopping after %s consecutive IK failures",
                        max_consecutive_failures,
                    )
                    break
                continue
            consecutive_fails = 0
            last_success_index = idx
            # ik_res.solution is (B, S, D) when return_seeds > 1 (e.g. 20x10x34).
            # Select a *successful* seed first, then among them pick the one with minimum joint diff
            # to the previous joint state (use batch element that has successes).
            if success_mask.ndim == 2:
                # pick a batch row with at least one success (usually all rows are identical)
                b_sel = int(np.argmax(success_mask.any(axis=1)))
                seed_mask = torch.as_tensor(
                    success_mask[b_sel], device=ik_res.solution.device, dtype=torch.bool
                )
                prev_q = cur_q[b_sel].view(1, -1)  # (1, D)
                cand = ik_res.solution[b_sel]  # (S, D)
                if seed_mask.any():
                    cand_ok = cand[seed_mask]  # (K, D)
                    diffs = torch.sum(torch.abs(cand_ok - prev_q), dim=-1)  # (K,)
                    best = int(torch.argmin(diffs).item())
                    q1 = cand_ok[best].view(1, -1).clone()
                else:
                    # should be rare: success elsewhere but not in selected row
                    q1 = cand[0].view(1, -1).clone()
            else:
                # return_seeds == 1 case: (B, D)
                b_sel = int(np.argmax(success_mask.astype(bool))) if success_mask.ndim == 1 else 0
                q1 = ik_res.solution[b_sel].view(1, -1).clone()
            pos_list.append(q1)
            cur_q = q1.expand(CUROBO_BATCH_SIZE, -1).clone()

        if len(pos_list) == 0:
            restore_tool_pose_criteria()
            self.cmd_plan = None
            self.cmd_idx = 0
            self.reached = True
            self.success = False
            total = time.time() - time_start
            avg = (ik_time_total / ik_calls) if ik_calls > 0 else 0.0
            logger.info(
                "follow_trajectory: total=%.4fs, ik_calls=%s, ik_time=%.4fs, ik_avg=%.6fs",
                total,
                ik_calls,
                ik_time_total,
                avg,
            )
            logger.info("follow_trajectory: no IK points solved")
            return

        solve_ratio = (last_success_index + 1) / float(n_input)
        ok = solve_ratio > float(solvable_threshold)

        if not ok:
            restore_tool_pose_criteria()
            self.reached = True
            self.success = False
            total = time.time() - time_start
            avg = (ik_time_total / ik_calls) if ik_calls > 0 else 0.0
            logger.info(
                "follow_trajectory: total=%.4fs, ik_calls=%s, ik_time=%.4fs, ik_avg=%.6fs",
                total,
                ik_calls,
                ik_time_total,
                avg,
            )
            logger.info(
                "follow_trajectory: solve ratio %.4f <= threshold %.4f, success=False",
                solve_ratio,
                solvable_threshold,
            )
        else:
            restore_tool_pose_criteria()
            pos = torch.cat(pos_list, dim=0)
            cmd = JointState(
                position=pos,
                joint_names=ik_solver.joint_names,
            )
            self.idx_list = []
            common_js_names = []
            for x in sim_js_names:
                if x in cmd.joint_names:
                    self.idx_list.append(self.robot.get_dof_index(x))
                    common_js_names.append(x)
            self.cmd_plan = cmd.reorder(common_js_names)
            self.cmd_idx = 0
            self.reached = False
            self.success = True
            total = time.time() - time_start
            avg = (ik_time_total / ik_calls) if ik_calls > 0 else 0.0
            logger.info(
                "follow_trajectory: total=%.4fs, ik_calls=%s, ik_time=%.4fs, ik_avg=%.6fs",
                total,
                ik_calls,
                ik_time_total,
                avg,
            )
            logger.info(
                "follow_trajectory: last solved index %s (n_input=%s), span ratio %.4f, %s IK waypoints",
                last_success_index,
                n_input,
                solve_ratio,
                len(pos_list),
            )

    def _get_current_joint_state(self):
        sim_js = physics.get_joints_state(self.robot, as_torch=False)
        js_names = physics.get_dof_names(self.robot)
        sim_js_names = []
        lock_idx = []
        sim_js_positions = []
        for idx, name in enumerate(js_names):
            if name not in self.lock_js_names:
                sim_js_names.append(name)
            else:
                lock_idx.append(idx)
        for idx, position in enumerate(sim_js.positions):
            if idx not in lock_idx:
                sim_js_positions.append(position)
        sim_js_positions = np.array(sim_js_positions)[np.newaxis, :]
        cu_js = JointState(
            position=self.tensor_args.to_device(
                np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))
            ),
            velocity=self.tensor_args.to_device(
                np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))
            )
            * 0.0,
            acceleration=self.tensor_args.to_device(
                np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))
            )
            * 0.0,
            jerk=self.tensor_args.to_device(
                np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))
            )
            * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.reorder(self.planner.kinematics.joint_names)
        # Ensure the current state excludes locked-joint names.
        # cuRobo augmentation rejects names shared with
        # its locked-joint snapshot.
        # Reordering normally leaves only active planner joints,
        # but lock-key changes during config updates
        # can make the stored KinematicsCfg snapshot
        # overlap the current active-joint selection.
        # For example, a mimic joint can become explicitly locked
        # and trigger the fast-path overlap assertion.
        # Filter again using lock_jointstate when available.
        try:
            lock_state = getattr(self.planner.kinematics, "lock_jointstate", None)
            if lock_state is not None and lock_state.joint_names is not None:
                lock_set = set(lock_state.joint_names)
                bad = [n for n in cu_js.joint_names if n in lock_set]
                if bad:
                    logger.warning(
                        f"[curobo] cu_js.joint_names contains lock joints: "
                        f"{bad} — filtering out; planner.lock_jointstate has "
                        f"{list(lock_state.joint_names)}"
                    )
                    keep = [n for n in cu_js.joint_names if n not in lock_set]
                    cu_js = cu_js.reorder(keep)
        except Exception as _e:
            logger.debug(f"[curobo] defensive lock-filter check failed: {_e}")

        return cu_js, sim_js_names

    def plan_bimanual_leader(
        self,
        target_position,
        target_rotation,
        follower_ee_link,
        follower_ee_pose,
        goal_offset=None,
        path_constraint=None,
        offset_and_constraint_in_goal_frame=True,
        disable_collision_links=None,
        from_current_pose=False,
        active_tool: Optional[str] = None,
    ):
        """Plan the leader target while holding the follower at follower_ee_pose.
        Put both tools in one GoalToolPose and return successful (batch_idx, JointState) paths."""
        t0 = time.time()
        self.reached = False
        self.success = False
        active_tool = active_tool or self.ee_link_name or self.tool_frames[0]
        cu_js, sim_js_names = self._get_current_joint_state()

        ee_translation_goal = np.asarray(target_position, dtype=np.float64).ravel()
        ee_orientation_goal = np.asarray(target_rotation, dtype=np.float64).ravel()

        if from_current_pose:
            kin_state = self.planner.compute_kinematics(cu_js[0:1])
            tp = kin_state.tool_poses.get_link_pose(active_tool)
            ee_translation_goal = tp.position.view(-1, 3)[0].detach().cpu().numpy()
            ee_orientation_goal = tp.quaternion.view(-1, 4)[0].detach().cpu().numpy()

        if goal_offset is not None:
            offset = Pose.from_list(goal_offset)
            goal_pose_list = np.concatenate([ee_translation_goal, ee_orientation_goal]).tolist()
            goal_pose = Pose.from_list(goal_pose_list)
            if offset_and_constraint_in_goal_frame:
                offset_goal_pose = goal_pose.clone().multiply(offset)
            else:
                offset_goal_pose = offset.clone().multiply(goal_pose.clone())
            ee_translation_goal = offset_goal_pose.position.squeeze().cpu().numpy()
            ee_orientation_goal = offset_goal_pose.quaternion.squeeze().cpu().numpy()

        if (
            path_constraint is not None
            and len(path_constraint) == 6
            and path_constraint[-3] > 0
            and path_constraint[-2] > 0
            and path_constraint[-1] > 0
        ):
            logger.info(
                "plan_bimanual_leader: path constraint constrains all positions, use ee_pose as goal position"
            )
            kin_state = self.planner.compute_kinematics(cu_js[0:1])
            tp = kin_state.tool_poses.get_link_pose(active_tool)
            ee_translation_goal = tp.position.view(-1, 3)[0].detach().cpu().numpy()

        # Put leader and follower in one GoalToolPose.
        # Use the caller follower target rather than inferring it from FK.
        goal_tool_poses = self._build_batch_goal(
            active_tool,
            ee_translation_goal,
            ee_orientation_goal,
            cu_js,
            hold_link_poses={follower_ee_link: follower_ee_pose},
        )
        criteria = self._build_tool_pose_criteria(
            active_tool, path_constraint, offset_and_constraint_in_goal_frame,
        )
        self.planner.update_tool_pose_criteria(criteria)

        disable_collision_links = list(
            filter(
                lambda link: any(
                    re.match(pattern, link) for pattern in (disable_collision_links or [])
                ),
                self.collision_link_names,
            )
        )
        self.planner.disable_link_collision(disable_collision_links)

        try:
            result = self.planner.plan_pose(
                goal_tool_poses,
                cu_js,
                max_attempts=self.max_attempts,
                success_ratio=self.success_ratio,
                enable_graph_attempt=self.enable_graph_attempt,
                finetune_attempts=0,
                initial_iters=25,
            )
            if (
                result is None
                or result.success is None
                or not bool(result.success.any().item())
            ):
                self.reached = True
                self.success = False
                logger.info(
                    "plan_bimanual_leader: plan_pose failed: %s",
                    "no result" if result is None else "no batch success",
                )
                self.planner.enable_link_collision(disable_collision_links)
                return None

            # extract per-batch candidate trajectories with same helper as calculate_ik_goal
            candidate_paths, pos_err_per_path, rot_err_per_path = self._extract_candidate_paths(result)
            # Retain batch_idx for callers expecting trajectory tuples.
            batch_success = result.success.any(dim=-1)
            successful_batch_indices = [
                bi for bi in range(batch_success.shape[0]) if bool(batch_success[bi].item())
            ]
            successful_trajectories = []
            for bi, path in zip(successful_batch_indices, candidate_paths):
                # Candidate paths already contain full DOFs
                # from trajectory optimization expansion.
                # Do not call get_full_js again; it duplicates locked joints.
                # "lock_joints is also listed in js.joint_names".
                traj = path.reorder(
                    [n for n in self.planner.kinematics.joint_names if n in path.joint_names]
                )
                successful_trajectories.append((bi, traj))

            position_filter_res = filter_paths_by_position_error(candidate_paths, pos_err_per_path)
            rotation_filter_res = filter_paths_by_rotation_error(candidate_paths, rot_err_per_path)
            crs_bound_filter_res = [True] * len(candidate_paths)
            if "G2" in self.robot_prim_path and self.filter_crs_bound:
                crs_bound_filter_res = filter_paths_by_crs_bound(
                    candidate_paths,
                    is_right_arm=(
                        self._active_is_right
                        if self._active_is_right is not None
                        else active_tool == (self.ee_link_name or "")
                    ),
                )
            filtered_trajectories = []
            for i, (bi, traj) in enumerate(successful_trajectories):
                if position_filter_res[i] and rotation_filter_res[i] and crs_bound_filter_res[i]:
                    filtered_trajectories.append((bi, traj))
            logger.info(
                "plan_bimanual_leader: paths %d -> filtered %d "
                "(pos_ok=%d, rot_ok=%d, crs_ok=%d)",
                len(successful_trajectories),
                len(filtered_trajectories),
                sum(position_filter_res),
                sum(rotation_filter_res),
                sum(crs_bound_filter_res),
            )
            if len(filtered_trajectories) == 0:
                filtered_trajectories = successful_trajectories

            # Indices retained from candidate_paths.
            filtered_paths_only = [candidate_paths[i] for i, tup in enumerate(successful_trajectories)
                                   if tup in filtered_trajectories]
            # Sort the filtered paths directly.
            sort_paths = [candidate_paths[
                successful_trajectories.index(t)
            ] for t in filtered_trajectories]
            sorted_indices = sort_by_difference_js(
                sort_paths,
                weights=self.tensor_args.to_device(
                    [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0]
                ) if sort_paths and sort_paths[0].position.shape[-1] == 14 else None,
            )
            successful_trajectories = [filtered_trajectories[si] for si in sorted_indices]

            self.planner.enable_link_collision(disable_collision_links)
            logger.info(
                "plan_bimanual_leader: %d successful trajectories, time=%.4fs",
                len(successful_trajectories),
                time.time() - t0,
            )
            return successful_trajectories

        except Exception as e:
            self.reached = True
            self.success = False
            logger.info("plan_bimanual_leader: exception: %s", e)
            self.planner.enable_link_collision(disable_collision_links)
            return None

    def build_follower_scatter_plan(
        self,
        leader_motion,
        leader_trajectory,
        pose_rel,
        max_consecutive_failures=1,
        leader_active_tool: Optional[str] = None,
        follower_active_tool: Optional[str] = None,
    ):
        """Build follower IK waypoints from leader_pose * pose_rel.
        Read v2 tool poses through planner KinematicsState. Explicit leader/follower tool
        arguments select the arms; omitted arguments fall back to ee_link_name."""
        t0 = time.time()
        cu_js, sim_js_names = self._get_current_joint_state()
        ik_solver = self.planner.ik_solver
        cur_js = cu_js.reorder(ik_solver.joint_names)
        cur_q = cur_js.position.view(CUROBO_BATCH_SIZE, -1)
        leader_planner = getattr(leader_motion, "planner", None) or leader_motion.motion_gen
        leader_tool = (
            leader_active_tool
            or getattr(leader_motion, "ee_link_name", None)
            or leader_planner.tool_frames[0]
        )
        follower_tool = (
            follower_active_tool or self.ee_link_name or self.tool_frames[0]
        )

        pos_list = []
        consecutive_fails = 0
        n_steps = len(leader_trajectory.position)

        for ti in range(n_steps):
            js_i = leader_trajectory[ti].reorder(
                leader_planner.kinematics.joint_names
            )
            kin_i = leader_planner.compute_kinematics(js_i)
            pose_leader_i = kin_i.tool_poses.get_link_pose(leader_tool)
            pose_follower_i = pose_leader_i.multiply(pose_rel)

            pos_np = pose_follower_i.position.view(-1, 3)[0].detach().cpu().numpy().astype(np.float64)
            quat_np = pose_follower_i.quaternion.view(-1, 4)[0].detach().cpu().numpy().astype(np.float64)

            # Build follower IK for its active tool.
            # Single-arm follower YAML matches previous behavior; dual-arm callers
            # must avoid jointly constraining the leader in this solve.
            active_pose = Pose(
                position=self.tensor_args.to_device(
                    np.tile(pos_np.reshape(1, -1), (CUROBO_BATCH_SIZE, 1))
                ),
                quaternion=self.tensor_args.to_device(
                    np.tile(quat_np.reshape(1, -1), (CUROBO_BATCH_SIZE, 1))
                ),
            )
            goal_tool_poses = GoalToolPose.from_poses(
                {follower_tool: active_pose},
                ordered_tool_frames=self.planner.tool_frames,
                num_goalset=1,
            )
            current_state = JointState.from_position(
                cur_q, joint_names=ik_solver.joint_names,
            )
            ik_res = ik_solver.solve_pose(
                goal_tool_poses,
                current_state=current_state,
                return_seeds=10,
            )

            success_mask = np.array(ik_res.success.cpu())
            if "G2" in self.robot_prim_path and self.filter_crs_bound:
                crs_filter = filter_ik_result_by_crs_bound(
                    ik_res.js_solution,
                    is_right_arm=(
                        self._active_is_right
                        if self._active_is_right is not None
                        else follower_tool == (self.ee_link_name or "")
                    ),
                )
                success_mask = success_mask & np.array(crs_filter)

            if not success_mask.any():
                consecutive_fails += 1
                if consecutive_fails >= max_consecutive_failures:
                    logger.info(
                        "build_follower_scatter_plan: stopped at step %d after %d consecutive IK failures",
                        ti, max_consecutive_failures,
                    )
                    break
                continue

            consecutive_fails = 0
            if success_mask.ndim == 2:
                b_sel = int(np.argmax(success_mask.any(axis=1)))
                seed_mask = torch.as_tensor(
                    success_mask[b_sel], device=ik_res.solution.device, dtype=torch.bool,
                )
                prev_q = cur_q[b_sel].view(1, -1)
                cand = ik_res.solution[b_sel]
                if seed_mask.any():
                    cand_ok = cand[seed_mask]
                    diffs = torch.sum(torch.abs(cand_ok - prev_q), dim=-1)
                    best = int(torch.argmin(diffs).item())
                    q1 = cand_ok[best].view(1, -1).clone()
                else:
                    q1 = cand[0].view(1, -1).clone()
            else:
                b_sel = int(np.argmax(success_mask.astype(bool))) if success_mask.ndim == 1 else 0
                q1 = ik_res.solution[b_sel].view(1, -1).clone()

            pos_list.append(q1)
            cur_q = q1.expand(CUROBO_BATCH_SIZE, -1).clone()

        if len(pos_list) == 0:
            logger.info("build_follower_scatter_plan: no IK points solved")
            self.success = False
            return

        if len(pos_list) != n_steps:
            logger.info(
                "build_follower_scatter_plan: %d/%d steps solved", len(pos_list), n_steps,
            )
            self.success = False
            return

        pos = torch.cat(pos_list, dim=0)
        cmd = JointState(
            position=pos,
            joint_names=ik_solver.joint_names,
        )
        self.idx_list = []
        common_js_names = []
        for x in sim_js_names:
            if x in cmd.joint_names:
                self.idx_list.append(self.robot.get_dof_index(x))
                common_js_names.append(x)
        self.cmd_plan = cmd.reorder(common_js_names)
        # Consume the same controller-resolved slowdown as the leader.
        _total = consume_resolved_slowdown_factor(
            getattr(self, "pending_slowdown_factor", 1.0)
        )
        if abs(_total - 1.0) > 1e-3:
            self.cmd_plan = self._resample_cmd_plan(self.cmd_plan, _total)
        self.pending_slowdown_factor = 1.0
        self.cmd_idx = 0
        self.reached = False
        self.success = True
        logger.info(
            "build_follower_scatter_plan: %d waypoints, time=%.4fs",
            len(pos_list), time.time() - t0,
        )

    def on_physics_step(
        self, run_ratio=1.0, additional_action: ArticulationAction = None
    ):
        self.time_index += 1
        if run_ratio <= 0.0 or run_ratio > 1.0:
            logger.info("run_ratio should be in the range (0, 1], setting to 1.0")
            run_ratio = 1.0

        if self.cmd_plan is not None:
            cmd_state = self.cmd_plan[self.cmd_idx]
            self.past_cmd = cmd_state.clone()
            art_action = self.get_articulation_action_without_lock_joints(cmd_state)
            # merge art_action with additional_action if provided
            if (
                additional_action is not None
                and additional_action.joint_positions is not None
            ):
                for idx in range(len(additional_action.joint_positions)):
                    if additional_action.joint_positions[idx] is None:
                        continue
                    if idx in art_action.joint_indices:
                        art_idx = art_action.joint_indices.index(idx)
                        art_action.joint_positions[art_idx] = (
                            additional_action.joint_positions[idx]
                        )
                        art_action.joint_velocities[art_idx] = (
                            additional_action.joint_velocities[idx]
                        )
                    else:
                        art_action.joint_indices = np.append(
                            art_action.joint_indices, idx
                        )
                        art_action.joint_positions = np.append(
                            art_action.joint_positions,
                            additional_action.joint_positions[idx],
                        )
                        art_action.joint_velocities = np.append(
                            art_action.joint_velocities,
                            additional_action.joint_velocities[idx],
                        )
            physics.apply_action(self.robot, art_action)
            self.cmd_idx += 1
            if self.cmd_idx >= len(self.cmd_plan.position) * run_ratio:
                # Begin settling and retain the final target for PD checks.
                # Use the action after merging additional gripper commands.
                self._reach_settle_target = np.asarray(
                    art_action.joint_positions, dtype=np.float64
                )
                self._reach_settle_dof_indices = np.asarray(
                    art_action.joint_indices, dtype=np.int64
                )
                self._reach_settle_counter = 0
                self._reach_settle_stall_counter = 0
                self._reach_settle_prev_cur = None
                self._reach_settle_active = True
                self.cmd_idx = 0
                self.cmd_plan = None
                self.past_cmd = None
        elif self._reach_settle_active:
            # Wait for target convergence, stall, or timeout after the path ends.
            self._reach_settle_counter += 1
            cur = np.asarray(
                physics.get_joint_positions(
                    self.robot,
                    joint_indices=self._reach_settle_dof_indices,
                    as_torch=False,
                ),
                dtype=np.float64,
            )
            diff_target = float(np.max(np.abs(cur - self._reach_settle_target)))
            # Count consecutive steps below eps_stall; reset on larger motion.
            if self._reach_settle_prev_cur is not None:
                diff_step = float(np.max(np.abs(cur - self._reach_settle_prev_cur)))
                if diff_step < self._reach_settle_eps_stall:
                    self._reach_settle_stall_counter += 1
                else:
                    self._reach_settle_stall_counter = 0
            else:
                diff_step = float("nan")
            self._reach_settle_prev_cur = cur

            physics_dt = self.my_world.get_physics_dt()
            max_steps = max(1, int(self._reach_settle_max_delay_s / physics_dt))
            reached_by_target = diff_target < self._reach_settle_eps_pos
            reached_by_stall = (
                self._reach_settle_stall_counter >= self._reach_settle_stall_window
            )
            reached_by_timeout = self._reach_settle_counter >= max_steps
            if reached_by_target or reached_by_stall or reached_by_timeout:
                reason = (
                    "target" if reached_by_target
                    else "stall" if reached_by_stall
                    else "timeout"
                )
                final_counter = self._reach_settle_counter
                final_stall = self._reach_settle_stall_counter
                self._reach_settle_active = False
                self._reach_settle_target = None
                self._reach_settle_dof_indices = None
                self._reach_settle_prev_cur = None
                self._reach_settle_stall_counter = 0
                self.reached = True
                logger.info(
                    f"Reached{self.reached} (reason={reason}, counter={final_counter}, "
                    f"max_joint_err={diff_target:.4f}rad, step_diff={diff_step:.4f}rad, "
                    f"stall={final_stall})"
                )
