# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
from typing import Mapping, Optional, Sequence, Tuple
import numpy as np
import threading
import queue
import json
from pathlib import Path
import asyncio
import subprocess
import signal, shutil
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdPhysics, PhysxSchema, UsdLux
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
from isaacsim.core.utils.stage import add_reference_to_stage, update_stage
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.xforms import get_world_pose

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()  # Create singleton instance
from geniesim_benchmark.app.utils import RobotCfg
from geniesim_benchmark.app.utils import material_changer, Light
from geniesim_benchmark.app.utils.utils import (
    get_rotation_matrix_from_quaternion,
    get_quaternion_from_euler,
    matrix_to_euler_angles,
    rotation_matrix_to_quaternion,
)
from geniesim_benchmark.utils import system_utils
from geniesim_benchmark.utils.usd_utils import *
from geniesim_benchmark.utils.ros_nodes.server_node import *
from geniesim_benchmark.config.params import Config
from geniesim_benchmark.app.ros_publisher.base import USDBase
from geniesim_benchmark.app.ros_publisher.robot_interface import RobotInterface
from geniesim_benchmark.plugins.output_system.local_recorder import LocalRecorder
from geniesim_benchmark.app.workflow.ui_builder import UIBuilder
from geniesim_benchmark.robot.utils import quaternion_rotate
from geniesim_benchmark.utils.name_utils import *
from geniesim_benchmark.utils.system_utils import *

import time


class APICore:
    def __init__(self, ui_builder: UIBuilder, config: Config):
        # ROS is opt-in: only init a context for the teleop / enable_ros flow.
        # The eval path creates no ROS nodes, so it needs no rclpy context.
        if config.app.enable_ros:
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
        # (robot_cfg, scene_usd_path) of the robot+/World already on the stage.
        # Lets serial instances reuse them and swap only /Workspace.
        self._loaded_scene_key = None
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
        self.teleop_recording = False
        self.arm_base_prim_path = "arm_base_link"
        self._current_mode = "realtime"
        self._stage = omni.usd.get_context().get_stage()
        # Vec-mode per-env state (see fork_for_env). Empty/None on root view.
        self.env_root = ""
        self._current_articulation = None
        self._cur_env_idx = 0

        # app config
        self.enable_physics = not config.app.disable_physics
        self.enable_curobo = config.app.enable_curobo
        self.enable_pub_depth_camera = getattr(config.app, "enable_pub_depth_camera", False)
        self.reset_fallen = config.app.reset_fallen
        self.rendering_step = config.app.rendering_step
        self.enable_ros = config.app.enable_ros
        self.record_images = config.app.record_img
        self.record_video = config.app.record_video
        self.data_convert = config.app.data_convert
        self.enable_playback = config.app.enable_playback
        self.on_demand_render = getattr(config.app, "on_demand_render", False)
        self.enable_gpu_dynamics = getattr(config.app, "enable_gpu_dynamics", False)

        self._is_vec_mode = getattr(config.benchmark, "enable_vec", 0) > 0
        # Vec mode forces on_demand_render: shared-cam renders are explicit
        # via render_once(); main-loop world.step(render=...) is wasted work.
        if self._is_vec_mode:
            self.on_demand_render = True
        # RTX subframes per shared-camera teleport (orchestrator.step rt_subframes).
        self.shared_cam_render_frames = getattr(config.benchmark, "shared_cam_render_frames", 8)
        self.shared_cam_first_render_frames = getattr(config.benchmark, "shared_cam_first_render_frames", 8)
        self._warmed_up_envs = set()

        # layout config
        self.seed = config.layout.seed
        self.autogen_ratio = config.layout.autogen_ratio
        self.num_obj = config.layout.num_obj

        # task config
        self.task_name = config.benchmark.task_name
        self.sub_task_name = config.benchmark.sub_task_name
        task_config_file = os.path.join(system_utils.benchmark_conf_path(), "eval_tasks", self.task_name + ".json")
        self.task_config = system_utils.load_json(task_config_file)

        # robot data
        self.sensor_base = USDBase()
        self.robot_interface: RobotInterface = RobotInterface()
        if not self.enable_ros:
            self.robot_interface.disable_ros_pub()
        self.ros_node_initialized = False

        # In-process video recorder (replaces the ROS rosbag pipeline).
        self.local_recorder = LocalRecorder(fps=30)
        self.playback_timerange = []
        self.playback_start = 0
        self.playback_end = 0
        self.add_object_flag = False
        self.reset_flag = False
        self.init_frame_info = {}
        self.sensor_base_initialized = False
        self.robot_initialized = False
        self.index = 0
        self.wait_recording = True
        self.recording_wait_num = 0
        self.robot_joint_indices = {}

        # On-demand render gating. ``frame_render_enabled`` is read by the
        # main loop's ``world.step(render=...)`` call. When on_demand_render
        # is False the flag stays True (legacy behaviour: render every frame).
        # When on_demand_render is True it follows the request refcount —
        # callers wrap rendering-dependent work in request_render() /
        # release_render() pairs (or use the render_scope() context manager).
        self._render_needed = threading.Event()
        self._render_needed.set()
        self._render_request_count = 0
        self._render_lock = threading.Lock()
        if self.on_demand_render:
            self._render_needed.clear()
        # One-shot render hold spanning task_manager.start() → first _play().
        self._startup_render_held = False

    def request_render(self):
        with self._render_lock:
            self._render_request_count += 1
            self._render_needed.set()

    def release_render(self):
        with self._render_lock:
            self._render_request_count = max(0, self._render_request_count - 1)
            if self._render_request_count == 0 and self.on_demand_render:
                self._render_needed.clear()

    @property
    def render_needed(self) -> bool:
        return self._render_needed.is_set()

    @property
    def frame_render_enabled(self) -> bool:
        if not self.on_demand_render:
            return True
        return self._render_needed.is_set()

    def run_on_render_loop(self, func, *args, timeout=120, **kwargs):
        done = threading.Event()
        result = {}

        def wrapper():
            try:
                result["value"] = func(*args, **kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                done.set()

        self.task_queue_on_render_loop.put(wrapper)
        if not done.wait(timeout=timeout):
            logger.error(f"run_on_render_loop timed out after {timeout}s for {func.__name__}")
            raise TimeoutError(f"run_on_render_loop: {func.__name__} did not complete within {timeout}s")

        if "error" in result:
            raise result["error"]
        return result.get("value")

    def run_on_physics_loop(self, func, *args, timeout=120, **kwargs):
        done = threading.Event()
        result = {}

        def wrapper():
            try:
                result["value"] = func(*args, **kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                done.set()

        self.task_queue_on_physics_loop.put(wrapper)
        if not done.wait(timeout=timeout):
            logger.error(f"run_on_physics_loop timed out after {timeout}s for {func.__name__}")
            raise TimeoutError(f"run_on_physics_loop: {func.__name__} did not complete within {timeout}s")

        if "error" in result:
            raise result["error"]
        return result.get("value")

    def render_step(self):
        try:
            self._on_recording()
            self._on_playback()
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

    @property
    def is_vec_mode(self) -> bool:
        return self._is_vec_mode

    def _env_path(self, prim_path: str) -> str:
        """No-op on root view; rewrites under env_root for forked views."""
        if not self.env_root:
            return prim_path
        if prim_path.startswith(self.env_root):
            return prim_path
        return self.env_root + prim_path

    def init_robot_cfg_multi(
        self, robot_cfg, scene_usd, init_position, init_rotation, sub_usd_paths, n_envs, env_spacing=20.0
    ):
        """Vec-mode entry point: build N cloned envs on a single stage."""
        return self.run_on_render_loop(
            self._init_robot_cfg_multi,
            robot_cfg,
            scene_usd,
            init_position,
            init_rotation,
            sub_usd_paths,
            n_envs,
            env_spacing,
        )

    def _init_robot_cfg_multi(
        self, robot_cfg, scene_usd, init_position, init_rotation, sub_usd_paths, n_envs, env_spacing
    ):
        from omni.isaac.cloner import GridCloner

        self.robot_cfg = RobotCfg(str(system_utils.app_root_path()) + "/robot_cfg/" + robot_cfg)
        robot_usd_path = str(system_utils.assets_path()) + "/" + self.robot_cfg.robot_usd
        scene_usd_str = str(np.random.choice(scene_usd)) if isinstance(scene_usd, list) else scene_usd
        scene_usd_path = str(system_utils.assets_path()) + "/" + str(scene_usd_str)

        cloner = GridCloner(spacing=env_spacing)
        cloner.define_base_env("/World/envs")
        env_paths = cloner.generate_paths("/World/envs/env", n_envs)
        env0 = env_paths[0]

        robot_prim_in_env0 = env0 + self.robot_cfg.robot_prim_path
        add_reference_to_stage(robot_usd_path, robot_prim_in_env0)
        add_reference_to_stage(scene_usd_path, env0)
        # Filter on env0 first; cloner copies edits to other env clones.
        self._filter_objects(scene_usd_path, world_root=env0)

        self.usd_objects["robot"] = SingleXFormPrim(
            prim_path=robot_prim_in_env0,
            translation=init_position,
            orientation=init_rotation,
        )

        self.scene = UsdPhysics.Scene.Define(self._stage, Sdf.Path("/physicsScene"))
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        self.scene.CreateGravityMagnitudeAttr().Set(9.81)

        cloner.clone(
            source_prim_path=env0,
            prim_paths=env_paths,
            copy_from_source=True,
            replicate_physics=False,
        )

        for i, env_path in enumerate(env_paths):
            if sub_usd_paths and i < len(sub_usd_paths) and sub_usd_paths[i]:
                ws_path = env_path + "/Workspace"
                add_reference_to_stage(sub_usd_paths[i], ws_path)

        cloner.filter_collisions(
            physicsscene_path="/physicsScene",
            collision_root_path="/World/collisions",
            prim_paths=env_paths,
            global_paths=[],
        )

        self.robot_init_position = init_position
        self.robot_init_rotation = init_rotation
        self.scene_usd = scene_usd_str
        self.scene_glb = os.path.join(os.path.dirname(scene_usd_str), "compressed_simplified.glb")
        if "multispace" in scene_usd_str:
            self.scene_name = scene_usd_str.split("/")[-3] + "/" + scene_usd_str.split("/")[-2]
        else:
            self.scene_name = scene_usd_str.split("/")[-2]
        self.robot_name = self.robot_cfg.robot_name
        self.material_changer = material_changer()

        physics_scene = PhysxSchema.PhysxSceneAPI.Get(self._stage, "/physicsScene")
        if getattr(self, "enable_gpu_dynamics", False):
            physics_scene.CreateGpuMaxRigidContactCountAttr(8388608 * n_envs)
            physics_scene.CreateGpuMaxRigidPatchCountAttr(163840 * n_envs)
            physics_scene.CreateGpuFoundLostPairsCapacityAttr(2097152 * n_envs)
            physics_scene.CreateGpuFoundLostAggregatePairsCapacityAttr(33554432 * n_envs)
            physics_scene.CreateGpuTotalAggregatePairsCapacityAttr(2097152 * n_envs)

        time.sleep(1)

        self.ui_builder.my_world.reset()
        self.frame_status = []
        # Vec doesn't call _play(), so release startup render hold here.
        if getattr(self, "_startup_render_held", False):
            self._startup_render_held = False
            self.release_render()

        time.sleep(1)
        self._initialize_all_scene_articulations()

        self._env_articulations = {}
        self._env_robot_xforms = {}
        for i, env_path in enumerate(env_paths):
            robot_prim = env_path + self.robot_cfg.robot_prim_path
            art = SingleArticulation(prim_path=robot_prim, name=f"{self.robot_name}_env{i}")
            art.initialize()
            self._env_articulations[i] = art
            self._env_robot_xforms[i] = SingleXFormPrim(prim_path=robot_prim)

        env0_art = self._env_articulations[0]
        self.robot_joint_indices = {name: idx for idx, name in enumerate(env0_art.dof_names)}

        needed_cam_names = set()
        for camera_prim in self.robot_cfg.cameras:
            needed_cam_names.add(camera_prim.split("/")[-1])
        for env_path in env_paths:
            self._deactivate_unused_cameras(env_path, needed_cam_names)

        self._isaacsim_annotators = {}
        self._isaacsim_depth_annotators = {}
        self._setup_shared_cameras(env_paths)

        return env_paths

    def _deactivate_unused_cameras(self, env_path, needed_cam_names):
        stage = self._stage
        env_prim = stage.GetPrimAtPath(env_path)
        if not env_prim.IsValid():
            return
        for prim in Usd.PrimRange(env_prim):
            if prim.IsA(UsdGeom.Camera):
                cam_name = prim.GetName()
                if cam_name not in needed_cam_names:
                    prim.SetActive(False)
                    logger.info(f"Deactivated unused camera: {prim.GetPath()}")

    # Shared-camera teleport rendering: one set of top-level cameras under
    # /World/shared_cameras, teleported per env then RTX-accumulated.

    _SHARED_CAM_ROOT = "/World/shared_cameras"

    def _setup_shared_cameras(self, env_paths):
        stage = self._stage
        env0 = env_paths[0]
        infos = []
        for cam_rel, resolution in self.robot_cfg.cameras.items():
            w, h = int(resolution[0]), int(resolution[1])
            cam_full = env0 + cam_rel
            cam_prim = stage.GetPrimAtPath(cam_full)
            if not cam_prim.IsValid():
                logger.warning(f"[shared_cam] camera prim not found at {cam_full}, skipping")
                continue
            parent_rel = "/".join(cam_rel.split("/")[:-1])
            name = cam_rel.split("/")[-1].lower()
            translate_attr = cam_prim.GetAttribute("xformOp:translate")
            orient_attr = cam_prim.GetAttribute("xformOp:orient")
            local_t = Gf.Vec3d(translate_attr.Get()) if translate_attr.IsValid() else Gf.Vec3d(0, 0, 0)
            local_q = Gf.Quatd(orient_attr.Get()) if orient_attr.IsValid() else Gf.Quatd(1, 0, 0, 0)

            # Clone non-xform attrs (focal/aperture/distortion) — needed for
            # renderer to match training distribution.
            attrs = {}
            for attr in cam_prim.GetAttributes():
                nm = attr.GetName()
                if nm.startswith("xformOp:") or nm == "xformOpOrder":
                    continue
                val = attr.Get()
                if val is None:
                    continue
                attrs[nm] = (val, attr.GetTypeName())

            infos.append(
                dict(
                    rel_path=cam_rel,
                    name=name,
                    parent_link_rel=parent_rel,
                    resolution=(w, h),
                    local_translate=local_t,
                    local_orient=local_q,
                    all_attrs=attrs,
                    applied_schemas=list(cam_prim.GetAppliedSchemas()),
                )
            )

        if not infos:
            logger.warning("[shared_cam] no cameras discovered; skip shared-camera setup")
            return

        stage.DefinePrim(self._SHARED_CAM_ROOT, "Xform")
        self._shared_cam_infos = infos
        self._shared_cam_paths = {}
        self._shared_cam_rgb_ann = {}
        self._shared_cam_depth_ann = {}
        self._shared_render_cache = {}
        # Native cam prims used for teleport target via GetLocalToWorldTransform.
        self._native_cam_prims: dict = {}
        for env_idx, env_path in enumerate(env_paths):
            self._native_cam_prims[env_idx] = {}
            for info in infos:
                native_cam_path = env_path + info["rel_path"]
                native_prim = stage.GetPrimAtPath(native_cam_path)
                if native_prim.IsValid():
                    self._native_cam_prims[env_idx][info["name"]] = native_prim
                else:
                    logger.warning(f"[shared_cam] native camera prim missing at {native_cam_path}")

        for info in infos:
            path = f"{self._SHARED_CAM_ROOT}/{info['name']}"
            cam = UsdGeom.Camera.Define(stage, path)
            prim = cam.GetPrim()

            # Apply API schemas before attrs (e.g. lens distortion attrs need it).
            for schema_name in info.get("applied_schemas", []):
                applied = False
                try:
                    prim.ApplyAPI(schema_name)
                    applied = True
                except (TypeError, AttributeError, Exception):
                    try:
                        from pxr import Usd as _Usd

                        schema_type = _Usd.SchemaRegistry().GetTypeFromName(schema_name)
                        if schema_type:
                            prim.ApplyAPI(schema_type)
                            applied = True
                    except Exception as e:
                        logger.warning(f"[shared_cam] could not apply API schema " f"'{schema_name}' on {path}: {e}")
                if not applied:
                    try:
                        from pxr import Sdf as _Sdf

                        listop = _Sdf.TokenListOp.Create(prependedItems=[schema_name])
                        prim.SetMetadata("apiSchemas", listop)
                    except Exception as e:
                        logger.warning(f"[shared_cam] metadata fallback for '{schema_name}' " f"on {path} failed: {e}")

            for nm, (val, tn) in info["all_attrs"].items():
                dst = prim.GetAttribute(nm)
                if not dst.IsValid():
                    dst = prim.CreateAttribute(nm, tn)
                try:
                    dst.Set(val)
                except Exception as e:
                    logger.warning(f"[shared_cam] failed to set {path}.{nm}: {e}")
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
            self._shared_cam_paths[info["name"]] = path

            w, h = info["resolution"]
            rp = rep.create.render_product(path, (w, h))
            rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_ann.attach(rp)
            self._shared_cam_rgb_ann[info["name"]] = rgb_ann
            rel = info["rel_path"]
            if "Fisheye" not in rel and "Top" not in rel:
                depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
                depth_ann.attach(rp)
                self._shared_cam_depth_ann[info["name"]] = depth_ann

            logger.info(
                f"[shared_cam] created {path} resolution={info['resolution']} "
                f"applied_schemas={info.get('applied_schemas', [])}"
            )

        # Disable ALL native cameras across all envs.
        for env_path in env_paths:
            env_prim = stage.GetPrimAtPath(env_path)
            if not env_prim.IsValid():
                continue
            for prim in Usd.PrimRange(env_prim):
                if prim.IsA(UsdGeom.Camera):
                    prim.SetActive(False)

        for i in range(len(env_paths)):
            self._isaacsim_annotators[i] = {}
            self._isaacsim_depth_annotators[i] = {}

        logger.info(f"[shared_cam] initialised {len(infos)} shared cameras under {self._SHARED_CAM_ROOT}")

    def _set_shared_cam_world_pose(self, cam_name, world_m4):
        path = self._shared_cam_paths[cam_name]
        prim = self._stage.GetPrimAtPath(path)
        translate = world_m4.ExtractTranslation()
        quat = world_m4.ExtractRotation().GetQuat()
        real = float(quat.GetReal())
        img = quat.GetImaginary()
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(translate[0], translate[1], translate[2]))
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(real, float(img[0]), float(img[1]), float(img[2])))

    def _teleport_shared_cams_to_env(self, env_idx, env_paths):
        """Write world xforms of the N shared cameras for env_idx."""
        if not hasattr(self, "_shared_cam_infos"):
            return
        xform_cache = UsdGeom.XformCache(0.0)
        native_map = self._native_cam_prims.get(env_idx, {})
        for info in self._shared_cam_infos:
            name = info["name"]
            native_prim = native_map.get(name)
            if native_prim is None or not native_prim.IsValid():
                logger.warning(f"[shared_cam] no native cam prim for env_{env_idx}/{name}; skip teleport")
                continue
            target_world = xform_cache.GetLocalToWorldTransform(native_prim)
            self._set_shared_cam_world_pose(name, target_world)

    def render_once(self, n_subframes: int = None):
        """Force one render with N RTX subframes. Task-thread safe.
        pause_timeline must be False to avoid deadlocking render_step."""
        if n_subframes is None:
            n_subframes = getattr(self, "shared_cam_render_frames", 8)
        n_subframes = max(1, int(n_subframes))

        def _do_render():
            try:
                rep.orchestrator.step(
                    rt_subframes=n_subframes,
                    pause_timeline=False,
                    delta_time=0.0,
                    wait_for_render=True,
                )
            except Exception as e:
                logger.warning(f"[shared_cam] orchestrator.step failed: {e}")

        self.run_on_render_loop(_do_render)

    def prepare_shared_render_cache(self):
        """Clear the per-step shared render cache before a new step's reads.

        Mutates in place so forked views (which hold the same dict reference
        from __dict__.update) see the clear; rebinding with ``= {}`` would
        only update the root view's reference.
        """
        if hasattr(self, "_shared_render_cache") and isinstance(self._shared_render_cache, dict):
            self._shared_render_cache.clear()
        else:
            self._shared_render_cache = {}

    def render_env(self, env_idx, env_paths=None):
        """Teleport+render+cache for env_idx. Returns (images, depths)."""
        if not hasattr(self, "_shared_cam_infos"):
            return {}, {}
        if env_paths is None:
            env_paths = [f"/World/envs/env_{i}" for i in range(len(self._env_articulations))]

        self.run_on_render_loop(self._teleport_shared_cams_to_env, env_idx, env_paths)

        # First render per env uses heavier subframes (shader compile/BVH warmup).
        root = getattr(self, "_root_api_core", None) or self
        warmed = root._warmed_up_envs
        if env_idx not in warmed:
            n_subframes = self.shared_cam_first_render_frames
            warmed.add(env_idx)
            logger.info(f"[shared_cam] env_{env_idx} first render — subframes={n_subframes}")
        else:
            n_subframes = self.shared_cam_render_frames
        self.render_once(n_subframes=n_subframes)

        images = {}
        depths = {}
        for info in self._shared_cam_infos:
            name = info["name"]
            rgb_ann = self._shared_cam_rgb_ann.get(name)
            if rgb_ann is not None:
                data = rgb_ann.get_data()
                if data is not None and data.size > 0:
                    images[name] = data[..., :3]
            depth_ann = self._shared_cam_depth_ann.get(name)
            if depth_ann is not None:
                d = depth_ann.get_data()
                if d is not None and d.size > 0:
                    depths[name] = np.asarray(d).squeeze()
        self._shared_render_cache[env_idx] = (images, depths)
        return images, depths

    def fork_for_env(self, env_idx: int) -> "APICore":
        """Per-env view sharing __dict__ but with env-scoped overrides."""
        view = object.__new__(APICore)
        view.__dict__.update(self.__dict__)
        env_root = f"/World/envs/env_{env_idx}"
        view.env_root = env_root
        view.target_joints_pose = []
        if hasattr(self, "robot_cfg") and self.robot_cfg is not None:
            view.robot_prim_path = env_root + self.robot_cfg.robot_prim_path
        if hasattr(self, "_env_articulations") and env_idx in self._env_articulations:
            view._current_articulation = self._env_articulations[env_idx]
        view._cur_env_idx = env_idx

        # Pin a back-ref so root-only state (e.g. _warmed_up_envs) isn't shadowed.
        view._root_api_core = self

        view.usd_objects = dict(self.usd_objects)
        view.usd_objects["robot"] = self._env_robot_xforms[env_idx]
        return view

    def clear_vectorized_stage(self):
        self.run_on_render_loop(self._clear_vectorized_stage)

    def _clear_vectorized_stage(self):
        # Detach shared-cam annotators and drop /World/shared_cameras + /World/envs.
        if hasattr(self, "_shared_cam_rgb_ann"):
            for ann in self._shared_cam_rgb_ann.values():
                try:
                    ann.detach()
                except Exception:
                    pass
            self._shared_cam_rgb_ann = {}
        if hasattr(self, "_shared_cam_depth_ann"):
            for ann in self._shared_cam_depth_ann.values():
                try:
                    ann.detach()
                except Exception:
                    pass
            self._shared_cam_depth_ann = {}
        if hasattr(self, "_shared_cam_paths"):
            for path in self._shared_cam_paths.values():
                p = self._stage.GetPrimAtPath(path)
                if p.IsValid():
                    delete_prim(path)
            self._shared_cam_paths = {}
        shared_root_prim = self._stage.GetPrimAtPath(self._SHARED_CAM_ROOT)
        if shared_root_prim.IsValid():
            delete_prim(self._SHARED_CAM_ROOT)
        self._shared_cam_infos = []
        self._shared_render_cache = {}
        self._warmed_up_envs = set()
        self._native_cam_prims = {}

        self._isaacsim_annotators = {}
        self._isaacsim_depth_annotators = {}
        self._env_articulations = {}
        self._env_robot_xforms = {}
        self.articulat_objects.clear()
        self.usd_objects.clear()
        self.object_prims = {"object_prims": [], "articulated_object_prims": []}

        world = self.ui_builder.my_world
        if world.is_playing():
            world.stop()
            time.sleep(0.5)

        world.scene.clear()

        envs_prim = self._stage.GetPrimAtPath("/World/envs")
        if envs_prim.IsValid():
            delete_prim("/World/envs")
        collisions_prim = self._stage.GetPrimAtPath("/World/collisions")
        if collisions_prim.IsValid():
            delete_prim("/World/collisions")
        physics_scene = self._stage.GetPrimAtPath("/physicsScene")
        if physics_scene.IsValid():
            delete_prim("/physicsScene")

        update_stage()
        time.sleep(0.5)
        world._physics_sim_view = None
        logger.info("Cleared vectorized stage")

    def _read_shared_images(self, dir):
        """Task-thread shared-cam read; renders on cache miss."""
        cache = getattr(self, "_shared_render_cache", None)
        if cache is None:
            self._shared_render_cache = {}
            cache = self._shared_render_cache
        if self._cur_env_idx not in cache:
            self.render_env(self._cur_env_idx)
        cached = cache.get(self._cur_env_idx)
        if cached is None:
            return {}
        images, _depths = cached
        ret = {}
        for k, v in dir.items():
            if v in images:
                img = images[v]
                ret[k] = img[..., :3] if img.ndim == 3 and img.shape[-1] >= 3 else img
        return ret

    def _read_shared_depths(self, dir):
        cache = getattr(self, "_shared_render_cache", None)
        if cache is None:
            self._shared_render_cache = {}
            cache = self._shared_render_cache
        if self._cur_env_idx not in cache:
            self.render_env(self._cur_env_idx)
        cached = cache.get(self._cur_env_idx)
        if cached is None:
            return {}
        _images, depths = cached
        ret = {}
        for k, v in dir.items():
            if v in depths:
                ret[k] = depths[v]
        return ret

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
        prim_path = self._env_path(prim_path) if prim_path != "robot" else prim_path
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
        prim_path = self._env_path(prim_path)
        cache = create_bbox_cache()
        aabb = compute_aabb(cache, prim_path=prim_path)
        return aabb

    def get_obj_joint(self, prim_path):
        prim_path = self._env_path(prim_path)
        return self.run_on_physics_loop(self._get_obj_joint, prim_path)

    def _get_obj_joint(self, prim_path):
        articulation = self.articulat_objects.get(prim_path) or self.articulat_objects.get(prim_path + "/entity")
        if articulation is None:
            return {}
        return {
            "joint_names": articulation.dof_names,
            "joint_positions": articulation.get_joint_positions(),
            "joint_velocities": articulation.get_joint_velocities(),
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

    def set_joint_positions_batched(self, items):
        """Batched variant: submit multiple (target_pose, joint_indices,
        is_trajectory) groups in a single physics-loop round-trip.

        Saves N-1 physics-tick waits when an env step writes arm + gripper
        + waist targets — instead of three serial run_on_physics_loop
        hand-offs, the positions and indices are concatenated and applied
        in one ArticulationAction.

        Groups are split into trajectory and non-trajectory buckets: in
        trajectory mode the articulation's controller integrates targets,
        so we can safely concatenate; in non-trajectory mode the call is a
        direct set, also safe to concatenate. Mixed trajectory flags fall
        back to sequential execution within one physics tick.
        """
        items = [it for it in items if it is not None and it[0] is not None]
        if not items:
            return
        self.run_on_physics_loop(self._set_joint_positions_batched, items)

    def apply_chassis_action(self, positions, velocities):
        self.run_on_physics_loop(self._apply_chassis_action, positions, velocities)

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
        light_position,
        light_rotation,
        light_texture,
    ):
        self.run_on_render_loop(
            self._set_light,
            light_type,
            light_prim,
            light_temperature,
            light_intensity,
            light_position,
            light_rotation,
            light_texture,
        )

    def apply_light_config(self, light_config):
        self.run_on_render_loop(self._apply_light_config, light_config)

    def reset(self):
        self.run_on_render_loop(self._on_reset)

    def stop(self):
        self.exit = True

    def post_process(self):
        pass

    def stop_all_recording(self):
        """Stop the active recording (if any) and finalize all per-episode bags.

        Safe to call multiple times during shutdown.
        """
        if self.record_process or self.local_recorder.is_recording:
            logger.info("stop_all_recording called")
            self._graceful_stop_recording()

    def shutdown_ros(self):
        """Shutdown all ROS2 nodes and context gracefully."""
        import rclpy as _rclpy

        logger.info("Shutting down ROS2...")
        if self.benchmark_ros_node is not None:
            try:
                self.benchmark_ros_node.destroy_node()
                logger.info("benchmark_ros_node destroyed")
            except Exception as e:
                logger.warning(f"Failed to destroy benchmark_ros_node: {e}")
            self.benchmark_ros_node = None

        if hasattr(self, "server_ros_node") and self.server_ros_node is not None:
            try:
                self.server_ros_node.destroy_node()
                logger.info("server_ros_node destroyed")
            except Exception as e:
                logger.warning(f"Failed to destroy server_ros_node: {e}")
            self.server_ros_node = None

        if hasattr(self, "robot_interface") and self.robot_interface is not None:
            # RobotInterface owns a backing ROS Node only on the teleop flow;
            # destroy() tears it down (no-op on the eval path, which has none).
            try:
                self.robot_interface.destroy()
                logger.info("robot_interface destroyed")
            except Exception as e:
                logger.warning(f"Failed to destroy robot_interface: {e}")
            self.robot_interface = None

        if _rclpy.ok():
            try:
                _rclpy.shutdown()
                logger.info("rclpy.shutdown() called")
            except Exception as e:
                logger.warning(f"rclpy.shutdown() failed: {e}")

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

    def start_local_recording(self, sub_task_name="", episode_idx=0, fps=30, camera_prim_list=None, output_root=None):
        """Direct entry point to begin in-process recording.

        Mirrors start_recording() but routes through LocalRecorder without the
        legacy ROS topic list. ``output_root`` overrides the default
        ``recording_output_path()`` so callers can co-locate recordings with
        their evaluation outputs.
        """
        self.run_on_render_loop(
            self._start_local_recording,
            sub_task_name,
            episode_idx,
            fps,
            camera_prim_list,
            output_root,
        )

    def stop_local_recording(self, episode_idx=None, discard=False):
        self.run_on_render_loop(self._stop_local_recording, episode_idx, discard)

    def concat_recordings(self, final_output_dir=None):
        self.local_recorder.concat_all(final_output_dir=final_output_dir)

    def _start_local_recording(self, sub_task_name, episode_idx, fps, camera_prim_list, output_root=None):
        # Eval recording path: in-process LocalRecorder (mp4), isolated from the
        # teleop rosbag path (_start_recording / record_rosbag).
        self._recording_episode_idx = episode_idx
        self._recording_output_root = output_root or system_utils.recording_output_path()
        if sub_task_name:
            self.sub_task_name = sub_task_name
        cams = (
            camera_prim_list
            if camera_prim_list
            else self.task_config.get("recording_setting", {}).get("camera_list", [])
        )
        self.fps = fps
        self.camera_prim_list = cams
        self.process_camera_info_list()

        if not self.enable_physics:
            self.ui_builder.my_world.stop()
            disable_physics(self._physics_info)
            self._play()

        camera_specs = self._build_camera_specs()
        sub_task = getattr(self, "sub_task_name", "") or ""
        bag_path = self.local_recorder.start(
            output_root=self._recording_output_root,
            episode_idx=episode_idx,
            camera_specs=camera_specs,
            sub_task_name=sub_task,
            fps=fps,
        )
        self.path_to_save_record = bag_path or self.path_to_save_record
        self.recording_started = True

    def _build_camera_specs(self):
        """Construct LocalRecorder camera_specs from robot_interface.parameters.

        Each entry mirrors the data already cached in robot_interface during
        register_camera. Returns [] if no cameras are registered yet.
        """
        params = getattr(self.robot_interface, "parameters", None) or {}
        specs = []
        for cam_id, cam_param in params.items():
            res = cam_param.get("resolution", {})
            width = int(res.get("width", 0))
            height = int(res.get("height", 0))
            if width <= 0 or height <= 0:
                continue
            specs.append(
                {
                    "camera_id": cam_id,
                    "prim_path": cam_param.get("path", ""),
                    "width": width,
                    "height": height,
                    "every_n_frame": cam_param.get("every_n_frame", 1),
                }
            )
        return specs

    def collect_init_physics(self):
        self.run_on_render_loop(self._collect_init_physics)

    def dump_recording_info(self):
        self.run_on_render_loop(self._dump_recording_info)

    def reset_env(self):
        self.run_on_render_loop(self._reset_env)

    def shuffle_scene(self):
        """Randomly adjust x and y positions of rigid body objects in the scene"""
        self.run_on_render_loop(self._shuffle_scene)

    def get_joint_state_dict(self):
        return self.run_on_physics_loop(self._get_joint_state_dict)

    def get_link_world_pose(self, link_name):
        return self.run_on_physics_loop(self._get_link_world_pose, link_name)

    def get_observation_image(self, dir):
        # Vec: read on task thread; nesting under run_on_physics_loop deadlocks.
        if self.is_vec_mode and hasattr(self, "_shared_cam_infos"):
            return self._read_shared_images(dir)
        return self.run_on_physics_loop(self._get_observation_image, dir)

    def get_observation_depth(self, dir):
        if self.is_vec_mode and hasattr(self, "_shared_cam_infos"):
            return self._read_shared_depths(dir)
        return self.run_on_physics_loop(self._get_observation_depth, dir)

    def get_obs_bundle(self, image_dirs=None, depth_dirs=None, link_names=None, want_joint_state=True):
        """Fetch image / depth / joint_state / link poses in one physics-loop
        round-trip. Each component is optional — pass ``None`` (or an empty
        sequence for ``link_names``) to skip it.

        The single-tick batching saves N-1 physics-tick waits per
        env.step() vs calling get_observation_image/depth/joint_state/
        link_world_pose serially. See pi_env.PiEnv.get_observation."""
        return self.run_on_physics_loop(self._get_obs_bundle, image_dirs, depth_dirs, link_names, want_joint_state)

    def count_visible_meshes(self, prim_path: str):
        prim_path = self._env_path(prim_path)
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
        return str(og.Controller.attribute(self._env_path(prim_path)).get())

    def set_prim_visibility(self, prim_path: str, visible: bool):
        """Set the visibility of a prim and all its descendants."""
        prim_path = self._env_path(prim_path)

        def _set_visibility():
            stage = get_current_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                logger.warning(f"Prim path {prim_path} is not valid")
                return

            # Set visibility for the prim and all its descendants
            for p in Usd.PrimRange(prim):
                if p.IsA(UsdGeom.Imageable):
                    imageable = UsdGeom.Imageable(p)
                    visibility_attr = imageable.GetVisibilityAttr()
                    if visibility_attr:
                        visibility_attr.Set("inherited" if visible else "invisible")

        self.run_on_render_loop(_set_visibility)

    def collect_material_info(self):
        return self.run_on_render_loop(self._collect_material_info)

    def change_material(self, mesh_path: str, material_path):
        return self.run_on_render_loop(self._change_material, mesh_path, material_path)

    def update_robot_base(self, pos, quat, with_physics=False):
        self.run_on_render_loop(self._update_robot_base, pos, quat)

    def set_articulation_joint_drive_gains(
        self,
        articulation_prim_path: str,
        joint_gains: Mapping[str, Tuple[float, float]],
    ):
        """Set PhysX joint drive stiffness and damping for named articulation DOFs.

        Args:
            articulation_prim_path: Root prim path of the articulation (e.g. robot Xform).
            joint_gains: Map each DOF/joint name to ``(stiffness, damping)`` as used by
                ``UsdPhysics.DriveAPI`` on the joint prim (typically revolute: ``angular`` drive).

        Notes:
            Joint prims are discovered under ``articulation_prim_path`` by matching
            ``prim.GetName()`` to the given DOF name. If your asset uses different naming,
            extend the lookup logic.
        """
        return self.run_on_render_loop(self._set_articulation_joint_drive_gains, articulation_prim_path, joint_gains)

    def set_robot_joint_drive_gains(self, joint_gains: Mapping[str, Tuple[float, float]]):
        """Same as :meth:`set_articulation_joint_drive_gains` for the current robot articulation."""
        art = self._get_articulation()
        path = getattr(self, "robot_prim_path", None)
        if art is not None:
            path = getattr(art, "prim_path", None) or path
        if not path:
            raise RuntimeError("robot_prim_path is not set; call after init_robot_cfg")
        return self.set_articulation_joint_drive_gains(str(path), joint_gains)

    def set_robot_camera_local_pose(
        self,
        camera_prim_path: str,
        position: Sequence[float],
        orientation: Optional[Sequence[float]] = None,
    ):
        """Set a robot camera prim's transform **in its parent's local space** (relative pose).

        Args:
            camera_prim_path: Full USD path to the camera Xform, e.g.
                ``/genie/gripper_l_base_link/Left_Camera``.
            position: Local translation ``[x, y, z]`` relative to the parent prim.
            orientation: Optional local orientation as quaternion ``[w, x, y, z]``.
                If omitted, only translation is updated.

        Notes:
            Only **updates existing** ``UsdGeom`` translate / orient xform ops on the prim.
            Does not create or rebuild the xform stack; if the expected ops are missing, logs
            a warning and skips.
        """
        return self.run_on_render_loop(self._set_robot_camera_local_pose, camera_prim_path, position, orientation)

    def get_prim_local_pose(
        self, prim_path: str
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """Get the local pose of a prim (position and orientation relative to its parent).

        Args:
            prim_path: Full USD path to the prim, e.g. ``/World/Objects/cup``.

        Returns:
            A tuple of ``(position, quaternion)`` where:
            - position: ``(x, y, z)`` local translation
            - quaternion: ``(w, x, y, z)`` local orientation

        Notes:
            Returns ``((0, 0, 0), (0, 0, 0, 0))`` if the prim is invalid or has no xform.
        """
        return self.run_on_render_loop(self._get_prim_local_pose, prim_path)

    def set_prim_local_pose(
        self,
        prim_path: str,
        position: Sequence[float],
        orientation: Optional[Sequence[float]] = None,
    ):
        """Set the local pose of a prim (position and orientation relative to its parent).

        Args:
            prim_path: Full USD path to the prim, e.g. ``/World/Objects/cup``.
            position: Local translation ``[x, y, z]`` relative to the parent prim.
            orientation: Optional local orientation as quaternion ``[w, x, y, z]``.
                If omitted, only translation is updated.

        Notes:
            Only **updates existing** ``UsdGeom`` translate / orient xform ops on the prim.
            Does not create or rebuild the xform stack; if the expected ops are missing, logs
            a warning and skips.
        """
        return self.run_on_render_loop(self._set_prim_local_pose, prim_path, position, orientation)

    ######################===================== New API END ======================================

    ######################===================== Private Methods BEGIN ===================================
    def _change_material(self, mesh_path: str, material_path):
        stage = get_current_stage()
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        material = UsdShade.Material.Get(stage, Sdf.Path(material_path))
        UsdShade.MaterialBindingAPI(mesh_prim).UnbindAllBindings()
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

    def _collect_material_info(self):
        stage = get_current_stage()

        if not stage:
            return {}

        env_root = getattr(self, "env_root", "")

        all_results = {}
        processed_meshes = set()

        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Xform):
                for mesh_prim in Usd.PrimRange(prim):
                    if mesh_prim.IsA(UsdGeom.Mesh):
                        mesh_path = str(mesh_prim.GetPath())

                        if mesh_path in processed_meshes:
                            continue
                        # Forked view: only this env's meshes are relevant.
                        if env_root and not mesh_path.startswith(env_root):
                            continue
                        if not any(keyword in mesh_path for keyword in ["objects", "Objects"]):
                            continue
                        processed_meshes.add(mesh_path)
                        material_paths = []
                        binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
                        material, binding_rel = binding_api.ComputeBoundMaterial()

                        if material:
                            material_path = str(material.GetPath())
                            material_prim = stage.GetPrimAtPath(material_path)
                            if material_prim.IsValid():
                                parent_prim = material_prim.GetParent()
                                if parent_prim.IsValid():
                                    for sub_prim in Usd.PrimRange(parent_prim):
                                        if sub_prim.IsA(UsdShade.Material):
                                            sub_material_path = str(sub_prim.GetPath())
                                            if sub_material_path not in material_paths:
                                                material_paths.append(sub_material_path)

                        if not material_paths:
                            direct_binding_rel = binding_api.GetDirectBindingRel()
                            if direct_binding_rel:
                                binding_targets = direct_binding_rel.GetTargets()
                                for target_path in binding_targets:
                                    target_prim = stage.GetPrimAtPath(target_path)
                                    if target_prim.IsValid():
                                        for sub_prim in Usd.PrimRange(target_prim):
                                            if sub_prim.IsA(UsdShade.Material):
                                                sub_material_path = str(sub_prim.GetPath())
                                                if sub_material_path not in material_paths:
                                                    material_paths.append(sub_material_path)

                        if len(material_paths) > 1:
                            all_results[mesh_path] = sorted(set(material_paths))

        return all_results

    def _collect_init_physics(self):
        robot_articulation = self._get_articulation()
        self._physics_info = {}
        collect_physics(self._physics_info, getattr(self, "robot_prim_path", None))
        if robot_articulation:
            for i in range(20):
                update_stage()
                time.sleep(0.01)
            self.init_frame_info = store_init_physics(robot_articulation, self._physics_info)

    def _get_joint_state_dict(self):
        # Vec view: read from per-env articulation.
        if getattr(self, "_current_articulation", None) is not None:
            art = self._current_articulation
            positions = art.get_joint_positions()
            if positions is None:
                return {}
            positions = positions.tolist()
            return {art.dof_names[i]: positions[i] for i in range(len(art.dof_names))}
        return self.robot_interface.get_joint_state_dict()

    def _get_link_world_pose(self, link_name):
        prim_path = f"{self.robot_prim_path}/{link_name}"
        return get_world_pose(prim_path)

    def _get_observation_image(self, dir):
        # Vec: read-only cache lookup; cache populated by vec_env beforehand.
        if self.is_vec_mode and hasattr(self, "_shared_cam_infos"):
            cache = getattr(self, "_shared_render_cache", None) or {}
            cached = cache.get(self._cur_env_idx)
            if cached is None:
                return {}
            images, _depths = cached
            ret = {}
            for k, v in dir.items():
                if v in images:
                    img = images[v]
                    ret[k] = img[..., :3] if img.ndim == 3 and img.shape[-1] >= 3 else img
            return ret
        return self.robot_interface.get_observation_image(dir)

    def _get_observation_depth(self, dir):
        if self.is_vec_mode and hasattr(self, "_shared_cam_infos"):
            cache = getattr(self, "_shared_render_cache", None) or {}
            cached = cache.get(self._cur_env_idx)
            if cached is None:
                return {}
            _images, depths = cached
            ret = {}
            for k, v in dir.items():
                if v in depths:
                    ret[k] = depths[v]
            return ret
        return self.robot_interface.get_observation_depth(dir)

    def _get_obs_bundle(self, image_dirs, depth_dirs, link_names, want_joint_state):
        out = {}
        if image_dirs is not None:
            out["images"] = self.robot_interface.get_observation_image(image_dirs)
        if depth_dirs is not None:
            out["depth"] = self.robot_interface.get_observation_depth(depth_dirs)
        if want_joint_state:
            out["joint_state"] = self.robot_interface.get_joint_state_dict()
        if link_names:
            poses = {}
            for name in link_names:
                try:
                    poses[name] = get_world_pose(f"{self.robot_prim_path}/{name}")
                except Exception:
                    pass
            out["link_poses"] = poses
        return out

    def _play(self):
        self.ui_builder.my_world.play()
        self._init_robot(self.robot_cfg, self.enable_curobo)

        self.frame_status = []

        if self._startup_render_held:
            self._startup_render_held = False
            self.release_render()

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
        self.robot_joint_indices = {name: idx for idx, name in enumerate(self.robot_interface.get_joint_state_names())}
        self.robot_interface.register_robot_tf(self._stage, self.robot_prim_path)
        if robot.perception:
            self.robot_interface.register_perception(self._stage, self.robot_prim_path)
        # cams
        for camera in self.robot_cfg.cameras:
            self.robot_interface.register_camera(
                camera,
                self.robot_cfg.cameras[camera],
                (
                    int(round((1 / self.ui_builder.my_world.get_physics_dt()) / self.robot_cfg.cameras[camera][2]))
                    if 2 < len(self.robot_cfg.cameras[camera])
                    else 10
                ),
            )
        if self.enable_pub_depth_camera:
            self.pub_depth_camera()
        if self.enable_ros and not self.ros_node_initialized:
            self.server_ros_node = ServerNode(robot_name=self.robot_name)
            # joint_states
            logger.info(f"sensor_ros.publish_joint {self.robot_prim_path} {self.robot_name}")
            self.sensor_base.publish_joint(robot_prim=self.robot_prim_path)
            self.ros_node_initialized = True

    def _reload_scenes(self, sub_usd_path):
        ws_prim = get_prim_at_path("/Workspace")
        workspace_replaced = False
        if sub_usd_path != "":
            if ws_prim.IsValid():
                delete_prim(ws_prim.GetPath())
                workspace_replaced = True
                logger.info(f"Deleted old Workspace prim")
                self.articulat_objects.clear()
                # Drop only the old Workspace objects; the robot prim under
                # /World persists across instances, so keep its handle.
                robot_obj = self.usd_objects.get("robot")
                self.usd_objects.clear()
                if robot_obj is not None:
                    self.usd_objects["robot"] = robot_obj

            add_reference_to_stage(
                sub_usd_path,
                "/Workspace",
            )
            logger.info(f"Added new Workspace from: {sub_usd_path}")
        if workspace_replaced:
            if hasattr(self.ui_builder, "my_world") and self.ui_builder.my_world:

                world = self.ui_builder.my_world

                was_playing = world.is_playing()
                if was_playing:
                    world.stop()
                    logger.info("Stopped physics simulation to reset views")
                    time.sleep(0.1)

                world.play()
                logger.info("Restarted physics simulation")

                update_stage()

                logger.info("Physics views initialized for new Workspace articulations")

            if hasattr(self.ui_builder, "initialize_articulation"):
                self.ui_builder.initialize_articulation()
                logger.info("Re-initialized robot articulation from scene")
            elif hasattr(self.ui_builder, "articulation") and self.ui_builder.articulation:
                self.ui_builder.articulation.initialize()
                logger.info("Re-initialized robot articulation directly")

            robot_articulation = self._get_articulation()
            if robot_articulation:
                try:
                    self.robot_interface.register_joint_state(robot_articulation)
                    logger.info("Re-registered robot joint state after Workspace replacement")
                except Exception as e:
                    logger.error(f"Failed to register joint state: {e}")
            else:
                logger.warning("Failed to get robot articulation after Workspace replacement")

    def _initialize_all_scene_articulations(self):
        stage = get_current_stage()
        if not stage:
            logger.warning("USD stage not available")
            return

        scene = self.ui_builder.my_world.scene
        robot_prim_path = getattr(self.ui_builder, "robot_prim_path", None)

        articulation_paths = []
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if robot_prim_path and prim_path == robot_prim_path:
                continue

            prim_type = get_prim_object_type(prim_path)
            if prim_type == "articulation":
                articulation_paths.append(prim_path)

        logger.info(f"Found {len(articulation_paths)} articulation(s) in scene")

        for prim_path in articulation_paths:
            print(f"initializing articulation {prim_path}")
            try:
                if scene._scene_registry.name_exists(prim_path):
                    articulation = scene.get_object(prim_path)
                    logger.info(f"Found existing articulation in scene for {prim_path}")
                else:
                    articulation = SingleArticulation(prim_path=prim_path)
                    logger.info(f"Created new articulation for {prim_path}")

                articulation.initialize()

                self.articulat_objects[prim_path] = articulation
                logger.info(f"Registered articulation {prim_path} to api_core")

            except Exception as e:
                logger.error(f"Failed to initialize articulation {prim_path}: {e}")
                continue

    def pub_depth_camera(self):
        sensors = []
        for camera in self.robot_cfg.cameras:
            sensor = {}
            sensor["path"] = camera
            sensor["frequency"] = 30.0
            sensor["resolution"] = {}
            sensor["resolution"]["width"] = self.robot_cfg.cameras[camera][0]
            sensor["resolution"]["height"] = self.robot_cfg.cameras[camera][1]
            sensors.append(sensor)
        self.sensor_base.init_depth_camera(sensors)

    def _init_robot_cfg(
        self,
        robot_cfg,
        scene_usd,
        init_position=[0, 0, 0],
        init_rotation=[1, 0, 0, 0],
        sub_usd_path="",
    ):
        scene_usd = str(np.random.choice(scene_usd)) if type(scene_usd) == list else scene_usd
        scene_usd_path = str(system_utils.assets_path()) + "/" + str(scene_usd)
        scene_key = (robot_cfg, scene_usd_path)

        # Robot + /World background are loaded once per scene and reused across
        # instances; only /Workspace is swapped. Re-adding the full scene every
        # instance stacks references on the stage and leaks RAM/VRAM until the
        # run hangs. A different scene (e.g. a list-valued scene_usd re-rolled
        # for background generalization) falls through to a full reload.
        if self._loaded_scene_key == scene_key:
            self._reload_scenes(sub_usd_path)
            self._initialize_all_scene_articulations()
            return

        self._reload_scenes(sub_usd_path)
        self.robot_cfg = RobotCfg(str(system_utils.app_root_path()) + "/robot_cfg/" + robot_cfg)
        robot_usd_path = str(system_utils.assets_path()) + "/" + self.robot_cfg.robot_usd
        # A full reload re-adds /World and the robot prim. add_reference_to_stage
        # appends a reference rather than replacing, so delete any previously loaded
        # copies first — otherwise the stage stacks references on every scene change
        # (e.g. a re-rolled list-valued scene_usd) and leaks RAM/VRAM until the run hangs.
        for stale_path in ("/World", self.robot_cfg.robot_prim_path):
            if self._stage.GetPrimAtPath(stale_path).IsValid():
                delete_prim(stale_path)
        add_reference_to_stage(robot_usd_path, self.robot_cfg.robot_prim_path)
        add_reference_to_stage(scene_usd_path, "/World")
        self._filter_objects(scene_usd_path)
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
            viewport.set_active_camera("/genie/head_link3/head_front_Camera")
        time.sleep(1)
        self._play()
        time.sleep(1)
        self._initialize_all_scene_articulations()
        self._loaded_scene_key = scene_key

    def _set_joint_positions(self, target_pose, target_joint_indices, is_trajectory):
        # Vec view: write to per-env articulation directly.
        if getattr(self, "_current_articulation", None) is not None:
            art = self._current_articulation
            positions = np.asarray(target_pose, dtype=np.float32)
            indices = np.asarray(target_joint_indices, dtype=np.int32)
            if is_trajectory:
                from isaacsim.core.utils.types import ArticulationAction

                art.apply_action(ArticulationAction(joint_positions=positions, joint_indices=indices))
            else:
                art.set_joint_positions(positions, joint_indices=indices)
            return
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

    def _set_joint_positions_batched(self, items):
        """Physics-loop handler: concatenate groups sharing the same
        is_trajectory flag into a single _joint_moveto call.

        Each item is ``(target_pose, joint_indices, is_trajectory)``. Items
        with matching trajectory flag are merged; groups with different
        flags are flushed separately but still within one physics tick, so
        the call site only pays one run_on_physics_loop round-trip.
        """
        merged_poses = {True: [], False: []}
        merged_indices = {True: [], False: []}
        for target_pose, joint_indices, is_trajectory in items:
            merged_poses[is_trajectory].extend(float(v) for v in target_pose)
            merged_indices[is_trajectory].extend(int(i) for i in joint_indices)

        # Vec view: write to per-env articulation directly.
        if getattr(self, "_current_articulation", None) is not None:
            from isaacsim.core.utils.types import ArticulationAction

            art = self._current_articulation
            for is_trajectory, pose_chunk in merged_poses.items():
                if not pose_chunk:
                    continue
                idx_chunk = merged_indices[is_trajectory]
                positions = np.asarray(pose_chunk, dtype=np.float32)
                indices = np.asarray(idx_chunk, dtype=np.int32)
                if is_trajectory:
                    art.apply_action(ArticulationAction(joint_positions=positions, joint_indices=indices))
                else:
                    art.set_joint_positions(positions, joint_indices=indices)
            return

        for is_trajectory, pose_chunk in merged_poses.items():
            if not pose_chunk:
                continue
            idx_chunk = merged_indices[is_trajectory]
            self._joint_moveto(
                pose_chunk,
                target_joint_indices=idx_chunk,
                is_trajectory=is_trajectory,
            )
            # target_joints_pose is only consulted by the single-group path;
            # keep it in sync so a subsequent legacy _set_joint_positions()
            # call doesn't see stale state.
            if not is_trajectory:
                self.target_joints_pose = []
            elif self.ui_builder.reached:
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
                        prim = self._stage.GetPrimAtPath(_prim)
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

    def _apply_light_config(self, light_config):
        stage = self._stage
        if not stage:
            logger.warning("Stage is not available, cannot apply light config")
            return

        temperature = light_config.get("temperature")
        intensity = light_config.get("intensity")
        if temperature is None or intensity is None:
            logger.warning("No light generalization")
            return

        light_prims = []
        for prim in stage.Traverse():
            prim_type = prim.GetTypeName()
            if prim_type in ["DomeLight", "SphereLight", "DiskLight", "RectLight", "DistantLight", "CylinderLight"]:
                light_prims.append(prim)

        for light_prim in light_prims:
            try:
                if light_prim.GetTypeName() == "DomeLight":
                    light = UsdLux.DomeLight(light_prim)
                elif light_prim.GetTypeName() == "SphereLight":
                    light = UsdLux.SphereLight(light_prim)
                elif light_prim.GetTypeName() == "DiskLight":
                    light = UsdLux.DiskLight(light_prim)
                elif light_prim.GetTypeName() == "RectLight":
                    light = UsdLux.RectLight(light_prim)
                elif light_prim.GetTypeName() == "DistantLight":
                    light = UsdLux.DistantLight(light_prim)
                elif light_prim.GetTypeName() == "CylinderLight":
                    light = UsdLux.CylinderLight(light_prim)
                else:
                    continue

                color_temp_attr = light.GetColorTemperatureAttr()
                if color_temp_attr.IsValid():
                    color_temp_attr.Set(temperature)
                else:
                    light.CreateColorTemperatureAttr().Set(temperature)

                enable_color_temp_attr = light.GetEnableColorTemperatureAttr()
                if enable_color_temp_attr.IsValid():
                    enable_color_temp_attr.Set(True)
                else:
                    light.CreateEnableColorTemperatureAttr().Set(True)

                intensity_attr = light.GetIntensityAttr()
                if intensity_attr.IsValid():
                    intensity_attr.Set(intensity)
                else:
                    light.CreateIntensityAttr().Set(intensity)

                logger.info(
                    f"Applied light config to {light_prim.GetPath()}: temperature={temperature}, intensity={intensity}"
                )

            except Exception as e:
                logger.error(f"Failed to apply light config to {light_prim.GetPath()}: {e}")

    def _filter_objects(self, scene_usd_path: str, world_root: str = ""):
        stage = self._stage
        if not stage:
            logger.warning("Stage is not available, cannot filter objects")
            return

        def _resolve_path(raw_path: str) -> str:
            # Rewrite /World/... under env_root for vec mode.
            if not world_root:
                return raw_path
            if raw_path.startswith("/World/"):
                return world_root + raw_path[len("/World") :]
            return world_root + raw_path

        json_path = os.path.splitext(scene_usd_path)[0] + ".json"

        if not os.path.exists(json_path):
            logger.info(f"JSON configuration file not found: {json_path}, skipping object filtering")
            return

        try:
            # Load JSON configuration
            config_data = system_utils.load_json(json_path)

            # Get objects dictionary
            if "objects" not in config_data:
                logger.warning(f"No 'objects' field found in {json_path}, skipping object filtering")
                return

            objects = config_data["objects"]
            if not isinstance(objects, dict):
                logger.warning(f"'objects' field is not a dictionary in {json_path}, skipping object filtering")
                return

            # Process each object group
            total_kept = 0
            total_deactivated = 0
            all_prim_paths = []

            for group_name, group_config in objects.items():
                if not isinstance(group_config, dict):
                    logger.warning(f"Group '{group_name}' is not a dictionary, skipping")
                    continue

                # Get prim_path list for this group
                if "prim_path" not in group_config:
                    logger.warning(f"No 'prim_path' field in group '{group_name}', skipping")
                    continue

                prim_paths = group_config["prim_path"]
                if not isinstance(prim_paths, list):
                    logger.warning(f"'prim_path' in group '{group_name}' is not a list, skipping")
                    continue

                if not prim_paths:
                    logger.warning(f"Empty prim_path list in group '{group_name}', skipping")
                    continue

                # Get Reserve_num for this group
                reserve_num = group_config.get("Reserve_num", len(prim_paths))
                if not isinstance(reserve_num, int) or reserve_num < 0:
                    logger.warning(f"Invalid Reserve_num ({reserve_num}) in group '{group_name}', using all objects")
                    reserve_num = len(prim_paths)

                # Limit reserve_num to the number of available prim_paths
                reserve_num = min(reserve_num, len(prim_paths))

                # First, activate all prim_paths in the JSON list to ensure they are active
                group_activated = 0
                for prim_path in prim_paths:
                    full_path = _resolve_path(prim_path)
                    prim = stage.GetPrimAtPath(full_path)
                    if prim.IsValid():
                        try:
                            prim.SetActive(True)
                            group_activated += 1
                        except Exception as e:
                            logger.error(f"Failed to activate prim {full_path}: {e}")
                    else:
                        logger.warning(f"Prim path not found in stage: {full_path}")

                # Randomly select prim_paths to keep for this group
                if reserve_num < len(prim_paths):
                    selected_indices = np.random.choice(len(prim_paths), size=reserve_num, replace=False)
                    prim_paths_to_keep = [prim_paths[i] for i in selected_indices]
                else:
                    prim_paths_to_keep = prim_paths

                # Deactivate prims that are not in the keep list for this group
                group_deactivated = 0
                for prim_path in prim_paths:
                    if prim_path not in prim_paths_to_keep:
                        full_path = _resolve_path(prim_path)
                        prim = stage.GetPrimAtPath(full_path)
                        if prim.IsValid():
                            try:
                                prim.SetActive(False)
                                group_deactivated += 1
                                logger.info(f"Deactivated object: {full_path} (group: {group_name})")
                            except Exception as e:
                                logger.error(f"Failed to deactivate prim {full_path}: {e}")
                        else:
                            logger.warning(f"Prim path not found in stage: {full_path}")

                total_kept += len(prim_paths_to_keep)
                total_deactivated += group_deactivated
                all_prim_paths.extend(prim_paths)

                logger.info(
                    f"Group '{group_name}': activated {group_activated} objects, "
                    f"kept {len(prim_paths_to_keep)}/{len(prim_paths)} objects, "
                    f"deactivated {group_deactivated} objects"
                )

            logger.info(
                f"Object filtering completed: kept {total_kept}/{len(all_prim_paths)} objects across all groups, "
                f"deactivated {total_deactivated} objects"
            )

        except Exception as e:
            logger.error(f"Error filtering objects from {json_path}: {e}")

    def _set_light(
        self,
        light_type,
        light_prim,
        light_temperature,
        light_intensity,
        light_position,
        light_rotation,
        light_texture,
    ):

        light = Light(
            light_type=light_type,
            prim_path=light_prim,
            stage=self._stage,
            intensity=light_intensity,
            color=light_temperature,
            position=light_position,
            orientation=light_rotation,
            texture_file=light_texture,
        )
        light.initialize()

    def randomize_hdr_textures(self):
        """Public API to randomize HDR textures for all DomeLights in the scene.

        Automatically scans the current HDR file directory and randomly selects
        a different HDR file for each DomeLight.
        """
        self.run_on_render_loop(self._randomize_hdr_textures)

    def _randomize_hdr_textures(self):
        """Randomly replace HDR texture files for all DomeLight prims in the stage.

        For each DomeLight, scans its current HDR directory for all .hdr files
        and randomly selects a different one to replace the current texture.
        """
        stage = self._stage
        if not stage:
            logger.warning("Stage not available, cannot randomize HDR textures")
            return

        # Find all DomeLight prims in the stage
        domelight_count = 0
        for prim in stage.Traverse():
            if prim.GetTypeName() == "DomeLight":
                domelight_count += 1
                dome_light = UsdLux.DomeLight(prim)

                texture_attr = dome_light.GetTextureFileAttr()
                current_path = texture_attr.Get()

                if not current_path:
                    logger.info(f"DomeLight {prim.GetPath()}: no HDR texture set")
                    continue

                absolute_assets_path = assets_path()
                relative_hdr_dir = os.path.dirname(current_path.path)
                background_path = os.path.dirname(self.scene_usd)
                hdr_directory = os.path.normpath(os.path.join(absolute_assets_path, background_path, relative_hdr_dir))

                try:
                    all_hdr_files = sorted([f for f in os.listdir(hdr_directory) if f.endswith(".hdr")])
                except FileNotFoundError:
                    logger.warning(f"DomeLight {prim.GetPath()}: directory not found: {hdr_directory}")
                    continue
                except PermissionError:
                    logger.warning(f"DomeLight {prim.GetPath()}: permission denied: {hdr_directory}")
                    continue

                if len(all_hdr_files) <= 1:
                    logger.info(
                        f"DomeLight {prim.GetPath()}: only {len(all_hdr_files)} HDR file in {hdr_directory}, skipping"
                    )
                    continue

                new_hdr = np.random.choice(all_hdr_files)
                new_path = os.path.join(hdr_directory, new_hdr)

                texture_attr.Set(new_path)
                logger.info(f"DomeLight {prim.GetPath()}: {current_path.path} -> {new_path}")

        if domelight_count == 0:
            logger.info("No DomeLight prims found in the stage")
        else:
            logger.info(f"Randomized HDR textures for {domelight_count} DomeLight(s)")

    def _on_reset(self):
        logger.info("api_core reset.")
        self.ui_builder._on_reset()
        self.usd_objects = {}
        self.target_position = [0, 0, 0]
        self.articulat_objects = {}
        self.frame_status = []

    def _start_recording(self, camera_prim_list, fps, extra_prim_paths, record_topic_list):
        # Teleop / legacy gRPC recording path: record ROS topics to a rosbag,
        # which teleop's data_recording/extract_ros_bag.py turns into HDF5.
        # Fully isolated from the eval LocalRecorder (see _start_local_recording).
        logger.info("Start recording (rosbag)")
        self.fps = fps
        self.process_recording_path()
        self.camera_prim_list = camera_prim_list
        self.process_camera_info_list()
        self.record_topic_list = record_topic_list

        if not self.enable_physics:
            self.ui_builder.my_world.stop()
            disable_physics(self._physics_info)
            self._play()

        self.record_rosbag()
        self.recording_started = True

    def _stop_recording(self, episode_idx=None, discard=False):
        # Teleop rosbag path: SIGINT the ros2 bag record process group(s).
        logger.info("Stop recording (rosbag)")
        self.recording_started = False
        for process in self.record_process:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                process.wait(timeout=10)
                logger.info(f"Recording process {process.pid} stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"Process {process.pid} did not stop in time, killing...")
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
                except (ProcessLookupError, OSError) as e:
                    logger.warning(f"Failed to kill process {process.pid}: {e}")
            except (ProcessLookupError, OSError) as e:
                logger.warning(f"Failed to stop process {process.pid}: {e}")
            time.sleep(0.5)
        self.record_process = []

    def _stop_local_recording(self, episode_idx=None, discard=False):
        # Eval path: stop the in-process LocalRecorder.
        logger.info("Stop recording (local)")
        self.recording_started = False
        self.local_recorder.stop(episode_idx=episode_idx, discard=discard)

    def _graceful_stop_recording(self):
        """Synchronous stop used during shutdown — handles both paths."""
        logger.info("Graceful recording stop (sync)")
        self.recording_started = False
        if self.record_process:
            for process in self.record_process:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGINT)
                except (ProcessLookupError, OSError):
                    pass
            time.sleep(0.5)
            for process in self.record_process:
                try:
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait()
                except (ProcessLookupError, OSError) as e:
                    logger.warning(f"Failed to cleanup process: {e}")
            self.record_process = []
        if self.local_recorder.is_recording:
            self.local_recorder.stop()

    def _dump_recording_info(self):
        if not self.path_to_save_record:
            return
        logger.info("Dump recording info")
        new_info = {
            "bag_file": self.path_to_save_record,
            "output_dir": self.path_to_save_record,
            "robot_init_position": self.robot_init_position,
            "robot_init_rotation": self.robot_init_rotation,
            "camera_info": self.camera_info_list,
            "scene_name": self.scene_name,
            "scene_usd": self.scene_usd,
            "scene_glb": self.scene_glb,
            "object_names": self.object_prims,
            "fps": self.fps,
            "robot_name": self.robot_name,
            "frame_status": self.frame_status,
            "light_config": self.light_config,
            "gripper_names": self.gripper_names,
            "with_img": self.record_images,
            "with_video": self.record_video,
            "arm_base_prim_path": self.arm_base_prim_path,
            "playback_timerange": self.playback_timerange,
        }
        file_path = os.path.join(self.path_to_save_record, "recording_info.json")
        os.makedirs(self.path_to_save_record, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(new_info, f, indent=4)

        logger.info(f"Recording info saved to {file_path}")

    def _reset_env(self):
        robot_articulation = self._get_articulation()
        # init_frame_info may be empty in vec mode pre-collect_init_physics.
        if robot_articulation and getattr(self, "init_frame_info", None):
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

    def _update_robot_base(self, pos, quat):
        # Vec view: write local pose; env_root xform carries clone offset.
        if getattr(self, "env_root", ""):
            self.usd_objects["robot"].set_local_pose(pos, quat)
        else:
            self.usd_objects["robot"].set_world_pose(pos, quat)

    def _apply_chassis_action(self, positions, velocities):
        from isaacsim.core.utils.types import ArticulationAction

        articulation = self._get_articulation()
        action = ArticulationAction(joint_positions=positions, joint_velocities=velocities)
        articulation.apply_action(action)
        self.ui_builder.my_world.step(render=True)

    def _set_robot_camera_local_pose(
        self,
        camera_prim_path: str,
        position: Sequence[float],
        orientation: Optional[Sequence[float]],
    ):
        """Render-thread: only mutate existing translate/orient ops (no create / no rebuild)."""
        stage = get_current_stage()
        if not stage:
            logger.warning("set_robot_camera_local_pose: no stage")
            return
        prim = stage.GetPrimAtPath(camera_prim_path)
        if not prim.IsValid():
            logger.warning(f"set_robot_camera_local_pose: invalid prim {camera_prim_path}")
            return
        xformable = UsdGeom.Xformable(prim)
        pos = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        quat = None
        if orientation is not None and len(orientation) >= 4:
            quat = Gf.Quatd(
                float(orientation[0]),
                float(orientation[1]),
                float(orientation[2]),
                float(orientation[3]),
            )

        translate_op = None
        orient_op = None
        for op in xformable.GetOrderedXformOps():
            ot = op.GetOpType()
            if ot == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif ot == UsdGeom.XformOp.TypeOrient:
                orient_op = op

        if translate_op is None:
            logger.warning(f"set_robot_camera_local_pose: no existing translate xform op on {camera_prim_path}; skip")
            return

        translate_op.Set(pos)
        if quat is not None:
            if orient_op is not None:
                orient_op.Set(quat)
            else:
                logger.warning(
                    f"set_robot_camera_local_pose: orientation given but no orient xform op on {camera_prim_path}; "
                    "translation updated only"
                )
        logger.info(f"set_robot_camera_local_pose: updated {camera_prim_path} local t={pos}")

    def _get_prim_local_pose(
        self, prim_path: str
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """Render-thread: get local pose of a prim."""
        prim_path = self._env_path(prim_path)
        stage = get_current_stage()
        if not stage:
            logger.warning("get_prim_local_pose: no stage")
            return ((0, 0, 0), (0, 0, 0, 0))

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            logger.warning(f"get_prim_local_pose: invalid prim {prim_path}")
            return ((0, 0, 0), (0, 0, 0, 0))

        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            logger.warning(f"get_prim_local_pose: prim {prim_path} is not xformable")
            return ((0, 0, 0), (0, 0, 0, 0))

        translate_op = None
        orient_op = None

        for op in xformable.GetOrderedXformOps():
            ot = op.GetOpType()
            if ot == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif ot == UsdGeom.XformOp.TypeOrient:
                orient_op = op

        position = (0.0, 0.0, 0.0)
        if translate_op is not None:
            translate_value = translate_op.Get()
            if translate_value is not None:
                position = (translate_value[0], translate_value[1], translate_value[2])

        quaternion = (0.0, 0.0, 0.0, 0.0)
        if orient_op is not None:
            orient_value = orient_op.Get()
            if orient_value is not None:
                quaternion = (
                    orient_value.GetReal(),
                    orient_value.GetImaginary()[0],
                    orient_value.GetImaginary()[1],
                    orient_value.GetImaginary()[2],
                )

        return (position, quaternion)

    def _set_prim_local_pose(
        self,
        prim_path: str,
        position: Sequence[float],
        orientation: Optional[Sequence[float]],
    ):
        """Render-thread: set local pose of a prim."""
        prim_path = self._env_path(prim_path)
        stage = get_current_stage()
        if not stage:
            logger.warning("set_prim_local_pose: no stage")
            return

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            logger.warning(f"set_prim_local_pose: invalid prim {prim_path}")
            return

        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            logger.warning(f"set_prim_local_pose: prim {prim_path} is not xformable")
            return

        pos = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        quat = None
        if orientation is not None and len(orientation) >= 4:
            quat = Gf.Quatd(
                float(orientation[0]),
                float(orientation[1]),
                float(orientation[2]),
                float(orientation[3]),
            )

        translate_op = None
        orient_op = None

        for op in xformable.GetOrderedXformOps():
            ot = op.GetOpType()
            if ot == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif ot == UsdGeom.XformOp.TypeOrient:
                orient_op = op

        if translate_op is None:
            logger.warning(f"set_prim_local_pose: no existing translate xform op on {prim_path}; skip")
            return

        translate_op.Set(pos)
        if quat is not None:
            if orient_op is not None:
                orient_op.Set(quat)
            else:
                logger.warning(
                    f"set_prim_local_pose: orientation given but no orient xform op on {prim_path}; "
                    "translation updated only"
                )

        logger.info(f"set_prim_local_pose: updated {prim_path} local t={pos}")

    @staticmethod
    def _find_joint_prim_for_dof_name(articulation_root: Usd.Prim, dof_name: str) -> Usd.Prim:
        """Return the joint prim whose leaf name matches ``dof_name`` (first match)."""
        for prim in Usd.PrimRange(articulation_root):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            if prim.GetName() == dof_name:
                return prim
        return Usd.Prim()

    @staticmethod
    def _apply_drive_stiffness_damping_to_joint_prim(joint_prim: Usd.Prim, stiffness: float, damping: float) -> bool:
        """Apply stiffness/damping on the first available ``UsdPhysics.DriveAPI`` (angular, then linear)."""
        if not joint_prim or not joint_prim.IsValid():
            return False
        for token in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(joint_prim, token)
            if not drive:
                continue
            s_attr = drive.GetStiffnessAttr()
            d_attr = drive.GetDampingAttr()
            if s_attr:
                s_attr.Set(float(stiffness))
            else:
                drive.CreateStiffnessAttr().Set(float(stiffness))
            if d_attr:
                d_attr.Set(float(damping))
            else:
                drive.CreateDampingAttr().Set(float(damping))
            return True
        # Some assets define drive only after Apply()
        for token in ("angular", "linear"):
            try:
                if hasattr(UsdPhysics.DriveAPI, "CanApply") and UsdPhysics.DriveAPI.CanApply(joint_prim, token):
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, token)
                    drive.CreateStiffnessAttr().Set(float(stiffness))
                    drive.CreateDampingAttr().Set(float(damping))
                    return True
            except Exception:
                continue
        return False

    def _set_articulation_joint_drive_gains(
        self, articulation_prim_path: str, joint_gains: Mapping[str, Tuple[float, float]]
    ):
        stage = get_current_stage()
        if not stage:
            logger.warning("set_articulation_joint_drive_gains: no stage")
            return
        root = stage.GetPrimAtPath(articulation_prim_path)
        if not root.IsValid():
            logger.warning(f"set_articulation_joint_drive_gains: invalid articulation path {articulation_prim_path}")
            return
        for dof_name, gains in joint_gains.items():
            if gains is None or len(gains) != 2:
                logger.warning(f"set_articulation_joint_drive_gains: bad gains for {dof_name}: {gains}")
                continue
            stiffness, damping = float(gains[0]), float(gains[1])
            joint_prim = self._find_joint_prim_for_dof_name(root, dof_name)
            if not joint_prim.IsValid():
                logger.warning(
                    f"set_articulation_joint_drive_gains: no joint prim named '{dof_name}' under {articulation_prim_path}"
                )
                continue
            if self._apply_drive_stiffness_damping_to_joint_prim(joint_prim, stiffness, damping):
                logger.info(f"Joint drive gains set: {dof_name} stiffness={stiffness} damping={damping}")
            else:
                logger.warning(
                    f"set_articulation_joint_drive_gains: no DriveAPI on joint {joint_prim.GetPath()} ({dof_name})"
                )

    ######################===================== Private Methods END ===================================
    def get_robot_joint_indices(self):
        return self.robot_joint_indices

    def process_recording_path(self):
        recording_path = os.path.join(system_utils.recording_output_path(), self.sub_task_name)
        folder_index = 1
        while os.path.isdir(os.path.join(recording_path, str(folder_index))):
            folder_index += 1
        recording_path = os.path.join(recording_path, str(folder_index))
        if not os.path.exists(recording_path):
            os.makedirs(recording_path, exist_ok=True)
        self.path_to_save_record = recording_path

    def get_camera_intrinsic_info(self):
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
        for prim in self.camera_prim_list:
            camera_intrinsic_info[prim] = {}
            camera_prim = stage.GetPrimAtPath(prim)
            for info in camera_info:
                value = camera_prim.GetAttribute(info).Get()
                info = info.split(":")[-1]
                camera_intrinsic_info[prim][info] = str(value)
        return camera_intrinsic_info

    def process_camera_info_list(self):
        self.camera_info_list = {}
        camera_intrinsic_info = self.get_camera_intrinsic_info()
        for prim in self.camera_prim_list:
            prim_name = prim.split("/")[-1]
            if "G1" in self.robot_name:
                if "Fisheye_Camera_R" in prim_name:
                    prim_name = "head_right_fisheye"
                elif "Fisheye_Camera" in prim_name:
                    prim_name = "head_left_fisheye"
                elif "Fisheye_Back_R" in prim_name:
                    prim_name = "back_right_fisheye"
                elif "Fisheye_Back" in prim_name:
                    prim_name = "back_left_fisheye"
                elif "head" in prim_name:
                    prim_name = "head"
                elif "right" in prim_name:
                    prim_name = "hand_right"
                elif "left" in prim_name:
                    prim_name = "hand_left"
                elif "top" in prim_name:
                    prim_name = "head_front_fisheye"
            if "G2" in self.robot_name:
                if "Right_Camera" in prim_name:
                    prim_name = "hand_right_color"
                elif "Left_Camera" in prim_name:
                    prim_name = "hand_left_color"
                elif "head_front" in prim_name:
                    prim_name = "head_color"
                elif "head_left" in prim_name:
                    prim_name = "head_stereo_left_color"
                elif "head_right" in prim_name:
                    prim_name = "head_stereo_right_color"
            self.camera_info_list[prim_name] = {
                "intrinsic": camera_intrinsic_info[prim],
                "output": {
                    "rgb": "camera/" + "{frame_num}/" + f"{prim_name}.jpg",
                    "video": f"{prim_name}.mp4",
                },
            }
            if "fisheye" not in prim_name:
                self.camera_info_list[prim_name]["output"]["depth"] = (
                    "camera/" + "{frame_num}/" + f"{prim_name[:-6]}_depth.png"
                )

            if "semantic" not in prim_name:
                self.camera_info_list[prim_name]["output"]["semantic"] = (
                    "camera/" + "{frame_num}/" + f"{prim_name}_semantic.png"
                )

    def record_rosbag(self):
        # Teleop recording path: record ROS topics to a rosbag, which the
        # teleop data_recording/extract_ros_bag.py later turns into HDF5.
        # This is isolated from the eval LocalRecorder (in-process mp4) path.
        if self.enable_ros:
            if os.path.isdir(self.path_to_save_record):
                shutil.rmtree(self.path_to_save_record)
            ros_distro = os.getenv("ROS_DISTRO", "jazzy")
            command_str = f"""
            unset PYTHONPATH
            unset LD_LIBRARY_PATH
            source /opt/ros/{ros_distro}/setup.bash
            ros2 bag record -o {self.path_to_save_record} {' '.join(self.record_topic_list)}
            """
            process = subprocess.Popen(
                command_str,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            logger.info(f"started record (pid={process.pid})")
            self.record_process.append(process)

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
            raise ValueError("Undefined robot")

    def on_ros_tick(self, step_size):
        # Sim-tick: refresh robot_interface caches every step, regardless of
        # whether ROS is enabled. The recorder reads _img_data_cache from here.
        current_time = self.ui_builder.my_world.current_time
        current_step = self.ui_builder.my_world.current_time_step_index
        self.robot_interface.tick(current_time, current_step)

        if self.local_recorder.is_recording:
            self.local_recorder.write_frames(self.robot_interface, current_step)

        if not self.enable_ros:
            return

        if self.ros_node_initialized:
            rclpy.spin_once(self.server_ros_node, timeout_sec=0)
            if self.benchmark_ros_node is not None:
                rclpy.spin_once(self.benchmark_ros_node, timeout_sec=0)

            self.server_ros_node.publish_clock(current_time)
            self._on_play_back = self.server_ros_node.get_playback_state()
            self.teleop_recording = self.server_ros_node.get_teleop_recording()

    def _on_recording(self):
        # Multi-episode recording: autoteleop.sh publishes on /sim/stop_episode
        # after the operator confirms an episode with y/n. On that signal:
        # finalize the current ros2 bag gracefully (SIGINT, so metadata.yaml is
        # written) and re-arm the one-shot latches so the next record-button
        # press starts a fresh episode. Without this, wait_recording locks after
        # the first episode and the whole simulator must be restarted per episode.
        server_node = getattr(self, "server_ros_node", None)
        if server_node is not None and server_node.get_stop_episode():
            server_node.clear_stop_episode()
            if self.recording_started:
                if self.recording_wait_num < 100:
                    # Episode stopped before recording_info.json was written at
                    # frame 100; write it now, otherwise post-processing skips
                    # the episode silently for lack of an info file.
                    self._dump_recording_info()
                self._stop_recording()
                logger.info("Episode finished, ready for next recording")
            self.wait_recording = True
            self.recording_wait_num = 0
            # Drop the cached record-button state and this frame's already-read
            # copy of it, so the next episode waits for a fresh button press
            # instead of auto-starting from stale signals.
            server_node.reset_recording_state()
            self.teleop_recording = False

        if self.teleop_recording and self.wait_recording:
            self.set_record_topics()
            camera_prim_list = self.task_config["recording_setting"]["camera_list"]
            self._start_recording(
                camera_prim_list=camera_prim_list,
                fps=30,
                extra_prim_paths=[],
                record_topic_list=self.record_topic_list,
            )
            self.wait_recording = False
        if not self.wait_recording and self.recording_wait_num < 100:
            self.recording_wait_num += 1
            if self.recording_wait_num == 100:
                self._dump_recording_info()

    def set_record_topics(self):
        # Topic list for the teleop rosbag recording path.
        if "G2" in self.task_config["robot"]["robot_cfg"]:
            self.record_topic_list = [
                "/tf",
                "/tf_static",
                "/joint_states",
                "/genie_sim/camera_rgb",
                "/genie_sim/head_front_camera_rgb",
                "/genie_sim/left_camera_rgb",
                "/genie_sim/right_camera_rgb",
                "/genie_sim/static_info",
                "/genie_sim/head_front_Camera_depth",
                "/genie_sim/Left_Camera_depth",
                "/genie_sim/Right_Camera_depth",
                "/genie_sim/head_left_camera_rgb",
                "/genie_sim/head_right_camera_rgb",
            ]
        elif "G1" in self.task_config["robot"]["robot_cfg"]:
            self.record_topic_list = [
                "/tf",
                "/joint_states",
                "/G1/head_camera_rgb",
                "/G1/left_camera_rgb",
                "/G1/right_camera_rgb",
            ]
        else:
            raise ValueError("Invalid robot cfg")

    def _on_playback(self):
        if self.enable_playback:
            robot_articulation = self._get_articulation()

            if robot_articulation:
                # playback
                if not self._physics_info or self.add_object_flag:
                    collect_physics(self._physics_info, getattr(self, "robot_prim_path", None))
                if self._on_play_back and self._current_mode == "realtime":
                    disable_physics(self._physics_info)
                    self._current_mode = "playback"

                if self._current_mode == "playback" and not self._on_play_back:
                    restore_physics(self._physics_info)
                    disable_physics(self._physics_info)
                    restore_physics(self._physics_info)
                    self._current_mode = "realtime"
                    self.playback_end = self.ui_builder.my_world.current_time
                    self.playback_timerange.append([self.playback_start, self.playback_end])
                    if self.teleop_recording:
                        self._dump_recording_info()

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
        if getattr(self, "_current_articulation", None) is not None:
            return self._current_articulation
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
