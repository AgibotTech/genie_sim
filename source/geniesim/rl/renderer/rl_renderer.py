#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
#
# Lightweight IsaacSim renderer for the GeneSim RL parallel pipeline.
#
# Architecture:
#   - ONE IsaacSim process handles rendering for ALL parallel environments
#   - GridCloner duplicates the scene N times in a grid layout
#   - Each clone subscribes to /env_i/tf_render to update body poses
#   - Camera images are captured per-clone and returned via shared memory
#
# Usage — explicit USD paths (legacy):
#   /isaac-sim/python.sh rl_renderer.py \
#       --scene-usd <foreground.usda> \
#       --robot-usd <robot.usda> \
#       --num-envs 4 \
#       --shm-name geniesim_frames \
#       --headless
#
# Usage — task-name mode (recommended):
#   /isaac-sim/python.sh rl_renderer.py \
#       --task-name place_block_into_box \
#       --robot-type G2 \
#       --num-envs 4 \
#       --shm-name geniesim_frames \
#       --headless
#

import argparse
import json
import math
import os
import sys
import threading
import time
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make main/source importable so resolve_task_config() can access geniesim modules.
_SIM_REPO_ROOT = os.getenv("SIM_REPO_ROOT", "")
if _SIM_REPO_ROOT:
    _src_path = os.path.join(_SIM_REPO_ROOT, "source")
    if _src_path not in sys.path:
        sys.path.insert(0, _src_path)

# IsaacSim must be imported before any other omni packages
import isaacsim  # noqa: F401
from isaacsim import SimulationApp

# ---------------------------------------------------------------------------- #
# Task-name resolution  (pure Python, no IsaacSim deps)
# ---------------------------------------------------------------------------- #

def resolve_task_config(task_name: str, robot_type: str = "G2", instance_id: int = 0) -> dict:
    """
    Resolve scene/robot assets and initial states from a task name.

    Returns a dict with keys:
      foreground_usd       – absolute path to llm_task/<task>/<id>/scene.usda
      robot_usd            – absolute path to assets/robot/...
      robot_prim_path      – USD prim path inside the env for the robot root
      robot_init_position  – [x, y, z] world position of robot base
      robot_init_quaternion – [w, x, y, z] orientation of robot base
      init_joint_positions – dict {arm, head, waist, gripper} or None
    """
    from geniesim.benchmark.config.task_config_mapping import TASK_MAPPING
    from geniesim.benchmark.config.robot_init_states import TASK_INFO_DICT
    from geniesim.utils import system_utils
    from geniesim.utils.infer_pre_process import TaskInfo

    if task_name not in TASK_MAPPING:
        raise ValueError(
            f"[resolve_task_config] Unknown task '{task_name}'. "
            f"Available: {list(TASK_MAPPING.keys())}"
        )
    mapping = TASK_MAPPING[task_name]
    if robot_type not in mapping.get("background", {}):
        raise ValueError(
            f"[resolve_task_config] Robot type '{robot_type}' not supported for "
            f"task '{task_name}'. Supported: {list(mapping['background'].keys())}"
        )

    # 1. background config name → eval_tasks JSON
    bg_name = mapping["background"][robot_type]   # e.g. "table_task_g2"
    bg_json_path = os.path.join(
        system_utils.benchmark_conf_path(), f"eval_tasks/{bg_name}.json"
    )
    with open(bg_json_path) as f:
        bg_cfg = json.load(f)

    robot_cfg_name = bg_cfg["robot"].get("robot_cfg", "G1_120s.json")
    workspace = "workspace_00"
    init_pos  = bg_cfg["robot"]["robot_init_pose"][workspace]["position"]
    init_quat = bg_cfg["robot"]["robot_init_pose"][workspace]["quaternion"]

    # 2. robot_cfg JSON → robot USD path and prim root
    robot_cfg_path = os.path.join(
        system_utils.app_root_path(), f"robot_cfg/{robot_cfg_name}"
    )
    with open(robot_cfg_path) as f:
        robot_cfg = json.load(f)["robot"]
    robot_usd = os.path.join(system_utils.assets_path(), robot_cfg["robot_usd"])
    robot_prim = robot_cfg.get("base_prim_path", "/robot")

    # 3. foreground scene USD (objects only, no background room)
    fg_usd = os.path.abspath(
        os.path.join(
            system_utils.benchmark_conf_path(),
            f"llm_task/{task_name}/{instance_id}/scene.usda",
        )
    )
    if not os.path.exists(fg_usd):
        raise FileNotFoundError(
            f"[resolve_task_config] Foreground USD not found: {fg_usd}"
        )

    # 4. initial joint positions from TASK_INFO_DICT
    robot_key = f"{robot_type}_omnipicker"
    task_states = TASK_INFO_DICT.get(task_name, {}).get(robot_key)
    init_joints = None
    if task_states is not None:
        ti = TaskInfo(task_states, robot_key)
        arm, head, waist, hand, gripper = ti.init_pose()
        init_joints = {"arm": arm, "head": head, "waist": waist, "gripper": gripper}

    print(
        f"[resolve_task_config] task={task_name} robot_type={robot_type} "
        f"instance={instance_id}\n"
        f"  foreground_usd  = {fg_usd}\n"
        f"  robot_usd       = {robot_usd}\n"
        f"  robot_prim      = {robot_prim}\n"
        f"  init_pos        = {init_pos}\n"
        f"  init_quat       = {init_quat}\n"
        f"  init_joints     = {'present' if init_joints else 'none'}"
    )
    return dict(
        foreground_usd=fg_usd,
        robot_usd=robot_usd,
        robot_prim_path=robot_prim,
        robot_init_position=init_pos,
        robot_init_quaternion=init_quat,
        init_joint_positions=init_joints,
    )


# ---------------------------------------------------------------------------- #
# CLI argument parsing  (must happen before SimulationApp)
# ---------------------------------------------------------------------------- #

_parser = argparse.ArgumentParser(description="GeneSim RL IsaacSim renderer")

# ---- task-name mode (auto-resolves scene/robot) ----
_parser.add_argument("--task-name", default="", help="Task name; auto-resolves scene/robot USDs")
_parser.add_argument(
    "--robot-type", default="G2", choices=["G1", "G2"],
    help="Robot family for task config lookup (G1 / G2)"
)
_parser.add_argument("--task-instance-id", type=int, default=0,
                     help="Sub-index under llm_task/<task_name>/<id>/")

# ---- explicit USD mode (legacy, required when --task-name is absent) ----
_parser.add_argument("--scene-usd", default="", help="Foreground scene USD path")
_parser.add_argument("--robot-usd", default="", help="Robot USDA path")
_parser.add_argument("--robot-prim", default="/robot", help="Robot prim path inside env")

# ---- simulation / rendering ----
_parser.add_argument("--num-envs", type=int, default=1)
_parser.add_argument("--clone-spacing", type=float, default=3.0, help="Grid spacing in metres")
_parser.add_argument("--render-hz", type=float, default=30.0)
_parser.add_argument("--cam-width", type=int, default=640)
_parser.add_argument("--cam-height", type=int, default=480)
_parser.add_argument(
    "--main-cam-prim", default="",
    help="Camera prim relative to env root.  "
         "Defaults to /default_viz_camera when --task-name is used."
)
_parser.add_argument("--wrist-cam-prim", default="", help="Wrist camera prim (optional)")
_parser.add_argument("--cameras-json", default="", help="JSON list of camera configs [{name, prim, width, height}, ...]")
_parser.add_argument("--shm-name", default="geniesim_frames", help="Shared memory segment name")
_parser.add_argument("--headless", action="store_true")
_parser.add_argument("--ros-domain-id", type=int, default=0)
_parser.add_argument("--auto-dome-light", action="store_true",
                     help="Auto add a dome light when scene has no lights")
_parser.add_argument("--auto-dome-light-intensity", type=float, default=600.0,
                     help="Intensity for auto dome light")

# ---- default viz camera position / target (used when --task-name is set) ----
_parser.add_argument(
    "--default-cam-pos", nargs=3, type=float,
    default=[-0.2, -1.8, 1.8],
    metavar=("X", "Y", "Z"),
    help="World position of the default viz camera",
)
_parser.add_argument(
    "--default-cam-target", nargs=3, type=float,
    default=[-0.4, 0.0, 0.9],
    metavar=("X", "Y", "Z"),
    help="Look-at target for the default viz camera",
)

_args = _parser.parse_args()

# ---- Resolve task config if --task-name provided ----
_task_resolved: Optional[dict] = None
if _args.task_name:
    _task_resolved = resolve_task_config(
        _args.task_name, _args.robot_type, _args.task_instance_id
    )
    _args.scene_usd   = _task_resolved["foreground_usd"]
    _args.robot_usd   = _task_resolved["robot_usd"]
    _args.robot_prim  = _task_resolved["robot_prim_path"]
    if not _args.main_cam_prim:
        _args.main_cam_prim = "/default_viz_camera"
else:
    if not _args.scene_usd:
        _parser.error("Either --task-name or --scene-usd must be provided.")
    if not _args.main_cam_prim:
        _args.main_cam_prim = "/camera_main"

# Disable multi-GPU to avoid USD-RT Hydra CUDA pointer crash (cuPointerGetAttributes)
# in libusdrt.hydra.fabric_scene_delegate.plugin.so when world.reset() is called
# with multiple NVIDIA GPUs in the system.
simulation_app = SimulationApp({"headless": _args.headless, "multi_gpu": False})

# ---- IsaacSim imports (must be after SimulationApp) ----
import omni.replicator.core as rep
from omni.isaac.cloner import GridCloner
from omni.isaac.core import World
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import (
    add_reference_to_stage,
    get_current_stage,
)
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from tf2_msgs.msg import TFMessage

QOS_BE = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

# ---------------------------------------------------------------------------- #
# Shared memory layout
# ---------------------------------------------------------------------------- #
# Constants and helpers are defined in shm_layout.py (no IsaacSim dependency)
# so that GenieSimVectorEnv can import them without pulling in omni/isaacsim.
from geniesim.rl.renderer.shm_layout import NUM_CAMS as _NUM_CAMS
from geniesim.rl.renderer.shm_layout import SHM_HEADER_BYTES as _SHM_HEADER_BYTES
from geniesim.rl.renderer.shm_layout import shm_total_bytes as _shm_total_bytes


# ---------------------------------------------------------------------------- #
# ROS subscriber per env
# ---------------------------------------------------------------------------- #

class EnvTFSubscriber(Node):
    """One ROS node that subscribes to /env_i/tf_render for a specific clone."""

    def __init__(self, env_id: int, stage, env_root_prim_path: str, body_name_map: Dict[str, str]):
        super().__init__(f"geniesim_renderer_env{env_id}")
        self.env_id = env_id
        self.stage = stage
        self.env_root = env_root_prim_path  # e.g. "/World/envs/env_0"
        self.body_name_map = body_name_map  # MuJoCo body name → USD prim name inside env
        self._tf_cache = None
        self._dirty = False          # set True by _on_tf; cleared by apply_tf
        self._robot_root_rel = _args.robot_prim.lstrip("/") if getattr(_args, "robot_prim", "") else "robot"
        # Per-body attribute handle cache: body_name -> (translate_attr, orient_attr, is_quatf)
        # or None if the prim doesn't exist.  Built lazily on first apply_tf call.
        self._attr_cache: Dict[str, Optional[Tuple]] = {}
        # Ordered list parallel to TF message transforms, built after first full cache pass.
        # Eliminates per-body dict lookups in the steady-state hot path.
        self._ordered_attrs: Optional[List] = None

        ns = f"env_{env_id}"
        self.create_subscription(
            TFMessage,
            f"/{ns}/tf_render",
            self._on_tf,
            QOS_BE,
        )

    @staticmethod
    def _normalize_frame_id(frame_id: str) -> str:
        """
        Convert ROS TF child_frame_id to a prim-relative path under env_root.

        Expected IsaacSim clone layout under each env root:
          /World/envs/env_i/objects/...
          /World/envs/env_i/robot/...

        But MuJoCo/ROS may publish absolute USD-ish paths like:
          /World/objects/foo
          World/objects/foo
        """
        s = (frame_id or "").strip()
        s = s.lstrip("/")  # "/World/objects/x" -> "World/objects/x"
        if s.startswith("World/"):
            s = s[len("World/") :]  # "World/objects/x" -> "objects/x"
        # Foreground USD uses "Objects" (capital O) as the container prim, but
        # MuJoCo/ROS publishes lower-case "objects". Normalize to USD naming.
        if s.startswith("objects/"):
            s = "Objects/" + s[len("objects/") :]
        return s

    def _on_tf(self, msg: TFMessage):
        self._tf_cache = msg
        self._dirty = True

    def apply_tf(self):
        """Write latest tf_render into the corresponding clone's USD prims."""
        if not self._dirty or self._tf_cache is None:
            return
        self._dirty = False
        transforms = self._tf_cache.transforms

        if self._ordered_attrs is None:
            # First pass: build attr cache for every body in the TF message.
            for tf in transforms:
                body_name = tf.child_frame_id
                if body_name in self._attr_cache:
                    continue
                if body_name in ("base_link", "chassis_site"):
                    mapped = self._robot_root_rel
                else:
                    mapped = self.body_name_map.get(body_name, body_name)
                prim_name = self._normalize_frame_id(mapped)
                prim = self.stage.GetPrimAtPath(f"{self.env_root}/{prim_name}")
                if not prim.IsValid():
                    self._attr_cache[body_name] = None
                    continue
                t_attr = prim.GetAttribute("xformOp:translate")
                o_attr = prim.GetAttribute("xformOp:orient")
                is_quatf = o_attr.IsValid() and str(o_attr.GetTypeName()) == "quatf"
                self._attr_cache[body_name] = (
                    t_attr if t_attr.IsValid() else None,
                    o_attr if o_attr.IsValid() else None,
                    is_quatf,
                )
            # Build ordered list parallel to TF message: eliminates dict lookups
            # in the steady-state hot path (every frame after the first).
            self._ordered_attrs = [
                self._attr_cache.get(tf.child_frame_id) for tf in transforms
            ]

        # Steady-state hot path: zip TF data with pre-built attr list.
        # No dict lookups, no string operations, only attr.Set() calls.
        for tf, entry in zip(transforms, self._ordered_attrs):
            if entry is None:
                continue
            t_attr, o_attr, is_quatf = entry
            t = tf.transform.translation
            r = tf.transform.rotation
            if t_attr is not None:
                t_attr.Set(Gf.Vec3d(t.x, t.y, t.z))
            if o_attr is not None:
                if is_quatf:
                    o_attr.Set(Gf.Quatf(r.w, r.x, r.y, r.z))
                else:
                    o_attr.Set(Gf.Quatd(r.w, r.x, r.y, r.z))


# ---------------------------------------------------------------------------- #
# Main renderer class
# ---------------------------------------------------------------------------- #

class RLRenderer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.num_envs = args.num_envs
        self.stage = None
        self.world = None
        self.env_subscribers: List[EnvTFSubscriber] = []
        self.cam_annotators_main: List = []  # rep annotator per env (main cam)
        self.cam_annotators_wrist: List = []  # rep annotator per env (wrist cam)
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.shm_array: Optional[np.ndarray] = None

        # rclpy reads ROS_DOMAIN_ID from environment at init time.
        # Keep CLI (--ros-domain-id) and runtime domain consistent even when the
        # parent shell exports a different ROS_DOMAIN_ID.
        if getattr(args, "ros_domain_id", None) is not None:
            os.environ["ROS_DOMAIN_ID"] = str(int(args.ros_domain_id))

        rclpy.init()

    def setup(self):
        """Build scene, clone envs, set up cameras and shared memory."""
        # ---- World ----
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=0.0,  # disable IsaacSim physics; MuJoCo owns physics
            rendering_dt=1.0 / self.args.render_hz,
        )
        self.stage = get_current_stage()

        # ---- Clone scene ----
        cloner = GridCloner(spacing=self.args.clone_spacing)
        cloner.define_base_env("/World/envs")
        env_paths = cloner.generate_paths("/World/envs/env", self.num_envs)

        # Load foreground scene into env_0
        add_reference_to_stage(self.args.scene_usd, env_paths[0])

        # Load robot USD and apply initial base pose when task-name mode is active
        if self.args.robot_usd:
            robot_prim_path = env_paths[0] + self.args.robot_prim
            add_reference_to_stage(self.args.robot_usd, robot_prim_path)
            if _task_resolved is not None:
                self._set_robot_base_pose(
                    robot_prim_path,
                    _task_resolved["robot_init_position"],
                    _task_resolved["robot_init_quaternion"],
                )

        # Create the default visualization camera before cloning so that every
        # clone gets its own copy of the camera prim.
        if self.args.main_cam_prim == "/default_viz_camera":
            self._create_default_viz_camera(
                env_paths[0],
                self.args.default_cam_pos,
                self.args.default_cam_target,
            )

        # Clone to all envs.
        # copy_from_source=True creates independent prim copies per env so that
        # each env can have its own object poses (driven independently by tf_render).
        # copy_from_source=False would create USD instances sharing one prototype,
        # causing all envs to move together when any single prim is modified.
        cloner.clone(
            source_prim_path=env_paths[0],
            prim_paths=env_paths,
            copy_from_source=True,
            replicate_physics=False,  # physics disabled; MuJoCo owns physics
        )

        # Add ground plane (dark blue, like Isaac Lab style) before cloning.
        self._add_ground_plane(env_paths[0])

        # Add a dome light when the scene has no lights (always-on default).
        if self._count_lights_under_env(env_paths[0]) == 0:
            dome = UsdLux.DomeLight.Define(self.stage, env_paths[0] + "/AutoDomeLight")
            dome.CreateIntensityAttr(float(self.args.auto_dome_light_intensity))

        self.world.reset()

        # ---- Build body name → USD prim name map ----
        body_name_map = self._build_body_name_map(env_paths[0])
        print(f"[RLRenderer] body_name_map built: {len(body_name_map)} prims indexed under env_0")

        # ---- Parse cameras list ----
        import json as _json
        self._camera_list = []
        if self.args.cameras_json:
            try:
                self._camera_list = _json.loads(self.args.cameras_json)
            except Exception:
                pass
        if not self._camera_list:
            self._camera_list = [{"name": "main", "prim": self.args.main_cam_prim}]
            if self.args.wrist_cam_prim:
                self._camera_list.append({"name": "wrist", "prim": self.args.wrist_cam_prim})
        self._num_cams = len(self._camera_list)
        self.cam_annotators_all: List[List] = [[] for _ in range(self._num_cams)]

        # ---- Set up cameras per clone ----
        for i, env_path in enumerate(env_paths):
            for cam_idx, cam_cfg in enumerate(self._camera_list):
                ann = self._setup_camera(env_path + cam_cfg["prim"])
                self.cam_annotators_all[cam_idx].append(ann)
            # backward compat
            self.cam_annotators_main.append(self.cam_annotators_all[0][i] if self._num_cams > 0 else None)
            self.cam_annotators_wrist.append(self.cam_annotators_all[1][i] if self._num_cams > 1 else None)

        # ---- ROS subscribers (one per env) ----
        for i, env_path in enumerate(env_paths):
            sub = EnvTFSubscriber(i, self.stage, env_path, body_name_map)
            self.env_subscribers.append(sub)

        # ---- Background ROS executor ----
        # spin_once(timeout_sec=0.0) in the render callback is unreliable with many
        # publishers: odd-indexed envs consistently miss their messages because DDS
        # delivers them just after the non-blocking poll returns.  Instead, run a
        # persistent MultiThreadedExecutor in a background thread so all subscriber
        # callbacks fire as soon as messages arrive, independent of render timing.
        self._ros_executor = rclpy.executors.SingleThreadedExecutor()
        for sub in self.env_subscribers:
            self._ros_executor.add_node(sub)
        self._ros_thread = threading.Thread(
            target=self._spin_ros_executor_safe, daemon=True, name="geniesim_ros_spin"
        )
        self._ros_thread.start()

        # ---- Shared memory ----
        h, w = self.args.cam_height, self.args.cam_width
        shm_bytes = _shm_total_bytes(self.num_envs, h, w, num_cams=self._num_cams)
        try:
            self.shm = shared_memory.SharedMemory(name=self.args.shm_name, create=True, size=shm_bytes)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.args.shm_name, create=False, size=shm_bytes)
        self.shm_array = np.ndarray(
            (self.num_envs, self._num_cams, h, w, 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
            offset=_SHM_HEADER_BYTES,
        )
        self.frame_counter = np.ndarray((1,), dtype=np.uint32, buffer=self.shm.buf, offset=0)
        self.frame_counter[0] = 0

        # Register render callback
        self.world.add_render_callback("geniesim_rl_render", self._render_callback)

        print(f"[RLRenderer] Ready | envs={self.num_envs} | shm={self.args.shm_name} | {shm_bytes//1024}KB")

    def _spin_ros_executor_safe(self):
        """Spin ROS executor; suppress expected shutdown exception."""
        try:
            self._ros_executor.spin()
        except Exception as exc:
            try:
                from rclpy.executors import ExternalShutdownException
                if isinstance(exc, ExternalShutdownException):
                    return
            except Exception:
                pass
            raise

    # ------------------------------------------------------------------
    # Robot pose helpers
    # ------------------------------------------------------------------

    def _set_robot_base_pose(
        self,
        robot_prim_path: str,
        position: List[float],
        quaternion: List[float],
    ):
        """
        Set the initial world-space pose of the robot root prim via USD
        xform ops.  position=[x,y,z], quaternion=[w,x,y,z].
        """
        prim = self.stage.GetPrimAtPath(robot_prim_path)
        if not prim.IsValid():
            print(f"[RLRenderer] Warning: robot prim not found at {robot_prim_path}, skipping pose init")
            return

        xf = UsdGeom.Xformable(prim)
        # Clear any existing ops authored by the reference to avoid conflicts.
        xf.ClearXformOpOrder()

        translate_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(*position))

        w, x, y, z = quaternion

        try:
            existing_orient_attr = prim.GetAttribute("xformOp:orient")
            existing_type = str(existing_orient_attr.GetTypeName()) if existing_orient_attr.IsValid() else ""
            use_quatd = existing_type == "quatd"

            if use_quatd:
                orient_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                orient_op.Set(Gf.Quatd(w, x, y, z))
            else:
                orient_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
                orient_op.Set(Gf.Quatf(w, x, y, z))
        except Exception as e:
            raise

        print(
            f"[RLRenderer] Robot base pose set: pos={position}  quat(wxyz)={quaternion}"
        )

    # ------------------------------------------------------------------
    # Default visualization camera
    # ------------------------------------------------------------------

    def _create_default_viz_camera(
        self,
        env_path: str,
        cam_pos: List[float],
        cam_target: List[float],
    ) -> str:
        """
        Create a UsdGeom.Camera at <env_path>/default_viz_camera positioned
        at cam_pos and looking toward cam_target (Z-up world).
        Returns the camera prim path.
        """
        cam_prim_path = env_path + "/default_viz_camera"
        cam = UsdGeom.Camera.Define(self.stage, cam_prim_path)
        xf = UsdGeom.Xformable(cam.GetPrim())

        w, x, y, z = self._lookat_quaternion(cam_pos, cam_target)

        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*cam_pos))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(w, x, y, z))

        # Reasonable defaults for a 640×480 overview camera
        cam.CreateFocalLengthAttr(18.0)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(15.2908)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

        print(
            f"[RLRenderer] Default viz camera created at {cam_prim_path}\n"
            f"  pos={cam_pos}  target={cam_target}  quat(wxyz)=({w:.4f},{x:.4f},{y:.4f},{z:.4f})"
        )
        return cam_prim_path

    @staticmethod
    def _lookat_quaternion(
        eye: List[float], target: List[float], world_up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    ) -> Tuple[float, float, float, float]:
        """
        Compute a (w, x, y, z) quaternion for a camera at `eye` looking toward
        `target` in a Z-up world.  The camera looks along its local -Z axis.

        Convention (right-handed):
          local X = image right  = cross(forward, world_up).normalized
          local Y = image up     = cross(local_X, forward)
          local Z = -forward     (camera looks in -Z)
        """
        def normalize(v):
            n = math.sqrt(sum(c * c for c in v))
            return tuple(c / n for c in v) if n > 1e-9 else v

        def cross(a, b):
            return (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )

        fwd = normalize(tuple(t - e for t, e in zip(target, eye)))  # local -Z
        right = normalize(cross(fwd, world_up))                      # local  X
        up = cross(right, fwd)                                        # local  Y
        lz = tuple(-f for f in fwd)                                   # local  Z

        # Rotation matrix: columns are [right, up, lz]
        m00, m01, m02 = right[0], up[0], lz[0]
        m10, m11, m12 = right[1], up[1], lz[1]
        m20, m21, m22 = right[2], up[2], lz[2]

        trace = m00 + m11 + m22
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m21 - m12) * s
            y = (m02 - m20) * s
            z = (m10 - m01) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * math.sqrt(max(0.0, 1.0 + m00 - m11 - m22))
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * math.sqrt(max(0.0, 1.0 + m11 - m00 - m22))
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * math.sqrt(max(0.0, 1.0 + m22 - m00 - m11))
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        # Normalize to guard against floating-point drift
        n = math.sqrt(w * w + x * x + y * y + z * z)
        return (w / n, x / n, y / n, z / n)

    # ------------------------------------------------------------------
    # Body-name map
    # ------------------------------------------------------------------

    def _build_body_name_map(self, env0_path: str) -> Dict[str, str]:
        """
        Build a mapping from MuJoCo body names to USD prim path suffixes
        (relative to env root, without leading slash).

        Priority:
          1. If a <scene>_body_map.json sidecar exists, use it verbatim.
          2. Auto-scan all prims under env0_path and index them by leaf name.
             This covers robot links (e.g. "base_link" → "robot/base_link")
             without requiring any hand-crafted mapping.
        """
        map_path = (
            self.args.scene_usd
            .replace(".usd", "_body_map.json")
            .replace(".usda", "_body_map.json")
        )
        if os.path.exists(map_path):
            with open(map_path) as f:
                return json.load(f)

        # Auto-build: walk every prim under env0 and map leaf name → rel path.
        prefix = env0_path.rstrip("/") + "/"
        leaf_map: Dict[str, str] = {}
        for prim in self.stage.Traverse():
            path_str = str(prim.GetPath())
            if not path_str.startswith(prefix):
                continue
            rel = path_str[len(prefix):]   # e.g. "robot/base_link"
            leaf = rel.split("/")[-1]      # e.g. "base_link"
            # Keep the shallowest match so top-level prims win over children.
            if leaf and leaf not in leaf_map:
                leaf_map[leaf] = rel
        return leaf_map

    def _setup_camera(self, cam_prim_path: str):
        """Create a replicator render product for the given camera prim."""
        resolved_path = cam_prim_path
        if not is_prim_path_valid(resolved_path):
            # Fall back to the first camera found under the same env root.
            parts = resolved_path.strip("/").split("/")
            env_root = "/" + "/".join(parts[:3]) if len(parts) >= 3 else resolved_path
            fallback = self._find_first_camera_under_env(env_root)
            if not fallback:
                print(f"[RLRenderer] Camera prim not found: {cam_prim_path}")
                return None
            resolved_path = fallback

        rp = rep.create.render_product(resolved_path, (self.args.cam_width, self.args.cam_height))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp])
        return annot

    def _find_first_camera_under_env(self, env_root: str) -> Optional[str]:
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(env_root + "/"):
                continue
            if prim.IsA(UsdGeom.Camera):
                return path
        return None

    def _add_ground_plane(self, env_path: str, size: float = 100.0) -> None:
        """Add a solid dark-blue ground plane (visual only) to the environment."""
        floor_path = env_path + "/GroundPlane"
        mat_path = env_path + "/GroundPlaneMat"

        # Quad mesh lying flat in the XY plane at Z=0.
        h = size / 2.0
        mesh = UsdGeom.Mesh.Define(self.stage, floor_path)
        mesh.CreatePointsAttr([
            Gf.Vec3f(-h, -h, 0.0),
            Gf.Vec3f( h, -h, 0.0),
            Gf.Vec3f( h,  h, 0.0),
            Gf.Vec3f(-h,  h, 0.0),
        ])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
        mesh.SetNormalsInterpolation("vertex")

        # Dark blue material (UsdPreviewSurface, similar to Isaac Lab ground).
        mat = UsdShade.Material.Define(self.stage, mat_path)
        shader = UsdShade.Shader.Define(self.stage, mat_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.459, 0.636, 0.922)  # sRGB [117, 162, 235] / 255
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh).Bind(mat)

    def _count_lights_under_env(self, env_root: str) -> int:
        cnt = 0
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(env_root + "/"):
                continue
            type_name = prim.GetTypeName()
            if isinstance(type_name, str) and "Light" in type_name:
                cnt += 1
        return cnt

    def _render_callback(self, step_size: float):
        """Called every render step by IsaacSim."""
        # Skip the entire frame (USD writes + GPU readback) when no env has new
        # MuJoCo data.  This happens when the render callback fires slightly before
        # the next TF message arrives, avoiding redundant work that would produce
        # an identical frame to the previous one.
        if not any(sub._dirty for sub in self.env_subscribers):
            return

        # 1. Apply latest TF poses from all env subscribers.
        # Sdf.ChangeBlock batches all USD attribute writes across all envs into a
        # single change notification, avoiding per-Set() notification overhead which
        # is the dominant cost when updating many bodies across 12+ environments.
        with Sdf.ChangeBlock():
            for sub in self.env_subscribers:
                sub.apply_tf()

        # 2. Capture camera images and write to shared memory
        h, w = self.args.cam_height, self.args.cam_width
        for i in range(self.num_envs):
            for cam_idx in range(self._num_cams):
                ann = self.cam_annotators_all[cam_idx][i]
                if ann is not None:
                    data = ann.get_data()
                    if data is not None and data.shape == (h, w, 4):
                        self.shm_array[i, cam_idx, :, :, :] = data[:, :, :3]

        # Increment frame counter (signals VectorEnv that new data is available)
        self.frame_counter[0] = (int(self.frame_counter[0]) + 1) % (2**32)

    def run(self):
        """Main blocking loop."""
        while simulation_app.is_running():
            self.world.step(render=True)

        # Cleanup
        self._ros_executor.shutdown(timeout_sec=2.0)
        for sub in self.env_subscribers:
            sub.destroy_node()
        rclpy.shutdown()
        if self.shm:
            self.shm.close()
            self.shm.unlink()
        self.world.stop()
        simulation_app.close()


# ---------------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    renderer = RLRenderer(_args)
    renderer.setup()
    renderer.run()
