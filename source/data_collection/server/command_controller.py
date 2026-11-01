import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import omni
import omni.replicator.core as rep
import omni.timeline
import yaml
from isaacsim.core.api.materials import OmniGlass, OmniPBR, PhysicsMaterial
from isaacsim.core.api.objects import cuboid, cylinder
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim
from isaacsim.core.prims import SingleRigidPrim as RigidPrim
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.prims import get_prim_at_path, get_prim_object_type, delete_prim, create_prim
from isaacsim.core.utils.stage import add_reference_to_stage, update_stage, print_stage_prim_paths
try:
    from isaacsim.core.utils.stage import remove_deleted_references  # noqa: F401
except ImportError:
    # Isaac Sim 4.5+ removed this helper; use None if the code does not require it.
    def remove_deleted_references(*args, **kwargs):  # type: ignore
        pass
from isaacsim.sensors.camera import Camera

from omni.kit.viewport.utility import get_active_viewport_and_window
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from omni.physx.scripts import utils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux

from common.base_utils.logger import logger
from common.base_utils.transform_utils import mat2quat_wxyz, quat2mat_wxyz
from common.data_filter.runtime_checker import CheckerStatus, create_checker
from sim_backend import physics, render
import sim_backend
from server.motion_generator.revolute_joint_trajectory import world_point_trajectory_with_body1_revolute, get_current_revolute_joint_info
from server.motion_generator.trajectory_timing import (
    resolve_slowdown_factor as resolve_trajectory_slowdown_factor,
)
from server.command_enum import Command, command_value_to_string
from server.generated_geometry_state import GeneratedGeometryRuntime
from server.controllers.parallel_gripper import ParallelGripper
from server.utils import Light
from server.robot import RobotCfg
from server.ros_publisher.base import USDBase
from server.ui_builder import UIBuilder
from server.utils import batch_matrices_to_quaternions_scipy_w_first

MAX_EXTRACT_PROCESS_NUM = 1


# task_name arrives over an unauthenticated gRPC channel, becomes a filesystem
# path component, and is interpolated into shell commands. Restrict it to a
# conservative character set so it cannot traverse directories or inject shell
# metacharacters. Legitimate task names look like "[pick_block_g1_0]".
_SAFE_TASK_NAME_RE = re.compile(r"^[A-Za-z0-9_.\[\]-]+$")


def sanitize_task_name(task_name):
    if (
        not isinstance(task_name, str)
        or task_name in (".", "..")
        or not _SAFE_TASK_NAME_RE.match(task_name)
    ):
        raise ValueError(f"Invalid task_name: {task_name!r}")
    return task_name


def _contains_traversal(user_path):
    # Reject empty, NUL bytes, and any ".." path component (either separator).
    if not isinstance(user_path, str) or not user_path or "\x00" in user_path:
        return True
    return ".." in user_path.replace("\\", "/").split("/")


def _resolves_within(base_dir, user_path):
    """True iff joining user_path onto base_dir stays inside base_dir.

    Mirrors the runtime's own os.path.join(base_dir, user_path): an absolute
    user_path makes join discard base_dir entirely, which realpath containment
    then rejects."""
    base_real = os.path.realpath(base_dir)
    resolved = os.path.realpath(os.path.join(base_real, user_path))
    return resolved == base_real or resolved.startswith(base_real + os.sep)


def validate_robot_cfg_file(robot_cfg, config_dir):
    # robot_cfg_file is always a plain filename resolved under config/robot_cfg;
    # there is no legitimate reason for it to be absolute or to traverse out.
    if _contains_traversal(robot_cfg) or not _resolves_within(config_dir, robot_cfg):
        raise ValueError(f"Invalid robot_cfg_file: {robot_cfg!r}")
    return robot_cfg


def validate_scene_usd_path(scene_usd, assets_root):
    # Two legitimate shapes: a relative path under SIM_ASSETS (normal mode) and
    # an absolute path to a generated scene (llm_gen mode, see _init_robot_cfg).
    # Block ".." traversal / NUL in both; additionally contain relative paths
    # inside the assets root. Absolute paths remain allowed by design.
    if _contains_traversal(scene_usd):
        raise ValueError(f"Invalid scene_usd_path: {scene_usd!r}")
    if not os.path.isabs(scene_usd) and not _resolves_within(assets_root, scene_usd):
        raise ValueError(f"Invalid scene_usd_path (escapes assets root): {scene_usd!r}")
    return scene_usd


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
        llm_gen_mode=False,
        output_depth=False,
        enable_render=False,
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
        # Track the previous LinearMove from_current_pose and goal_offset for replanning.
        # With from_current_pose, the server derives the goal from current EE plus offset.
        # An unchanged client target alone cannot detect an offset-based step change.
        self._last_from_current_pose = False
        self._last_goal_offset = [0, 0, 0, 1, 0, 0, 0]
        # Cache SingleRigidPrim views by path to avoid repeated object-pose construction.
        # update_objects and TF publication can query these on every frame.
        self._rigid_view_cache: dict = {}
        # Retain retired views across recordings because the physics scene can still
        # reference their native resources; see _clear_temporary_publishers.
        self._rigid_view_graveyard: list = []
        # FEM pin state maintained by attach/release commands and physics-step servos:
        # _active_pin_groups[group_id] = {
        #   cloth_prim: cloth mesh path.
        #   anchor_link: pin anchor link path.
        #   node_idx: selected simulation-node indices as a CUDA long tensor (M,).
        #   offset: frozen node-minus-anchor CUDA offsets (M, 3).
        #   "gain":         float,                       # vcmd = clamp((tgt-cur)/dt * gain, ±vmax)
        #   "vmax":         float,
        #   fem_body_idx: CUDA int tensor (1,), currently body zero.
        #   anchor_view: cached SingleRigidPrim for the anchor.
        # }
        self._active_pin_groups: dict = {}
        # Rigid pin state maintained by attach/release commands and
        # the rigid-body 6D servo in _pin_servo_step:
        # _active_rigid_pin_groups[group_id] = {
        #   rigid_prim: pinned prim path; rigid bodies or other valid Xforms.
        #   anchor_link: path, or None for a fixed world pose.
        #   p_rel: frozen position relative to the anchor, shape (3,).
        #   q_rel: frozen wxyz orientation, shape (4,).
        #   "gain_lin":     float,
        #   "gain_ang":     float,
        #   "vmax_lin":     float,
        #   "vmax_ang":     float,
        #   "rigid_view":   SingleRigidPrim cached,
        #   "anchor_view":  SingleRigidPrim cached | None,
        # }
        self._active_rigid_pin_groups: dict = {}
        self.target_joints_pose = None
        self.task_name = None
        self.cameras = {}
        self.step = 0
        self.path_to_save = None
        self.exit = False
        # Record FEM vertices asynchronously for each captured frame.
        # Write deformable/<name>/<frame_idx>.npz under path_to_save.
        # Reset counters and start the executor at recording start;
        # wait for all NPZ writes at recording stop.
        self._cloth_record_executor = None
        self._cloth_record_counter = 0
        self._cloth_record_name_map: Dict[str, str] = {}
        # Track each frame deformable index independently and write it
        # to _cloth_frames.json, later merged into the final ROS state.json.
        # ({"frames":[{"time_stamp", "deformable":[{name, prim_path, file, n_nodes}]}, ...]}).
        # Keep it separate from frame_state.json, which contains stage events.
        self._cloth_frames: List[Dict[str, Any]] = []
        # Runtime-generated append-only meshes share action checkpoints and
        # recording time with robot/object state. Sources are discovered from
        # declarations on the live USD stage instead of configured paths.
        self._generated_geometry = GeneratedGeometryRuntime()
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
        self.ros_publishers_persistent = []  # Persistent nodes (camera, articulation_action, main robot joint_state)
        self.ros_publishers_temporary = []  # Temporary nodes (tf, articulat_objects joint_state)
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
            import rclpy  # noqa: PLC0415
            rclpy.init()
        # llm_gen
        self.llm_gen_mode = llm_gen_mode
        self.already_loaded = False
        # Blocked simulation-step state.
        self.block_step = True
        # Commands that allow physics steps, mapped to their settling margins.
        # Margins are baseline 60 Hz step counts, converted using physics_dt
        # by _scaled_margin_frames. Ten baseline steps become sixty at 360 Hz
        # while preserving the same approximately 167 ms settling time.
        self.allow_simulation_commands = {
            Command.LINEAR_MOVE: 10,
            Command.FOLLOW_TRAJECTORY: 10,
            Command.BIMANUAL_RIGID_MOVE: 10,
            Command.SET_JOINT_POSITION: 10,
            Command.SET_GRIPPER_STATE: 30,
            Command.CONTROL_GRIPPER: 0,
            Command.RESET: 60,
            Command.PLAYBACK: 0,
            Command.GET_CHECKER_STATUS: 0,
            Command.RUN_STEP: 0,
            Command.ADD_OBJECT: 3,
            Command.DETACH_OBJ: 10,
            Command.INIT_ROBOT: 30,
            # Synchronized dual-arm motion uses the single-arm LINEAR_MOVE margin.
            Command.DUAL_LINEAR_MOVE: 10,
            Command.DUAL_CUROBO_MOVE: 10,
        }
        # Whether a step-permitting command is being processed.
        self.is_processing_simulation_command = False
        # Remaining settling-margin counter.
        self.margin_frame_counter = 0
        # Settling margin for the current command.
        self.current_margin_frames = 0
        # Previous command, used to detect command transitions.
        self.last_processed_command = 0
        # Remaining RUN_STEP iterations.
        self.run_step_remaining = 0
        self.run_step_total = 0
        self.output_depth = output_depth
        self.enable_render = enable_render
        # Track simulation recording intervals: append at start and update the
        # latest end timestamp in the physics callback; combine intervals at stop.
        # Initialize here so total_sim_time can be computed during server shutdown
        # without relying on unreachable code after another method return.
        self.recording_times = []

    # Plans historically consumed one point per 60 Hz physics step.
    # Resample according to (1/60) / physics_dt at higher frequencies
    # while preserving duration and manageable PD target spacing.
    def _resolve_slowdown_factor(self, user_sf) -> float:
        """Combine explicit slowdown and physics-timestep alignment.
        factor = max(1, user_sf) * max(1, (1/60) / phys_dt); 360 Hz gives an auto factor of 6."""
        sf = max(1.0, float(user_sf or 1.0))
        try:
            phys_dt = float(self.ui_builder.my_world.get_physics_dt())
        except Exception:
            return sf
        if phys_dt <= 0:
            return sf
        return resolve_trajectory_slowdown_factor(sf, phys_dt)

    # ------------------------------------------------------------------
    # Asynchronous FEM vertex recording to NPZ.
    # ------------------------------------------------------------------

    def _cloth_recording_name(self, prim_path: str) -> str:
        """Resolve a cloth recording name using explicit mappings first.
        Otherwise select the last path segment other than body/entity/deformable;
        fall back to replacing path separators with underscores."""
        if prim_path in self._cloth_record_name_map:
            return self._cloth_record_name_map[prim_path]
        parts = [p for p in prim_path.split("/") if p]
        skip = {"body", "entity", "deformable"}
        for tok in reversed(parts):
            if tok not in skip:
                return tok
        return prim_path.replace("/", "_").strip("_") or "cloth"

    @staticmethod
    def _save_cloth_npz(abs_path: str, points: np.ndarray, prim_path: str, time_stamp: float):
        """Write NPZ data in the thread pool without GPU operations.
        Wait for ros2bag to create path_to_save before writing child directories;
        precreating its output directory would make ros2bag refuse recording."""
        try:
            parent = os.path.dirname(abs_path)  # <path_to_save>/deformable/<name>/
            path_to_save = os.path.dirname(os.path.dirname(parent))  # Two parent levels give path_to_save.
            if not os.path.isdir(path_to_save):
                return  # ros2bag has not created its output yet; skip and retry on the next tick.
            os.makedirs(parent, exist_ok=True)
            np.savez_compressed(
                abs_path,
                points=points,           # (N, 3) float32 world coord
                prim_path=str(prim_path),
                time_stamp=float(time_stamp),
            )
        except Exception as exc:
            logger.warning(f"[cloth_record] save {abs_path} failed: {exc}")

    def _snapshot_cloth_states(self, time_stamp: float) -> List[Dict[str, Any]]:
        """Synchronously copy GPU points to CPU, then enqueue asynchronous disk writes.
        Return name/prim_path/file/node-count entries for the intermediate cloth index;
        file paths are relative to path_to_save. Return [] without views."""
        if not self.path_to_save:
            return []
        cloth_views = getattr(self.ui_builder, "cloth_views", None) or {}
        if not cloth_views:
            return []
        if self._cloth_record_executor is None:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
            self._cloth_record_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="cloth-rec",
            )
        self._cloth_record_counter += 1
        frame_idx = self._cloth_record_counter
        entries: List[Dict[str, Any]] = []
        try:
            from pxr import UsdGeom  # noqa: PLC0415
            _stage = omni.usd.get_context().get_stage()
        except Exception:
            _stage = None
        for prim_path, view in cloth_views.items():
            try:
                # Legacy SoftBodyView get_nodal_positions returns concatenated
                # collision and render vertices. The first N belong to collision geometry;
                # the remaining M are skinned render vertices updated by PhysX.
                # Replay writes the render segment to mesh.points, matching asset topology
                # and the authored point count.
                # Surface views have only the triangular surface mesh.
                # get_simulation_nodal_positions returns surface/render points
                # without collision tetrahedra, so no split is required.
                pos_t = None
                api_used = None
                if hasattr(view, "get_nodal_positions"):
                    pos_t = view.get_nodal_positions()
                    api_used = "get_nodal_positions"
                elif hasattr(view, "get_simulation_nodal_positions"):
                    pos_t = view.get_simulation_nodal_positions()
                    api_used = "get_simulation_nodal_positions"
                else:
                    raise AttributeError(
                        f"view {type(view).__name__} has no known nodal positions API"
                    )
                pos_np = pos_t[0].detach().cpu().numpy().astype(np.float32)  # (N, 3) world coords

                # Split concatenated soft-body points only when collisionRestPoints plus
                # asset point counts match the complete returned array length.
                if _stage is not None and api_used == "get_nodal_positions":
                    _prim = _stage.GetPrimAtPath(prim_path)
                    if _prim and _prim.IsValid():
                        col_rest_attr = _prim.GetAttribute("physxDeformable:collisionRestPoints")
                        pts_attr = _prim.GetAttribute("points")
                        if (col_rest_attr and col_rest_attr.IsValid()
                                and pts_attr and pts_attr.IsValid()):
                            col_rest = col_rest_attr.Get()
                            render_pts = pts_attr.Get()
                            if col_rest is not None and render_pts is not None:
                                n_col = len(col_rest)
                                n_render = len(render_pts)
                                if pos_np.shape[0] == n_col + n_render:
                                    pos_np = pos_np[n_col:n_col + n_render]
            except Exception as exc:
                logger.warning(f"[cloth_record] read nodal {prim_path} failed: {exc}")
                continue
            name = self._cloth_recording_name(prim_path)
            rel_path = f"deformable/{name}/{frame_idx:06d}.npz"
            abs_path = os.path.join(self.path_to_save, rel_path)
            # GPU-to-CPU synchronization is done; run disk I/O in the thread pool.
            self._cloth_record_executor.submit(
                self._save_cloth_npz, abs_path, pos_np, prim_path, time_stamp,
            )
            entries.append({
                "name": name,
                "prim_path": prim_path,
                "file": rel_path,
                "n_nodes": int(pos_np.shape[0]),
            })
        return entries

    def _wait_cloth_recording_done(self):
        """Wait for all cloth NPZ writes at recording stop, then retire the executor."""
        if self._cloth_record_executor is not None:
            try:
                self._cloth_record_executor.shutdown(wait=True)
            except Exception as exc:
                logger.warning(f"[cloth_record] executor shutdown failed: {exc}")
            self._cloth_record_executor = None

    def _scaled_margin_frames(self, margin60: int) -> int:
        """Convert a margin measured in 60 Hz steps to equivalent steps at physics_dt.
        Preserve margin60 / 60 seconds across physics frequencies: ten baseline steps
        become twenty at 120 Hz or sixty at 360 Hz."""
        try:
            phys_dt = float(self.ui_builder.my_world.get_physics_dt())
        except Exception:
            phys_dt = 1.0 / 60.0
        if phys_dt <= 0:
            return int(margin60)
        # Convert margin60 / 60 seconds into steps at the current physics_dt.
        scaled = margin60 / (60.0 * phys_dt)
        return max(0, int(round(scaled)))

    def _clear_temporary_publishers(self):
        """Clear all temporary publisher nodes"""
        for node in self.ros_publishers_temporary:
            try:
                if hasattr(node, 'destroy_node'):
                    node.destroy_node()
            except Exception as e:
                logger.warning(f"Failed to destroy temporary node: {e}")
        self.ros_publishers_temporary.clear()
        logger.info("Cleared all temporary publisher nodes")
        self._rotate_rigid_view_caches()

    def _rotate_rigid_view_caches(self):
        # Reloading a scene destroys PhysX objects referenced by old views.
        # Rotate caches instead of clearing them: garbage collection can free
        # native resources still referenced by the scene and cause crashes.
        # Rotate both long-lived path-to-view caches together:
        #   ros_publisher.base cache serves recorded TF poses.
        #   controller cache serves gRPC object-pose queries.
        # Leaving either stale can crash subsequent get_world_poses calls.
        try:
            from server.ros_publisher.base import rotate_rigid_view_cache  # noqa: PLC0415

            rotate_rigid_view_cache()
        except Exception as e:
            logger.warning(f"Failed to rotate rigid view cache: {e}")
        if self._rigid_view_cache:
            self._rigid_view_graveyard.append(self._rigid_view_cache)
            logger.info(
                f"Rotated controller rigid view cache "
                f"({len(self._rigid_view_cache)} stale entries retired, "
                f"graveyard={len(self._rigid_view_graveyard)})"
            )
            self._rigid_view_cache = {}

    def _reset_publisher_phase(self):
        """Reset every publisher phase so topics start recording together.
        Temporary nodes are recreated each round while persistent nodes retain counters;
        reset both to avoid a one-step TF/joint-state offset in later recordings."""
        reset = 0
        for node in list(self.ros_publishers_persistent) + list(self.ros_publishers_temporary):
            fn = getattr(node, "reset_publish_phase", None)
            if callable(fn):
                try:
                    fn()
                    reset += 1
                except Exception as e:
                    logger.warning(f"Failed to reset publish phase: {e}")
        logger.info(
            f"Reset publish phase on {reset} node(s) "
            f"(persistent={len(self.ros_publishers_persistent)}, "
            f"temporary={len(self.ros_publishers_temporary)})"
        )

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
            current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            robot_config_dir = os.path.join(current_directory, "config/robot_cfg")
            robot = RobotCfg(os.path.join(robot_config_dir, robot_cfg))
            if self.llm_gen_mode and self.already_loaded:
                self._rotate_rigid_view_caches()
                self.ui_builder.my_world.stop()
                delete_prim("/World")
                # for usd_object in self.usd_objects.values():
                #     default_state = usd_object.get_default_state()
                #     usd_object.set_world_pose(default_state.position, default_state.orientation)
                update_stage()
                self.usd_objects.clear()
                self.articulat_objects.clear()
                self.rigid_bodies.clear()
                # self.ui_builder.my_world.reset()
                [curoboMotion.reset() for curoboMotion in self.ui_builder.curoboMotion.values()]
                # reload scene
                self.scene_usd = scene_usd
                self.scene_usd_path = os.path.join(self.sim_assets_root, scene_usd)
                add_reference_to_stage(self.scene_usd_path, "/World")
                self._play()
                return
                # for usd_object in self.usd_objects.values():
                #     default_state = usd_object.get_default_state()
                #     usd_object.set_world_pose(default_state.position, default_state.orientation)
            # create robot first to use robot dof names
            self.scene_usd = scene_usd
            if os.path.isabs(self.scene_usd):
                self.llm_gen_mode = True # TODO: use environment variable to determine llm gen mode
                logger.info(f"llm gen mode, scene_usd: {self.scene_usd}")
            else:
                logger.info(f"normal mode, scene_usd: {self.scene_usd}")
            self.robot_usd_path = os.path.join(self.sim_assets_root, robot.robot_usd)
            self.scene_usd_path = os.path.join(self.sim_assets_root, scene_usd)
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
            elif "ARX" in robot.robot_name:
                pass
            elif "Aloha" in robot.robot_name:
                pass
            with robot_rep:
                rep.modify.semantics([("class", "robot")])
            self.robot_cfg = robot
            self._play()

            # init kinematic solver
            articulation = self._initialize_articulation()
            self.dof_names = physics.get_dof_names(articulation)
            if "ARX" in robot.robot_name:
                logger.warning(f"=== ARX BRANCH HIT, robot_name={robot.robot_name} ===")
                # Match curobo example pick_place_arx.py: every active joint gets
                # max_effort=5000, and gripper joints additionally need explicit
                # gains (the URDF mimic relationship leaves outer kds=0 by default).
                arx_arm_names = [
                    "arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
                    "arm_l_joint4", "arm_l_joint5", "arm_l_joint6",
                    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3",
                    "arm_r_joint4", "arm_r_joint5", "arm_r_joint6",
                ]
                arx_arm_idx = np.array([
                    articulation.get_dof_index(n) for n in arx_arm_names
                ])
                physics.set_max_efforts(
                    articulation,
                    np.array([5000.0] * len(arx_arm_idx)),
                    joint_indices=arx_arm_idx,
                )
                arx_grip_names = [
                    "gripper_l_active_joint",
                    "gripper_l_passive_joint",
                    "gripper_r_active_joint",
                    "gripper_r_passive_joint",
                ]
                arx_grip_idx = np.array([
                    articulation.get_dof_index(n) for n in arx_grip_names
                ])
                physics.set_gains(
                    articulation,
                    kps=np.array([50000.0] * 4),
                    kds=np.array([5000.0] * 4),
                    joint_indices=arx_grip_idx,
                )
                physics.set_max_efforts(
                    articulation,
                    np.array([50000.0] * 4),
                    joint_indices=arx_grip_idx,
                )
                logger.info("ARX: arm max_effort=5000, gripper kps=50000 kds=5000 max_effort=50000")
            elif "Aloha" in robot.robot_name:
                logger.warning(f"=== Aloha BRANCH HIT, robot_name={robot.robot_name} ===")
                # Same idea as ARX: bump arm efforts, and explicitly drive
                # gripper kps/kds since the URDF mimic relation otherwise
                # leaves the passive (follower) joint with kds=0.
                aloha_arm_names = [
                    "arm_fl_joint1", "arm_fl_joint2", "arm_fl_joint3",
                    "arm_fl_joint4", "arm_fl_joint5", "arm_fl_joint6",
                    "arm_fr_joint1", "arm_fr_joint2", "arm_fr_joint3",
                    "arm_fr_joint4", "arm_fr_joint5", "arm_fr_joint6",
                ]
                aloha_arm_idx = np.array([
                    articulation.get_dof_index(n) for n in aloha_arm_names
                ])
                physics.set_max_efforts(
                    articulation,
                    np.array([5000.0] * len(aloha_arm_idx)),
                    joint_indices=aloha_arm_idx,
                )
                aloha_grip_names = [
                    "gripper_fl_active_joint",
                    "gripper_fl_passive_joint",
                    "gripper_fr_active_joint",
                    "gripper_fr_passive_joint",
                ]
                aloha_grip_idx = np.array([
                    articulation.get_dof_index(n) for n in aloha_grip_names
                ])
                physics.set_gains(
                    articulation,
                    kps=np.array([50000.0] * 4),
                    kds=np.array([5000.0] * 4),
                    joint_indices=aloha_grip_idx,
                )
                physics.set_max_efforts(
                    articulation,
                    np.array([50000.0] * 4),
                    joint_indices=aloha_grip_idx,
                )
                logger.info("Aloha: arm max_effort=5000, gripper kps=50000 kds=5000 max_effort=50000")
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
            if self.llm_gen_mode:
                # add default lights
                stage = omni.usd.get_context().get_stage()
                dome_light = UsdLux.DomeLight.Define(stage, "/dome_light")
                dome_light.AddRotateXYZOp().Set((-90.0, -30.0, 0.0))
                dome_light.GetEnableColorTemperatureAttr().Set(True)
                dome_light.GetColorTemperatureAttr().Set(6150.0)
                dome_light.GetIntensityAttr().Set(1.0)
                dome_light.GetExposureAttr().Set(9.0)
                dome_light.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)

                distant_light = UsdLux.DistantLight.Define(stage, "/distant_light")
                distant_light.AddRotateXYZOp().Set((-35.0, 45.0, 0.0))
                distant_light.GetEnableColorTemperatureAttr().Set(True)
                distant_light.GetColorTemperatureAttr().Set(7250.0)
                distant_light.GetIntensityAttr().Set(1.0)
                distant_light.GetExposureAttr().Set(10.0)

                # add default plane
                from isaacsim.core.api.objects.ground_plane import GroundPlane

                GroundPlane(prim_path="/default_ground_plane", z_position=0, visible=False)

                self.already_loaded = True

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
        # This method is called by the server only after ``World.step`` returns.
        # SoftBodyView reads are invalid from ``callback_physics`` while PhysX is
        # still simulating, so deformable recording must stay on this side of
        # the step boundary.
        self._cloth_record_tick()
        with self._timing_context("on_physics_step"):
            with self._timing_context("on_physics_step:ui_builder"):
                self.ui_builder._on_every_frame_trajectory_list()
                self.ui_builder._on_every_frame_dual_trajectory()
                # curobo step
                stepped_curobo_motion_ids = set()
                for key, curobo_motion in self.ui_builder.curoboMotion.items():
                    if id(curobo_motion) in stepped_curobo_motion_ids:
                        continue
                    stepped_curobo_motion_ids.add(id(curobo_motion))
                    additional_action = None
                    if self.gripper_action_timing is not None:
                        state = self.gripper_action_timing.get("state", None)
                        timing = self.gripper_action_timing.get("timing", None)
                        is_right = self.gripper_action_timing.get("is_right", True)
                        active_key = "right" if is_right else "left"
                        active_motion = self.ui_builder.curoboMotion.get(active_key)
                        if active_motion is curobo_motion and state is not None and timing is not None:
                            if (
                                curobo_motion.cmd_plan is not None
                                and curobo_motion.cmd_idx
                                >= len(curobo_motion.cmd_plan.position) * timing
                            ):
                                additional_action = self._get_gripper_action(state, is_right)
                    curobo_motion.on_physics_step(self.motion_run_ratio, additional_action)

            # Inject FEM pin servo velocities every physics step
            # to counteract internal forces pulling selected nodes away from the target.
            self._pin_servo_step()
            self.on_command_step()

    def _pin_servo_step(self):
        """Inject node velocity servos for active FEM pins each physics step.
        vcmd = clamp((anchor_world + frozen_offset - node_world) / dt * gain, +/-vmax).
        Skip when there are no active pins or deformable simulation is disabled."""
        # FEM cloth pin
        self._fem_pin_servo_step()
        # Rigid pins are maintained independently of cloth_views.
        self._rigid_pin_servo_step()

    def _fem_pin_servo_step(self):
        if not self._active_pin_groups:
            return
        cloth_views = getattr(self.ui_builder, "cloth_views", None)
        if not cloth_views:
            return
        try:
            import torch  # noqa: PLC0415
        except Exception:
            return
        try:
            dt = float(self.ui_builder.my_world.get_physics_dt())
        except Exception:
            dt = 1.0 / 360.0
        if dt <= 0:
            return

        # Group pin sets by cloth so both arms share one nodal velocity write
        # and avoid repeated GPU synchronization.
        per_cloth_writes: dict = {}
        for group_id, g in self._active_pin_groups.items():
            view = cloth_views.get(g["cloth_prim"])
            if view is None:
                continue
            try:
                anchor_pos = self._read_anchor_world_pos(
                    g["anchor_link"], g.get("anchor_view")
                )
                if anchor_pos is None:
                    continue
                pos_t = view.get_simulation_nodal_positions()  # (1, N, 3) torch.cuda
                # Current positions of selected nodes.
                cur = pos_t[0, g["node_idx"]]                  # (M, 3)
                tgt = anchor_pos + g["offset"]                  # (M, 3)
                vcmd = (tgt - cur) / dt * g["gain"]
                vmax = g["vmax"]
                vcmd = torch.clamp(vcmd, min=-vmax, max=vmax)
                per_cloth_writes.setdefault(g["cloth_prim"], (view, [])
                                            )[1].append((g["node_idx"], vcmd))
            except Exception as exc:
                logger.warning(f"[pin] Failed to apply group {group_id}: {exc}")
                continue

        for cloth_prim, (view, group_writes) in per_cloth_writes.items():
            try:
                vel = view.get_simulation_nodal_velocities().clone()  # (1, N, 3) cuda
                for node_idx, vcmd in group_writes:
                    vel[0, node_idx] = vcmd
                view.set_simulation_nodal_velocities(vel, view._fem_body_idx_cache)
            except Exception as exc:
                logger.warning(f"[pin] Failed to write cloth={cloth_prim} velocities: {exc}")

    @staticmethod
    def _quat_mul_wxyz(a, b):
        """Multiply torch wxyz quaternions shaped (..., 4)."""
        aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        import torch  # noqa: PLC0415
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dim=-1)

    @staticmethod
    def _quat_conj_wxyz(q):
        import torch  # noqa: PLC0415
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

    @staticmethod
    def _quat_rotate_vec(q, v):
        """Rotate vector v (3,) using quaternion q in wxyz order."""
        import torch  # noqa: PLC0415
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
        # Quaternion vector rotation: v + 2 * cross(qv, cross(qv, v) + w*v).
        # Expand the expression directly to avoid two cross-product calls.
        rx = (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz
        ry = 2 * (x * y + w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - w * x) * vz
        rz = 2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (1 - 2 * (x * x + y * y)) * vz
        return torch.stack([rx, ry, rz], dim=-1)

    @staticmethod
    def _np_quat_mul(a, b):
        """Multiply numpy wxyz quaternions, each shaped (4,)."""
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dtype=np.float64)

    @staticmethod
    def _np_quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

    @staticmethod
    def _np_quat_rotate(q, v):
        """Rotate numpy vector v (3,) using quaternion q (4,)."""
        w, x, y, z = q
        vx, vy, vz = v
        return np.array([
            (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz,
            2 * (x * y + w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - w * x) * vz,
            2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (1 - 2 * (x * x + y * y)) * vz,
        ], dtype=np.float64)

    @staticmethod
    def _set_disable_gravity(rigid_prim_path: str, disable: bool):
        """Set PhysxRigidBodyAPI.disableGravity and return (success, previous_value).
        None means the attribute was not authored; restore it by clearing the override."""
        try:
            from pxr import PhysxSchema  # noqa: PLC0415
        except Exception:
            return False, None
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(rigid_prim_path) if stage else None
            if not (prim and prim.IsValid()):
                return False, None
            api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not api:
                # Apply PhysxRigidBodyAPI when missing on a rigid-body prim.
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            attr = api.GetDisableGravityAttr()
            had_value = attr.HasAuthoredValue()
            prev = bool(attr.Get()) if had_value else None
            if disable:
                api.CreateDisableGravityAttr(True, False)
            else:
                # Restore an authored value, or clear it to use the default False.
                if prev is True:
                    attr.Set(True)
                elif prev is False:
                    attr.Set(False)
                else:
                    attr.Clear()
            return True, prev
        except Exception:
            return False, None

    @staticmethod
    def _restore_gravity(rigid_prim_path: str, prev_value):
        """Restore the authored gravity flag, or clear it if previously unauthored."""
        try:
            from pxr import PhysxSchema  # noqa: PLC0415
        except Exception:
            return False, None
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(rigid_prim_path) if stage else None
            if not (prim and prim.IsValid()):
                return False, None
            api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not api:
                return False, None
            attr = api.GetDisableGravityAttr()
            if not attr:
                return False, None
            if prev_value is True:
                attr.Set(True)
            elif prev_value is False:
                attr.Set(False)
            else:
                attr.Clear()
            return True, None
        except Exception:
            return False, None

    def _read_anchor_pose(self, anchor_link: str, anchor_view):
        """Read anchor world position and wxyz orientation as float64 numpy arrays.
        Prefer the cached rigid tensor view; otherwise use physics.get_prim_world_pose
        for any valid Xform/link. Return (None, None) on failure."""
        try:
            if anchor_view is not None:
                a_pos, a_ori = anchor_view._rigid_prim_view.get_world_poses(usd=False)
                return a_pos[0], a_ori[0]
            pose = physics.get_prim_world_pose(anchor_link)
            if pose is None:
                return None, None
            return pose[0], pose[1]
        except Exception:
            return None, None

    def _rigid_pin_servo_step(self):
        """Enforce each active pin's frozen anchor-relative or world pose.
        Pose mode writes the target and zeros rigid velocities. Velocity mode limits
        the translation step and injects 6D velocity for collision response; nonrigid
        prims fall back to pose mode. Without an anchor, use the frozen world pose."""
        if not self._active_rigid_pin_groups:
            return
        try:
            from sim_backend import backend as _bk  # noqa: PLC0415
        except Exception:
            return
        try:
            dt = float(self.ui_builder.my_world.get_physics_dt())
        except Exception:
            dt = 1.0 / 360.0
        if dt <= 0:
            return

        for gid, g in list(self._active_rigid_pin_groups.items()):
            try:
                target_view = g["target_view"]
                if target_view is None:
                    continue
                # Read the current prim pose through the wrapper and convert to numpy.
                cur_pos, cur_ori = target_view.get_world_pose()
                p_r = np.asarray(_bk.to_numpy(cur_pos), dtype=np.float64)
                q_r = np.asarray(_bk.to_numpy(cur_ori), dtype=np.float64)

                anchor_link = g.get("anchor_link")
                if anchor_link:
                    a_pos, a_ori = self._read_anchor_pose(anchor_link, g.get("anchor_view"))
                    if a_pos is None:
                        continue
                    p_a = np.asarray(_bk.to_numpy(a_pos), dtype=np.float64)
                    q_a = np.asarray(_bk.to_numpy(a_ori), dtype=np.float64)
                    # Compose the current anchor pose with the frozen relative pose.
                    p_t = p_a + self._np_quat_rotate(q_a, g["p_rel"])
                    q_t = self._np_quat_mul(q_a, g["q_rel"])
                else:
                    p_t = g["p_rel"]      # Absolute target pose in world coordinates.
                    q_t = g["q_rel"]

                mode = g.get("mode", "pose")
                rigid_view = g.get("rigid_view")
                use_velocity = (mode == "velocity") and (rigid_view is not None)

                if use_velocity:
                    # Velocity mode: limit the pose step by vmax_lin * dt and inject velocities.
                    delta_p = p_t - p_r
                    max_step = g["vmax_lin"] * dt
                    norm = np.linalg.norm(delta_p)
                    if norm > max_step:
                        delta_p = delta_p / norm * max_step
                    p_step = p_r + delta_p
                    target_view.set_world_pose(position=p_step, orientation=q_t)
                    # Linear servo velocity: clamp((target - current) / dt * gain, +/-vmax).
                    v_lin = delta_p / dt * g["gain_lin"]
                    v_lin = np.clip(v_lin, -g["vmax_lin"], g["vmax_lin"])
                    # Angular velocity uses axis-angle from q_target * inverse(q_current).
                    q_err = self._np_quat_mul(q_t, self._np_quat_conj(q_r))
                    if q_err[0] < 0:
                        q_err = -q_err
                    w_e = float(np.clip(q_err[0], -1.0, 1.0))
                    xyz = q_err[1:4]
                    sin_half = float(np.linalg.norm(xyz))
                    if sin_half < 1e-8:
                        axis_angle = xyz * 2.0
                    else:
                        angle = 2.0 * float(np.arctan2(sin_half, w_e))
                        axis_angle = xyz / sin_half * angle
                    v_ang = axis_angle / dt * g["gain_ang"]
                    v_ang = np.clip(v_ang, -g["vmax_ang"], g["vmax_ang"])
                    physics.set_rigid_velocity(rigid_view, linear=v_lin, angular=v_ang)
                else:
                    # Pose mode, including nonrigid fallback: write the fixed target pose.
                    target_view.set_world_pose(position=p_t, orientation=q_t)
                    # Zero rigid velocities to avoid inferred displacement velocities on the next step.
                    if rigid_view is not None or g.get("kind") == "rigid":
                        if rigid_view is None:
                            try:
                                from isaacsim.core.prims import SingleRigidPrim  # noqa: PLC0415
                                rigid_view = SingleRigidPrim(prim_path=g["rigid_prim"])
                                g["rigid_view"] = rigid_view
                            except Exception:
                                rigid_view = None
                        if rigid_view is not None:
                            try:
                                physics.set_rigid_velocity(rigid_view, linear=[0, 0, 0], angular=[0, 0, 0])
                            except Exception:
                                pass
            except Exception as exc:
                logger.warning(f"[rigid_pin] Failed to apply group {gid}: {exc}")
                continue

    def _read_anchor_world_pos(self, anchor_link: str, cached_view=None):
        """Read live anchor position from its PhysX rigid view as a CUDA tensor (3,).
        Require RigidBodyAPI before creating the view to avoid invalid-prim errors;
        return None on failure."""
        try:
            import torch  # noqa: PLC0415
            from sim_backend import backend as _bk  # noqa: PLC0415
        except Exception:
            return None

        # USD schema precheck.
        try:
            stage = omni.usd.get_context().get_stage()
            _prim = stage.GetPrimAtPath(anchor_link) if stage else None
            if not (_prim and _prim.IsValid() and _prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                return None
        except Exception:
            return None

        rp = cached_view
        if rp is None:
            try:
                from isaacsim.core.prims import SingleRigidPrim  # noqa: PLC0415
                rp = SingleRigidPrim(prim_path=anchor_link)
            except Exception as exc:
                logger.warning(f"[pin] Failed to create SingleRigidPrim({anchor_link}): {exc}")
                return None
        try:
            poss, _oris = rp._rigid_prim_view.get_world_poses(usd=False)
            # Positions may be CUDA tensors or numpy arrays depending on the pipeline.
            # FEM pin servos use CUDA, so convert numpy values to CUDA tensors.
            p0 = poss[0]
            if hasattr(p0, "device"):
                return p0.to("cuda:0").float()
            arr = _bk.to_numpy(p0)
            return torch.tensor(arr, device="cuda:0", dtype=torch.float32)
        except Exception as exc:
            logger.warning(f"[pin] Failed to read {anchor_link} world pose: {exc}")
            return None

    def callback_physics(self):
        with self._timing_context("callback_physics:publish_ros"):
            if self.publish_ros:
                if len(self.ros_publishers_temporary) > 0:
                    self.recording_times[-1][1] = self.get_current_simulation_time()
                # Tick both persistent and temporary nodes
                for ros_publisher_node in self.ros_publishers_persistent:
                    ros_publisher_node.tick(self.get_current_simulation_time())
                for ros_publisher_node in self.ros_publishers_temporary:
                    ros_publisher_node.tick(self.get_current_simulation_time())
        # Sample every physics callback on the state/camera simulation clock.
        # Each recorder only stores (prefix, revision) changes, so this remains
        # linear in generated geometry rather than physics-frame count.
        stage = omni.usd.get_context().get_stage()
        if stage:
            self._generated_geometry.sample(
                stage, self.get_current_simulation_time()
            )
        if self.publish_ros:
            return

    def _cloth_record_tick(self):
        """Record cloth at the requested FPS and append to the independent cloth frame list.
        Write _cloth_frames.json on stop for merging into final state.json."""
        if not self.path_to_save:
            return
        # With ROS, temporary publishers indicate active recording.
        # Without ROS, inspect whether the latest recording interval is active.
        if self.publish_ros and not self.ros_publishers_temporary:
            return
        cloth_views = getattr(self.ui_builder, "cloth_views", None) or {}
        if not cloth_views:
            return
        fps = float(getattr(self, "fps", 0.0) or 0.0)
        if fps <= 0:
            fps = 30.0
        now = self.get_current_simulation_time()
        last = getattr(self, "_cloth_last_record_time", None)
        if last is not None and (now - last) < (1.0 / fps - 1e-6):
            return
        self._cloth_last_record_time = now
        cloth_entries = self._snapshot_cloth_states(now)
        if not cloth_entries:
            return
        # Store file paths relative to deformable/ so replay resolves them
        # as <recording_dir>/deformable/<entry.file>.
        rel_entries = []
        for e in cloth_entries:
            rel_entries.append({
                "name": e["name"],
                "prim_path": e["prim_path"],
                "file": e["file"][len("deformable/"):] if e["file"].startswith("deformable/") else e["file"],
                "n_nodes": e["n_nodes"],
            })
        self._cloth_frames.append({
            "time_stamp": now,
            "deformable": rel_entries,
        })

    def get_current_simulation_time(self):
        return self.ui_builder.my_world.current_time

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
            info_key = info.split(":")[-1]
            camera_intrinsic_info[info_key] = str(value)
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
            ] = physics.get_joint_positions(articulation_obj, as_torch=False)
            state_info["articulation_objects"][prim_path][
                "joint_velocities"
            ] = physics.get_joint_velocities(articulation_obj, as_torch=False)
        articulation = self._initialize_articulation()
        state_info["articulation"]["joint_positions"] = physics.get_joint_positions(articulation, as_torch=False)
        state_info["articulation"]["joint_velocities"] = physics.get_joint_velocities(articulation, as_torch=False)

        # curobo
        state_info["attach_states"] = self.attach_states.copy()

        # Append-only generated geometry is restored by source-keyed prefixes
        # plus producer continuity state.
        state_info["generated_geometry"] = self._generated_geometry.capture_checkpoints(stage)

        # timestamp
        state_info["timestamp"] = self.get_current_simulation_time()

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
        state_info["target_point"] = self.target_point.copy() if self.target_point is not None else None
        return state_info

    def restore_state(self, state_info):
        MAX_PLAYBACK_WAITED_FRAME_NUM = 20
        self.playback_waited_frame_num += 1
        if not state_info:
            return False
        # Pin rigid bodies / articulation to the snapshot on EVERY wait frame so
        # detached objects can't drift under gravity or residual velocity during
        # the wait window — that drift is what leaks into the frame just past
        # playback_timerange.end before the attach constraint settles.
        stage = omni.usd.get_context().get_stage()
        if not stage:
            logger.info("stage not found")
            return False
        if self.playback_waited_frame_num == 1:
            self._generated_geometry.begin_restore(
                stage, state_info.get("generated_geometry")
            )
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
            # SingleRigidPrim velocity setters can be no-ops on the GPU pipeline.
            # Use the view combined 6D set_velocities call for torch.
            # Keep the existing individual setters for CPU physics
            # to preserve its established behavior.
            if sim_backend.tensors_as() == "torch":
                physics.set_rigid_velocity(
                    rigidbody,
                    linear=dynamic_state.linear_velocity,
                    angular=dynamic_state.angular_velocity,
                )
            else:
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
            physics.set_joint_positions(articulation_obj, articulation_obj_info["joint_positions"])
            physics.set_joint_velocities(articulation_obj, articulation_obj_info["joint_velocities"])

        articulation = self._initialize_articulation()
        physics.set_joint_positions(articulation, state_info["articulation"]["joint_positions"])
        physics.set_joint_velocities(articulation, state_info["articulation"]["joint_velocities"])
        # other local variables
        self.target_joints_pose = state_info["target_joints_pose"]
        self.target_position = state_info["target_position"]
        self.target_rotation = state_info["target_rotation"]
        self.gripper_state = state_info["gripper_state"]
        self._reset_stiffness()
        self._set_gripper_state(state_info["gripper_state_R"], True, 0.8) if state_info["gripper_state_R"] in ["open", "close"] else None
        self._set_gripper_state(state_info["gripper_state_L"], False, 0.8) if state_info["gripper_state_L"] in ["open", "close"] else None
        self.ros_step = state_info["ros_step"]
        self.target_point = state_info["target_point"]
        # curobo related — detach on frame 1, reattach on the final wait frame
        if state_info["attach_states"] and state_info["attach_states"] != self.attach_states:
            if self.playback_waited_frame_num == 1:
                self.ui_builder.detach_objs()
            elif self.playback_waited_frame_num >= MAX_PLAYBACK_WAITED_FRAME_NUM:
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
            # Do not let the live writer resume while robot/attachments are
            # still settling onto the checkpoint pose.
            self._generated_geometry.finish_restore(stage)
            # timestamp
            start_timstamp = state_info["timestamp"]
            end_timstamp = self.get_current_simulation_time()
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
        # Slowdown defaults to 1 when omitted or zero; fractional values below 1 clamp to 1.
        # Slowdown is applied by trajectory resampling rather than planner time dilation.
        slowdown_factor = max(1.0, float(self.data.get("slowdown_factor", 0.0) or 1.0))
        is_Right = False
        if self.data["isArmRight"]:
            is_Right = True
        if not is_backend:
            self.ui_builder.rmp_flow = False
            if (
                np.linalg.norm(self.target_position - target_position) != 0.0
                or np.linalg.norm(self.target_rotation - target_rotation) != 0.0
                or self.ui_builder.get_curobo_motion(is_Right).success is False
                or from_current_pose != self._last_from_current_pose
                or list(goal_offset) != list(self._last_goal_offset)
            ):
                self.motion_run_ratio = self.data.get("motion_run_ratio", 1.0)
                self.gripper_action_timing = self.data.get("gripper_action_timing", None)
                self.target_position = target_position
                self.target_rotation = target_rotation
                self._last_from_current_pose = from_current_pose
                self._last_goal_offset = list(goal_offset)
                # Combine user slowdown with automatic physics-timestep alignment
                # using _resolve_slowdown_factor and store it on curobo_motion.
                # After selecting a plan, calculate_ik_goal resamples joint positions.
                # Do not use plan_config.time_dilation_factor: batch planning cannot
                # call result.retime_trajectory, which supports only single results
                # and reports an error for plan_batch.
                # Reset the pending factor to 1 after each planning call.
                _curobo = self.ui_builder.get_curobo_motion(is_Right)
                _curobo.pending_slowdown_factor = self._resolve_slowdown_factor(slowdown_factor)
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
            self.ui_builder._check_and_update_kinematics_solver()
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
        is_reached = self.gripper_R.check_gripper_state() if isRight else self.gripper_L.check_gripper_state()
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
                prim_name = "hand_left"
            elif "right_camera" in prim_name.lower() and "head" not in prim_name.lower():
                prim_name = "hand_right"
        if "ARX" in self.robot_name:
            if "head" in prim_name.lower():
                prim_name = "head"
            elif "right" in prim_name.lower():
                prim_name = "hand_right"
            elif "left" in prim_name.lower():
                prim_name = "hand_left"
        if "Aloha" in self.robot_name:
            if "head" in prim_name.lower():
                prim_name = "head"
            elif "right" in prim_name.lower():
                prim_name = "hand_right"
            elif "left" in prim_name.lower():
                prim_name = "hand_left"
        return prim_name

    def handle_get_observation(self):
        """Handle Command 11: GetObservation / StartRecording / StopRecording"""
        if self.data["startRecording"]:
            with self._timing_context("start_recording"):
                self.recording_times.append([self.get_current_simulation_time(), self.get_current_simulation_time()])
                self.task_name = sanitize_task_name(self.data["task_name"])
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
                stage = omni.usd.get_context().get_stage()
                if stage:
                    self._generated_geometry.start_recording(
                        stage,
                        self.path_to_save,
                        self.get_current_simulation_time(),
                    )
                # Reset FEM counters and wait for any executor left from the previous recording.
                # Clear frame indices and the last timestamp so the first frame captures immediately.
                self._wait_cloth_recording_done()
                self._cloth_record_counter = 0
                self._cloth_last_record_time = None
                self._cloth_frames = []
                # Scan deformable views before recording starts, even if the task never invokes
                # FEM pin or nodal-position commands that would otherwise trigger lazy loading.
                try:
                    self.ui_builder.init_cloth_views()
                except Exception as exc:
                    logger.warning(f"[cloth] init_cloth_views on start_recording failed: {exc}")
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
                    if "fisheye" not in prim_name and self.output_depth:
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
                    # Fixed script, zero interpolation: $1 sources the ROS env,
                    # then "$@" carries the ros2 args as data. recording_path is
                    # a separate argv element, so it can never be parsed as shell.
                    record_script = (
                        "unset PYTHONPATH\n"
                        "unset LD_LIBRARY_PATH\n"
                        'source /opt/ros/"$1"/setup.bash\n'
                        "shift\n"
                        'exec ros2 bag record "$@"\n'
                    )
                    record_argv = [
                        "/bin/bash",
                        "-c",
                        record_script,
                        "bash",
                        ros_cmd_distro,
                        "-o",
                        recording_path,
                        exclude_args,
                        ".*_rgb(?!_)",
                        "-a",
                    ]
                    logger.info("publish_ros argv: %s", record_argv)
                    process = subprocess.Popen(
                        record_argv,
                        preexec_fn=os.setsid,
                    )
                    self.process.append(process)
                    # ROS publication cadence is fixed at 30 fps, independent of task_info fps.
                    # Compute step_size from physics frequency divided by 30
                    # (two steps at 60 Hz, four at 120 Hz). Warn when a different requested fps
                    # could misalign topic publication and the recorded timeline.
                    _RECORDING_FPS = 30
                    if self.data["fps"] != _RECORDING_FPS:
                        logger.warning(
                            f"task_info['fps']={self.data['fps']} != {_RECORDING_FPS}; "
                            f"ROS publish step_size still uses {_RECORDING_FPS}Hz; the recording timeline may be misaligned. "
                            f"For other frame rates, update the step_size calculation."
                        )
                    frequency = (int)(
                        1 / (self.ui_builder.my_world.get_physics_dt() * _RECORDING_FPS)
                    )  # step_size = physics_step / 30
                    logger.info(
                        f"frequency: {frequency}, fps: {self.data['fps']}, physics_dt: {self.ui_builder.my_world.get_physics_dt()}"
                    )
                    additional_cam_parameters = self.data.get("additional_cam_parameters", {})
                    if additional_cam_parameters:
                        additional_cam_parameters = json.loads(additional_cam_parameters)
                    noised_probability = additional_cam_parameters.get("noised_probability", 0.0)
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

                    # Clear temporary nodes at the start of recording
                    self._clear_temporary_publishers()

                    for prim in self.object_asset_dict.keys():
                        if prim not in tf_target:
                            tf_target.append(prim)
                        type_prim = get_prim_object_type(prim)
                        if type_prim == "articulation":
                            stage = omni.usd.get_context().get_stage()
                            for child in Usd.PrimRange(stage.GetPrimAtPath(prim)):
                                rigid_body_api = UsdPhysics.RigidBodyAPI.Get(stage, child.GetPath())
                                if rigid_body_api:
                                    tf_target.append(child.GetPath())
                    for prim in self.end_effector_prim_path.values():
                        if prim not in tf_target:
                            tf_target.append(prim)
                    for prim in self.end_effector_center_prim_path.values():
                        if prim not in tf_target:
                            tf_target.append(prim)
                    tf_target.append(self.robot_prim_path)
                    if self.arm_base_prim_path not in tf_target:
                        tf_target.append(self.arm_base_prim_path)
                    logger.info(f"tf_target{tf_target}")

                    # Create temporary tf node
                    tf_node = self.sensor_base.publish_tf(
                        robot_prim=self.robot_prim_path,
                        targets=tf_target,
                        step_size=frequency,
                        topic_name="/tf",
                    )
                    self.ros_publishers_temporary.append(tf_node)

                    # Create temporary joint_state nodes for articulat_objects
                    for prim in self.articulat_objects.keys():
                        joint_node = self.sensor_base.publish_joint(
                            robot_prim=prim,
                            robot=self.articulat_objects[prim],
                            step_size=frequency,
                            topic_name=prim,
                        )
                        self.ros_publishers_temporary.append(joint_node)

                    self.fps = self.data["fps"]
                    self.graph_path = [
                        "/World/RobotTFActionGraph",
                        "/World/RobotJointActionGraph",
                        "/ClockActionGraph",
                    ]

                    # Persistent nodes (only create once)
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
                                if noised_probability > 0.0:
                                    logger.error("[WARN] NOISE IS SHUT OFF, PLEASE CHANGE CODE IF NEEDED")
                                camera_param["noised"] = False
                                camera_param["noise_parameters"] = noise_parameters
                            else:
                                camera_param["noised"] = False
                            if camera_param["noised"]:
                                logger.info(f"noised camera: {camera}")
                            if "head_right_Camera" in camera or "head_left_Camera" in camera or not self.output_depth:
                                camera_param["publish"].remove("depth:/" + camera.split("/")[-1])
                            self.camera_info_list[self.get_camera_prim_name(camera)]["noised"] = (
                                camera_param["noised"]
                            )
                            if self.enable_render:
                                camera_graph, ros_nodes = self.sensor_base._init_camera(camera_param)
                                self.ros_publishers_persistent += ros_nodes
                        self.camera_graph_path.append(self.data["camera_prim_list"])
                        # joint action
                        articulation_action_node = self.sensor_base.publish_articulation_action(
                            robot=self.robot,
                            step_size=frequency,
                        )
                        self.ros_publishers_persistent.append(articulation_action_node)
                        # Main robot joint_state (persistent)
                        joint_node = self.sensor_base.publish_joint(
                            robot_prim=self.robot_prim_path,
                            robot=self.robot,
                            step_size=frequency,
                            topic_name="/joint_states",
                        )
                        self.ros_publishers_persistent.append(joint_node)

                    # Align all publication phases after creating the nodes.
                    # Temporary TF nodes start with zero counters each recording;
                    # persistent joint-state, action, and camera nodes retain their counters.
                    # Without a reset, the phase depends on the physical steps between recordings
                    # and can differ by one step in later rounds.
                    # At 60 Hz physics and 30 fps, a step is about 16.7 ms.
                    # An offset pairs TF object poses with the wrong joint-state frame during extraction,
                    # making objects lag the gripper in replay.
                    # The displacement increases with gripper speed.
                    self._reset_publisher_phase()

                    if self.enable_render:
                        for camera in self.data["camera_prim_list"]:
                            logger.info(f"republish camera{camera}")
                            topic_name = "/" + camera.split("/")[-1] + "_rgb"
                            compressed_name = topic_name + "_compressed"
                            ros_cmd_distro = os.getenv("ROS_CMD_DISTRO", "humble")
                            extra_args = (
                                ["--remap", "_out_transport:=compressed"]
                                if ros_cmd_distro != "humble"
                                else []
                            )
                            # Fixed script + argv: topic_name / compressed_name
                            # ride in as data via "$@", so a crafted camera prim
                            # name cannot inject commands.
                            run_script = (
                                "unset PYTHONPATH\n"
                                "unset LD_LIBRARY_PATH\n"
                                'source /opt/ros/"$1"/setup.bash\n'
                                "shift\n"
                                'exec ros2 run image_transport republish raw compressed "$@"\n'
                            )
                            run_argv = [
                                "/bin/bash",
                                "-c",
                                run_script,
                                "bash",
                                ros_cmd_distro,
                                *extra_args,
                                "--ros-args",
                                "--remap",
                                f"/in:={topic_name}",
                                "--remap",
                                f"/out:={compressed_name}",
                            ]
                            logger.info("republish argv: %s", run_argv)
                            subpro = subprocess.Popen(
                                run_argv,
                                preexec_fn=os.setsid,
                            )
                            self.process.append(subpro)
                    if not self.ros_node_initialized:
                        from common.base_utils.ros_nodes.server_node import ServerNode  # noqa: PLC0415
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
                    self._clear_temporary_publishers()
                    self.process = []
                    stage = omni.usd.get_context().get_stage()
                    if stage:
                        self._generated_geometry.stop_recording(
                            stage,
                            self.get_current_simulation_time(),
                        )
                    # copy usda scene directory if llm_mode
                    if self.llm_gen_mode:
                        # Get the scene.usda parent directory name.
                        scene_dir = os.path.dirname(self.scene_usd_path)
                        scene_dir_name = os.path.basename(scene_dir)
                        # Destination: path_to_save/scene_dir_name.
                        target_dir = os.path.join(self.path_to_save, scene_dir_name)
                        # Copy the entire scene directory.
                        if os.path.exists(target_dir):
                            shutil.rmtree(target_dir)
                        shutil.copytree(scene_dir, target_dir)
                        logger.info(f"Copied scene directory from {scene_dir} to {target_dir}")
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
        is_success = self.data["isSuccess"]
        config = self.task_metric
        keep_on_failure = os.environ.get("RUN_MODE", "") == "trial"
        if self.task_name is not None:
            if is_success is True or keep_on_failure:
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
                # Wait for all asynchronous cloth NPZ writes, then write the per-frame
                # index to _cloth_frames.json. extract_ros_bag.merge_state_json merges
                # it into each final state.json frame by timestamp
                # and removes the intermediate file.
                self._wait_cloth_recording_done()
                if self._cloth_frames:
                    with open(
                        self.path_to_save + "/_cloth_frames.json",
                        "w", encoding="utf-8",
                    ) as f:
                        json.dump({"frames": self._cloth_frames}, f, indent=2, ensure_ascii=False)
                    logger.info(
                        f"[cloth_record] wrote {len(self._cloth_frames)} frames to "
                        f"{self.path_to_save}/_cloth_frames.json (will be merged into state.json by extract)"
                    )
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
                    for p, log_file, rec_path in self.extract_process:
                        if p.poll() is None:
                            new_process.append((p, log_file, rec_path))
                        else:
                            self._print_last_log_lines(log_file)
                            self._check_extract_result(rec_path)
                    self.extract_process = new_process
                    time.sleep(0.1)
                    total_time += 0.1
                    if total_time > 180:
                        process, log_file, _ = self.extract_process[0]
                        os.killpg(os.getpgid(process.pid), signal.SIGINT)
                        logger.info("Extract process waiting timeout 120s, kill it")
                        self._print_last_log_lines(log_file)

                log_file = open(self.path_to_save + "/extract.log", "w")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                extract_cmd = [
                    sys.executable,
                    f"{current_dir}/recording/extract_and_convert_data.py",
                    "--path_to_save",
                    self.path_to_save,
                    "--task_info_path",
                    task_info_path,
                    "--metric_config_path",
                    metric_config_path,
                    "--with_img",
                    str(self.enable_render),
                    "--with_video",
                    str(self.enable_render),
                ]
                if not is_success:
                    extract_cmd.append("--stage_failed")
                extract_sub_process = subprocess.Popen(
                    extract_cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                )
                logger.info("Extract process started")
                self.extract_process.append((extract_sub_process, log_file, self.path_to_save))
            else:
                # remove folder if exist
                if os.path.exists(self.path_to_save):
                    shutil.rmtree(self.path_to_save)

        self.object_asset_dict = {}
        self.data_to_send = str(is_success)

    def _print_last_log_lines(self, log_file, lines=50):
        """Read and print the last N log lines, or all lines if fewer exist."""
        try:
            # Flush the log buffer and obtain its file path.
            log_file.flush()
            log_path = log_file.name
            log_file.close()

            # Reopen the file for reading.
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    all_lines = f.readlines()
                    # Read the last N lines.
                    last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                    if last_lines:
                        logger.info(f"Extract process log (last {len(last_lines)} lines):")
                        for line in last_lines:
                            # Remove trailing newlines; logger.info supplies its own.
                            logger.info(line.rstrip('\n\r'))
        except Exception as e:
            logger.warning(f"Failed to read log file: {e}")

    def _check_extract_result(self, rec_path):
        if not rec_path or not os.path.exists(rec_path):
            return
        state_file = os.path.join(rec_path, "state.json")
        if not os.path.exists(state_file):
            logger.warning(
                f"Extract failed: state.json not found in {rec_path}, removing directory"
            )
            shutil.rmtree(rec_path)

    def handle_exit(self):
        """Handle Command 17: Exit"""
        # wait for extract process to finish
        for process, log_file, rec_path in self.extract_process:
            try:
                if process.poll() is None:
                    process.wait(timeout=300)
                self._print_last_log_lines(log_file)
                self._check_extract_result(rec_path)
            except subprocess.TimeoutExpired:
                logger.info("Exit:Extract process waiting timeout 300s, kill it")
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                self._print_last_log_lines(log_file)
        self.exit = self.data["exit"]
        self.data_to_send = "exit"

    def handle_get_ee_pose(self):
        """Handle Command 18: GetEEPose"""
        is_right = self.data["isRight"]
        self.data_to_send = self._get_ee_pose(is_right)

    def handle_get_ik_status(self):
        """Handle Command 19: GetIKStatus"""
        before_check = time.time()
        self.ui_builder._check_and_update_kinematics_solver()
        after_check = time.time()
        logger.info(f"check and update kinematics sovler: {after_check - before_check} s")
        target_poses = np.array(self.data["target_poses"])
        ik_result = []
        is_Right = self.data["isRight"]
        ObsAvoid = self.data["ObsAvoid"]
        output_link_pose = self.data["output_link_pose"]
        ik_result = self._get_ik_status(target_poses, is_Right, ObsAvoid, output_link_pose)
        self.data_to_send = ik_result

    def handle_init_robot(self):
        """Handle Command 21: InitRobot"""
        # robot_cfg_file / scene_usd_path arrive over the unauthenticated gRPC
        # channel and are turned into filesystem paths that get opened / loaded.
        # Contain them before use (robot_cfg under config/robot_cfg, scene_usd
        # under SIM_ASSETS or an intentional absolute llm_gen path).
        current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        robot_config_dir = os.path.join(current_directory, "config/robot_cfg")
        robot_cfg_file = validate_robot_cfg_file(self.data["robot_cfg_file"], robot_config_dir)
        scene_usd_path = validate_scene_usd_path(self.data["scene_usd_path"], self.sim_assets_root)
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
        time_stamp = self.get_current_simulation_time()
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

    def handle_deactivate_object(self):
        """Handle Command 57: DeactivateObject"""
        prim_path = self.data["object_prim_path"]
        result = self._deactivate_object(prim_path)
        self.data_to_send = result

    def handle_run_step(self):
        """Handle Command 60: RunStep"""
        # Run on_command_step the requested number of times, then report success.
        steps = int(self.data.get("steps", 0))
        if steps <= 0:
            self.run_step_remaining = 0
            self.run_step_total = 0
            self.data_to_send = "success"
            return

        # Initialize on first entry, then decrement once per step.
        if self.run_step_remaining == 0:
            self.run_step_total = steps
            self.run_step_remaining = steps

        # Count each on_command_step invocation as one step.
        self.run_step_remaining -= 1
        if self.run_step_remaining <= 0:
            self.run_step_remaining = 0
            self.data_to_send = "success"
        else:
            self.data_to_send = None

    def handle_command_any(self):
        """Handle Command 99: CommandAny"""
        command_config = self.data["command_config"]
        if command_config["command"] == "GetRevoluteJointTrajectory":
            self.Command = Command.GET_REVOLUTE_JOINT_TRAJECTORY
            return self.on_get_revolute_joint_trajectory()
        elif command_config["command"] == "ControlGripper":
            self.Command = Command.CONTROL_GRIPPER
            return self.handle_control_gripper()
        elif command_config["command"] == "FollowTrajectory":
            self.Command = Command.FOLLOW_TRAJECTORY
            return self.on_follow_trajectory()
        elif command_config["command"] == "GetArticulationJointInfo":
            self.Command = Command.GET_ARTICULATION_JOINT_INFO
            return self.on_get_articulation_joint_info()
        elif command_config["command"] == "BimanualRigidMove":
            self.Command = Command.BIMANUAL_RIGID_MOVE
            return self.on_bimanual_rigid_move()
        elif command_config["command"] == "AddObjsToObstacle":
            return self.on_add_objs_to_obstacle()
        elif command_config["command"] == "SetObjectParameter":
            return self.on_set_object_parameter()
        elif command_config["command"] == "DualLinearMove":
            self.Command = Command.DUAL_LINEAR_MOVE
            return self.handle_dual_linear_move()
        elif command_config["command"] == "DualCuroboMove":
            self.Command = Command.DUAL_CUROBO_MOVE
            return self.handle_dual_curobo_move()
        elif command_config["command"] == "FEMPinAttach":
            return self.handle_fem_pin_attach()
        elif command_config["command"] == "FEMPinRelease":
            return self.handle_fem_pin_release()
        elif command_config["command"] == "RigidPinAttach":
            return self.handle_rigid_pin_attach()
        elif command_config["command"] == "RigidPinRelease":
            return self.handle_rigid_pin_release()
        elif command_config["command"] == "GetClothNodalPositions":
            return self.handle_get_cloth_nodal_positions()
        elif command_config["command"] == "SpawnDebugCube":
            return self.handle_spawn_debug_cube()
        elif command_config["command"] == "ClearDebugCubes":
            return self.handle_clear_debug_cubes()
        elif command_config["command"] == "SetGripperColliders":
            return self.handle_set_gripper_colliders()
        elif command_config["command"] == "GetPhysicsDt":
            return self.handle_get_physics_dt()
        elif command_config["command"] == "create_fix_joint":
            return self.handle_create_fix_joint()
        elif command_config["command"] == "get_fix_joints":
            return self.handle_get_fix_joints()

    def handle_control_gripper(self):
        """Independent, simulation-stepped opt-in command; no legacy gripper state writes."""
        from common.gripper_control import make_gripper_command
        from server.controllers.gripper_control import GripperControlSession
        from isaacsim.core.utils.types import ArticulationAction

        session = getattr(self, "_gripper_control_session", None)
        try:
            if session is None:
                options = dict(self.data["command_config"])
                options.pop("command")
                config = make_gripper_command(**options)
                robot = self._init_grippers()
                gripper = self.gripper_R if config["arm"] == "right" else self.gripper_L
                session = GripperControlSession(robot, gripper, physics, ArticulationAction,
                    config, self.ui_builder.my_world.current_time)
                self._gripper_control_session = session
            result = session.step(self.ui_builder.my_world.current_time)
            if result is not None:
                self._gripper_control_session = None
                self.data_to_send = result
        except Exception as exc:
            self._gripper_control_session = None
            self.data_to_send = {"success": False, "reason": "error", "error": str(exc)}
            if session is not None:
                try:
                    session.abort()
                except Exception as cleanup_exc:
                    self.data_to_send["cleanup_error"] = str(cleanup_exc)

    def handle_create_fix_joint(self):
        """Create a FixedJoint between two rigid bodies at their current poses."""
        import omni.usd
        from server.fixed_joint import create_fix_joint

        config = self.data["command_config"]
        try:
            self.data_to_send = create_fix_joint(
                omni.usd.get_context().get_stage(),
                config["body0_path"], config["body1_path"], config["joint_path"],
                config.get("collision_enabled", True),
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.data_to_send = {"success": False, "error": str(exc)}

    def handle_get_fix_joints(self):
        import omni.usd
        from server.fixed_joint import get_fix_joints

        self.data_to_send = get_fix_joints(omni.usd.get_context().get_stage())

    def handle_get_physics_dt(self):
        """Return World physics_dt so clients can convert recording delays to step counts."""
        self.data_to_send = {"physics_dt": float(self.ui_builder.my_world.get_physics_dt())}

    def handle_dual_linear_move(self):
        """Handle simultaneous dual-arm linear move (backend/trajectory mode).

        Expected command_config format:
        {
            "command": "DualLinearMove",
            "moves": [
                {
                    "arm": "left" | "right",
                    "target_position": [x, y, z],
                    "target_rotation": [w, x, y, z],
                    "ee_interpolation": bool,      # optional, default False
                    "distance_frame": float,       # optional, default 0.0008
                },
                ...  # second arm entry
            ]
        }
        """
        command_config = self.data["command_config"]
        moves = command_config["moves"]
        arm_keys = [m["arm"] for m in moves]

        if not getattr(self, "_dual_move_initialized", False):
            for move in moves:
                arm_key = move["arm"]
                target_position = np.array(move["target_position"])
                target_rotation = np.array(move["target_rotation"])
                ee_interpolation = move.get("ee_interpolation", False)
                distance_frame = move.get("distance_frame", 0.0008)
                self.ui_builder._trajectory_list_follow_target_arm(
                    target_position, target_rotation, arm_key,
                    ee_interpolation=ee_interpolation,
                    distance_frame=distance_frame,
                )
            self._dual_move_initialized = True

        if all(self.ui_builder.dual_reached.get(k, True) for k in arm_keys):
            self._dual_move_initialized = False
            self.data_to_send = {"success": True}

    def handle_dual_curobo_move(self):
        """Handle simultaneous cuRobo moves from command_config.moves.
        Each entry selects left/right arm, robot-local xyz target_position and wxyz
        target_rotation, with optional goal_offset, path_constraint, collision-link
        exclusions, from_current_pose, goal-frame constraints, and slowdown_factor."""
        command_config = self.data["command_config"]
        moves = command_config["moves"]

        # COMMAND_ANY normally does not permit simulation playback.
        # Without routing dual motion to a step-permitting command, plans can be consumed
        # without advancing physics, producing false reached flags.
        # handle_command_any routes it to DUAL_CUROBO_MOVE, registered in
        # allow_simulation_commands; the handler need not play World explicitly.
        if not getattr(self, "_dual_curobo_initialized", False):
            # Both arms can share a single v2 BatchMotionPlanner.
            # Two independent _hand_moveto calls would let the second calculation
            # overwrite the first planner target, active tool, and cmd_plan.
            # The resulting symptom is only one moving arm
            # in a synchronized two-arm manipulation.
            #
            # When planners are shared, combine both goals into one calculate_ik_goal call.
            # Use the first move as the active tool and supply the second
            # as an additional goal instead of holding its current FK pose.
            _planner_to_moves: Dict[int, List[dict]] = {}
            for move in moves:
                is_right = move["arm"] == "right"
                _p = self.ui_builder.get_curobo_motion(is_right)
                _planner_to_moves.setdefault(id(_p), []).append((move, _p, is_right))
            for _pid, _grouped in _planner_to_moves.items():
                if len(_grouped) == 1:
                    move, _curobo, is_right = _grouped[0]
                    _sf = max(1.0, float(move.get("slowdown_factor", 1.0) or 1.0))
                    _curobo.pending_slowdown_factor = self._resolve_slowdown_factor(_sf)
                    self._hand_moveto(
                        position=np.array(move["target_position"]),
                        rotation=np.array(move["target_rotation"]),
                        isRight=is_right,
                        goal_offset=move.get("goal_offset", [0, 0, 0, 1, 0, 0, 0]),
                        path_constraint=move.get("path_constraint", None),
                        offset_and_constraint_in_goal_frame=move.get(
                            "offset_and_constraint_in_goal_frame", True
                        ),
                        disable_collision_links=move.get("disable_collision_links", []),
                        from_current_pose=move.get("from_current_pose", False),
                    )
                else:
                    # Use one plan_pose call for shared planners to prevent overwriting.
                    # The primary arm follows the usual calculation path; the second arm
                    # becomes another target in goal_dict through additional_goals.
                    primary_move, _curobo, primary_is_right = _grouped[0]
                    other_move, _, other_is_right = _grouped[1]
                    # Read the second tool frame from the configured left/right end-effector mapping.
                    _ee_map = getattr(self.ui_builder, "end_effector_name", None)
                    if isinstance(_ee_map, dict):
                        other_tool = _ee_map.get("right" if other_is_right else "left")
                    else:
                        _tools = list(_curobo.planner.tool_frames)
                        primary_tool = (_ee_map if isinstance(_ee_map, str) else
                                        _curobo.ee_link_name or _tools[0])
                        other_tool = next((t for t in _tools if t != primary_tool), None)
                    if not other_tool:
                        logger.error(
                            "handle_dual_curobo_move: cannot resolve second tool_frame; "
                            "falling back to sequential path (may overwrite cmd_plan)"
                        )
                        for move, _cm, _ir in _grouped:
                            self._hand_moveto(
                                position=np.array(move["target_position"]),
                                rotation=np.array(move["target_rotation"]),
                                isRight=_ir,
                                goal_offset=move.get("goal_offset", [0, 0, 0, 1, 0, 0, 0]),
                                path_constraint=move.get("path_constraint", None),
                                offset_and_constraint_in_goal_frame=move.get(
                                    "offset_and_constraint_in_goal_frame", True
                                ),
                                disable_collision_links=move.get("disable_collision_links", []),
                                from_current_pose=move.get("from_current_pose", False),
                            )
                        continue
                    _sf = max(1.0, float(primary_move.get("slowdown_factor", 1.0) or 1.0))
                    _curobo.pending_slowdown_factor = self._resolve_slowdown_factor(_sf)
                    logger.info(
                        f"[dual_curobo] shared planner (id={_pid}), merging both arms: "
                        f"primary={'right' if primary_is_right else 'left'} "
                        f"other={other_tool}"
                    )
                    # As in _hand_moveto, set the following target marker values first,
                    # then invoke _follow_target with additional_goals.
                    primary_pos = np.array(primary_move["target_position"])
                    primary_rot = np.array(primary_move["target_rotation"])
                    other_pos = np.array(other_move["target_position"])
                    other_rot = np.array(other_move["target_rotation"])
                    self.ui_builder._followingPos = primary_pos
                    self.ui_builder._followingOrientation = primary_rot
                    self._initialize_articulation()
                    self.ui_builder._Generate_following_position(primary_is_right)
                    self.ui_builder.set_locked_joint_positions(primary_is_right)
                    _curobo.set_obstacles()
                    _curobo.calculate_ik_goal(
                        goal_offset=primary_move.get("goal_offset", [0, 0, 0, 1, 0, 0, 0]),
                        path_constraint=primary_move.get("path_constraint", None),
                        offset_and_constraint_in_goal_frame=primary_move.get(
                            "offset_and_constraint_in_goal_frame", True
                        ),
                        disable_collision_links=primary_move.get("disable_collision_links", []),
                        from_current_pose=primary_move.get("from_current_pose", False),
                        target_position=primary_pos,
                        target_orientation=primary_rot,
                        # Forward the second arm goal offset and path constraint independently;
                        # its goal_dict target uses its own tool criteria.
                        additional_goals={
                            other_tool: {
                                "target_position": other_pos,
                                "target_orientation": other_rot,
                                "goal_offset": other_move.get(
                                    "goal_offset", [0, 0, 0, 1, 0, 0, 0]
                                ),
                                "from_current_pose": other_move.get(
                                    "from_current_pose", False
                                ),
                                "path_constraint": other_move.get("path_constraint", None),
                                "offset_and_constraint_in_goal_frame": other_move.get(
                                    "offset_and_constraint_in_goal_frame", True
                                ),
                            }
                        },
                    )
                    # Both arms are active, so do not exclude the opposite arm joints.
            self._dual_curobo_initialized = True

        all_reached = all(
            self.ui_builder.get_curobo_motion(m["arm"] == "right").reached for m in moves
        )
        if all_reached:
            all_success = all(
                self.ui_builder.get_curobo_motion(m["arm"] == "right").success for m in moves
            )
            self._dual_curobo_initialized = False
            self.data_to_send = {"success": all_success}

    def on_add_objs_to_obstacle(self):
        """Handle AddObjsToObstacle via CommandAny"""
        command_config = self.data["command_config"]
        obj_prims = command_config["obj_prim_paths"]
        items = []
        stage = omni.usd.get_context().get_stage()
        if stage:
            for prim_path in obj_prims:
                for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                    path = str(prim.GetPath())
                    prim = get_prim_at_path(path)
                    if prim.IsA(UsdGeom.Mesh):
                        items.append(path)
        for path in items:
            usd_path = self.object_asset_dict.get(path)
            for curobo_motion in self.ui_builder.curoboMotion.values():
                if curobo_motion:
                    curobo_motion.add_obstacle_from_prim_path(path, usd_path)
        self.data_to_send = f"{obj_prims} added to obstacle"

    # ------------------------------------------------------------------
    # Generic FEM cloth pin and nodal-position interfaces.
    # Use surface-deformable prims and selected anchor link paths.
    # Client task code composes the manipulation sequence.
    # ------------------------------------------------------------------
    def _resolve_cloth_view(self, cloth_prim: str):
        """Lazily resolve a deformable view by cloth prim path; return None on failure."""
        if not getattr(self.ui_builder, "enable_deformable", False):
            logger.warning(
                "[cloth] Ignoring FEM command because the server was started without --enable_deformable."
            )
            return None
        # Lazy-load after physics steps have allowed FEM cooking to complete.
        self.ui_builder.init_cloth_views()
        view = self.ui_builder.cloth_views.get(cloth_prim)
        if view is None:
            logger.warning(
                f"[cloth] cloth_prim={cloth_prim} is not registered as a surface deformable body. "
                f"Registered: {list(self.ui_builder.cloth_views.keys())}"
            )
        return view

    def handle_fem_pin_attach(self):
        """Pin cloth FEM nodes near an anchor using velocity servos.
        command_config selects cloth_prim, anchor_link, radius (0.03), gain (3),
        vmax (3 m/s), optional group_id (defaults to anchor), and xy/3d distance.
        Return success, attached_count, group_id, and n_total."""
        cfg = self.data["command_config"]
        cloth_prim = cfg["cloth_prim"]
        anchor_link = cfg["anchor_link"]
        radius = float(cfg.get("radius", 0.03))
        gain = float(cfg.get("gain", 3.0))
        vmax = float(cfg.get("vmax", 3.0))
        axis = cfg.get("axis", "xy")
        group_id = cfg.get("group_id") or anchor_link

        view = self._resolve_cloth_view(cloth_prim)
        if view is None:
            self.data_to_send = {"success": False, "error": "cloth_view_unavailable"}
            return

        try:
            import torch  # noqa: PLC0415
        except Exception as exc:
            self.data_to_send = {"success": False, "error": f"torch_unavailable:{exc}"}
            return

        anchor_pos = self._read_anchor_world_pos(anchor_link)
        if anchor_pos is None:
            self.data_to_send = {"success": False, "error": "anchor_pose_unavailable"}
            return

        try:
            P = view.get_simulation_nodal_positions()  # (1, N, 3) torch.cuda
            n_total = int(P.shape[1])
            P0 = P[0]                                  # (N, 3)
            if axis == "3d":
                d = torch.linalg.vector_norm(P0 - anchor_pos.unsqueeze(0), dim=1)
            else:
                # Select nodes by xy-plane distance for planar cloth pinning.
                dxy = torch.sqrt(
                    (P0[:, 0] - anchor_pos[0]) ** 2 + (P0[:, 1] - anchor_pos[1]) ** 2
                )
                d = dxy
            mask = d < radius
            node_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1).to(torch.long)
            attached_count = int(node_idx.numel())
            if attached_count == 0:
                self.data_to_send = {
                    "success": False, "error": "no_node_in_radius",
                    "n_total": n_total, "anchor_world": anchor_pos.detach().cpu().tolist(),
                    "radius": radius,
                }
                return
            offset = (P0[node_idx] - anchor_pos.unsqueeze(0)).contiguous()  # (M, 3)
        except Exception as exc:
            logger.warning(f"[FEMPinAttach] Failed to compute nodes: {exc}")
            self.data_to_send = {"success": False, "error": f"node_query_failed:{exc}"}
            return

        # Cache the anchor rigid view for subsequent physics-step servo reads.
        try:
            from isaacsim.core.prims import SingleRigidPrim  # noqa: PLC0415
            anchor_view = SingleRigidPrim(prim_path=anchor_link)
        except Exception:
            anchor_view = None

        self._active_pin_groups[group_id] = {
            "cloth_prim":   cloth_prim,
            "anchor_link":  anchor_link,
            "node_idx":     node_idx,
            "offset":       offset,
            "gain":         gain,
            "vmax":         vmax,
            "anchor_view":  anchor_view,
        }
        logger.info(
            f"[FEMPinAttach] group={group_id} cloth={cloth_prim} anchor={anchor_link} "
            f"radius={radius} gain={gain} vmax={vmax} attached={attached_count}/{n_total}"
        )
        self.data_to_send = {
            "success": True,
            "attached_count": attached_count,
            "group_id": group_id,
            "n_total": n_total,
        }

    def handle_fem_pin_release(self):
        """Release the selected FEM pin group or all groups and zero node velocities.
        Return success and released group IDs."""
        cfg = self.data["command_config"]
        group_id = cfg.get("group_id", "all")
        if not self._active_pin_groups:
            self.data_to_send = {"success": True, "released": []}
            return

        targets = (list(self._active_pin_groups.keys()) if group_id == "all"
                   else [group_id] if group_id in self._active_pin_groups
                   else [])
        try:
            import torch  # noqa: PLC0415
        except Exception:
            torch = None

        for gid in targets:
            g = self._active_pin_groups[gid]
            view = self.ui_builder.cloth_views.get(g["cloth_prim"]) if hasattr(
                self.ui_builder, "cloth_views"
            ) else None
            if view is not None and torch is not None:
                try:
                    vel = view.get_simulation_nodal_velocities().clone()
                    vel[0, g["node_idx"]] = 0.0
                    view.set_simulation_nodal_velocities(vel, view._fem_body_idx_cache)
                except Exception as exc:
                    logger.warning(f"[FEMPinRelease] Failed to zero group {gid} velocities: {exc}")
            del self._active_pin_groups[gid]
            logger.info(f"[FEMPinRelease] group={gid} released")

        self.data_to_send = {"success": True, "released": targets}

    def handle_rigid_pin_attach(self):
        """Freeze a prim relative to an anchor, or in world coordinates without one.
        Support rigid bodies and other valid Xforms; nonrigid velocity pins use pose mode.
        command_config selects rigid_prim, optional anchor_link/group_id, pose/velocity
        mode, linear/angular gains, and speed caps. Return success, group_id, mode,
        and the frozen relative position/orientation."""
        cfg = self.data["command_config"]
        rigid_prim = cfg["rigid_prim"]
        anchor_link = cfg.get("anchor_link") or None
        mode = str(cfg.get("mode", "pose")).lower()
        if mode not in ("pose", "velocity"):
            self.data_to_send = {"success": False, "error": f"invalid_mode:{mode}"}
            return
        gain_lin = float(cfg.get("gain_lin", 3.0))
        gain_ang = float(cfg.get("gain_ang", 3.0))
        vmax_lin = float(cfg.get("vmax_lin", 3.0))
        vmax_ang = float(cfg.get("vmax_ang", 6.0))
        group_id = cfg.get("group_id") or rigid_prim

        try:
            import torch  # noqa: PLC0415
            from sim_backend import backend as _bk  # noqa: PLC0415
        except Exception as exc:
            self.data_to_send = {"success": False, "error": f"torch_unavailable:{exc}"}
            return

        # Allow any valid pinned prim: rigid bodies use SingleRigidPrim
        # for a true PhysX teleport through set_transforms;
        # other prims use SingleXFormPrim for USD/Fabric pose updates.
        # Velocity mode also retains a rigid view for physical velocity commands.
        stage = omni.usd.get_context().get_stage()
        rprim = stage.GetPrimAtPath(rigid_prim) if stage else None
        if not (rprim and rprim.IsValid()):
            self.data_to_send = {"success": False, "error": "rigid_prim_invalid"}
            return
        is_rigid_body = bool(rprim.HasAPI(UsdPhysics.RigidBodyAPI))
        kind = "rigid" if is_rigid_body else "xform"
        rigid_view = None  # Available only for rigid-body prims.
        try:
            if is_rigid_body:
                from isaacsim.core.prims import SingleRigidPrim  # noqa: PLC0415
                rigid_view = SingleRigidPrim(prim_path=rigid_prim)
                target_view = rigid_view  # Write poses through the rigid view to teleport PhysX.
            else:
                from isaacsim.core.prims import SingleXFormPrim  # noqa: PLC0415
                target_view = SingleXFormPrim(prim_path=rigid_prim)
        except Exception as exc:
            self.data_to_send = {"success": False, "error": f"target_view_failed:{exc}"}
            return

        anchor_view = None
        if anchor_link:
            aprim = stage.GetPrimAtPath(anchor_link) if stage else None
            if not (aprim and aprim.IsValid()):
                self.data_to_send = {"success": False, "error": "anchor_link_invalid"}
                return
            # Anchors may be ordinary Xforms or links without RigidBodyAPI.
            # Read them through physics.get_prim_world_pose during servo updates;
            # cache a SingleRigidPrim only for rigid anchors.
            if aprim.HasAPI(UsdPhysics.RigidBodyAPI):
                try:
                    from isaacsim.core.prims import SingleRigidPrim  # noqa: PLC0415
                    anchor_view = SingleRigidPrim(prim_path=anchor_link)
                except Exception as exc:
                    logger.warning(
                        f"[RigidPinAttach] SingleRigidPrim({anchor_link}) construction failed; falling back to prim_world_pose: {exc}"
                    )
                    anchor_view = None

        # Compute relative poses in float64 numpy to avoid mixed backend types.
        # Convert again when writing poses or velocities to the backend.
        def _read_pose_np(prim_path: str, view):
            """Read current position (3,) and wxyz quaternion (4,) as float64 numpy arrays."""
            try:
                if view is not None and hasattr(view, "_rigid_prim_view"):
                    p, q = view._rigid_prim_view.get_world_poses(usd=False)
                    return _bk.to_numpy(p[0]).astype(np.float64), \
                           _bk.to_numpy(q[0]).astype(np.float64)
                pose = physics.get_prim_world_pose(prim_path)
                if pose is None:
                    return None, None
                return np.asarray(pose[0], dtype=np.float64), \
                       np.asarray(pose[1], dtype=np.float64)
            except Exception:
                return None, None

        # Compute relative poses in float64 numpy to avoid mixed backend types.
        # Isaac Sim APIs handle conversion when writing world poses and velocities.
        def _read_pose_np(prim_path: str):
            try:
                pose = physics.get_prim_world_pose(prim_path)
                if pose is None:
                    return None, None
                return np.asarray(pose[0], dtype=np.float64), \
                       np.asarray(pose[1], dtype=np.float64)
            except Exception:
                return None, None

        try:
            p_r, q_r = _read_pose_np(rigid_prim)
            if p_r is None:
                self.data_to_send = {"success": False, "error": "rigid_pose_unreadable"}
                return
            if anchor_link is not None:
                a_pos, a_ori = self._read_anchor_pose(anchor_link, anchor_view)
                if a_pos is None:
                    self.data_to_send = {"success": False, "error": "anchor_pose_unreadable"}
                    return
                p_a = np.asarray(_bk.to_numpy(a_pos), dtype=np.float64)
                q_a = np.asarray(_bk.to_numpy(a_ori), dtype=np.float64)
                # Relative pose: p_rel = inverse(R_anchor) * (p_rigid - p_anchor).
                p_rel = self._np_quat_rotate(self._np_quat_conj(q_a), p_r - p_a)
                q_rel = self._np_quat_mul(self._np_quat_conj(q_a), q_r)
            else:
                # For a world lock, store the current world pose directly as p_rel/q_rel.
                p_rel = p_r.copy()
                q_rel = q_r.copy()
        except Exception as exc:
            self.data_to_send = {"success": False, "error": f"read_pose_failed:{exc}"}
            return

        # Disable rigid-body gravity while pinned so it cannot displace the object.
        # Nonrigid prims have no physical gravity and need no override.
        # Restore the previous gravity setting on release.
        gravity_prev = None
        gravity_disabled = False
        if is_rigid_body:
            ok, gravity_prev = self._set_disable_gravity(rigid_prim, disable=True)
            gravity_disabled = ok

        self._active_rigid_pin_groups[group_id] = {
            "rigid_prim":  rigid_prim,
            "anchor_link": anchor_link,
            "kind":        kind,
            "mode":        mode,
            "p_rel":       p_rel,
            "q_rel":       q_rel,
            "gain_lin":    gain_lin,
            "gain_ang":    gain_ang,
            "vmax_lin":    vmax_lin,
            "vmax_ang":    vmax_ang,
            "target_view": target_view,   # Pose wrapper used by set_world_pose.
            "rigid_view":  rigid_view,    # Rigid view, or None; used to inject velocities in velocity mode.
            "anchor_view": anchor_view,
            # Track whether attachment changed the gravity attribute.
            "gravity_disabled": gravity_disabled,
            "gravity_prev":     gravity_prev,
        }
        logger.info(
            f"[RigidPinAttach] group={group_id} rigid={rigid_prim} anchor={anchor_link} "
            f"mode={mode} gain_lin={gain_lin} gain_ang={gain_ang} "
            f"vmax_lin={vmax_lin} vmax_ang={vmax_ang} "
            f"gravity_disabled={gravity_disabled}(prev={gravity_prev})"
        )
        self.data_to_send = {
            "success": True,
            "group_id": group_id,
            "mode": mode,
            "kind": kind,
            "gravity_disabled": gravity_disabled,
            "p_rel": p_rel.tolist(),
            "q_rel": q_rel.tolist(),
        }

    def handle_rigid_pin_release(self):
        """Release one or all rigid pin groups, zero velocities, and return released IDs."""
        cfg = self.data["command_config"]
        group_id = cfg.get("group_id", "all")
        active_keys = list(self._active_rigid_pin_groups.keys())
        if not self._active_rigid_pin_groups:
            logger.info(f"[RigidPinRelease] no active groups (requested group_id={group_id!r})")
            self.data_to_send = {"success": True, "released": []}
            return

        if group_id == "all":
            targets = active_keys
        elif group_id in self._active_rigid_pin_groups:
            targets = [group_id]
        else:
            logger.warning(
                f"[RigidPinRelease] group_id={group_id!r} not in active groups {active_keys}; nothing released"
            )
            self.data_to_send = {"success": True, "released": [], "active_groups": active_keys}
            return
        for gid in targets:
            g = self._active_rigid_pin_groups[gid]
            rigid_view = g.get("rigid_view")
            if rigid_view is None and g.get("kind") == "rigid":
                # Try constructing a rigid view on release if needed to zero velocities.
                try:
                    from isaacsim.core.prims import SingleRigidPrim  # noqa: PLC0415
                    rigid_view = SingleRigidPrim(prim_path=g["rigid_prim"])
                except Exception:
                    rigid_view = None
            if rigid_view is not None:
                try:
                    physics.set_rigid_velocity(rigid_view, linear=[0, 0, 0], angular=[0, 0, 0])
                except Exception as exc:
                    logger.warning(f"[RigidPinRelease] Failed to zero group {gid} velocities: {exc}")
            # Restore any recorded gravity_prev value unconditionally for rigid bodies.
            # Do not rely on gravity_disabled, since partial attachment failures can
            # otherwise leave an inconsistent gravity state.
            gravity_restored = False
            if g.get("kind") == "rigid":
                ok, _ = self._restore_gravity(g["rigid_prim"], g.get("gravity_prev"))
                gravity_restored = ok
            del self._active_rigid_pin_groups[gid]
            logger.info(
                f"[RigidPinRelease] group={gid} released "
                f"(gravity_restored={gravity_restored}, remaining={len(self._active_rigid_pin_groups)})"
            )

        self.data_to_send = {"success": True, "released": targets}

    def handle_get_cloth_nodal_positions(self):
        """Read cloth FEM node world positions; optional indices select a subset.
        Return success, positions, and n_total."""
        cfg = self.data["command_config"]
        cloth_prim = cfg["cloth_prim"]
        indices = cfg.get("indices")

        view = self._resolve_cloth_view(cloth_prim)
        if view is None:
            self.data_to_send = {"success": False, "error": "cloth_view_unavailable"}
            return
        try:
            P = view.get_simulation_nodal_positions()  # (1, N, 3)
            P0 = P[0]
            n_total = int(P0.shape[0])
            if indices is None:
                positions = P0.detach().cpu().numpy().tolist()
            else:
                # Preserve explicit index order for client array-based access.
                import torch  # noqa: PLC0415
                idx_t = torch.tensor(indices, dtype=torch.long, device=P0.device)
                positions = P0[idx_t].detach().cpu().numpy().tolist()
            self.data_to_send = {
                "success": True,
                "positions": positions,
                "n_total": n_total,
            }
        except Exception as exc:
            logger.warning(f"[GetClothNodalPositions] Failed: {exc}")
            self.data_to_send = {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Debug visualization: create small cube markers for grasp poses
    # and FEM nodes through the client draw_debug_cube API.
    # ------------------------------------------------------------------
    def _debug_cube_root(self) -> str:
        return "/World/_DebugCubes"

    def handle_spawn_debug_cube(self):
        """Create or update a named cube marker with world xyz and wxyz orientation.
        Defaults are identity orientation, 0.02 m size, and red RGB color.
        Return success and prim_path."""
        cfg = self.data["command_config"]
        name = cfg.get("name", "marker")
        position = list(cfg.get("position", [0.0, 0.0, 0.0]))
        orientation = list(cfg.get("orientation", [1.0, 0.0, 0.0, 0.0]))
        size = float(cfg.get("size", 0.02))
        color = list(cfg.get("color", [1.0, 0.0, 0.0]))

        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(name))
        prim_path = f"{self._debug_cube_root()}/{safe_name}"
        try:
            from isaacsim.core.api.objects import VisualCuboid  # noqa: PLC0415
        except Exception as exc:
            self.data_to_send = {"success": False, "error": f"VisualCuboid import: {exc}"}
            return

        stage = omni.usd.get_context().get_stage()
        existed = stage and stage.GetPrimAtPath(prim_path).IsValid()
        try:
            if not existed:
                # Create the cube.
                VisualCuboid(
                    prim_path=prim_path,
                    name=safe_name,
                    position=np.asarray(position),
                    orientation=np.asarray(orientation),
                    size=size,
                    color=np.asarray(color),
                )
            else:
                # Update pose, scale, and color when the marker already exists.
                from isaacsim.core.prims import SingleXFormPrim  # noqa: PLC0415
                xp = SingleXFormPrim(prim_path=prim_path)
                xp.set_world_pose(
                    position=np.asarray(position),
                    orientation=np.asarray(orientation),
                )
            self.data_to_send = {"success": True, "prim_path": prim_path}
        except Exception as exc:
            logger.warning(f"[SpawnDebugCube] {prim_path} failed: {exc}")
            self.data_to_send = {"success": False, "error": str(exc)}

    def handle_clear_debug_cubes(self):
        """Delete all cube markers under _DebugCubes."""
        stage = omni.usd.get_context().get_stage()
        n = 0
        if stage:
            root = stage.GetPrimAtPath(self._debug_cube_root())
            if root.IsValid():
                for child in list(root.GetChildren()):
                    stage.RemovePrim(child.GetPath())
                    n += 1
        self.data_to_send = {"success": True, "removed": n}

    def handle_set_gripper_colliders(self):
        """Toggle gripper base/finger colliders for left, right, or both arms.
        Disable contacts during cloth pins and restore on release. Optional links
        selects suffixes; return success, toggled prim paths/states, and count."""
        cfg = self.data["command_config"]
        arm = cfg.get("arm", "both")
        enabled = bool(cfg.get("enabled", True))
        # Default to base plus six finger links; center/camera links have no meshes.
        default_link_suffixes = [
            "base_link",
            "left_inner_link", "left_outer_link", "left_support_link",
            "right_inner_link", "right_outer_link", "right_support_link",
        ]
        link_suffixes = cfg.get("links") or default_link_suffixes
        sides = ["l", "r"] if arm == "both" else (["l"] if arm == "left" else ["r"])

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self.data_to_send = {"success": False, "error": "stage_unavailable"}
            return

        toggled = []
        try:
            for side in sides:
                for suffix in link_suffixes:
                    link_name = f"gripper_{side}_{suffix}"
                    # Find articulation link prims by traversing paths with the robot prefix.
                    for prim in stage.Traverse():
                        if prim.GetName() != link_name:
                            continue
                        # Toggle CollisionAPI on meshes under the selected link.
                        for sub in Usd.PrimRange(prim):
                            if not sub.IsA(UsdGeom.Mesh):
                                continue
                            collision_api = UsdPhysics.CollisionAPI.Get(stage, sub.GetPath())
                            if not collision_api:
                                # Skip visual-only meshes without CollisionAPI.
                                continue
                            attr = collision_api.GetCollisionEnabledAttr()
                            if attr:
                                attr.Set(enabled)
                                toggled.append({
                                    "prim_path": str(sub.GetPath()),
                                    "enabled": enabled,
                                })
        except Exception as exc:
            logger.warning(f"[SetGripperColliders] Failed to toggle colliders: {exc}")
            self.data_to_send = {"success": False, "error": f"toggle_failed:{exc}"}
            return

        logger.info(
            f"[SetGripperColliders] arm={arm} enabled={enabled} toggled {len(toggled)} mesh"
        )
        self.data_to_send = {"success": True, "toggled": toggled, "n": len(toggled)}

    def on_set_object_parameter(self):
        """Handle SetObjectParameter via CommandAny.

        Applies a USD attribute on the given prim. Frame delays before/after
        are scheduled by the client via run_step.

        command_config keys:
            prim_path, attribute_key, attribute_value
            reference: "local" (default) or "world". When "world":
                - xformOp:translate is converted into the parent's local frame
                - xformOp:orient is converted via parent_world_rot.inverse() * R
        """
        command_config = self.data["command_config"]
        prim_path = command_config["prim_path"]
        attribute_key = command_config["attribute_key"]
        attribute_value = command_config["attribute_value"]
        reference = command_config.get("reference", "world")
        stage = omni.usd.get_context().get_stage()
        applied = False
        if stage:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                attr = prim.GetAttribute(attribute_key)
                if attr and attr.IsValid():
                    value = attribute_value
                    if reference == "world":
                        value = self._world_to_local_attr(
                            prim, attribute_key, value
                        )
                    attr.Set(self._coerce_usd_attribute_value(attr, value))
                    applied = True
                    # On GPU physics, authoring a USD attribute alone may not update PhysX.
                    # For transform attributes, also write the live pose through the
                    # SingleXFormPrim or SingleRigidPrim wrapper.
                    # The wrapper routes Xform updates appropriately and rigid updates
                    # directly to PhysX set_transforms.
                    # CPU physics already reads USD, so this extra route can trigger
                    # non-root articulation-link warnings and non-kinematic body errors.
                    # Enable the synchronization route only for torch backends.
                    if (
                        attribute_key in ("xformOp:translate", "xformOp:orient")
                        and sim_backend.tensors_as() == "torch"
                    ):
                        self._sync_transform_to_fabric(
                            prim_path, attribute_key, attribute_value, reference
                        )
                else:
                    logger.warning(
                        f"SetObjectParameter: attribute {attribute_key} not found on {prim_path}"
                    )
            else:
                logger.warning(f"SetObjectParameter: prim {prim_path} not found")
        self.data_to_send = "success" if applied else "failed"

    def _sync_transform_to_fabric(
        self, prim_path: str, attribute_key: str, raw_value, reference: str
    ) -> None:
        """Synchronize an authored transform with the live prim pose.
        Rigid bodies use SingleRigidPrim to teleport PhysX; other prims use the
        SingleXFormPrim wrapper. World values apply directly; local values are composed
        with the parent transform. Complex hierarchies should supply world values."""
        # Compute world-frame value
        if reference == "world":
            world_value = raw_value
        else:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            parent_xform = self._parent_world_transform(prim)
            if parent_xform is None:
                world_value = raw_value
            else:
                parent_mat = np.asarray(parent_xform).T
                if attribute_key == "xformOp:translate":
                    local_t = np.asarray([raw_value[0], raw_value[1], raw_value[2], 1.0])
                    world_t = parent_mat @ local_t
                    world_value = world_t[:3].tolist()
                else:
                    logger.warning(
                        "SetObjectParameter: Fabric synchronization for reference='local' + xformOp:orient "
                        "is incomplete; use reference='world' on the client. USD has been updated."
                    )
                    return

        # Detect rigid body via UsdPhysics.RigidBodyAPI
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        is_rigid = prim and prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI)

        if is_rigid:
            # RigidPrim set_world_poses teleports the PhysX body through set_transforms.
            # PhysX synchronizes Fabric and USD, with readback enabled
            # by init_app settings.
            rp = RigidPrim(prim_path=prim_path)
            if attribute_key == "xformOp:translate":
                rp.set_world_pose(position=np.asarray(world_value, dtype=np.float32))
            elif attribute_key == "xformOp:orient":
                rp.set_world_pose(orientation=np.asarray(world_value, dtype=np.float32))
        else:
            # Nonrigid prims use the XFormPrim wrapper.
            xfp = XFormPrim(prim_path=prim_path)
            if attribute_key == "xformOp:translate":
                xfp.set_world_pose(position=np.asarray(world_value, dtype=np.float32))
            elif attribute_key == "xformOp:orient":
                xfp.set_world_pose(orientation=np.asarray(world_value, dtype=np.float32))

    @staticmethod
    def _parent_world_transform(prim):
        parent = prim.GetParent()
        if not parent or not parent.IsValid():
            return None
        xformable = UsdGeom.Xformable(parent)
        if not xformable:
            return None
        return xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    @classmethod
    def _world_to_local_attr(cls, prim, attribute_key, world_value):
        """Convert a world-space xform value into the prim's local attribute."""
        parent_world = cls._parent_world_transform(prim)
        if parent_world is None:
            return world_value
        if attribute_key == "xformOp:translate":
            if not isinstance(world_value, (list, tuple)) or len(world_value) != 3:
                return world_value
            local_point = parent_world.GetInverse().Transform(
                Gf.Vec3d(float(world_value[0]), float(world_value[1]), float(world_value[2]))
            )
            return [local_point[0], local_point[1], local_point[2]]
        if attribute_key == "xformOp:orient":
            if not isinstance(world_value, (list, tuple)) or len(world_value) != 4:
                return world_value
            # Input convention: [w, x, y, z]
            world_quat = Gf.Quatd(
                float(world_value[0]),
                Gf.Vec3d(float(world_value[1]), float(world_value[2]), float(world_value[3])),
            )
            parent_rot = parent_world.ExtractRotationQuat().GetNormalized()
            local_quat = parent_rot.GetInverse() * world_quat
            imag = local_quat.GetImaginary()
            return [local_quat.GetReal(), imag[0], imag[1], imag[2]]
        return world_value

    @staticmethod
    def _coerce_usd_attribute_value(attr, value):
        """Coerce a JSON-RPC payload into the USD attribute's native type.

        JSON-RPC strips Gf/Sdf types down to lists/scalars; USD attr.Set
        requires the exact type (e.g. xformOp:translate expects GfVec3f).
        """
        if not isinstance(value, (list, tuple)):
            return value
        current = attr.Get()
        if current is None:
            return value
        try:
            return type(current)(*value)
        except (TypeError, ValueError):
            return value

    def on_get_articulation_joint_info(self):
        """Handle Command 73: GetArticulationJointInfo"""
        with self._timing_context(
                    f"rpc_server.step_command_{command_value_to_string[self.Command]}"
                ):
            command_config = self.data["command_config"]
            articulation_object_prim_path = command_config["articulation_object_prim_path"]
            joint_prim_path = command_config.get("joint_prim_path", None)
            stage = omni.usd.get_context().get_stage()
            joint_info = get_current_revolute_joint_info(stage, articulation_object_prim_path, joint_prim_path=joint_prim_path)
            self.data_to_send = {
                "joint_path": joint_info.joint_path,
                "joint_prim_name": joint_info.joint_prim_name,
                "body0_path": joint_info.body0_path,
                "body1_path": joint_info.body1_path,
                "axis": joint_info.axis,
                "current_angle_deg": joint_info.current_angle_deg,
                "lower_limit_deg": joint_info.lower_limit_deg,
                "upper_limit_deg": joint_info.upper_limit_deg,
            }

    def on_bimanual_rigid_move(self):
        with self._timing_context("rpc_server.step_command_BimanualRigidMove"):
            command_config = self.data["command_config"]
            leader_arm = command_config["leader_arm"]
            target_position = np.array(command_config["target_position"])
            target_rotation = np.array(command_config["target_rotation"])
            goal_offset = command_config.get("goal_offset", [0, 0, 0, 1, 0, 0, 0])
            path_constraint = command_config.get("path_constraint", [])
            offset_and_constraint_in_goal_frame = command_config.get(
                "offset_and_constraint_in_goal_frame", True
            )
            disable_collision_links = command_config.get("disable_collision_links", [])
            motion_run_ratio = command_config.get("motion_run_ratio", 1.0)
            max_consecutive_failures = command_config.get("max_consecutive_failures", 1)
            from_current_pose = command_config.get("from_current_pose", False)
            slowdown_factor = max(1.0, float(command_config.get("slowdown_factor", 1.0) or 1.0))
            is_right = leader_arm == "right"
            if not hasattr(self, "in_bimanual_rigid_move") or not self.in_bimanual_rigid_move:
                self.in_bimanual_rigid_move = True
                self.motion_run_ratio = motion_run_ratio
                if isinstance(self.end_effector_name, dict):
                    leader_ee_name = self.end_effector_name[leader_arm]
                else:
                    leader_ee_name = self.end_effector_name
                # Apply combined user and timestep slowdown to leader and follower.
                # The leader resamples in calculate_ik_goal; the follower resamples
                # in build_follower_scatter_plan.
                _leader = self.ui_builder.get_curobo_motion(is_right)
                _follower = self.ui_builder.get_curobo_motion(not is_right)
                _resolved = self._resolve_slowdown_factor(slowdown_factor)
                _leader.pending_slowdown_factor = _resolved
                _follower.pending_slowdown_factor = _resolved
                self.ui_builder._follow_bimanual_rigid(
                    target_position=target_position,
                    target_rotation=target_rotation,
                    leader_arm=leader_arm,
                    leader_ee_name=leader_ee_name,
                    goal_offset=goal_offset,
                    path_constraint=path_constraint,
                    offset_and_constraint_in_goal_frame=offset_and_constraint_in_goal_frame,
                    disable_collision_links=disable_collision_links,
                    max_consecutive_failures=max_consecutive_failures,
                    from_current_pose=from_current_pose,
                )
            leader_motion = self.ui_builder.get_curobo_motion(is_right)
            follower_motion = self.ui_builder.get_curobo_motion(not is_right)
            if leader_motion.reached and follower_motion.reached:
                self.data_to_send = {
                    "success": leader_motion.success and follower_motion.success
                }
                self.in_bimanual_rigid_move = False
                self.motion_run_ratio = 1.0

    def on_follow_trajectory(self):
        """Handle Command 72: FollowTrajectory"""
        with self._timing_context(
                    f"rpc_server.step_command_{command_value_to_string[self.Command]}"
                ):
            command_config = self.data["command_config"]
            trajectory_positions = command_config["trajectory_positions"]
            trajectory_orientations = command_config["trajectory_orientations"]
            init_rotation_matrix = quat2mat_wxyz(self.robot_init_rotation)
            robot_translation_matrix = np.zeros((4, 4))
            robot_translation_matrix[:3, :3] = init_rotation_matrix
            robot_translation_matrix[:3, 3] = self.robot_init_position
            robot_translation_matrix[3, 3] = 1
            target_poses = []
            for i in range(len(trajectory_positions)):
                target_pose = np.eye(4)
                target_pose[:3, :3] = quat2mat_wxyz(trajectory_orientations[i])
                target_pose[:3, 3] = trajectory_positions[i]
                target_poses.append(target_pose)
            target_poses = np.array(target_poses)
            target_poses_local = (
                np.linalg.inv(robot_translation_matrix[np.newaxis, ...]) @ target_poses
            )
            target_rotations_local = batch_matrices_to_quaternions_scipy_w_first(target_poses_local)
            target_positions_local = target_poses_local[:, :3, 3]
            is_right = command_config["is_right"]
            solvable_threshold = command_config.get("solvable_threshold", 0.5)
            max_consecutive_failures = command_config.get("max_consecutive_failures", 3)
            if not hasattr(self, "in_follow_trajectory") or not self.in_follow_trajectory:
                self.in_follow_trajectory = True
                if isinstance(self.end_effector_name, dict):
                    end_effector_name = self.end_effector_name["left"]
                    if is_right:
                        end_effector_name = self.end_effector_name["right"]
                else:
                    end_effector_name = self.end_effector_name
                self.ui_builder._follow_trajectory(
                    trajectory_positions=target_positions_local,
                    trajectory_orientations=target_rotations_local,
                    is_right=is_right,
                    solvable_threshold=solvable_threshold,
                    max_consecutive_failures=max_consecutive_failures,
                    active_ee_name=end_effector_name,
                )
            if self.ui_builder.get_curobo_motion(is_right).reached:
                self.data_to_send = {"success": self.ui_builder.get_curobo_motion(is_right).success}
                self.in_follow_trajectory = False

    def on_get_revolute_joint_trajectory(self):
        """Handle Command 71: GetRevoluteJointTrajectory"""
        with self._timing_context(
                    f"rpc_server.step_command_{command_value_to_string[self.Command]}"
                ):
            command_config = self.data["command_config"]
            stage = omni.usd.get_context().get_stage()
            articulation_object_prim_path = command_config["articulation_object_prim_path"]
            object_prim_path = command_config["object_prim_path"]
            target_limit = command_config.get("target_limit", "min")
            trajectory_step_deg = command_config.get("trajectory_step_deg", 1.0)
            joint_prim_path = command_config.get("joint_prim_path", None)
            object_position = []
            object_orientation = []
            if object_prim_path == "left_gripper":
                object_position, object_orientation = self._get_ee_pose(is_right=False)
            elif object_prim_path == "right_gripper":
                object_position, object_orientation = self._get_ee_pose(is_right=True)
            else:
                object_position, object_orientation = self._get_object_pose(
                    object_prim_path
                )

            trajectory_positions, trajectory_orientations = (
                world_point_trajectory_with_body1_revolute(
                    stage=stage,
                    root_prim_path=articulation_object_prim_path,
                    world_position=object_position,
                    world_orientation_wxyz=object_orientation,
                    target_limit=target_limit,
                    trajectory_step_deg=trajectory_step_deg,
                    joint_prim_path=joint_prim_path,
                    debug=False,
                )
            )
            self.data_to_send = {
                "trajectory_positions": trajectory_positions.tolist(),
                "trajectory_orientations": trajectory_orientations.tolist(),
            }

    # update
    def on_command_step(self):
        with self._timing_context("on_command_step"):
            # Blocked-step mechanism: play/pause for step-permitting commands.
            current_command = self.Command
            # logger.info(f"current_command: {current_command}, step: {self.ui_builder.my_world.current_time_step_index}")
            if self.data and current_command and self.block_step:
                # Capture the command locally to avoid changes during the checks.
                # Detect a new valid command.
                is_new_command = current_command != 0 and current_command != self.last_processed_command
                # Start simulation for a new command in the step-permitting map.
                if is_new_command and current_command in self.allow_simulation_commands:
                    if not self.ui_builder.my_world.is_playing():
                        logger.info(f"play world")
                        self.ui_builder.my_world.play()
                    self.is_processing_simulation_command = True
                    self.margin_frame_counter = 0
                    # Read its settling margin using the captured command.
                    # Convert the stored baseline 60 Hz margin to the current timestep.
                    self.current_margin_frames = self._scaled_margin_frames(
                        self.allow_simulation_commands[current_command]
                    )
                elif is_new_command:
                    # Leave World unchanged for new commands outside the allow list.
                    self.is_processing_simulation_command = False
                    self.margin_frame_counter = 0
                    self.current_margin_frames = 0

                # Start counting settling steps after a step-permitting command produces a result.
                if self.is_processing_simulation_command and self.data_to_send is not None:
                    self.margin_frame_counter += 1
                    if self.margin_frame_counter >= self.current_margin_frames:
                        # Pause simulation when the settling margin is exhausted.
                        if self.ui_builder.my_world.is_playing():
                            logger.info(f"pause world")
                            self.ui_builder.my_world.pause()
                        self.is_processing_simulation_command = False
                        self.margin_frame_counter = 0
                        self.current_margin_frames = 0
            self.last_processed_command = current_command
            if not self.data or not self.Command:
                return
            elif not self.data_to_send:
              try:
                with self._timing_context(
                    f"rpc_server.step_command_{command_value_to_string[self.Command]}"
                ): #SKIP IF HAS DATA_TO_SEND
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
                    elif self.Command == Command.DEACTIVATE_OBJECT:
                        self.handle_deactivate_object()
                    elif self.Command == Command.RUN_STEP:
                        self.handle_run_step()
                    elif self.Command == Command.COMMAND_ANY or self.Command >= Command.COMMAND_ANY_BEGIN:
                        self.handle_command_any()
                    else:
                        raise ValueError(f"Invalid command: {self.Command}")
              except Exception as e:
                # A handler exception (e.g. a rejected malicious task_name /
                # robot_cfg / scene_usd) must never leave the waiting gRPC worker
                # hung: record it as this command's result and clear the sim-step
                # gate so the notify_all below still wakes the caller.
                logger.error(f"Command step failed for command {self.Command}: {e}")
                self.is_processing_simulation_command = False
                if self.data_to_send is None:
                    self.data_to_send = f"ERROR: {e}"

        # When a result is ready, notify the blocking thread and wait for acknowledgement.
        if self.Command and self.data_to_send is not None and (not self.is_processing_simulation_command or not self.block_step):
            with self.condition:
                # Notify the blocking thread that the result is ready.
                self.condition.notify_all()
                # Wait until the result is consumed and Command/data_to_send are reset.
                # Keep stepping blocked while the result remains unread.
                while self.Command and self.data_to_send is not None:
                    self.condition.wait()

    def _get_observation(self):
        for camera in self.cameras:
            self._capture_camera(
                prim_path=camera, isRGB=True, isDepth=True, isSemantic=True, isGN=False
            )

    def _on_reset(self):
        self.ui_builder._on_reset()
        stage = omni.usd.get_context().get_stage()
        if stage:
            self._generated_geometry.reset_live(stage)
        self._reset_stiffness()
        self.target_position = [0, 0, 0]
        self._reset_scene_material()
        self._get_observation()
        self.frame_status = []
        self.playback_frames = {}
        self.playback_timerange = []
        self.playback_waited_frame_num = 0
        if self.ui_builder._lula_rotated:
            self.ui_builder._init_kinematic_solver(self.robot_cfg)
        if self.llm_gen_mode:
            for key, curoboMotion in self.ui_builder.curoboMotion.items():
                curoboMotion.refresh_obstacle_cache()
                break
            self._add_objects_in_scene("/World/Objects")

    def _on_blocking_thread(self, data, Command):
        self.data = data
        self.Command = Command
        with self.condition:
            while self.data_to_send is None:
                self.condition.wait()
            result = self.data_to_send
            self.Command = 0
            self.data_to_send = None
            # Notify the step thread that the result was consumed and Command reset.
            self.condition.notify_all()
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
        camera = render.make_camera(prim_path=camera_prim, resolution=[width, height])
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
        dof_names = physics.get_dof_names(self.articulat_objects[prim_path])
        positions = physics.get_joint_positions(self.articulat_objects[prim_path], as_torch=False)
        velocities = physics.get_joint_velocities(self.articulat_objects[prim_path], as_torch=False)
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
        physics.set_joint_positions(self.articulat_objects[prim_path], target_positions)

    def _add_objects_in_scene(self, root_prim_path: str):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            logger.error("Stage not found")
            return

        def _prim_or_subtree_has_deformable(p) -> bool:
            try:
                for sub in Usd.PrimRange(p):
                    if any("Deformable" in s for s in (sub.GetAppliedSchemas() or [])):
                        return True
            except Exception:
                pass
            return False

        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_prim_path)):
            prim_path = str(prim.GetPath())
            if not prim_path.endswith("/entity"):
                continue
            # Do not wrap deformable entities as rigid prims.
            # Their missing rigid PhysX handles can make view initialization fail
            # with a NoneType error; skip these entities.
            if _prim_or_subtree_has_deformable(prim):
                logger.info(
                    f"_add_objects_in_scene: skip {prim_path} — contains Deformable"
                )
                continue
            type = get_prim_object_type(prim_path)
            usd_object = XFormPrim(
                prim_path=prim_path,
            )
            items = []
            all_prim_paths = []
            if stage:
                for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                    path = str(prim.GetPath())
                    prim = get_prim_at_path(path)
                    if prim.IsA(UsdGeom.Mesh):
                        items.append(path)
                    all_prim_paths.append(path)
            if type == "articulation":
                self.ui_builder.my_world.play()
                for path in items:
                    if not self.enable_physics:
                        collisionAPI = UsdPhysics.CollisionAPI.Get(stage, path)
                        if collisionAPI:
                            collisionAPI.GetCollisionEnabledAttr().Set(False)
                articulation = Articulation(prim_path)
                self.articulat_objects[prim_path] = articulation
                self.usd_objects[prim_path] = usd_object
                for path in all_prim_paths:
                    rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, path)
                    if rigid_api:
                        rigid_prim = RigidPrim(prim_path=path)
                        self.rigid_bodies[path] = rigid_prim
            else:
                self.usd_objects[prim_path] = usd_object
                rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, prim_path)
                if rigid_api:
                    rigid_prim = RigidPrim(prim_path=prim_path)
                    self.rigid_bodies[prim_path] = rigid_prim
        return items
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
                # Prims with deformable schemas must not receive a rigid-body wrapper
                # or become static cuRobo obstacles. Traverse the subtree to detect
                # any schema containing Deformable.
                _has_deformable = False
                try:
                    _prim_check = stage.GetPrimAtPath(prim_path)
                    if _prim_check and _prim_check.IsValid():
                        for sub in Usd.PrimRange(_prim_check):
                            schemas = list(sub.GetAppliedSchemas() or [])
                            if any("Deformable" in s for s in schemas):
                                _has_deformable = True
                                break
                except Exception:
                    _has_deformable = False

                if self.enable_physics and add_rigid_body and not _has_deformable:
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
                elif self.enable_physics and add_rigid_body and _has_deformable:
                    logger.warning(
                        f"add_object: skip RigidBodyAPI wrap for {prim_path} —"
                        f" contains Deformable schema(FEM cloth / softbody)。"
                        f" Clients should pass add_rigid_body=False to avoid this path."
                    )
                    # Set center of gravity offset (unit: meters)
                    # physics_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(object_com[0], object_com[1],object_com[2]))
            for curobo_motion in self.ui_builder.curoboMotion.values():
                if curobo_motion:
                    # Deformable vertices change without updating the static cuRobo cache;
                    # deformable meshes may also lack regular triangulation.
                    if _has_deformable or not add_rigid_body:
                        logger.info(
                            f"add_object: skip curobo obstacle for {prim_path}"
                            f"(deformable={_has_deformable}, add_rigid_body={add_rigid_body})"
                        )
                        break
                    prim_usd_path = usd_path
                    if not (np.array(scale) == 1.0).all():
                        prim_usd_path = None
                        logger.info("scale is not 1.0, so simplified mesh cache cannot be used")
                    curobo_motion.add_obstacle_from_prim_path(prim_path, prim_usd_path)
                    break
        elif not self.llm_gen_mode:
            usd_object = XFormPrim(
                prim_path=prim_path,
                position=position,
                orientation=rotation,
                scale=scale,
            )
        else:
            logger.info("llm gen mode does not support adding object")
            return

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
        joint_positions = physics.get_joint_positions(articulation, as_torch=False)
        ids = {}
        dof_names = physics.get_dof_names(articulation)
        for idx in range(len(dof_names)):
            name = dof_names[idx]
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
        # Reuse initialized grippers to preserve internal PD state.
        # Repeatedly authoring drive stiffness/force limits on GPU
        # can destabilize close behavior during the settling window.
        # update_objects may set gripper state each frame and restore_state
        # may reset stiffness, so initialization must remain repeatable.
        if self.gripper_R is not None and self.gripper_L is not None:
            return self.robot
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
            articulation_apply_action_func=lambda *a, **kw: physics.apply_action(
                robot, a[0] if a else kw.get("control_actions")
            ),
            get_joint_positions_func=lambda: physics.get_joint_positions(robot, as_torch=False),
            set_joint_positions_func=lambda positions, joint_indices=None: physics.set_joint_positions(
                robot, positions, joint_indices=joint_indices
            ),
            dof_names=physics.get_dof_names(robot),
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
            articulation_apply_action_func=lambda *a, **kw: physics.apply_action(
                robot, a[0] if a else kw.get("control_actions")
            ),
            get_joint_positions_func=lambda: physics.get_joint_positions(robot, as_torch=False),
            set_joint_positions_func=lambda positions, joint_indices=None: physics.set_joint_positions(
                robot, positions, joint_indices=joint_indices
            ),
            dof_names=physics.get_dof_names(robot),
        )
        self.robot = robot
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
        physics.apply_action(self.robot, action)

    # Get pose of any object, Input: prim_path
    def _get_object_pose(self, object_prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read live prim poses using cached PhysX rigid views where available.
        USD/Fabric queries can be stale after attachment; cache views to avoid repeated
        construction on gRPC update_objects calls. Check RigidBodyAPI before creating
        a rigid view. For nonrigid descendants of a rigid body, compose the ancestor's
        live tensor pose with the static local delta using numpy column-vector matrices."""
        for value in self.articulat_objects.values():
            value.initialize()
        if object_prim_path == "robot":
            position, rotation = self.usd_objects["robot"].get_world_pose()
            for value in self.articulat_objects.values():
                value.initialize()
            return position, rotation

        # Check RigidBodyAPI before constructing SingleRigidPrim.
        # Use Fabric/USD fallback for nonrigid prims to avoid native plugin errors.
        # Error/Warning log("Pattern '...' did not match any rigid bodies"),
        # Python exception handling cannot suppress plugin errors written to stderr.
        _is_rigid = False
        _prim_check = None
        try:
            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                _prim_check = stage.GetPrimAtPath(object_prim_path)
                if _prim_check and _prim_check.IsValid():
                    _is_rigid = _prim_check.HasAPI(UsdPhysics.RigidBodyAPI)
        except Exception:
            _is_rigid = False
            _prim_check = None

        if _is_rigid:
            try:
                from sim_backend import backend as _bk  # noqa: PLC0415
                rp = self._rigid_view_cache.get(object_prim_path)
                if rp is None:
                    from isaacsim.core.prims import SingleRigidPrim  # noqa: PLC0415
                    rp = SingleRigidPrim(prim_path=object_prim_path)
                    self._rigid_view_cache[object_prim_path] = rp
                # Use usd=False for current PhysX/Fabric state instead of USD spawn snapshots.
                # Even enabled readback may not synchronize USD in time on the GPU pipeline.
                # Request the live view explicitly.
                poss, oris = rp._rigid_prim_view.get_world_poses(usd=False)
                position = np.asarray(_bk.to_numpy(poss[0]), dtype=np.float64)
                rotation = np.asarray(_bk.to_numpy(oris[0]), dtype=np.float64)
            except Exception:
                # If the rigid path fails, use the Fabric/USD fallback.
                position, rotation = physics.get_prim_world_pose(object_prim_path)
        else:
            # Nonrigid prims have two cases:
            #   With a rigid ancestor, such as a static hole child of a rigid entity,
            #   compose its live ancestor pose with the static
            #   ancestor-to-target local transform.
            #   Without a rigid ancestor, read static markers/decorations from USD.
            rigid_ancestor = None
            if _prim_check is not None:
                walker = _prim_check.GetParent()
                while walker and walker.IsValid() and not walker.IsPseudoRoot():
                    if walker.HasAPI(UsdPhysics.RigidBodyAPI):
                        rigid_ancestor = walker
                        break
                    walker = walker.GetParent()
            if rigid_ancestor is None:
                position, rotation = physics.get_prim_world_pose(object_prim_path)
            else:
                try:
                    from sim_backend import backend as _bk  # noqa: PLC0415
                    anc_path = str(rigid_ancestor.GetPath())
                    rp = self._rigid_view_cache.get(anc_path)
                    if rp is None:
                        from isaacsim.core.prims import SingleRigidPrim  # noqa: PLC0415
                        rp = SingleRigidPrim(prim_path=anc_path)
                        self._rigid_view_cache[anc_path] = rp
                    anc_poss, anc_oris = rp._rigid_prim_view.get_world_poses(usd=False)
                    anc_pos = np.asarray(_bk.to_numpy(anc_poss[0]), dtype=np.float64)
                    anc_quat = np.asarray(_bk.to_numpy(anc_oris[0]), dtype=np.float64)
                    anc_world = np.eye(4, dtype=np.float64)
                    anc_world[:3, :3] = quat2mat_wxyz(anc_quat)
                    anc_world[:3, 3] = anc_pos
                    # Gf matrices use row vectors: world = local @ M.
                    # Transpose when converting to numpy column-vector convention.
                    # Static body-to-hole transform = inverse(body_static) @ hole_static.
                    # T_world_hole_current = T_world_body_current @ T_body_hole_static
                    tgt_static_gf = UsdGeom.Xformable(_prim_check).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default())
                    anc_static_gf = UsdGeom.Xformable(rigid_ancestor).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default())
                    tgt_static = np.array(tgt_static_gf, dtype=np.float64).T
                    anc_static = np.array(anc_static_gf, dtype=np.float64).T
                    delta = np.linalg.inv(anc_static) @ tgt_static  # T_body_hole (local static)
                    tgt_world = anc_world @ delta  # T_world_body_current @ T_body_hole
                    position = tgt_world[:3, 3]
                    rotation = mat2quat_wxyz(tgt_world[:3, :3])
                except Exception:
                    position, rotation = physics.get_prim_world_pose(object_prim_path)
        for value in self.articulat_objects.values():
            value.initialize()
        return position, rotation

    def _get_ee_pose(self, is_right: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Return the current EE-center world pose as numpy position and wxyz quaternion.
        Use ui_builder's Lula FK from current joint positions. The G2_crsB_90D URDF
        fixed gripper-center rotations match the corresponding USDA link orientations."""
        position, rotation_matrix = self.ui_builder._get_ee_pose(is_right)
        rotation = mat2quat_wxyz(rotation_matrix)
        return position, rotation

    def _deactivate_object(self, prim_path: str) -> dict:
        """Deactivate an object by setting its visibility to invisible and disabling physics."""
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return {"success": False, "msg": "Stage not found"}

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"success": False, "msg": f"Prim path {prim_path} does not exist"}

        try:
            # Set visibility to invisible
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()

            # Disable collision if it has collision API
            for child_prim in Usd.PrimRange(prim):
                collision_api = UsdPhysics.CollisionAPI.Get(stage, child_prim.GetPath())
                if collision_api:
                    collision_api.GetCollisionEnabledAttr().Set(False)

            # Remove from curobo obstacle if exists
            for curobo_motion in self.ui_builder.curoboMotion.values():
                if curobo_motion:
                    try:
                        curobo_motion.remove_obstacle_from_prim_path(prim_path)
                    except Exception as e:
                        logger.error(f"Failed to remove obstacle from curobo: {e}")
                    break

            return {"success": True, "msg": f"Object {prim_path} deactivated successfully"}
        except Exception as e:
            return {"success": False, "msg": f"Failed to deactivate object: {str(e)}"}

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
                all_names = physics.get_dof_names(self.ui_builder.articulation)
                for i, idx in enumerate(joint_state.joint_indices):
                    joint_positions[all_names[idx]] = joint_state.joint_positions[i]
                results.append((is_success, joint_positions))
            time01 = time.time()
            logger.info(f"ik lula time cost{time01 - time00}, points number: {target_poses.shape[0]}")
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
            curoboMotion = self.ui_builder.get_curobo_motion(isRight)
            curoboMotion.set_obstacles()
            self.ui_builder.set_locked_joint_positions(isRight)
            results = curoboMotion.solve_batch_ik(
                target_positions_local,
                target_rotations_local,
                end_effector_name,
                output_link_pose=output_link_pose,
            )
            time_3 = time.time()
            logger.info(f"ik curobo time cost{time_3 - time0}")
        return results
