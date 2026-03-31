#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
#
# MuJoCo physics node for GeneSim RL lightweight pipeline.
# Runs the MuJoCo physics simulation and bridges it to ROS 2:
#   - Publishes /tf_render so IsaacSim can render body poses
#   - Publishes /joint_states for robot joint feedback
#   - Publishes /clock for simulation time
#   - Subscribes /joint_command for robot arm control
#   - Provides /get_object_pose and /get_object_aabb services for reward computation
#   - Provides /reset_scene service for episode reset
#
# Usage:
#   python mujoco_ros_node.py \
#       --mjcf <scene.xml> \
#       --physics-hz 1000 \
#       --ros-domain-id 0 \
#       --namespace env_0
#

import argparse
import json
import math
import os
import sys
import threading
import time
from multiprocessing import shared_memory as _shm_mod
from typing import Dict, List, Optional

import mujoco
import numpy as np

from geniesim.rl.renderer.shm_layout import (
    ctrl_shm_name as _ctrl_shm_name,
    ctrl_total_bytes as _ctrl_total_bytes,
    CTRL_HEADER_BYTES as _CTRL_HEADER_BYTES,
    CTRL_SYNC_BYTES as _CTRL_SYNC_BYTES,
    EE_STATE_DIM as _EE_STATE_DIM,
    BODY_POSE_DIM as _BODY_POSE_DIM,
    RESET_IDLE as _RESET_IDLE,
    RESET_REQUESTED as _RESET_REQUESTED,
    RESET_DONE as _RESET_DONE,
    MUJOCO_PHASE_WAIT as _MJ_WAIT,
    MUJOCO_PHASE_GO as _MJ_GO,
    MUJOCO_PHASE_DONE as _MJ_DONE,
)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.executors import SingleThreadedExecutor

from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage

from geniesim_rl_interfaces.srv import GetObjectPose, GetObjectAABB, ResetScene  # noqa: E402

# ---------------------------------------------------------------------------- #
# Helper math utilities
# ---------------------------------------------------------------------------- #

def xmat_to_quat_wxyz(xmat: np.ndarray) -> np.ndarray:
    """Convert MuJoCo 3×3 rotation matrix (row-major) to quaternion [w, x, y, z]."""
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, xmat.flatten())
    return q  # [w, x, y, z]


def compute_aabb_world(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple:
    """
    Approximate world-space AABB for a body by iterating its geoms.
    Returns (aabb_min [3], aabb_max [3]).
    """
    mins = []
    maxs = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue
        geom_type = model.geom_type[geom_id]
        size = model.geom_size[geom_id]
        pos = data.geom_xpos[geom_id]

        # Approximate half-extents
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            half = size[:3]
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            half = np.array([size[0]] * 3)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half = np.array([size[0], size[0], size[1]])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r = size[0]
            half_len = size[1] + r
            half = np.array([r, r, half_len])
        else:
            half = np.abs(size[:3]) + 0.01

        mins.append(pos - half)
        maxs.append(pos + half)

    if not mins:
        # Fallback: body position ± small margin
        pos = data.xpos[body_id]
        return pos - 0.05, pos + 0.05

    return np.min(mins, axis=0), np.max(maxs, axis=0)


def _euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert intrinsic XYZ Euler angles to 3×3 rotation matrix (no scipy needed)."""
    ca, sa = np.cos(roll), np.sin(roll)
    cb, sb = np.cos(pitch), np.sin(pitch)
    cg, sg = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cg*cb, -sg*ca + cg*sb*sa,  sg*sa + cg*sb*ca],
        [sg*cb,  cg*ca + sg*sb*sa, -cg*sa + sg*sb*ca],
        [-sb,    cb*sa,             cb*ca            ],
    ])


def _matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Extract intrinsic XYZ Euler angles from a 3×3 rotation matrix."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 0.9999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw  = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw  = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _rand_sample(rng: np.random.Generator, distribution: str, param) -> float:
    """
    Sample one scalar perturbation.

    distribution='uniform': param=[lo, hi]  → uniform sample in [lo, hi]
    distribution='gaussian': param=std (float or 1-element list)  → N(0, std)
    Returns 0.0 if param range is zero (lo==hi for uniform).
    """
    if distribution == "uniform":
        lo, hi = float(param[0]), float(param[1])
        if lo == hi:
            return 0.0
        return float(rng.uniform(lo, hi))
    else:  # gaussian
        std = float(param[0]) if isinstance(param, (list, tuple)) else float(param)
        if std == 0.0:
            return 0.0
        return float(rng.normal(0.0, std))


# ---------------------------------------------------------------------------- #
# Main ROS Node
# ---------------------------------------------------------------------------- #

QOS_RELIABLE = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
QOS_BEST_EFFORT = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class MuJoCoRosNode(Node):
    """
    Python-native MuJoCo physics node.

    Coordinate convention: MuJoCo uses Z-up, meters, same as IsaacSim/ROS.
    """

    def __init__(self, args: argparse.Namespace):
        node_name = f"mujoco_physics_{args.namespace.replace('/', '_').strip('_')}"
        super().__init__(node_name)

        self.args = args
        self.physics_hz = args.physics_hz
        self.dt = 1.0 / self.physics_hz

        # Load model
        self.model = mujoco.MjModel.from_xml_path(args.mjcf)
        self.model.opt.timestep = self.dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Parse initial joint config overlay
        self.init_qpos: Optional[np.ndarray] = None
        if args.init_qpos_json:
            self.init_qpos = np.array(json.loads(args.init_qpos_json))

        # Cache body names → ids for fast lookup
        self._body_name_to_id: Dict[str, int] = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i): i
            for i in range(self.model.nbody)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        }
        # Base link body id for base_link-frame EE coordinate conversion.
        self._base_link_id: int = self._body_name_to_id.get("base_link", -1)

        # Cache joint names → actuator ids for control
        self._joint_name_to_qpos_idx: Dict[str, int] = {}
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name:
                self._joint_name_to_qpos_idx[name] = self.model.jnt_qposadr[j]

        self._actuator_name_to_id: Dict[str, int] = {}
        for a in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if name:
                self._actuator_name_to_id[name] = a

        # ---- Body joints metadata (used for reset ctrl sync) ----
        # We record: qposadr and the actuators that target this joint (if any).
        self._body_joint_debug: List[dict] = []
        self._body_joint_debug_err: str = ""
        self._rebuild_body_joint_debug()

        # Commanded control (protected by lock; fallback when no ctrl SHM)
        self._ctrl_lock = threading.Lock()
        self._ctrl: Optional[np.ndarray] = None

        # ---- Per-env control SHM ----
        # Created when --shm-name is provided.  Layout (see shm_layout.py):
        #   [0:4]      uint32  state_counter
        #   [4:8]      uint32  reset_flag
        #   [8:8+S]    float32 states (pos then vel for each non-free joint)
        #   [8+S:8+S+A]  float32 actions (one per actuator)
        #   [8+S+A:end]  float32 info_buf (ground-truth body poses)
        self._ctrl_shm: Optional[_shm_mod.SharedMemory] = None
        self._ctrl_counter: Optional[np.ndarray] = None
        self._ctrl_reset_flag: Optional[np.ndarray] = None
        self._ctrl_states_buf: Optional[np.ndarray] = None
        self._ctrl_actions_buf: Optional[np.ndarray] = None
        self._ctrl_info_buf: Optional[np.ndarray] = None
        self._ctrl_mj_phase: Optional[np.ndarray] = None
        self._ctrl_steps_per_step: Optional[np.ndarray] = None
        self._info_body_ids: List[int] = []
        self._sync_mode: bool = False

        if args.shm_name:
            # Host-specified dims override full-model dims so the SHM layout
            # matches exactly what the host expects.
            self._ctrl_state_dim = args.state_dim if args.state_dim > 0 else sum(
                1 for j in range(self.model.njnt)
                if self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
                and mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            ) * 2
            self._ctrl_action_dim = args.action_dim if args.action_dim > 0 else self.model.nu

            if args.info_body_names:
                for bname in args.info_body_names:
                    bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
                    if bid < 0:
                        self.get_logger().warning(f"info_body_names: '{bname}' not found in model, skipped")
                    else:
                        self._info_body_ids.append(bid)
            self._ctrl_info_dim = len(self._info_body_ids) * _BODY_POSE_DIM

            # Joint/actuator offset parameters (allow targeting arm subset)
            self._state_joint_offset: int = args.state_joint_offset
            self._ctrl_offset: int = args.ctrl_offset
            # Right-arm ctrl offset.  Defaults to ctrl_offset+7 for models where
            # both arms' actuators are contiguous.  Override when position actuators
            # for left and right arm are separated (e.g. G2_t2v2 interleaves motor
            # actuators between the two arms: left-pos[24:31], left-motor[31:38],
            # right-pos[38:45] → set ctrl_offset=24, ctrl_offset_r=38).
            _r = args.ctrl_offset_r
            self._ctrl_offset_r: int = _r if _r >= 0 else args.ctrl_offset + 7

            # Parse env_id from namespace "env_N"
            try:
                _env_id = int(args.namespace.split("_")[-1])
            except ValueError:
                _env_id = 0

            _ctrl_name = _ctrl_shm_name(args.shm_name, _env_id)
            _ctrl_bytes = _ctrl_total_bytes(self._ctrl_state_dim, self._ctrl_action_dim, self._ctrl_info_dim)
            try:
                self._ctrl_shm = _shm_mod.SharedMemory(
                    name=_ctrl_name, create=True, size=max(_ctrl_bytes, 1)
                )
            except FileExistsError:
                # Stale segment from a previous crashed run — unlink and recreate.
                _stale = _shm_mod.SharedMemory(name=_ctrl_name, create=False)
                _stale.close()
                _stale.unlink()
                self._ctrl_shm = _shm_mod.SharedMemory(
                    name=_ctrl_name, create=True, size=max(_ctrl_bytes, 1)
                )
            # Allow host user (differs from container uid 1234) to open SHM
            try:
                os.chmod(f"/dev/shm/{_ctrl_name}", 0o666)
            except PermissionError:
                pass

            _S = self._ctrl_state_dim * 4
            self._ctrl_counter = np.ndarray(
                (1,), dtype=np.uint32, buffer=self._ctrl_shm.buf, offset=0
            )
            self._ctrl_reset_flag = np.ndarray(
                (1,), dtype=np.uint32, buffer=self._ctrl_shm.buf, offset=4
            )
            self._ctrl_states_buf = np.ndarray(
                (self._ctrl_state_dim,), dtype=np.float32,
                buffer=self._ctrl_shm.buf, offset=_CTRL_HEADER_BYTES,
            )
            self._ctrl_actions_buf = np.ndarray(
                (self._ctrl_action_dim,), dtype=np.float32,
                buffer=self._ctrl_shm.buf, offset=_CTRL_HEADER_BYTES + _S,
            )
            _A = self._ctrl_action_dim * 4
            if self._ctrl_info_dim > 0:
                self._ctrl_info_buf = np.ndarray(
                    (self._ctrl_info_dim,), dtype=np.float32,
                    buffer=self._ctrl_shm.buf, offset=_CTRL_HEADER_BYTES + _S + _A,
                )
            _I = self._ctrl_info_dim * 4
            _sync_off = _CTRL_HEADER_BYTES + _S + _A + _I
            self._ctrl_mj_phase = np.ndarray(
                (1,), dtype=np.uint32,
                buffer=self._ctrl_shm.buf, offset=_sync_off,
            )
            self._ctrl_steps_per_step = np.ndarray(
                (1,), dtype=np.uint32,
                buffer=self._ctrl_shm.buf, offset=_sync_off + 4,
            )
            self._ctrl_counter[0] = 0
            self._ctrl_reset_flag[0] = _RESET_IDLE
            self._ctrl_states_buf[:] = 0.0
            self._ctrl_actions_buf[:] = 0.0
            if self._ctrl_info_buf is not None:
                self._ctrl_info_buf[:] = 0.0
            self._ctrl_mj_phase[0] = _MJ_WAIT
            _sps = max(1, args.steps_per_step)
            self._ctrl_steps_per_step[0] = _sps
            self._sync_mode = args.sync_mode
            self.get_logger().info(
                f"Ctrl SHM '{_ctrl_name}' created | "
                f"state_dim={self._ctrl_state_dim} action_dim={self._ctrl_action_dim} "
                f"info_dim={self._ctrl_info_dim} sync={self._sync_mode} sps={_sps}"
            )

        # ---- Control mode setup ----
        self._control_mode: str = args.control_mode
        self._gripper_ctrl_l: int = args.gripper_ctrl_l
        self._gripper_ctrl_r: int = args.gripper_ctrl_r
        self._setup_control_mode()

        # Right-arm EE reset pose (ee mode only); stored as [x,y,z,roll,pitch,yaw].
        self._reset_ee_r: Optional[np.ndarray] = None
        if getattr(args, "reset_ee_r_json", "") and self._control_mode == "ee":
            self._reset_ee_r = np.array(json.loads(args.reset_ee_r_json), dtype=np.float64)

        # Publishers
        ns = args.namespace.rstrip("/") + "/"
        self._pub_clock = self.create_publisher(Clock, f"{ns}clock", QOS_RELIABLE)
        self._pub_joint_states = self.create_publisher(
            JointState, f"{ns}joint_states", QOS_BEST_EFFORT
        )
        self._pub_tf_render = self.create_publisher(
            TFMessage, f"{ns}tf_render", QOS_BEST_EFFORT
        )

        # Subscribers
        self._sub_cmd = self.create_subscription(
            JointState,
            f"{ns}joint_command",
            self._on_joint_command,
            QOS_BEST_EFFORT,
        )

        # Services
        self._srv_pose = self.create_service(
            GetObjectPose, f"{ns}get_object_pose", self._handle_get_object_pose
        )
        self._srv_aabb = self.create_service(
            GetObjectAABB, f"{ns}get_object_aabb", self._handle_get_object_aabb
        )
        self._srv_reset = self.create_service(
            ResetScene, f"{ns}reset_scene", self._handle_reset_scene
        )

        # Sim time state
        self._sim_time: float = 0.0
        self._step_count: int = 0

        # Only publish tf_render at reduced rate (avoids flooding IsaacSim)
        self._render_hz = args.render_hz
        self._render_interval = max(1, round(self.physics_hz / self._render_hz))

        self.get_logger().info(
            f"MuJoCoRosNode ready | mjcf={args.mjcf} | ns={ns} | "
            f"physics={self.physics_hz}Hz | render={self._render_hz}Hz"
        )

        self._load_rand_cfg()

    def _rebuild_body_joint_debug(self) -> None:
        """(Re)build body-joint metadata. Safe to call multiple times."""
        try:
            joint_to_actuators: Dict[int, List[int]] = {}
            for a in range(self.model.nu):
                jid = int(self.model.actuator_trnid[a, 0])
                joint_to_actuators.setdefault(jid, []).append(a)

            items: List[dict] = []
            for j in range(self.model.njnt):
                jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                if not jname or "_body_joint" not in jname:
                    continue
                if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                qadr = int(self.model.jnt_qposadr[j])
                acts = []
                for a in joint_to_actuators.get(j, []):
                    aname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                    acts.append({"act_id": int(a), "act_name": aname})
                items.append({
                    "joint_id": int(j),
                    "joint": jname,
                    "qposadr": qadr,
                    "actuators": acts,
                })
            self._body_joint_debug = items
            self._body_joint_debug_err = ""
        except Exception as e:
            self._body_joint_debug = []
            self._body_joint_debug_err = repr(e)
    def _sync_body_position_ctrl_to_qpos(self) -> None:
        """
        Body joints may have position actuators whose targets live in data.ctrl.
        If those ctrl targets are left at 0 after reset, the actuators will pull
        body joints back toward 0, overriding init_qpos. To prevent that, set the
        position-actuator targets to the current qpos after reset.
        """
        try:
            if self.model.nu <= 0:
                return
            if not self._body_joint_debug:
                self._rebuild_body_joint_debug()
            if not self._body_joint_debug:
                return

            updated = []
            for it in self._body_joint_debug:
                qadr = int(it["qposadr"])
                q = float(self.data.qpos[qadr])
                for a in it.get("actuators", []):
                    aname = a.get("act_name") or ""
                    aid = int(a.get("act_id", -1))
                    # Only touch position actuators; leave torque motors unchanged.
                    if aid < 0 or aid >= self.model.nu:
                        continue
                    if not aname.startswith("position_"):
                        continue
                    self.data.ctrl[aid] = q
                    updated.append({"joint": it["joint"], "act_id": aid, "act_name": aname, "target": q})

        except Exception:
            pass

    def _load_rand_cfg(self):
        """
        Parse the randomization config JSON and pre-compute joint/body addresses.
        Called once at the end of __init__, after the model is fully loaded.
        """
        # Per-env seed: combine global seed with env_id so parallel envs get
        # independent random streams.
        try:
            _env_id = int(self.args.namespace.split("_")[-1])
        except ValueError:
            _env_id = 0
        seed = (getattr(self.args, "seed", 0) + _env_id) % (2 ** 31)
        self._rng = np.random.default_rng(seed)

        cfg_json = getattr(self.args, "randomization_cfg", "")
        if not cfg_json:
            self._rand_cfg: dict = {}
            self._robot_base_free_joint_adr: int = -1
            self._arm_rand_qpos_adrs: list = []
            self._arm_rand_joint_ids: list = []
            self._obj_free_joint_adrs: dict = {}
            return

        self._rand_cfg = json.loads(cfg_json)

        # ---- Object alias map (_object_map embedded by geniesim_env.py) ----
        # Maps simple alias (e.g. "block") → {"mujoco_body": "...", "usd_prim": "..."}
        self._object_map: dict = self._rand_cfg.pop("_object_map", {})

        # ---- Robot base free joint ----
        self._robot_base_free_joint_adr = -1
        rb_body = self._rand_cfg.get("robot_base", {}).get("body_name", "base_link")
        if rb_body:
            bid = self._body_name_to_id.get(rb_body, -1)
            if bid >= 0:
                for j in range(self.model.njnt):
                    if (self.model.jnt_bodyid[j] == bid and
                            self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
                        self._robot_base_free_joint_adr = int(self.model.jnt_qposadr[j])
                        break
            if self._robot_base_free_joint_adr < 0:
                self.get_logger().warning(
                    f"[rand] robot_base body '{rb_body}' has no free joint; "
                    f"robot_base randomization disabled."
                )

        # ---- Arm joint qpos addresses (from ctrl_offset actuators) ----
        ctrl_off = self.args.ctrl_offset
        self._arm_rand_qpos_adrs = []
        self._arm_rand_joint_ids = []
        if self.model.nu >= ctrl_off + 14:
            for ci in range(ctrl_off, ctrl_off + 14):
                jid = int(self.model.actuator_trnid[ci, 0])
                self._arm_rand_qpos_adrs.append(int(self.model.jnt_qposadr[jid]))
                self._arm_rand_joint_ids.append(jid)

        # ---- Object body free joints ----
        # Item resolution order:
        #   1. item["name"]  → look up mujoco_body via self._object_map
        #   2. item["body_name"]  → use directly (backward-compatible)
        self._obj_free_joint_adrs = {}
        for item in self._rand_cfg.get("objects", {}).get("items", []):
            alias = item.get("name", "")
            bname = item.get("body_name", "")
            if alias and not bname:
                entry = self._object_map.get(alias, {})
                bname = entry.get("mujoco_body", "")
                if not bname:
                    self.get_logger().warning(
                        f"[rand] alias '{alias}' not found in object_map; skipped."
                    )
                    continue
            if not bname:
                self.get_logger().warning("[rand] object item has no 'name' or 'body_name'; skipped.")
                continue
            bid = self._body_name_to_id.get(bname, -1)
            if bid < 0:
                self.get_logger().warning(
                    f"[rand] object body '{bname}' (alias='{alias}') not found in model; skipped."
                )
                continue
            for j in range(self.model.njnt):
                if (self.model.jnt_bodyid[j] == bid and
                        self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
                    self._obj_free_joint_adrs[bname] = int(self.model.jnt_qposadr[j])
                    break
            if bname not in self._obj_free_joint_adrs:
                self.get_logger().warning(
                    f"[rand] object body '{bname}' has no free joint; skipped."
                )

        aliases = [it.get("name") or it.get("body_name", "?")
                   for it in self._rand_cfg.get("objects", {}).get("items", [])]
        self.get_logger().info(
            f"[rand] Config loaded | "
            f"robot_base_adr={self._robot_base_free_joint_adr} "
            f"arm_joints={len(self._arm_rand_qpos_adrs)} "
            f"objects={aliases} → resolved={list(self._obj_free_joint_adrs.keys())}"
        )

    def _apply_randomization(self):
        """
        Apply per-reset random perturbations to data.qpos.
        Call after mj_resetData (and optional init_qpos overlay) but BEFORE mj_forward.
        """
        if not self._rand_cfg:
            return

        # ---- Robot base ----
        rb = self._rand_cfg.get("robot_base", {})
        if rb.get("enabled", False) and self._robot_base_free_joint_adr >= 0:
            adr = self._robot_base_free_joint_adr
            dist = rb.get("distribution", "uniform")
            dx = _rand_sample(self._rng, dist, rb.get("pos_x", [0.0, 0.0]))
            dy = _rand_sample(self._rng, dist, rb.get("pos_y", [0.0, 0.0]))
            self.data.qpos[adr]     += dx
            self.data.qpos[adr + 1] += dy
            # Yaw rotation about Z axis
            yaw_param = rb.get("yaw", [0.0, 0.0])
            dyaw = _rand_sample(self._rng, dist, yaw_param)
            if abs(dyaw) > 1e-9:
                cy, sy = np.cos(dyaw * 0.5), np.sin(dyaw * 0.5)
                dq = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
                q_cur = np.array(self.data.qpos[adr + 3:adr + 7], dtype=np.float64)
                q_new = np.empty(4)
                mujoco.mju_mulQuat(q_new, dq, q_cur)
                self.data.qpos[adr + 3:adr + 7] = q_new

        # ---- Arm joints ----
        aj = self._rand_cfg.get("arm_joints", {})
        if aj.get("enabled", False) and self._arm_rand_qpos_adrs:
            dist = aj.get("distribution", "gaussian")
            std = aj.get("std", 0.05)
            for qp_adr, jid in zip(self._arm_rand_qpos_adrs, self._arm_rand_joint_ids):
                noise = _rand_sample(self._rng, dist, std)
                lo = float(self.model.jnt_range[jid, 0])
                hi = float(self.model.jnt_range[jid, 1])
                new_val = float(self.data.qpos[qp_adr]) + noise
                if lo < hi:
                    self.data.qpos[qp_adr] = float(np.clip(new_val, lo, hi))
                else:
                    self.data.qpos[qp_adr] = float(new_val)

        # ---- Objects ----
        obj_cfg = self._rand_cfg.get("objects", {})
        if obj_cfg.get("enabled", False):
            for item in obj_cfg.get("items", []):
                bname = item.get("body_name", "")
                adr = self._obj_free_joint_adrs.get(bname, -1)
                if adr < 0:
                    continue
                dist = item.get("distribution", "uniform")
                for axis_idx, key in enumerate(["pos_x", "pos_y", "pos_z"]):
                    param = item.get(key)
                    if param is None:
                        continue
                    delta = _rand_sample(self._rng, dist, param)
                    self.data.qpos[adr + axis_idx] += delta

    # ---------------------------------------------------------------------- #
    # Callbacks
    # ---------------------------------------------------------------------- #

    def _on_joint_command(self, msg: JointState):
        """Receive joint position commands and update ctrl buffer."""
        ctrl = np.copy(self.data.ctrl)
        for name, pos in zip(msg.name, msg.position):
            aid = self._actuator_name_to_id.get(name)
            if aid is not None:
                ctrl[aid] = pos
        with self._ctrl_lock:
            self._ctrl = ctrl

    def _handle_get_object_pose(self, request, response):
        """Return world pose (position + quaternion wxyz) for a named body."""
        body_id = self._body_name_to_id.get(request.object_name)
        if body_id is None:
            response.success = False
            response.message = f"Unknown body: {request.object_name}"
            return response
        pos = self.data.xpos[body_id]
        xmat = self.data.xmat[body_id].reshape(3, 3)
        quat_wxyz = xmat_to_quat_wxyz(xmat)
        response.success = True
        response.position = pos.tolist()
        response.quaternion_wxyz = quat_wxyz.tolist()
        return response

    def _handle_get_object_aabb(self, request, response):
        """Return world-space AABB [x_min,y_min,z_min,x_max,y_max,z_max] for a named body."""
        body_id = self._body_name_to_id.get(request.object_name)
        if body_id is None:
            response.success = False
            response.message = f"Unknown body: {request.object_name}"
            return response
        aabb_min, aabb_max = compute_aabb_world(self.model, self.data, body_id)
        response.success = True
        response.aabb = [*aabb_min.tolist(), *aabb_max.tolist()]
        return response

    def _handle_reset_scene(self, request, response):
        """Reset simulation to initial state, optionally with a new qpos."""
        mujoco.mj_resetData(self.model, self.data)
        if request.init_qpos and len(request.init_qpos) == self.model.nq:
            self.data.qpos[:] = np.array(request.init_qpos)
        elif self.init_qpos is not None:
            self.data.qpos[: len(self.init_qpos)] = self.init_qpos
        # If body joints are position-actuated, sync ctrl targets to init_qpos.
        self._sync_body_position_ctrl_to_qpos()
        self._apply_randomization()
        # Sync arm ctrl to qpos so position actuators hold init_qpos instead of
        # fighting it (mj_resetData leaves ctrl=0, causing kp*(0-qpos) torques).
        _l_ctrl = getattr(self, "_l_arm_ctrl_indices", None)
        _l_qpos = getattr(self, "_l_arm_qpos_indices", None)
        _r_ctrl = getattr(self, "_r_arm_ctrl_indices", None)
        _r_qpos = getattr(self, "_r_arm_qpos_indices", None)
        if _l_ctrl and _l_qpos:
            for ci, qi in zip(_l_ctrl, _l_qpos):
                self.data.ctrl[ci] = self.data.qpos[qi]
        if _r_ctrl and _r_qpos:
            for ci, qi in zip(_r_ctrl, _r_qpos):
                self.data.ctrl[ci] = self.data.qpos[qi]
        self._apply_reset_ee_r()
        mujoco.mj_forward(self.model, self.data)
        self._sim_time = 0.0
        self._step_count = 0
        with self._ctrl_lock:
            self._ctrl = None
        self._ctrl_actions_buf[:] = 0.0
        # In EE mode, seed actions buffer with current EE poses so the first
        # _apply_ee_actions call holds the reset configuration.
        if self._control_mode == 'ee':
            _l_id = getattr(self, "_l_ee_body_id", -1)
            _r_id = getattr(self, "_r_ee_body_id", -1)
            if _l_id >= 0:
                _pl, _rl = self._world_to_base_link(
                    self.data.xpos[_l_id].copy(),
                    self.data.xmat[_l_id].reshape(3, 3).copy())
                self._ctrl_actions_buf[0:3] = _pl.astype(np.float32)
                self._ctrl_actions_buf[3:6] = _rl.astype(np.float32)
            if _r_id >= 0:
                _pr, _rr = self._world_to_base_link(
                    self.data.xpos[_r_id].copy(),
                    self.data.xmat[_r_id].reshape(3, 3).copy())
                self._ctrl_actions_buf[6:9]  = _pr.astype(np.float32)
                self._ctrl_actions_buf[9:12] = _rr.astype(np.float32)
        response.success = True
        self.get_logger().info("Scene reset complete.")
        return response

    # ---------------------------------------------------------------------- #
    # Publishing helpers
    # ---------------------------------------------------------------------- #

    def _publish_clock(self):
        sec_float = self._sim_time
        sec = int(sec_float)
        nanosec = int((sec_float - sec) * 1e9)
        msg = Clock()
        msg.clock.sec = sec
        msg.clock.nanosec = nanosec
        self._pub_clock.publish(msg)

    def _publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        names, positions, velocities, efforts = [], [], [], []
        for jname, qidx in self._joint_name_to_qpos_idx.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            jtype = self.model.jnt_type[jid]
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                continue  # skip free joints (object bodies)
            names.append(jname)
            positions.append(float(self.data.qpos[qidx]))
            vidx = self.model.jnt_dofadr[jid]
            velocities.append(float(self.data.qvel[vidx]))
            efforts.append(float(self.data.qfrc_actuator[vidx]) if vidx < len(self.data.qfrc_actuator) else 0.0)
        msg.name = names
        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts
        self._pub_joint_states.publish(msg)

    def _publish_tf_render(self):
        """
        Publish absolute world poses of ALL bodies as TFMessage on /ns/tf_render.
        IsaacSim renderer subscribes and writes poses to USD prims.
        """
        transforms = []
        ns_prefix = self.args.namespace.rstrip("/")
        stamp = self.get_clock().now().to_msg()
        for bname, bid in self._body_name_to_id.items():
            if bid == 0:  # world body
                continue
            pos = self.data.xpos[bid]
            xmat = self.data.xmat[bid].reshape(3, 3)
            q = xmat_to_quat_wxyz(xmat)
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = "world"
            # Child frame id encodes the full prim path so renderer can map it
            tf.child_frame_id = bname
            tf.transform.translation = Vector3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            tf.transform.rotation = Quaternion(
                w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3])
            )
            transforms.append(tf)
        if transforms:
            self._pub_tf_render.publish(TFMessage(transforms=transforms))

    # ---------------------------------------------------------------------- #
    # Ctrl SHM helpers
    # ---------------------------------------------------------------------- #

    def _write_ctrl_states(self):
        """Write current joint pos+vel to the ctrl SHM states buffer.

        Joint layout: [arm_l_pos(7), arm_r_pos(7), arm_l_vel(7), arm_r_vel(7)]
        Uses the actuator-based arm joint indices (_l/_r_arm_qpos_indices and
        _l/_r_arm_dof_indices) so that non-arm joints (gripper, body, chassis)
        are never mixed into the arm state slots — regardless of their position
        in the joint tree.

        Falls back to the legacy offset-based scan when arm indices are not
        yet initialised (before _setup_control_mode runs).
        """
        if self._ctrl_states_buf is None:
            return

        l_qp = getattr(self, "_l_arm_qpos_indices", None)
        r_qp = getattr(self, "_r_arm_qpos_indices", None)
        l_dof = getattr(self, "_l_arm_dof_indices", None)
        r_dof = getattr(self, "_r_arm_dof_indices", None)

        if l_qp and r_qp:
            # Fast path: write arm joints directly (correct even when gripper
            # joints sit between the two arms in the MuJoCo joint tree).
            n_l, n_r = len(l_qp), len(r_qp)
            # positions: [0:n_l] = left, [n_l:n_l+n_r] = right
            for i, qi in enumerate(l_qp):
                self._ctrl_states_buf[i] = float(self.data.qpos[qi])
            for i, qi in enumerate(r_qp):
                self._ctrl_states_buf[n_l + i] = float(self.data.qpos[qi])
            # velocities: [n_l+n_r:] = left+right
            off = n_l + n_r
            for i, di in enumerate(l_dof):
                self._ctrl_states_buf[off + i] = float(self.data.qvel[di])
            for i, di in enumerate(r_dof):
                self._ctrl_states_buf[off + n_l + i] = float(self.data.qvel[di])
            return

        # Legacy fallback: sequential non-free joint scan from state_joint_offset.
        joint_state_dim = self._ctrl_state_dim - _EE_STATE_DIM
        half = joint_state_dim // 2
        raw_idx = 0
        buf_idx = 0
        for jname, qidx in self._joint_name_to_qpos_idx.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if raw_idx < self._state_joint_offset:
                raw_idx += 1
                continue
            if buf_idx >= half:
                break
            self._ctrl_states_buf[buf_idx] = self.data.qpos[qidx]
            vidx = self.model.jnt_dofadr[jid]
            self._ctrl_states_buf[half + buf_idx] = self.data.qvel[vidx]
            raw_idx += 1
            buf_idx += 1

    def _write_ctrl_info(self):
        if self._ctrl_info_buf is None:
            return
        off = 0
        for bid in self._info_body_ids:
            pos = self.data.xpos[bid]
            mat = self.data.xmat[bid].reshape(3, 3)
            quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat, mat.ravel())
            self._ctrl_info_buf[off:off + 3] = pos.astype(np.float32)
            self._ctrl_info_buf[off + 3:off + 7] = quat.astype(np.float32)
            off += _BODY_POSE_DIM

    def _get_base_link_tf(self):
        """Return (pos [3], R [3×3]) of base_link in world frame.

        Falls back to (zeros, identity) if base_link is not in the model.
        """
        if self._base_link_id < 0:
            return np.zeros(3), np.eye(3)
        p = self.data.xpos[self._base_link_id].copy()
        R = self.data.xmat[self._base_link_id].reshape(3, 3).copy()
        return p, R

    def _base_link_to_world(self, pos_local: np.ndarray, eul_local: np.ndarray):
        """Convert pos+euler from base_link frame to world frame."""
        p_base, R_base = self._get_base_link_tf()
        pos_world = R_base @ pos_local + p_base
        R_local = _euler_xyz_to_matrix(*eul_local)
        eul_world = _matrix_to_euler_xyz(R_base @ R_local)
        return pos_world, eul_world

    def _world_to_base_link(self, pos_world: np.ndarray, R_world: np.ndarray):
        """Convert world-frame pos+rotation matrix to base_link frame.

        Returns (pos_local [3], euler_xyz_local [3]).
        """
        p_base, R_base = self._get_base_link_tf()
        pos_local = R_base.T @ (pos_world - p_base)
        R_local = R_base.T @ R_world
        eul_local = _matrix_to_euler_xyz(R_local)
        return pos_local, eul_local

    def _write_ctrl_ee_states(self):
        """Write left/right EE pose and velocity (base_link frame) to the EE slots of ctrl SHM states.

        EE layout in states[joint_state_dim : joint_state_dim+24]:
            [ee_l_pos(3), ee_l_rpy(3), ee_l_lin_vel(3), ee_l_ang_vel(3),
             ee_r_pos(3), ee_r_rpy(3), ee_r_lin_vel(3), ee_r_ang_vel(3)]
        Only writes when EE body ids are configured (ee control mode).
        """
        if self._ctrl_states_buf is None:
            return
        joint_state_dim = self._ctrl_state_dim - _EE_STATE_DIM
        base_idx = joint_state_dim  # first EE slot in the buffer

        def _write_ee(body_id: int, slot: int):
            if body_id < 0:
                return
            pos_w = self.data.xpos[body_id].copy()
            R_w = self.data.xmat[body_id].reshape(3, 3).copy()
            pos_local, eul_local = self._world_to_base_link(pos_w, R_w)
            self._ctrl_states_buf[slot:slot + 3] = pos_local.astype(np.float32)
            self._ctrl_states_buf[slot + 3:slot + 6] = eul_local.astype(np.float32)
            
            # Write EE velocity (cvel format: [angular(3), linear(3)])
            # Convert from world frame to base_link frame
            cvel_w = self.data.cvel[body_id].copy()  # [wx, wy, wz, vx, vy, vz]
            ang_vel_w = cvel_w[0:3]  # angular velocity in world frame
            lin_vel_w = cvel_w[3:6]  # linear velocity in world frame
            p_base, R_base = self._get_base_link_tf()
            ang_vel_local = R_base.T @ ang_vel_w
            lin_vel_local = R_base.T @ lin_vel_w
            self._ctrl_states_buf[slot + 6:slot + 9] = lin_vel_local.astype(np.float32)
            self._ctrl_states_buf[slot + 9:slot + 12] = ang_vel_local.astype(np.float32)

        l_id = getattr(self, "_l_ee_body_id", -1)
        r_id = getattr(self, "_r_ee_body_id", -1)
        _write_ee(l_id, base_idx)
        _write_ee(r_id, base_idx + 12)

    def _do_reset_from_shm(self):
        """Perform a physics reset triggered by the ctrl SHM reset flag."""
        mujoco.mj_resetData(self.model, self.data)
        if self.init_qpos is not None:
            self.data.qpos[: len(self.init_qpos)] = self.init_qpos
        # If body joints are position-actuated, sync ctrl targets to init_qpos.
        self._sync_body_position_ctrl_to_qpos()
        self._apply_randomization()
        # Sync ctrl to qpos for ALL arm joints BEFORE _apply_reset_ee_r.
        # mj_resetData sets data.ctrl=0; after init_qpos the joints are at their
        # configured rest angles.  Without this sync, position actuators (kp=10000)
        # generate kp*(0-qpos) torques that violently snap any arm without an IK
        # target (e.g. the left arm when only reset_ee_r is configured).
        _l_ctrl = getattr(self, "_l_arm_ctrl_indices", None)
        _l_qpos = getattr(self, "_l_arm_qpos_indices", None)
        _r_ctrl = getattr(self, "_r_arm_ctrl_indices", None)
        _r_qpos = getattr(self, "_r_arm_qpos_indices", None)
        if _l_ctrl and _l_qpos:
            for ci, qi in zip(_l_ctrl, _l_qpos):
                self.data.ctrl[ci] = self.data.qpos[qi]
        if _r_ctrl and _r_qpos:
            for ci, qi in zip(_r_ctrl, _r_qpos):
                self.data.ctrl[ci] = self.data.qpos[qi]
        self._apply_reset_ee_r()
        mujoco.mj_forward(self.model, self.data)
        self._sim_time = 0.0
        self._step_count = 0
        with self._ctrl_lock:
            self._ctrl = None
        self._ctrl_actions_buf[:] = 0.0
        # In EE mode, seed the actions buffer with the current EE poses so that
        # the very first _apply_ee_actions call holds the reset configuration.
        # Without this, IK would target (0,0,0) world-frame EE (the cleared buffer),
        # producing a wild joint command and causing the arm to flail immediately
        # after reset.
        if self._control_mode == 'ee':
            _l_id = getattr(self, "_l_ee_body_id", -1)
            _r_id = getattr(self, "_r_ee_body_id", -1)
            if _l_id >= 0:
                _pl, _rl = self._world_to_base_link(
                    self.data.xpos[_l_id].copy(),
                    self.data.xmat[_l_id].reshape(3, 3).copy(),
                )
                self._ctrl_actions_buf[0:3] = _pl.astype(np.float32)
                self._ctrl_actions_buf[3:6] = _rl.astype(np.float32)
            if _r_id >= 0:
                _pr, _rr = self._world_to_base_link(
                    self.data.xpos[_r_id].copy(),
                    self.data.xmat[_r_id].reshape(3, 3).copy(),
                )
                self._ctrl_actions_buf[6:9] = _pr.astype(np.float32)
                self._ctrl_actions_buf[9:12] = _rr.astype(np.float32)
        self._write_ctrl_states()
        self._write_ctrl_ee_states()
        self._write_ctrl_info()
        self._ctrl_counter[0] += 1
        self._ctrl_reset_flag[0] = _RESET_DONE
        self.get_logger().info("Scene reset complete (ctrl SHM).")

    def cleanup_shm(self):
        """Unlink the per-env ctrl SHM.  Call once before process exit."""
        if self._ctrl_shm is not None:
            try:
                self._ctrl_shm.close()
                self._ctrl_shm.unlink()
            except Exception:
                pass
            self._ctrl_shm = None

    def _setup_control_mode(self):
        """
        Discover arm joint qpos/dof indices and set up IK solver for ee mode.

        Called once after the MuJoCo model is loaded.  Works by iterating over
        the actuators at ctrl[ctrl_offset … ctrl_offset+14) and looking up the
        corresponding joint kinematics data.
        """
        if self._ctrl_shm is None:
            # SHM not yet created (host mode); IK requires SHM mode (ee actions arrive via SHM).
            return

        if self.model.nu < self._ctrl_offset + 14:
            self.get_logger().warning(
                f"model.nu={self.model.nu} < ctrl_offset+14={self._ctrl_offset+14}; "
                f"skipping control-mode setup (arm actuators not reachable)."
            )
            return

        # Left arm:  ctrl[ctrl_offset   : ctrl_offset+7]
        # Right arm: ctrl[ctrl_offset_r : ctrl_offset_r+7]
        # ctrl_offset_r defaults to ctrl_offset+7 (contiguous layout), but can be
        # set explicitly for models like G2_t2v2 where motor actuators sit between
        # the two arms' position actuators (left-pos[24:31], motor[31:38], right-pos[38:45]).
        n_per_arm = 7
        l_ctrl = list(range(self._ctrl_offset,   self._ctrl_offset   + n_per_arm))
        r_ctrl = list(range(self._ctrl_offset_r, self._ctrl_offset_r + n_per_arm))

        def _kinematics_info(ctrl_indices):
            joint_ids, qpos_indices, dof_indices = [], [], []
            for ci in ctrl_indices:
                jid = int(self.model.actuator_trnid[ci, 0])
                joint_ids.append(jid)
                qpos_indices.append(int(self.model.jnt_qposadr[jid]))
                dof_indices.append(int(self.model.jnt_dofadr[jid]))
            return joint_ids, qpos_indices, dof_indices

        self._l_arm_ctrl_indices = l_ctrl
        self._r_arm_ctrl_indices = r_ctrl
        self._l_arm_joint_ids, self._l_arm_qpos_indices, self._l_arm_dof_indices = \
            _kinematics_info(l_ctrl)
        self._r_arm_joint_ids, self._r_arm_qpos_indices, self._r_arm_dof_indices = \
            _kinematics_info(r_ctrl)

        if self._control_mode == 'ee':
            ee_body_l = self.args.ee_body_l
            ee_body_r = self.args.ee_body_r
            self._l_ee_body_id = self._body_name_to_id.get(ee_body_l, -1)
            self._r_ee_body_id = self._body_name_to_id.get(ee_body_r, -1)
            if self._l_ee_body_id < 0:
                raise ValueError(f"EE body '{ee_body_l}' not found in MuJoCo model")
            if self._r_ee_body_id < 0:
                raise ValueError(f"EE body '{ee_body_r}' not found in MuJoCo model")
            self._ik_data = mujoco.MjData(self.model)
            self._ik_max_iter: int = self.args.ik_max_iter
            self._ik_damp: float = self.args.ik_damp
            self.get_logger().info(
                f"EE control mode | ee_l={ee_body_l}(id={self._l_ee_body_id}) "
                f"ee_r={ee_body_r}(id={self._r_ee_body_id}) "
                f"ik_iters={self._ik_max_iter} ik_damp={self._ik_damp}"
            )
        else:
            self.get_logger().info(
                f"Joint control mode | ctrl_offset={self._ctrl_offset} "
                f"gripper_l={self._gripper_ctrl_l} gripper_r={self._gripper_ctrl_r}"
            )

    def _solve_ik_arm(
        self,
        target_pos: np.ndarray,
        target_euler_xyz: np.ndarray,
        ee_body_id: int,
        dof_indices: list,
        qpos_indices: list,
        joint_ids: list,
        max_dq: float = None,
        ns_gain: float = 0.0,
    ) -> np.ndarray:
        """
        Solve IK for one arm using damped least squares Jacobian iterations.

        Uses self._ik_data (seeded from self.data.qpos) so the simulation
        state is never perturbed.  Returns the joint position targets [7].

        max_dq: if set, scales the per-iteration joint change so the largest
                component does not exceed this value (radians).  Prevents
                large first-step overshoots when the initial error is large.
        ns_gain: null-space gain for joint midpoint tracking (0=disabled).
                 When > 0, adds a null-space velocity that pulls joints toward
                 their midpoints, avoiding joint limits during IK convergence.
        """
        # Convert target orientation to quaternion wxyz via rotation matrix
        target_mat = _euler_xyz_to_matrix(*target_euler_xyz)
        target_quat = np.empty(4)
        mujoco.mju_mat2Quat(target_quat, target_mat.flatten())

        # Seed IK data from current simulation state
        np.copyto(self._ik_data.qpos, self.data.qpos)
        np.copyto(self._ik_data.ctrl, self.data.ctrl)

        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        q_neg = np.empty(4)
        q_err = np.empty(4)
        n_arm = len(dof_indices)

        # Precompute joint midpoints for null-space control
        if ns_gain > 0.0:
            q_lo = np.array([float(self.model.jnt_range[jid, 0]) for jid in joint_ids])
            q_hi = np.array([float(self.model.jnt_range[jid, 1]) for jid in joint_ids])
            q_mid = (q_lo + q_hi) / 2.0

        for _iter in range(self._ik_max_iter):
            mujoco.mj_kinematics(self.model, self._ik_data)
            mujoco.mj_comPos(self.model, self._ik_data)

            ee_pos = self._ik_data.xpos[ee_body_id].copy()
            ee_xmat = self._ik_data.xmat[ee_body_id].reshape(3, 3).copy()
            ee_quat = np.empty(4)
            mujoco.mju_mat2Quat(ee_quat, ee_xmat.flatten())

            # Position error
            dp = target_pos - ee_pos

            # Orientation error (quaternion difference → rotation vector)
            mujoco.mju_negQuat(q_neg, ee_quat)
            mujoco.mju_mulQuat(q_err, target_quat, q_neg)
            sign = 1.0 if q_err[0] >= 0.0 else -1.0
            dr = sign * 2.0 * q_err[1:4]

            err = np.concatenate([dp, dr])  # [6]

            # Jacobian at current EE position
            jacp[:] = 0.0
            jacr[:] = 0.0
            mujoco.mj_jac(self.model, self._ik_data, jacp, jacr, ee_pos, ee_body_id)

            # Extract arm DOF columns → [6, n_arm]
            J = np.vstack([jacp[:, dof_indices], jacr[:, dof_indices]])

            # Damped least squares: J^+ = J^T (J J^T + λ² I)⁻¹
            JJT = J @ J.T + self._ik_damp ** 2 * np.eye(6)
            JJT_inv_err = np.linalg.solve(JJT, err)  # [6]
            dq = J.T @ JJT_inv_err  # primary IK step [n_arm]

            # Null-space: pull joints toward their midpoints to avoid limits.
            # dq += (I - J^+ J) * ns_gain * (q_mid - q_cur)
            if ns_gain > 0.0:
                q_cur = np.array([self._ik_data.qpos[qi] for qi in qpos_indices])
                dq_null = ns_gain * (q_mid - q_cur)
                JJT_inv_J = np.linalg.solve(JJT, J)         # [6, n_arm]
                N = np.eye(n_arm) - J.T @ JJT_inv_J         # null-space projector
                dq = dq + N @ dq_null

            # Scale the step so the largest joint change does not exceed max_dq.
            if max_dq is not None:
                max_abs = np.max(np.abs(dq))
                if max_abs > max_dq:
                    dq *= max_dq / max_abs

            # Update qpos and clamp to joint limits
            for i, (qp_idx, jid) in enumerate(zip(qpos_indices, joint_ids)):
                self._ik_data.qpos[qp_idx] += dq[i]
                lo = float(self.model.jnt_range[jid, 0])
                hi = float(self.model.jnt_range[jid, 1])
                self._ik_data.qpos[qp_idx] = float(
                    np.clip(self._ik_data.qpos[qp_idx], lo, hi)
                )

        return np.array([self._ik_data.qpos[qi] for qi in qpos_indices], dtype=np.float32)

    def _apply_reset_ee_r(self):
        """
        Set right-arm joint positions via IK so the EE is at the configured reset pose.

        Called after mj_resetData + init_qpos + randomization, before mj_forward.
        Only active in 'ee' control mode when --reset-ee-r-json is provided.
        reset_ee_r is specified in base_link frame; converted to world frame for IK.

        Strategy for robust convergence:
          1. Pre-seed right arm joints to their midpoints (avoids immediately hitting
             joint limits when starting from a neutral/zero configuration).
          2. Run warm-start IK rounds with null-space joint-midpoint tracking (ns_gain)
             to continuously pull joints away from limits during convergence.
          3. Each round seeds _ik_data from the updated data.qpos so progress
             accumulates across rounds.
        """
        if self._reset_ee_r is None or not hasattr(self, "_r_ee_body_id"):
            return

        # Run kinematics on _ik_data to get correct body poses before mj_forward.
        # (self.data.xpos is stale at this point — mj_forward hasn't been called yet.)
        np.copyto(self._ik_data.qpos, self.data.qpos)
        mujoco.mj_kinematics(self.model, self._ik_data)

        # Read base_link transform directly from _ik_data (avoid touching data.xpos).
        if self._base_link_id >= 0:
            p_base = self._ik_data.xpos[self._base_link_id].copy()
            R_base = self._ik_data.xmat[self._base_link_id].reshape(3, 3).copy()
        else:
            p_base = np.zeros(3)
            R_base = np.eye(3)

        # Convert reset pose from base_link frame to world frame.
        pos_world = R_base @ self._reset_ee_r[:3] + p_base
        eul_world = _matrix_to_euler_xyz(R_base @ _euler_xyz_to_matrix(*self._reset_ee_r[3:]))

        # Pre-seed right arm joints to their midpoints so the IK starts from a
        # configuration that is away from joint limits.  This avoids the common
        # failure mode where a large initial error pushes a joint to its limit
        # on the first step, from which the Jacobian IK cannot recover.
        for qp_idx, jid in zip(self._r_arm_qpos_indices, self._r_arm_joint_ids):
            lo = float(self.model.jnt_range[jid, 0])
            hi = float(self.model.jnt_range[jid, 1])
            self.data.qpos[qp_idx] = (lo + hi) / 2.0

        # IK warm-start loop: apply result to data.qpos each round so the next
        # round seeds from the accumulated solution.  Null-space gain (small) keeps
        # joints away from hard limits without fighting the primary task too much.
        _RESET_IK_ROUNDS = 500
        for _ in range(_RESET_IK_ROUNDS):
            q_r = self._solve_ik_arm(
                pos_world, eul_world,
                self._r_ee_body_id,
                self._r_arm_dof_indices,
                self._r_arm_qpos_indices,
                self._r_arm_joint_ids,
                max_dq=0.1,
                ns_gain=0.1,
            )
            for qp_idx, val in zip(self._r_arm_qpos_indices, q_r):
                self.data.qpos[qp_idx] = float(val)

        # CRITICAL: sync data.ctrl to match the IK-computed joint positions.
        # mj_resetData sets data.ctrl=0, but _apply_reset_ee_r puts joints at
        # non-zero angles.  Position actuators (kp=10000) would then generate
        # huge torques (kp * error) that violently snap the arm back, causing
        # the arm to "flail" at the start of each episode.
        for ci, qp_idx in zip(self._r_arm_ctrl_indices, self._r_arm_qpos_indices):
            self.data.ctrl[ci] = self.data.qpos[qp_idx]

    def _apply_joint_actions(self):
        """
        Apply joint-space actions from ctrl SHM.

        action_dim=14: all arm joints (no gripper)
        action_dim=16: 14 arm joints + 2 gripper (when gripper_ctrl >= 0)
        """
        n = len(self._ctrl_actions_buf)
        has_gripper = self._gripper_ctrl_l >= 0
        arm_n = n - 2 if (has_gripper and n >= 16) else n
        # Write left arm (actions[0:7]) and right arm (actions[7:14]) separately so
        # that non-contiguous actuator layouts (e.g. G2_t2v2) are handled correctly.
        l_end = min(self._ctrl_offset   + 7, len(self.data.ctrl))
        r_end = min(self._ctrl_offset_r + 7, len(self.data.ctrl))
        np.copyto(self.data.ctrl[self._ctrl_offset:l_end],
                  self._ctrl_actions_buf[:l_end - self._ctrl_offset])
        if arm_n > 7:
            np.copyto(self.data.ctrl[self._ctrl_offset_r:r_end],
                      self._ctrl_actions_buf[7:7 + (r_end - self._ctrl_offset_r)])
        # Write gripper actions (last 2 slots when enabled)
        if has_gripper and n >= 16:
            self.data.ctrl[self._gripper_ctrl_l] = float(self._ctrl_actions_buf[arm_n])
            self.data.ctrl[self._gripper_ctrl_r] = float(self._ctrl_actions_buf[arm_n + 1])

    def _apply_ee_actions(self):
        """
        Apply end-effector space actions from ctrl SHM.

        Action layout (action_dim=14):
            [0:3]   left EE position (base_link frame, metres)
            [3:6]   left EE orientation (intrinsic XYZ Euler, radians, base_link frame)
            [6:9]   right EE position (base_link frame, metres)
            [9:12]  right EE orientation (intrinsic XYZ Euler, radians, base_link frame)
            [12]    left gripper (0=open, 0.024=closed for place_workpiece hand)
            [13]    right gripper

        Safety: if an EE position target is at the base-link origin (norm < 1 cm),
        the buffer is uninitialized or the host sent a zero action.  Skip IK for
        that arm and hold the current ctrl to avoid the arm being commanded to an
        unreachable body-interior target.
        """
        # Minimum distance from base-link origin to treat as a valid EE target.
        # Any physically reachable EE point is several centimetres away from the
        # base-link origin; zero (or near-zero) means the buffer was never set.
        _EE_POS_ZERO_THRESH = 0.01  # metres

        a = self._ctrl_actions_buf
        pos_l_loc = np.array(a[0:3], dtype=np.float64)
        eul_l_loc = np.array(a[3:6], dtype=np.float64)
        pos_r_loc = np.array(a[6:9], dtype=np.float64)
        eul_r_loc = np.array(a[9:12], dtype=np.float64)

        # Left arm IK
        if np.linalg.norm(pos_l_loc) >= _EE_POS_ZERO_THRESH:
            pos_l_w, eul_l_w = self._base_link_to_world(pos_l_loc, eul_l_loc)
            q_l = self._solve_ik_arm(
                pos_l_w, eul_l_w,
                self._l_ee_body_id,
                self._l_arm_dof_indices,
                self._l_arm_qpos_indices,
                self._l_arm_joint_ids,
            )
            np.copyto(self.data.ctrl[self._ctrl_offset:self._ctrl_offset + 7], q_l)

        # Right arm IK
        if np.linalg.norm(pos_r_loc) >= _EE_POS_ZERO_THRESH:
            pos_r_w, eul_r_w = self._base_link_to_world(pos_r_loc, eul_r_loc)
            q_r = self._solve_ik_arm(
                pos_r_w, eul_r_w,
                self._r_ee_body_id,
                self._r_arm_dof_indices,
                self._r_arm_qpos_indices,
                self._r_arm_joint_ids,
            )
            np.copyto(self.data.ctrl[self._ctrl_offset_r:self._ctrl_offset_r + 7], q_r)

        # Gripper
        if self._gripper_ctrl_l >= 0 and len(a) >= 14:
            self.data.ctrl[self._gripper_ctrl_l] = float(a[12])
            self.data.ctrl[self._gripper_ctrl_r] = float(a[13])

    # ---------------------------------------------------------------------- #
    # Physics loop (runs in dedicated thread)
    # ---------------------------------------------------------------------- #

    def step(self):
        """Advance physics by one timestep."""
        # ---- Reset check (ctrl SHM path) ----
        if self._ctrl_shm is not None and self._ctrl_reset_flag[0] == _RESET_REQUESTED:
            self._do_reset_from_shm()
            return

        # ---- Apply actions ----
        if self._ctrl_shm is not None:
            if self._control_mode == 'ee':
                self._apply_ee_actions()
            else:
                self._apply_joint_actions()
        else:
            with self._ctrl_lock:
                if self._ctrl is not None:
                    np.copyto(self.data.ctrl, self._ctrl)

        mujoco.mj_step(self.model, self.data)
        self._sim_time += self.dt
        self._step_count += 1

        # ---- Write states to ctrl SHM ----
        if self._ctrl_shm is not None:
            self._write_ctrl_states()
            self._write_ctrl_ee_states()
            self._write_ctrl_info()
            self._ctrl_counter[0] += 1

        self._publish_clock()
        self._publish_joint_states()

        if self._step_count % self._render_interval == 0:
            self._publish_tf_render()

    def run_physics_loop(self, viewer=None):
        self.get_logger().info(
            f"Physics loop started (sync={self._sync_mode})."
        )
        if self._sync_mode:
            self._run_physics_sync(viewer)
        else:
            self._run_physics_free(viewer)

    def _run_physics_free(self, viewer=None):
        target_period = self.dt
        next_wall = time.perf_counter()
        while rclpy.ok():
            if viewer is not None and not viewer.is_running():
                break
            self.step()
            if viewer is not None:
                viewer.sync()
            next_wall += target_period
            sleep_time = next_wall - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _run_physics_sync(self, viewer=None):
        while rclpy.ok():
            if viewer is not None and not viewer.is_running():
                break
            if self._ctrl_mj_phase is None:
                time.sleep(0.001)
                continue
            if self._ctrl_mj_phase[0] != _MJ_GO:
                time.sleep(0.0001)
                continue
            sps = int(self._ctrl_steps_per_step[0])
            if sps < 1:
                sps = 1
            for _ in range(sps):
                self.step()
            if viewer is not None:
                viewer.sync()
            self._ctrl_mj_phase[0] = _MJ_DONE


# ---------------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeneSim MuJoCo ROS 2 physics node")
    parser.add_argument("--mjcf", required=True, help="Path to MuJoCo XML scene file")
    parser.add_argument("--namespace", default="env_0", help="ROS namespace (e.g. env_0)")
    parser.add_argument("--physics-hz", type=int, default=1000, help="Physics frequency (Hz)")
    parser.add_argument("--render-hz", type=int, default=30, help="tf_render publish rate (Hz)")
    parser.add_argument("--init-qpos-json", type=str, default="", help="JSON array of initial qpos")
    parser.add_argument(
        "--ros-domain-id", type=int, default=None,
        help="Override ROS_DOMAIN_ID for this node (applied before rclpy.init)",
    )
    parser.add_argument(
        "--shm-name", type=str, default="",
        help="Frame SHM name; when set, a per-env ctrl SHM is created for "
             "host↔node states/actions/reset (bypasses ROS2)",
    )
    parser.add_argument(
        "--state-joint-offset", type=int, default=0,
        help="Skip this many non-free joints before writing to the states buffer. "
             "Use to skip body/base joints and start from arm joints (e.g. 5 on G2).",
    )
    parser.add_argument(
        "--ctrl-offset", type=int, default=0,
        help="Offset into data.ctrl[] where host action commands are applied. "
             "e.g. 24 to target arm position actuators on G2.",
    )
    parser.add_argument(
        "--ctrl-offset-r", type=int, default=-1,
        help="ctrl[] offset for the RIGHT arm.  Default -1 = ctrl_offset+7 (contiguous). "
             "Set explicitly when left/right arm actuators are non-contiguous, "
             "e.g. 38 for G2_t2v2 which has motor actuators between the two arms.",
    )
    parser.add_argument(
        "--state-dim", type=int, default=0,
        help="Host state_dim (pos+vel count); defines where velocities start in the "
             "SHM states buffer (at state_dim//2). 0 = use full model joint count.",
    )
    parser.add_argument(
        "--action-dim", type=int, default=0,
        help="Host action_dim; number of ctrl values the host sends. "
             "0 = use model.nu (full actuator count).",
    )
    parser.add_argument(
        "--control-mode", type=str, default="joint", choices=["joint", "ee"],
        help="Control mode: 'joint' (position targets) or 'ee' (end-effector pose targets with IK).",
    )
    parser.add_argument(
        "--gripper-ctrl-l", type=int, default=-1,
        help="Index into data.ctrl[] for the left gripper actuator (-1 = no gripper control).",
    )
    parser.add_argument(
        "--gripper-ctrl-r", type=int, default=-1,
        help="Index into data.ctrl[] for the right gripper actuator (-1 = no gripper control).",
    )
    parser.add_argument(
        "--ee-body-l", type=str, default="arm_l_link7",
        help="MuJoCo body name to use as the left-arm end-effector for IK (ee mode only).",
    )
    parser.add_argument(
        "--ee-body-r", type=str, default="arm_r_link7",
        help="MuJoCo body name to use as the right-arm end-effector for IK (ee mode only).",
    )
    parser.add_argument(
        "--ik-max-iter", type=int, default=10,
        help="Number of Jacobian iterations per control step in ee mode.",
    )
    parser.add_argument(
        "--ik-damp", type=float, default=0.05,
        help="Damping factor λ for damped least squares IK in ee mode.",
    )
    parser.add_argument(
        "--reset-ee-r-json", type=str, default="",
        help="JSON array [x, y, z, roll, pitch, yaw] for the right-arm EE reset pose "
             "(ee mode only). IK is solved at every reset so the arm starts at this pose.",
    )
    parser.add_argument(
        "--randomization-cfg", type=str, default="",
        help="JSON string describing per-reset randomization config "
             "(robot_base, arm_joints, objects).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base RNG seed for randomization; env_id is added so parallel envs diverge.",
    )
    parser.add_argument(
        "--info-body-names", nargs="*", default=[],
        help="Body names whose world-frame poses are written into ctrl SHM info_buf.",
    )
    parser.add_argument(
        "--sync-mode", action="store_true",
        help="Run in synchronous mode: wait for GO signal before stepping physics.",
    )
    parser.add_argument(
        "--steps-per-step", type=int, default=33,
        help="Number of physics steps per control step in sync mode.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open MuJoCo passive (native) viewer window (needs DISPLAY / GPU)",
    )
    return parser.parse_args(argv)


def _shutdown_rclpy_safely():
    """Avoid RCLError when Ctrl+C or destructor already shut down the context."""
    try:
        rclpy.shutdown()
    except Exception:
        pass


def main(argv=None):
    args = parse_args(argv)

    if args.ros_domain_id is not None:
        import os as _os
        _os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)

    rclpy.init()
    node = MuJoCoRosNode(args)

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def _run_headless():
        physics_thread = threading.Thread(target=node.run_physics_loop, daemon=True)
        physics_thread.start()
        node.get_logger().info(
            "Headless mode: no MuJoCo window. Set headless: false in env YAML to enable viewer for env_0."
        )
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            node.cleanup_shm()
            node.destroy_node()
            _shutdown_rclpy_safely()

    if not args.viewer:
        _run_headless()
        return

    # Passive viewer: GLFW runs on main thread; physics + viewer.sync in worker.
    try:
        from mujoco import viewer as mj_viewer
    except Exception as e:  # pragma: no cover
        node.get_logger().error(
            f"MuJoCo viewer unavailable (install GLFW deps / use display): {e}"
        )
        node.destroy_node()
        _shutdown_rclpy_safely()
        sys.exit(1)

    with mj_viewer.launch_passive(node.model, node.data) as viewer:
        physics_thread = threading.Thread(
            target=lambda: node.run_physics_loop(viewer),
            daemon=True,
        )
        physics_thread.start()
        try:
            while rclpy.ok() and viewer.is_running():
                try:
                    executor.spin_once(timeout_sec=0.001)
                except Exception as exc:
                    # Avoid noisy traceback on normal shutdown (e.g. container manager stop).
                    try:
                        from rclpy.executors import ExternalShutdownException
                        if isinstance(exc, ExternalShutdownException):
                            break
                    except Exception:
                        pass
                    # rclpy can also raise an RCLError during teardown when the
                    # context is already shut down. Treat this as normal shutdown.
                    try:
                        from rclpy._rclpy_pybind11 import RCLError  # type: ignore
                        if isinstance(exc, RCLError) and "context is not valid" in str(exc).lower():
                            break
                    except Exception:
                        pass
                    raise
        except KeyboardInterrupt:
            pass
        finally:
            node.cleanup_shm()
            node.destroy_node()
            _shutdown_rclpy_safely()


if __name__ == "__main__":
    main()
