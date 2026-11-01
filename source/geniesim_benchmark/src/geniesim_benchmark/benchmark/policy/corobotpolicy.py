# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import hashlib
import math
import numbers
import os
import threading
import time
import pickle
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict

from .base import BasePolicy
from geniesim_benchmark.plugins.logger import Logger
from geniesim_benchmark.utils.comm.retry import is_transient_error, run_with_inference_retry
from geniesim_benchmark.utils.generalization_utils import apply_camera_image_augmentation
from collections import deque
import numpy as np
import cv2
import msgpack
import websockets.sync.client
from scipy.spatial.transform import Rotation as R
from geniesim_benchmark.utils.infer_post_process import process_action, get_arm_states
from geniesim_benchmark.utils.name_utils import ROBOT_CONFIGS, DEFAULT_ROBOT_CONFIG
from geniesim_benchmark.utils.ikfk_utils import get_shared_ikfk_solver
from geniesim_benchmark.utils.comm.websocket_client import ws_connect_compat
from geniesim_benchmark.benchmark.shared_control import (
    ACTUATION_CONTRACT_REVISION,
    ControlError,
    ControlReplyGuard,
    PROTOCOL_VERSION,
    control_write_is_current,
)

logger = Logger()

ROOT_DIR = os.environ.get("SIM_REPO_ROOT")

_OPEN_TIMEOUT_SEC = 30
_PING_INTERVAL_SEC = 20
_PING_TIMEOUT_SEC = 60
# Application-level cap on a single recv(): the ping/pong keepalive above only
# proves the TCP connection is alive, not that the server is still working on
# our request. Without this, a server that stalls mid-inference (e.g. model
# forward hangs) blocks the single-threaded main loop inside recv() forever,
# so it never reaches env.step() — which is where the wallclock episode
# timeout is actually checked. This turns that silent hang into a
# TimeoutError that run_with_inference_retry already treats as transient.
_RECV_TIMEOUT_SEC = 90

# Control-mode episode_id length: sha256 hexdigest truncated to this many hex chars.
_EPISODE_ID_HASH_LEN = 64


class StaleControlResult(Exception):
    """A well-formed result superseded by a newer control event."""


class ControlShutdownRequested(Exception):
    """The control runtime closed an in-flight inference during shutdown."""


class ControlObservationStalled(RuntimeError):
    """A required post-hold observation generation never became current."""


def _is_finite_real(value):
    return not isinstance(value, bool) and isinstance(value, numbers.Real) and math.isfinite(value)


def _transform_pose_to_frame(pose_xyzquat, tf_mat):
    """Transform an EEF pose from base_link to arm_base_link.

    Args:
        pose_xyzquat: [x, y, z, qx, qy, qz, qw] in base_link (xyzw quaternion).
        tf_mat: 4x4 homogeneous transform (arm_base_link -> base_link).
            Inverted internally to apply the base_link -> arm_base_link direction.

    Returns:
        [x, y, z, qx, qy, qz, qw] in arm_base_link (xyzw quaternion).
    """
    inv_tf = np.eye(4)
    inv_tf[:3, :3] = tf_mat[:3, :3].T
    inv_tf[:3, 3] = -inv_tf[:3, :3] @ tf_mat[:3, 3]

    pos_bl = np.array(pose_xyzquat[:3])
    rot_bl = R.from_quat(pose_xyzquat[3:7]).as_matrix()

    pos_abl = inv_tf[:3, :3] @ pos_bl + inv_tf[:3, 3]
    rot_abl = inv_tf[:3, :3] @ rot_bl

    quat_abl_xyzw = R.from_matrix(rot_abl).as_quat()

    return np.concatenate([pos_abl, quat_abl_xyzw])



def _eef_xyzrpy_in_arm_base(pose_xyzquat, declared_frame, arm_base_tf):
    """Convert one declared EEF pose to the arm-base xyz/rpy IK convention."""

    if declared_frame == "arm_base_link":
        pose = np.asarray(pose_xyzquat, dtype=np.float64)
    elif declared_frame == "base_link" and arm_base_tf is not None:
        pose = _transform_pose_to_frame(pose_xyzquat, arm_base_tf)
    elif declared_frame == "base_link":
        raise RuntimeError(
            "EEF_ABS action declared in base_link but arm_base_transform is unavailable; cannot convert to the IK frame"
        )
    else:
        raise RuntimeError(f"Unsupported EEF frame: {declared_frame}")
    return np.concatenate(
        [
            pose[:3],
            R.from_quat(pose[3:7], scalar_first=False).as_euler("xyz"),
        ]
    )



class CoRobotPolicy(BasePolicy):
    def __init__(
        self,
        task_name,
        host_ip,
        port,
        sub_task_name="",
        debug=False,
        preview=False,
        robot_cfg="",
        realprobe=False,
        ik_backend="ik_sdk",
    ):
        super().__init__(task_name=task_name, sub_task_name=sub_task_name)
        self.ts_str = time.strftime("%Y%m%d_%H%M", time.localtime(time.time()))
        self.initialized = False
        self.preview = preview
        self.debug = debug
        # Server-selected history interval, recent-frame limit and per-camera
        # resolution, with legacy hist_frame_interval compatibility.
        self._hist_frame_interval = 0
        self._hist_last_n = 0
        self._hist_resolution = {}
        # Benchmark episodes start with depth and matching calibration. Subsequent
        # requests require an enabling reply; omission disables depth. Control
        # reset retains its depth_read-only path.
        self._need_depth = True
        self._history_buffer = []
        self._since_infer = 0
        self._ws_uri = f"ws://{host_ip}:{port}" if port is not None else f"ws://{host_ip}"
        self._ws = None
        self.infer_cnt = 0
        # Lifetime accepted action batches, independent of horizon/debug IDs.
        # Evaluators take episode deltas; reset() must not reset this counter.
        self.policy_dispatch_count = 0
        self._camera_dirt_cache: Dict[tuple, np.ndarray] = {}
        self._current_episode_idx = 0
        self._episode_done = False
        self._task_progress = []
        self.robot_cfg = robot_cfg
        self._robot_config = ROBOT_CONFIGS.get(robot_cfg, DEFAULT_ROBOT_CONFIG)
        # Embodiment tag the server branches on; falls back to the raw cfg key.
        self._robot_type = self._robot_config.get("robot_type", robot_cfg)
        self._eef_type = self._robot_config.get("eef_type", "")
        # realprobe evals target the real robot, so drop the "_sim" suffix
        # (e.g. g2a_sim -> g2a).
        if realprobe and self._robot_type.endswith("_sim"):
            self._robot_type = self._robot_type[: -len("_sim")]
        self._label_state = self._robot_config["label_state"]
        self._process_gripper_action = self._robot_config["process_gripper_action"]
        self._arm_dim = len(self._robot_config.get("arm_joints", [])) or 14
        self._gripper_dim = len(self._robot_config.get("gripper_joints", [])) or 2
        # Process-wide shared IK/FK solver, used for EEF_ABS control and FK
        # observations; JOINT_ABS control does not depend on it.
        self._ikfk_solver = get_shared_ikfk_solver(
            arm_init_joint_position=[0.0] * self._arm_dim,
            head_init_position=[0.0] * 3,
            waist_init_position=[0.0] * 5,
            robot_cfg=robot_cfg,
            ik_backend=ik_backend,
        )
        self.chunk_len = 0
        # True only on the act() call that starts a freshly-received inference
        # chunk (as opposed to any later replay step within it). PiEnv uses
        # this -- together with chunk_len -- to size the WBC chassis's
        # per-chunk pose anchor to the policy's actual chunk length instead of
        # assuming a fixed one.
        self.chunk_started_this_step = False
        # Latches True once the server signals episode_done (e.g. mock server
        # exhausted its recording). Sticky for the rest of the episode.
        self.server_episode_done = False
        # Shared-control runtime state. Only the explicit control session
        # (task_benchmark sets control_runtime=True) drives these; benchmark
        # evaluation keeps the single action_buffer path below.
        self.control_runtime = False
        self._action_lock = threading.RLock()
        self._control_lock = threading.RLock()
        self._control_service_id = None
        self._control_generation = -1
        self._control_event_seq = 0
        self._control_write_revision = 0
        self._control_run_id = None
        self._control_required = False
        self._control_subscription_live = False
        self._control_pending_ack = None
        self._control_completion = None
        self._control_completion_basis = None
        self._control_target_failure = None
        self._control_last_action = None
        self._control_user_text = ""
        self._control_refresh_required = False
        self._last_observation_object_id = None
        self._refresh_after_observation_id = None
        self._required_control_observation_generation = None
        self._control_event_ws = None
        self._control_event_thread = None
        self._control_event_stop = threading.Event()
        self._control_event_ready = threading.Event()
        self._control_shutdown_requested = threading.Event()
        self._control_consumer_id = f"corobot-{id(self)}"
        self._returned_control_token = None
        self._control_stop_callback = None
        self._control_arm_plan_cancel = None
        self._control_arm_plan_state = None
        self.private_task_observer = None
        self._private_observer_failed = False
        self._private_event_lock = threading.Lock()
        self._private_event_service_id = None
        self._private_event_seq = 0
        self.raw_action_buffer = deque()
        self._chunk_raw_index = 0
        self._chunk_frame_index = 0
        self._last_action_entry = None
        self.current_action_substeps = 1
        self.raw_waypoint_started = False
        self._episode_id = self._compute_episode_id()
        self._on_reset = True

    def _start_new_chunk(self, actions):
        with self._action_lock:
            self._cancel_control_arm_plan_locked()
            self.raw_action_buffer = deque(actions)
            # infer() replaces action_buffer with a bounded deque; control
            # expands one raw waypoint at a time into an unbounded queue.
            self.action_buffer = deque()
            self.chunk_len = len(actions)
            self._chunk_raw_index = 0
            self._chunk_frame_index = 0
            self._control_refresh_required = False
            self._refresh_after_observation_id = None
            self._required_control_observation_generation = None

    def _clear_action_state_locked(self, *, refresh_required):
        self._cancel_control_arm_plan_locked()
        self.action_buffer.clear()
        self.raw_action_buffer.clear()
        self.chunk_len = 0
        self._chunk_raw_index = 0
        self._chunk_frame_index = 0
        self._last_action_entry = None
        self._control_completion = None
        self._control_completion_basis = None
        self._control_target_failure = None
        self._control_last_action = None
        self._control_refresh_required = refresh_required
        self._refresh_after_observation_id = self._last_observation_object_id if refresh_required else None
        if not refresh_required:
            self._required_control_observation_generation = None

    def _cancel_control_arm_plan_locked(self):
        cancel = self._control_arm_plan_cancel
        if cancel is not None:
            cancel.set()
        self._control_arm_plan_cancel = None
        self._control_arm_plan_state = None

    def _invalidate_control_writes_locked(
        self,
        *,
        advance_revision=True,
        reason="control_invalidated",
        source=None,
        event_seq=None,
        generation=None,
    ):
        recorder = getattr(self, "control_evidence", None)
        if recorder is not None and self._control_completion is not None:
            recorder.interrupted(
                deepcopy(self._control_completion),
                reason=reason,
                generation=self._control_generation if generation is None else generation,
                source=source,
                event_seq=event_seq,
            )
        self._clear_action_state_locked(refresh_required=True)
        if advance_revision:
            self._control_write_revision += 1
        self._control_pending_ack = None
        self.server_episode_done = False

    def control_target_failure(self):
        """Return the current failed-target marker for physical completion."""

        with self._action_lock:
            with self._control_lock:
                return deepcopy(self._control_target_failure)

    def _capture_control_completion_basis(self, observation, token):
        """Bind an arm or gripper goal to the sample before its first write."""

        from geniesim_benchmark.control_observation import filter_measured

        basis = filter_measured(observation.get("measured"))
        if not basis.get("sample") or not basis.get("profile_revision"):
            return
        with self._action_lock:
            with self._control_lock:
                metadata = self._control_completion or {}
                operations = set((metadata.get("operations") or {}).values())
                current = (
                    len(operations) == 1
                    and bool(operations & {"arm.pose", "arm.translate", "arm.rotate", "gripper.move"})
                    and self._control_completion_basis is None
                    and control_write_is_current(
                        token,
                        service_id=self._control_service_id,
                        run_id=self._control_run_id,
                        generation=self._control_generation,
                        refresh_required=self._control_refresh_required,
                    )
                    and token.get("write_revision") == self._control_write_revision
                )
                if current:
                    self._control_completion_basis = deepcopy(basis)

    def _recover_control_target_failure(self, raw_entry, reason, *, deadline=None):
        """Fence a failed G2 target while retaining its ACK identity."""

        token = raw_entry.get("_control")
        if not isinstance(token, dict):
            raise RuntimeError(
                "control runtime cannot recover kinematics rejection without a validated dispatch identity"
            )
        hold_token = None
        marker = None
        with self._action_lock:
            with self._control_lock:
                if self._control_shutdown_requested.is_set():
                    raise ControlShutdownRequested("control runtime shutdown requested")
                current = (
                    control_write_is_current(
                        token,
                        service_id=self._control_service_id,
                        run_id=self._control_run_id,
                        generation=self._control_generation,
                        refresh_required=False,
                    )
                    and token.get("write_revision") == self._control_write_revision
                )
                if not current:
                    return None
                metadata = deepcopy(self._control_completion)
                acknowledgement = self._control_pending_ack
                if (
                    not isinstance(metadata, dict)
                    or not isinstance(acknowledgement, dict)
                    or metadata.get("dispatch_id") != token.get("dispatch_id")
                    or acknowledgement.get("dispatch_id") != token.get("dispatch_id")
                ):
                    raise RuntimeError(
                        "control runtime cannot recover kinematics rejection without correlated completion state"
                    )
                if deadline is None and isinstance(self._control_arm_plan_state, dict):
                    deadline = self._control_arm_plan_state.get("deadline")
                self.action_buffer.clear()
                self.raw_action_buffer.clear()
                self.chunk_len = 0
                self._chunk_raw_index = 0
                self._chunk_frame_index = 0
                self._last_action_entry = None
                self._control_completion_basis = None
                self._cancel_control_arm_plan_locked()
                self._control_write_revision += 1
                hold_token = deepcopy(token)
                hold_token["write_revision"] = self._control_write_revision
                marker = {
                    "_control_execution": "failed_target",
                    "_control_failure_reason": reason,
                }
                self._returned_control_token = hold_token
                self._control_last_action = deepcopy(marker)
                self._control_target_failure = {
                    "reason": reason,
                    "token": deepcopy(hold_token),
                    "metadata": metadata,
                }
                if deadline is not None:
                    self._control_target_failure["deadline"] = deadline
                self._control_refresh_required = True
                self._refresh_after_observation_id = self._last_observation_object_id
                self._required_control_observation_generation = None
                self.server_episode_done = False
        self._dispatch_control_hold(hold_token, reason)
        return marker

    _monotonic = staticmethod(time.monotonic)
    _generation_wait = None

    @classmethod
    def _control_generation_wait_s(cls):
        # Wall-clock budget for a hold's physical confirmation. Simulation can
        # run well below real time, so allow four simulated goal budgets.
        return 4.0 * cls._control_action_timeout_s()

    def _check_generation_wait_locked(self):
        """Bound the wait for a required generation; never spin silently."""
        required = self._required_control_observation_generation
        now = self._monotonic()
        if self._generation_wait is None or self._generation_wait[0] != required:
            self._generation_wait = (required, now)
            return
        if now - self._generation_wait[1] > self._control_generation_wait_s():
            raise ControlObservationStalled(
                f"control observation generation {required} was not confirmed within "
                f"{self._control_generation_wait_s():.0f} s"
            )

    def _observation_is_fresh_after_control_event(self, observation):
        observation_id = id(observation)
        with self._action_lock:
            if not self._control_refresh_required:
                self._last_observation_object_id = observation_id
                return True
            if self._required_control_observation_generation is not None:
                observed_generation = observation.get("_control_observation_generation")
                if (
                    isinstance(observed_generation, bool)
                    or not isinstance(observed_generation, int)
                    or observed_generation < self._required_control_observation_generation
                ):
                    self._last_observation_object_id = observation_id
                    self._check_generation_wait_locked()
                    return False
                self._generation_wait = None
                self._control_refresh_required = False
                self._refresh_after_observation_id = None
                self._required_control_observation_generation = None
                self._last_observation_object_id = observation_id
                return True
            if self._refresh_after_observation_id is None:
                self._refresh_after_observation_id = observation_id
            if observation_id == self._refresh_after_observation_id:
                self._last_observation_object_id = observation_id
                return False
            self._control_refresh_required = False
            self._refresh_after_observation_id = None
            self._last_observation_object_id = observation_id
            return True

    def _expand_next_raw_waypoint(self):
        with self._action_lock:
            raw_entry = self.raw_action_buffer.popleft()
            self.chunk_started_this_step = self._chunk_raw_index == 0
            self._chunk_raw_index += 1
            # Control writes each raw (or planned) waypoint as-is; there is no
            # waypoint interpolation on this path.
            self.action_buffer.append(dict(raw_entry))
            self._last_action_entry = dict(raw_entry)
            self.current_action_substeps = 1
            self.raw_waypoint_started = True

    @staticmethod
    def _control_simulation_time(observation):
        from geniesim_benchmark.control_observation import filter_measured

        sample = filter_measured(observation.get("measured")).get("sample") or {}
        return sample.get("simulation_time_s")

    @staticmethod
    def _control_action_timeout_s(profile=None):
        if profile is None:
            from geniesim_benchmark.g2_90d_profile import G2_90D_PROFILE

            profile = G2_90D_PROFILE
        timeout = float((profile.get("chassis", {}).get("controller") or {}).get("goal_timeout_s", 15.0))
        if not math.isfinite(timeout) or timeout <= 0:
            raise RuntimeError("control completion budget must be finite and positive")
        return timeout

    @staticmethod
    def _control_deadline_failure_reason(timeout_cause):
        return (
            "physical_sample_invalid"
            if timeout_cause in {"simulation_clock_invalid", "wall_stalled"}
            else "physical_target_timeout"
        )

    def _control_token_is_current_locked(self, token):
        acknowledgement = self._control_pending_ack or {}
        return (
            control_write_is_current(
                token,
                service_id=self._control_service_id,
                run_id=self._control_run_id,
                generation=self._control_generation,
                refresh_required=self._control_refresh_required,
            )
            and token.get("write_revision") == self._control_write_revision
            and acknowledgement.get("dispatch_id") == token.get("dispatch_id")
        )

    def _prepare_control_body_plan(self, observation):
        # Imported lazily: g2_lula_ikfk loads the G2_90d profile at import time.
        from geniesim_benchmark.utils.g2_lula_ikfk import KinematicsTargetRejected

        with self._action_lock:
            if not self.raw_action_buffer:
                return None
            raw = self.raw_action_buffer[0]
            if "_control_body" not in raw or raw.get("_control_planned_waypoint"):
                return None
            token = raw["_control"]
        model = self._ikfk_solver.body_kinematics
        measured = observation["measured"]
        state = {
            name: value
            for group in ("waist", "head")
            for name, value in zip(measured["joints"][group]["names"], measured["joints"][group]["positions"])
        }
        period = self._ikfk_solver.waypoint_period_s
        count = max(1, math.ceil(raw["_control_direct"]["duration_s"] / period))
        entries = []

        def cancelled():
            with self.control_action_guard(token) as current:
                return not current

        for _ in range(count):
            if cancelled():
                return "discarded"
            from geniesim_benchmark.body_kinematics import BodyPlanningCancelled

            try:
                state, _residual = model.solve(raw["_control_body"], state, period_s=period, should_cancel=cancelled)
            except BodyPlanningCancelled:
                return "discarded"
            except ValueError as error:
                logger.warning(f"Control body target rejected by kinematics planning: {error}")
                from geniesim_benchmark.control_completion import PhysicalCompletionDeadline

                self._recover_control_target_failure(
                    raw,
                    KinematicsTargetRejected.reason,
                    deadline=PhysicalCompletionDeadline(self._control_action_timeout_s()),
                )
                return "failed"
            entry = dict(raw)
            entry["_control_planned_waypoint"] = True
            if "head" in raw["_control_body"]:
                entry["head"] = [state[name] for name in model.head_joints]
            if "torso" in raw["_control_body"]:
                entry["waist"] = [state[name] for name in model.torso_joints]
            entries.append(entry)
        from geniesim_benchmark.control_completion import PhysicalCompletionDeadline

        deadline = PhysicalCompletionDeadline(self._control_action_timeout_s())
        with self._action_lock:
            with self._control_lock:
                if (
                    not self._control_token_is_current_locked(token)
                    or not self.raw_action_buffer
                    or self.raw_action_buffer[0] is not raw
                ):
                    return "discarded"
                self.raw_action_buffer = deque(entries)
                self.action_buffer.clear()
                self.chunk_len = len(entries)
                self._control_arm_plan_state = {
                    "token": deepcopy(token),
                    "period_s": period,
                    "last_sim_time_s": None,
                    "next_write_sim_time_s": None,
                    "deadline": deadline,
                    "deadline_cause": None,
                }
        return "installed"

    def _prepare_control_arm_plan(self, observation):
        """Plan a semantic arm goal without holding control event locks."""

        from geniesim_benchmark.control_completion import PhysicalCompletionDeadline
        from geniesim_benchmark.utils.g2_lula_ikfk import KinematicsPlanCancelled, KinematicsTargetRejected

        with self._action_lock:
            with self._control_lock:
                if not self.raw_action_buffer:
                    return None
                raw_entry = self.raw_action_buffer[0]
                if raw_entry.get("_control_arm_execution") not in {"cartesian_side", "direct_segment"} or raw_entry.get(
                    "_control_planned_waypoint"
                ):
                    return None
                token = raw_entry.get("_control")
                if not isinstance(token, dict) or not self._control_token_is_current_locked(token):
                    return "discarded"
                if self._control_arm_plan_cancel is not None:
                    return "discarded"
                cancel = threading.Event()
                self._control_arm_plan_cancel = cancel

        side = raw_entry.get("_control_arm_side")
        direct = raw_entry.get("_control_direct")
        sides = tuple(direct["arms"]) if isinstance(direct, dict) else (side,)
        if any(part not in {"left", "right"} for part in sides):
            raise RuntimeError("semantic arm command has no validated side")
        planner_name = (
            "plan_pose_action_for_side" if raw_entry.get("_control_pose_joint_path") else "plan_eef_action_for_side"
        )
        planner = getattr(self._ikfk_solver, planner_name, None)
        if not callable(planner):
            raise RuntimeError("semantic arm command requires a selected-side trajectory planner")
        measured_arm = np.asarray(get_arm_states(observation["states"], self._arm_dim), dtype=np.float64)
        if measured_arm.shape != (self._arm_dim,) or not np.all(np.isfinite(measured_arm)):
            raise RuntimeError("semantic arm command requires a finite measured arm state")
        declared_frame = raw_entry.get("eef_frame", "base_link")
        if declared_frame != "arm_base_link":
            raise RuntimeError("control runtime accepts semantic arm targets only in arm_base_link")
        simulation_time = self._control_simulation_time(observation)
        deadline = PhysicalCompletionDeadline(
            self._control_action_timeout_s(),
            clock=lambda: time.monotonic(),
        )
        deadline.observe({"simulation_time_s": simulation_time})
        try:
            plans = {}
            for side in sides:
                eef_slice = slice(0, 7) if side == "left" else slice(7, 14)
                target = _eef_xyzrpy_in_arm_base(raw_entry["arm"][eef_slice], declared_frame, None)
                plans[side] = planner(side, target.tolist(), measured_arm.tolist(), should_cancel=cancel.is_set)
        except KinematicsPlanCancelled:
            if self._control_shutdown_requested.is_set():
                raise ControlShutdownRequested("control runtime shutdown requested") from None
            return "discarded"
        except KinematicsTargetRejected as error:
            logger.warning(f"Control target rejected by kinematics planning: {error}")
            with self._action_lock:
                if self._control_arm_plan_cancel is cancel:
                    self._control_arm_plan_cancel = None
            self._recover_control_target_failure(
                raw_entry,
                KinematicsTargetRejected.reason,
                deadline=deadline,
            )
            return "failed"

        planned_entries = []
        count = max((len(plan) for plan in plans.values()), default=1)
        dof = self._arm_dim // 2
        for index in range(count):
            full_arm = measured_arm.copy()
            for side, plan in plans.items():
                selected = np.asarray(plan[min(index, len(plan) - 1)], dtype=np.float64)
                if selected.shape != (dof,) or not np.all(np.isfinite(selected)):
                    raise RuntimeError(f"{side} arm trajectory planner returned an invalid joint state")
                arm_slice = slice(0, dof) if side == "left" else slice(dof, 2 * dof)
                full_arm[arm_slice] = selected
            planned_entry = dict(raw_entry)
            planned_entry["_control_planned_arm"] = full_arm.tolist()
            planned_entry["_control_planned_waypoint"] = True
            planned_entries.append(planned_entry)
        if not planned_entries:
            raise RuntimeError("semantic arm trajectory planner returned no joint targets")
        waypoint_period = getattr(self._ikfk_solver, "waypoint_period_s", None)
        if not _is_finite_real(waypoint_period) or waypoint_period <= 0:
            raise RuntimeError("semantic arm trajectory planner returned an invalid waypoint period")

        with self._action_lock:
            with self._control_lock:
                current = (
                    self._control_arm_plan_cancel is cancel
                    and not cancel.is_set()
                    and self.raw_action_buffer
                    and self.raw_action_buffer[0] is raw_entry
                    and raw_entry.get("_control") == token
                    and self._control_token_is_current_locked(token)
                )
                if not current:
                    if self._control_arm_plan_cancel is cancel:
                        self._control_arm_plan_cancel = None
                    return "discarded"
                remaining = list(self.raw_action_buffer)[1:]
                self.raw_action_buffer = deque(planned_entries + remaining)
                self.action_buffer.clear()
                self.chunk_len = len(self.raw_action_buffer)
                self._chunk_raw_index = 0
                self._chunk_frame_index = 0
                self._last_action_entry = None
                self._control_arm_plan_cancel = None
                self._control_arm_plan_state = {
                    "token": deepcopy(token),
                    "period_s": float(waypoint_period),
                    "last_sim_time_s": None,
                    "next_write_sim_time_s": None,
                    "deadline": deadline,
                    "deadline_cause": None,
                    "segment_duration_s": direct["duration_s"] if isinstance(direct, dict) else None,
                    "segment_start_s": None,
                }
        return "installed"

    def _planned_control_waypoint_due_locked(self, raw_entry, simulation_time):
        if not raw_entry.get("_control_planned_waypoint"):
            return True
        state = self._control_arm_plan_state
        token = raw_entry.get("_control")
        if (
            not isinstance(state, dict)
            or token != state.get("token")
            or not self._control_token_is_current_locked(token)
        ):
            return False
        timeout_cause = state["deadline"].observe({"simulation_time_s": simulation_time})
        if timeout_cause is not None:
            state["deadline_cause"] = timeout_cause
            return False
        if not _is_finite_real(simulation_time) or simulation_time < 0:
            return False
        last_sim_time = state["last_sim_time_s"]
        if last_sim_time is None:
            state["last_sim_time_s"] = float(simulation_time)
            state["next_write_sim_time_s"] = float(simulation_time) + state["period_s"]
            return False
        if simulation_time <= last_sim_time:
            return False
        state["last_sim_time_s"] = float(simulation_time)
        if simulation_time + 1e-12 < state["next_write_sim_time_s"]:
            return False
        state["next_write_sim_time_s"] = float(simulation_time) + state["period_s"]
        return True

    def _ensure_connection(self):
        """Make a single connect attempt if currently disconnected.

        Transient failures propagate so the outer retry helper counts them
        against the budget — looping here would block one infer() call
        forever and bypass the budget."""
        if self._control_shutdown_requested.is_set():
            raise ControlShutdownRequested("control runtime shutdown requested")
        if self._ws is not None:
            return
        logger.info(f"Connecting to policy server at {self._ws_uri}...")
        self._ws = ws_connect_compat(
            self._ws_uri,
            compression=None,
            max_size=None,
            open_timeout=_OPEN_TIMEOUT_SEC,
            ping_interval=_PING_INTERVAL_SEC,
            ping_timeout=_PING_TIMEOUT_SEC,
        )
        logger.info("Connected to policy server")

    def _drop_connection(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def close(self):
        self._control_event_stop.set()
        # Closing the simulator request socket ends the server-side run, so
        # new Agent goals are rejected while the event socket is being torn down.
        self._drop_connection()
        with self._control_lock:
            event_ws = self._control_event_ws
        if event_ws is not None:
            try:
                event_ws.close()
            except Exception:
                pass

    def request_control_shutdown(self, reason):
        """Fence queued control, queue one guarded hold, then unblock sockets."""

        if self._control_shutdown_requested.is_set():
            return {"status": "already_requested"}
        self._control_shutdown_requested.set()
        with self._action_lock:
            with self._control_lock:
                self._invalidate_control_writes_locked(reason=reason)
                hold_token = self._current_control_token_locked()
                hold_queued = self._control_stop_callback is not None
        try:
            self._dispatch_control_hold(hold_token, reason)
        finally:
            threading.Thread(
                target=self.close,
                daemon=True,
                name="corobot-control-shutdown",
            ).start()
        return {
            "status": "requested",
            "guarded_hold": "queued" if hold_queued else "unavailable",
        }

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _start_control_subscription(self):
        with self._control_lock:
            if self._control_event_thread is not None:
                return
            thread = threading.Thread(
                target=self._control_event_loop,
                daemon=True,
                name=f"{self._control_consumer_id}-events",
            )
            self._control_event_thread = thread
        thread.start()

    def _control_event_loop(self):
        while not self._control_event_stop.is_set():
            ws = None
            try:
                ws = ws_connect_compat(
                    self._ws_uri,
                    compression=None,
                    max_size=None,
                    open_timeout=_OPEN_TIMEOUT_SEC,
                    ping_interval=_PING_INTERVAL_SEC,
                    ping_timeout=_PING_TIMEOUT_SEC,
                )
                with self._control_lock:
                    self._control_event_ws = ws
                ws.send(
                    msgpack.packb(
                        {
                            "method": "control.subscribe",
                            "request_id": self._control_consumer_id,
                            "params": {
                                "consumer_id": self._control_consumer_id,
                                "consumer_role": "simulator",
                                "after_event_seq": self._control_event_seq,
                            },
                        }
                    )
                )
                response = ws.recv(timeout=_OPEN_TIMEOUT_SEC)
                decoded = msgpack.unpackb(response, raw=False)
                if not isinstance(decoded, dict) or decoded.get("error") or not isinstance(decoded.get("result"), dict):
                    raise RuntimeError("control.subscribe returned an invalid acknowledgement")
                snapshot = decoded["result"]
                self._reconcile_control_snapshot(snapshot)
                for event in snapshot.get("events", []):
                    if isinstance(event, dict):
                        self._handle_control_event(event)
                        self._acknowledge_control_event(ws, event)
                with self._control_lock:
                    self._control_subscription_live = True
                    self._control_event_ready.set()
                while not self._control_event_stop.is_set():
                    try:
                        raw = ws.recv(timeout=1.0)
                    except TimeoutError:
                        continue
                    if isinstance(raw, str):
                        continue
                    message = msgpack.unpackb(raw, raw=False)
                    if isinstance(message, dict) and isinstance(message.get("event"), dict):
                        self._handle_control_event(message["event"])
                        self._acknowledge_control_event(ws, message["event"])
            except Exception as e:
                if not self._control_event_stop.is_set():
                    self._mark_control_channel_failed()
                    logger.info(f"Shared control event subscription reconnecting: {type(e).__name__}: {e}")
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                with self._control_lock:
                    if self._control_event_ws is ws:
                        self._control_event_ws = None
            self._control_event_stop.wait(1.0)
        with self._control_lock:
            self._control_event_thread = None

    def _acknowledge_control_event(self, ws, event):
        ws.send(
            msgpack.packb(
                {
                    "method": "control.event_ack",
                    "request_id": f"{self._control_consumer_id}:{event.get('event_seq')}",
                    "params": {
                        "consumer_id": self._control_consumer_id,
                        "event_seq": event.get("event_seq"),
                    },
                }
            )
        )

    def _reconcile_control_snapshot(self, snapshot):
        if self._control_shutdown_requested.is_set():
            return
        service_id = snapshot.get("service_id")
        run_id = snapshot.get("run_id")
        generation = snapshot.get("generation")
        event_seq = snapshot.get("event_seq")
        if (
            snapshot.get("actuation_contract_revision") != ACTUATION_CONTRACT_REVISION
            or not isinstance(service_id, str)
            or not service_id
            or not isinstance(run_id, str)
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or isinstance(event_seq, bool)
            or not isinstance(event_seq, int)
        ):
            raise RuntimeError("control.subscribe snapshot identity is invalid")
        hold_token = None
        with self._action_lock:
            with self._control_lock:
                should_hold = self._control_required
                service_changed = self._control_service_id not in {None, service_id}
                identity_changed = (
                    service_changed
                    or (self._control_run_id is not None and self._control_run_id != run_id)
                    or generation > self._control_generation
                )
                if identity_changed:
                    self._invalidate_control_writes_locked(reason="snapshot_reconciled")
                self._control_service_id = service_id
                self._control_run_id = run_id
                self._control_generation = generation
                self._control_event_seq = event_seq
                if identity_changed and should_hold:
                    hold_token = self._current_control_token_locked()
        self._dispatch_control_hold(hold_token, "snapshot_reconciled")

    def _mark_control_channel_failed(self):
        if self._control_shutdown_requested.is_set():
            return
        hold_token = None
        with self._action_lock:
            with self._control_lock:
                self._control_subscription_live = False
                self._control_event_ready.clear()
                if not self._control_required:
                    return
                self._invalidate_control_writes_locked(reason="event_channel_failed")
                hold_token = self._current_control_token_locked()
        self._dispatch_control_hold(hold_token, "event_channel_failed")

    def domain_control_facts(self):
        """Return a nonblocking private snapshot of the subscribed control identity."""
        if not self._control_lock.acquire(blocking=False):
            return None
        try:
            return {
                "actuation_contract_revision": ACTUATION_CONTRACT_REVISION,
                "service_id": self._control_service_id,
                "run_id": self._control_run_id,
                "generation": self._control_generation,
                "event_seq": self._control_event_seq,
                "write_revision": self._control_write_revision,
                "private_events_complete": not self._private_observer_failed,
                # A normal completion hold fences writes until a fresh
                # observation; it does not invalidate the subscribed identity.
                "binding_available": bool(
                    self._control_subscription_live
                    and self._control_service_id
                    and self._control_run_id
                    and self._control_generation >= 0
                    and not self._private_observer_failed
                    and not self._control_shutdown_requested.is_set()
                ),
                "ready": bool(
                    self._control_subscription_live
                    and self._control_run_id
                    and not self._control_refresh_required
                    and not self._control_shutdown_requested.is_set()
                ),
            }
        finally:
            self._control_lock.release()

    def _offer_private_event(self, event):
        observer = getattr(self, "private_task_observer", None)
        if observer is None:
            return
        service_id = event.get("service_id")
        event_seq = event.get("event_seq")
        with self._private_event_lock:
            if service_id != self._private_event_service_id:
                self._private_event_service_id = service_id
                self._private_event_seq = 0
            if event_seq <= self._private_event_seq:
                return
            self._private_event_seq = event_seq
        try:
            observer.note_event(event)
        except Exception:
            # Private instrumentation never suppresses a control fence.
            self._private_observer_failed = True

    def _offer_private_source(self, observation, *, control=None):
        """Offer compact state, action, and acknowledgement facts without waiting."""
        observer = getattr(self, "private_task_observer", None)
        if observer is None:
            return
        snapshot = {
            "schema": "geniesim.private-source-observation.v1",
            "offered_at_ns": time.time_ns(),
            "offered_monotonic_ns": time.monotonic_ns(),
            "control": (deepcopy(control) if isinstance(control, dict) else self.domain_control_facts()),
            "observation": observation,
        }
        try:
            if observer.note_source_snapshot(snapshot) is False:
                self._private_observer_failed = True
        except Exception:
            self._private_observer_failed = True

    def _handle_control_event(self, event):
        if self._control_shutdown_requested.is_set():
            return
        if (
            event.get("schema") != PROTOCOL_VERSION
            or event.get("actuation_contract_revision") != ACTUATION_CONTRACT_REVISION
        ):
            return
        service_id = event.get("service_id")
        generation = event.get("generation")
        run_id = event.get("run_id")
        if (
            not isinstance(service_id, str)
            or not service_id
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            return
        event_type = event.get("event")
        event_seq = event.get("event_seq")
        operation = event.get("operation")
        if isinstance(event_seq, bool) or not isinstance(event_seq, int) or event_seq < 1:
            return
        self._offer_private_event(event)
        invalidates_chunk = (
            event_type
            in {
                "control.manual_override",
                "run.started",
                "run.ended",
            }
            or (event_type in {"agent.disconnected", "agent.released"} and event.get("control_invalidated") is True)
            or (
                event_type == "control.command_accepted"
                and operation in {"control.stop", "run.reset", "forward.connect", "forward.disconnect"}
            )
        )
        hold_token = None
        with self._action_lock:
            with self._control_lock:
                if self._control_service_id not in {None, service_id}:
                    self._control_event_seq = 0
                    self._control_generation = -1
                    self._control_run_id = None
                    invalidates_chunk = True
                self._control_service_id = service_id
                if event_seq <= self._control_event_seq:
                    return
                self._control_event_seq = event_seq
                if generation < self._control_generation:
                    return
                if (
                    self._control_run_id is not None
                    and run_id != self._control_run_id
                    and event_type
                    not in {
                        "run.started",
                        "run.ended",
                    }
                ):
                    return
                generation_advanced = generation > self._control_generation
                invalidated = False
                if generation_advanced:
                    self._invalidate_control_writes_locked(
                        reason=event_type,
                        source=event.get("source"),
                        event_seq=event_seq,
                        generation=generation,
                    )
                    invalidated = True
                self._control_generation = generation
                if event_type == "run.started" or self._control_run_id is None:
                    self._control_run_id = run_id
                if not invalidates_chunk and not generation_advanced:
                    return
                if not invalidated:
                    self._invalidate_control_writes_locked(
                        reason=event_type,
                        source=event.get("source"),
                        event_seq=event_seq,
                    )
                if event_type == "run.ended":
                    self._control_refresh_required = False
                    self._refresh_after_observation_id = None
                    self._required_control_observation_generation = None
                hold_token = self._current_control_token_locked()
                logger.info(
                    f"Shared control invalidated queued actions: event={event_type}, "
                    f"generation={generation}, run_id={run_id}"
                )
        self._dispatch_control_hold(hold_token, event_type)

    def set_control_stop_callback(self, callback):
        self._control_stop_callback = callback

    def _current_control_token_locked(self):
        return {
            "schema": PROTOCOL_VERSION,
            "actuation_contract_revision": ACTUATION_CONTRACT_REVISION,
            "service_id": self._control_service_id,
            "run_id": self._control_run_id,
            "generation": self._control_generation,
            "write_revision": self._control_write_revision,
        }

    def _dispatch_control_hold(self, token, reason):
        if token is not None and self._control_stop_callback is not None:
            observation_generation = self._control_stop_callback(token, self.control_stop_guard, reason)
            if isinstance(observation_generation, int) and not isinstance(observation_generation, bool):
                with self._action_lock:
                    with self._control_lock:
                        if token.get("write_revision") == self._control_write_revision:
                            self._required_control_observation_generation = observation_generation

    def _fence_control_for_hold(self, token, reason):
        """Invalidate the old write token before a guarded hold is enqueued."""

        with self._action_lock:
            with self._control_lock:
                current = (
                    control_write_is_current(
                        token,
                        service_id=self._control_service_id,
                        run_id=self._control_run_id,
                        generation=self._control_generation,
                        refresh_required=False,
                    )
                    and token.get("write_revision") == self._control_write_revision
                )
                if not current:
                    return None
                self.action_buffer.clear()
                self.raw_action_buffer.clear()
                self.chunk_len = 0
                self._chunk_raw_index = 0
                self._chunk_frame_index = 0
                self._last_action_entry = None
                self._cancel_control_arm_plan_locked()
                self._control_write_revision += 1
                hold_token = deepcopy(token)
                hold_token["write_revision"] = self._control_write_revision
                self._returned_control_token = hold_token
                self._control_last_action = {
                    "_control_execution": "hold",
                    "_control_failure_reason": reason,
                }
                self._control_refresh_required = True
                self._refresh_after_observation_id = self._last_observation_object_id
                self._required_control_observation_generation = None
                return hold_token

    @contextmanager
    def control_stop_guard(self, token):
        self._action_lock.acquire()
        self._control_lock.acquire()
        allowed = token.get("write_revision") == self._control_write_revision
        try:
            yield allowed
        finally:
            self._control_lock.release()
            self._action_lock.release()

    def _control_identity_from_result(self, inner, payload):
        metadata = inner.pop("_control", None)
        if metadata is None:
            with self._control_lock:
                if self._control_required:
                    raise RuntimeError("Rejected result without required shared control identity")
            return None
        request_id = payload.get("params", {}).get("control_request_id")
        with self._control_lock:
            newest_generation = self._control_generation
            current_run_id = self._control_run_id
            current_service_id = self._control_service_id
        try:
            identity = ControlReplyGuard.validate(
                metadata,
                request_id=request_id,
                newest_generation=newest_generation,
                current_run_id=current_run_id,
                current_service_id=current_service_id,
            )
        except ControlError as e:
            if e.code == "stale_control_result":
                raise StaleControlResult(e.message) from e
            raise RuntimeError(f"Rejected control result [{e.code}]: {e.message}") from e
        hold_token = None
        with self._action_lock:
            with self._control_lock:
                if self._control_service_id not in {None, identity.service_id}:
                    self._invalidate_control_writes_locked()
                    self._control_generation = -1
                    self._control_event_seq = 0
                    self._control_run_id = None
                    self._control_subscription_live = False
                    self._control_event_ready.clear()
                    hold_token = self._current_control_token_locked()
                self._control_service_id = identity.service_id
                self._control_required = True
        self._dispatch_control_hold(hold_token, "service_replaced")
        return identity

    def _validate_control_result_actions(self, metadata, actions):
        operations_by_id = (metadata or {}).get("operations", {})
        source = (metadata or {}).get("source")
        semantic_item = next(iter(operations_by_id.items())) if len(operations_by_id) == 1 else None
        semantic_operation = semantic_item[1] if semantic_item else None
        if (
            getattr(self, "control_runtime", False)
            and source == "forwarded"
            and any(action.get("chassis") is not None for action in actions)
        ):
            raise RuntimeError("forwarded control chassis requires an explicit wheel pulse operation")
        if (
            source != "forwarded"
            and semantic_operation
            in {
                "arm.pose",
                "arm.translate",
                "arm.rotate",
                "gripper.move",
                "head.rotate",
                "waist.rotate",
                "chassis.move",
                "chassis.turn",
                "chassis.move_distance",
                "chassis.turn_angle",
            }
            and len(actions) != 1
        ):
            raise RuntimeError("local semantic command must produce exactly one complete goal")

    def _install_control_chunk(self, identity, actions, on_accept=None, metadata=None):
        self._validate_control_result_actions(metadata, actions)
        operations_by_id = (metadata or {}).get("operations", {})
        source = (metadata or {}).get("source")
        semantic_item = next(iter(operations_by_id.items())) if len(operations_by_id) == 1 else None
        semantic_operation = semantic_item[1] if semantic_item else None
        if self._control_shutdown_requested.is_set():
            raise ControlShutdownRequested("control runtime shutdown requested")
        self._start_control_subscription()
        if not self._control_event_ready.wait(timeout=_OPEN_TIMEOUT_SEC):
            self._mark_control_channel_failed()
            raise ConnectionError("shared control event subscription is unavailable")
        with self._action_lock:
            with self._control_lock:
                if self._control_shutdown_requested.is_set():
                    raise ControlShutdownRequested("control runtime shutdown requested")
                if not self._control_subscription_live or identity.service_id != self._control_service_id:
                    raise ConnectionError("shared control event subscription identity changed")
                if identity.generation < self._control_generation:
                    raise StaleControlResult("control result was superseded before queue installation")
                if (
                    self._control_run_id is not None
                    and identity.run_id != self._control_run_id
                    and identity.generation <= self._control_generation
                ):
                    raise StaleControlResult("control result belongs to an obsolete run")
                self._control_generation = identity.generation
                self._control_event_seq = max(self._control_event_seq, identity.event_seq)
                self._control_run_id = identity.run_id
                self._control_write_revision += 1
                if on_accept is not None:
                    on_accept()
                token = identity.write_token()
                token["write_revision"] = self._control_write_revision
                operations = set(operations_by_id.values())
                if source == "forwarded":
                    execution = "command"
                elif operations == {"control.stop"}:
                    execution = "stop"
                elif not operations or all(operation.startswith("forward.") for operation in operations):
                    execution = "hold"
                else:
                    execution = "command"
                arm_side = None
                if semantic_operation == "direct.segment":
                    arm_execution = "direct_segment"
                elif source == "forwarded":
                    arm_execution = "cartesian"
                elif getattr(self, "control_runtime", False) and semantic_operation in {
                    "arm.pose",
                    "arm.translate",
                    "arm.rotate",
                }:
                    command_arguments = (metadata or {}).get("command_arguments")
                    arguments = command_arguments.get(semantic_item[0]) if isinstance(command_arguments, dict) else None
                    arm_side = arguments.get("side") if isinstance(arguments, dict) else None
                    if arm_side not in {"left", "right"}:
                        raise RuntimeError("semantic arm command has no validated side")
                    arm_execution = "cartesian_side"
                elif semantic_operation in {"arm.pose", "arm.translate", "arm.rotate"}:
                    arm_execution = "cartesian"
                else:
                    arm_execution = "hold_measured"
                chassis_pulse = None
                if getattr(self, "control_runtime", False) and semantic_operation in {
                    "chassis.move",
                    "chassis.turn",
                    "chassis.move_distance",
                    "chassis.turn_angle",
                }:
                    command_arguments = (metadata or {}).get("command_arguments")
                    chassis_pulse = (
                        command_arguments.get(semantic_item[0]) if isinstance(command_arguments, dict) else None
                    )
                    if not isinstance(chassis_pulse, dict):
                        raise RuntimeError("semantic chassis command has no validated velocity pulse")
                    chassis_pulse = {"operation": semantic_operation, **chassis_pulse}
                for action in actions:
                    action["_control"] = token
                    action["_control_execution"] = execution
                    if execution == "command":
                        action["_control_arm_execution"] = arm_execution
                        if semantic_operation in {"direct.segment", "body.segment"}:
                            action["_control_direct"] = deepcopy(metadata["command_arguments"][semantic_item[0]])
                            action["_control_direct_basis"] = deepcopy(
                                metadata.get("direct_bases", {}).get(semantic_item[0])
                            )
                        if semantic_operation == "body.segment":
                            action["_control_body"] = deepcopy(metadata["resolved_body"][semantic_item[0]])
                        if arm_side is not None:
                            action["_control_arm_side"] = arm_side
                        if semantic_operation == "arm.pose":
                            action["_control_pose_joint_path"] = True
                        if chassis_pulse is not None:
                            action["_control_chassis_pulse"] = deepcopy(chassis_pulse)
                self._start_new_chunk(actions)
                self._control_pending_ack = identity.acknowledgement()
                self._control_completion = deepcopy(metadata) if metadata else None
                self._control_completion_basis = None
                self._control_target_failure = None
                self._control_last_action = None

    @contextmanager
    def control_action_guard(self, token):
        """Linearize final robot writes against control invalidation events."""
        self._action_lock.acquire()
        self._control_lock.acquire()
        allowed = (
            (not self._control_required or (self._control_subscription_live and token is not None))
            and control_write_is_current(
                token,
                service_id=self._control_service_id,
                run_id=self._control_run_id,
                generation=self._control_generation,
                refresh_required=self._control_refresh_required,
            )
            and (token is None or token.get("write_revision") == self._control_write_revision)
        )
        try:
            yield allowed
        finally:
            self._control_lock.release()
            self._action_lock.release()

    @property
    def returned_control_token(self):
        with self._control_lock:
            return self._returned_control_token

    def set_episode_idx(self, idx):
        self._current_episode_idx = idx
        self._episode_id = self._compute_episode_id()

    def update_task_status(self, done, task_progress):
        self._episode_done = done
        self._task_progress = task_progress
        if done:
            if not getattr(self, "control_runtime", False):
                self.action_buffer.clear()
                return
            with self._action_lock:
                chunk_was_drained = not self.action_buffer and not self.raw_action_buffer
                self._clear_action_state_locked(refresh_required=False)
            if not chunk_was_drained:
                with self._control_lock:
                    self._control_pending_ack = None

    def _compute_episode_id(self):
        """Per-episode id carried by control-mode requests (control_request_id).

        sha256 of task_name|background|episode_idx|ms_timestamp, computed once
        per episode so every request within it shares the same id."""
        raw = f"{self.sub_task_name}|{self.task_name}|{self._current_episode_idx}|{int(time.time() * 1000)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_EPISODE_ID_HASH_LEN]

    def _task_evaluation(self):
        """Build a local diagnostic snapshot of the live checker tree and budgets.

        This contains privileged scoring specifications and progress. It must
        not be attached to inference requests. Native evaluation retains its
        checker objects and scoring independently of the transport payload.
        """
        entries = [e for e in (self._task_progress or []) if isinstance(e, dict) and e.get("acion_obj") is not None]
        if not entries:
            return None
        specs = {id(e["acion_obj"]): e.get("spec") for e in entries}
        root = entries[0]["acion_obj"]
        current_step = getattr(getattr(root, "env", None), "current_step", None)

        def scalar(value):
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value

        def node(act):
            progress = act.progress_info if isinstance(getattr(act, "progress_info", None), dict) else {}
            state = getattr(getattr(act, "state", None), "name", None)
            out = {
                "checker": act.__class__.__name__,
                "state": state,
                "score": scalar(progress.get("SCORE")),
                "status": progress.get("STATUS"),
            }
            spec = specs.get(id(act))
            if spec is not None:
                out["spec"] = spec
            if act.__class__.__name__ == "StepOut":
                max_step = int(act.max_step)
                budget = {"max_steps": max_step}
                if state == "RUNNING" and current_step is not None:
                    elapsed = max(int(current_step - act.ref_step), 0)
                    budget.update(elapsed_steps=elapsed, remaining_steps=max(max_step - elapsed, 0))
                elif state == "INIT":
                    budget["remaining_steps"] = max_step
                elif state in ("FINISHED", "CANCELED"):
                    budget.update(remaining_steps=0, exhausted=bool(getattr(act, "_done_flag", False)))
                out["budget"] = budget
            children = getattr(act, "_actions", None)
            if children:
                out["children"] = [node(child) for child in children]
            return out

        return {
            "current_step": None if current_step is None else int(current_step),
            "step_unit": "environment action steps; one issued action row is one step",
            "tree": node(root),
        }

    @staticmethod
    def _extract_scores(task_progress):
        scores = []
        ignored = {"ActionList", "ActionSetWaitAny", "StepOut"}
        for item in task_progress:
            cls = item.get("class_name", "")
            if cls in ignored:
                continue
            prog = item.get("progress") or {}
            entry = {
                "name": cls,
                "score": prog.get("SCORE", 0) if isinstance(prog, dict) else 0,
                "status": prog.get("STATUS", "PENDING") if isinstance(prog, dict) else "PENDING",
            }
            scores.append(entry)
        return scores

    @staticmethod
    def _encode_image_jpeg(image_rgb: np.ndarray, quality: int = 95) -> dict:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return {
            "encoding": "JPEG",
            "image_data": buf.tobytes(),
            "height": image_rgb.shape[0],
            "width": image_rgb.shape[1],
        }

    @staticmethod
    def _resize_half(image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        return cv2.resize(image_rgb, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _resize_to(image_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
        # Never upscale or exceed the source resolution: a server-specified
        # `resolution` is a *downscale* target, and clamping here also bounds
        # the memory/spend an absurd request could otherwise trigger.
        h, w = image_rgb.shape[:2]
        width = min(max(1, width), w)
        height = min(max(1, height), h)
        return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _encode_depth(depth_map: np.ndarray, scale: int) -> dict:
        """Encode metre depth with per-pixel uniform noise: +/-2% up to 1 m, +/-3% beyond."""
        depth = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        noise_limit = np.where(depth <= 1.0, 0.02, 0.03)
        # Keep the simulator's global random sequence independent of transport noise.
        depth = depth * (1.0 + np.random.default_rng().uniform(-noise_limit, noise_limit))
        depth = np.clip(depth * scale, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        return {
            "encoding": "RAW_UINT16",
            "image_data": depth.tobytes(),
            "height": depth.shape[0],
            "width": depth.shape[1],
        }

    def _split_states(self, states):
        if isinstance(states, dict):
            arm = list(states["left_arm"]) + list(states["right_arm"])
            gripper = list(states["left_gripper"]) + list(states["right_gripper"])
            chassis = list(states.get("chassis", []))
            return arm, gripper, list(states.get("waist", [])), list(states.get("head", [])), chassis
        # legacy flat list: arm + gripper + waist + head (no chassis)
        s = list(states)
        arm = s[0 : self._arm_dim]
        gripper = s[self._arm_dim : self._arm_dim + self._gripper_dim]
        remaining = s[self._arm_dim + self._gripper_dim :]
        n = len(remaining)
        if n >= 5:
            waist = remaining[:5]
            head = remaining[5:]
        elif n >= 2:
            waist = remaining[:2]
            head = remaining[2:]
        else:
            waist = remaining
            head = []
        return arm, gripper, waist, head, []

    def need_depth(self):
        """Whether the env should capture depth + calibration for the next payload."""
        return getattr(self, "_need_depth", False)

    def need_infer(self):
        # Force a render either when a new inference is due, or one step ahead
        # of a history-capture step so the env produces fresh images for it.
        if self.inference_due():
            return True
        if self._hist_frame_interval > 0 and (self._since_infer + 1) % self._hist_frame_interval == 0:
            return True
        return False

    def inference_due(self):
        # True only on the chunk's last step (buffer just emptied), i.e. the
        # observation that feeds the next fresh inference. Unlike need_infer(),
        # this excludes the history-capture render steps.
        if getattr(self, "control_runtime", False):
            # Control also waits on the raw queue and on a post-event refresh.
            with self._action_lock:
                return self._control_refresh_required or (not self.raw_action_buffer and not self.action_buffer)
        return len(self.action_buffer) == 0

    # Payload camera key -> obs["images"] key.
    _FRAME_CAMERAS = {"head": "head", "hand_left": "left_hand", "hand_right": "right_hand"}

    def _encode_frame(self, images, cameras=None):
        """Encode the given obs images into payload frames.

        `cameras` selects which payload cameras to include (keys of
        `_FRAME_CAMERAS`); defaults to all three.
        """
        cameras = cameras if cameras is not None else self._FRAME_CAMERAS
        return {cam: self._encode_image_jpeg(images[self._FRAME_CAMERAS[cam]]) for cam in cameras}

    _AUGMENTATIONS = ("camera_noise", "camera_occlusion", "camera_drop_frame")

    def _attach_depth_and_calibration(self, payload, obs, gen_config):
        """Publish uint16 millimetre depth and its camera calibration when asked.

        Depth and RGB share one capture, so both carry the calibration's
        snapshot_id; a client can reject a mismatched trio instead of
        backprojecting one frame's pixels with another frame's pose.
        """
        if not getattr(self, "_need_depth", False):
            return
        depth = obs.get("depth") or {}
        calibrations = obs.get("calibration") or {}
        images = payload["params"]["images"]
        augmented = bool(gen_config) and any(
            gen_config.get(key, {}).get("enable", False) for key in self._AUGMENTATIONS
        )
        for camera, source_camera in self._FRAME_CAMERAS.items():
            calibration = deepcopy(
                calibrations.get(source_camera, {"available": False, "reason": "sensor calibration not supplied"})
            )
            if augmented:
                # Augmented pixels no longer match the rendered geometry.
                calibration.update(
                    available=False,
                    rgb_depth_aligned=False,
                    reason="camera augmentation alignment not verified",
                )
            images[camera]["calibration"] = calibration
            images[camera]["snapshot_id"] = calibration.get("snapshot_id")
            sample = depth.get(source_camera)
            if sample is None or getattr(sample, "size", 1) == 0:
                continue
            images[camera + "_depth"] = self._encode_depth(sample, 1000)
            images[camera + "_depth"]["snapshot_id"] = calibration.get("snapshot_id")

    def _capture_history(self, obs, gen_config):
        images = obs.get("images") or {}
        if images.get("head") is None:
            return
        if gen_config is not None:
            images = apply_camera_image_augmentation(self._camera_dirt_cache, deepcopy(images), gen_config)
        if self.debug:
            self._dump_history_frame(images, len(self._history_buffer))
        resized = {}
        for payload_cam, obs_cam in self._FRAME_CAMERAS.items():
            img = images[obs_cam]
            target = self._hist_resolution.get(payload_cam)
            if target is not None:
                img = self._resize_to(img, target[0], target[1])
            elif payload_cam in ("hand_left", "hand_right"):
                # Default (no server-specified resolution): hand cameras are
                # higher-resolution than head; downscale only those to keep
                # the history payload small over the network.
                img = self._resize_half(img)
            resized[obs_cam] = img
        self._history_buffer.append(self._encode_frame(resized))

    def _dump_history_frame(self, images, frame_idx):
        debug_dir = os.path.join(ROOT_DIR, "debug_history", f"chunk_{self.infer_cnt:04d}")
        os.makedirs(debug_dir, exist_ok=True)
        for cam in ("head", "left_hand", "right_hand"):
            cv2.imwrite(
                os.path.join(debug_dir, f"frame_{frame_idx:03d}_step_{self._since_infer:03d}_{cam}.png"),
                cv2.cvtColor(images[cam], cv2.COLOR_RGB2BGR),
            )
        logger.info(f"[History] chunk={self.infer_cnt} frame={frame_idx} step={self._since_infer} -> {debug_dir}")

    def _build_end_pose(self, obs):
        """Build end_pose in base_link frame from FK-computed EEF (arm_base_link frame).

        Returns None if eef or arm_base_transform is unavailable. The control
        runtime instead exposes only the measured model-FK endpoints in
        arm_base_link.
        """
        if getattr(self, "control_runtime", False):
            from geniesim_benchmark.control_observation import filter_measured

            measured = filter_measured(obs.get("measured"))
            poses = measured.get("arm_end_poses", {})
            kinematics = measured.get("kinematics") or {}
            sides = poses.get("arm_base_link", {})
            if (
                not measured.get("profile_name")
                or kinematics.get("basis") != "robot_model_fk"
                or kinematics.get("frame") != "arm_base_link"
                or not sides
            ):
                raise RuntimeError(
                    "control runtime requires at least one model-FK arm endpoint in arm_base_link"
                )
            return {
                "arm_base_link": {
                    side
                    + "_arm": {
                        "position": pose["position"],
                        "orientation": pose["orientation_xyzw"],
                    }
                    for side, pose in sides.items()
                }
            }
        eef = obs.get("eef")
        arm_base_tf = obs.get("arm_base_transform")
        if eef is None or arm_base_tf is None:
            return None

        def _to_base_link(eef_wxyz):
            pos_abl = np.asarray(eef_wxyz[:3], dtype=np.float64)
            quat_wxyz = eef_wxyz[3:7]
            rot_abl = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
            pos_bl = arm_base_tf[:3, :3] @ pos_abl + arm_base_tf[:3, 3]
            rot_bl = arm_base_tf[:3, :3] @ rot_abl
            quat_xyzw = R.from_matrix(rot_bl).as_quat()
            return {
                "position": pos_bl.tolist(),
                "orientation": quat_xyzw.tolist(),
            }

        def _to_arm_base_link(eef_wxyz):
            # FK EEF is already in arm_base_link; just expose it (orientation wxyz -> xyzw).
            # The XR-0 server reads the proprio state EEF from this frame (trained in arm_base_link).
            quat_wxyz = eef_wxyz[3:7]
            return {
                "position": np.asarray(eef_wxyz[:3], dtype=np.float64).tolist(),
                "orientation": np.asarray(
                    [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
                ).tolist(),
            }

        return {
            "base_link": {
                "left_arm": _to_base_link(eef["left"]),
                "right_arm": _to_base_link(eef["right"]),
            },
            "arm_base_link": {
                "left_arm": _to_arm_base_link(eef["left"]),
                "right_arm": _to_arm_base_link(eef["right"]),
            },
        }

    def _pre_process_obs(self, obs, gen_config):
        obs = deepcopy(obs)
        self._label_state(obs, self._robot_config)
        if gen_config is not None:
            obs["images"] = apply_camera_image_augmentation(self._camera_dirt_cache, obs["images"], gen_config)
        return obs

    def _control_sensor_fields(self, obs, user_text, *, private_ack=None):
        """One sensor encoder for control inference and in-progress observation updates.

        RGB is always published; depth with its calibration travels separately
        and only while the control service has a pending control.depth_read.
        No raw chassis states are published.
        """
        from geniesim_benchmark.control_observation import control_depth_capture, filter_camera_samples, filter_measured

        arm, gripper, waist, head, _chassis = self._split_states(obs["states"])
        now = time.time_ns()
        params = {
            "timestamps": {"head": now, "states": now},
            "images": {},
            "states": {
                "head_joint_states": head,
                "arm_joint_states": arm,
                "waist_joint_states": waist,
                "gripper_states": gripper,
                "end_pose": self._build_end_pose(obs),
            },
            "user_text": user_text,
            "robot_type": self._robot_type,
            "ee_type": self._eef_type,
        }
        if self._eef_type == "wuji":
            params["states"]["hand_joint_states"] = gripper
        measured = filter_measured(obs.get("measured"))
        if measured:
            params["measured"] = measured
            if measured.get("sample"):
                params["timestamps"]["states"] = measured["sample"]["captured_at_ns"]
        samples = {}
        for camera, wire_name in {"head": "head", "left_hand": "hand_left", "right_hand": "hand_right"}.items():
            image = (obs.get("images") or {}).get(camera)
            if image is not None and image.size:
                params["images"][wire_name] = self._encode_image_jpeg(image)
            sample = (obs.get("camera_samples") or {}).get(camera)
            if sample:
                samples[wire_name] = sample
                params["timestamps"][wire_name] = sample["captured_at_ns"]
        if samples:
            params["camera_samples"] = filter_camera_samples(samples)
        depth_capture = control_depth_capture(obs)
        if depth_capture:
            params["depth_capture"] = depth_capture
        if getattr(self, "private_task_observer", None) is not None:
            private = {
                "timestamps": {
                    "states": params["timestamps"]["states"],
                },
                "states": deepcopy(params["states"]),
                "user_text": params["user_text"],
                "robot_type": params["robot_type"],
                "ee_type": params["ee_type"],
                "episode_id": self._episode_id,
                "on_reset": self._on_reset,
            }
            if "measured" in params:
                private["measured"] = deepcopy(params["measured"])
            if private_ack is not None:
                private["control_ack"] = deepcopy(private_ack)
            self._offer_private_source(private, control=self.domain_control_facts())
        return params

    def get_payload(self, obs, task_instruction, gen_config, user_text="", task_segments=None):
        obs = self._pre_process_obs(obs, gen_config)
        if getattr(self, "control_runtime", False):
            # Control sends only filtered sensors plus the operator's explicit
            # user_text; benchmark prompt/progress fields are not part of it.
            from geniesim_benchmark.benchmark.shared_control import filter_observation

            with self._control_lock:
                private_ack = deepcopy(self._control_pending_ack)
            fields = self._control_sensor_fields(obs, user_text, private_ack=private_ack)
            sensors = filter_observation(fields)
            if fields.get("depth_capture"):
                sensors["depth_capture"] = fields["depth_capture"]
            sensors.update(
                episode_id=self._episode_id,
                control_request_id=f"{self._episode_id}:{self.infer_cnt}",
                on_reset=self._on_reset,
            )
            if private_ack is not None:
                sensors["control_ack"] = private_ack
            self._on_reset = False
            return {"method": "infer", "params": sensors}

        arm_states, gripper_states, waist_states, head_states, chassis_states = self._split_states(obs["states"])

        ts_ns = time.time_ns()

        # Checker requirements, rules and live progress remain local to evaluation.
        payload = {
            "method": "infer",
            "params": {
                "timestamps": {
                    "head": ts_ns,
                    "states": ts_ns,
                },
                "images": {
                    "head": self._encode_image_jpeg(obs["images"]["head"]),
                    "hand_left": self._encode_image_jpeg(obs["images"]["left_hand"]),
                    "hand_right": self._encode_image_jpeg(obs["images"]["right_hand"]),
                },
                "states": {
                    "head_joint_states": head_states,
                    "arm_joint_states": arm_states,
                    "waist_joint_states": waist_states,
                    "gripper_states": gripper_states,
                    "end_pose": self._build_end_pose(obs),
                },
                "prompt": task_instruction,
                "robot_type": self._robot_type,
                "task_name": self.sub_task_name,
                "episode_idx": self._current_episode_idx,
                "episode_done": self._episode_done,
            },
        }
        if task_segments:
            # Optional context for model adapters; the policy prompt stays intact.
            payload["params"]["segments"] = list(task_segments)
        self._attach_depth_and_calibration(payload, obs, gen_config)
        # Only WBC robots carry a chassis; omit the key entirely elsewhere so
        # non-WBC payloads stay byte-identical to before.
        if chassis_states:
            payload["params"]["states"]["chassis_states"] = chassis_states
        if self._hist_frame_interval > 0:
            # Attach the history captured while replaying the previous chunk.
            # On the first inference (before the server has enabled history)
            # this branch is skipped; on a chunk that captured nothing the
            # buffer is empty and an empty images list is sent.
            history_images = self._history_buffer
            if self._hist_last_n > 0:
                history_images = history_images[-self._hist_last_n :]
            payload["params"]["history"] = {
                "interval": self._hist_frame_interval,
                "images": history_images,
            }
        self._history_buffer = []
        if self.debug:
            logger.debug(f"task_name: {payload['params']['task_name']}")
            logger.debug(f"states: {payload['params']['states']}")
            logger.debug(f"prompt: {payload['params']['prompt']}")
            logger.debug(f"episode_done: {payload['params']['episode_done']}")
            cv2.imwrite("head.png", cv2.cvtColor(obs["images"]["head"], cv2.COLOR_RGB2BGR))
            cv2.imwrite("left_hand.png", cv2.cvtColor(obs["images"]["left_hand"], cv2.COLOR_RGB2BGR))
            cv2.imwrite("right_hand.png", cv2.cvtColor(obs["images"]["right_hand"], cv2.COLOR_RGB2BGR))
            debug_dir = os.path.join(ROOT_DIR, "debug_preview")
            os.makedirs(debug_dir, exist_ok=True)
            pkl_path = os.path.join(debug_dir, f"debug_{self.infer_cnt:04d}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump({"payload": payload, "obs": obs}, f)
            logger.debug(f"Dumped debug pkl to {pkl_path}")

        if self.preview:
            ts = int(time.time() * 1000)
            debug_dir = os.path.join(ROOT_DIR, "debug_preview")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(debug_dir, f"preview_{self.infer_cnt:04d}_{ts}_head.png"),
                cv2.cvtColor(obs["images"]["head"], cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(debug_dir, f"preview_{self.infer_cnt:04d}_{ts}_left_hand.png"),
                cv2.cvtColor(obs["images"]["left_hand"], cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(debug_dir, f"preview_{self.infer_cnt:04d}_{ts}_right_hand.png"),
                cv2.cvtColor(obs["images"]["right_hand"], cv2.COLOR_RGB2BGR),
            )
            logger.info(f"[Preview] Saved images to {debug_dir}/preview_{self.infer_cnt:04d}_{ts}_*.png")
            self.infer_cnt += 1
            return None

        return payload

    def reset(self):
        self.action_buffer.clear()
        self._episode_done = False
        self._task_progress = []
        # Drop any half-collected history and let the server re-enable it on
        # the new episode's first inference response.
        self._hist_frame_interval = 0
        self._hist_last_n = 0
        self._hist_resolution = {}
        self._need_depth = not getattr(self, "control_runtime", False)
        self._history_buffer = []
        self._since_infer = 0
        self.server_episode_done = False
        if getattr(self, "control_runtime", False):
            with self._action_lock:
                self._clear_action_state_locked(refresh_required=False)
                self.chunk_started_this_step = False
                self.raw_waypoint_started = False
            with self._control_lock:
                self._control_pending_ack = None
                self._returned_control_token = None
            # Next episode's first request again follows env.reset().
            self._on_reset = True

    def _update_server_episode_done(self, result_dict):
        """Latch server_episode_done=True if the inference reply signals it."""
        if result_dict.get("episode_done"):
            self.server_episode_done = True

    @staticmethod
    def _parse_result(result_dict):
        left_arm = result_dict.get("left_arm", {})
        right_arm = result_dict.get("right_arm", {})
        waist = result_dict.get("waist", {})

        left_arm_kind = left_arm.get("kind", "JOINT_ABS")
        right_arm_kind = right_arm.get("kind", "JOINT_ABS")

        has_waist = bool(waist and waist.get("kind") in ("JOINT_ABS", "ABS_JOINT"))

        if left_arm_kind not in ("JOINT_ABS", "EEF_ABS") or right_arm_kind not in ("JOINT_ABS", "EEF_ABS"):
            raise ValueError(f"Unsupported action kind: " f"left_arm={left_arm_kind}, " f"right_arm={right_arm_kind}")

        if left_arm_kind != right_arm_kind:
            raise ValueError(f"Left/right arm kind must match: left_arm={left_arm_kind}, right_arm={right_arm_kind}")

        # Frame the server declares its EEF_ABS poses in; accepted per-arm or
        # top-level. Absent means base_link (the historical assumption).
        eef_frame = left_arm.get("base_link") or result_dict.get("base_link") or "base_link"
        if eef_frame not in ("base_link", "arm_base_link"):
            raise ValueError(f"Unsupported EEF frame: {eef_frame}")

        left_arm_vals = np.array(left_arm["values"])
        right_arm_vals = np.array(right_arm["values"])

        chunk = left_arm_vals.shape[0]

        def _get_optional(key):
            raw = result_dict.get(key)
            if raw is None or (isinstance(raw, dict) and not raw.get("values")):
                return None
            arr = np.array(raw if not isinstance(raw, dict) else raw["values"])
            return arr if arr.size > 0 else None

        left_eff_vals = np.array(result_dict["left_effector"])
        right_eff_vals = np.array(result_dict["right_effector"])
        head_vals = _get_optional("head")
        chassis_vals = _get_optional("wheels")
        waist_vals = np.array(waist["values"]) if has_waist else None

        # "wheels" is normally a [Δx,Δy,Δyaw] pose-target delta (POSE_DELTA,
        # the default when the block carries no "kind" -- old servers/replies
        # stay byte-compatible). POSE_ABS_TELEPORT instead carries a single
        # world-frame [x,y,yaw] absolute target and tells PiEnv to teleport
        # the chassis there via reset_robot_base rather than servoing.
        wheels_block = result_dict.get("wheels")
        chassis_kind = wheels_block.get("kind", "POSE_DELTA") if isinstance(wheels_block, dict) else "POSE_DELTA"
        if chassis_kind not in ("POSE_DELTA", "POSE_ABS_TELEPORT"):
            raise ValueError(f"Unsupported wheels kind: {chassis_kind}")

        # Chassis follows only the LAST step of each chunk. The per-step chassis
        # value is a pose target [Δx, Δy, Δyaw] relative to the chunk-start
        # anchor, so broadcasting the final step's target to every step makes the
        # base servo toward one fixed end-of-chunk pose for the whole chunk
        # instead of chasing the small per-step wobble — higher tracking
        # accuracy. The env-side servo logic (PiEnv._apply_chassis_pose_target) is
        # unchanged; it simply sees a constant target across the chunk. A
        # POSE_ABS_TELEPORT target is likewise constant across the chunk, so
        # the same broadcast is harmless (PiEnv only acts on it once).
        if chassis_vals is not None and chunk > 0:
            chassis_vals = np.repeat(chassis_vals[-1:], chunk, axis=0)

        actions = []
        for i in range(chunk):
            entry = {
                "arm": np.concatenate([left_arm_vals[i], right_arm_vals[i]]),
                "gripper": np.concatenate([left_eff_vals[i], right_eff_vals[i]]),
                "kind": left_arm_kind,
                "eef_frame": eef_frame,
            }
            if head_vals is not None:
                entry["head"] = head_vals[i]
            if waist_vals is not None:
                entry["waist"] = waist_vals[i]
            if chassis_vals is not None:
                entry["chassis"] = chassis_vals[i]
                entry["chassis_kind"] = chassis_kind
            actions.append(entry)
        return actions

    def _post_process_action(self, raw_entry, cur_arm, arm_base_tf=None):
        """Post-process action based on kind (JOINT_ABS or EEF_ABS).

        Args:
            raw_entry: dict with "arm", "gripper", and "kind" keys
            cur_arm: current arm joint states
            arm_base_tf: transform matrix for arm base frame computation

        Returns:
            Processed action dict with "arm", "gripper" keys
        """
        kind = raw_entry.get("kind", "JOINT_ABS")
        raw_arm = raw_entry["arm"]
        raw_gripper = raw_entry["gripper"]

        if kind == "EEF_ABS" and self._ikfk_solver is not None:
            left_eef = raw_arm[:7]
            right_eef = raw_arm[7:14]

            # IK solves in arm_base_link, so base_link poses need one transform;
            # arm_base_link poses go straight through.
            if raw_entry.get("eef_frame", "base_link") == "arm_base_link":
                left_eef_abl = np.asarray(left_eef, dtype=np.float64)
                right_eef_abl = np.asarray(right_eef, dtype=np.float64)
            elif arm_base_tf is not None:
                left_eef_abl = _transform_pose_to_frame(left_eef, arm_base_tf)
                right_eef_abl = _transform_pose_to_frame(right_eef, arm_base_tf)
            else:
                raise RuntimeError(
                    "EEF_ABS action declared in base_link but arm_base_transform is "
                    "unavailable; cannot convert to the IK frame"
                )

            left_xyzrpy = np.concatenate(
                [
                    left_eef_abl[:3],
                    R.from_quat(left_eef_abl[3:7]).as_euler("xyz"),
                ]
            )
            right_xyzrpy = np.concatenate(
                [
                    right_eef_abl[:3],
                    R.from_quat(right_eef_abl[3:7]).as_euler("xyz"),
                ]
            )

            eef_action = np.concatenate([left_xyzrpy, right_xyzrpy, raw_gripper[:1], raw_gripper[1:2]])
            joint_action = self._ikfk_solver.eef_actions_to_joint([eef_action.tolist()], cur_arm, [0.0, 0.0])[0]
            arm = [float(v) for v in joint_action[: self._arm_dim]]
            gripper_raw = joint_action[self._arm_dim : self._arm_dim + self._gripper_dim]

            logger.info(f"[EEF_ABS] IK result joints: {[round(v, 4) for v in arm]}")
        else:
            # JOINT_ABS: process directly
            action_flat = np.concatenate([raw_arm, raw_gripper])
            action_flat = process_action(None, cur_arm, action_flat, type="abs_joint", smooth_alpha=0.5)
            arm = [float(v) for v in action_flat[: self._arm_dim]]
            gripper_raw = action_flat[self._arm_dim : self._arm_dim + self._gripper_dim]

        gripper = self._process_gripper_action(gripper_raw, self._robot_config)

        result = {"arm": arm, "gripper": gripper}

        if "head" in raw_entry:
            result["head"] = [float(v) for v in raw_entry["head"]]

        if "waist" in raw_entry:
            result["waist"] = [float(v) for v in raw_entry["waist"]]

        if "chassis" in raw_entry:
            result["chassis"] = [float(v) for v in raw_entry["chassis"]]
            result["chassis_kind"] = raw_entry.get("chassis_kind", "POSE_DELTA")

        return result

    def _post_process_control_action(self, raw_entry, cur_arm, arm_base_tf=None):
        """Post-process one shared-control waypoint.

        Semantic arm goals carry a planned joint trajectory (or hold the
        measured arm); everything else reuses _post_process_action. Direct and
        body segments write only the parts they name.
        """
        kind = raw_entry.get("kind", "JOINT_ABS")
        arm_execution = raw_entry.get("_control_arm_execution")
        if kind == "EEF_ABS" and arm_execution in {"hold_measured", "cartesian_side", "direct_segment"}:
            measured_arm = np.asarray(cur_arm, dtype=np.float64)
            if arm_execution == "hold_measured":
                if measured_arm.shape != (self._arm_dim,) or not np.all(np.isfinite(measured_arm)):
                    raise RuntimeError("non-arm semantic command requires a finite measured arm state")
                arm = measured_arm.tolist()
            else:
                side = raw_entry.get("_control_arm_side")
                if arm_execution == "cartesian_side" and side not in {"left", "right"}:
                    raise RuntimeError("semantic arm command has no validated side")
                if measured_arm.shape != (self._arm_dim,) or not np.all(np.isfinite(measured_arm)):
                    raise RuntimeError("semantic arm command requires a finite measured arm state")
                planned_arm = np.asarray(raw_entry.get("_control_planned_arm"), dtype=np.float64)
                if planned_arm.shape != (self._arm_dim,) or not np.all(np.isfinite(planned_arm)):
                    raise RuntimeError("semantic arm command has no validated joint trajectory target")
                arm = planned_arm.tolist()
            result = {"arm": arm, "gripper": self._process_gripper_action(raw_entry["gripper"], self._robot_config)}
        else:
            base_entry = {key: value for key, value in raw_entry.items() if key not in ("waist", "chassis")}
            result = self._post_process_action(base_entry, cur_arm, arm_base_tf=arm_base_tf)

        if raw_entry.get("_control_direct") is not None:
            result["_direct_duration_s"] = raw_entry["_control_direct"]["duration_s"]
            result["_direct_basis"] = raw_entry.get("_control_direct_basis")
            result["_direct_arms"] = tuple(raw_entry["_control_direct"].get("arms", {}))
            result["_direct_grippers"] = tuple(raw_entry["_control_direct"].get("grippers", {}))

        if "head" in raw_entry and ("_control_direct" not in raw_entry or "head" in raw_entry.get("_control_body", {})):
            result["head"] = [float(v) for v in raw_entry["head"]]
        if "waist" in raw_entry and (
            "_control_direct" not in raw_entry or "torso" in raw_entry.get("_control_body", {})
        ):
            result["waist"] = [float(v) for v in raw_entry["waist"]]
        if "chassis" in raw_entry and "_control_direct" not in raw_entry:
            result["chassis"] = [float(v) for v in raw_entry["chassis"]]
            result["chassis_kind"] = raw_entry.get("chassis_kind", "POSE_DELTA")
        if "_control_chassis_pulse" in raw_entry:
            result["chassis_pulse"] = deepcopy(raw_entry["_control_chassis_pulse"])
        return result


    def _parse_history_config(self, result_dict):
        """Parse the server's history-capture config from its response.

        Prefers the nested `history: {interval, last_n, resolution}` block;
        falls back to the legacy top-level `hist_frame_interval` for servers
        that only set that field. Any malformed sub-field is treated as
        absent (falls back to default behavior) rather than failing the
        whole response.

        Returns (interval, last_n, resolution) where `resolution` maps a
        payload camera name (subset of `_FRAME_CAMERAS` keys) to a validated
        (width, height) tuple.
        """

        def _to_nonneg_int(value):
            try:
                return max(int(value or 0), 0)
            except (TypeError, ValueError):
                return 0

        history_cfg = result_dict.get("history")
        if not isinstance(history_cfg, dict):
            history_cfg = {}

        interval = _to_nonneg_int(history_cfg.get("interval"))
        if interval == 0:
            interval = _to_nonneg_int(result_dict.get("hist_frame_interval"))

        last_n = _to_nonneg_int(history_cfg.get("last_n"))

        resolution = {}
        raw_resolution = history_cfg.get("resolution")
        if isinstance(raw_resolution, dict):
            for cam, size in raw_resolution.items():
                if cam not in self._FRAME_CAMERAS:
                    continue
                # Only a 2-sequence of scalars is a valid size; anything else
                # (dict, short list, string, non-iterable) is treated as absent
                # rather than raising — a malformed field must not take the
                # response (and the whole inference) down.
                if not isinstance(size, (list, tuple)) or len(size) < 2:
                    continue
                try:
                    width, height = int(size[0]), int(size[1])
                except (TypeError, ValueError):
                    continue
                if width > 0 and height > 0:
                    resolution[cam] = (width, height)

        return interval, last_n, resolution

    @staticmethod
    def _to_bool(value):
        """Coerce a response field to bool without the `bool("false") is True`
        trap — a server sending a JSON-ish string instead of a real msgpack
        boolean would otherwise silently mean the opposite of what it wrote.
        """
        if isinstance(value, str):
            return value.strip().lower() not in ("", "0", "false", "no", "off", "none", "null")
        return bool(value)

    def _parse_need_depth(self, result_dict):
        # Each reply independently opts the next request into depth.
        return self._to_bool(result_dict.get("need_depth", False))

    def infer(self, payload):
        """Send one inference request. Returns True on success.

        On transient connection failures the socket is dropped and the
        exception is re-raised so the outer retry helper can classify and
        count it. Server-side semantic errors (RuntimeError) propagate as
        fatal — they will not be retried.
        """
        if getattr(self, "control_runtime", False):
            return self._infer_control(payload)
        try:
            self._ensure_connection()
            data = msgpack.packb(payload)
            logger.info(
                f"Sending payload to server, size={len(data)} bytes, "
                f"robot_type={self._robot_type}, ee_type={self._eef_type}"
            )
            self._ws.send(data)
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Server error: {response}")
            result = msgpack.unpackb(response, raw=False)
            if result.get("error"):
                raise RuntimeError(f"Server returned error: {result['error']}")
            inner = result["result"]
            self._update_server_episode_done(inner)
            actions = self._parse_result(inner)
            # Match main's history protocol; omitted history disables capture.
            self._hist_frame_interval, self._hist_last_n, self._hist_resolution = self._parse_history_config(inner)
            self._need_depth = self._parse_need_depth(inner)
            n = max(len(actions), 1)
            self.action_buffer = deque(actions, maxlen=n)
            self.chunk_len = n
            # A terminal notification may return a hold chunk, but is not a
            # policy decision made within the episode's action budget.
            if actions and not payload.get("params", {}).get("episode_done", False):
                self.policy_dispatch_count += 1
            return True
        except Exception as e:
            logger.warning(f"Model inference failed: {type(e).__name__}: {str(e)}")
            self._drop_connection()
            raise

    def _infer_control(self, payload):
        """Control-runtime infer(): bind the reply to its shared-control identity."""
        try:
            self._ensure_connection()
            data = msgpack.packb(payload)
            logger.info(
                f"Sending payload to server, size={len(data)} bytes, "
                f"robot_type={self._robot_type}, ee_type={self._eef_type}"
            )
            t0 = time.perf_counter()
            ws = self._ws
            if ws is None:
                raise ConnectionError("policy server connection closed")
            ws.send(data)
            response = ws.recv(timeout=_RECV_TIMEOUT_SEC)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if isinstance(response, str):
                raise RuntimeError(f"Server error: {response}")
            result = msgpack.unpackb(response, raw=False)
            if result.get("error"):
                raise RuntimeError(f"Server returned error: {result['error']}")
            if self._control_shutdown_requested.is_set():
                raise ControlShutdownRequested("control runtime shutdown requested")
            inner = result["result"]
            # A pending control.depth_read asks the next capture for depth.
            self._need_depth = inner.get("need_depth") is True
            control_metadata = deepcopy(inner.get("_control"))
            actions = self._parse_result(inner)
            self._validate_control_result_actions(control_metadata, actions)
            control_identity = self._control_identity_from_result(inner, payload)
            logger.info(f"Inference round-trip {elapsed_ms:.0f} ms, chunk={len(actions)}")

            def accept_result_side_effects():
                self._update_server_episode_done(inner)

            if control_identity is None:
                accept_result_side_effects()
                self._start_new_chunk(actions)
            else:
                self._install_control_chunk(
                    control_identity, actions, on_accept=accept_result_side_effects, metadata=control_metadata
                )
            logger.info(f"Queued raw chunk with {len(actions)} waypoint(s)")
            return True
        except ControlShutdownRequested:
            raise
        except StaleControlResult as e:
            with self._action_lock:
                with self._control_lock:
                    self._invalidate_control_writes_locked(advance_revision=False)
            logger.info(f"Discarded superseded control result: {e}")
            return True
        except Exception as e:
            if self._control_shutdown_requested.is_set() and is_transient_error(e):
                raise ControlShutdownRequested("control runtime shutdown requested") from None
            logger.warning(f"Model inference failed: {type(e).__name__}: {str(e)}")
            self._drop_connection()
            raise

    def act(self, observation, **kwargs):
        if getattr(self, "control_runtime", False):
            return self._act_control(observation, **kwargs)
        if len(self.action_buffer) == 0:
            logger.info("CoRobotPolicy: calling model infer")
            task_instruction = kwargs.get("task_instruction", "")
            gen_config = kwargs.get("gen_config")
            logger.info(f"\nInstruction: {task_instruction}\n")
            # get_payload attaches the previous chunk's history and clears it.
            payload = self.get_payload(
                observation, task_instruction, gen_config, task_segments=kwargs.get("task_segments")
            )

            if payload is None:
                return None

            run_with_inference_retry(
                lambda: self.infer(payload),
                log=logger,
                label="CoRobotPolicy.infer",
            )
            self.infer_cnt += 1
            self._since_infer = 0
            self.chunk_started_this_step = True
        else:
            self.chunk_started_this_step = False
            if self._hist_frame_interval > 0:
                self._since_infer += 1
                if self._since_infer % self._hist_frame_interval == 0:
                    self._capture_history(observation, kwargs.get("gen_config"))

        raw_entry = self.action_buffer.popleft()
        cur_arm = get_arm_states(observation["states"], self._arm_dim)
        arm_base_tf = observation.get("arm_base_transform")
        return self._post_process_action(raw_entry, cur_arm, arm_base_tf=arm_base_tf)

    def _act_control(self, observation, **kwargs):
        """Control-runtime act(): one guarded write from the shared-control queue.

        Returns {} whenever nothing may be written this step (stale
        observation, plan being installed, waypoint not yet due, hold/stop).
        """
        from geniesim_benchmark.utils.g2_lula_ikfk import KinematicsTargetRejected

        if self._control_shutdown_requested.is_set():
            raise ControlShutdownRequested("control runtime shutdown requested")
        self.raw_waypoint_started = False
        self.chunk_started_this_step = False
        if not self._observation_is_fresh_after_control_event(observation):
            return {}
        if self.inference_due():
            logger.info("CoRobotPolicy: calling model infer")
            gen_config = kwargs.get("gen_config")
            user_text = kwargs.get("user_text", "")
            self._control_user_text = user_text
            payload = self.get_payload(observation, kwargs.get("task_instruction", ""), gen_config, user_text=user_text)
            if payload is None:
                return None

            run_with_inference_retry(
                lambda: self.infer(payload),
                log=logger,
                label="CoRobotPolicy.infer",
            )
            if self._control_shutdown_requested.is_set():
                raise ControlShutdownRequested("control runtime shutdown requested")
            self.infer_cnt += 1
            self._since_infer = 0

        body_plan = self._prepare_control_body_plan(observation)
        if body_plan is not None:
            return {}
        plan_result = self._prepare_control_arm_plan(observation)
        if plan_result is not None:
            return {}

        simulation_time = self._control_simulation_time(observation)
        with self._action_lock:
            with self._control_lock:
                retry_after_override = self._control_refresh_required or (
                    not self.action_buffer and not self.raw_action_buffer
                )
                if not retry_after_override:
                    if not self.action_buffer:
                        if not self._planned_control_waypoint_due_locked(
                            self.raw_action_buffer[0],
                            simulation_time,
                        ):
                            retry_after_override = True
                        else:
                            self._expand_next_raw_waypoint()
                    if not retry_after_override:
                        raw_entry = self.action_buffer.popleft()
                        self._chunk_frame_index += 1
        if retry_after_override:
            return {}
        execution = raw_entry.get("_control_execution")
        if execution in {"hold", "stop"}:
            action = {"_control_execution": execution}
            returned_action = {}
            if execution == "stop":
                self._dispatch_control_hold(raw_entry.get("_control"), "control.stop")
            else:
                self._dispatch_control_hold(raw_entry.get("_control"), "idle_hold")
        else:
            cur_arm = get_arm_states(observation["states"], self._arm_dim)
            arm_base_tf = observation.get("arm_base_transform")
            try:
                action = self._post_process_control_action(raw_entry, cur_arm, arm_base_tf=arm_base_tf)
            except KinematicsTargetRejected as error:
                logger.warning(f"Control target rejected by kinematics: {error}")
                action = self._recover_control_target_failure(raw_entry, KinematicsTargetRejected.reason)
                if action is None:
                    return {}
            returned_action = {} if action.get("_control_execution") == "failed_target" else action
        if action.get("_control_execution") != "failed_target":
            if execution == "command":
                self._capture_control_completion_basis(
                    observation,
                    raw_entry.get("_control"),
                )
            with self._control_lock:
                self._returned_control_token = raw_entry.get("_control")
                self._control_last_action = deepcopy(action)
        return returned_action

    def _record_control_evidence(self, observation):
        recorder = getattr(self, "control_evidence", None)
        if recorder is not None:
            with self._action_lock:
                with self._control_lock:
                    metadata = deepcopy(self._control_completion)
                    acknowledgement = deepcopy(self._control_pending_ack)
            recorder.record(observation, metadata, acknowledgement)

    def _fail_expired_control_arm_plan(self, observation, env):
        with self._action_lock:
            with self._control_lock:
                state = self._control_arm_plan_state
                queued = bool(self.action_buffer or self.raw_action_buffer)
                if not isinstance(state, dict) or not queued:
                    return False
                token = deepcopy(state["token"])
                deadline = state["deadline"]
                timeout_cause = state["deadline_cause"]
        expected_timeout = self._control_action_timeout_s(env.api_core.robot_profile)
        if not math.isclose(deadline.timeout_s, expected_timeout, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("semantic arm deadline does not match the active robot profile")
        if timeout_cause is None:
            timeout_cause = deadline.observe({"simulation_time_s": self._control_simulation_time(observation)})
        if timeout_cause is None:
            return False
        reason = self._control_deadline_failure_reason(timeout_cause)
        logger.warning(f"Semantic arm trajectory stopped by {timeout_cause}")
        return (
            self._recover_control_target_failure(
                {"_control": token},
                reason,
                deadline=deadline,
            )
            is not None
        )

    def reject_direct_observation(self, token):
        """Retain this dispatch's failure and require a new selected observation."""
        return self._recover_control_target_failure({"_control": token}, "direct_observation_required")

    def cutoff_direct_segment(self, token):
        """Retire the queue and transfer hold once, inside the physics callback."""
        with self._control_lock:
            target = deepcopy(self._last_action_entry)
        held = self._fence_control_for_hold(token, "segment_window_ended")
        if held is None:
            return None
        transfer = {"token": held, "target": target, "failure": None}
        try:
            generation = self._control_stop_callback(
                held, self.control_stop_guard, "segment_window_ended", on_physics_tick=True
            )
        except Exception as error:
            transfer["failure"] = f"{type(error).__name__}: {error}"
            generation = None
        with self._control_lock:
            self._required_control_observation_generation = generation
        return transfer

    def _finish_direct_window(self, observation, env):
        """Wait for the physics-owned cutoff, then prove the transferred hold."""
        with self._control_lock:
            metadata = deepcopy(self._control_completion)
            token = deepcopy(self._returned_control_token)
            raw_target = deepcopy(self._last_action_entry)
        if (
            not metadata
            or set((metadata.get("operations") or {}).values()) not in ({"direct.segment"}, {"body.segment"})
            or not token
        ):
            return None
        if (self._control_last_action or {}).get("_control_execution") == "failed_target":
            return None
        state = env.direct_window(token)
        if state is None:
            return observation
        with self._action_lock:
            drained = not self.action_buffer and not self.raw_action_buffer
        if not state["ended"] and not drained:
            self._publish_control_observation(observation)
            return observation
        deadline = time.monotonic() + self._control_action_timeout_s(env.api_core.robot_profile)
        while not state["ended"]:
            with self.control_action_guard(token) as current:
                if not current:
                    return observation
            if time.monotonic() >= deadline:
                break
            self._publish_control_observation(observation)
            time.sleep(0.01)
            observation = env.get_observation(fetch_images=True)
            state = env.direct_window(token)
            if state is None:
                return observation
        transfer = state.get("transfer")
        if transfer is not None:
            hold_token = transfer["token"]
            raw_target = transfer["target"]
        else:
            hold_token = self._fence_control_for_hold(token, "segment_window_ended")
            if hold_token is None:
                return observation
            self._dispatch_control_hold(hold_token, "segment_window_ended")
        observation, stopped, summary = self._wait_for_control_standstill(
            observation, env, hold_token, self._control_action_timeout_s(env.api_core.robot_profile)
        )
        with self.control_stop_guard(hold_token) as current:
            if not current:
                return observation
            if not summary:
                raise RuntimeError("Direct segment has no post-window measured sample")
            ack = self._control_pending_ack
            if ack is not None and ack.get("dispatch_id") == metadata["dispatch_id"]:
                ack["status"] = "target_failed"
                ack["reason"] = "segment_window_ended" if stopped and state["ended"] else "physical_stop_unconfirmed"
                ack["physical"] = summary
                if raw_target is not None:
                    from geniesim_benchmark.direct_targets import segment_feedback

                    command_id = metadata["command_ids"][0]
                    if "body.segment" in metadata["operations"].values():
                        from geniesim_benchmark.direct_targets import body_feedback

                        ack["direct_parts"] = body_feedback(
                            raw_target["_control_body"],
                            observation["measured"],
                            holding=stopped,
                            started_at_s=state["started_at_s"],
                            ended_at_s=state.get("ended_at_unix_s"),
                            limits=env.api_core.robot_profile["completion"],
                        )
                    else:
                        ack["direct_parts"] = segment_feedback(
                            metadata["command_arguments"][command_id],
                            raw_target["arm"],
                            observation["measured"],
                            env.api_core.robot_profile,
                            holding=stopped,
                            started_at_s=state["started_at_s"],
                            ended_at_s=state.get("ended_at_unix_s"),
                        )
                self._control_completion = None
                self._control_completion_basis = None
                self._control_arm_plan_state = None
        self._record_control_evidence(observation)
        return observation

    def finish_control_step(self, observation, env):
        """Keep a command's write identity until its physical result is known.

        Called only by the explicit control runtime after env.step. Physics and
        control events continue while this task thread measures convergence.
        Sensor-only updates keep the operator view live without fetching a new
        command that would replace an unfinished goal.
        """
        from geniesim_benchmark.control_completion import (
            CompletionWindow,
            PhysicalCompletionDeadline,
            TargetProgressWindow,
            refresh_target_evidence,
        )

        self._record_control_evidence(observation)
        self._fail_expired_control_arm_plan(observation, env)
        direct = self._finish_direct_window(observation, env)
        if direct is not None:
            return direct
        with self._action_lock:
            with self._control_lock:
                metadata = deepcopy(self._control_completion)
                action = deepcopy(self._control_last_action)
                token = deepcopy(self._returned_control_token)
                raw_target = deepcopy(self._last_action_entry)
                completion_basis = deepcopy(self._control_completion_basis)
                failure = deepcopy(self._control_target_failure)
                drained = not self.action_buffer and not self.raw_action_buffer
                motion_token = deepcopy((self._control_arm_plan_state or {}).get("token"))
                action_deadline = (self._control_arm_plan_state or {}).get("deadline")
                if action_deadline is None and failure is not None:
                    action_deadline = failure.get("deadline")
        if not drained and motion_token is not None:
            with self.control_action_guard(motion_token) as current:
                if not current:
                    return observation
            self._publish_control_observation(observation)
            return observation
        if (
            not drained
            or not metadata
            or not metadata.get("command_ids")
            or metadata.get("terminal")
            or not action
            or not token
        ):
            return observation
        operations = set((metadata.get("operations") or {}).values())
        if len(operations) != 1:
            raise RuntimeError("physical completion requires one semantic operation")
        operation = next(iter(operations))
        command_id = metadata["command_ids"][0]
        arguments = (metadata.get("command_arguments") or {}).get(command_id, {})
        failed_target = bool(failure and action.get("_control_execution") == "failed_target")
        stopping = operations == {"control.stop"} or failed_target
        targets = {}
        for field, names_key in (
            ("arm", "arm_joints"),
            ("gripper", "gripper_joints"),
            ("waist", "waist_joints"),
            ("head", "head_joints"),
        ):
            values = action.get(field)
            if values is not None:
                names = env.cfg[names_key]
                if len(names) != len(values):
                    raise RuntimeError("physical completion target/joint shape mismatch")
                targets.update(zip(names, (float(value) for value in values)))
        require_chassis = not failed_target and bool(
            operations & {"chassis.move", "chassis.turn", "chassis.move_distance", "chassis.turn_angle"}
        )
        arm_targets = None
        if not failed_target and operation.startswith("arm."):
            if not raw_target or raw_target.get("kind") != "EEF_ABS":
                raise RuntimeError("semantic arm target has no endpoint pose")
            if raw_target.get("_control_arm_execution") == "cartesian_side":
                side = raw_target.get("_control_arm_side")
                if side not in {"left", "right"}:
                    raise RuntimeError("semantic arm target has no validated side")
                arm_slice = slice(0, 7) if side == "left" else slice(7, 14)
                arm_targets = {side: list(raw_target["arm"][arm_slice])}
            else:
                arm_targets = {"left": list(raw_target["arm"][:7]), "right": list(raw_target["arm"][7:14])}
        joint_limits = {}
        gripper_target = None
        if operation == "gripper.move" and not failed_target:
            from geniesim_benchmark.control_observation import gripper_travel

            profile = env.api_core.robot_profile
            gripper = profile["gripper"][arguments["side"]]
            limits = profile["completion"]
            open_position, closed_position, unit = gripper_travel(gripper)
            linear = unit == "m"
            tolerance = min(
                limits["linear_joint_position_tolerance_m" if linear else "joint_position_tolerance_rad"],
                max(
                    limits["gripper_joint_min_tolerance_m" if linear else "gripper_joint_min_tolerance_rad"],
                    abs(closed_position - open_position) * arguments["amount"] * limits["gripper_tolerance_fraction"],
                ),
            )
            joint_limits[gripper["control_joint"]] = tolerance
            gripper_names = list(env.cfg["gripper_joints"])
            if gripper["control_joint"] not in gripper_names:
                raise RuntimeError("semantic gripper target has no selected control joint")
            target_position = float(action["gripper"][gripper_names.index(gripper["control_joint"])])
            gripper_target = {
                "side": arguments["side"],
                "direction": arguments["direction"],
                "control_joint": gripper["control_joint"],
                "mimic_joint": gripper["mimic_joint"],
                "target_position_rad": target_position,
                "open_position_rad": open_position,
                "closed_position_rad": closed_position,
                "position_tolerance_rad": tolerance,
                "tolerance_fraction": limits["gripper_tolerance_fraction"],
                # Only calibrated jaw geometry carries probes; without them the
                # aperture effect stays unverified and the target never qualifies.
                "aperture_probes": gripper.get("aperture_probes", []),
                "basis": completion_basis,
            }
        window = CompletionWindow(
            targets,
            stopping=stopping,
            require_chassis=require_chassis,
            operation=operation,
            arguments=arguments,
            arm_targets=arm_targets,
            arm_frame=(raw_target or {}).get("eef_frame", "arm_base_link"),
            joint_limit_overrides=joint_limits,
            gripper_target=gripper_target,
        )
        timeout = self._control_action_timeout_s(env.api_core.robot_profile)
        deadline = action_deadline or PhysicalCompletionDeadline(timeout)
        progress_window = TargetProgressWindow()
        if not math.isclose(deadline.timeout_s, timeout, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("physical completion deadline does not match the active robot profile")
        summary = {}
        chassis_cursor = 0
        status = "target_failed"
        physically_qualified = False
        timeout_cause = None
        terminal_reason = None
        trajectory_failure_reason = None
        while True:
            if self._control_shutdown_requested.is_set():
                raise ControlShutdownRequested("control runtime shutdown requested")
            guard = self.control_stop_guard if failed_target else self.control_action_guard
            with guard(token) as current:
                if not current:
                    return observation
            self._publish_control_observation(observation)
            self._record_control_evidence(observation)
            chassis = env.get_g2_90d_chassis_control_state(after_tick_seq=chassis_cursor) if require_chassis else None
            if isinstance(chassis, dict):
                chassis_cursor = int(chassis.get("tick_seq", chassis_cursor))
                identity = chassis.get("control_identity") or {}
                if any(
                    identity.get(key) != token.get(key) for key in ("service_id", "run_id", "generation", "dispatch_id")
                ):
                    chassis = None
            reached, summary = window.observe(observation.get("measured"), chassis)
            if failed_target or stopping:
                proof = self._control_retirement_proof(observation, token)
                if proof is not None:
                    summary["command_retirement"] = proof
            trajectory_failure_reason = summary.get("trajectory_failure_reason")
            if trajectory_failure_reason is not None:
                break
            timeout_cause = deadline.observe(summary)
            if not stopping and not failed_target and progress_window.observe(summary):
                timeout_cause = "target_stalled"
            if timeout_cause is not None:
                break
            if reached:
                if (
                    require_chassis
                    and not stopping
                    and not failed_target
                    and not env.retire_g2_90d_chassis_goal(control_identity=token)
                ):
                    reached = False
                else:
                    target_summary = deepcopy(summary)
                    if failed_target or stopping:
                        proof = self._control_retirement_proof(observation, token)
                        if proof is None:
                            reached = False
                        else:
                            summary["command_retirement"] = proof
                            physically_qualified = True
                            status = "target_failed" if failed_target else "stop_confirmed"
                            break
                    else:
                        hold_token = self._fence_control_for_hold(token, "command_completed")
                        if hold_token is None:
                            return observation
                        token = hold_token
                        self._dispatch_control_hold(token, "command_completed")
                        observation, stopped, stop_summary = self._wait_for_control_standstill(
                            observation,
                            env,
                            token,
                            timeout,
                            require_chassis=require_chassis,
                        )
                        if stopped:
                            if require_chassis:
                                chassis = env.get_g2_90d_chassis_control_state(after_tick_seq=None)
                            target_satisfied = refresh_target_evidence(
                                stop_summary,
                                window,
                                observation.get("measured"),
                                chassis,
                            )
                            summary = stop_summary
                            physically_qualified = True
                            if target_satisfied:
                                status = "target_reached"
                            else:
                                status = "target_failed"
                                terminal_reason = "physical_target_timeout"
                        else:
                            summary = stop_summary or target_summary
                            status = "target_failed"
                            terminal_reason = "physical_stop_unconfirmed"
                        break
            if require_chassis and isinstance(chassis, dict) and chassis.get("status") in {"failed", "cancelled"}:
                break
            time.sleep(0.05)
            observation = self._control_completion_observation(env)
        if trajectory_failure_reason is not None:
            hold_token = self._fence_control_for_hold(
                token,
                trajectory_failure_reason,
            )
            if hold_token is None:
                return observation
            token = hold_token
            self._dispatch_control_hold(token, trajectory_failure_reason)
            observation, stopped, stop_summary = self._wait_for_control_standstill(
                observation,
                env,
                token,
                timeout,
            )
            refresh_target_evidence(
                stop_summary,
                window,
                observation.get("measured"),
                chassis,
            )
            summary = stop_summary or summary
            terminal_reason = trajectory_failure_reason if stopped else "physical_stop_unconfirmed"
        elif status == "target_failed" and not failed_target and not physically_qualified:
            requested_reason = (
                "physical_sample_invalid"
                if timeout_cause in {"simulation_clock_invalid", "wall_stalled"}
                else "physical_target_stalled" if timeout_cause == "target_stalled" else "physical_target_timeout"
            )
            hold_token = self._fence_control_for_hold(token, requested_reason)
            if hold_token is None:
                return observation
            token = hold_token
            self._dispatch_control_hold(token, requested_reason)
            observation, stopped, stop_summary = self._wait_for_control_standstill(observation, env, token, timeout)
            refresh_target_evidence(
                stop_summary,
                window,
                observation.get("measured"),
                chassis,
            )
            summary = stop_summary or summary
            terminal_reason = requested_reason if stopped else "physical_stop_unconfirmed"
        if not summary:
            raise RuntimeError("physical completion has no valid measured sample")
        with self._action_lock:
            with self._control_lock:
                if token.get("write_revision") != self._control_write_revision:
                    return observation
                ack = self._control_pending_ack
                if ack is not None and ack.get("dispatch_id") == metadata.get("dispatch_id"):
                    ack["status"] = status
                    ack["physical"] = summary
                    if failed_target:
                        ack["reason"] = failure["reason"] if physically_qualified else "physical_stop_unconfirmed"
                        self._control_target_failure = None
                    elif status == "target_failed":
                        ack["reason"] = terminal_reason
                    self._control_completion = None
                    self._control_completion_basis = None
                    self._control_arm_plan_state = None
        if not observation.get("images"):
            observation = env.get_observation(fetch_images=True)
        self._record_control_evidence(observation)
        return observation

    def _wait_for_control_standstill(self, observation, env, token, timeout, *, require_chassis=False):
        """Qualify the guarded hold after an irreversible trajectory violation."""

        from geniesim_benchmark.control_completion import CompletionWindow, PhysicalCompletionDeadline

        window = CompletionWindow({}, stopping=True)
        deadline = PhysicalCompletionDeadline(timeout)
        summary = {}
        while True:
            if self._control_shutdown_requested.is_set():
                raise ControlShutdownRequested("control runtime shutdown requested")
            with self.control_stop_guard(token) as current:
                if not current:
                    # A newer Stop owns this observation. The prior report is
                    # not evidence for it and must not be merged with it.
                    return observation, False, {}
            self._publish_control_observation(observation)
            self._record_control_evidence(observation)
            reached, summary = window.observe(observation.get("measured"))
            # Retirement proves the hold write, not that the robot is still.
            # Failed standstill must retain this identity-bound evidence too.
            proof = self._control_retirement_proof(observation, token)
            if proof is not None:
                summary["command_retirement"] = proof
                chassis = env.get_g2_90d_chassis_control_state(after_tick_seq=None) if require_chassis else None
                chassis_ready = not require_chassis or (
                    isinstance(chassis, dict)
                    and chassis.get("status") == "reached"
                    and chassis.get("goal_retired") is True
                )
                if reached and chassis_ready:
                    return observation, True, summary
            if deadline.observe(summary) is not None:
                return observation, False, summary
            time.sleep(0.05)
            observation = self._control_completion_observation(env)

    def _control_completion_observation(self, env):
        # Keep RGB fresh for admission/preview without encoding three cameras
        # on every measured-joint poll. Final inference still gets full RGB.
        now = time.monotonic()
        images_due = now - getattr(self, "_control_last_completion_rgb", -math.inf) >= 0.2
        if images_due:
            self._control_last_completion_rgb = now
        return env.get_observation(fetch_images=images_due)

    def _control_retirement_proof(self, observation, token):
        """Bind a post-fence actuator sample to the guarded hold write."""

        observed_generation = observation.get("_control_observation_generation")
        with self._action_lock:
            with self._control_lock:
                required_generation = self._required_control_observation_generation
                current_revision = self._control_write_revision
        if (
            not isinstance(required_generation, int)
            or isinstance(required_generation, bool)
            or not isinstance(observed_generation, int)
            or isinstance(observed_generation, bool)
            or observed_generation < required_generation
            or token.get("write_revision") != current_revision
        ):
            return None
        fields = ("service_id", "run_id", "generation", "dispatch_id", "write_revision")
        if any(token.get(field) is None for field in fields):
            return None
        return {
            **{field: token[field] for field in fields},
            "status": "retired",
            "zero_drive_fenced": True,
            "hold_write_applied": True,
        }

    def control_reset_requested(self):
        with self._action_lock:
            with self._control_lock:
                metadata = self._control_completion or {}
                return bool(
                    metadata.get("terminal")
                    and set((metadata.get("operations") or {}).values()) == {"run.reset"}
                    and not self.action_buffer
                    and not self.raw_action_buffer
                    and not self._control_refresh_required
                )

    def acknowledge_control_reset(self, observation):
        with self._action_lock:
            with self._control_lock:
                if not self.control_reset_requested():
                    return False
                token = deepcopy(self._returned_control_token)
                ack = deepcopy(self._control_pending_ack)
        if not token or not ack:
            return False
        identity = self._publish_control_observation(observation, acknowledgement=ack)
        with self._action_lock:
            with self._control_lock:
                if token.get("write_revision") != self._control_write_revision or self._control_refresh_required:
                    return False
                if any(identity.get(key) != token.get(key) for key in ("service_id", "run_id", "generation")):
                    return False
                self._control_pending_ack = None
                self._control_completion = None
                self._control_completion_basis = None
                return True

    def _publish_control_observation(self, observation, acknowledgement=None):
        if self._control_shutdown_requested.is_set():
            raise ControlShutdownRequested("control runtime shutdown requested")
        params = self._control_sensor_fields(
            observation,
            self._control_user_text,
            private_ack=acknowledgement,
        )
        params.update(episode_id=self._episode_id, on_reset=False)
        if acknowledgement is not None:
            params["control_ack"] = acknowledgement
        request_id = f"observation:{self._episode_id}:{time.monotonic_ns()}"
        request = {"method": "control.observe", "request_id": request_id, "params": params}
        socket = self._ws
        if socket is None:
            if self._control_shutdown_requested.is_set():
                raise ControlShutdownRequested("control runtime shutdown requested")
            raise ConnectionError("control observation connection is unavailable")
        try:
            socket.send(msgpack.packb(request))
            response = socket.recv(timeout=_RECV_TIMEOUT_SEC)
        except Exception as error:
            if self._control_shutdown_requested.is_set() and is_transient_error(error):
                raise ControlShutdownRequested("control runtime shutdown requested") from None
            raise
        result = msgpack.unpackb(response, raw=False)
        if not isinstance(result, dict) or result.get("request_id") != request_id:
            raise RuntimeError("uncorrelated control observation acknowledgement")
        if result.get("error"):
            raise RuntimeError("control observation publication failed: " + str(result["error"].get("code", "unknown")))
        identity = result.get("result")
        if not isinstance(identity, dict):
            raise RuntimeError("invalid control observation acknowledgement")
        self._need_depth = identity.get("need_depth") is True
        return identity
