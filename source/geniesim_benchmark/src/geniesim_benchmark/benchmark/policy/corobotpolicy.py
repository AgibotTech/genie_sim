# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import time
import pickle
from copy import deepcopy
from typing import Dict

from .base import BasePolicy
from geniesim_benchmark.plugins.logger import Logger
from geniesim_benchmark.utils.comm.retry import run_with_inference_retry
from geniesim_benchmark.utils.generalization_utils import apply_camera_image_augmentation
from collections import deque
import numpy as np
import cv2
import msgpack
from scipy.spatial.transform import Rotation as R
from geniesim_benchmark.utils.infer_post_process import process_action, get_arm_states
from geniesim_benchmark.utils.name_utils import ROBOT_CONFIGS, DEFAULT_ROBOT_CONFIG
from geniesim_benchmark.utils.ikfk_utils import get_shared_ikfk_solver
from geniesim_benchmark.utils.comm.websocket_client import ws_connect_compat


logger = Logger()

ROOT_DIR = os.environ.get("SIM_REPO_ROOT")

_OPEN_TIMEOUT_SEC = 30
_PING_INTERVAL_SEC = 20
_PING_TIMEOUT_SEC = 60


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
        ik_backend="auto",
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
        # requests require an enabling reply; omission disables depth.
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

    def _ensure_connection(self):
        """Make a single connect attempt if currently disconnected.

        Transient failures propagate so the outer retry helper counts them
        against the budget — looping here would block one infer() call
        forever and bypass the budget."""
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
        self._drop_connection()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def set_episode_idx(self, idx):
        self._current_episode_idx = idx

    def update_task_status(self, done, task_progress):
        self._episode_done = done
        self._task_progress = task_progress

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

        Returns None if eef or arm_base_transform is unavailable.
        """
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

    def get_payload(self, obs, task_instruction, gen_config, task_segments=None):
        obs = self._pre_process_obs(obs, gen_config)

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
            debug_dir = os.environ.get("GENIESIM_PREVIEW_OUTPUT_DIR") or os.path.join(ROOT_DIR, "debug_preview")
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
        self._need_depth = True
        self._history_buffer = []
        self._since_infer = 0
        self.server_episode_done = False

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
        right_frame = right_arm.get("base_link") or result_dict.get("base_link") or "base_link"
        if left_arm_kind == "EEF_ABS" and eef_frame != right_frame:
            raise ValueError(f"Left/right EEF frames must match: {eef_frame}, {right_frame}")
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

        # Wheels carry a pose delta relative to the chunk-start anchor.
        wheels_block = result_dict.get("wheels")
        chassis_kind = wheels_block.get("kind", "POSE_DELTA") if isinstance(wheels_block, dict) else "POSE_DELTA"
        if chassis_kind != "POSE_DELTA":
            raise ValueError(f"Unsupported wheels kind: {chassis_kind}")

        # Chassis follows only the LAST step of each chunk. The per-step chassis
        # value is a pose target [Δx, Δy, Δyaw] relative to the chunk-start
        # anchor, so broadcasting the final step's target to every step makes the
        # base servo toward one fixed end-of-chunk pose for the whole chunk
        # instead of chasing the small per-step wobble — higher tracking
        # accuracy. The env-side servo logic (PiEnv._apply_chassis_pose_target) is
        # unchanged; it simply sees a constant target across the chunk.
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

        if kind == "EEF_ABS":
            if self._ikfk_solver is None:
                raise RuntimeError("EEF_ABS requires an available IK solver")
            from geniesim_benchmark.utils.eef_control import eef_poses_to_joints

            joints = eef_poses_to_joints(
                self._ikfk_solver,
                [raw_arm[:7]],
                [raw_arm[7:14]],
                cur_arm,
                raw_entry.get("eef_frame", "base_link"),
                arm_base_tf,
            )[0]
            arm = [float(v) for v in joints]
            # IK owns arm joints only; retain all effector targets (e.g. 40 WUJI joints).
            gripper_raw = raw_gripper

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

    def act(self, observation, **kwargs):
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
