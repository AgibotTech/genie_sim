# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory, resource_tracker as _rt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from geniesim.rl.envs.process_manager import ProcessManager
from geniesim.rl.renderer.shm_layout import (
    NUM_CAMS,
    SHM_HEADER_BYTES,
    shm_total_bytes,
    ctrl_shm_name as _ctrl_shm_name,
    ctrl_total_bytes as _ctrl_total_bytes,
    step_shm_name as _step_shm_name,
    step_total_bytes as _step_total_bytes,
    CTRL_HEADER_BYTES,
    CTRL_SYNC_BYTES,
    BODY_POSE_DIM,
    STEP_HEADER_BYTES,
    STEP_OUTPUT_SCALARS,
    RESET_IDLE,
    RESET_REQUESTED,
    RESET_DONE,
    MUJOCO_PHASE_WAIT,
    MUJOCO_PHASE_GO,
    MUJOCO_PHASE_DONE,
    STEP_PHASE_IDLE,
    STEP_PHASE_STEP_REQUEST,
    STEP_PHASE_STEP_DONE,
    STEP_PHASE_RESET_REQUEST,
    STEP_PHASE_RESET_DONE,
    STEP_PHASE_CLOSE,
)


@dataclass
class GenieSimVectorEnvConfig:
    mjcf_path: str = ""
    scene_usd: str = ""
    robot_usd: str = ""
    robot_prim: str = "/robot"

    task_file: str = ""
    task_name: str = ""
    task_description: str = ""
    robot_cfg: str = "G2_omnipicker"
    robot_type: str = "G2"
    task_instance_id: int = 0

    num_envs: int = 1
    max_episode_steps: int = 300

    cam_width: int = 640
    cam_height: int = 480
    main_cam_prim: str = "/camera_main"
    wrist_cam_prim: str = ""
    cameras_json: str = ""

    ignore_terminations: bool = False
    auto_reset: bool = True

    physics_hz: int = 1000
    render_hz: float = 30.0
    shm_name: str = "geniesim_frames"
    headless: bool = True
    ros_domain_id: int = 0
    isaac_python: str = "/isaac-sim/python.sh"
    mujoco_python: str = ""

    state_dim: int = 28
    action_dim: int = 14
    state_joint_offset: int = 0
    ctrl_offset: int = 0
    ctrl_offset_r: int = -1

    control_mode: str = "joint"
    gripper_ctrl_l: int = -1
    gripper_ctrl_r: int = -1
    ee_body_l: str = "arm_l_link7"
    ee_body_r: str = "arm_r_link7"
    ik_max_iter: int = 10
    ik_damp: float = 0.05

    randomization_cfg_json: str = ""
    init_qpos_json: str = ""
    reset_ee_r_json: str = ""
    seed: int = 42

    info_body_names: List[str] = field(default_factory=list)

    sync_mode: bool = True
    steps_per_step: int = 33

    shm_open_timeout_sec: int = 180
    attach_to_running: bool = False
    enable_reward: bool = True # deprecated
    reward_coef: float = 1.0 #deprecated


class GenieSimVectorEnv:

    def __init__(self, cfg: GenieSimVectorEnvConfig):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self._elapsed_steps = np.zeros(cfg.num_envs, dtype=np.int32)
        self._episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
        self._success_once = np.zeros(cfg.num_envs, dtype=bool)

        self._info_body_names: List[str] = list(cfg.info_body_names or [])
        self._info_dim = len(self._info_body_names) * BODY_POSE_DIM
        self._steps_per_step = cfg.steps_per_step if cfg.steps_per_step > 0 else max(1, int(cfg.physics_hz / cfg.render_hz))

        if cfg.attach_to_running:
            self._proc_manager = None
        else:
            self._proc_manager = ProcessManager(
                num_envs=cfg.num_envs,
                mjcf_path=cfg.mjcf_path,
                scene_usd=cfg.scene_usd,
                robot_usd=cfg.robot_usd,
                robot_prim=cfg.robot_prim,
                shm_name=cfg.shm_name,
                physics_hz=cfg.physics_hz,
                render_hz=cfg.render_hz,
                cam_width=cfg.cam_width,
                cam_height=cfg.cam_height,
                main_cam_prim=cfg.main_cam_prim,
                wrist_cam_prim=cfg.wrist_cam_prim,
                cameras_json=getattr(cfg, "cameras_json", ""),
                headless=cfg.headless,
                ros_domain_id=cfg.ros_domain_id,
                isaac_python=cfg.isaac_python,
                mujoco_python=cfg.mujoco_python or None,
                task_name=cfg.task_name,
                robot_type=cfg.robot_type,
                task_instance_id=cfg.task_instance_id,
                state_joint_offset=cfg.state_joint_offset,
                ctrl_offset=cfg.ctrl_offset,
                ctrl_offset_r=cfg.ctrl_offset_r,
                state_dim=cfg.state_dim,
                action_dim=cfg.action_dim,
                control_mode=cfg.control_mode,
                gripper_ctrl_l=cfg.gripper_ctrl_l,
                gripper_ctrl_r=cfg.gripper_ctrl_r,
                ee_body_l=cfg.ee_body_l,
                ee_body_r=cfg.ee_body_r,
                ik_max_iter=cfg.ik_max_iter,
                ik_damp=cfg.ik_damp,
                randomization_cfg_json=cfg.randomization_cfg_json,
                init_qpos_json=cfg.init_qpos_json,
                reset_ee_r_json=cfg.reset_ee_r_json,
                seed=cfg.seed,
                info_body_names=self._info_body_names,
                sync_mode=cfg.sync_mode,
                steps_per_step=self._steps_per_step,
            )
            self._proc_manager.start(wait_ready_sec=120.0)

        self._frame_shm: Optional[shared_memory.SharedMemory] = None
        self._frames: Optional[np.ndarray] = None
        self._frame_counter: Optional[np.ndarray] = None
        self._open_frame_shm()

        self._ctrl_shms: List[shared_memory.SharedMemory] = []
        self._ctrl_counters: List[np.ndarray] = []
        self._ctrl_reset_flags: List[np.ndarray] = []
        self._ctrl_states_bufs: List[np.ndarray] = []
        self._ctrl_actions_bufs: List[np.ndarray] = []
        self._ctrl_info_bufs: List[Optional[np.ndarray]] = []
        self._ctrl_mj_phases: List[np.ndarray] = []
        self._ctrl_sps_bufs: List[np.ndarray] = []
        self._open_ctrl_shms()

        self._step_shm: Optional[shared_memory.SharedMemory] = None
        self._step_phase: Optional[np.ndarray] = None
        self._step_reset_mask: Optional[np.ndarray] = None
        self._step_actions: Optional[np.ndarray] = None
        self._step_rewards: Optional[np.ndarray] = None
        self._step_terminated: Optional[np.ndarray] = None
        self._step_truncated: Optional[np.ndarray] = None
        self._step_elapsed: Optional[np.ndarray] = None
        self._step_returns: Optional[np.ndarray] = None
        self._step_success: Optional[np.ndarray] = None
        self._step_info_poses: Optional[np.ndarray] = None
        self._create_step_shm()

        print(
            f"[GenieSimVectorEnv] Initialised | num_envs={self.num_envs} "
            f"state_dim={cfg.state_dim} action_dim={cfg.action_dim} "
            f"info_dim={self._info_dim} steps_per_step={self._steps_per_step}"
        )

    def _open_frame_shm(self):
        import json as _json
        h, w = self.cfg.cam_height, self.cfg.cam_width
        cameras = []
        if getattr(self.cfg, "cameras_json", ""):
            try:
                cameras = _json.loads(self.cfg.cameras_json)
            except Exception:
                pass
        self._num_cams = len(cameras) if cameras else NUM_CAMS
        _total = shm_total_bytes(self.num_envs, h, w, num_cams=self._num_cams)
        for _ in range(self.cfg.shm_open_timeout_sec):
            try:
                self._frame_shm = shared_memory.SharedMemory(
                    name=self.cfg.shm_name, create=False, size=_total
                )
                _rt.unregister(f"/{self.cfg.shm_name}", "shared_memory")
                break
            except FileNotFoundError:
                time.sleep(1.0)
        else:
            raise RuntimeError(
                f"Frame SHM '{self.cfg.shm_name}' not available"
            )
        self._frames = np.ndarray(
            (self.num_envs, self._num_cams, h, w, 3),
            dtype=np.uint8,
            buffer=self._frame_shm.buf,
            offset=SHM_HEADER_BYTES,
        )
        self._frame_counter = np.ndarray(
            (1,), dtype=np.uint32, buffer=self._frame_shm.buf, offset=0
        )

    def _open_ctrl_shms(self):
        _S = self.cfg.state_dim * 4
        _A = self.cfg.action_dim * 4
        _I = self._info_dim * 4
        _total = _ctrl_total_bytes(
            self.cfg.state_dim, self.cfg.action_dim, self._info_dim
        )
        for i in range(self.num_envs):
            name = _ctrl_shm_name(self.cfg.shm_name, i)
            shm = None
            for _ in range(self.cfg.shm_open_timeout_sec):
                try:
                    shm = shared_memory.SharedMemory(
                        name=name, create=False, size=_total
                    )
                    break
                except FileNotFoundError:
                    time.sleep(1.0)
            if shm is None:
                raise RuntimeError(f"Ctrl SHM '{name}' not available")
            _rt.unregister(f"/{name}", "shared_memory")
            self._ctrl_shms.append(shm)
            self._ctrl_counters.append(
                np.ndarray((1,), dtype=np.uint32, buffer=shm.buf, offset=0)
            )
            self._ctrl_reset_flags.append(
                np.ndarray((1,), dtype=np.uint32, buffer=shm.buf, offset=4)
            )
            self._ctrl_states_bufs.append(
                np.ndarray(
                    (self.cfg.state_dim,), dtype=np.float32,
                    buffer=shm.buf, offset=CTRL_HEADER_BYTES,
                )
            )
            self._ctrl_actions_bufs.append(
                np.ndarray(
                    (self.cfg.action_dim,), dtype=np.float32,
                    buffer=shm.buf, offset=CTRL_HEADER_BYTES + _S,
                )
            )
            if self._info_dim > 0:
                self._ctrl_info_bufs.append(
                    np.ndarray(
                        (self._info_dim,), dtype=np.float32,
                        buffer=shm.buf, offset=CTRL_HEADER_BYTES + _S + _A,
                    )
                )
            else:
                self._ctrl_info_bufs.append(None)
            _sync_off = CTRL_HEADER_BYTES + _S + _A + _I
            self._ctrl_mj_phases.append(
                np.ndarray(
                    (1,), dtype=np.uint32,
                    buffer=shm.buf, offset=_sync_off,
                )
            )
            self._ctrl_sps_bufs.append(
                np.ndarray(
                    (1,), dtype=np.uint32,
                    buffer=shm.buf, offset=_sync_off + 4,
                )
            )
        print(
            f"[GenieSimVectorEnv] Ctrl SHMs attached | "
            f"state_dim={self.cfg.state_dim} action_dim={self.cfg.action_dim} "
            f"info_dim={self._info_dim}"
        )

    def _create_step_shm(self):
        N = self.num_envs
        A = self.cfg.action_dim
        _total = _step_total_bytes(N, A, self._info_dim)
        name = _step_shm_name(self.cfg.shm_name)
        try:
            old = shared_memory.SharedMemory(name=name, create=False)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        self._step_shm = shared_memory.SharedMemory(
            name=name, create=True, size=_total
        )
        off = 0
        self._step_phase = np.ndarray(
            (1,), dtype=np.uint32, buffer=self._step_shm.buf, offset=off
        )
        off += STEP_HEADER_BYTES
        self._step_reset_mask = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        self._step_actions = np.ndarray(
            (N, A), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * A * 4
        self._step_rewards = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        self._step_terminated = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        self._step_truncated = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        self._step_elapsed = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        self._step_returns = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        self._step_success = np.ndarray(
            (N,), dtype=np.float32, buffer=self._step_shm.buf, offset=off
        )
        off += N * 4
        if self._info_dim > 0:
            self._step_info_poses = np.ndarray(
                (N, self._info_dim), dtype=np.float32,
                buffer=self._step_shm.buf, offset=off,
            )
        self._step_phase[0] = STEP_PHASE_IDLE
        self._step_reset_mask[:] = 0.0
        self._step_actions[:] = 0.0
        self._step_rewards[:] = 0.0
        self._step_terminated[:] = 0.0
        self._step_truncated[:] = 0.0
        self._step_elapsed[:] = 0.0
        self._step_returns[:] = 0.0
        self._step_success[:] = 0.0
        if self._step_info_poses is not None:
            self._step_info_poses[:] = 0.0
        print(f"[GenieSimVectorEnv] Step SHM created: {_step_shm_name(self.cfg.shm_name)}")

    def _trigger_mujoco_step(self):
        for phase_buf in self._ctrl_mj_phases:
            phase_buf[0] = MUJOCO_PHASE_GO

    def _wait_mujoco_done(self, timeout: float = 10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(int(p[0]) == MUJOCO_PHASE_DONE for p in self._ctrl_mj_phases):
                for p in self._ctrl_mj_phases:
                    p[0] = MUJOCO_PHASE_WAIT
                return True
            time.sleep(0.0001)
        print("[GenieSimVectorEnv] WARNING: MuJoCo step timeout")
        return False

    def _read_states(self) -> np.ndarray:
        return np.stack(
            [np.copy(buf) for buf in self._ctrl_states_bufs], axis=0
        )

    def _read_body_poses(self) -> Dict[str, np.ndarray]:
        if self._info_dim == 0:
            return {}
        poses = {}
        for idx, bname in enumerate(self._info_body_names):
            arr = np.zeros((self.num_envs, BODY_POSE_DIM), dtype=np.float32)
            for env_i in range(self.num_envs):
                buf = self._ctrl_info_bufs[env_i]
                if buf is not None:
                    off = idx * BODY_POSE_DIM
                    arr[env_i] = buf[off:off + BODY_POSE_DIM]
            poses[bname] = arr
        return poses

    def _get_obs(self) -> Dict[str, Any]:
        import json as _json
        obs: Dict[str, Any] = {}
        cameras = []
        if getattr(self.cfg, "cameras_json", ""):
            try:
                cameras = _json.loads(self.cfg.cameras_json)
            except Exception:
                pass
        if cameras:
            for cam_idx, cam_cfg in enumerate(cameras):
                name = cam_cfg["name"]
                key = f"{name}_images"
                if cam_idx < self._frames.shape[1]:
                    obs[key] = np.copy(self._frames[:, cam_idx, :, :, :])
        else:
            h, w = self.cfg.cam_height, self.cfg.cam_width
            obs["main_images"] = np.copy(self._frames[:, 0, :, :, :])
            if self.cfg.wrist_cam_prim:
                obs["wrist_images"] = np.copy(self._frames[:, 1, :, :, :])
        states = self._read_states()
        obs["states"] = states
        obs["task_descriptions"] = [self.cfg.task_description] * self.num_envs
        return obs

    def _reset_env(self, env_idx: int):
        flag = self._ctrl_reset_flags[env_idx]
        flag[0] = RESET_REQUESTED
        self._ctrl_mj_phases[env_idx][0] = MUJOCO_PHASE_GO
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if int(flag[0]) == RESET_DONE:
                flag[0] = RESET_IDLE
                self._ctrl_mj_phases[env_idx][0] = MUJOCO_PHASE_WAIT
                break
            time.sleep(0.001)
        else:
            print(f"[GenieSimVectorEnv] reset timeout for env_{env_idx}")
        self._elapsed_steps[env_idx] = 0
        self._episode_returns[env_idx] = 0.0
        self._success_once[env_idx] = False

    def reset(self, env_idx=None) -> Tuple[Dict, Dict]:
        if env_idx is None:
            indices = list(range(self.num_envs))
        elif isinstance(env_idx, int):
            indices = [env_idx]
        else:
            indices = list(env_idx)
        for i in indices:
            self._reset_env(i)
        obs = self._get_obs()
        body_poses = self._read_body_poses()
        info = self._build_infos(
            np.zeros(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=bool),
            np.zeros(self.num_envs, dtype=bool),
            body_poses,
        )
        self._write_step_output(
            np.zeros(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.float32),
            body_poses,
        )
        return obs, info

    def step(
        self,
        actions: np.ndarray,
        auto_reset: bool = True,
    ) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, Dict]:
        if self._proc_manager is not None:
            self._proc_manager.restart_dead()

        for i, buf in enumerate(self._ctrl_actions_bufs):
            n = min(actions.shape[1], len(buf))
            np.copyto(buf[:n], actions[i, :n].astype(np.float32))

        self._trigger_mujoco_step()
        self._wait_mujoco_done()

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        body_poses = self._read_body_poses()

        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        dones = terminated | truncated
        self._episode_returns += rewards
        self._success_once |= terminated

        obs = self._get_obs()
        infos = self._build_infos(rewards, terminated, truncated, body_poses)

        if self.cfg.ignore_terminations:
            infos["episode"]["success_at_end"] = terminated.copy()
            terminated = np.zeros_like(terminated)

        self._write_step_output(rewards, terminated, truncated, body_poses)

        _do_auto_reset = auto_reset and self.cfg.auto_reset
        if dones.any() and _do_auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, rewards, terminated, truncated, infos

    def _build_infos(
        self, rewards, terminated, truncated, body_poses
    ) -> Dict:
        return {
            "episode": {
                "success_once": self._success_once.copy(),
                "return": self._episode_returns.copy(),
                "episode_len": self._elapsed_steps.copy(),
                "reward": np.where(
                    self._elapsed_steps > 0,
                    self._episode_returns / np.maximum(self._elapsed_steps, 1),
                    0.0,
                ),
            },
            "body_poses": body_poses,
            "task_progress": [[] for _ in range(self.num_envs)],
        }

    def _write_step_output(self, rewards, terminated, truncated, body_poses):
        self._step_rewards[:] = rewards.astype(np.float32)
        self._step_terminated[:] = terminated.astype(np.float32)
        self._step_truncated[:] = truncated.astype(np.float32)
        self._step_elapsed[:] = self._elapsed_steps.astype(np.float32)
        self._step_returns[:] = self._episode_returns.astype(np.float32)
        self._step_success[:] = self._success_once.astype(np.float32)
        if self._step_info_poses is not None and body_poses:
            for idx, bname in enumerate(self._info_body_names):
                arr = body_poses.get(bname)
                if arr is not None:
                    off = idx * BODY_POSE_DIM
                    self._step_info_poses[:, off:off + BODY_POSE_DIM] = arr

    def _handle_auto_reset(self, dones, final_obs, infos):
        _final_obs = copy.deepcopy(final_obs)
        _final_info = copy.deepcopy(infos)
        done_indices = np.where(dones)[0].tolist()
        obs, new_infos = self.reset(env_idx=done_indices)
        new_infos["final_observation"] = _final_obs
        new_infos["final_info"] = _final_info
        new_infos["_final_observation"] = dones
        new_infos["_final_info"] = dones
        new_infos["_elapsed_steps"] = dones
        return obs, new_infos

    def run_step_loop(self):
        print("[GenieSimVectorEnv] Step loop started. Waiting for requests...")
        while True:
            phase = int(self._step_phase[0])
            if phase == STEP_PHASE_CLOSE:
                print("[GenieSimVectorEnv] Received CLOSE signal.")
                break
            elif phase == STEP_PHASE_STEP_REQUEST:
                actions = np.copy(self._step_actions)
                self.step(actions, auto_reset=False)
                self._step_phase[0] = STEP_PHASE_STEP_DONE
            elif phase == STEP_PHASE_RESET_REQUEST:
                mask = self._step_reset_mask
                indices = np.where(mask > 0.5)[0].tolist()
                if not indices:
                    indices = list(range(self.num_envs))
                self.reset(env_idx=indices)
                self._step_phase[0] = STEP_PHASE_RESET_DONE
            else:
                time.sleep(0.0001)

    def close(self):
        for shm in self._ctrl_shms:
            try:
                shm.close()
            except Exception:
                pass
        self._ctrl_shms.clear()
        if self._step_shm is not None:
            try:
                self._step_shm.close()
                self._step_shm.unlink()
            except Exception:
                pass
            self._step_shm = None
        if self._frame_shm is not None:
            try:
                self._frame_shm.close()
            except Exception:
                pass
            self._frame_shm = None
        if self._proc_manager is not None:
            self._proc_manager.stop()

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps.copy()
