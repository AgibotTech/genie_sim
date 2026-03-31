# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
#
# ProcessManager — launches and supervises N MuJoCo physics processes and 1
# IsaacSim renderer process for the parallel RL pipeline.
#
# Each MuJoCo process gets its own ROS namespace (/env_i) so their topics
# do not collide.  The IsaacSim renderer subscribes to all N namespaces.
#

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


_MUJOCO_NODE_SCRIPT = Path(__file__).parent.parent / "mujoco" / "mujoco_ros_node.py"
_RENDERER_SCRIPT   = Path(__file__).parent.parent / "renderer" / "rl_renderer.py"

# Source root containing the `geniesim` package (3 dirs above this file).
# Added to PYTHONPATH of child processes so they can import geniesim.*.
_GENIESIM_SRC_ROOT = str(Path(__file__).parent.parent.parent.parent)

# Log directory for child-process stdout/stderr.
_LOG_DIR = Path("/tmp/geniesim_logs")


class ProcessManager:
    """
    Manages the lifecycle of all simulation processes:
      - N MuJoCo ROS physics processes (one per parallel environment)
      - 1 IsaacSim renderer process

    Parameters
    ----------
    num_envs : int
        Number of parallel environments.
    mjcf_path : str
        Path to the MuJoCo XML scene file (same scene for all envs).
    scene_usd : str
        Path to the IsaacSim USD scene.
    robot_usd : str
        Path to the robot USDA asset.
    robot_prim : str
        Prim path of robot root inside the env (e.g. "/robot").
    shm_name : str
        Shared memory segment name for camera images.
    physics_hz : int
        MuJoCo physics frequency.
    render_hz : float
        IsaacSim rendering frequency.
    cam_width / cam_height : int
        Camera resolution.
    main_cam_prim / wrist_cam_prim : str
        Camera prim paths relative to env root.
    headless : bool
        Run IsaacSim headless.
    ros_domain_id : int
        ROS_DOMAIN_ID for all processes.
    isaac_python : str
        Path to the Isaac Sim python executable.
    extra_env : dict | None
        Additional environment variables to forward.
    """

    def __init__(
        self,
        num_envs: int,
        mjcf_path: str,
        scene_usd: str = "",
        robot_usd: str = "",
        robot_prim: str = "/robot",
        shm_name: str = "geniesim_frames",
        physics_hz: int = 1000,
        render_hz: float = 30.0,
        cam_width: int = 640,
        cam_height: int = 480,
        main_cam_prim: str = "/camera_main",
        wrist_cam_prim: str = "",
        cameras_json: str = "",
        headless: bool = True,
        ros_domain_id: int = 0,
        isaac_python: str = "/isaac-sim/python.sh",
        extra_env: Optional[Dict[str, str]] = None,
        # Task-name mode: when provided, renderer auto-resolves all USD assets.
        task_name: str = "",
        robot_type: str = "G2",
        task_instance_id: int = 0,
        init_qpos_json: str = "",
        # Python executable for MuJoCo subprocesses.
        # When None, falls back to sys.executable.
        # Override when the caller's Python (e.g. a training venv) doesn't
        # have `mujoco` installed.
        mujoco_python: Optional[str] = None,
        # Ctrl SHM joint selection.
        # state_joint_offset: first non-free joint index to expose as state
        # ctrl_offset: first ctrl[] index the host actions map to
        # state_dim / action_dim: host API dimensions (must match GenieSimVectorEnvConfig)
        state_joint_offset: int = 0,
        ctrl_offset: int = 0,
        ctrl_offset_r: int = -1,
        state_dim: int = 0,
        action_dim: int = 0,
        control_mode: str = "joint",
        gripper_ctrl_l: int = -1,
        gripper_ctrl_r: int = -1,
        ee_body_l: str = "arm_l_link7",
        ee_body_r: str = "arm_r_link7",
        ik_max_iter: int = 10,
        ik_damp: float = 0.05,
        randomization_cfg_json: str = "",
        reset_ee_r_json: str = "",
        seed: int = 0,
        info_body_names: Optional[List[str]] = None,
        sync_mode: bool = True,
        steps_per_step: int = 33,
    ):
        self.num_envs = num_envs
        self.mjcf_path = mjcf_path
        self.scene_usd = scene_usd
        self.robot_usd = robot_usd
        self.robot_prim = robot_prim

        if task_name:
            self._resolve_mjcf_from_task(task_name)

        self.shm_name = shm_name
        self.physics_hz = physics_hz
        self.render_hz = render_hz
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.main_cam_prim = main_cam_prim
        self.wrist_cam_prim = wrist_cam_prim
        self.cameras_json = cameras_json
        self.headless = headless
        self.ros_domain_id = ros_domain_id
        self.isaac_python = isaac_python
        self.task_name = task_name
        self.robot_type = robot_type
        self.task_instance_id = task_instance_id
        self.init_qpos_json = init_qpos_json
        self.mujoco_python = mujoco_python or sys.executable
        self.state_joint_offset = state_joint_offset
        self.ctrl_offset = ctrl_offset
        self.ctrl_offset_r = ctrl_offset_r
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.control_mode = control_mode
        self.gripper_ctrl_l = gripper_ctrl_l
        self.gripper_ctrl_r = gripper_ctrl_r
        self.ee_body_l = ee_body_l
        self.ee_body_r = ee_body_r
        self.ik_max_iter = ik_max_iter
        self.ik_damp = ik_damp
        self.randomization_cfg_json = randomization_cfg_json
        self.reset_ee_r_json = reset_ee_r_json
        self.seed = seed
        self.info_body_names = info_body_names or []
        self.sync_mode = sync_mode
        self.steps_per_step = steps_per_step

        self._base_env = os.environ.copy()
        self._base_env["ROS_DOMAIN_ID"] = str(ros_domain_id)
        # Ensure child processes can import geniesim.* even when PYTHONPATH is
        # not pre-populated (e.g. when launched from inside a Docker container).
        _existing_pp = self._base_env.get("PYTHONPATH", "")
        if _GENIESIM_SRC_ROOT not in _existing_pp.split(":"):
            self._base_env["PYTHONPATH"] = (
                f"{_GENIESIM_SRC_ROOT}:{_existing_pp}" if _existing_pp
                else _GENIESIM_SRC_ROOT
            )
        if extra_env:
            self._base_env.update(extra_env)

        self._mujoco_procs: List[subprocess.Popen] = []
        self._mujoco_log_files: List = []   # open file handles
        self._renderer_proc: Optional[subprocess.Popen] = None
        self._renderer_log_file = None

        # Ensure clean kill on Ctrl+C or process exit.
        atexit.register(self.stop)
        signal.signal(signal.SIGINT, self._sigint_handler)
        signal.signal(signal.SIGTERM, self._sigterm_handler)

        _LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _resolve_mjcf_from_task(self, task_name: str):
        from geniesim.benchmark.config.task_config_mapping import TASK_MAPPING

        mapping = TASK_MAPPING.get(task_name, {})
        mjcf_rel = mapping.get("mjcf", "")
        file_path = os.path.realpath(__file__)
        assets_dir = os.path.join(os.path.dirname(file_path), "../../assets")
        if mjcf_rel:
            resolved = os.path.join(assets_dir, mjcf_rel)
            print(f"[ProcessManager] Resolved mjcf_path from task '{task_name}': {resolved}")
            self.mjcf_path = resolved

    # ---------------------------------------------------------------------- #
    # Signal / atexit handlers
    # ---------------------------------------------------------------------- #

    def _sigint_handler(self, signum, frame):
        print("\n[ProcessManager] SIGINT received — stopping all child processes...")
        self.stop()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)

    def _sigterm_handler(self, signum, frame):
        print("\n[ProcessManager] SIGTERM received — stopping all child processes...")
        self.stop()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    # ---------------------------------------------------------------------- #
    # Start / stop
    # ---------------------------------------------------------------------- #

    def start(self, wait_ready_sec: float = 15.0):
        """Launch all processes and wait until they are ready."""
        self._start_renderer()
        self._start_mujoco_procs()
        self._wait_ready(wait_ready_sec)

    def stop(self):
        """Terminate all managed processes."""
        self._kill_procs(self._mujoco_procs)
        self._mujoco_procs.clear()
        for fh in self._mujoco_log_files:
            try:
                fh.close()
            except Exception:
                pass
        self._mujoco_log_files.clear()
        if self._renderer_proc is not None:
            _kill_proc(self._renderer_proc)
            self._renderer_proc = None
        if self._renderer_log_file is not None:
            try:
                self._renderer_log_file.close()
            except Exception:
                pass
            self._renderer_log_file = None

    def health_check(self) -> bool:
        """Return True if all processes are still alive."""
        for p in self._mujoco_procs:
            if p.poll() is not None:
                return False
        if self._renderer_proc is not None and self._renderer_proc.poll() is not None:
            return False
        return True

    def restart_dead(self):
        """Restart any crashed physics process."""
        for i, p in enumerate(self._mujoco_procs):
            if p.poll() is not None:
                print(f"[ProcessManager] env_{i} crashed (exit={p.returncode}), restarting...")
                log_path = _LOG_DIR / f"mujoco_env_{i}.log"
                log_fh = open(log_path, "w")
                self._mujoco_log_files[i] = log_fh
                self._mujoco_procs[i] = self._launch_mujoco(i, log_fh)

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _start_renderer(self):
        cmd = [self.isaac_python, str(_RENDERER_SCRIPT)]

        if self.task_name:
            cmd += [
                "--task-name", self.task_name,
                "--robot-type", self.robot_type,
                "--task-instance-id", str(self.task_instance_id),
            ]
        else:
            cmd += [
                "--scene-usd", self.scene_usd,
                "--robot-usd", self.robot_usd,
                "--robot-prim", self.robot_prim,
                "--main-cam-prim", self.main_cam_prim,
            ]

        cmd += [
            "--num-envs", str(self.num_envs),
            "--render-hz", str(self.render_hz),
            "--cam-width", str(self.cam_width),
            "--cam-height", str(self.cam_height),
            "--shm-name", self.shm_name,
            "--ros-domain-id", str(self.ros_domain_id),
        ]
        if self.wrist_cam_prim:
            cmd += ["--wrist-cam-prim", self.wrist_cam_prim]
        if self.cameras_json:
            cmd += ["--cameras-json", self.cameras_json]
        if self.headless:
            cmd.append("--headless")

        # Build a renderer-specific env for Isaac Sim (Python 3.11).
        #
        # Problem: the driver process (Python 3.12) propagates ROS2 Python-3.12
        # site-packages via PYTHONPATH.  Isaac Sim's Python 3.11 would find those
        # Python-3.12 packages first, then fail trying to load cpython-312 C
        # extensions (e.g. rclpy, rcl_interfaces).
        #
        # Fix:
        #   1. Prepend Isaac Sim's bundled Python-3.11 ROS2 packages so Python 3.11
        #      finds the correct rclpy / rcl_interfaces before the Python-3.12 ones.
        #   2. Strip all Python-3.12 site-packages from PYTHONPATH so the bundled
        #      Python-3.11 packages are used consistently (no ABI mixing).
        #   3. Add the bundled native ROS2 libs to LD_LIBRARY_PATH so the C
        #      extensions (_rclpy_pybind11.cpython-311-*.so etc.) can dlopen them.
        _ISAAC_HOME = self.isaac_python.replace("/python.sh", "")
        _BRIDGE_ROOT = os.path.join(_ISAAC_HOME, "exts", "isaacsim.ros2.bridge")
        _BRIDGE_RCLPY = os.path.join(_BRIDGE_ROOT, "jazzy", "rclpy")
        _BRIDGE_LIB   = os.path.join(_BRIDGE_ROOT, "jazzy", "lib")

        renderer_env = self._base_env.copy()

        # PYTHONPATH: bundled path first, then non-Python-3.12 entries.
        _pypath = renderer_env.get("PYTHONPATH", "")
        _filtered = [p for p in _pypath.split(":") if p and "python3.12" not in p]
        renderer_env["PYTHONPATH"] = ":".join([_BRIDGE_RCLPY] + _filtered)

        # LD_LIBRARY_PATH: add bundled native libs so C extensions resolve.
        _ldpath = renderer_env.get("LD_LIBRARY_PATH", "")
        renderer_env["LD_LIBRARY_PATH"] = (
            f"{_BRIDGE_LIB}:{_ldpath}" if _ldpath else _BRIDGE_LIB
        )

        log_path = _LOG_DIR / "renderer.log"
        self._renderer_log_file = open(log_path, "w")
        self._renderer_proc = subprocess.Popen(
            cmd,
            env=renderer_env,
            stdout=self._renderer_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(
            f"[ProcessManager] IsaacSim renderer launched "
            f"(pid={self._renderer_proc.pid}, log={log_path})"
        )

    def _launch_mujoco(self, env_id: int, log_fh) -> subprocess.Popen:
        namespace = f"env_{env_id}"
        cmd = [
            self.mujoco_python,
            str(_MUJOCO_NODE_SCRIPT),
            "--mjcf", self.mjcf_path,
            "--namespace", namespace,
            "--physics-hz", str(self.physics_hz),
            "--render-hz", str(int(self.render_hz)),
            "--ros-domain-id", str(self.ros_domain_id),
            "--shm-name", self.shm_name,
        ]
        if self.init_qpos_json:
            cmd += ["--init-qpos-json", self.init_qpos_json]
        if self.state_joint_offset:
            cmd += ["--state-joint-offset", str(self.state_joint_offset)]
        if self.ctrl_offset:
            cmd += ["--ctrl-offset", str(self.ctrl_offset)]
        if self.ctrl_offset_r >= 0:
            cmd += ["--ctrl-offset-r", str(self.ctrl_offset_r)]
        if self.state_dim:
            cmd += ["--state-dim", str(self.state_dim)]
        if self.action_dim:
            cmd += ["--action-dim", str(self.action_dim)]
        if self.control_mode != "joint":
            cmd += ["--control-mode", self.control_mode]
        if self.gripper_ctrl_l >= 0:
            cmd += ["--gripper-ctrl-l", str(self.gripper_ctrl_l)]
        if self.gripper_ctrl_r >= 0:
            cmd += ["--gripper-ctrl-r", str(self.gripper_ctrl_r)]
        if self.control_mode == "ee":
            cmd += ["--ee-body-l", self.ee_body_l]
            cmd += ["--ee-body-r", self.ee_body_r]
            cmd += ["--ik-max-iter", str(self.ik_max_iter)]
            cmd += ["--ik-damp", str(self.ik_damp)]
        cmd += ["--seed", str(self.seed + env_id)]
        if self.randomization_cfg_json:
            cmd += ["--randomization-cfg", self.randomization_cfg_json]
        if self.reset_ee_r_json:
            cmd += ["--reset-ee-r-json", self.reset_ee_r_json]
        if self.info_body_names:
            cmd += ["--info-body-names"] + self.info_body_names
        if self.sync_mode:
            cmd += ["--sync-mode", "--steps-per-step", str(self.steps_per_step)]
        if not self.headless and env_id == 0:
            cmd.append("--viewer")
        proc = subprocess.Popen(
            cmd,
            env=self._base_env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_path = _LOG_DIR / f"mujoco_env_{env_id}.log"
        print(f"[ProcessManager] MuJoCo env_{env_id} launched (pid={proc.pid}, log={log_path})")
        return proc

    def _start_mujoco_procs(self):
        for i in range(self.num_envs):
            log_path = _LOG_DIR / f"mujoco_env_{i}.log"
            log_fh = open(log_path, "w")
            self._mujoco_log_files.append(log_fh)
            self._mujoco_procs.append(self._launch_mujoco(i, log_fh))

    def _wait_ready(self, timeout: float):
        """
        Wait until physics processes print their 'ready' message to the log.

        Tails per-process log files to detect "MuJoCoRosNode ready",
        avoiding any dependency on the `ros2` CLI.
        """
        deadline = time.time() + timeout
        namespaces_ready: set = set()
        target = {f"env_{i}" for i in range(self.num_envs)}

        print(f"[ProcessManager] Waiting for {self.num_envs} MuJoCo envs to be ready...")

        while time.time() < deadline:
            for i in range(self.num_envs):
                ns = f"env_{i}"
                if ns in namespaces_ready:
                    continue
                # Fail fast: if the process crashed, there is no point waiting.
                p = self._mujoco_procs[i] if i < len(self._mujoco_procs) else None
                if p is not None and p.poll() is not None:
                    log_path = _LOG_DIR / f"mujoco_env_{i}.log"
                    try:
                        last_lines = log_path.read_text().splitlines()[-20:]
                    except Exception:
                        last_lines = []
                    raise RuntimeError(
                        f"[ProcessManager] env_{i} crashed (rc={p.returncode}) "
                        f"during startup wait.\n"
                        f"Last log lines:\n"
                        + "\n".join(f"  {l}" for l in last_lines)
                    )
                log_path = _LOG_DIR / f"mujoco_env_{i}.log"
                try:
                    if "MuJoCoRosNode ready" in log_path.read_text():
                        namespaces_ready.add(ns)
                        print(f"[ProcessManager] {ns} ready")
                except Exception:
                    pass

            if namespaces_ready == target:
                print("[ProcessManager] All MuJoCo envs ready.")
                return
            time.sleep(0.5)

        missing = target - namespaces_ready
        print(f"[ProcessManager] WARNING: Timeout waiting for envs: {missing}")
        print(f"[ProcessManager] Check logs in {_LOG_DIR} for details.")

    @staticmethod
    def _kill_procs(procs: List[subprocess.Popen]):
        for p in procs:
            _kill_proc(p)


def _kill_proc(p: subprocess.Popen, wait_sec: float = 10.0):
    """Send SIGTERM to the entire process group, wait, then always SIGKILL.

    Using `start_new_session=True` means the launched shell script (e.g.
    python.sh) is the PGID leader.  When python.sh exits after SIGTERM, its
    heavy children (e.g. IsaacSim's kit/python3) may still be alive in the same
    PGID.  We must SIGKILL the PGID *after* the wait to clean those up.
    """
    try:
        pgid = os.getpgid(p.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(timeout=wait_sec)
    except subprocess.TimeoutExpired:
        pass
    # Always SIGKILL the group to handle children that outlive the shell wrapper.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
