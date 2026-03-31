#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
#
# GenieSim simulation server.
#
# Intended to run as the main process of the geniesim-rlinf Docker container
# when launched by RLinf's SimContainerManager.  Starts and supervises
# Isaac Sim + MuJoCo physics processes via ProcessManager, writes a readiness
# marker to a bind-mounted path so the host can detect when the simulation is
# ready, then runs a supervision loop until terminated.
#
# Usage (inside container, after entrypoint sources ROS + workspace):
#   python3 sim_server.py \
#       --config-json /geniesim/main/.sim_server_config.json \
#       --ready-file  /geniesim/main/.geniesim_ready
#
# The config JSON must contain ProcessManager.__init__() keyword arguments.
# See SimContainerManager.pm_kwargs_from_vec_cfg() for the exact schema.
#

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

_PROGRESS_FILE = ".geniesim_progress"
_STOP_FILE     = ".geniesim_stop"   # host writes → stop processes, enter idle
_START_FILE    = ".geniesim_start"  # host writes → restart processes from idle
_IDLE_FILE     = ".geniesim_idle"   # sim_server writes → container idle
_ERROR_FILE    = ".geniesim_error"  # sim_server writes → startup failed (renderer crash)


def _bootstrap_geniesim() -> None:
    """Add geniesim source to sys.path using SIM_REPO_ROOT env var."""
    repo_root = os.environ.get("SIM_REPO_ROOT", "/geniesim/main/main")
    src = os.path.join(repo_root, "source")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def main() -> None:
    parser = argparse.ArgumentParser(description="GenieSim simulation server")
    parser.add_argument(
        "--config-json", required=True,
        help="Path to JSON file with ProcessManager kwargs",
    )
    parser.add_argument(
        "--ready-file", default="/tmp/geniesim_ready",
        help="Path to write readiness marker (must be on a bind-mounted volume "
             "so the host can observe it)",
    )
    args = parser.parse_args()

    _bootstrap_geniesim()
    from geniesim.rl.envs.geniesim_vec_env import GenieSimVectorEnv, GenieSimVectorEnvConfig  # noqa: E402
    from geniesim.rl.renderer.shm_layout import (  # noqa: E402
        ctrl_shm_name as _ctrl_shm_name,
        step_shm_name as _step_shm_name,
    )

    ready_path    = Path(args.ready_file)
    base_dir      = ready_path.parent
    stop_path     = base_dir / _STOP_FILE
    start_path    = base_dir / _START_FILE
    idle_path     = base_dir / _IDLE_FILE
    progress_path = base_dir / _PROGRESS_FILE
    error_path    = base_dir / _ERROR_FILE
    cfg_path      = Path(args.config_json)

    # Clear any stale sentinel files from a previous run.
    for _p in [ready_path, stop_path, start_path, idle_path, progress_path, error_path]:
        _p.unlink(missing_ok=True)

    def _write_progress(stage: str) -> None:
        try:
            progress_path.write_text(stage)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Signal handling  (SIGTERM / SIGINT from `docker stop` or Ctrl+C)
    # ------------------------------------------------------------------ #
    pm_ref: list = []  # mutable cell so the handler can reach the current pm

    def _shutdown(signum, _frame) -> None:
        print(f"\n[SimServer] Signal {signum} received — shutting down...", flush=True)
        for _p in [ready_path, idle_path, stop_path, start_path]:
            _p.unlink(missing_ok=True)
        if pm_ref:
            pm_ref[0].stop()
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # SIGCHLD reaper — sim_server.py runs as PID 1 in the container.
    # When Isaac Sim forks grandchildren (omni.telemetry, carb.tasking, …)
    # and they are later orphaned by SIGKILL, they are re-parented to PID 1.
    # Without a SIGCHLD handler, these orphans accumulate as zombies.
    #
    # IMPORTANT: waitpid(-1, WNOHANG) reaps ANY child, including the renderer
    # and MuJoCo Popen objects.  If we silently discard the exit status,
    # Popen.poll() sees ChildProcessError and returns None forever, breaking
    # crash detection.  We preserve the exit code by writing it directly into
    # the Popen object's returncode attribute so poll() works correctly.
    # ------------------------------------------------------------------ #
    _managed_pids: dict = {}  # pid → Popen object; updated after each pm.start()

    def _reap_children(signum, _frame) -> None:
        """Reap all exited children; preserve exit codes for Popen-managed procs."""
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid <= 0:
                    break
                # Decode raw wait status into a return code.
                if os.WIFEXITED(status):
                    rc = os.WEXITSTATUS(status)
                elif os.WIFSIGNALED(status):
                    rc = -os.WTERMSIG(status)
                else:
                    rc = -1
                # If this is a Popen-managed child, store the exit code so
                # that poll() can return it correctly despite us having
                # already consumed the wait status.
                popen_obj = _managed_pids.pop(pid, None)
                if popen_obj is not None:
                    popen_obj.returncode = rc
            except ChildProcessError:
                break

    signal.signal(signal.SIGCHLD, _reap_children)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # ------------------------------------------------------------------ #
    # Main loop — supports multiple start / stop cycles (keep_alive mode)
    # ------------------------------------------------------------------ #
    first_cycle = True

    while True:
        if not first_cycle:
            # Enter idle state: wait for the host to send a start signal.
            print("[SimServer] Idle. Waiting for start signal from host...", flush=True)
            idle_path.write_text("idle")
            while not start_path.exists():
                time.sleep(0.5)
            start_path.unlink(missing_ok=True)
            idle_path.unlink(missing_ok=True)
            print("[SimServer] Start signal received — restarting sim processes...",
                  flush=True)
        first_cycle = False

        # Load (possibly updated) config each cycle.
        if not cfg_path.exists():
            print(f"[SimServer] ERROR: config file not found: {cfg_path}", flush=True)
            sys.exit(1)
        cfg: dict = json.loads(cfg_path.read_text())
        print(
            f"[SimServer] Config loaded | num_envs={cfg.get('num_envs')} "
            f"task={cfg.get('task_name')!r}",
            flush=True,
        )

        # ------------------------------------------------------------------ #
        # Clean up stale SHM from a previous run.
        # ------------------------------------------------------------------ #
        from geniesim.rl.renderer.shm_layout import ctrl_shm_name as _ctrl_shm_name
        shm_name: str = cfg.get("shm_name", "geniesim_frames")
        shm_path = Path("/dev/shm") / shm_name
        if shm_path.exists():
            try:
                shm_path.unlink()
                print(f"[SimServer] Removed stale SHM '{shm_name}'", flush=True)
            except OSError as exc:
                print(f"[SimServer] WARNING: could not remove stale SHM "
                      f"'{shm_name}': {exc}", flush=True)
        for _env_id in range(cfg.get("num_envs", 1)):
            _ctrl_path = Path("/dev/shm") / _ctrl_shm_name(shm_name, _env_id)
            if _ctrl_path.exists():
                try:
                    _ctrl_path.unlink()
                    print(f"[SimServer] Removed stale ctrl SHM "
                          f"'{_ctrl_path.name}'", flush=True)
                except OSError as exc:
                    print(f"[SimServer] WARNING: could not remove stale ctrl SHM "
                          f"'{_ctrl_path.name}': {exc}", flush=True)
        _step_path = Path("/dev/shm") / _step_shm_name(shm_name)
        if _step_path.exists():
            try:
                _step_path.unlink()
                print(f"[SimServer] Removed stale step SHM "
                      f"'{_step_path.name}'", flush=True)
            except OSError as exc:
                print(f"[SimServer] WARNING: could not remove stale step SHM "
                      f"'{_step_path.name}': {exc}", flush=True)

        # ------------------------------------------------------------------ #
        # Launch simulation via GenieSimVectorEnv.
        # ------------------------------------------------------------------ #
        _write_progress("mujoco_launching")
        print("[SimServer] Creating GenieSimVectorEnv...", flush=True)
        vec_env = None
        try:
            vec_cfg = GenieSimVectorEnvConfig(**cfg)
            vec_env = GenieSimVectorEnv(vec_cfg)
            pm_ref.clear()
            if vec_env._proc_manager is not None:
                pm_ref.append(vec_env._proc_manager)
            _write_progress("mujoco_ready")
            _managed_pids.clear()
            pm = vec_env._proc_manager
            if pm is not None:
                if pm._renderer_proc is not None:
                    _managed_pids[pm._renderer_proc.pid] = pm._renderer_proc
                for _p in pm._mujoco_procs:
                    _managed_pids[_p.pid] = _p
        except Exception as exc:
            msg = str(exc)
            print(f"[SimServer] FAILED to start simulation: {msg}", flush=True)
            error_path.write_text(msg)
            _write_progress("error")
            ready_path.unlink(missing_ok=True)
            _managed_pids.clear()
            if vec_env is not None:
                try:
                    vec_env.close()
                except Exception:
                    pass
                vec_env = None
            elif pm_ref:
                try:
                    pm_ref[0].stop()
                except Exception:
                    pass
            pm_ref.clear()
            # Enter idle so the container stays alive for debugging.
            first_cycle = False
            continue

        # ------------------------------------------------------------------ #
        # Wait for Isaac Sim SHM, then fix permissions.
        # Check every second whether the renderer process is still alive so
        # that a renderer crash is detected immediately rather than after the
        # full 300-second timeout.
        # ------------------------------------------------------------------ #
        shm_wait_sec = 300
        _write_progress("isaac_launching")
        print(
            f"[SimServer] Waiting for Isaac Sim SHM '{shm_name}' "
            f"(up to {shm_wait_sec}s)...",
            flush=True,
        )
        deadline = time.time() + shm_wait_sec
        _shm_found = False
        _renderer_crashed = False
        while time.time() < deadline:
            if shm_path.exists():
                try:
                    shm_path.chmod(0o666)
                except PermissionError:
                    pass
                _write_progress("isaac_ready")
                print(
                    f"[SimServer] SHM ready — permissions set to 0o666 ({shm_path})",
                    flush=True,
                )
                _shm_found = True
                break
            # Detect renderer crash early: if it died, there is no point waiting.
            if pm_ref and pm_ref[0]._renderer_proc is not None:
                if pm_ref[0]._renderer_proc.poll() is not None:
                    rc = pm_ref[0]._renderer_proc.returncode
                    _renderer_crashed = True
                    msg = (
                        f"Isaac Sim renderer exited unexpectedly (rc={rc}) before "
                        f"creating SHM '{shm_name}'.\n"
                        f"See /tmp/geniesim_logs/renderer.log for details."
                    )
                    print(f"[SimServer] ERROR: {msg}", flush=True)
                    error_path.write_text(msg)
                    _write_progress("error")
                    pm_ref[0].stop()
                    pm_ref.clear()
                    break
            # Detect renderer still alive but broken: ExternalShutdownException
            # kills Isaac Sim's ROS spin thread while the process keeps running,
            # so poll() stays None forever and SHM is never created.
            # Check the renderer log for this and similar crash signatures.
            _renderer_log = Path("/tmp/geniesim_logs/renderer.log")
            _BROKEN_PATTERNS = [
                "ExternalShutdownException",
                "rclpy.executors.ExternalShutdownException",
            ]
            try:
                if not _renderer_crashed and _renderer_log.exists():
                    _log_text = _renderer_log.read_text()
                    for _pat in _BROKEN_PATTERNS:
                        if _pat in _log_text:
                            _renderer_crashed = True
                            msg = (
                                f"Isaac Sim renderer ROS spin thread crashed "
                                f"('{_pat}' in renderer.log) before SHM "
                                f"'{shm_name}' was created.\n"
                                f"The renderer process is alive but broken — "
                                f"killing it and entering idle.\n"
                                f"See /tmp/geniesim_logs/renderer.log for details."
                            )
                            print(f"[SimServer] ERROR: {msg}", flush=True)
                            error_path.write_text(msg)
                            _write_progress("error")
                            if pm_ref:
                                pm_ref[0].stop()
                                pm_ref.clear()
                            break
            except Exception:
                pass
            if _renderer_crashed:
                break
            time.sleep(1.0)
        else:
            # Timeout — SHM still not found.
            if pm_ref and pm_ref[0]._renderer_proc is not None:
                if pm_ref[0]._renderer_proc.poll() is not None:
                    _renderer_crashed = True

            if _renderer_crashed or not _shm_found:
                rc = (pm_ref[0]._renderer_proc.returncode
                      if pm_ref and pm_ref[0]._renderer_proc else "?")
                msg = (
                    f"Isaac Sim renderer did not create SHM '{shm_name}' "
                    f"within {shm_wait_sec}s (renderer rc={rc}).\n"
                    f"See /tmp/geniesim_logs/renderer.log for details."
                )
                print(f"[SimServer] ERROR: {msg}", flush=True)
                error_path.write_text(msg)
                _write_progress("error")
                if pm_ref:
                    pm_ref[0].stop()
                    pm_ref.clear()
                _renderer_crashed = True

        # If renderer crashed, enter idle without signalling ready to the host.
        if _renderer_crashed:
            first_cycle = False
            continue

        # ------------------------------------------------------------------ #
        # Signal readiness to the host.
        # ------------------------------------------------------------------ #
        ready_path.write_text("ready\n")
        print(f"[SimServer] READY — {ready_path}", flush=True)

        # ------------------------------------------------------------------ #
        # Run the step loop — GenieSimVectorEnv handles step requests from
        # RLinf.  A background thread watches for stop signals and renderer
        # crashes, sending STEP_PHASE_CLOSE into the step SHM to break the
        # loop when needed.
        # ------------------------------------------------------------------ #
        import threading

        _stop_event = threading.Event()

        def _supervision_thread():
            while not _stop_event.is_set():
                if stop_path.exists():
                    stop_path.unlink(missing_ok=True)
                    print(
                        "[SimServer] Stop requested by host — "
                        "signalling step loop to exit...",
                        flush=True,
                    )
                    from geniesim.rl.renderer.shm_layout import STEP_PHASE_CLOSE
                    if vec_env._step_phase is not None:
                        vec_env._step_phase[0] = STEP_PHASE_CLOSE
                    _stop_event.set()
                    break

                _mid_run_crash = False
                if pm_ref and pm_ref[0]._renderer_proc is not None:
                    if pm_ref[0]._renderer_proc.poll() is not None:
                        rc = pm_ref[0]._renderer_proc.returncode
                        msg = (
                            f"Isaac Sim renderer exited mid-run (rc={rc}).\n"
                            f"See /tmp/geniesim_logs/renderer.log for details."
                        )
                        print(f"[SimServer] ERROR: {msg}", flush=True)
                        error_path.write_text(msg)
                        _mid_run_crash = True
                    else:
                        _renderer_log = Path("/tmp/geniesim_logs/renderer.log")
                        try:
                            if _renderer_log.exists():
                                _log_text = _renderer_log.read_text()
                                if "ExternalShutdownException" in _log_text:
                                    msg = (
                                        f"Isaac Sim renderer ROS spin thread crashed "
                                        f"mid-run (ExternalShutdownException in renderer.log).\n"
                                        f"See /tmp/geniesim_logs/renderer.log for details."
                                    )
                                    print(f"[SimServer] ERROR: {msg}", flush=True)
                                    error_path.write_text(msg)
                                    _mid_run_crash = True
                        except Exception:
                            pass
                if _mid_run_crash:
                    from geniesim.rl.renderer.shm_layout import STEP_PHASE_CLOSE
                    if vec_env._step_phase is not None:
                        vec_env._step_phase[0] = STEP_PHASE_CLOSE
                    _stop_event.set()
                    break

                time.sleep(1.0)

        sup_thread = threading.Thread(target=_supervision_thread, daemon=True)
        sup_thread.start()

        try:
            vec_env.run_step_loop()
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[SimServer] Error in step loop: {exc}", flush=True)

        _stop_event.set()
        sup_thread.join(timeout=5.0)

        ready_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

        if vec_env is not None:
            try:
                vec_env.close()
            except Exception:
                pass
            vec_env = None
        pm_ref.clear()


if __name__ == "__main__":
    main()
