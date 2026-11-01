#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Shared server lifecycle and isolated LLM collection episode execution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RECORDING_ROOT = REPO_ROOT / "recording_data"
SERVER_PORT = 50051
SERVER_READY_TEXT = "server ready"


@dataclass(frozen=True)
class ServerConfig:
    backend: str
    curobo_batch_size: int
    physics_step: int
    enable_deformable: bool
    publish_ros: bool = True
    enable_render: bool = False
    headless: bool = True

    def argv(self) -> list[str]:
        args = [
            sys.executable,
            "scripts/data_collector_server.py",
            "--enable_physics",
            "--enable_curobo",
            "--headless",
            "--backend",
            self.backend,
            "--curobo_batch_size",
            str(self.curobo_batch_size),
            "--physics_step",
            str(self.physics_step),
        ]
        if not self.headless:
            args.remove("--headless")
        if self.enable_render:
            args.append("--enable_render")
        if self.publish_ros:
            args.append("--publish_ros")
        if self.enable_deformable:
            args.append("--enable_deformable")
        return args

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "curobo_batch_size": self.curobo_batch_size,
            "physics_step": self.physics_step,
            "enable_deformable": self.enable_deformable,
            "publish_ros": self.publish_ros,
            "enable_render": self.enable_render,
            "headless": self.headless,
        }


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def recording_dirs(task_name: str) -> dict[str, float]:
    if not task_name or not RECORDING_ROOT.is_dir():
        return {}
    dirs: dict[str, float] = {}
    for path in RECORDING_ROOT.iterdir():
        if not path.is_dir():
            continue
        suffix = path.name[len(task_name) :]
        if path.name == task_name or (path.name.startswith(task_name) and suffix.isdigit()):
            try:
                dirs[str(path.resolve())] = path.stat().st_mtime
            except OSError:
                pass
    return dirs


def newest_recording_dir(task_name: str, before: dict[str, float], started: float) -> Path | None:
    after = recording_dirs(task_name)
    candidates: list[tuple[float, Path]] = []
    for path_str, mtime in after.items():
        if path_str not in before or mtime >= started - 1:
            candidates.append((mtime, Path(path_str)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def wait_for_task_result(recording_dir: Path, timeout_s: int) -> tuple[Path | None, dict[str, Any] | None]:
    task_result_path = recording_dir / "task_result.json"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if task_result_path.is_file():
            try:
                return task_result_path, load_json(task_result_path)
            except json.JSONDecodeError:
                time.sleep(1)
                continue
        extract_log = recording_dir / "extract.log"
        if extract_log.is_file():
            try:
                text = extract_log.read_text(encoding="utf-8", errors="ignore")[-4000:]
                if "Traceback" in text or "ERROR" in text:
                    # Keep polling briefly; extract can log non-fatal ERROR lines.
                    pass
            except OSError:
                pass
        time.sleep(2)
    return None, None


def port_open(host: str = "127.0.0.1", port: int = SERVER_PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def scene_dirs(task_dir: Path, limit: int) -> list[str]:
    candidates = [
        p.name
        for p in sorted(task_dir.iterdir(), key=lambda x: x.name)
        if p.is_dir() and (p / "scene.usda").is_file() and (p / "scene_info.json").is_file()
    ]
    return candidates[:limit]


def server_config_from_task_info(
    task_info: dict[str, Any], publish_ros: bool = True
) -> ServerConfig:
    # Map optional task metadata to server arguments, using
    # data_collector_server.py defaults for unspecified values.
    return ServerConfig(
        backend=str(task_info.get("backend", "cpu")),
        curobo_batch_size=int(task_info.get("curobo_batch_size", 10)),
        physics_step=int(task_info.get("physics_frequency", 60)),
        enable_deformable=bool(task_info.get("enable_deformable", False)),
        publish_ros=publish_ros,
    )


def start_server(cfg: ServerConfig, log_path: Path, timeout_s: int) -> subprocess.Popen:
    if port_open():
        raise RuntimeError(
            f"gRPC port {SERVER_PORT} is already open; stop the existing server "
            "before starting collection"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    env = os.environ.copy()
    env.setdefault("ROS_DOMAIN_ID", "77")
    env.setdefault("ROS_LOCALHOST_ONLY", "1")
    env.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    proc = subprocess.Popen(
        cfg.argv(),
        cwd=str(REPO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    deadline = time.time() + timeout_s
    last_size = 0
    while time.time() < deadline:
        if proc.poll() is not None:
            log_file.close()
            raise RuntimeError(f"server exited during startup with code {proc.returncode}")
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="ignore")
                last_size = len(text)
                if SERVER_READY_TEXT in text or port_open():
                    return proc
            except OSError:
                pass
        time.sleep(2)
    log_file.close()
    raise TimeoutError(f"server was not ready in {timeout_s}s; log_size={last_size}")


def stop_server(proc: subprocess.Popen | None, timeout_s: int = 30) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def run_episode_subprocess(
    task_root: Path,
    task: str,
    scene: str,
    log_path: Path,
    result_path: Path,
    client_host: str,
    record: bool,
    timeout_s: int,
    task_result_timeout_s: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--task-root",
        str(task_root),
        "--task",
        task,
        "--scene",
        scene,
        "--client-host",
        client_host,
        "--episode-result",
        str(result_path),
    ]
    if record:
        cmd.append("--record")
        cmd.extend(["--task-result-timeout", str(task_result_timeout_s)])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
            return {
                "task": task,
                "scene": scene,
                "success": False,
                "returncode": None,
                "duration_s": round(time.time() - started, 3),
                "error": f"timeout after {timeout_s}s",
                "log": str(log_path),
            }
    result: dict[str, Any]
    if result_path.is_file():
        result = load_json(result_path)
    else:
        result = {
            "task": task,
            "scene": scene,
            "success": False,
            "error": "episode result json was not written",
        }
    result["returncode"] = rc
    result["duration_s"] = round(time.time() - started, 3)
    result["log"] = str(log_path)
    if rc != 0 and result.get("success") is not True:
        result.setdefault("error", f"client exited with returncode {rc}")
    return result


def load_task_module(task_dir: Path, task: str):
    """Load data_gen.py with its sibling helper modules available to the episode."""
    task_path = str(task_dir.resolve())
    if task_path not in sys.path:
        sys.path.insert(0, task_path)
    spec = importlib.util.spec_from_file_location(
        f"collection_data_gen_{task}_{int(time.time() * 1000)}",
        task_dir / "data_gen.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load data_gen.py for {task}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_episode(args: argparse.Namespace) -> int:
    started = time.time()
    robot = None
    result = {
        "task": args.task,
        "scene": args.scene,
        "scene_dir": str(Path(args.task_root) / args.task / args.scene),
        "started_at": now_iso(),
        "success": False,
    }
    try:
        task_root = Path(args.task_root)
        task_dir = task_root / args.task
        scene_dir = task_dir / args.scene
        scene_usda = scene_dir / "scene.usda"
        scene_info = scene_dir / "scene_info.json"
        task_info_json = scene_dir / "task_info.json"
        if not task_info_json.is_file():
            task_info_json = task_dir / "task_info.json"
        if not scene_usda.is_file() or not scene_info.is_file():
            raise FileNotFoundError(f"missing scene files under {scene_dir}")

        from client.llm_gen.launcher import change_payload
        from client.llm_gen.robot_api import RobotAPI, RobotAPIConfig

        change_payload(str(scene_usda))
        config = RobotAPIConfig.from_json(str(task_info_json))
        config.auto_record = args.record
        recording_task_name = config.task_name or args.task
        recording_before = recording_dirs(recording_task_name) if args.record else {}
        recording_started = time.time()
        robot = RobotAPI(
            config=config,
            scene_usd=str(scene_usda),
            scene_info=str(scene_info),
            client_host=args.client_host,
        )
        if args.record:
            result["recording_task_name"] = recording_task_name
            recording_dir = newest_recording_dir(recording_task_name, recording_before, recording_started)
            if recording_dir is not None:
                result["recording_dir"] = str(recording_dir)
        module = load_task_module(task_dir, args.task)
        try:
            success = bool(module.execute_task(robot))
        except Exception as exc:
            result["exception"] = repr(exc)
            result["traceback"] = traceback.format_exc()
            success = False
        try:
            robot.stop_recording()
        except Exception as exc:
            result["stop_recording_error"] = repr(exc)
        try:
            robot.client.send_task_status(success, [robot.stage_id, -1])
        except Exception as exc:
            result["send_task_status_error"] = repr(exc)
        result["execute_task_success"] = success
        result["success"] = success
        if args.record:
            recording_dir = Path(result["recording_dir"]) if result.get("recording_dir") else None
            if recording_dir is None:
                recording_dir = newest_recording_dir(recording_task_name, recording_before, recording_started)
                if recording_dir is not None:
                    result["recording_dir"] = str(recording_dir)
            if success and recording_dir is not None:
                task_result_path, task_result = wait_for_task_result(recording_dir, args.task_result_timeout)
                if task_result_path is not None and task_result is not None:
                    result["task_result_path"] = str(task_result_path)
                    result["task_result"] = task_result
                    copy_path = Path(args.episode_result).with_suffix(".task_result.json")
                    shutil.copy2(task_result_path, copy_path)
                    result["task_result_report_copy"] = str(copy_path)
                    result["success"] = task_result.get("task_status") is True
                    if result["success"] is not True:
                        result.setdefault(
                            "error",
                            f"task_result.task_status={task_result.get('task_status')}",
                        )
                else:
                    result["success"] = False
                    result.setdefault(
                        "error",
                        f"task_result.json not found within {args.task_result_timeout}s",
                    )
            else:
                result["success"] = False
                result.setdefault("error", result.get("exception", "execute_task returned false"))
        elif not success:
            result.setdefault("error", result.get("exception", "execute_task returned false"))
    except Exception as exc:
        result["success"] = False
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        if robot is not None:
            try:
                robot.exit()
            except Exception as exc:
                result["exit_error"] = repr(exc)
        result["finished_at"] = now_iso()
        result["duration_s"] = round(time.time() - started, 3)
        write_json(Path(args.episode_result), result)
    return 0 if result["success"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--episode-result", required=True)
    parser.add_argument("--client-host", default="localhost:50051")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--task-result-timeout", type=int, default=600)
    return run_episode(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
