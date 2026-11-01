# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run each Wuji layout in a fresh, normally assembled ROS engine."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Callable

import yaml


@dataclass(frozen=True)
class LayoutRun:
    layout_id: int
    scene: Path
    evaluation: Path
    launcher: Path
    assets_root: Path
    manifest: Path
    run_dir: Path
    runtime_launcher: Path
    runtime_evaluation: Path
    engine_command: list[str]
    runner_command: list[str]


def _bundle_file(root: Path, value: str) -> Path:
    if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("bundle paths must be relative and stay inside the bundle")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("bundle path escapes the bundle")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_plans(
    bundle: Path,
    assets_root: Path,
    output_dir: Path,
    infer_host: str,
    *,
    layout_ids: list[int] | None = None,
    python_cmd: str | None = None,
    shutdown_timeout_s: float = 60,
) -> list[LayoutRun]:
    """Validate the complete index and plan commands without writing or starting anything."""
    root = Path(bundle).expanduser().resolve(strict=True)
    assets = Path(assets_root).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve()
    if not assets.is_dir():
        raise ValueError("assets root must be a directory")
    value = json.loads((root / "layouts.json").read_text())
    entries = value.get("layouts")
    if type(value.get("version")) is not int or value["version"] != 1 or value.get("task_name") != "wok_flip":
        raise ValueError("unsupported layout bundle protocol")
    if (
        not isinstance(entries, list)
        or not entries
        or type(value.get("count")) is not int
        or value["count"] != len(entries)
    ):
        raise ValueError("bundle count must match its nonempty layouts")
    interpreter = python_cmd or os.environ.get("GENIESIM_PY_CMD") or sys.executable
    plans, ids, stems = [], set(), set()
    for item in entries:
        layout_id = item.get("layout_id")
        if type(layout_id) is not int or layout_id < 0 or layout_id in ids:
            raise ValueError("layout_id must be a unique nonnegative integer")
        ids.add(layout_id)
        scene, evaluation, launcher = (_bundle_file(root, item[key]) for key in ("scene", "evaluation", "launcher"))
        expected = Path("background/common/runtime") / scene.stem / "manifest.json"
        legacy = (Path("scenes") / scene.stem / "manifest.json").as_posix()
        if item.get("manifest_subpath") not in (expected.as_posix(), legacy) or scene.stem in stems:
            raise ValueError("each layout needs its own scene stem and matching manifest_subpath")
        stems.add(scene.stem)
        scene_config = yaml.safe_load(scene.read_text())
        plugin_name = scene_config.get("newton", {}).get("scene_plugin")
        source_plugin = value.get("sources", {}).get("plugin", {})
        if not isinstance(plugin_name, str) or Path(plugin_name).name != plugin_name:
            raise ValueError("layout scene plugin must remain a sibling filename")
        plugin = _bundle_file(root, str(Path(item["scene"]).parent / plugin_name))
        if (
            source_plugin.get("name") != plugin_name
            or source_plugin.get("sha256") != hashlib.sha256(plugin.read_bytes()).hexdigest()
        ):
            raise ValueError("copied scene plugin sha256 does not match the source plugin")
        for key, path in (("scene", scene), ("evaluation", evaluation), ("launcher", launcher)):
            expected_sha = item.get("sha256", {}).get(key)
            if expected_sha is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
                raise ValueError(f"layout {layout_id} {key} sha256 does not match bundle")
        config = yaml.safe_load(evaluation.read_text())
        if config.get("runtime") != "newton" or config.get("task_name") != "wok_flip":
            raise ValueError("layout evaluations require runtime newton and task_name wok_flip")
        if type(config.get("episodes")) is not int or config["episodes"] != 1:
            raise ValueError("layout evaluations require episodes: 1; food has no reset contract")
        if not config.get("task_progress_topic"):
            raise ValueError("layout evaluations require authoritative task_progress_topic")
        if (evaluation.parent / config["launcher"]).resolve() != launcher:
            raise ValueError("evaluation launcher does not match its bundle entry")
        run = output / f"{layout_id:02d}"
        runtime_launcher, runtime_evaluation = run / "launcher.yaml", run / "evaluation.yaml"
        manifest = assets / expected
        engine = [
            "ros2",
            "launch",
            "genie_sim_bringup",
            "app.launch.py",
            f"scene:={scene}",
            f"launcher_config:={runtime_launcher}",
            "headless:=false",
            f"sigterm_timeout:={shutdown_timeout_s:g}",
            "sigkill_timeout:=5",
        ]
        runner = [
            interpreter,
            "-m",
            "geniesim_benchmark.newton.runner",
            "--config",
            str(runtime_evaluation),
            "--manifest",
            str(manifest),
            "--infer-host",
            infer_host,
            "--output-dir",
            str(run),
        ]
        plans.append(
            LayoutRun(
                layout_id,
                scene,
                evaluation,
                launcher,
                assets,
                manifest,
                run,
                runtime_launcher,
                runtime_evaluation,
                engine,
                runner,
            )
        )
    if layout_ids is not None:
        if not layout_ids or len(set(layout_ids)) != len(layout_ids) or set(layout_ids) - ids:
            raise ValueError("layout selection must contain unique known layout ids")
        plans = [plan for plan in plans if plan.layout_id in layout_ids]
    return plans


def _prepare(plan: LayoutRun) -> None:
    plan.run_dir.mkdir()
    (plan.run_dir / "assets").symlink_to(plan.assets_root, target_is_directory=True)
    launcher = yaml.safe_load(plan.launcher.read_text())
    params = launcher["genie_sim_engine"]["ros__parameters"]
    params["newton_video_path"] = str(plan.run_dir / "observer.mp4")
    params["isaac_render_log_path"] = str(plan.run_dir / "isaac.log")
    plan.runtime_launcher.write_text(yaml.safe_dump(launcher, sort_keys=False))
    evaluation = yaml.safe_load(plan.evaluation.read_text())
    evaluation["launcher"] = str(plan.runtime_launcher)
    plan.runtime_evaluation.write_text(yaml.safe_dump(evaluation, sort_keys=False))


def _wait_for_manifest(plan: LayoutRun, engine: subprocess.Popen, started_ns: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if engine.poll() is not None:
            raise RuntimeError(f"engine exited before fresh manifest, code {engine.returncode}")
        try:
            if plan.manifest.stat().st_mtime_ns >= started_ns:
                value = json.loads(plan.manifest.read_text())
                if Path(value["scene_yaml"]).resolve() != plan.scene:
                    raise RuntimeError("fresh manifest references another layout scene")
                shutil.copyfile(plan.manifest, plan.run_dir / "manifest.json")
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"no fresh manifest for layout {plan.layout_id}: {plan.manifest}")


def _assert_engine_idle() -> None:
    """Refuse existing engine services in this ROS domain; never stop another engine."""
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    context = Context()
    with ExitStack() as stack:
        stack.callback(context.try_shutdown)
        rclpy.init(args=[], context=context)
        node = Node(f"wuji_layout_guard_{os.getpid()}", context=context)
        stack.callback(node.destroy_node)
        executor = SingleThreadedExecutor(context=context)
        stack.callback(executor.shutdown)
        executor.add_node(node)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            existing = [
                name
                for name, _ in node.get_service_names_and_types()
                if name.startswith(("/genie_sim_engine/", "/genie_sim_engine_newton/"))
            ]
            if existing:
                raise RuntimeError(f"ROS domain already has engine services; stop that engine yourself: {existing}")
            executor.spin_once(timeout_sec=0.1)


def _process_row(pid: int) -> tuple[str, int, int, int] | None:
    """Read Linux state, parent, group and start ticks for one process."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], int(fields[1]), int(fields[2]), int(fields[19])
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
        return None


def _process_rows() -> dict[int, tuple[str, int, int, int]]:
    return {
        int(path.name): row
        for path in Path("/proc").glob("[0-9]*")
        if (row := _process_row(int(path.name))) is not None
    }


def _group_alive(pgid: int) -> bool:
    return any(state != "Z" and group == pgid for state, _, group, _ in _process_rows().values())


def _stop_group(pgid: int, timeout: float, poll: Callable, *, expected_start: int | None = None) -> dict:
    forced = False
    for signum, grace in ((signal.SIGINT, timeout), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        leader = _process_row(pgid)
        if expected_start is not None and leader is not None and leader[3] != expected_start:
            return {"forced": forced, "pid_reused": True}
        if not _group_alive(pgid):
            return {"forced": forced}
        forced = forced or signum != signal.SIGINT
        try:
            os.killpg(pgid, signum)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            poll()
            if not _group_alive(pgid):
                return {"forced": forced}
            time.sleep(0.05)
    raise RuntimeError(f"owned process group {pgid} did not exit after SIGKILL")


def stop_process_group(process: subprocess.Popen, timeout: float = 60) -> dict:
    """Close a child created with start_new_session, allowing MP4 finalization before escalation."""
    row = _process_row(process.pid)
    start = getattr(process, "_layout_start_ticks", row[3] if row is not None else None)
    result = _stop_group(process.pid, timeout, process.poll, expected_start=start)
    process.wait()
    return result


class OwnedProcessTree:
    """Track owned descendants, including Isaac's separate session, before parents can disappear."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        row = _process_row(process.pid)
        self.known = {process.pid: row[3]} if row is not None else {}
        process._layout_start_ticks = row[3] if row is not None else None
        self._done = threading.Event()
        self._capture()
        self._thread = threading.Thread(target=self._watch, name="wuji-owned-processes", daemon=True)
        self._thread.start()

    def _capture(self):
        known = dict(self.known)
        pending = list(known)
        while pending:
            pid = pending.pop()
            row = _process_row(pid)
            if row is None or row[3] != known[pid]:
                continue
            try:
                children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            for child in map(int, children):
                child_row = _process_row(child)
                if child not in known and child_row is not None and child_row[1] == pid:
                    known[child] = child_row[3]
                    pending.append(child)
        self.known = known

    def _watch(self):
        while not self._done.wait(0.03):
            self._capture()

    def close(self, timeout: float = 60) -> dict:
        detached, error, result = [], None, {"forced": False}
        try:
            try:
                result = stop_process_group(self.process, timeout)
            except Exception as failure:
                error = failure
            rows = _process_rows()
            for pid, start in dict(self.known).items():
                row = rows.get(pid)
                if pid == self.process.pid or row is None or row[3] != start or row[0] == "Z" or row[2] != pid:
                    continue
                _stop_group(pid, timeout, lambda: None, expected_start=start)
                detached.append(pid)
            if error is not None:
                raise error
            return {"forced": result["forced"] or bool(detached), "detached_cleanup": detached}
        finally:
            self._done.set()
            self._thread.join(timeout=2)


def _wait_runner(runner: subprocess.Popen, engine: subprocess.Popen) -> int:
    while True:
        if engine.poll() is not None:
            raise RuntimeError(f"engine exited while evaluating, code {engine.returncode}")
        try:
            return runner.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass


def run_batch(
    plans: list[LayoutRun],
    *,
    startup_timeout_s: float = 600,
    shutdown_timeout_s: float = 60,
    popen: Callable | None = None,
    wait_ready: Callable | None = None,
    stop: Callable | None = None,
    assert_idle: Callable | None = None,
) -> dict:
    """Execute fresh engines serially, preserving task failures separately from infrastructure errors."""
    track_processes = popen is None
    popen, wait_ready = popen or subprocess.Popen, wait_ready or _wait_for_manifest
    stop, assert_idle = stop or stop_process_group, assert_idle or _assert_engine_idle
    output = plans[0].run_dir.parent
    output.mkdir(parents=True, exist_ok=False)
    rows = [
        {"layout_id": p.layout_id, "status": "not_run", "score": None, "task_success": None, "run_dir": str(p.run_dir)}
        for p in plans
    ]
    summary = {"version": 1, "task_name": "wok_flip", "layouts": rows, "evaluated_count": 0, "task_success_count": 0}
    sessions = set()

    def save():
        summary["evaluated_count"] = sum(row["status"] == "evaluated" for row in rows)
        summary["task_success_count"] = sum(row["task_success"] is True for row in rows)
        temporary = output / ".batch_summary.json.tmp"
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        temporary.replace(output / "batch_summary.json")

    save()
    for plan, row in zip(plans, rows):
        owned, trees, cleanup_failed, interrupted = [], {}, False, False
        try:
            assert_idle()
            _prepare(plan)
            row["status"] = "running"
            save()
            with ExitStack() as files:
                engine_log = files.enter_context((plan.run_dir / "engine.log").open("w"))
                runner_log = files.enter_context((plan.run_dir / "runner.log").open("w"))
                started_ns = time.time_ns()
                engine = popen(
                    plan.engine_command,
                    cwd=plan.run_dir,
                    stdin=subprocess.PIPE,
                    stdout=engine_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                owned.append(engine)
                if track_processes:
                    trees[engine.pid] = OwnedProcessTree(engine)
                if engine.stdin is not None:
                    engine.stdin.write(b"\n")
                    engine.stdin.close()
                wait_ready(plan, engine, started_ns, startup_timeout_s)
                runner = popen(
                    plan.runner_command,
                    cwd=plan.run_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=runner_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                owned.append(runner)
                if track_processes:
                    trees[runner.pid] = OwnedProcessTree(runner)
                code = _wait_runner(runner, engine)
                row["runner_returncode"] = code
                if code != 0:
                    raise RuntimeError(f"runner exited with code {code}; see runner.log")
                results = json.loads((plan.run_dir / "results.json").read_text())
                if not isinstance(results, list) or len(results) != 1:
                    raise RuntimeError("runner must produce exactly one episode result")
                result = results[0]
                session = result["evaluation"]["session_id"]
                if not isinstance(session, str) or not session or session in sessions:
                    raise RuntimeError("evaluation reused an engine session")
                sessions.add(session)
                if type(result.get("score")) is not int or result["score"] not in (0, 1, 2):
                    raise RuntimeError("runner result has no valid observed task score")
                row.update(
                    status="evaluated",
                    score=result["score"],
                    max_score=result["max_score"],
                    task_success=result["task_success"],
                    failed=result["failed"],
                    session_id=session,
                    issued_steps=result["issued_steps"],
                    result=result,
                )
        except KeyboardInterrupt:
            interrupted = True
            row.update(status="interrupted", error="interrupted by user")
        except Exception as error:
            row.update(status="infrastructure_error", error=f"{type(error).__name__}: {error}")
        finally:
            cleanup = []
            previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            for process in reversed(owned):
                try:
                    tree = trees.get(process.pid)
                    outcome = (
                        tree.close(shutdown_timeout_s + 10) if tree is not None else stop(process, shutdown_timeout_s)
                    )
                    cleanup.append({"pid": process.pid, **outcome})
                    if outcome.get("forced"):
                        raise RuntimeError("owned process required forced shutdown; video finalization unconfirmed")
                except BaseException as error:
                    cleanup_failed = True
                    row.update(status="infrastructure_error", error=f"cleanup: {type(error).__name__}: {error}")
            signal.signal(signal.SIGINT, previous_sigint)
            row["cleanup"] = cleanup
            if row["status"] != "evaluated":
                row["score"], row["task_success"] = None, None
            save()
        print(f"layout {plan.layout_id:02d}: {row['status']} score={row['score']}", flush=True)
        if interrupted or cleanup_failed:
            break
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--infer-host", required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new artifact directory; existing directories are refused"
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=os.environ.get("GENIESIM_ASSETS_PATH"),
        help="asset root (or GENIESIM_ASSETS_PATH); no downloads are performed",
    )
    parser.add_argument("--layout-ids", help="comma-separated layout ids; default: all in bundle order")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and print commands without writing or starting anything"
    )
    parser.add_argument("--startup-timeout-s", type=float, default=600)
    parser.add_argument("--shutdown-timeout-s", type=float, default=60)
    options = parser.parse_args(argv)
    if options.assets_root is None:
        parser.error("provide --assets-root or GENIESIM_ASSETS_PATH")
    for value in (options.startup_timeout_s, options.shutdown_timeout_s):
        if not math.isfinite(value) or value <= 0:
            parser.error("timeouts must be finite and positive")
    try:
        ids = [int(value) for value in options.layout_ids.split(",")] if options.layout_ids is not None else None
        plans = load_plans(
            options.bundle,
            options.assets_root,
            options.output_dir,
            options.infer_host,
            layout_ids=ids,
            shutdown_timeout_s=options.shutdown_timeout_s,
        )
        if options.dry_run:
            for plan in plans:
                print(f"layout {plan.layout_id:02d}; cwd={plan.run_dir}; manifest={plan.manifest}")
                print(shlex.join(plan.engine_command))
                print(shlex.join(plan.runner_command))
            return 0
        result = run_batch(
            plans, startup_timeout_s=options.startup_timeout_s, shutdown_timeout_s=options.shutdown_timeout_s
        )
        if any(row["status"] == "interrupted" for row in result["layouts"]):
            return 130
        return int(any(row["status"] != "evaluated" for row in result["layouts"]))
    except (ValueError, OSError, KeyError) as error:
        print(f"run-layouts: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
