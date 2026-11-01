# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Start normal ROS bringup, retain its artifacts and close every owned process."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml


BRINGUP = Path("source/geniesim_ros/src/ros_ws/src/genie_sim_bringup")
ENGINES = {"newton": "genie_sim_engine_newton"}


def resolve_bringup_file(value: str, config_path: Path) -> Path:
    """Prefer an explicit task-relative file or its checkout, then ROS share."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=True)
    local = config_path.parent / path
    if local.is_file():
        return local.resolve()
    for parent in config_path.parents:
        candidate = parent / BRINGUP / "config" / path
        if candidate.is_file():
            return candidate.resolve()
    from ament_index_python.packages import get_package_share_directory

    return (Path(get_package_share_directory("genie_sim_bringup")) / "config" / path).resolve(strict=True)


def resolve_assets(config_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("--assets-root must name an asset directory")
        return root
    # Use exactly bringup's asset lookup, rather than adding another environment
    # variable or assuming the benchmark package owns the RT Engine assets.
    helper = resolve_bringup_file("../launch/utils.py", config_path)
    spec = importlib.util.spec_from_file_location("benchmark_bringup_assets", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return Path(module.resolve_assets_folder()).resolve(strict=True)


@dataclass(frozen=True)
class LaunchPlan:
    runtime: str
    scene: Path
    launcher: Path
    assets: Path
    output: Path
    manifest: Path
    runtime_launcher: Path
    runtime_config: Path
    command: tuple[str, ...]
    ready_timeout_s: float

    def as_dict(self):
        return {key: str(value) if isinstance(value, Path) else value for key, value in vars(self).items()}


def plan_launch(config_path: Path, config: dict, output: Path, assets_root: Path | None = None) -> LaunchPlan:
    runtime = config["runtime"]
    if runtime not in ENGINES:
        raise ValueError(f"unsupported RT Engine runtime: {runtime}")
    scene = resolve_bringup_file(config["scene"], config_path)
    launcher = resolve_bringup_file(config["launcher"], config_path)
    launch_doc = yaml.safe_load(launcher.read_text())
    physics = launch_doc["launcher"]["physics"]
    selected = physics["engines"][physics["engine"]]
    if selected["package"] != ENGINES[runtime]:
        raise ValueError("task runtime and launcher physics package disagree")
    scene_doc = yaml.safe_load(scene.read_text())
    if scene_doc["robot"]["robot_name"] != "G2_crsB_wuji":
        raise ValueError("Wuji tasks require a G2_crsB_wuji scene")
    output = output.expanduser().resolve()
    assets = resolve_assets(config_path, assets_root)
    runtime_launcher = output / "launcher.yaml"
    command = (
        sys.executable,
        shutil.which("ros2") or "ros2",
        "launch",
        "genie_sim_bringup",
        "app.launch.py",
        f"scene:={scene}",
        f"launcher_config:={runtime_launcher}",
        f"headless:={str(config.get('headless', False)).lower()}",
        "rviz:=false",
    )
    return LaunchPlan(
        runtime,
        scene,
        launcher,
        assets,
        output,
        assets / "scenes" / scene.stem / "manifest.json",
        runtime_launcher,
        output / "evaluation.yaml",
        command,
        float(config["ready_timeout_s"]),
    )


def _wait_manifest(plan: LaunchPlan, process, started_ns: int):
    deadline = time.monotonic() + plan.ready_timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"bringup exited with code {process.returncode}; see {plan.output / 'engine.log'}")
        try:
            if plan.manifest.stat().st_mtime_ns >= started_ns:
                document = json.loads(plan.manifest.read_text())
                base = Path(document.get("base_path") or plan.manifest.parent)
                if not base.is_absolute():
                    base = plan.manifest.parent / base
                scene = Path(document["scene_yaml"])
                if (scene if scene.is_absolute() else base / scene).resolve() != plan.scene:
                    raise RuntimeError("fresh bringup manifest selects a different scene")
                shutil.copyfile(plan.manifest, plan.output / "manifest.json")
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    raise TimeoutError(f"no fresh assembled manifest; see {plan.output / 'engine.log'}")


@contextmanager
def launched_scene(plan: LaunchPlan, config: dict):
    """Own bringup and its render descendants, leaving attached sessions alone."""
    from geniesim_benchmark.newton.layout_batch import OwnedProcessTree

    if plan.output.exists() and any(plan.output.iterdir()):
        raise FileExistsError("choose a fresh --output-dir; existing run artifacts are never overwritten")
    plan.output.mkdir(parents=True, exist_ok=True)
    (plan.output / "assets").symlink_to(plan.assets, target_is_directory=True)
    launcher = yaml.safe_load(plan.launcher.read_text())
    physics = launcher["launcher"]["physics"]
    params = launcher[physics["engines"][physics["engine"]].get("name", "genie_sim_engine")]["ros__parameters"]
    if plan.runtime == "newton":
        params["newton_video_path"] = str(plan.output / "observer.mp4")
        params["isaac_render_log_path"] = str(plan.output / "isaac.log")
    plan.runtime_launcher.write_text(yaml.safe_dump(launcher, sort_keys=False))
    evaluation = {**config, "launcher": str(plan.runtime_launcher)}
    plan.runtime_config.write_text(yaml.safe_dump(evaluation, sort_keys=False, allow_unicode=True))
    (plan.output / "launch-plan.json").write_text(json.dumps(plan.as_dict(), indent=2) + "\n")
    with (plan.output / "engine.log").open("w") as log:
        started_ns = time.time_ns()
        process = subprocess.Popen(
            plan.command, cwd=plan.output, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        owner = OwnedProcessTree(process)
        try:
            _wait_manifest(plan, process, started_ns)
            yield plan.runtime_config, plan.manifest
        finally:
            previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                receipt = owner.close(timeout=60)
            finally:
                signal.signal(signal.SIGINT, previous_sigint)
            (plan.output / "shutdown.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if receipt["forced"]:
                raise RuntimeError("bringup needed forced shutdown; inspect shutdown.json and engine.log")


def run_managed(options, config_path: Path, config: dict, attached_main) -> int:
    plan = plan_launch(config_path, config, options.output_dir, options.assets_root)
    if options.dry_run:
        print(json.dumps(plan.as_dict(), indent=2))
        return 0
    print(f"Benchmark output: {plan.output}", flush=True)
    with launched_scene(plan, config) as (evaluation, manifest):
        args = [
            "--config",
            str(evaluation),
            "--manifest",
            str(manifest),
            "--infer-host",
            options.infer_host,
            "--output-dir",
            str(plan.output),
        ]
        for name in ("max_steps", "episodes"):
            value = getattr(options, name)
            if value is not None:
                args += ["--" + name.replace("_", "-"), str(value)]
        return attached_main(args)
