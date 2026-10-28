# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Host-side LLM task catalog, collection, and replay workflows."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid

from geniesim_cli import _env
from geniesim_cli._style import BOLD, CYAN, DIM, RED, RST


def _dc():
    from geniesim_cli.commands import data_collection

    return data_collection


def _settings(module):
    spec = importlib.util.spec_from_file_location("collection_settings", module / "scripts/collection_settings.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _validate_run_times(module, task_dir, scene, run_times):
    """Apply the collection module's metadata policy before creating a run or starting Docker."""
    spec = importlib.util.spec_from_file_location("llm_task_policy", module / "scripts/llm_task_policy.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.validate_run_times(task_dir, scene, run_times)


def _root(module, script, override=None):
    env = dict(os.environ)
    if override:
        name = _env.GENIESIM_LLM_TASK_ROOT.name if "llm" in script else _env.GENIESIM_ASSETS_PATH.name
        env[name] = str(Path(override).expanduser().resolve())
    result = subprocess.run([sys.executable, str(module / "scripts" / script)], env=env, text=True, capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.strip())
    return Path(result.stdout.strip()) if result.stdout.strip() else None


def install(args):
    parser = argparse.ArgumentParser(prog="geniesim data-collection install")
    parser.add_argument("--task-root")
    parser.add_argument("--assets-root")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(args)
    module = _dc()._dc_root()
    tasks = _root(module, "resolve_llm_task_root.py", options.task_root)
    # Assets can be supplied later; a stale explicit/saved asset path still fails.
    try:
        assets = _root(module, "resolve_assets.py", options.assets_root)
    except ValueError as exc:
        if "Cannot locate assets" not in str(exc):
            raise
        assets = None
    helper = _settings(module)
    command = [sys.executable, "-m", "pip", "install", "-e", str(module)]
    print(f"{BOLD}Install module:{RST} {shlex.join(command)}")
    print(f"{CYAN}Settings:{RST} {helper.settings_path()}")
    print(f'{CYAN}LLM task root:{RST} {tasks or "not found; configure it before running LLM tasks"}')
    if assets:
        print(f"{CYAN}Assets:{RST} {assets}")
    if options.dry_run:
        return
    result = subprocess.run(command)
    if result.returncode:
        raise SystemExit(result.returncode)
    helper.save_settings({"module_root": module, "llm_task_root": tasks, "assets_root": assets})
    print("Saved defaults; subsequent CLI commands and launchers load this file automatically.")


def config(args):
    parser = argparse.ArgumentParser(prog="geniesim data-collection config")
    parser.parse_args(args)
    module = _dc()._dc_root()
    helper = _settings(module)
    print(
        json.dumps(
            {
                "settings_file": str(helper.settings_path()),
                "saved": helper.load_settings(),
                "effective_llm_task_root": str(_root(module, "resolve_llm_task_root.py") or ""),
            },
            indent=2,
        )
    )


def _catalog(root, module):
    if root is None:
        raise ValueError("No LLM task root; run data-collection install --task-root PATH")
    catalog = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / "task_info.json").is_file():
            continue
        try:
            metadata = json.loads((folder / "task_info.json").read_text())
            robot = metadata.get("robot") or {}
            cfg = robot.get("robot_cfg", "")
            scenes = sorted(
                p.name
                for p in folder.iterdir()
                if p.is_dir() and (p / "scene.usda").is_file() and (p / "scene_info.json").is_file()
            )
            supported = bool(cfg and (module / "config/robot_cfg" / cfg).is_file())
            catalog.append(
                {
                    "name": folder.name,
                    "robot": robot.get("robot_id") or cfg,
                    "robot_config": cfg,
                    "scenes": scenes,
                    "supported": supported,
                    "ready": supported and bool(scenes) and (folder / "data_gen.py").is_file(),
                    "metadata": metadata,
                    "path": str(folder),
                }
            )
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            catalog.append(
                {
                    "name": folder.name,
                    "robot": "?",
                    "scenes": [],
                    "ready": False,
                    "error": str(exc),
                    "path": str(folder),
                }
            )
    return catalog


def _select(catalog, name):
    matches = [task for task in catalog if task["name"] == name]
    if not matches:
        matches = [task for task in catalog if name.lower() in task["name"].lower()]
    if len(matches) != 1:
        raise ValueError(f"Task {name!r}: expected one match, found {len(matches)}; use llm list")
    return matches[0]


def _writable(path):
    path.mkdir(parents=True, exist_ok=True)
    if shutil.which("setfacl"):
        subprocess.run(["setfacl", "-R", "-m", "u:1234:rwX", str(path)], check=True)
        subprocess.run(["setfacl", "-R", "-d", "-m", "u:1234:rwX", str(path)], check=True)
    else:
        # Only the new run's private files are affected by this fallback.
        for item in (path, *path.rglob("*")):
            if not item.is_symlink():
                item.chmod(0o777 if item.is_dir() else item.stat().st_mode | 0o666)


def _run_dir(module, kind, task, override):
    if override:
        path = Path(override).expanduser().resolve()
        if path.exists():
            raise ValueError(f"Output directory already exists: {path}; use a new directory")
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    return module / "recording_data" / kind / task / stamp


def _docker(module, assets, run, options, tasks=False):
    name = options.container_name or _dc()._DC_CONTAINER
    image = options.image or os.environ.get("DATA_COLLECTION_IMAGE") or _dc()._DC_IMAGE
    cache = module / "recording_data" / ".isaac_cache"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--gpus",
        "all",
        "--shm-size=2g",
        "-e",
        "SIM_ASSETS=/sim_assets",
        "-v",
        f"{assets}:/sim_assets:ro",
        "-v",
        f"{run}:/run:rw",
        "-v",
        f"{cache}:/isaac-sim/.cache:rw",
    ]
    if tasks:
        command += ["-e", "LLM_TASK_ROOT=/run/tasks", "-e", "GENIESIM_LLM_TASK_ROOT=/run/tasks"]
    if not options.headless:
        display = os.environ.get("DISPLAY")
        if not display:
            raise ValueError("GUI mode requires DISPLAY; otherwise use the default headless mode")
        command += ["-e", f"DISPLAY={display}", "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw"]
    command += [image, "/isaac-sim/python.sh"]
    return command, cache


def _common(parser):
    parser.add_argument("--output-dir")
    parser.add_argument("--assets-root")
    parser.add_argument("--image")
    parser.add_argument("--container-name")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")


def _execute(command, run, cache, manifest, dry_run):
    print(f"{BOLD}Run directory:{RST} {run}", flush=True)
    print(f"{DIM}{shlex.join(command)}{RST}", flush=True)
    if dry_run:
        return None
    _dc()._ensure_docker()
    if _dc()._container_state(command[command.index("--name") + 1]) is not None:
        raise ValueError("Container name already exists; choose --container-name or close that container")
    _writable(run)
    if not cache.exists():
        _writable(cache)
    manifest["command"] = command
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f'{CYAN}Log:{RST} {run / "run.log"}', flush=True)
    with (run / "run.log").open("wb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    manifest["exit_code"] = result.returncode
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result.returncode


def llm(args, root_run=False):
    parser = argparse.ArgumentParser(prog="geniesim data-collection" if root_run else "geniesim data-collection llm")
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("list", "show"):
        p = commands.add_parser(action)
        p.add_argument("--task-root")
        p.add_argument("--json", action="store_true")
        if action == "list":
            p.add_argument("--robot")
        else:
            p.add_argument("task")
    p = commands.add_parser(
        "run",
        description="Collect an LLM task from the configured task library.",
        epilog="JSON jobs: geniesim data-collection run --task-json PATH [options]",
    )
    p.add_argument("task")
    p.add_argument("--task-root")
    p.add_argument("--scene", default="random")
    p.add_argument("--run-times", type=int, default=1)
    p.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--render", action="store_true")
    p.add_argument("--server-start-timeout", type=int, default=420)
    p.add_argument("--episode-timeout", type=int, default=600)
    p.add_argument("--task-result-timeout", type=int, default=600)
    _common(p)
    options = parser.parse_args(args)
    module = _dc()._dc_root()
    root = _root(module, "resolve_llm_task_root.py", options.task_root)
    catalog = _catalog(root, module)
    if options.action == "list":
        tasks = [t for t in catalog if not options.robot or options.robot.upper() in t["robot"].upper()]
        if options.json:
            print(json.dumps({"task_root": str(root), "tasks": tasks}, ensure_ascii=False, indent=2))
        else:
            print(f"{CYAN}Task root:{RST} {root}\n{len(tasks)} LLM tasks")
            for task in tasks:
                print(
                    f'  {task["name"]:<52} {task["robot"]:<10} {len(task["scenes"]):>3} scenes  '
                    f'{"ready" if task["ready"] else "unavailable"}'
                )
        return
    task = _select(catalog, options.task)
    if options.action == "show":
        print(json.dumps(task, ensure_ascii=False, indent=2))
        return
    if not task["ready"]:
        raise ValueError(f'Task is not runnable (robot configuration/code/scenes): {task["name"]}')
    if options.run_times < 1:
        parser.error("--run-times must be positive")
    if options.scene != "random" and options.scene not in task["scenes"]:
        parser.error(f'Scene not found: {options.scene}; use llm show {task["name"]}')
    _validate_run_times(module, task["path"], options.scene, options.run_times)
    assets = _root(module, "resolve_assets.py", options.assets_root)
    run = _run_dir(module, "llm", task["name"], options.output_dir)
    if run.is_relative_to(root):
        raise ValueError("Output directory must be outside the task library")
    command, cache = _docker(module, assets, run, options, tasks=True)
    image_index = command.index("/isaac-sim/python.sh") - 1
    command[image_index:image_index] = ["-v", f'{run / "episodes"}:/workspace/recording_data:rw']
    command += [
        "scripts/collect_llm_task.py",
        "--task",
        task["name"],
        "--scene",
        options.scene,
        "--run-times",
        str(options.run_times),
        "--report-dir",
        "/run/report",
        "--server-start-timeout",
        str(options.server_start_timeout),
        "--episode-timeout",
        str(options.episode_timeout),
        "--task-result-timeout",
        str(options.task_result_timeout),
    ]
    if not options.record:
        command.append("--no-record")
    if not options.headless:
        command.append("--no-headless")
    if options.render:
        command.append("--render")
    if not options.dry_run:
        run.mkdir(parents=True)
        shutil.copytree(task["path"], run / "tasks" / task["name"])
        (run / "episodes").mkdir()
    manifest = {
        "kind": "llm_collection",
        "task": task["name"],
        "task_root": str(root),
        "scene": options.scene,
        "run_times": options.run_times,
        "record": options.record,
        "assets_root": str(assets),
        "run_dir": str(run),
    }
    rc = _execute(command, run, cache, manifest, options.dry_run)
    if rc is None:
        return
    report_path = run / "report/collection_result.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {"success": False}
    episodes = []
    for entry in report.get("episodes", []):
        path = entry.get("recording_dir")
        if path:
            host = run / "episodes" / Path(path).name
            episodes.append(
                {
                    "path": str(host),
                    "scene": entry.get("scene"),
                    "success": entry.get("success"),
                    "frames": entry.get("frames"),
                    "error": entry.get("error"),
                }
            )
    summary = {
        "task": task["name"],
        "success": rc == 0 and report.get("success") is True,
        "record": options.record,
        "episodes": episodes,
        "exit_code": rc,
    }
    (run / "run_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["success"] and options.record:
        index = module / "recording_data/llm" / task["name"] / "latest.json"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps({"result_path": str(run / "run_result.json")}) + "\n")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["success"] else rc or 1)


def _episode(module, source):
    literal = Path(source).expanduser()
    if literal.is_dir():
        return literal.resolve()
    if Path(source).name != source or source in (".", ".."):
        raise ValueError(f"Episode directory not found: {source}")
    folder = module / "recording_data/llm" / source
    candidates = list(folder.glob("*/run_result.json"))
    if (folder / "latest.json").is_file():
        indexed = Path(json.loads((folder / "latest.json").read_text())["result_path"])
        if indexed.is_file() and indexed not in candidates:
            candidates.append(indexed)
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        result = json.loads(path.read_text())
        if result.get("success") and result.get("record"):
            for episode in reversed(result.get("episodes", [])):
                if episode.get("success") and Path(episode["path"]).is_dir():
                    return Path(episode["path"])
    raise ValueError(f"No successful recorded episode for {source}; pass an explicit episode directory")


def _stage_episode(source, target):
    shutil.copytree(source, target)
    state = json.loads((target / "state.json").read_text())
    raw = state.get("scene", {}).get("scene_usd", "")
    basename = Path(raw).name
    matches = sorted(target.rglob(basename)) if basename else []
    relative = Path(raw)
    local = target / relative if not relative.is_absolute() else None
    if local is not None and local.is_file():
        scene = local
    elif len(matches) == 1:
        scene = matches[0]
    elif relative.is_absolute() and relative.is_file():
        scene = target / "scene" / basename
        shutil.copytree(relative.parent, scene.parent)
    else:
        raise ValueError(f"Cannot isolate scene {raw!r}; include the scene directory in the episode")
    state["scene"]["scene_usd"] = scene.relative_to(target).as_posix()
    (target / "state.json").write_text(json.dumps(state, indent=2) + "\n")


def replay(args):
    parser = argparse.ArgumentParser(prog="geniesim data-collection replay")
    parser.add_argument("episode", help="Episode directory or an LLM task name with a successful CLI recording")
    parser.add_argument("--config")
    parser.add_argument("--preset", choices=("data_collection", "teleop"))
    parser.add_argument("--max-concurrent-envs", type=int)
    parser.add_argument("--num-rounds", type=int)
    parser.add_argument("--render-hz", type=int)
    parser.add_argument("--depth", choices=("NONE", "ALL"))
    parser.add_argument("--no-fuzz", action="store_true")
    _common(parser)
    options = parser.parse_args(args)
    module = _dc()._dc_root()
    source = _episode(module, options.episode)
    if not (source / "state.json").is_file() or not (source / "metadata.yaml").is_file():
        raise ValueError("Replay requires state.json and metadata.yaml in the episode directory")
    top = json.loads(Path(options.config).expanduser().read_text()) if options.config else {}
    if not isinstance(top, dict):
        parser.error("Replay config must be a JSON object")
    preset = options.preset or top.get("preset", "data_collection")
    if preset not in ("data_collection", "teleop"):
        parser.error("Only data_collection and teleop presets are supported")
    top["preset"] = preset
    if options.preset or "sub_mode" not in top:
        top["sub_mode"] = preset == "teleop"
    cfg = top.setdefault("replay_config", {})
    fuzz = top.setdefault("fuzzing_config", {})
    if not isinstance(cfg, dict) or not isinstance(fuzz, dict):
        parser.error("replay_config and fuzzing_config must be JSON objects")
    for key, value in [
        ("max_concurrent_envs", options.max_concurrent_envs),
        ("num_rounds", options.num_rounds),
        ("render_hz", options.render_hz),
    ]:
        if value is not None:
            if value < 1:
                parser.error(f'--{key.replace("_", "-")} must be positive')
            cfg[key] = value
    if options.num_rounds is not None:
        fuzz.pop("num_rounds", None)
    if options.depth:
        cfg["record_depth"] = options.depth
    if options.no_fuzz:
        fuzz.update({key: False for key in ("enable_light_fuzz", "enable_background_fuzz", "enable_material_fuzz")})
    preset_cfg = json.loads((module / "replay/preset_configs" / f"{preset}.json").read_text())
    effective = {**preset_cfg.get("replay_config", {}), **cfg}
    effective_fuzz = {**preset_cfg.get("fuzzing_config", {}), **fuzz}
    rounds = effective_fuzz.get("num_rounds")
    if rounds is None:
        rounds = effective.get("num_rounds", 1)
    envs = effective.get("max_concurrent_envs", 3)
    if type(rounds) is not int or rounds < 1 or type(envs) is not int or envs < 1:
        parser.error("CLI replay needs positive finite num_rounds and max_concurrent_envs")
    if effective.get("enable_recording") is False:
        parser.error("CLI replay output verification requires enable_recording=true")
    assets = _root(module, "resolve_assets.py", options.assets_root)
    run = _run_dir(module, "replay_runs", source.name, options.output_dir)
    if run.is_relative_to(source):
        raise ValueError("Replay output must be outside its source episode")
    command, cache = _docker(module, assets, run, options)
    command += [
        "scripts/replay_episode.py",
        "--input-dir",
        "/run/input",
        "--config",
        "/run/config.json",
        "--output-dir",
        "/run/output",
        "--expected-episodes",
        str(envs * rounds),
    ]
    if not options.headless:
        command.append("--no-headless")
    if not options.dry_run:
        run.mkdir(parents=True)
        _stage_episode(source, run / "input")
        (run / "config.json").write_text(json.dumps(top, indent=2) + "\n")
    manifest = {
        "kind": "replay",
        "source_episode": str(source),
        "assets_root": str(assets),
        "config": top,
        "expected_episodes": envs * rounds,
        "run_dir": str(run),
    }
    rc = _execute(command, run, cache, manifest, options.dry_run)
    if rc is not None:
        print(f'{CYAN}Replay output:{RST} {run / "output"}\nVerification: {run / "output_check.json"}')
        raise SystemExit(rc)


def run(action, args):
    try:
        {
            "install": install,
            "config": config,
            "llm": llm,
            "replay": replay,
            "run": lambda values: llm(["run", *values], root_run=True),
        }[action](args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"{RED}Error: {exc}{RST}", file=sys.stderr)
        raise SystemExit(1)
