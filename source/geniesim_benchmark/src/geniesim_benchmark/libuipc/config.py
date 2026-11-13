# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Evaluation YAML resolution and path-only episode overlays."""

from __future__ import annotations
import copy
from importlib import resources
import json
import math
from pathlib import Path
import uuid

import yaml

TASKS = ("cloth_hanger", "fold_cloth", "drawstring_bag")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def discover_repo_root(value=None, *, config_path=None):
    if value is not None:
        candidates = [Path(value).expanduser().resolve(strict=True)]
    else:
        candidates = list(Path(config_path or __file__).resolve().parents)
    for root in candidates:
        if (root / "source/geniesim_benchmark/pyproject.toml").is_file() and (
            root / "source/geniesim_ros/src/ros_ws/src/genie_sim_bringup"
        ).is_dir():
            return root
    raise ValueError("a configured source checkout is required; pass --repo-root for a wheel installation")


def validate_settings(config):
    from .materials import validate_material_settings

    validate_material_settings(config)
    if config.get("runtime") != "libuipc" or not isinstance(config.get("libuipc"), dict):
        raise ValueError("Wuji evaluation requires explicit runtime: libuipc")
    settings = config["libuipc"]
    benchmark = config.get("benchmark", {})
    for key, default in (("record", False), ("record_state", True), ("resume", False)):
        if type(benchmark.get(key, default)) is not bool:
            raise ValueError(f"benchmark.{key} must be a boolean")
    resume_id = benchmark.get("resume_id", "")
    if not isinstance(resume_id, str) or resume_id in (".", "..") or "/" in resume_id or "\\" in resume_id:
        raise ValueError("benchmark.resume_id must be a local job/model identity")
    if settings.get("task") not in TASKS:
        raise ValueError("unsupported Wuji IPC task")
    for key, minimum in (("max_steps", 1), ("action_chunk_steps", 0)):
        if type(settings.get(key)) is not int or settings[key] < minimum:
            raise ValueError(f"libuipc.{key} must be an integer >= {minimum}")
    for key in ("fps", "wait_timeout", "infer_timeout", "scoring_drain_timeout_s"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"libuipc.{key} must be finite and positive")
    if type(config.get("benchmark", {}).get("num_episode")) is not int or config["benchmark"]["num_episode"] < 1:
        raise ValueError("benchmark.num_episode must be a positive integer")
    count = config["benchmark"].get("num_instances", 1)
    if type(count) is not int or count < 1:
        raise ValueError("benchmark.num_instances must be a positive integer")
    material = config.get("generalization", {}).get("material", {})
    offsets = settings.get("instance_base_xy_offsets")
    if offsets is None and material.get("selection") == "enumerate" and count > len(material["allowed_materials"]):
        raise ValueError("num_instances exceeds the material bank")
    if offsets is not None and (
        material.get("selection") != "enumerate"
        or not isinstance(offsets, list)
        or len(offsets) != count
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in pair)
            for pair in offsets
        )
    ):
        raise ValueError("libuipc.instance_base_xy_offsets requires one finite XY pair per enumerated instance")
    if offsets is not None and len({tuple(pair) for pair in offsets}) != count:
        raise ValueError("libuipc.instance_base_xy_offsets must be distinct")
    if settings.get("scoring_execution") not in ("offline", "streaming"):
        raise ValueError("scoring_execution must be offline or streaming")
    if (
        type(settings.get("stop_on_success")) is not bool
        or settings["stop_on_success"]
        and settings["scoring_execution"] != "streaming"
    ):
        raise ValueError("stop_on_success requires streaming scoring")
    if settings.get("scoring_framework") != "ader":
        raise ValueError("Wuji evaluation requires native ADER scoring")
    if type(config.get("app", {}).get("headless", True)) is not bool:
        raise ValueError("app.headless must be a boolean")


def _bringup_path(value, root):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / "source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/config" / path
    return path.resolve(strict=True)


def load_config(path, *, repo_root=None):
    candidate = Path(path).expanduser()
    if not candidate.is_file() and len(candidate.parts) == 1:
        name = candidate.name if candidate.suffix else candidate.name + ".yaml"
        candidate = Path(str(resources.files("geniesim_benchmark").joinpath("config", name)))
    path = candidate.resolve(strict=True)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("benchmark configuration must be a mapping")
    validate_settings(config)
    config = copy.deepcopy(config)
    root = discover_repo_root(repo_root, config_path=path)
    settings = config["libuipc"]
    settings["cameras"] = str((path.parent / settings["cameras"]).resolve(strict=True))
    for key in ("evaluation_scene", "launcher"):
        settings[key] = str(_bringup_path(settings[key], root))
    # All physical rates come from these two YAML files. This only validates the
    # action request duration; there is no Python/CLI physics override.
    from genie_sim_client.ipc import IPCClientConfig

    protocol = IPCClientConfig.from_files(settings["evaluation_scene"], settings["launcher"])
    ticks = protocol.physics_hz / settings["fps"]
    if ticks < 1 or not math.isclose(ticks, round(ticks), rel_tol=0, abs_tol=1e-7):
        raise ValueError("fps must produce an integer number of YAML IPC steps per action")
    config["repo_root"] = str(root)
    config["config_path"] = str(path)
    return config


def materialize_scene(config, episode_dir, *, assets_root, episode_index=0, episode_id=None):
    settings = config["libuipc"]
    directory = Path(episode_dir).resolve(strict=True)
    scene = yaml.safe_load(Path(settings["evaluation_scene"]).read_text())
    if "newton" in scene:
        raise ValueError("target evaluation scene must use its independent libuipc definition")
    if "state_recording" in scene["libuipc"]:
        raise ValueError("target scene may not supply a shared recording output")
    offsets = settings.get("instance_base_xy_offsets")
    if offsets is not None:
        x, y = offsets[episode_index]
        base = scene["robot"]["init_base_pose"]
        base["x"] += x
        base["y"] += y
    scene["cameras"] = yaml.safe_load(Path(settings["cameras"]).read_text())["cameras"]
    scene["libuipc"]["state_recording"] = {"enabled": True, "output_dir": str(directory / "physical_state")}
    from .materials import create_material_plan

    plan = create_material_plan(
        config, scene, assets_root, episode_index=episode_index, episode_id=episode_id or uuid.uuid4().hex
    )
    write_json(directory / "material_generalization.json", plan)
    scene["generalization"] = {
        "material": {
            "episode_plan": plan,
            "receipt_path": str(directory / "material_generalization_applied.json"),
        }
    }
    effective = directory / ("scene_" + uuid.uuid4().hex + ".yaml")
    effective.write_text(yaml.safe_dump(scene, sort_keys=False))
    return effective


def resolve_assets_root(repo_root, assets_root=None):
    """Use the bringup mount priority and reject a shadowed explicit root."""
    candidates = (Path(repo_root) / "assets", Path("/geniesim_assets"))
    root = next((path for path in candidates if path.is_dir()), None)
    if root is None:
        from geniesim_assets import ASSETS_PATH

        root = Path(ASSETS_PATH)
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("selected assets root must be a directory")
    if assets_root is not None and Path(assets_root).expanduser().resolve(strict=True) != root:
        raise ValueError(f"bringup assets root {root} differs from --assets-root {assets_root}")
    return root


def materialize_sequence(config, run_dir, *, assets_root):
    """Preload every episode plan into one YAML; physics/topology must be identical."""
    root = Path(run_dir)
    count = config["benchmark"].get("num_instances", 1)
    attempts = config["benchmark"]["num_episode"]
    enumerated = config.get("generalization", {}).get("material", {}).get("selection") == "enumerate"
    entries, episodes, baseline = [], [], None
    for index in range(count * attempts):
        directory = root / f"episode_{index + 1:03d}"
        directory.mkdir(exist_ok=False)
        token = uuid.uuid4().hex
        plan_index = index // attempts if enumerated else index
        scene_path = materialize_scene(
            config, directory, assets_root=assets_root, episode_index=plan_index, episode_id=token
        )
        scene = yaml.safe_load(scene_path.read_text())
        entry = {
            "state_recording": scene["libuipc"]["state_recording"],
            "material": scene["generalization"]["material"],
        }
        physical = copy.deepcopy(scene)
        del physical["libuipc"]["state_recording"]
        del physical["generalization"]
        pose = physical.get("robot", {}).get("init_base_pose")
        if pose is not None:
            # This is an explicit preloaded pose plan, not an ignored physical
            # difference. The engine reconstructs native state for changed XY.
            if not isinstance(pose, dict) or not {"x", "y"} <= set(pose):
                raise ValueError("sequence robot base pose requires x/y fields")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in pose.values()):
                raise ValueError("sequence robot base pose must be finite")
            entry["robot_base_pose"] = copy.deepcopy(pose)
            del pose["x"], pose["y"]
        if baseline is not None and physical != baseline:
            raise ValueError("libuipc in-process layouts require identical physical geometry and solver YAML")
        if baseline is None:
            baseline, combined = physical, scene
        entries.append(entry)
        episodes.append(
            dict(
                directory=directory,
                token=token,
                plan_index=plan_index,
                instance_id=index // attempts,
                attempt_index=index % attempts,
            )
        )
    combined["libuipc"]["episode_sequence"] = entries
    scene_path = root / ("scene_sequence_" + uuid.uuid4().hex + ".yaml")
    scene_path.write_text(yaml.safe_dump(combined, sort_keys=False))
    return scene_path, episodes
