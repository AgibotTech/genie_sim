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
    if material.get("selection") == "enumerate" and count > len(material["allowed_materials"]):
        raise ValueError("num_instances exceeds the material bank")
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
    scene["cameras"] = yaml.safe_load(Path(settings["cameras"]).read_text())["cameras"]
    scene["libuipc"]["state_recording"] = {"enabled": True, "output_dir": str(directory / "physical_state")}
    from .materials import create_material_plan

    plan = create_material_plan(config, scene, assets_root, episode_index=episode_index,
                                episode_id=episode_id or uuid.uuid4().hex)
    write_json(directory / "material_generalization.json", plan)
    scene["generalization"] = {"material": {
        "episode_plan": plan,
        "receipt_path": str(directory / "material_generalization_applied.json"),
    }}
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
