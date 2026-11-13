# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Strict configuration for the native MuJoCo Rubik benchmark."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math
import os
import yaml

from .cameras import policy_cameras, projection_parameters


@dataclass
class AppConfig:
    headless: bool = True


@dataclass
class BenchmarkConfig:
    backend: str = "mujoco"
    python: str = ""
    task_name: str = "rubik_cube_solve_1turn"
    platform: str = "g2_crsb_wuji"
    infer_host: str = ""
    policy: str = "websocket"
    num_episode: int = 1
    num_instances: int = 1
    seed: int = 0
    output_dir: str = ""
    record: bool = False
    record_state: bool = True
    record_video_dir: str = ""
    fps: int = 30
    resume: bool = False
    resume_id: str = ""
    preview: bool = False


@dataclass
class RubikConfig:
    package_root: str = ""
    prompt: str = "Solve the Rubik's Cube"
    segments: list[str] = field(default_factory=list)
    scramble: str = ""
    timeout_seconds: float = 60.0
    control_hz: float = 30.0
    action_time_scale: float = 2.0
    hold_seconds: float = 0.0
    alignment_degrees: float = 5.0
    pickup_clearance: float = 0.02
    drop_distance: float = 0.04
    minimum_support_fingers: int = 2
    detent_stiffness: float = 0.02
    center_detent_stiffness: float = 0.02
    guide_stiffness: float = 0.2
    internal_friction: float = 0.05
    cube_friction: float = 0.8
    hand_friction_scale: float = 1.25
    inference_timeout_seconds: float = 30.0
    max_chunk_size: int = 256
    hand_target_tolerance: float = 0.0
    hand_controller: str = "native"
    terminate_on_robot_collision: bool = True
    render: bool = True
    renderer: str = field(default_factory=lambda: os.environ.get("RUBIK_RENDERER", "mujoco"))
    isaac_python: str = ""
    scene_archive: str = field(default_factory=lambda: os.environ.get("RUBIK_SCENE_ARCHIVE", ""))
    bundled_scene_archive: str = ""
    cameras: dict = field(default_factory=policy_cameras)


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    rubik: RubikConfig = field(default_factory=RubikConfig)


def load_config(path, overrides=()):
    """Apply dotted CLI overrides and reject unused or ambiguous settings."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")
    for arg in overrides:
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(f"Expected --section.key=value: {arg}")
        key, value = arg[2:].split("=", 1)
        section, separator, name = key.partition(".")
        if not separator or "." in name or section not in ("app", "benchmark", "rubik"):
            raise ValueError(f"Unsupported MuJoCo override: {key}")
        if not isinstance(raw.setdefault(section, {}), dict):
            raise ValueError(f"{section} must be a mapping")
        raw[section][name] = yaml.safe_load(value)
    unknown = set(raw) - {"app", "benchmark", "rubik"}
    if unknown:
        raise ValueError(f"Unsupported config sections: {sorted(unknown)}")
    cfg = Config()
    for section in ("app", "benchmark", "rubik"):
        values = raw.get(section, {})
        target = getattr(cfg, section)
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a mapping")
        extras = set(values) - {f.name for f in fields(target)}
        if extras:
            raise ValueError(f"Unknown {section} keys: {sorted(extras)}")
        for key, value in values.items():
            expected = type(getattr(target, key))
            if expected is float:
                valid = type(value) in (int, float) and math.isfinite(value)
                value = float(value) if valid else value
            else:
                valid = type(value) is expected
            if not valid:
                raise ValueError(f"Invalid type/value for {section}.{key}")
            setattr(target, key, value)
    b, r = cfg.benchmark, cfg.rubik
    if (
        b.backend != "mujoco"
        or b.task_name
        not in ("rubik_cube_solve", "rubik_cube_solve_1turn", "rubik_cube_solve_2turn", "rubik_cube_solve_3turn")
        or b.platform != "g2_crsb_wuji"
    ):
        raise ValueError("This backend supports Rubik scene tasks on g2_crsb_wuji")
    if not r.prompt.strip():
        raise ValueError("rubik.prompt must be nonempty")
    if any(not isinstance(segment, str) or not segment.strip() for segment in r.segments):
        raise ValueError("rubik.segments must contain nonempty strings")
    if b.fps < 1:
        raise ValueError("benchmark.fps must be positive")
    if b.num_episode < 1 or b.num_instances < 1 or b.seed < 0:
        raise ValueError("Require positive num_episode/num_instances and nonnegative seed")
    if b.resume_id in (".", "..") or "/" in b.resume_id or "\\" in b.resume_id:
        raise ValueError("benchmark.resume_id must be a local job/model identity")
    if b.policy not in ("websocket", "hold"):
        raise ValueError("policy must be websocket or hold (interface smoke test)")
    r.package_root = r.package_root or os.environ.get("RUBIK_PACKAGE_ROOT", "")
    if not r.package_root:
        from .isaac_render import assets_directory

        try:
            r.package_root = str(assets_directory() / "mujoco" / "rubik_cube")
        except RuntimeError as error:
            raise ValueError(str(error)) from error
    if r.bundled_scene_archive:
        name = Path(r.bundled_scene_archive)
        if name.name != r.bundled_scene_archive or name.suffix != ".zip":
            raise ValueError("rubik.bundled_scene_archive must be a catalog ZIP filename")
        if raw["rubik"].get("scene_archive"):
            raise ValueError("Choose either rubik.scene_archive or rubik.bundled_scene_archive")
        r.scene_archive = str(Path(__file__).parent / "scene_catalogs" / name)
    if not r.scene_archive and b.task_name == "rubik_cube_solve_2turn":
        r.scene_archive = str(Path(__file__).parent / "scene_catalogs/rubik_2turn_5_public.zip")
    if not r.scene_archive and r.renderer == "isaac":
        candidate = Path(r.package_root) / "rubik_scenes_only_10_20261012.zip"
        if candidate.is_file():
            r.scene_archive = str(candidate)
    if b.task_name != "rubik_cube_solve_1turn" and not r.scene_archive:
        raise ValueError("Multi-turn tasks require the supplied scene archive")
    if r.renderer == "isaac" and not r.scene_archive:
        raise ValueError("Isaac rendering requires the supplied scene archive")
    if r.scene_archive and r.scramble:
        raise ValueError("Scene initial state and scramble override cannot both be selected")
    if b.policy == "websocket" and not b.infer_host and not b.preview:
        raise ValueError("Websocket policy requires --infer-host=HOST:PORT")
    for name in (
        "timeout_seconds",
        "control_hz",
        "drop_distance",
        "internal_friction",
        "cube_friction",
        "inference_timeout_seconds",
    ):
        if getattr(r, name) <= 0:
            raise ValueError(f"rubik.{name} must be positive")
    if r.action_time_scale < 1:
        raise ValueError("rubik.action_time_scale must be at least 1")
    if r.control_hz > 1000 or not 0 <= r.hold_seconds <= r.timeout_seconds or not 0 <= r.alignment_degrees < 45:
        raise ValueError("Require control_hz<=1000, 0<=hold_seconds<=timeout and 0<=alignment_degrees<45")
    if r.pickup_clearance <= 0 or r.minimum_support_fingers < 2 or r.minimum_support_fingers > 10:
        raise ValueError("Require positive pickup_clearance and 2..10 supporting fingers")
    if not 1 <= r.hand_friction_scale <= 2 or not 1 <= r.max_chunk_size <= 4096:
        raise ValueError("Invalid friction scale or chunk size")
    if r.renderer not in ("mujoco", "isaac"):
        raise ValueError("renderer must be mujoco or isaac")
    if r.hand_controller not in ("natural", "native"):
        raise ValueError("hand_controller must be natural or native")
    if not 0 <= r.hand_target_tolerance <= 0.05:
        raise ValueError("Require hand_target_tolerance in [0,0.05] radians")
    if min(r.detent_stiffness, r.center_detent_stiffness, r.guide_stiffness) < 0:
        raise ValueError("Passive stiffness must be nonnegative")
    if r.scramble and r.scramble not in [f + s for f in "URFDLB" for s in ("", "'")]:
        raise ValueError("One-turn benchmark requires exactly one quarter-turn scramble")
    if not r.render and (b.policy != "hold" or b.preview):
        raise ValueError("Only hold smoke tests may disable rendering")
    if r.render and set(r.cameras) != {"head", "hand_left", "hand_right"}:
        raise ValueError("Configure exactly the three policy cameras")
    for camera in r.cameras.values():
        projection_parameters(camera)
    return cfg


def config_dict(cfg):
    return asdict(cfg)
