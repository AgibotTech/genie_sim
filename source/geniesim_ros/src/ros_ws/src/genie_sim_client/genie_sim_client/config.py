# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Resolve the existing launcher + assembled scene inputs without app imports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ClientConfig:
    """Frozen launch command and pacing metadata; no second physics knob channel.

    The manifest is an assemble artifact referring to the scene YAML, not a
    third physics config. Use the normal bringup assembly workflow first.
    """

    launcher_yaml: Path
    stage_manifest: Path
    scene_yaml: Path
    num_envs: int
    physics_hz: float
    ticks_per_step: int
    command: tuple[str, ...]

    @classmethod
    def from_files(cls, launcher_yaml: str | Path, stage_manifest: str | Path, *, headless: bool = True):
        launcher = Path(launcher_yaml).expanduser().resolve(strict=True)
        manifest = Path(stage_manifest).expanduser().resolve(strict=True)
        doc = yaml.safe_load(launcher.read_text())
        physics = doc["launcher"]["physics"]
        engine = physics["engines"][physics["engine"]]
        if engine["package"] != "genie_sim_engine_newton":
            raise ValueError("genie_sim_client currently supports the standalone Newton engine only")
        params = doc[engine.get("name", "genie_sim_engine")]["ros__parameters"]
        if params.get("step_mode") != "lockstep":
            raise ValueError("launcher must explicitly set step_mode: lockstep")
        ticks = params.get("step_lockstep_ticks")
        worlds = params.get("num_envs", 1)  # documented engine default
        hz = float(params["physics_hz"])
        if type(ticks) is not int or ticks < 1 or type(worlds) is not int or worlds < 1:
            raise ValueError("step_lockstep_ticks and num_envs must be positive integers")
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError("physics_hz must be finite and positive")
        if params.get("use_tf_render") is not True:
            raise ValueError("launcher must set use_tf_render: true for the body-state channel")
        # Never let a hidden launcher artifact override the explicit assembly input.
        if "stage_manifest" in params:
            raise ValueError("stage_manifest is an assembly artifact; pass it separately from launcher parameters")
        data = json.loads(manifest.read_text())
        base = Path(data.get("base_path") or manifest.parent)
        if not base.is_absolute():
            base = manifest.parent / base
        if not data.get("scene_yaml"):
            raise ValueError("assembled manifest must reference its scene_yaml")
        scene = Path(data["scene_yaml"])
        scene = (scene if scene.is_absolute() else base / scene).resolve(strict=True)
        # Same ros2 run + ROS params path as RLinf. All physics values, including
        # scene_yaml_defaults and physics_params_file, come from the launcher.
        command = ["ros2", "run", engine["package"], engine["executable"]]
        if not headless:
            command.append("--gui")
        command.append("--ros-args")
        for key, value in params.items():
            encoded = yaml.safe_dump(value, default_flow_style=True, width=1_000_000).strip()
            if encoded.endswith("\n..."):
                encoded = encoded[:-4]
            command += ["-p", f"{key}:={encoded}"]
        command += ["-p", f"stage_manifest:={manifest}", "-p", f"launcher_yaml_path:={launcher}"]
        return cls(launcher, manifest, scene, worlds, hz, ticks, tuple(command))

    @property
    def step_duration(self) -> float:
        """Requested simulation duration per command; not a measured commit."""
        return self.ticks_per_step / self.physics_hz
