# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Normalize physics, sensor and render randomization domains without sampling.

Explicit YAML keys override task defaults, which override benchmark settings.
Null task defaults suppress a dimension. Render settings use explicit/task
settings and built-in defaults without the benchmark layer.
Reject simultaneous per-instance robot-base poses and base-offset randomization:
both would write the engine's base_mocap_off slot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class BaseXYChannelConflictError(RuntimeError):
    """Per-instance base poses and base randomization target the same engine state."""


def _benchmark_config_root() -> Path:
    """Locate geniesim_benchmark's ``benchmark/config`` directory.

    This is the same definition as ``task_loader._benchmark_config_root``,
    pointing at the same directory; it is kept here as its own copy instead
    of importing it from task_loader in order to avoid a reverse dependency
    -- ``task_loader`` needs to import this module, so this module cannot
    turn around and import ``task_loader``.
    """
    import geniesim_benchmark  # noqa: PLC0415

    return Path(geniesim_benchmark.__file__).resolve().parent / "benchmark" / "config"


def resolve_benchmark_generalization(
    task_name: str,
    *,
    bench_root: Optional[str] = None,
) -> dict:
    """Read the eval task's ``generalization`` block and map the PHYSICS-side
    keys onto our seeded-reset spec (render-side keys — lights/hdr/material/
    camera — are handled by the render node, not here; instruction by VecEnv).

    benchmark semantics (generalization_utils.update_init_env):
      * init_joint.thresh  -> per-arm-joint offsets ~ U(-thresh, thresh) [rad]
      * init_base.x/y_thresh -> base xy offset ~ U(-thresh, thresh) [m]
    """
    bench = Path(bench_root) if bench_root else _benchmark_config_root()
    eval_task = json.loads((bench / "eval_tasks" / f"{task_name}.json").read_text())
    g = eval_task.get("generalization") or {}
    out: dict = {}
    ij = g.get("init_joint") or {}
    if ij.get("enable"):
        out["init_joint_thresh"] = float(ij.get("thresh", 0.1))
    ib = g.get("init_base") or {}
    if ib.get("enable"):
        out["init_base_xy_thresh"] = [float(ib.get("x_thresh", 0.01)), float(ib.get("y_thresh", 0.01))]
    return out


# Built-in default for render.lights -- "enable" always starts off (the
# same rule every render/physics dimension below follows: "built-in default
# produces no key = off"); intensity_scale/distant_rotate_deg are the
# values used once enabled but no concrete magnitude was given, matching
# data-collection's set_lights_random (INTENSITY_SCALE_LO/HI = 0.5/1.5,
# ROTATE_DELTA = 45.0).
_DEFAULT_LIGHTS_SPEC: dict = {
    "enable": False,
    "intensity_scale": [0.5, 1.5],
    "distant_rotate_deg": 45.0,
}

# Built-in default for render.material -- "enable" always starts off.
#
# "mode" is one of three choices, see
# randomize.material.scan_material_candidates:
#   * "ml_prefix" (default) -- candidate materials are restricted to
#     sibling Materials under the same-level Looks with an "ml_" prefix;
#     meshes are not filtered by name (scans the whole scene). The "ml_"
#     prefix itself is already the signal "this is a deliberately left
#     variant," with no need to also fall back on a mesh-path keyword.
#   * "keyword" -- candidate materials get no "ml_" filtering (any sibling
#     Material counts), but the mesh must hit one of "keywords",
#     replicating benchmark's apply_material_generalization semantics
#     (swapping its objects/Objects path filter for a keyword filter).
#   * "both" -- both filters apply: the mesh must hit keywords AND the
#     candidate material must have the "ml_" prefix. The most conservative
#     combination.
# "keywords" only takes effect when "mode" is "keyword"/"both".
# "exclude_prims" always takes effect, regardless of "mode" -- hitting the
# substring removes a mesh from the candidates. The robot should never
# appear in the scan sources for candidate materials
# (foreground_usda/background_usda) in the first place; this is a second
# line of defense, not the only one.
_DEFAULT_MATERIAL_SPEC: dict = {
    "enable": False,
    "mode": "ml_prefix",
    "keywords": ["table", "door"],
    "exclude_prims": ["/genie/", "/G1/", "/G2/"],
}

_MATERIAL_MODES = ("ml_prefix", "keyword", "both")

# Built-in default for render.background -- "enable" always starts off.
#
# "rooms": an explicit list of candidate room keys (e.g.
# ["room/room_1", "home/home_3"]; for the format see the
# randomize.background module docstring -- no "background/" prefix, a
# different set of keys from task_background.room_key(), used for a
# different purpose; don't confuse the two). An empty list means
# auto-discover via the max_footprint_m size gate
# (randomize.background.discover_background_rooms), not "off."
#
# "max_footprint_m": None means use the actual env_spacing the caller
# passes in (for the size-gate criterion, see
# randomize.background.passes_size_gate: a candidate room's dx/dy/dz must
# all be below this threshold) -- deliberately not hardcoded to a literal
# here, so that if some task changes env_spacing, the default here doesn't
# silently drift out of sync with the real grid spacing.
#
# This dimension only matters once task_background.resolve_use_background
# has already decided "this task should have a background room attached"
# -- that table controls "attach or not," this one controls "which room to
# attach," and the latter has no consumer without the former. The two
# concerns are layered across two separate modules; this does not repeat
# that table's decision.
_DEFAULT_BACKGROUND_SPEC: dict = {
    "enable": False,
    "rooms": [],
    "max_footprint_m": None,
}

# Built-in default for render.camera_pose -- "enable" always starts off,
# same as the three above, but this dimension **doesn't even get a
# conservative small magnitude**: all six axes of "threshold" are 0.
#
# This is not an oversight -- camera pose changes the **observation
# geometry**, which affects a policy's spatial prior, making trained
# policies and existing eval numbers incomparable; lighting/material ship a
# conservative-looking default because they don't change the geometric
# structure of the observation (brightness/texture doesn't change how the
# camera sees the scene in the world) -- camera pose is different, and any
# non-zero default would silently invalidate every existing baseline.
# Anyone who wants to use this must explicitly configure "threshold"; it
# can't rely on "turning this dimension on automatically gives it some
# reasonable amount."
#
# "threshold": x/y/z are local translation thresholds (meters),
# roll/pitch/yaw are local rotation thresholds (in **degrees**, not
# radians -- matching the units of benchmark's
# _dynamic_camera_position_sample/_apply_camera_position_perturbation; the
# render node converts to radians internally before composing). Each of
# the six axes is drawn independently from [-threshold, threshold], not
# "draw one direction then multiply by one shared magnitude."
#
# Takes benchmark's local-pose-perturbation logic (take the current local
# pose -> add a per-axis offset/compose a rotation -> write back the local
# pose), but **drops** its unconditional "z + 0.01" -- benchmark's
# _apply_camera_position_perturbation still raises the camera 1cm along z
# even when the threshold is 0; that's a pre-existing quirk in its own
# code, and reproducing it here serves no purpose.
_DEFAULT_CAMERA_POSE_SPEC: dict = {
    "enable": False,
    "threshold": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    },
}


def _normalize_render_spec(raw: Optional[dict]) -> dict:
    """Fill defaults for lights, material, background and camera-pose domains.

    Merge each domain field, including individual camera threshold axes. Validate
    material mode here before candidate scanning; benchmark settings are not merged."""
    lights = dict(_DEFAULT_LIGHTS_SPEC)
    lights.update((raw or {}).get("lights") or {})
    material = dict(_DEFAULT_MATERIAL_SPEC)
    material.update((raw or {}).get("material") or {})
    if material["mode"] not in _MATERIAL_MODES:
        raise ValueError(
            f"randomization.render.material.mode must be one of " f"{_MATERIAL_MODES!r}, got {material['mode']!r}"
        )
    background = dict(_DEFAULT_BACKGROUND_SPEC)
    background.update((raw or {}).get("background") or {})
    raw_camera_pose = (raw or {}).get("camera_pose") or {}
    camera_pose = dict(_DEFAULT_CAMERA_POSE_SPEC)
    camera_pose.update(raw_camera_pose)
    threshold = dict(_DEFAULT_CAMERA_POSE_SPEC["threshold"])
    threshold.update(raw_camera_pose.get("threshold") or {})
    camera_pose["threshold"] = threshold
    return {"lights": lights, "material": material, "background": background, "camera_pose": camera_pose}


# Built-in default for randomization.joint_pd (physics side) -- "enable" is
# always off, the same "built-in default = off" rule as the render
# dimensions above, but this dimension **likewise doesn't even get a
# conservative small magnitude** (same reasoning as camera pose):
# kp_scale/kd_scale default to the identity interval [1.0, 1.0].
#
# **Not under render** -- this dimension changes the robot joints' drive
# gains (kp/kd), and the consumer is the engine
# (``genie_sim_engine_newton``), not the render node, so it sits at the top
# level of ``randomization:``, alongside ``object_xy_radius`` /
# ``init_joint_thresh`` / ``init_base_xy_thresh``, not under
# ``randomization.render`` -- those three keys are the existing precedent
# for this rule.
#
# joint_pd changes the **robot's dynamic response**, making trained
# policies' action distributions incomparable, and any non-zero default
# would silently invalidate every existing baseline. Anyone who wants to use
# this must explicitly configure kp_scale/kd_scale; it can't rely on
# "turning this dimension on automatically gives it some reasonable amount"
# -- the same reasoning as camera pose, not an oversight.
#
# kp_scale/kd_scale are **relative multiplier intervals** ([lo, hi], drawn
# independently per env via rng.uniform(lo, hi)), not an absolute candidate
# list -- the reasoning is the same as lighting/camera pose: absolute
# values would need to be calibrated per robot, and this layer shouldn't
# guess a "reasonable" absolute kp/kd for every piece of hardware.
#
# Only three joint kinds are allowed through -- arm, waist, and head
# (engine-side JK_BODY/JK_ARM*/JK_HEAD); the gripper's (JK_GRIPPER) and the
# base's (JK_CHASSIS_*) gains are never scaled -- this constraint lives on
# the engine side (the eligible_kinds set in
# lifecycle.py::_compute_joint_pd_baseline); the spec here doesn't need to,
# and shouldn't, re-express "which joints get scaled" -- it only owns the
# on/off switch and the multiplier interval.
_DEFAULT_JOINT_PD_SPEC: dict = {
    "enable": False,
    "kp_scale": [1.0, 1.0],
    "kd_scale": [1.0, 1.0],
}


def _normalize_joint_pd_spec(raw: Optional[dict]) -> Optional[dict]:
    """Normalize enabled PD gain domains with identity defaults; omit disabled gains."""
    if raw is None:
        return None
    spec = dict(_DEFAULT_JOINT_PD_SPEC)
    spec.update(raw)
    return spec


def resolve_benchmark_camera_generalization(
    task_name: str,
    *,
    bench_root: Optional[str] = None,
) -> dict:
    """Read enabled noise, occlusion and drop-frame domains from the eval task.

    Preserve each enabled sub-dictionary for sampling. Camera selection belongs to
    VecEnv image_keys and is not part of this domain mapping."""
    bench = Path(bench_root) if bench_root else _benchmark_config_root()
    eval_task = json.loads((bench / "eval_tasks" / f"{task_name}.json").read_text())
    camera = ((eval_task.get("generalization") or {}).get("camera")) or {}
    out: dict = {}
    noise = camera.get("noise") or {}
    if noise.get("enable"):
        out["noise"] = noise
    occlusion = camera.get("occlusion") or {}
    if occlusion.get("enable"):
        out["occlusion"] = occlusion
    drop_frame = camera.get("drop_frame") or {}
    if drop_frame.get("enable"):
        out["drop_frame"] = drop_frame
    return out


class RandomizationSpec:
    """Normalized physics, sensor and render parameter domains.

    Only physics is serialized into the engine scene config. VecEnv consumes sensor
    parameters. Render bridges consume lights/material/camera-pose parameters; the
    stage builder consumes background selection. Sampling is separate."""

    def __init__(self, physics: dict, sensor: Optional[dict] = None, render: Optional[dict] = None):
        self.physics: dict = dict(physics)
        self.sensor: dict = dict(sensor or {})
        self.render: dict = dict(render or {})

    @staticmethod
    def normalize_render(yaml_render: Optional[dict]) -> dict:
        """Normalize the render subsection independently of physics and benchmark files."""
        return _normalize_render_spec(yaml_render)

    @classmethod
    def resolve(
        cls,
        task_name: str,
        yaml_randomization: Optional[dict],
        *,
        bench_root: Optional[str] = None,
        task_defaults: Optional[dict] = None,
    ) -> "RandomizationSpec":
        """Resolve explicit YAML > task defaults > benchmark settings > built-in defaults.

        Merge at top-level keys; null suppresses a physics key. Optional switches disable
        task or benchmark defaults. Sensor and render use separate namespaces, while
        joint_pd remains in physics. Log benchmark parse failures and continue with the
        remaining layers. Return normalized domains without drawing random values."""
        rand = dict(yaml_randomization or {})
        # Task-default generalization is merged first (a top-level
        # per-key setdefault), so its sensor/render/joint_pd sections go
        # through the same normalization below as the yaml's own.
        if task_defaults and rand.get("use_task_generalization", True):
            for k, v in task_defaults.items():
                rand.setdefault(k, v)
        sensor = dict(rand.pop("sensor", None) or {})
        render = _normalize_render_spec(rand.pop("render", None))
        joint_pd = _normalize_joint_pd_spec(rand.pop("joint_pd", None))
        if joint_pd is not None:
            rand["joint_pd"] = joint_pd
        if rand.get("use_benchmark_generalization", True):
            try:
                bench_gen = resolve_benchmark_generalization(task_name, bench_root=bench_root)
                for k, v in bench_gen.items():
                    rand.setdefault(k, v)
            except Exception as exc:  # noqa: BLE001
                print(f"[randomize.spec] benchmark generalization skipped: {exc!r}")
            try:
                bench_sensor = resolve_benchmark_camera_generalization(task_name, bench_root=bench_root)
                for k, v in bench_sensor.items():
                    sensor.setdefault(k, v)
            except Exception as exc:  # noqa: BLE001
                print(f"[randomize.spec] benchmark camera generalization " f"skipped: {exc!r}")
        # null = this dimension is off: it occupies the key to block the
        # eval task block, and is stripped back out here, since the
        # engine only understands "this key is absent."
        rand = {k: v for k, v in rand.items() if v is not None}
        return cls(rand, sensor, render)

    def to_dict(self) -> dict:
        return dict(self.physics)

    def __bool__(self) -> bool:
        # Deliberately physics-only: task_loader.py gates
        # ``scene_cfg["randomization"] = spec.to_dict()`` on this, and
        # ``to_dict()`` itself never includes ``sensor``/``render`` (the
        # engine doesn't consume either -- see the class docstring). Folding
        # them into this predicate would start writing an (empty)
        # ``randomization`` key into scene_cfg purely because sensor- or
        # render-only config was given, which is an engine-facing behavior
        # change this layer has no business making.
        return bool(self.physics)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"RandomizationSpec(physics={self.physics!r}, " f"sensor={self.sensor!r}, render={self.render!r})"


def check_base_xy_channel_conflict(
    *,
    task_name: str,
    sub_task_name: str,
    instance_id: int,
    instance_switch: Optional[dict],
    randomization: Optional[dict],
) -> None:
    """Reject simultaneous per-instance base poses and base-offset randomization.

    Both non-empty channels write base_mocap_off; even [0, 0] counts as an active
    randomization channel. Check after the scene configuration is fully assembled."""
    robot_base_xy = (instance_switch or {}).get("robot_base_xy")
    init_base_xy_thresh = (randomization or {}).get("init_base_xy_thresh")
    if robot_base_xy and init_base_xy_thresh:
        raise BaseXYChannelConflictError(
            f"task={task_name!r} sub_task={sub_task_name!r} "
            f"instance={instance_id!r}: both "
            f"instance_switch.robot_base_xy={robot_base_xy!r} and "
            f"randomization.init_base_xy_thresh={init_base_xy_thresh!r} are "
            f"set. Both write the SAME base_mocap_off[w] slot in engine "
            f"lifecycle.py (_apply_instance_poses writes it first, then "
            f"_randomize_world_objects silently overwrites it) -- turning "
            f"on init_base_xy_thresh would silently clobber the "
            f"per-instance pose this task was set up to replicate exactly. "
            f"These are two different things (a verbatim-replay set vs. a "
            f"generalization set), and mixing them is almost always a "
            f"config mistake, not a real requirement. Fix: drop "
            f"randomization.init_base_xy_thresh for this sub-task (unset "
            f"it, or set generalization.init_base.enable: false / "
            f"randomization.use_benchmark_generalization: false), or drop "
            f"the per-instance robot_init.json table if this sub-task "
            f"should generalize the base pose instead of replicating it."
        )


__all__ = [
    "RandomizationSpec",
    "resolve_benchmark_generalization",
    "resolve_benchmark_camera_generalization",
    "check_base_xy_channel_conflict",
    "BaseXYChannelConflictError",
]
