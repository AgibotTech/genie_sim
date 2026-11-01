# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Sample reproducible randomization plans using independent per-dimension streams.

Object, arm and base pose draws remain in the engine's RNG stream. This sampler
records their thresholds and draws PD gains, sensors, materials, backgrounds and
camera poses independently per environment. Lights are drawn from the smallest
env id in each call and shared across the grid. Background and PD allocation
happen at stage construction; other render dimensions can vary on reset."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .plan import RandomizationPlan
from .spec import RandomizationSpec


def substreams(seed: int, env_id: int) -> dict:
    """Derive independent rng streams for this env from one seed.

    Physics keeps its own stream (unchanged, see the module docstring).
    Render and sensor get separate streams so that adding a render
    dimension never shifts the physics draw sequence.

    ``seed`` must be >= 0. ``seed == -1`` is the sentinel for "replicate
    verbatim" and has no corresponding rng stream (``np.random.SeedSequence``
    rejects negative entropy). The caller (``Sampler.sample``) must filter
    out envs with seed < 0 before calling this function; the check here is
    a fallback, not the primary guard.
    """
    if seed < 0:
        raise ValueError(
            f"substreams() got seed={seed} -- negative seeds are the "
            f"verbatim-replay sentinel (seed == -1) and must never reach a "
            f"stream derivation; the caller should have filtered this env "
            f"out already (see Sampler.sample)"
        )
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(int(env_id),))
    render, sensor = ss.spawn(2)
    return {"render": np.random.default_rng(render), "sensor": np.random.default_rng(sensor)}


def _sample_from_list(rng: np.random.Generator, choices: list) -> Any:
    """Pick one value from a candidate list, or ``None`` if ``choices`` is
    empty or not configured."""
    if not choices:
        return None
    return rng.choice(choices)


def _draw_camera_noise(rng: np.random.Generator, noise_spec: dict) -> dict:
    """Draw a concrete noise config from the ``spec.sensor["noise"]``
    parameter domain."""
    if not noise_spec.get("enable", False):
        return {}
    types = noise_spec.get("types") or []
    if not types:
        return {}
    noise_type = str(rng.choice(types))
    params: dict = {"enable": True, "type": noise_type}
    if noise_type == "gaussian":
        std = _sample_from_list(rng, (noise_spec.get("gaussian") or {}).get("std_range") or [])
        if std is not None:
            params["std"] = float(std)
    elif noise_type == "uniform":
        u = noise_spec.get("uniform") or {}
        params["low"] = float(u.get("low", -0.1))
        params["high"] = float(u.get("high", 0.1))
    elif noise_type == "salt_pepper":
        sp = noise_spec.get("salt_pepper") or {}
        amount = _sample_from_list(rng, sp.get("amount_range") or [])
        params["salt_vs_pepper"] = float(sp.get("salt_vs_pepper", 0.5))
        if amount is not None:
            params["amount"] = float(amount)
    elif noise_type == "exponential":
        scale = _sample_from_list(rng, (noise_spec.get("exponential") or {}).get("scale_range") or [])
        if scale is not None:
            params["scale"] = float(scale)
    return params


def _draw_camera_occlusion(rng: np.random.Generator, occlusion_spec: dict) -> dict:
    """Draw a concrete occlusion config from ``spec.sensor["occlusion"]``."""
    if not occlusion_spec.get("enable", False):
        return {}
    ratio = _sample_from_list(rng, occlusion_spec.get("ratio_range") or [])
    if ratio is None:
        return {}
    return {"enable": True, "ratio": float(ratio), "prob": float(occlusion_spec.get("prob", 1.0))}


def _draw_camera_drop_frame(rng: np.random.Generator, drop_frame_spec: dict) -> dict:
    """Draw a concrete frame-drop config from ``spec.sensor["drop_frame"]``."""
    if not drop_frame_spec.get("enable", False):
        return {}
    drop_prob = _sample_from_list(rng, drop_frame_spec.get("drop_prob_range") or [])
    if drop_prob is None:
        return {}
    return {"enable": True, "drop_prob": float(drop_prob)}


def _draw_sensor_plan(rng: np.random.Generator, sensor_spec: dict) -> dict:
    """``spec.sensor`` (parameter domain) -> this env's concrete
    ``camera_gen_config`` (``camera_noise`` / ``camera_occlusion`` /
    ``camera_drop_frame``)."""
    if not sensor_spec:
        return {}
    out: dict = {}
    noise = _draw_camera_noise(rng, sensor_spec.get("noise") or {})
    if noise:
        out["camera_noise"] = noise
    occlusion = _draw_camera_occlusion(rng, sensor_spec.get("occlusion") or {})
    if occlusion:
        out["camera_occlusion"] = occlusion
    drop_frame = _draw_camera_drop_frame(rng, sensor_spec.get("drop_frame") or {})
    if drop_frame:
        out["camera_drop_frame"] = drop_frame
    return out


def _draw_light_plan(rng: np.random.Generator, lights_spec: dict, source_env_id: int) -> dict:
    """Draw one lighting perturbation shared by the whole grid from
    ``spec.render["lights"]``. ``rng`` must be the render substream of the
    env with id ``source_env_id``; that id is recorded in the result so a
    dumped plan shows which env's seed produced it (see the module
    docstring).

    The produced shape is the ``perturbation`` dict that
    ``task_lighting.perturb_light_rig`` consumes directly:
    ``dome_intensity_scale``/``sun_intensity_scale`` are each drawn
    independently (two lights, not a shared multiplier), and the three
    axes of ``sun_rotate_delta_deg`` are each drawn independently (only
    the sun rotates, not the dome). Returns ``{}`` when disabled.
    """
    if not lights_spec.get("enable", False):
        return {}
    lo, hi = lights_spec.get("intensity_scale", [1.0, 1.0])
    lo, hi = float(lo), float(hi)
    rot = float(lights_spec.get("distant_rotate_deg", 0.0))
    return {
        "source_env_id": int(source_env_id),
        "dome_intensity_scale": float(rng.uniform(lo, hi)),
        "sun_intensity_scale": float(rng.uniform(lo, hi)),
        "sun_rotate_delta_deg": [float(rng.uniform(-rot, rot)) for _ in range(3)],
    }


def _material_stream(seed: int, env_id: int) -> np.random.Generator:
    """Independent rng stream for material, isolated from the ``"render"``
    stream ``substreams()`` derives (the one lighting uses), so toggling
    material never shifts lighting's draw order."""
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(int(env_id), 1))
    return np.random.default_rng(ss)


def _background_stream(seed: int, env_id: int) -> np.random.Generator:
    """Independent rng stream for background room selection, isolated from
    the render and material streams."""
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(int(env_id), 2))
    return np.random.default_rng(ss)


def _draw_background_plan(rng: np.random.Generator, background_spec: dict, candidates: Sequence[str]) -> dict:
    """Independently pick one background room for this env from
    ``candidates`` (the output of
    ``randomize.background.discover_background_rooms``: room keys that
    passed the size gate, already sorted). Re-sorting here is defensive,
    not a reliance on caller discipline.

    Returns ``{}`` when ``background_spec.enable`` is false or
    ``candidates`` is empty. The produced ``{"room": ...}`` is directly the
    content of ``plan.render["background"]``.
    """
    if not background_spec.get("enable", False) or not candidates:
        return {}
    room = str(rng.choice(sorted(candidates)))
    return {"room": room}


def _draw_material_plan(rng: np.random.Generator, material_spec: dict, candidates: Mapping[str, list]) -> dict:
    """Independently choose a material for each mesh in ``candidates``
    (the output of ``randomize.material.scan_material_candidates``:
    ``{mesh relative path: candidate material relative paths}``), skipping
    meshes matched by ``exclude_prims``.

    Meshes and candidate lists are iterated in sorted order so the result
    depends only on the seed, not on dict/list iteration order.

    Returns ``{}`` when ``material_spec.enable`` is false or ``candidates``
    is empty (an asset with no material variants).
    """
    if not material_spec.get("enable", False) or not candidates:
        return {}
    exclude = material_spec.get("exclude_prims") or []
    picks: dict[str, str] = {}
    for mesh_path in sorted(candidates):
        if any(pat in mesh_path for pat in exclude):
            continue
        options = candidates[mesh_path]
        if not options:
            continue
        picks[mesh_path] = str(rng.choice(sorted(options)))
    return picks


def _camera_pose_stream(seed: int, env_id: int) -> np.random.Generator:
    """Independent rng stream for camera-pose perturbation, isolated from
    the render, material, and background streams."""
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(int(env_id), 3))
    return np.random.default_rng(ss)


_CAMERA_POSE_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def _draw_camera_pose_plan(rng: np.random.Generator, camera_pose_spec: dict, camera_paths: Sequence[str]) -> dict:
    """Independently draw a 6-axis (x, y, z, roll, pitch, yaw) local-pose
    perturbation for each camera in ``camera_paths``, for this one env.

    ``camera_paths`` is the list of camera identifiers shared by all envs
    in this call (``CameraConfig.path`` strings, e.g.
    ``"head_link3/head_front_Camera"``, read directly from the manifest's
    ``cameras[].path`` field).

    Cameras are iterated in sorted order so the result does not depend on
    the caller's list order. Each axis is drawn independently via
    ``rng.uniform(-threshold, threshold)`` (the six axes are uncorrelated).
    When an axis's threshold is exactly 0 (the default for all axes), it is
    written as ``0.0`` directly without calling ``rng.uniform``, so whether
    an axis consumes a draw does not depend on a floating-point corner
    case. This is also the first half of the "all-zero magnitude -> pose
    unchanged byte-for-byte" guarantee; the application side
    (``_apply_camera_pose_perturbation``) holds the other half by skipping
    ``Set()`` entirely on an all-zero perturbation.

    Returns ``{}`` when ``camera_pose_spec.enable`` is false or
    ``camera_paths`` is empty.
    """
    if not camera_pose_spec.get("enable", False) or not camera_paths:
        return {}
    threshold = camera_pose_spec.get("threshold") or {}
    plan: dict[str, dict] = {}
    for cam_path in sorted(camera_paths):
        pert: dict[str, float] = {}
        for axis in _CAMERA_POSE_AXES:
            t = float(threshold.get(axis, 0.0))
            pert[axis] = float(rng.uniform(-t, t)) if t != 0.0 else 0.0
        plan[cam_path] = pert
    return plan


def _joint_pd_stream(seed: int, env_id: int) -> np.random.Generator:
    """Independent rng stream for joint PD gain scaling, isolated from the
    render-side streams and from the engine's own physics rng stream
    (``lifecycle.py``, which takes ``seed`` directly as entropy rather than
    deriving a ``SeedSequence`` spawn_key)."""
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(int(env_id), 4))
    return np.random.default_rng(ss)


def _draw_joint_pd_plan(rng: np.random.Generator, joint_pd_spec: dict) -> dict:
    """Independently draw ``kp_scale``/``kd_scale`` multipliers for this
    env from their configured ``[lo, hi]`` intervals (drawn independently
    of each other).

    When ``lo == hi`` (the default identity interval, ``[1.0, 1.0]``), the
    value is written as ``lo`` directly without calling ``rng.uniform``,
    for the same reason as camera pose's zero-threshold case. This is the
    first half of the "scale is identity -> unchanged byte-for-byte"
    guarantee; the application side (the engine's ``_scale_joint_pd``)
    holds the other half by skipping the gain-array read/write entirely
    when both scales are 1.0.

    Unlike ``object_xy_radius``/``init_joint_thresh``/``init_base_xy_thresh``,
    which the engine redraws by seed on its own, this dimension's result is
    the concrete multiplier itself -- the engine only applies it, never
    draws it. Returns ``{}`` when ``joint_pd_spec.enable`` is false.
    """
    if not joint_pd_spec.get("enable", False):
        return {}
    kp_lo, kp_hi = joint_pd_spec.get("kp_scale", [1.0, 1.0])
    kd_lo, kd_hi = joint_pd_spec.get("kd_scale", [1.0, 1.0])
    kp_lo, kp_hi = float(kp_lo), float(kp_hi)
    kd_lo, kd_hi = float(kd_lo), float(kd_hi)
    kp_scale = kp_lo if kp_lo == kp_hi else float(rng.uniform(kp_lo, kp_hi))
    kd_scale = kd_lo if kd_lo == kd_hi else float(rng.uniform(kd_lo, kd_hi))
    return {"kp_scale": kp_scale, "kd_scale": kd_scale}


class Sampler:
    """Given an already-normalized ``RandomizationSpec``, produce each
    env's ``RandomizationPlan`` by seed."""

    def __init__(self, spec: RandomizationSpec):
        self._spec = spec

    def sample(
        self,
        seeds: Mapping[int, int],
        material_candidates: Optional[Mapping[str, list]] = None,
        background_candidates: Optional[Sequence[str]] = None,
        camera_pose_candidates: Optional[Sequence[str]] = None,
    ) -> dict[int, RandomizationPlan]:
        """Return plans for non-negative seeds in the {env_id: seed} mapping.

        Negative seeds produce no plan. Each environment uses independent streams for
        sensor, material, background, camera-pose and PD draws; lights are shared.
        Material candidates map mesh paths to sorted variant paths; background candidates
        are validated room keys; camera candidates come from manifest camera paths.
        Empty candidates suppress that dimension. PD draws replace the gain domain in
        each plan; pose thresholds remain for the engine to sample."""
        physics = self._spec.to_dict()
        joint_pd_spec = physics.get("joint_pd") or {}
        sensor_spec = self._spec.sensor
        lights_spec = (self._spec.render or {}).get("lights") or {}
        material_spec = (self._spec.render or {}).get("material") or {}
        background_spec = (self._spec.render or {}).get("background") or {}
        camera_pose_spec = (self._spec.render or {}).get("camera_pose") or {}
        plans: dict[int, RandomizationPlan] = {}
        streams_by_env: dict[int, dict] = {}
        for env_id, seed in seeds.items():
            seed = int(seed)
            if seed < 0:
                continue
            env_id = int(env_id)
            streams = substreams(seed, env_id)
            streams_by_env[env_id] = streams
            # The render substream feeds lighting (broadcast below); the
            # sensor substream is consumed independently per env, drawing
            # this env's concrete noise/occlusion/drop-frame parameters.
            sensor_plan = _draw_sensor_plan(streams["sensor"], sensor_spec) if sensor_spec else {}
            render_plan: dict[str, Any] = {}
            # material: drawn independently per env, using its own
            # material substream (uncorrelated with lighting's render
            # substream, see _material_stream).
            if material_spec.get("enable", False) and material_candidates:
                material_plan = _draw_material_plan(_material_stream(seed, env_id), material_spec, material_candidates)
                if material_plan:
                    render_plan["material"] = material_plan
            # background: drawn independently per env, using its own
            # background substream (uncorrelated with lighting's and
            # material's substreams, see _background_stream).
            if background_spec.get("enable", False) and background_candidates:
                background_plan = _draw_background_plan(
                    _background_stream(seed, env_id), background_spec, background_candidates
                )
                if background_plan:
                    render_plan["background"] = background_plan
            # camera_pose: drawn independently per env, using its own
            # camera_pose substream (uncorrelated with lighting's,
            # material's, and background's substreams, see
            # _camera_pose_stream).
            if camera_pose_spec.get("enable", False) and camera_pose_candidates:
                camera_pose_plan = _draw_camera_pose_plan(
                    _camera_pose_stream(seed, env_id), camera_pose_spec, camera_pose_candidates
                )
                if camera_pose_plan:
                    render_plan["camera_pose"] = camera_pose_plan
            env_physics = dict(physics)
            # joint_pd: drawn independently per env, using its own
            # joint_pd substream (uncorrelated with all four render
            # sub-dimensions' substreams, see _joint_pd_stream). Strip the
            # parameter domain dict(physics) copied in above --
            # env_physics["joint_pd"] must only ever be the concrete
            # multiplier this env drew, or the key must be absent, never
            # the domain (even one with enable: false): a caller could
            # construct RandomizationSpec(physics={"joint_pd": {...}})
            # directly, bypassing the normalization in
            # spec._normalize_joint_pd_spec, so this guarantees the
            # invariant itself rather than trusting physics to arrive
            # clean.
            env_physics.pop("joint_pd", None)
            if joint_pd_spec.get("enable", False):
                joint_pd_plan = _draw_joint_pd_plan(_joint_pd_stream(seed, env_id), joint_pd_spec)
                if joint_pd_plan:
                    env_physics["joint_pd"] = joint_pd_plan
            plans[env_id] = RandomizationPlan(
                seed=seed, env_id=env_id, physics=env_physics, render=render_plan, sensor=sensor_plan
            )

        # lights: derive the set of values shared by the whole grid from
        # the render substream of whichever env has the smallest env_id in
        # this sample() call, and broadcast it to every plan this call
        # produces (see the module docstring). When every env has
        # seed < 0, plans is empty and this is a no-op.
        if plans and lights_spec.get("enable", False):
            source_env_id = min(plans)
            light_plan = _draw_light_plan(streams_by_env[source_env_id]["render"], lights_spec, source_env_id)
            if light_plan:
                for plan in plans.values():
                    plan.render["lights"] = dict(light_plan)
        return plans


__all__ = ["substreams", "Sampler"]
