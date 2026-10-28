# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Apply render-side generalization (lights, material, camera pose) to a live stage.

Shared by the two Isaac nodes that render the multi-env GridCloner stage: the
render-only tiled node (``isaacsim_tiled_render.py``) and the fused
physics+render node (``isaacsim_fused_sim.py``). The bridge side samples a
reset's values (``randomize.Sampler``) and sends them as ``randomize_cmd.json``;
the node caches the light and camera baselines once per stage build and hands
each command to ``_apply_randomize_cmd``, which applies it and writes
``randomize_ack.json`` with per-dimension readback counts.

This module only depends on ``pxr``, not Kit, so it can be imported from
either node even though ``isaacsim_tiled_render.py`` itself starts Kit at
import time and cannot be imported directly.
"""
from __future__ import annotations

from typing import List, Optional

from rlinf_geniesim.render.bridges._hotswap_protocol import write_randomize_ack


# ---------------------------------------------------------------------------
# Lights: the first dimension of the render-side command channel. Only
# changes UsdLux attributes and triggers no stage rebuild, so it is tracked
# by its own randomize_cmd.json/randomize_ack.json pair, separate from
# reload_cmd.json/reload_ack.json and with its own generation counter.
#
# ``/RLinfEnv/Dome``/``/RLinfEnv/Sun`` are written by
# task_loader._wrap_scene_with_default_env when the stage is built (for
# tasks with no background room); ``/RenderOVRTX/RLinfDomeLight`` is a
# fill-light dome written by a different code path for foreground-only
# scenes (task_loader.author_task_render_layer, into render_layer.usda).
#
# ``build_gridcloner_stage`` (the assembly path this node uses) never
# references ``manifest.render_layer_usda`` -- only the single-camera
# ``isaacsim_render.py`` path (``init_stage()``) refs it into
# ``/RenderOVRTX`` -- so ``/RenderOVRTX/RLinfDomeLight`` is currently
# always missing from this node's stage. Keeping it in the candidate list
# is harmless: failing to find a prim only warns, never crashes (see
# ``_apply_light_perturbation`` below). Tasks with a background room never
# get ``/RLinfEnv/Dome``+``Sun`` written at all -- the room's own lights
# are not this channel's concern.
# ---------------------------------------------------------------------------

_DOME_PATHS = ("/RLinfEnv/Dome", "/RenderOVRTX/RLinfDomeLight")
_SUN_PATH = "/RLinfEnv/Sun"


def _get_light_num(prim, name: str) -> Optional[float]:
    """Reads ``inputs:{name}`` or the bare ``{name}`` (``task_loader``'s
    ``_author_light`` may write either, and they are equal at write time).
    Returns ``None`` if neither exists."""
    for attr_name in (f"inputs:{name}", name):
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            v = attr.Get()
            if v is not None:
                return float(v)
    return None


def _set_light_num(prim, name: str, value: float) -> None:
    """Writes ``inputs:{name}`` and the bare ``{name}`` -- whichever
    exists gets written (both may exist, in which case both get written,
    since it is not certain which one the renderer reads)."""
    for attr_name in (f"inputs:{name}", name):
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            attr.Set(float(value))


def _read_light_baseline(stg) -> dict:
    """Reads the current lights attributes once right after the stage is
    (re)built, and caches that as this build's perturbation **baseline**.

    Every subsequent randomize command computes relative to this cached
    baseline rather than the attribute's current value -- otherwise
    repeated resets would compound the perturbation, and brightness would
    drift with the number of resets instead of perturbing around the
    original value each time.

    A prim that cannot be found is simply absent from the result (``"dome"``
    is a dict, and a path that cannot be found does not appear in it);
    ``_apply_light_perturbation`` warns based on this rather than
    crashing."""
    doms: dict = {}
    for p in _DOME_PATHS:
        prim = stg.GetPrimAtPath(p)
        if prim and prim.IsValid():
            v = _get_light_num(prim, "intensity")
            doms[p] = v if v is not None else 1.0
    sun_intensity, sun_rotate, has_sun = None, None, False
    sun_prim = stg.GetPrimAtPath(_SUN_PATH)
    if sun_prim and sun_prim.IsValid():
        has_sun = True
        v = _get_light_num(sun_prim, "intensity")
        sun_intensity = v if v is not None else 1.0
        rot_attr = sun_prim.GetAttribute("xformOp:rotateXYZ")
        rv = rot_attr.Get() if (rot_attr and rot_attr.IsValid()) else None
        sun_rotate = (float(rv[0]), float(rv[1]), float(rv[2])) if rv is not None \
            else (0.0, 0.0, 0.0)
    return {"dome": doms, "has_sun": has_sun,
            "sun_intensity": sun_intensity, "sun_rotate": sun_rotate}


def _apply_light_perturbation(stg, baseline: dict, cmd: dict) -> "tuple[int, int]":
    """Applies ``cmd`` (drawn by ``randomize.sampler._draw_light_plan``)
    onto ``stg``'s lights, computed relative to ``baseline`` (see
    ``_read_light_baseline``), not the prim's current value.

    A prim missing from ``baseline`` only prints a WARNING and is skipped
    -- one missing light should not sink the whole render node. Any
    exception this function itself raises is still caught by the caller
    (``_apply_randomize_cmd``) and recorded in the ack's ``error`` field.

    Returns ``(applied, requested)``: ``requested`` counts the lights
    prims present in ``baseline``, ``applied`` counts how many were
    actually set."""
    applied = 0
    requested = 0
    if not baseline.get("dome") and not baseline.get("has_sun"):
        print(f"[render] WARN: randomize lights — neither a dome light under "
              f"{_DOME_PATHS} nor a sun light at {_SUN_PATH} was found when "
              f"the baseline was cached; nothing to perturb this reset "
              f"(background task? see task_lighting.py's module docstring)",
              flush=True)
        return applied, requested

    dome_scale = float(cmd.get("dome_intensity_scale", 1.0))
    for path, i0 in baseline.get("dome", {}).items():
        requested += 1
        prim = stg.GetPrimAtPath(path)
        if not (prim and prim.IsValid()):
            print(f"[render] WARN: randomize lights — dome prim {path} "
                  f"existed when the baseline was cached but is gone now; "
                  f"skipping", flush=True)
            continue
        _set_light_num(prim, "intensity", i0 * dome_scale)
        applied += 1

    if baseline.get("has_sun"):
        requested += 1
        sun_prim = stg.GetPrimAtPath(_SUN_PATH)
        if sun_prim and sun_prim.IsValid():
            sun_scale = float(cmd.get("sun_intensity_scale", 1.0))
            _set_light_num(sun_prim, "intensity",
                          baseline["sun_intensity"] * sun_scale)
            applied += 1
            delta = cmd.get("sun_rotate_delta_deg") or (0.0, 0.0, 0.0)
            ox, oy, oz = baseline["sun_rotate"]
            dx, dy, dz = (float(v) for v in delta)
            rot_attr = sun_prim.GetAttribute("xformOp:rotateXYZ")
            if rot_attr and rot_attr.IsValid():
                from pxr import Gf  # noqa: PLC0415

                rot_attr.Set(Gf.Vec3f(ox + dx, oy + dy, oz + dz))
            else:
                print(f"[render] WARN: randomize lights — sun prim "
                      f"{_SUN_PATH} has no xformOp:rotateXYZ; skipping "
                      f"rotation (intensity still applied)", flush=True)
        else:
            print(f"[render] WARN: randomize lights — sun prim {_SUN_PATH} "
                  f"existed when the baseline was cached but is gone now; "
                  f"skipping", flush=True)
    else:
        print(f"[render] WARN: randomize lights — no sun light prim at "
              f"{_SUN_PATH} (background task? the room's own lights are not "
              f"managed by this channel); skipping sun perturbation, "
              f"dome (if any) was still applied above", flush=True)
    return applied, requested


def _apply_material_bindings(stg, env_id: int, bindings: dict):
    from _instance_materials import _apply_material_bindings as apply
    return apply(stg, env_id, bindings)


# ---------------------------------------------------------------------------
# Camera pose: drawn independently per env, per camera, and reapplied on
# every reset (not a one-time allocation like background).
#
# This node's obs cameras (``manifest.obs_cameras()``, resolved by step 4 of
# ``build_gridcloner_stage``) are real ``Camera`` child prims on the robot
# USD (e.g. ``.../head_link3/Head_Camera``), with a standard
# ``xformOp:translate``+``xformOp:orient``+``xformOp:scale`` local xform --
# not something baked once at build time into a render product. The OVRTX
# single-camera path (``isaacsim_render.py``/``render/scene_assembly.py``'s
# ``author_render_layer``) does author its camera into ``render_layer.usda``
# with a world-anchored transform, but ``build_gridcloner_stage`` (the
# assembly path this node uses) never references that file (see the lights
# comment above), so this channel never touches that kind of camera.
#
# GridCloner's ``copy_from_source=True`` deep copy means each env's camera
# prim is an independent spec, so changing the local xform per env does not
# cross-contaminate other envs (the same mechanism as background's per-env
# override).
#
# A camera's own local xform is not like the driven ``/entity`` (a single
# ``xformOp:transform`` matrix overwritten every frame by ``/tf_render``) --
# it is a static child xform under the (driven) parent link, and only this
# channel ever touches it, so a write stays in effect and is never
# overwritten by the next frame's drive.
# ---------------------------------------------------------------------------

_CAMERA_POSE_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def _read_camera_pose_baseline(stg, cam_paths: List[str], obs: list,
                               n_cams: int, n_envs: int) -> dict:
    """Reads each env's each camera's local pose (``xformOp:translate``+
    ``xformOp:orient``) once right after the stage is (re)built, and
    caches it as this build's perturbation baseline -- the same
    discipline as ``_read_light_baseline`` and for the same reason:
    computing each command relative to a fixed baseline instead of the
    current value avoids perturbations compounding across resets.

    ``cam_paths``/``obs``/``n_cams``/``n_envs`` come from this node's
    ``build_scene()`` call: ``cam_paths`` is the flat ``[env_i][cam_j]``
    list of absolute prim paths (flat index = ``env_i*n_cams+cam_j``), and
    ``obs[cam_j].path`` (e.g. ``"head_link3/Head_Camera"``) is this
    camera's stable identifier, matching the ``cameras[].path`` field the
    bridge process reads from the same manifest.

    A camera whose prim or xformOps cannot be found is skipped in the
    baseline; ``_apply_camera_pose_perturbation`` warns and skips it based
    on that, rather than crashing.

    Returns ``{env_id: {cam_path: (absolute prim path, Gf.Vec3d
    translation, Gf.Quatd orientation)}}``."""
    from pxr import Gf  # noqa: PLC0415

    baseline: dict = {}
    for env_i in range(n_envs):
        cams: dict = {}
        for cam_j in range(n_cams):
            if cam_j >= len(obs):
                continue
            flat = env_i * n_cams + cam_j
            if flat >= len(cam_paths):
                continue
            prim_path = cam_paths[flat]
            prim = stg.GetPrimAtPath(prim_path)
            if not (prim and prim.IsValid()):
                print(f"[render] WARN: randomize camera_pose — camera prim "
                      f"{prim_path} not found while caching the pose "
                      f"baseline (env {env_i}); this camera will be "
                      f"skipped by any camera_pose command", flush=True)
                continue
            t_attr = prim.GetAttribute("xformOp:translate")
            o_attr = prim.GetAttribute("xformOp:orient")
            if not (t_attr and t_attr.IsValid() and o_attr and o_attr.IsValid()):
                print(f"[render] WARN: randomize camera_pose — camera prim "
                      f"{prim_path} has no xformOp:translate/xformOp:orient "
                      f"while caching the pose baseline (env {env_i}); this "
                      f"camera will be skipped by any camera_pose command",
                      flush=True)
                continue
            t0 = t_attr.Get()
            o0 = o_attr.Get()
            if t0 is None or o0 is None:
                continue
            cams[obs[cam_j].path] = (prim_path, Gf.Vec3d(t0), Gf.Quatd(o0))
        if cams:
            baseline[env_i] = cams
    return baseline


def _vec3_close(a, b, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(3))


def _quat_close(a, b, tol: float = 1e-9) -> bool:
    """Quaternion comparison that tolerates the ``q``/``-q`` double cover
    (two equally valid representations of the same rotation) -- the point
    of the independent readback is whether the composed result is the
    rotation we meant to write, not whether the sign bit is identical."""
    if a is None or b is None:
        return False
    ar, ai = float(a.GetReal()), a.GetImaginary()
    br, bi = float(b.GetReal()), b.GetImaginary()
    same = abs(ar - br) <= tol and all(abs(float(ai[i]) - float(bi[i])) <= tol for i in range(3))
    neg = abs(ar + br) <= tol and all(abs(float(ai[i]) + float(bi[i])) <= tol for i in range(3))
    return same or neg


def _apply_camera_pose_perturbation(stg, camera_pose_baseline: dict, env_id: int,
                                    perturbations: dict) -> "tuple[int, int, Optional[str]]":
    """Applies the per-camera offset for this env (``{camera identifier:
    {x,y,z,roll,pitch,yaw}}``, drawn by
    ``randomize.sampler._draw_camera_pose_plan``) onto this env's camera
    prims under ``/World/envs/env_<env_id>``, relative to
    ``camera_pose_baseline`` (see ``_read_camera_pose_baseline``).

    A camera missing from the baseline, or whose prim/xformOps are gone,
    only gets a WARN and is skipped -- it does not sink the whole
    command.

    **An all-zero perturbation skips ``Set()`` entirely**: the baseline
    pose is already the intended state, so the underlying attribute is
    never touched.

    Non-zero perturbation: translation adds the offset in local space;
    rotation composes a perturbation rotation from ``roll``/``pitch``/
    ``yaw`` (degrees, extrinsic X->Y->Z) and **pre-multiplies** it onto
    the baseline orientation (``rot_new = rot_perturb * rot_original``).
    pxr ``Gf.Rotation``'s multiplication applies the left operand first
    (the opposite of scipy's convention), which is why the composition
    here reads as ``rot_original * (Rx(roll) * Ry(pitch) * Rz(yaw))``.

    After writing, the value is **read back independently** and compared
    against the intended value; a mismatch (a stronger layer in the stage
    overrode the write) does not count towards ``applied`` even though
    ``Set()`` itself raised no exception.

    Returns ``(applied, requested, first_missing_camera)``; ``requested``
    is the number of camera entries received, and ``first_missing_camera``
    identifies the first camera that failed (missing baseline, missing
    prim, or a readback mismatch), or ``None`` if everything succeeded."""
    from pxr import Gf  # noqa: PLC0415

    env_cams = camera_pose_baseline.get(int(env_id)) or {}
    requested = 0
    applied = 0
    first_missing: Optional[str] = None

    def _mark_missing(ident: str) -> None:
        nonlocal first_missing
        if first_missing is None:
            first_missing = ident

    for cam_path, pert in (perturbations or {}).items():
        requested += 1
        entry = env_cams.get(cam_path)
        if entry is None:
            print(f"[render] WARN: randomize camera_pose — no cached pose "
                  f"baseline for camera {cam_path!r} in env {env_id} "
                  f"(missing when the baseline was cached, or a manifest/"
                  f"payload mismatch); skipping", flush=True)
            _mark_missing(f"env_{env_id}:{cam_path}")
            continue
        prim_path, t0, q0 = entry
        pert = pert or {}
        dx = float(pert.get("x", 0.0))
        dy = float(pert.get("y", 0.0))
        dz = float(pert.get("z", 0.0))
        roll = float(pert.get("roll", 0.0))
        pitch = float(pert.get("pitch", 0.0))
        yaw = float(pert.get("yaw", 0.0))
        if dx == 0.0 and dy == 0.0 and dz == 0.0 and roll == 0.0 and pitch == 0.0 and yaw == 0.0:
            # Nothing to do: the baseline is already the intended state.
            applied += 1
            continue
        prim = stg.GetPrimAtPath(prim_path)
        if not (prim and prim.IsValid()):
            print(f"[render] WARN: randomize camera_pose — camera prim "
                  f"{prim_path} existed when the baseline was cached but "
                  f"is gone now (env {env_id}); skipping", flush=True)
            _mark_missing(prim_path)
            continue
        t_attr = prim.GetAttribute("xformOp:translate")
        o_attr = prim.GetAttribute("xformOp:orient")
        if not (t_attr and t_attr.IsValid() and o_attr and o_attr.IsValid()):
            print(f"[render] WARN: randomize camera_pose — camera prim "
                  f"{prim_path} lost its xformOp:translate/xformOp:orient "
                  f"(env {env_id}); skipping", flush=True)
            _mark_missing(prim_path)
            continue
        new_t = Gf.Vec3d(t0) + Gf.Vec3d(dx, dy, dz)
        if roll == 0.0 and pitch == 0.0 and yaw == 0.0:
            new_q = Gf.Quatd(q0)
        else:
            rot0 = Gf.Rotation(Gf.Quatd(q0))
            rot_perturb = (Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), roll)
                          * Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), pitch)
                          * Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw))
            new_q = (rot0 * rot_perturb).GetQuat()
        t_attr.Set(new_t)
        o_attr.Set(new_q)
        # Independent readback -- does not trust "Set() did not raise".
        t_read = t_attr.Get()
        o_read = o_attr.Get()
        if _vec3_close(t_read, new_t) and _quat_close(o_read, new_q):
            applied += 1
        else:
            print(f"[render] WARN: randomize camera_pose — Set() on "
                  f"{prim_path} did not raise, but the independent readback "
                  f"disagrees (read translate={t_read}, orient={o_read}; "
                  f"intended translate={new_t}, orient={new_q}) — a "
                  f"stronger opinion elsewhere in the stage may be winning "
                  f"composition; not counting this camera as applied "
                  f"(env {env_id})", flush=True)
            _mark_missing(prim_path)
    return applied, requested, first_missing


def _apply_randomize_cmd(stg, baseline: dict, camera_pose_baseline: dict,
                         render_payload: dict, manifest_dir, generation: int) -> None:
    """Dispatch point for the render-side command channel: ``"lights"``
    (one shared value for the whole grid), ``"material"`` (independent
    per env, ``{env_id string: bindings}``), and ``"camera_pose"``
    (independent per env, ``{env_id string: {camera identifier:
    {x,y,z,roll,pitch,yaw}}}``). Adding a new render sub-dimension only
    needs a new branch here.

    Writes an ack (``write_randomize_ack``) on both success and failure,
    with ``error`` set on failure. This channel never rebuilds the stage,
    so a failed command simply does not take effect -- the next command
    still computes relative to the same cached baseline.

    ``ok=True`` additionally carries an ``applied`` readback: each apply
    function warns and skips per-prim/per-binding/per-camera failures
    internally, but this function is what counts how many actually
    succeeded, rather than treating "did not raise" as success. The
    bridge side uses this readback to decide whether to raise (nothing
    landed), warn (partially landed), or pass quietly (nothing was
    requested)."""
    try:
        applied: dict = {}
        lights_cmd = (render_payload or {}).get("lights")
        if lights_cmd:
            light_applied, light_requested = _apply_light_perturbation(
                stg, baseline, lights_cmd)
            applied["lights"] = {"applied": light_applied,
                                 "requested": light_requested}
        material_cmd = (render_payload or {}).get("material") or {}
        material_applied: dict = {}
        for env_id_str, bindings in material_cmd.items():
            if bindings:
                m_applied, m_requested, first_missing = _apply_material_bindings(
                    stg, int(env_id_str), bindings)
                material_applied[env_id_str] = {
                    "applied": m_applied, "requested": m_requested,
                    "first_missing_mesh": first_missing,
                }
        if material_applied:
            applied["material"] = material_applied
        camera_pose_cmd = (render_payload or {}).get("camera_pose") or {}
        camera_pose_applied: dict = {}
        for env_id_str, perturbations in camera_pose_cmd.items():
            if perturbations:
                c_applied, c_requested, first_missing_cam = _apply_camera_pose_perturbation(
                    stg, camera_pose_baseline, int(env_id_str), perturbations)
                camera_pose_applied[env_id_str] = {
                    "applied": c_applied, "requested": c_requested,
                    "first_missing_camera": first_missing_cam,
                }
        if camera_pose_applied:
            applied["camera_pose"] = camera_pose_applied
        write_randomize_ack(manifest_dir, generation, ok=True, error=None,
                            applied=applied)
    except Exception as e:  # noqa: BLE001 — node stays alive; ack carries the failure
        print(f"[render] randomize FAILED: generation={generation}: {e!r}",
              flush=True)
        write_randomize_ack(manifest_dir, generation, ok=False, error=repr(e))
