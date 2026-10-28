#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Dynamic / behavioral analysis for one or two MJCFs in pip-installed mujoco.

What this exists for
--------------------

When a MJCF (typically the disposable-solver dump from
``IsaacNewtonEngine``) tracks fine on bringup but oscillates,
swings, or blows up at runtime, the structural diff in
``test_robot_xml_static.py`` can't help — the structure is fine,
the *behaviour* is not.  This tool answers two complementary
questions:

  1. **Single XML — is this model stable on its own?**
     Run a fixed-control rollout in CPU MuJoCo, report per-joint
     amplitude, oscillation count, energy growth.  No reference
     needed.  Useful for "the master gripper joint shakes; does
     a bare CPU-MuJoCo simulation reproduce that, or is the
     wrapper / mjwarp / Newton stack adding the noise"?

  2. **Two XMLs — do they behave the same?**
     Same rollout on both; report per-joint cross-drift in
     addition to per-side stats.  Useful for "the dumped MJCF
     diverges from the reference at runtime, in which joints"?

The tool intentionally uses pip-installed ``mujoco`` (CPU only,
no Newton, no Warp, no Isaac).  That's the point — comparing
against MuJoCo's own integrator isolates Newton/mjwarp's
behaviour.

What it does
------------

Loads the MJCF(s), seeds ``data.qpos`` from ``--init-pose``, sets
``data.ctrl`` to "hold init pose" for AFFINE actuators (so the
zero-disturbance case sits at the operating point), then runs
``mj_step`` for ``--steps × --substeps`` ticks.  Trajectories are
analysed per joint:

  * ``qpos_amplitude`` — peak-to-peak swing (the "is this joint
    wobbling" number)
  * ``qpos_drift_from_init``
  * ``qvel_max_abs``, ``qvel_rms``
  * ``oscillation_count`` — zero crossings of ``qvel - mean(qvel)``;
    3 = quiet decay, 30 = sustained oscillation, 200+ = blow-up
  * ``ke_growth_ratio`` — ``KE(end) / KE(start)`` smoothed over
    a 5-step window; > 10 means kinetic energy is growing without
    bound

Output is a JSON report + a two-section terminal summary:
PER-SIDE STATS first, then CROSS-DRIFT (only when --xml-b given),
then a SUSPICIOUS digest.

Usage
-----

  # Single-XML stability check (default control = hold init pose)
  python3 scripts/tools/test_robot_xml_dynamic.py \\
      --xml /geniesim_assets/scenes/scene_flat_g2_sp/robot_runtime.xml \\
      --steps 200 --substeps 5 \\
      --report-out /tmp/dyn_a.json

  # Same XML but excite the system with a sinusoidal sweep on
  # the master gripper joint
  python3 scripts/tools/test_robot_xml_dynamic.py --xml runtime.xml \\
      --steps 500 --ctrl-mode sweep --sweep-actuator 'position:idx31_gripper_l_inner_joint1' \\
      --sweep-amp 0.3

  # Compare two MJCFs by stepping both with the same ctrl seed
  python3 scripts/tools/test_robot_xml_dynamic.py \\
      --xml-a runtime.xml --xml-b reference.xml \\
      --steps 200 --substeps 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

try:
    import mujoco
    import numpy as np
except ImportError as exc:  # pragma: no cover
    print(
        f"[test_robot_xml_dynamic] ERROR: this tool needs pip-installed `mujoco` "
        f"and `numpy` ({exc}).  Install with `pip install mujoco numpy`.",
        file=sys.stderr,
    )
    sys.exit(2)


# ----------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------


class _Logger:
    def info(self, msg: str) -> None:
        print(f"[test_robot_xml_dynamic] {msg}")

    def warn(self, msg: str) -> None:
        print(f"[test_robot_xml_dynamic] WARN: {msg}")

    def error(self, msg: str) -> None:
        print(f"[test_robot_xml_dynamic] ERROR: {msg}", file=sys.stderr)


# ----------------------------------------------------------------------
# Prefix-strip — same conventions as test_robot_xml_static.py so
# wrapper-namespaced names line up with bare reference names.
# ----------------------------------------------------------------------


def _strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix) :] if prefix and s.startswith(prefix) else s


def _detect_common_prefix_from_model(model: mujoco.MjModel) -> str:
    """Find the longest common leading substring across all joint
    names, trimmed back to a separator boundary.  Used by the
    auto-prefix path — same logic as the static tool's helper.
    """
    names = []
    for j in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n:
            names.append(n)
    if len(names) < 2:
        return ""
    p = names[0]
    for k in names[1:]:
        i = 0
        while i < len(p) and i < len(k) and p[i] == k[i]:
            i += 1
        p = p[:i]
        if not p:
            return ""
    for sep in ("_", "/", "."):
        idx = p.rfind(sep)
        if idx > 0:
            return p[: idx + 1]
    return p if len(p) >= 4 else ""


# ----------------------------------------------------------------------
# Init pose + control seeding
# ----------------------------------------------------------------------


def _parse_init_pose(spec: str) -> Dict[str, float]:
    """Parse ``"name=deg,name=deg,..."`` into ``{joint_name: deg_or_m}``.

    Convention matches scene yamls: HINGE joints in degrees,
    PRISMATIC in metres.  The caller scales to radians per joint
    type at apply time.
    """
    out: Dict[str, float] = {}
    if not spec:
        return out
    for kv in spec.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"--init-pose: expected 'name=value' tokens, got {kv!r}")
        k, v = kv.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def _apply_init_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    init_pose: Dict[str, float],
    strip_prefix: str = "",
) -> List[str]:
    """Write init pose into ``data.qpos`` for joints that exist.

    ``strip_prefix`` lets the same init-pose spec line up with both
    a wrapper-namespaced (``_genie_Physics_idx22_arm_l_joint2``) and
    a bare-name (``idx22_arm_l_joint2``) MJCF — pose names are
    matched after stripping.
    """
    missing: List[str] = []
    # Build name → joint-id map with the strip applied so the init
    # pose works against either namespace.
    jname_to_id: Dict[str, int] = {}
    for j in range(model.njnt):
        raw = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        jname_to_id[_strip_prefix(raw, strip_prefix)] = j
    for jname, val in init_pose.items():
        jid = jname_to_id.get(jname, -1)
        if jid < 0:
            missing.append(jname)
            continue
        jtype = int(model.jnt_type[jid])
        qadr = int(model.jnt_qposadr[jid])
        if jtype == int(mujoco.mjtJoint.mjJNT_HINGE):
            data.qpos[qadr] = math.radians(val)
        elif jtype == int(mujoco.mjtJoint.mjJNT_SLIDE):
            data.qpos[qadr] = val
        else:
            missing.append(f"{jname} (unsupported jnt_type={jtype})")
    return missing


def _seed_ctrl_hold(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """For every position-PD actuator (``biastype == AFFINE``), set
    ``ctrl`` to the joint's current ``qpos`` so the actuator sits at
    zero error.  All other actuators get 0.

    This is the "no disturbance" rollout seed: any motion observed
    afterwards is the model's own behaviour under the static target,
    not a step response transient.
    """
    ctrl = np.zeros(model.nu, dtype=np.float64)
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        jid = int(model.actuator_trnid[a, 0])
        if jid < 0 or jid >= model.njnt:
            continue
        qadr = int(model.jnt_qposadr[jid])
        if int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE):
            ctrl[a] = data.qpos[qadr]
    return ctrl


def _resolve_actuator_id(
    model: mujoco.MjModel,
    name_or_role_target: str,
) -> int:
    """Resolve an actuator name OR a ``"role:joint_name"`` form
    (matching the static tool's keying) to an actuator id.

    Returns -1 if not found.
    """
    # Direct name first.
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name_or_role_target)
    if aid >= 0:
        return aid
    # Fall through to role:joint syntax.
    if ":" not in name_or_role_target:
        return -1
    role, joint_name = name_or_role_target.split(":", 1)
    role = role.strip().lower()
    want_pos = role == "position"
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        jid = int(model.actuator_trnid[a, 0])
        if jid < 0:
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        # Allow either bare or namespaced match.
        if jname != joint_name and not jname.endswith("_" + joint_name) and not jname.endswith(joint_name):
            continue
        is_pos = int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
        if want_pos == is_pos:
            return a
    return -1


# ----------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------


def _step_rollout(
    model: mujoco.MjModel,
    init_pose: Dict[str, float],
    n_steps: int,
    substeps: int,
    ctrl_mode: str,
    sweep_actuator: Optional[str],
    sweep_amp: float,
    sweep_freq_hz: float,
    strip_prefix: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Run ``model`` for ``n_steps × substeps`` mj_steps.

    ``ctrl_mode``:
      * ``"hold"`` — ctrl = init qpos for AFFINE actuators (default).
      * ``"zero"`` — ctrl = 0 everywhere (free fall under gravity).
      * ``"sweep"`` — base = hold; add a sine on ``--sweep-actuator``
        with amplitude ``--sweep-amp`` (rad / m) at ``--sweep-freq-hz``.
      * ``"step"`` — base = hold; at frame 0 add a step of
        ``--sweep-amp`` to ``--sweep-actuator``.
      * ``"ramp"`` — base = hold; linearly ramp ``--sweep-actuator``
        from 0 to ``--sweep-amp`` over the FIRST HALF of the rollout,
        then hold.

    Returns ``(qpos_traj, qvel_traj, ke_traj, missing_init_pose,
    warnings)``:
      * qpos / qvel: shape ``(n_steps + 1, nq|nv)``.
      * ke: shape ``(n_steps + 1,)`` — total kinetic energy
        ``0.5 · qvel² · dof_M0`` (diagonal-mass approximation).
    """
    data = mujoco.MjData(model)
    missing = _apply_init_pose(model, data, init_pose, strip_prefix=strip_prefix)
    mujoco.mj_forward(model, data)

    base_ctrl = _seed_ctrl_hold(model, data) if ctrl_mode != "zero" else np.zeros(model.nu, dtype=np.float64)

    sweep_aid = -1
    sweep_base = 0.0
    warnings: List[str] = []
    if ctrl_mode in ("sweep", "step", "ramp"):
        if not sweep_actuator:
            warnings.append(f"ctrl-mode={ctrl_mode} requires --sweep-actuator; falling back to hold.")
            ctrl_mode = "hold"
        else:
            sweep_aid = _resolve_actuator_id(model, sweep_actuator)
            if sweep_aid < 0:
                warnings.append(f"sweep actuator {sweep_actuator!r} not found; falling back to hold.")
                ctrl_mode = "hold"
            else:
                sweep_base = float(base_ctrl[sweep_aid])

    qpos_traj = np.zeros((n_steps + 1, model.nq), dtype=np.float64)
    qvel_traj = np.zeros((n_steps + 1, model.nv), dtype=np.float64)
    ke_traj = np.zeros(n_steps + 1, dtype=np.float64)
    qpos_traj[0] = data.qpos
    qvel_traj[0] = data.qvel
    ke_traj[0] = 0.0  # qvel starts at 0; sanity floor

    sim_dt_per_substep = float(model.opt.timestep)
    ramp_horizon = max(1, n_steps // 2)
    for s in range(n_steps):
        ctrl = base_ctrl.copy()
        if sweep_aid >= 0:
            if ctrl_mode == "sweep":
                t = s * substeps * sim_dt_per_substep
                ctrl[sweep_aid] = sweep_base + sweep_amp * math.sin(2.0 * math.pi * sweep_freq_hz * t)
            elif ctrl_mode == "step":
                # Apply step from frame 0 onward — the trajectory's
                # first stored point is pre-step, then every
                # post-step frame holds the new target.  Matches
                # control-engineering convention for step-response
                # measurement.
                ctrl[sweep_aid] = sweep_base + sweep_amp
            elif ctrl_mode == "ramp":
                # Linear ramp over the first half of the rollout,
                # then hold for the second half — gives the model
                # enough time to settle after the ramp ends.
                frac = min(1.0, s / float(ramp_horizon))
                ctrl[sweep_aid] = sweep_base + sweep_amp * frac
        data.ctrl[:] = ctrl
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        qpos_traj[s + 1] = data.qpos
        qvel_traj[s + 1] = data.qvel
        ke_traj[s + 1] = 0.5 * float(np.sum(data.qvel * data.qvel * model.dof_M0))
    return qpos_traj, qvel_traj, ke_traj, missing, warnings


# ----------------------------------------------------------------------
# Step / ramp response metrics for a single actuator
# ----------------------------------------------------------------------


def _response_metrics(
    response: np.ndarray,
    qvel_response: np.ndarray,
    initial: float,
    target_delta: float,
    n_steps: int,
    substeps: int,
    sim_dt_per_substep: float,
    settle_band: float = 0.05,
    rise_frac: float = 0.9,
) -> Dict[str, Any]:
    """Compute classical step-response metrics for one DOF's
    trajectory under a step or ramp input.

    Args:
      response: qpos trajectory of the responding joint, length
        ``n_steps + 1``.
      qvel_response: matching qvel trajectory (same length).
      initial: qpos at frame 0 — the pre-step baseline.
      target_delta: commanded change (the step / ramp magnitude).
      n_steps: number of behavioural frames in the rollout.
      substeps: mj_step calls per frame (for time bookkeeping).
      sim_dt_per_substep: ``model.opt.timestep``.
      settle_band: fraction of |target_delta| inside which the
        response must stay for ``settle_time``.
      rise_frac: fraction of the target that defines "risen"
        (default 0.9, i.e. 10–90% rise time).

    Returns metrics dict.  ``rise_time`` / ``settle_time`` are
    in simulated SECONDS, not frames.  Both can be ``None`` if the
    response never reaches the threshold within the rollout — the
    UI surface that as "not reached".
    """
    target = initial + target_delta
    dt_per_frame = substeps * sim_dt_per_substep
    direction = 1.0 if target_delta >= 0 else -1.0
    progress = (response - initial) * direction
    full = abs(target_delta)

    # Rise time: first frame where progress >= rise_frac × full.
    rise_time: Optional[float] = None
    if full > 0:
        threshold = rise_frac * full
        crossed = np.where(progress >= threshold)[0]
        if crossed.size:
            rise_time = float(crossed[0] * dt_per_frame)

    # Settle time: last frame where |response - target| > band.
    # Time-after-that is the "stayed-in-band" window.  If the
    # response never leaves the band, settle_time = 0.  If it
    # never enters the band, settle_time = None.
    settle_time: Optional[float] = None
    if full > 0:
        band = settle_band * full
        outside = np.where(np.abs(response - target) > band)[0]
        if outside.size == 0:
            settle_time = 0.0
        elif int(outside[-1]) < n_steps:
            settle_time = float((int(outside[-1]) + 1) * dt_per_frame)

    # Overshoot: max excursion past the target IN THE COMMANDED
    # DIRECTION, as a fraction of full.  Undershoot would show up
    # as negative.
    overshoot_frac = 0.0
    if full > 0:
        if direction > 0:
            peak = float(np.max(response))
        else:
            peak = float(np.min(response))
        overshoot_frac = max(0.0, (peak - target) * direction / full)

    steady_state_error = float(response[-1] - target)

    return {
        "initial": float(initial),
        "target": float(target),
        "target_delta": float(target_delta),
        "final": float(response[-1]),
        "rise_time_s": rise_time,
        "settle_time_s": settle_time,
        "overshoot_frac": float(overshoot_frac),
        "steady_state_error": steady_state_error,
        "steady_state_error_frac": float(steady_state_error / full) if full > 0 else 0.0,
        "qvel_max_abs": float(np.max(np.abs(qvel_response))),
        "oscillation_count": _zero_crossings(qvel_response),
    }


def _list_position_actuators(
    model: mujoco.MjModel,
    strip_prefix: str = "",
) -> List[Tuple[int, str, int]]:
    """Return ``[(actuator_id, joint_name_stripped, qpos_addr), ...]``
    for every AFFINE position actuator that drives a HINGE / SLIDE
    joint.  Survey mode iterates over this list.
    """
    out: List[Tuple[int, str, int]] = []
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        if int(model.actuator_biastype[a]) != int(mujoco.mjtBias.mjBIAS_AFFINE):
            continue
        jid = int(model.actuator_trnid[a, 0])
        if jid < 0 or jid >= model.njnt:
            continue
        jtype = int(model.jnt_type[jid])
        if jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        raw_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"<joint#{jid}>"
        out.append((a, _strip_prefix(raw_name, strip_prefix), int(model.jnt_qposadr[jid])))
    return out


def _survey_responses(
    model: mujoco.MjModel,
    init_pose: Dict[str, float],
    n_steps: int,
    substeps: int,
    survey_mode: str,
    survey_amp: float,
    only_joints: Optional[List[str]] = None,
    skip_joints: Optional[List[str]] = None,
    strip_prefix: str = "",
) -> List[Dict[str, Any]]:
    """Run one rollout per position actuator, applying a step
    (``survey_mode="step"``) or ramp (``survey_mode="ramp"``) of
    magnitude ``survey_amp`` to that actuator while the rest hold
    init pose.

    Returns a list of per-actuator metrics dicts.  Each dict carries:

      * ``actuator_id`` / ``joint_name`` / ``qpos_addr``
      * the step input (``initial``, ``target``, ``target_delta``)
      * response metrics from ``_response_metrics``
      * the *driven* joint's stats; coupled-joint stats are NOT
        broken out here to keep the table compact — query the JSON
        for the full per-joint trace if needed.
    """
    sim_dt_per_substep = float(model.opt.timestep)
    rows: List[Dict[str, Any]] = []
    only_set = set(only_joints) if only_joints else None
    skip_set = set(skip_joints) if skip_joints else set()
    for aid, jname_stripped, qadr in _list_position_actuators(model, strip_prefix=strip_prefix):
        if only_set is not None and jname_stripped not in only_set:
            continue
        if jname_stripped in skip_set:
            continue
        qpos_traj, qvel_traj, ke_traj, _, _ = _step_rollout(
            model,
            init_pose,
            n_steps,
            substeps,
            ctrl_mode=survey_mode,
            # The rollout looks ``sweep_actuator`` up via the
            # ``role:joint_name`` form so it works with un-named
            # actuators (Newton's mjwarp converter doesn't author
            # actuator names).  ``position:<stripped>`` matches the
            # static tool's keying.
            sweep_actuator=f"position:{jname_stripped}",
            sweep_amp=survey_amp,
            sweep_freq_hz=0.0,
            strip_prefix=strip_prefix,
        )
        initial = float(qpos_traj[0, qadr])
        response = qpos_traj[:, qadr]
        qvel_response = qvel_traj[:, int(model.jnt_dofadr[int(model.actuator_trnid[aid, 0])])]
        metrics = _response_metrics(
            response,
            qvel_response,
            initial=initial,
            target_delta=survey_amp,
            n_steps=n_steps,
            substeps=substeps,
            sim_dt_per_substep=sim_dt_per_substep,
        )
        rows.append(
            {
                "actuator_id": aid,
                "joint_name": jname_stripped,
                "qpos_addr": qadr,
                "mode": survey_mode,
                **metrics,
                # KE drift over the rollout — flag if a joint's
                # step input pumps energy into the whole system
                # (typically means coupled instability further down
                # the chain).
                "ke_start": float(np.mean(ke_traj[:5])),
                "ke_end": float(np.mean(ke_traj[-5:])),
                "ke_ratio": float(np.mean(ke_traj[-5:]) / max(np.mean(ke_traj[:5]), 1e-9)),
            }
        )
    return rows


def _cross_impact_survey(
    model: mujoco.MjModel,
    init_pose: Dict[str, float],
    n_steps: int,
    substeps: int,
    cross_mode: str,
    sources: List[str],
    targets: List[str],
    cross_amp: float,
    cross_freq_hz: float,
    strip_prefix: str = "",
) -> List[Dict[str, Any]]:
    """Per-source × per-target coupling matrix.

    For each joint in ``sources``, run a dedicated rollout that
    drives that source (sweep / step / ramp on its position
    actuator), then record the response stats of every joint in
    ``targets``.  Output is one row per (source, target) pair so
    the printer can pivot into a matrix without much code.

    Why this exists: the per-side stats already surface "this
    joint is wobbling", but they don't tell you WHY.  Cross-impact
    answers the next question — "which source joints are driving
    which target joints' motion".  The matrix view makes it
    obvious whether a gripper swing is sourced in the arm shoulder
    or the body, and how strong each coupling is.

    Each row carries:

      * ``source`` / ``target`` joint names (stripped)
      * ``source_amp`` — actual peak-to-peak motion of the source
        DOF (sanity check that the drive actually moved the source)
      * ``target_amp`` — peak-to-peak qpos amplitude of the target
      * ``target_drift`` — final qpos minus initial qpos of the target
      * ``target_qvel_max`` — peak speed of the target during the rollout
      * ``target_osc`` — zero-crossings of target qvel
      * ``coupling_ratio`` — ``target_amp / max(source_amp, 1e-9)``
        — the headline number, makes cross-source-target ranking
        straightforward.  > 1.0 means the target swings MORE than
        the source.
    """
    sim_dt_per_substep = float(model.opt.timestep)

    # Resolve every target's qpos / dof index up front.  Skip
    # non-single-DOF joints (FREE / BALL) — out of scope for the
    # simple matrix view.
    target_idx: Dict[str, Tuple[int, int]] = {}
    for tname in targets:
        # Match both stripped and namespaced names — the user might
        # pass either.
        jid = -1
        for j in range(model.njnt):
            raw = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if raw == tname or _strip_prefix(raw, strip_prefix) == tname:
                jid = j
                break
        if jid < 0:
            continue
        jtype = int(model.jnt_type[jid])
        if jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        target_idx[tname] = (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))

    rows: List[Dict[str, Any]] = []
    for src in sources:
        aid = _resolve_actuator_id(model, f"position:{src}")
        if aid < 0:
            continue
        src_jid = int(model.actuator_trnid[aid, 0])
        if src_jid < 0 or src_jid >= model.njnt:
            continue
        src_qadr = int(model.jnt_qposadr[src_jid])
        qpos_traj, qvel_traj, _, _, _ = _step_rollout(
            model,
            init_pose,
            n_steps,
            substeps,
            ctrl_mode=cross_mode,
            sweep_actuator=f"position:{src}",
            sweep_amp=cross_amp,
            sweep_freq_hz=cross_freq_hz,
            strip_prefix=strip_prefix,
        )
        src_qpos = qpos_traj[:, src_qadr]
        src_amp = float(np.max(src_qpos) - np.min(src_qpos))
        for tname, (qadr, vadr) in target_idx.items():
            t_qpos = qpos_traj[:, qadr]
            t_qvel = qvel_traj[:, vadr]
            t_amp = float(np.max(t_qpos) - np.min(t_qpos))
            rows.append(
                {
                    "source": src,
                    "target": tname,
                    "mode": cross_mode,
                    "source_amp": src_amp,
                    "target_amp": t_amp,
                    "target_drift": float(t_qpos[-1] - t_qpos[0]),
                    "target_qvel_max": float(np.max(np.abs(t_qvel))),
                    "target_osc": _zero_crossings(t_qvel),
                    "coupling_ratio": float(t_amp / max(src_amp, 1e-9)),
                    "self": (src == tname),
                }
            )
    return rows


# ----------------------------------------------------------------------
# Per-joint statistics
# ----------------------------------------------------------------------


def _zero_crossings(arr: np.ndarray) -> int:
    """Count sign changes of (arr - mean(arr)).  Stable oscillation
    indicator that doesn't fire on monotonic drift.
    """
    centered = arr - float(np.mean(arr))
    signs = np.sign(centered)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.sum(signs[:-1] != signs[1:]))


def _per_axis_stats(qpos_col: np.ndarray, qvel_col: np.ndarray) -> Dict[str, float]:
    return {
        "qpos_initial": float(qpos_col[0]),
        "qpos_final": float(qpos_col[-1]),
        "qpos_min": float(np.min(qpos_col)),
        "qpos_max": float(np.max(qpos_col)),
        "qpos_amplitude": float(np.max(qpos_col) - np.min(qpos_col)),
        "qpos_drift_from_init": float(qpos_col[-1] - qpos_col[0]),
        "qvel_max_abs": float(np.max(np.abs(qvel_col))),
        "qvel_rms": float(np.sqrt(np.mean(qvel_col * qvel_col))),
        "oscillation_count": _zero_crossings(qvel_col),
    }


def _per_joint_stats(
    model: mujoco.MjModel,
    qpos_traj: np.ndarray,
    qvel_traj: np.ndarray,
    strip_prefix: str = "",
) -> List[Dict[str, Any]]:
    """One stats dict per single-DOF joint.  Multi-DOF joints are
    skipped with a note (would need quaternion / per-axis
    decomposition).
    """
    rows: List[Dict[str, Any]] = []
    for j in range(model.njnt):
        raw = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"<joint#{j}>"
        name = _strip_prefix(raw, strip_prefix)
        jtype = int(model.jnt_type[j])
        if jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            rows.append({"name": name, "skipped": True, "reason": f"non-scalar jnt_type={jtype}"})
            continue
        qadr = int(model.jnt_qposadr[j])
        vadr = int(model.jnt_dofadr[j])
        rows.append({"name": name, **_per_axis_stats(qpos_traj[:, qadr], qvel_traj[:, vadr])})
    return rows


def _ke_growth(ke_traj: np.ndarray) -> Tuple[float, float, float]:
    """KE diagnostic: returns ``(start, end, ratio)`` smoothed over
    a 5-step window so a single-step transient doesn't dominate.

    Ratio > 10 is a strong instability signal — the system is
    pumping energy from the actuators / numerical integration into
    the joints faster than dissipation can remove it.
    """
    if ke_traj.size < 6:
        return float(ke_traj[0]) if ke_traj.size else 0.0, float(ke_traj[-1]) if ke_traj.size else 0.0, 0.0
    win = 5
    start = float(np.mean(ke_traj[:win]))
    end = float(np.mean(ke_traj[-win:]))
    ratio = end / max(start, 1e-9)
    return start, end, ratio


# ----------------------------------------------------------------------
# Cross-model trajectory diff (only when --xml-b is given)
# ----------------------------------------------------------------------


def _trajectory_cross_diff(
    model_a: mujoco.MjModel,
    qpos_a: np.ndarray,
    qvel_a: np.ndarray,
    model_b: mujoco.MjModel,
    qpos_b: np.ndarray,
    qvel_b: np.ndarray,
    strip_prefix_a: str = "",
    strip_prefix_b: str = "",
) -> Dict[str, Any]:
    """Match joints across models by stripped name and report
    per-joint max ``qpos`` / ``qvel`` drift between A and B.
    """

    def _ids(model: mujoco.MjModel, prefix: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for j in range(model.njnt):
            raw = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            out[_strip_prefix(raw, prefix)] = j
        return out

    names_a = _ids(model_a, strip_prefix_a)
    names_b = _ids(model_b, strip_prefix_b)
    shared = sorted(set(names_a) & set(names_b))
    per_joint: List[Dict[str, Any]] = []
    g_q = 0.0
    g_v = 0.0
    for name in shared:
        ja = names_a[name]
        jb = names_b[name]
        type_a = int(model_a.jnt_type[ja])
        type_b = int(model_b.jnt_type[jb])
        if type_a != type_b or type_a not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        qa = qpos_a[:, int(model_a.jnt_qposadr[ja])]
        qb = qpos_b[:, int(model_b.jnt_qposadr[jb])]
        va = qvel_a[:, int(model_a.jnt_dofadr[ja])]
        vb = qvel_b[:, int(model_b.jnt_dofadr[jb])]
        dq = float(np.max(np.abs(qa - qb)))
        dv = float(np.max(np.abs(va - vb)))
        per_joint.append({"name": name, "max_qpos_drift": dq, "max_qvel_drift": dv})
        g_q = max(g_q, dq)
        g_v = max(g_v, dv)
    return {
        "per_joint": per_joint,
        "global_max_qpos_drift": g_q,
        "global_max_qvel_drift": g_v,
        "shared_dof_count": len(shared),
        "unmatched_a": sorted(set(names_a) - set(names_b)),
        "unmatched_b": sorted(set(names_b) - set(names_a)),
    }


# ----------------------------------------------------------------------
# Suspicion heuristics — dynamic only
# ----------------------------------------------------------------------

OSC_HIGH = 50  # zero-crossings; sustained-oscillation threshold
OSC_BLOWUP = 200
AMP_THRESHOLD = 0.05  # rad / m, ≈3° for a hinge — "actively swinging"
KE_GROWTH_HIGH = 5.0
KE_GROWTH_BLOWUP = 50.0


def _flag_suspicious(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Distill per-side and cross-drift findings into a short list of
    callouts.  Severity is ordered critical > high > medium so an
    AI consumer can pick the top few entries to act on.
    """
    flags: List[Dict[str, Any]] = []

    for side in ("a", "b"):
        side_data = report.get(side)
        if side_data is None:
            continue
        # Energy growth — fast indicator of integration blow-up.
        ke = side_data.get("ke", {})
        ratio = float(ke.get("growth_ratio", 0.0))
        if ratio >= KE_GROWTH_BLOWUP:
            flags.append(
                {
                    "severity": "critical",
                    "kind": f"ke_growth_{side}",
                    "msg": (
                        f"side {side.upper()}: kinetic energy grew "
                        f"{ratio:.2f}× across the rollout — numerical "
                        f"blow-up.  Drop --substeps or fix model "
                        f"actuator gains."
                    ),
                }
            )
        elif ratio >= KE_GROWTH_HIGH:
            flags.append(
                {
                    "severity": "high",
                    "kind": f"ke_growth_{side}",
                    "msg": (
                        f"side {side.upper()}: kinetic energy grew "
                        f"{ratio:.2f}× across the rollout — sustained "
                        f"oscillation pumping energy in."
                    ),
                }
            )

        # Top-noise joints by amplitude.
        noisy = sorted(
            (j for j in side_data.get("per_joint", []) if not j.get("skipped")),
            key=lambda j: -j.get("qpos_amplitude", 0.0),
        )
        for j in noisy[:5]:
            amp = float(j.get("qpos_amplitude", 0.0))
            if amp < AMP_THRESHOLD:
                break
            sev = "high" if amp > 0.3 else "medium"
            osc = int(j.get("oscillation_count", 0))
            flags.append(
                {
                    "severity": sev,
                    "kind": f"amplitude_{side}",
                    "name": j["name"],
                    "msg": (
                        f"side {side.upper()}: joint {j['name']!r} qpos "
                        f"peak-to-peak {amp:.4f} rad/m, "
                        f"oscillation count {osc}."
                    ),
                }
            )

        # Sustained-oscillation flag: any joint with osc >= threshold.
        max_osc = max(
            (int(j.get("oscillation_count", 0)) for j in side_data.get("per_joint", []) if not j.get("skipped")),
            default=0,
        )
        if max_osc >= OSC_BLOWUP:
            flags.append(
                {
                    "severity": "critical",
                    "kind": f"oscillation_{side}",
                    "msg": (
                        f"side {side.upper()}: at least one joint crossed "
                        f"zero velocity {max_osc} times across the rollout "
                        f"— numerical blow-up territory."
                    ),
                }
            )
        elif max_osc >= OSC_HIGH:
            flags.append(
                {
                    "severity": "high",
                    "kind": f"oscillation_{side}",
                    "msg": (
                        f"side {side.upper()}: at least one joint crossed "
                        f"zero velocity {max_osc} times across the rollout "
                        f"— sustained oscillation; check actuator force "
                        f"cap vs. damping."
                    ),
                }
            )

    cross = report.get("cross")
    if cross is not None:
        gq = float(cross.get("global_max_qpos_drift", 0.0))
        gv = float(cross.get("global_max_qvel_drift", 0.0))
        if gq > 1e-2:
            flags.append(
                {
                    "severity": "high" if gq > 0.1 else "medium",
                    "kind": "cross.qpos_drift",
                    "msg": (
                        f"trajectory max qpos drift A↔B = {gq:.4f} across "
                        f"{cross.get('shared_dof_count', '?')} matched DOFs."
                    ),
                }
            )
        if gv > 1.0:
            flags.append(
                {
                    "severity": "high" if gv > 10.0 else "medium",
                    "kind": "cross.qvel_drift",
                    "msg": f"trajectory max qvel drift A↔B = {gv:.4f}.",
                }
            )
    return flags


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------


def _build_report(
    xml_a: str,
    xml_b: Optional[str],
    n_steps: int,
    substeps: int,
    init_pose_spec: str,
    ctrl_mode: str,
    sweep_actuator: Optional[str],
    sweep_amp: float,
    sweep_freq_hz: float,
    logger: _Logger,
    strip_prefix_a: str = "",
    strip_prefix_b: str = "",
    auto_strip_prefix: bool = True,
    survey_mode: Optional[str] = None,
    survey_amp: float = 0.1,
    survey_only: Optional[List[str]] = None,
    survey_skip: Optional[List[str]] = None,
    cross_mode: Optional[str] = None,
    cross_sources: Optional[List[str]] = None,
    cross_targets: Optional[List[str]] = None,
    cross_amp: float = 0.3,
    cross_freq_hz: float = 0.5,
) -> Dict[str, Any]:
    """Run rollout(s) and return the report dict.

    Single-XML mode: only ``a`` block in the report.
    Two-XML mode: ``a`` + ``b`` + ``cross`` blocks.

    When ``survey_mode`` is set to ``"step"`` or ``"ramp"``, a
    separate per-actuator rollout is run for every position actuator
    and the resulting response metrics land in ``a.survey`` /
    ``b.survey``.

    When ``cross_mode`` is set to ``"sweep"`` / ``"step"`` / ``"ramp"``,
    one rollout per source joint runs with that source driven, and
    every joint in ``cross_targets`` has its response stats
    recorded.  Lands in ``a.cross_impact`` (and ``b.cross_impact``).
    Use this to map "if I move arm joint X, which gripper joints
    swing how much".
    """
    init_pose = _parse_init_pose(init_pose_spec)

    logger.info(f"loading A: {xml_a}")
    model_a = mujoco.MjModel.from_xml_path(xml_a)
    if auto_strip_prefix and not strip_prefix_a:
        strip_prefix_a = _detect_common_prefix_from_model(model_a)
        if strip_prefix_a:
            logger.info(f"auto-detected strip_prefix_a={strip_prefix_a!r}")

    model_b: Optional[mujoco.MjModel] = None
    if xml_b:
        logger.info(f"loading B: {xml_b}")
        model_b = mujoco.MjModel.from_xml_path(xml_b)
        if auto_strip_prefix and not strip_prefix_b:
            strip_prefix_b = _detect_common_prefix_from_model(model_b)
            if strip_prefix_b:
                logger.info(f"auto-detected strip_prefix_b={strip_prefix_b!r}")

    summary = {
        "xml_a": os.path.abspath(xml_a),
        "xml_b": os.path.abspath(xml_b) if xml_b else None,
        "n_steps": n_steps,
        "substeps": substeps,
        "ctrl_mode": ctrl_mode,
        "sweep_actuator": sweep_actuator,
        "sweep_amp": sweep_amp,
        "sweep_freq_hz": sweep_freq_hz,
        "strip_prefix_a": strip_prefix_a,
        "strip_prefix_b": strip_prefix_b,
        "init_pose": init_pose,
        "survey_mode": survey_mode,
        "survey_amp": survey_amp,
        "cross_mode": cross_mode,
        "cross_amp": cross_amp,
        "cross_freq_hz": cross_freq_hz,
        "cross_sources": cross_sources,
        "cross_targets": cross_targets,
    }
    report: Dict[str, Any] = {"summary": summary}

    logger.info(f"rollout A: {n_steps} steps × {substeps} substeps " f"({n_steps * substeps} mj_step calls)")
    qpos_a, qvel_a, ke_a, miss_a, warns_a = _step_rollout(
        model_a,
        init_pose,
        n_steps,
        substeps,
        ctrl_mode=ctrl_mode,
        sweep_actuator=sweep_actuator,
        sweep_amp=sweep_amp,
        sweep_freq_hz=sweep_freq_hz,
        strip_prefix=strip_prefix_a,
    )
    for w in warns_a:
        logger.warn(f"A: {w}")
    ke0_a, ke1_a, ke_ratio_a = _ke_growth(ke_a)
    report["a"] = {
        "per_joint": _per_joint_stats(model_a, qpos_a, qvel_a, strip_prefix=strip_prefix_a),
        "ke": {"start_smoothed": ke0_a, "end_smoothed": ke1_a, "growth_ratio": ke_ratio_a},
        "init_pose_missing": miss_a,
    }
    if survey_mode in ("step", "ramp"):
        n_act = len(_list_position_actuators(model_a, strip_prefix=strip_prefix_a))
        logger.info(
            f"survey A: {survey_mode} response on every position actuator "
            f"({n_act} actuator(s) × {n_steps * substeps} mj_step calls each)"
        )
        report["a"]["survey"] = _survey_responses(
            model_a,
            init_pose,
            n_steps,
            substeps,
            survey_mode=survey_mode,
            survey_amp=survey_amp,
            only_joints=survey_only,
            skip_joints=survey_skip,
            strip_prefix=strip_prefix_a,
        )
    if cross_mode in ("sweep", "step", "ramp") and cross_sources and cross_targets:
        logger.info(
            f"cross-impact A: {cross_mode} drive on {len(cross_sources)} source(s) "
            f"× {len(cross_targets)} target(s)  ({n_steps * substeps} mj_step calls per source)"
        )
        report["a"]["cross_impact"] = _cross_impact_survey(
            model_a,
            init_pose,
            n_steps,
            substeps,
            cross_mode=cross_mode,
            sources=cross_sources,
            targets=cross_targets,
            cross_amp=cross_amp,
            cross_freq_hz=cross_freq_hz,
            strip_prefix=strip_prefix_a,
        )

    if model_b is not None and xml_b is not None:
        logger.info(f"rollout B: {n_steps} steps × {substeps} substeps " f"({n_steps * substeps} mj_step calls)")
        qpos_b, qvel_b, ke_b, miss_b, warns_b = _step_rollout(
            model_b,
            init_pose,
            n_steps,
            substeps,
            ctrl_mode=ctrl_mode,
            sweep_actuator=sweep_actuator,
            sweep_amp=sweep_amp,
            sweep_freq_hz=sweep_freq_hz,
            strip_prefix=strip_prefix_b,
        )
        for w in warns_b:
            logger.warn(f"B: {w}")
        ke0_b, ke1_b, ke_ratio_b = _ke_growth(ke_b)
        report["b"] = {
            "per_joint": _per_joint_stats(model_b, qpos_b, qvel_b, strip_prefix=strip_prefix_b),
            "ke": {"start_smoothed": ke0_b, "end_smoothed": ke1_b, "growth_ratio": ke_ratio_b},
            "init_pose_missing": miss_b,
        }
        if survey_mode in ("step", "ramp"):
            n_act = len(_list_position_actuators(model_b, strip_prefix=strip_prefix_b))
            logger.info(
                f"survey B: {survey_mode} response on every position actuator "
                f"({n_act} actuator(s) × {n_steps * substeps} mj_step calls each)"
            )
            report["b"]["survey"] = _survey_responses(
                model_b,
                init_pose,
                n_steps,
                substeps,
                survey_mode=survey_mode,
                survey_amp=survey_amp,
                only_joints=survey_only,
                skip_joints=survey_skip,
                strip_prefix=strip_prefix_b,
            )
        if cross_mode in ("sweep", "step", "ramp") and cross_sources and cross_targets:
            logger.info(
                f"cross-impact B: {cross_mode} drive on {len(cross_sources)} source(s) × {len(cross_targets)} target(s)"
            )
            report["b"]["cross_impact"] = _cross_impact_survey(
                model_b,
                init_pose,
                n_steps,
                substeps,
                cross_mode=cross_mode,
                sources=cross_sources,
                targets=cross_targets,
                cross_amp=cross_amp,
                cross_freq_hz=cross_freq_hz,
                strip_prefix=strip_prefix_b,
            )
        report["cross"] = _trajectory_cross_diff(
            model_a,
            qpos_a,
            qvel_a,
            model_b,
            qpos_b,
            qvel_b,
            strip_prefix_a=strip_prefix_a,
            strip_prefix_b=strip_prefix_b,
        )

    report["suspicious"] = _flag_suspicious(report)
    return report


# ----------------------------------------------------------------------
# Pretty-printer
# ----------------------------------------------------------------------


def _print_summary(report: Dict[str, Any], logger: _Logger) -> None:
    s = report["summary"]
    logger.info("=" * 72)
    logger.info(f"A: {s['xml_a']}")
    if s.get("xml_b"):
        logger.info(f"B: {s['xml_b']}")
    logger.info(f"rollout:  {s['n_steps']} steps × {s['substeps']} substeps  " f"ctrl_mode={s['ctrl_mode']!r}")
    if s["ctrl_mode"] == "sweep" and s.get("sweep_actuator"):
        logger.info(
            f"sweep:    actuator={s['sweep_actuator']!r}  " f"amp={s['sweep_amp']}  freq={s['sweep_freq_hz']} Hz"
        )

    for side in ("a", "b"):
        side_data = report.get(side)
        if side_data is None:
            continue
        logger.info("-" * 72)
        logger.info(f"PER-SIDE {side.upper()}")
        logger.info("-" * 72)
        ke = side_data.get("ke", {})
        logger.info(
            f"KE:  start={ke.get('start_smoothed', 0):.4g}  "
            f"end={ke.get('end_smoothed', 0):.4g}  "
            f"growth={ke.get('growth_ratio', 0):.2f}×"
        )
        per = sorted(
            (j for j in side_data.get("per_joint", []) if not j.get("skipped")),
            key=lambda j: -j.get("qpos_amplitude", 0.0),
        )
        logger.info("top-10 noisiest joints (qpos peak-to-peak):")
        for j in per[:10]:
            logger.info(
                f"  {j['name']:50s}  amp={j.get('qpos_amplitude', 0):.4f}  "
                f"osc={j.get('oscillation_count', 0):3d}  "
                f"|qvel|max={j.get('qvel_max_abs', 0):.3f}  "
                f"drift={j.get('qpos_drift_from_init', 0):+.4f}"
            )

        # Survey block: one row per position actuator with response
        # metrics from its dedicated step / ramp rollout.
        survey = side_data.get("survey")
        if survey:
            mode_label = (report.get("summary", {}).get("survey_mode") or "?").upper()
            amp_label = report.get("summary", {}).get("survey_amp", "?")
            logger.info(f"survey ({mode_label} response, Δ={amp_label}):")
            logger.info(
                f"  {'joint':50s}  {'rise':>7s}  {'settle':>7s}  "
                f"{'over%':>6s}  {'ss_err':>9s}  {'osc':>3s}  {'KE×':>5s}"
            )
            for row in survey:
                rise = row.get("rise_time_s")
                settle = row.get("settle_time_s")
                logger.info(
                    f"  {row['joint_name']:50s}  "
                    f"{('—' if rise is None else f'{rise:.3f}s'):>7s}  "
                    f"{('—' if settle is None else f'{settle:.3f}s'):>7s}  "
                    f"{row.get('overshoot_frac', 0) * 100:>5.1f}%  "
                    f"{row.get('steady_state_error', 0):>+9.4f}  "
                    f"{row.get('oscillation_count', 0):>3d}  "
                    f"{row.get('ke_ratio', 0):>5.2f}"
                )

        # Cross-impact block: source × target coupling matrix.
        # Sort by coupling_ratio so the most problematic
        # (source, target) pairs appear first.  Self-pairs
        # (source == target) are excluded from the headline list
        # but kept in the JSON so the consumer can see how the
        # source itself moved.
        cross = side_data.get("cross_impact")
        if cross:
            mode_label = (report.get("summary", {}).get("cross_mode") or "?").upper()
            amp_label = report.get("summary", {}).get("cross_amp", "?")
            freq_label = report.get("summary", {}).get("cross_freq_hz", "?")
            mode_suffix = f"amp={amp_label}"
            if mode_label.lower() == "sweep":
                mode_suffix += f" freq={freq_label}Hz"
            logger.info(f"cross-impact ({mode_label} drive, {mode_suffix}):")
            logger.info(
                f"  {'source':38s}  {'target':38s}  "
                f"{'src_amp':>8s}  {'tgt_amp':>8s}  {'coup×':>6s}  "
                f"{'drift':>8s}  {'qvel':>7s}  {'osc':>3s}"
            )
            ranked = sorted(
                (r for r in cross if not r.get("self")),
                key=lambda r: -r.get("coupling_ratio", 0.0),
            )
            n_show = min(20, len(ranked))
            for r in ranked[:n_show]:
                logger.info(
                    f"  {r['source']:38s}  {r['target']:38s}  "
                    f"{r.get('source_amp', 0):>8.4f}  "
                    f"{r.get('target_amp', 0):>8.4f}  "
                    f"{r.get('coupling_ratio', 0):>5.2f}×  "
                    f"{r.get('target_drift', 0):>+8.4f}  "
                    f"{r.get('target_qvel_max', 0):>7.3f}  "
                    f"{r.get('target_osc', 0):>3d}"
                )
            if len(ranked) > n_show:
                logger.info(f"  ... and {len(ranked) - n_show} more (see JSON for full matrix)")

    cross = report.get("cross")
    if cross is not None:
        logger.info("-" * 72)
        logger.info("CROSS A↔B")
        logger.info("-" * 72)
        logger.info(
            f"shared DOFs: {cross.get('shared_dof_count', 0)}  "
            f"unmatched_a={len(cross.get('unmatched_a', []))}  "
            f"unmatched_b={len(cross.get('unmatched_b', []))}"
        )
        logger.info(
            f"max qpos drift = {cross.get('global_max_qpos_drift', 0):.6f}  "
            f"max qvel drift = {cross.get('global_max_qvel_drift', 0):.6f}"
        )
        worst = sorted(cross.get("per_joint", []), key=lambda x: -x.get("max_qpos_drift", 0))[:5]
        if worst:
            logger.info("top-5 most-divergent joints:")
            for j in worst:
                logger.info(
                    f"  {j['name']:50s}  qpos_drift={j['max_qpos_drift']:.4f}  " f"qvel_drift={j['max_qvel_drift']:.4f}"
                )

    flags = report.get("suspicious", [])
    logger.info("-" * 72)
    if not flags:
        logger.info("SUSPICIOUS: 0 — nothing stands out.  See JSON for full per-joint table.")
    else:
        logger.info(f"SUSPICIOUS: {len(flags)} flag(s)")
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        for f in sorted(flags, key=lambda x: sev_order.get(x.get("severity", "low"), 9)):
            logger.info(f"  [{f.get('severity', '?'):8s}] {f.get('kind', '?')}: {f.get('msg', '')}")
    logger.info("=" * 72)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Backwards-compatible interface: --xml is shorthand for --xml-a;
    # --xml-b makes it a comparison run.
    parser.add_argument(
        "--xml-a",
        "--xml",
        dest="xml_a",
        required=True,
        help="Path to MJCF A (the model under test).",
    )
    parser.add_argument(
        "--xml-b",
        default="",
        help=(
            "Path to MJCF B for a two-model comparison.  Empty (default) "
            "runs single-XML stability mode and only the ``a`` block "
            "appears in the report."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of behavioural rollout frames per model (default: %(default)s).",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=5,
        help="MuJoCo ``mj_step`` calls per behavioural frame (default: %(default)s).",
    )
    parser.add_argument(
        "--init-pose",
        default="",
        help='Comma-separated "name=value" overrides for ``data.qpos`` (degrees for hinge, metres for slide).',
    )
    parser.add_argument(
        "--ctrl-mode",
        choices=("hold", "zero", "sweep", "step", "ramp"),
        default="hold",
        help=(
            "How to seed ``data.ctrl`` for the MAIN rollout (separate "
            "from ``--survey-mode`` which runs N additional dedicated "
            "rollouts).  ``hold`` (default) sets ctrl = init qpos for "
            "AFFINE actuators.  ``zero`` sets ctrl = 0.  ``sweep`` adds "
            "a sinusoid on --sweep-actuator on top of hold.  ``step`` "
            "applies an immediate step of --sweep-amp.  ``ramp`` linearly "
            "ramps to --sweep-amp over the first half of the rollout."
        ),
    )
    parser.add_argument(
        "--survey-mode",
        choices=("none", "step", "ramp"),
        default="none",
        help=(
            "When set to ``step`` or ``ramp``, run one DEDICATED rollout "
            "per position actuator with a step / ramp input on JUST that "
            "actuator (other joints hold init pose).  Per-actuator "
            "response metrics (rise time, settle time, overshoot, "
            "steady-state error, oscillation count, KE coupling ratio) "
            "land in the report's ``a.survey`` (and ``b.survey``) "
            "blocks.  Default ``none`` skips the survey."
        ),
    )
    parser.add_argument(
        "--survey-amp",
        type=float,
        default=0.1,
        help=(
            "Step / ramp magnitude for the survey (rad for hinge, m for "
            "slide).  Default %(default)s = ~5.7°.  Pick small enough "
            "not to drive joints past their limits but large enough to "
            "give a measurable response."
        ),
    )
    parser.add_argument(
        "--survey-only",
        default="",
        help=(
            "Comma-separated joint-name allowlist for the survey.  "
            "Empty (default) = include every position actuator."
        ),
    )
    parser.add_argument(
        "--survey-skip",
        default="",
        help="Comma-separated joint-name skiplist for the survey.",
    )
    parser.add_argument(
        "--cross-impact-mode",
        choices=("none", "sweep", "step", "ramp"),
        default="none",
        help=(
            "Run a cross-impact analysis.  For each joint in "
            "``--cross-sources``, one rollout drives that source "
            "(sweep / step / ramp), and the response of every joint "
            "in ``--cross-targets`` is recorded.  Output is a "
            "matrix view with ``coupling_ratio = target_amp / "
            "source_amp`` per (source, target) pair.  Use this to "
            "map "
            "arms move → grippers swing"
            " cross-coupling."
        ),
    )
    parser.add_argument(
        "--cross-sources",
        default="",
        help=(
            "Comma-separated joint names to use as cross-impact "
            "SOURCES (driven joints).  Required when "
            "--cross-impact-mode != none."
        ),
    )
    parser.add_argument(
        "--cross-targets",
        default="",
        help=(
            "Comma-separated joint names to OBSERVE for cross-impact.  " "Required when --cross-impact-mode != none."
        ),
    )
    parser.add_argument(
        "--cross-amp",
        type=float,
        default=0.3,
        help="Cross-impact drive amplitude (rad / m).  Default %(default)s.",
    )
    parser.add_argument(
        "--cross-freq-hz",
        type=float,
        default=0.5,
        help="Cross-impact sweep frequency in Hz (only used with cross-impact-mode=sweep).",
    )
    parser.add_argument(
        "--sweep-actuator",
        default="",
        help=(
            "Actuator to sinusoidally drive when --ctrl-mode=sweep.  "
            "Accepts an explicit MJCF actuator name or the static-tool "
            "key form ``role:joint_name`` (e.g. "
            "``position:idx31_gripper_l_inner_joint1``)."
        ),
    )
    parser.add_argument(
        "--sweep-amp",
        type=float,
        default=0.1,
        help="Sweep amplitude (rad / m) added to the hold ctrl (default: %(default)s).",
    )
    parser.add_argument(
        "--sweep-freq-hz",
        type=float,
        default=1.0,
        help="Sweep frequency in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--report-out",
        default="",
        metavar="PATH",
        help="Where to write the JSON report.  Empty = print to stdout.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the human-readable summary; suppress the JSON dump on stdout.",
    )
    parser.add_argument(
        "--strip-prefix-a",
        default=None,
        metavar="PREFIX",
        help="Strip namespace from joint names in A.  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--strip-prefix-b",
        default=None,
        metavar="PREFIX",
        help="Strip namespace from joint names in B.  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--no-auto-strip-prefix",
        action="store_true",
        help="Disable auto-prefix detection.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logger = _Logger()

    try:
        report = _build_report(
            xml_a=args.xml_a,
            xml_b=args.xml_b or None,
            n_steps=args.steps,
            substeps=args.substeps,
            init_pose_spec=args.init_pose,
            ctrl_mode=args.ctrl_mode,
            sweep_actuator=args.sweep_actuator or None,
            sweep_amp=args.sweep_amp,
            sweep_freq_hz=args.sweep_freq_hz,
            logger=logger,
            strip_prefix_a=args.strip_prefix_a or "",
            strip_prefix_b=args.strip_prefix_b or "",
            auto_strip_prefix=not args.no_auto_strip_prefix,
            survey_mode=args.survey_mode if args.survey_mode != "none" else None,
            survey_amp=args.survey_amp,
            survey_only=[s.strip() for s in args.survey_only.split(",") if s.strip()] or None,
            survey_skip=[s.strip() for s in args.survey_skip.split(",") if s.strip()] or None,
            cross_mode=args.cross_impact_mode if args.cross_impact_mode != "none" else None,
            cross_sources=[s.strip() for s in args.cross_sources.split(",") if s.strip()] or None,
            cross_targets=[s.strip() for s in args.cross_targets.split(",") if s.strip()] or None,
            cross_amp=args.cross_amp,
            cross_freq_hz=args.cross_freq_hz,
        )
    except FileNotFoundError as exc:
        logger.error(f"input not found: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        import traceback

        logger.error(f"report build failed: {exc}")
        traceback.print_exc()
        return 3

    _print_summary(report, logger)

    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.report_out:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.report_out)) or ".", exist_ok=True)
            with open(args.report_out, "w") as f:
                f.write(text)
            logger.info(f"JSON report written to {args.report_out}")
        except OSError as exc:
            logger.error(f"could not write report: {exc}")
            return 4
    elif not args.summary_only:
        print(text)

    return 0 if not report.get("suspicious") else 1


if __name__ == "__main__":
    sys.exit(main())
