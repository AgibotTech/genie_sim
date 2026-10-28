#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Auto-tune PD (kp / kd) per joint class against a dumped MJCF.

Loads ``robot_runtime.xml`` (or any Newton-dumped MJCF following our
``position_<joint>`` actuator convention), runs scripted step-response
tests in pure CPU MuJoCo, and picks the (kp, kd) pair per class that
minimises::

    cost = settling_time_s + λ × overshoot_fraction

per the standard control-theory metrics:

  * settling_time : first time after which |q − target| stays inside
                    ``--tol-frac`` × step for the rest of the rollout
  * overshoot     : peak excursion past target, normalised by step size
  * ss_error      : final-frame residual, normalised by step size
                    (informational only; not in cost by default)

Default cost weights settling time linearly and adds 2× the overshoot
fraction — so a 10 % overshoot costs as much as 0.2 s of extra settle.
Tune via ``--cost-overshoot``.

Loop structure
--------------

For each joint class (body / head / arm_shoulder / arm_mid / arm_wrist
/ gripper_master) we:

  1. Pick a representative joint matching that class's regex.  Distal
     joints (the END of the kinematic chain in each class) have the
     lowest effective inertia and tightest saturation envelopes — they
     are the worst-case for control-loop tuning, so tuning them first
     means the gain choice is conservative for the easier sibling
     joints in the same class.
  2. Reset the model to the ``home`` keyframe (the dump's pre-built
     init pose) and set every actuator's ctrl to the home target so
     the rest of the robot holds while we exercise one joint.
  3. Sweep ``kp`` log-uniformly between ``kp_min`` and a per-class
     ``kp_max`` derived from ``max_effort / step_size`` (so the
     actuator doesn't saturate for >50 % of the step).
  4. For each ``kp``, sweep ``ζ`` (damping ratio) over a small set
     centred on 1.0; for each ζ compute ``kd = ζ · 2·√(kp·M_eff)``
     where ``M_eff`` is the joint's diagonal generalised inertia at
     the home pose.  Run the step response, score it.
  5. Pick the lowest-cost (kp, kd, ζ) for that class.

The chosen tuning is independent of the YAML's current values — we
write fresh actuator gain/bias on each trial, restore afterwards.  The
MJCF is never modified on disk.

Output
------

A YAML fragment ready to paste into
``physics_params.yaml::articulation_view_runtime``, e.g.::

    arm_shoulder:
      kp: 6400
      kd: 130
      max_effort: 108

plus a per-class table with the step-response metrics.  Use
``--output PATH`` to write the fragment to disk.

Viewer
------

By default the winning step response for each class is played back in
``mujoco.viewer.launch_passive`` at wall-clock speed.  Use ``--headless``
on a remote / CI machine.  Add ``--watch-all`` to see every candidate
(slow — ~25 × class × duration of playback).

Usage
-----

    # Default — headless tuning + viewer of winners.
    python auto_tune_pd.py --mjcf /path/to/robot_runtime.xml

    # CI-friendly headless.
    python auto_tune_pd.py --mjcf … --headless

    # Custom step size, longer simulation per trial.
    python auto_tune_pd.py --mjcf … --step-deg 2 --duration-s 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Joint classification — duplicated here so the tool runs without ROS / colcon.
# Mirrors common.joint_classification's regex set.  Keep in sync.
# ----------------------------------------------------------------------------

_RE_BODY = re.compile(r"(?:^|_)body_joint\d+$")
_RE_HEAD = re.compile(r"(?:^|_)head_joint\d+$")
_RE_ARM = re.compile(r"(?:^|_)arm_[lr]_joint(\d+)$")
_RE_GRIPPER_MASTER = re.compile(r"gripper_[lr]_inner_joint1$")
_RE_CHASSIS_STEER = re.compile(r"chassis_[lr]wheel_(?:front|rear)_joint1$")
_RE_CHASSIS_DRIVE = re.compile(r"chassis_[lr]wheel_(?:front|rear)_joint2$")

_ARM_SHOULDER_IDX = frozenset({1, 2})
_ARM_MID_IDX = frozenset({3, 4, 5})
_ARM_WRIST_IDX = frozenset({6, 7})


def classify(joint_name: str) -> Optional[str]:
    if _RE_BODY.search(joint_name):
        return "body"
    if _RE_HEAD.search(joint_name):
        return "head"
    if _RE_GRIPPER_MASTER.search(joint_name):
        return "gripper_master"
    m = _RE_ARM.search(joint_name)
    if m:
        n = int(m.group(1))
        if n in _ARM_SHOULDER_IDX:
            return "arm_shoulder"
        if n in _ARM_MID_IDX:
            return "arm_mid"
        if n in _ARM_WRIST_IDX:
            return "arm_wrist"
        return "arm"
    return None


# ----------------------------------------------------------------------------
# Step response + metrics
# ----------------------------------------------------------------------------


def step_response(
    m: Any,
    d: Any,
    target_aid: int,
    target_qadr: int,
    kp: float,
    kd: float,
    step_size_rad: float,
    sim_duration_s: float,
    home_qpos: np.ndarray,
    home_ctrl: np.ndarray,
    hold_others_stiff: bool = True,
    stiff_kp: float = 5.0e4,
    zero_gravity: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run one step-response trial.  Mutates the actuator's gain/bias
    transiently and restores at end so the model is unchanged across
    calls.

    When ``hold_others_stiff`` is True (default), ALL other position
    actuators are temporarily switched to a stiff PD (kp=stiff_kp,
    kd at critical damping computed from each joint's M_eff) so they
    rigidly hold the home pose while the target joint is exercised.
    This isolates the target joint's response from the rest of the
    robot's potentially-mistuned PD chain — without it, a stiff
    actuator can excite a soft neighbor and the metrics misrepresent
    the target joint's true response.

    Position actuators with zero kp (e.g. gripper mimic followers)
    and motor_* raw-torque actuators are left alone in both modes —
    overriding them would break the constraint chain or inject
    unintended torque."""
    import mujoco

    # Save target actuator's original PD so we restore after the trial.
    orig_gain = m.actuator_gainprm[target_aid].copy()
    orig_bias = m.actuator_biasprm[target_aid].copy()

    # Optionally lock all other position actuators stiff at home.
    saved_others: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if hold_others_stiff:
        # Pre-compute each joint's home-pose M_eff once to size kd
        # critically.  ``mj_fullM`` populates the full nv×nv matrix;
        # we read diagonals.  The home keyframe was already loaded
        # before this call (mujoco.mj_resetData + mj_forward), so
        # qM reflects home-pose inertia.
        M_full = np.zeros((m.nv, m.nv), dtype=np.float64)
        mujoco.mj_fullM(m, M_full, d.qM)
        for aid in range(m.nu):
            if aid == target_aid:
                continue
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            if not nm or not nm.startswith("position_"):
                # motor_* actuators apply ctrl·gear directly; their
                # ctrl stays at home_ctrl[aid] = 0 so they emit zero
                # torque — no need to "hold" them.
                continue
            if abs(m.actuator_gainprm[aid][0]) < 1e-9:
                # Inactive (kp=0) — typically a gripper mimic
                # follower whose motion is governed by the
                # equality constraint, not by an actuator.  Leave it
                # alone or we'll fight the constraint solver.
                continue
            jid_act = int(m.actuator_trnid[aid][0])
            if jid_act < 0 or jid_act >= m.njnt:
                continue
            vadr = int(m.jnt_dofadr[jid_act])
            m_eff = float(M_full[vadr, vadr])
            if m_eff < 1e-6:
                continue
            # ζ = 0.7: slightly underdamped → fast settle without
            # excessive overshoot when the test joint kicks them.
            kd_stiff = 1.4 * math.sqrt(stiff_kp * m_eff)
            saved_others[aid] = (
                m.actuator_gainprm[aid].copy(),
                m.actuator_biasprm[aid].copy(),
            )
            m.actuator_gainprm[aid] = 0.0
            m.actuator_gainprm[aid][0] = stiff_kp
            m.actuator_biasprm[aid] = 0.0
            m.actuator_biasprm[aid][1] = -stiff_kp
            m.actuator_biasprm[aid][2] = -kd_stiff

    m.actuator_gainprm[target_aid] = 0.0
    m.actuator_gainprm[target_aid][0] = float(kp)
    m.actuator_biasprm[target_aid] = 0.0
    m.actuator_biasprm[target_aid][1] = -float(kp)
    m.actuator_biasprm[target_aid][2] = -float(kd)

    mujoco.mj_resetData(m, d)
    d.qpos[:] = home_qpos
    d.ctrl[:] = home_ctrl

    # Optionally zero gravity during the trial.  Reasoning: PD tuning
    # is fundamentally an INERTIAL characterization — we want
    # ω_n = √(kp/M) and ζ = kd/(2·√(kp·M)) to determine the response.
    # Under gravity, a constant load adds a steady-state error
    # (proportional to mg·lever / kp) that the metrics misinterpret
    # as poor tracking, biasing the tuner toward higher kp than
    # needed.  Gravity-free trials produce clean overdamped step
    # responses that depend only on kp/kd/M.
    saved_gravity = None
    if zero_gravity:
        saved_gravity = m.opt.gravity.copy()
        m.opt.gravity[:] = 0.0
        # Re-run forward dynamics with zero gravity so qfrc_passive
        # reflects the gravity-free state at home.
        mujoco.mj_forward(m, d)

    initial = float(home_ctrl[target_aid])
    target = initial + float(step_size_rad)
    d.ctrl[target_aid] = target

    n_steps = int(sim_duration_s / m.opt.timestep)
    history = np.zeros(n_steps, dtype=np.float64)
    for i in range(n_steps):
        mujoco.mj_step(m, d)
        history[i] = d.qpos[target_qadr]

    if saved_gravity is not None:
        m.opt.gravity[:] = saved_gravity
    m.actuator_gainprm[target_aid] = orig_gain
    m.actuator_biasprm[target_aid] = orig_bias
    for aid, (gain, bias) in saved_others.items():
        m.actuator_gainprm[aid] = gain
        m.actuator_biasprm[aid] = bias

    times = np.arange(n_steps, dtype=np.float64) * m.opt.timestep
    return times, history, target


def step_metrics(
    times: np.ndarray,
    history: np.ndarray,
    target: float,
    initial: float,
    tol_frac: float,
) -> Dict[str, float]:
    step = target - initial
    if abs(step) < 1e-9:
        return {"settling_s": 0.0, "overshoot": 0.0, "ss_err": 0.0, "rise_s": 0.0}

    tol = abs(tol_frac * step)
    err = np.abs(history - target)

    # Settling time: index of LAST sample outside tolerance + 1.  If
    # the response never enters the tolerance band, settling = full
    # simulation duration (worst case).
    outside = err > tol
    if outside.any():
        last = int(np.where(outside)[0][-1])
        if last < len(times) - 1:
            settling = float(times[last + 1])
        else:
            settling = float(times[-1])
    else:
        settling = 0.0

    # Rise time: first time inside ± tol_frac of step.  Used for
    # display only; not weighted in cost.
    inside = err <= tol
    rise = float(times[int(np.argmax(inside))]) if inside.any() else float(times[-1])

    # Overshoot: peak past target, normalised by step magnitude.
    if step > 0:
        peak = float(history.max() - target)
    else:
        peak = float(target - history.min())
    overshoot = max(0.0, peak) / abs(step)

    ss_err = abs(float(history[-1]) - target) / abs(step)

    return {"settling_s": settling, "rise_s": rise, "overshoot": overshoot, "ss_err": ss_err}


# ----------------------------------------------------------------------------
# Per-class tuning
# ----------------------------------------------------------------------------


def diagonal_inertia(m: Any, d: Any, jid: int) -> float:
    import mujoco

    M = np.zeros((m.nv, m.nv), dtype=np.float64)
    mujoco.mj_fullM(m, M, d.qM)
    vadr = m.jnt_dofadr[jid]
    return float(M[vadr, vadr])


def pick_representative(class_name: str, joints_in_class: List[str]) -> str:
    """For each class, pick the joint we tune against.

    Distal joints have the smallest effective inertia, smallest
    saturation envelope, and the highest natural frequency — so they
    are the worst-case for stability.  Tuning them conservatively
    means the chosen gains are safe across the whole class.

    Naming conventions:
      arm_shoulder = arm_*_joint1 or joint2 — distal of pair = joint1
      arm_mid      = arm_*_joint3..5        — distal = joint3
      arm_wrist    = arm_*_joint6..7        — distal = joint6
      body         = body_joint1..5         — distal (most exposed) = joint5
      head         = head_joint1..3         — joint3
      gripper      = gripper_*_inner_joint1 — only one master per side

    The choice is heuristic — when in doubt, pass ``--joint`` to
    override and tune against a specific joint by name."""
    if not joints_in_class:
        return ""

    def _rank(name: str) -> Tuple[int, ...]:
        m = re.search(r"joint(\d+)$", name)
        if m:
            return (int(m.group(1)), name)
        return (0, name)

    if class_name == "arm_shoulder":
        # distal = joint1
        return min(joints_in_class, key=_rank)
    if class_name in ("arm_mid", "arm_wrist"):
        return min(joints_in_class, key=_rank)
    if class_name == "body":
        return max(joints_in_class, key=_rank)
    if class_name == "head":
        return max(joints_in_class, key=_rank)
    return sorted(joints_in_class)[0]


def tune_class(
    m: Any,
    d: Any,
    class_name: str,
    target_joint: str,
    home_qpos: np.ndarray,
    home_ctrl: np.ndarray,
    step_size_rad: float,
    sim_duration_s: float,
    tol_frac: float,
    cost_overshoot_weight: float,
    n_kp: int,
    zetas: Tuple[float, ...],
    effort_cap_override: Optional[float] = None,
    hold_others_stiff: bool = True,
    stiff_kp: float = 5.0e4,
    saturation_margin: float = 1.0,
    zero_gravity: bool = True,
    log_fn=print,
) -> Optional[Dict[str, Any]]:
    import mujoco

    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"position_{target_joint}")
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, target_joint)
    if aid < 0 or jid < 0:
        log_fn(f"[auto-tune] {class_name}: no actuator/joint for {target_joint!r}; skipping")
        return None

    qadr = int(m.jnt_qposadr[jid])
    M_eff = diagonal_inertia(m, d, jid)

    # Resolve max_effort.  Priority:
    #   1. CLI override (``--effort-cap CLASS=N``) — explicit operator
    #      intent, trumps everything.
    #   2. Joint-level actuator force range (``<joint actuatorfrcrange>``)
    #      — the URDF-effort cap our pipeline authors per joint.
    #   3. Actuator forcerange — only meaningful when non-zero in the
    #      MJCF; position_* actuators in our dump default to [0, 0]
    #      (= unlimited), in which case we fall through to "infinity".
    #
    # Without a finite cap the tuner picks kp purely on settling /
    # overshoot, which usually means VERY stiff values — fine for
    # exploring the cost surface but not realistic for hardware.
    if effort_cap_override is not None:
        max_effort = float(effort_cap_override)
        effort_source = "cli"
    else:
        jfr = m.jnt_actfrcrange[jid] if hasattr(m, "jnt_actfrcrange") else None
        if jfr is not None and (jfr[0] != 0.0 or jfr[1] != 0.0):
            max_effort = float(max(abs(jfr[0]), abs(jfr[1])))
            effort_source = "joint.actuatorfrcrange"
        else:
            fr = m.actuator_forcerange[aid]
            if fr[0] != 0.0 or fr[1] != 0.0:
                max_effort = float(max(abs(fr[0]), abs(fr[1])))
                effort_source = "actuator.forcerange"
            else:
                max_effort = float("inf")
                effort_source = "unlimited"

    # kp upper bound — derived from the saturation envelope:
    #
    #     saturation_envelope = max_effort / kp     (rad of error
    #                                                where actuator clips)
    #
    # We want this envelope to be AT LEAST ``saturation_margin × step``
    # so the PD never bang-bangs during a normal step command:
    #
    #     kp_max = max_effort / (step × saturation_margin)
    #
    # Defaults to ``saturation_margin = 1.0`` (envelope = step ≈
    # marginal — saturation can fire at the very start of a step but
    # PD takes over by ~1° of progress).  Bump to 2.0 for a "no
    # saturation under any commanded step" guarantee — the tuner
    # then picks softer kp but the response stays in the linear PD
    # regime end-to-end.  Set < 1.0 to allow saturation if you're
    # tuning AGAINST a torque-limited real-robot loop and want to
    # match its bang-bang signature.
    if math.isfinite(max_effort):
        kp_max = max_effort / (step_size_rad * max(saturation_margin, 1e-6))
    else:
        # No forcerange → arbitrary ceiling.  Picked so kp/M_eff×dt²
        # stays well inside explicit-step stability for typical dt.
        kp_max = 5.0 / (m.opt.timestep**2) * M_eff
    # Lower bound — never tune below something that produces a
    # noticeable response within sim_duration_s.
    kp_min = max(50.0, 0.5 / M_eff)
    if kp_max <= kp_min * 1.5:
        kp_max = kp_min * 10.0

    kp_grid = np.geomspace(kp_min, kp_max, n_kp)

    trials = []
    eff_str = "∞" if not math.isfinite(max_effort) else f"{max_effort:.2f}"
    log_fn(
        f"\n[auto-tune] {class_name}: joint {target_joint!r}  M_eff={M_eff:.4f} kg·m²  "
        f"max_effort={eff_str} N·m ({effort_source})  step={math.degrees(step_size_rad):.2f}°"
    )
    log_fn(
        f"           sweep kp ∈ [{kp_min:.0f}, {kp_max:.0f}] ({n_kp} points × {len(zetas)} ζ values "
        f"= {n_kp * len(zetas)} trials)"
    )
    for kp in kp_grid:
        kd_crit = 2.0 * math.sqrt(kp * M_eff)
        for zeta in zetas:
            kd = float(zeta) * kd_crit
            times, hist, target = step_response(
                m,
                d,
                aid,
                qadr,
                kp,
                kd,
                step_size_rad,
                sim_duration_s,
                home_qpos,
                home_ctrl,
                hold_others_stiff=hold_others_stiff,
                stiff_kp=stiff_kp,
                zero_gravity=zero_gravity,
            )
            metrics = step_metrics(times, hist, target, float(home_ctrl[aid]), tol_frac)
            # NaN guard — unstable trials show up as huge values or NaN.
            if not np.isfinite(hist).all():
                cost = float("inf")
                metrics["unstable"] = 1.0
            else:
                cost = metrics["settling_s"] + cost_overshoot_weight * metrics["overshoot"]
            trials.append(
                {
                    "kp": float(kp),
                    "kd": float(kd),
                    "zeta": float(zeta),
                    "cost": float(cost),
                    **metrics,
                }
            )

    finite_trials = [t for t in trials if math.isfinite(t["cost"])]
    if not finite_trials:
        log_fn(f"[auto-tune] {class_name}: every trial unstable; widen kp range or check MJCF")
        return None

    winner = min(finite_trials, key=lambda t: t["cost"])
    log_fn(
        f"           winner kp={winner['kp']:.0f}  kd={winner['kd']:.1f}  ζ={winner['zeta']:.2f}  "
        f"settle={winner['settling_s']*1000:.0f}ms  rise={winner['rise_s']*1000:.0f}ms  "
        f"overshoot={winner['overshoot']*100:.1f}%  ss_err={winner['ss_err']*100:.2f}%"
    )

    return {
        "class": class_name,
        "joint": target_joint,
        "M_eff": M_eff,
        "max_effort": max_effort if math.isfinite(max_effort) else None,
        "kp": winner["kp"],
        "kd": winner["kd"],
        "zeta": winner["zeta"],
        "metrics": {
            "settling_ms": winner["settling_s"] * 1000.0,
            "rise_ms": winner["rise_s"] * 1000.0,
            "overshoot_pct": winner["overshoot"] * 100.0,
            "ss_err_pct": winner["ss_err"] * 100.0,
        },
        "n_trials": len(trials),
        "_aid": aid,
        "_qadr": qadr,
    }


# ----------------------------------------------------------------------------
# Viewer playback
# ----------------------------------------------------------------------------


def viewer_playback(
    m: Any,
    d: Any,
    winners: List[Dict[str, Any]],
    home_qpos: np.ndarray,
    home_ctrl: np.ndarray,
    step_size_rad: float,
    sim_duration_s: float,
    hold_others_stiff: bool = True,
    stiff_kp: float = 1.0e6,
    log_fn=print,
) -> None:
    """Replay each winning step test at wall-clock speed in a passive
    viewer.  Blocks until the viewer is closed or all clips finish.

    Uses the same ``hold_others_stiff`` mode that tuning used so the
    visual playback matches the metrics — otherwise neighbour-joint
    motion would distort what the operator sees vs what the tuner
    measured."""
    import mujoco
    import mujoco.viewer

    log_fn(
        f"\n[auto-tune] launching viewer; replaying {len(winners)} winning step test(s) "
        f"at wall-clock speed (close the window to quit)…"
    )
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for w in winners:
            if not viewer.is_running():
                break
            aid, qadr = w["_aid"], w["_qadr"]
            log_fn(f"           replaying {w['class']}/{w['joint']}: kp={w['kp']:.0f} kd={w['kd']:.1f}")

            # Stiff-hold neighbours, mirroring step_response.
            saved_others: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            if hold_others_stiff:
                # Need M at home pose — reset+forward first.
                mujoco.mj_resetData(m, d)
                d.qpos[:] = home_qpos
                d.ctrl[:] = home_ctrl
                mujoco.mj_forward(m, d)
                M_full = np.zeros((m.nv, m.nv), dtype=np.float64)
                mujoco.mj_fullM(m, M_full, d.qM)
                for oaid in range(m.nu):
                    if oaid == aid:
                        continue
                    onm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, oaid)
                    if not onm or not onm.startswith("position_"):
                        continue
                    if abs(m.actuator_gainprm[oaid][0]) < 1e-9:
                        continue
                    jid_o = int(m.actuator_trnid[oaid][0])
                    if jid_o < 0 or jid_o >= m.njnt:
                        continue
                    vadr_o = int(m.jnt_dofadr[jid_o])
                    m_eff_o = float(M_full[vadr_o, vadr_o])
                    if m_eff_o < 1e-6:
                        continue
                    kd_stiff = 1.4 * math.sqrt(stiff_kp * m_eff_o)
                    saved_others[oaid] = (
                        m.actuator_gainprm[oaid].copy(),
                        m.actuator_biasprm[oaid].copy(),
                    )
                    m.actuator_gainprm[oaid] = 0.0
                    m.actuator_gainprm[oaid][0] = stiff_kp
                    m.actuator_biasprm[oaid] = 0.0
                    m.actuator_biasprm[oaid][1] = -stiff_kp
                    m.actuator_biasprm[oaid][2] = -kd_stiff

            # Set winning gains on this actuator only.
            orig_gain = m.actuator_gainprm[aid].copy()
            orig_bias = m.actuator_biasprm[aid].copy()
            m.actuator_gainprm[aid] = 0.0
            m.actuator_gainprm[aid][0] = w["kp"]
            m.actuator_biasprm[aid] = 0.0
            m.actuator_biasprm[aid][1] = -w["kp"]
            m.actuator_biasprm[aid][2] = -w["kd"]

            mujoco.mj_resetData(m, d)
            d.qpos[:] = home_qpos
            d.ctrl[:] = home_ctrl
            d.ctrl[aid] = float(home_ctrl[aid]) + step_size_rad

            n_steps = int(sim_duration_s / m.opt.timestep)
            for _ in range(n_steps):
                if not viewer.is_running():
                    break
                t0 = time.monotonic()
                mujoco.mj_step(m, d)
                viewer.sync()
                # Pace at wall-clock — sleep the leftover of one
                # timestep so visual playback matches real time.
                slack = m.opt.timestep - (time.monotonic() - t0)
                if slack > 0:
                    time.sleep(slack)

            m.actuator_gainprm[aid] = orig_gain
            m.actuator_biasprm[aid] = orig_bias
            for oaid, (gain, bias) in saved_others.items():
                m.actuator_gainprm[oaid] = gain
                m.actuator_biasprm[oaid] = bias
            # Brief hold so the operator can see the settled state.
            hold_end = time.monotonic() + 0.5
            while viewer.is_running() and time.monotonic() < hold_end:
                viewer.sync()
                time.sleep(0.02)


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------


def emit_yaml_fragment(winners: List[Dict[str, Any]]) -> str:
    """Render a yaml block fit for pasting under
    ``articulation_view_runtime:``.  Comments include the metrics for
    each class so the operator can sanity-check the choice."""
    lines = ["articulation_view_runtime:"]
    for w in winners:
        mtr = w["metrics"]
        lines.append(
            f"  {w['class']}:           "
            f"# rep={w['joint']}  M_eff={w['M_eff']:.4f}  "
            f"settle={mtr['settling_ms']:.0f}ms  "
            f"overshoot={mtr['overshoot_pct']:.1f}%  ζ={w['zeta']:.2f}"
        )
        lines.append(f"    kp: {w['kp']:.0f}")
        lines.append(f"    kd: {w['kd']:.1f}")
        if w["max_effort"] is not None:
            lines.append(f"    max_effort: {w['max_effort']:.0f}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mjcf", required=True, help="Path to the MJCF dump (robot_runtime.xml).")
    p.add_argument(
        "--classes",
        nargs="+",
        default=["body", "head", "arm_shoulder", "arm_mid", "arm_wrist", "gripper_master"],
        help="Classes to tune.  Default: all standard classes.",
    )
    p.add_argument(
        "--joint",
        default=None,
        help="Override the representative joint (tune ONLY this joint, class auto-detected).",
    )
    p.add_argument("--step-deg", type=float, default=1.0, help="Step magnitude in degrees (default: 1).")
    p.add_argument("--duration-s", type=float, default=1.5, help="Sim duration per trial (default: 1.5 s).")
    p.add_argument(
        "--tol-frac", type=float, default=0.05, help="Settling tolerance as fraction of step (default 0.05)."
    )
    p.add_argument(
        "--cost-overshoot",
        type=float,
        default=2.0,
        dest="cost_overshoot_weight",
        help="Cost = settling_s + λ × overshoot_fraction.  Higher λ = more overshoot-averse.  Default 2.0.",
    )
    p.add_argument("--n-kp", type=int, default=8, help="Number of kp grid points (geometric).  Default 8.")
    p.add_argument(
        "--zetas",
        nargs="+",
        type=float,
        default=[0.7, 1.0, 1.5, 2.0],
        help="Damping ratios to sweep at each kp.  Default 0.7 / 1.0 / 1.5 / 2.0.",
    )
    p.add_argument(
        "--effort-cap",
        action="append",
        default=[],
        metavar="CLASS=N",
        help=(
            "Per-class effort cap override (N·m).  Repeatable.  "
            "Example: ``--effort-cap arm_shoulder=108 --effort-cap arm_mid=35``.  "
            "When unset for a class, falls back to the MJCF's joint "
            "``actuatorfrcrange`` then to the actuator ``forcerange`` "
            "then to unlimited (which lets the tuner pick aggressive kp)."
        ),
    )
    p.add_argument(
        "--no-hold-others-stiff",
        dest="hold_others_stiff",
        action="store_false",
        default=True,
        help=(
            "Disable the default behaviour of stiffening every "
            "non-target position actuator with a high kp during a "
            "trial.  By default, all other PD-driven joints are "
            "switched to ``kp=stiff_kp`` (with critically-damped kd "
            "from each joint's home M_eff) to lock the rest of the "
            "robot at home pose so the target joint's response is "
            "isolated from neighbour-PD-mistuning.  Pass this flag "
            "to keep each non-target actuator at whatever PD it has "
            "in the dump — useful for tuning against a known-good "
            "reference, harmful when the dump's PD is itself in "
            "flux (the typical case)."
        ),
    )
    p.add_argument(
        "--stiff-kp",
        type=float,
        default=5.0e4,
        help=(
            "kp value applied to non-target position actuators when "
            "``--hold-others-stiff`` is on (default).  kd is sized "
            "for ζ=0.7 from each joint's home M_eff.  Default 5e4 is "
            "stable at dt=1ms for any joint with M ≥ 0.05 kg·m² "
            "(critical dt = √(4·M/kp_stiff) > sim dt).  Bumping much "
            "above 1e5 risks numerical instability with implicitfast "
            "integrator at typical timesteps."
        ),
    )
    p.add_argument(
        "--no-zero-gravity",
        dest="zero_gravity",
        action="store_false",
        default=True,
        help=(
            "Disable the default behaviour of zeroing gravity during "
            "step trials.  PD tuning is fundamentally an inertial "
            "characterization (ω_n = √(kp/M), ζ = kd/(2·√(kp·M))) "
            "and gravity adds a constant load that biases ss_err and "
            "settling-time metrics.  Default ON: trials run gravity-"
            "free for clean PD characterization.  Pass this flag to "
            "tune AGAINST the loaded condition (e.g. when you want "
            "kp sized to overcome gravity steady-state error)."
        ),
    )
    p.add_argument(
        "--saturation-margin",
        type=float,
        default=1.0,
        help=(
            "Saturation envelope expressed as a multiple of step size. "
            "kp_max is chosen so saturation_envelope = step × this margin. "
            "Default 1.0: envelope = step (marginal — saturation can fire "
            "briefly).  2.0: no saturation under commanded step (softer kp, "
            "always linear).  0.5: deeper saturation allowed (stiffer kp, "
            "bang-bang response — useful when tuning against a torque-"
            "limited real-robot loop)."
        ),
    )
    p.add_argument("--headless", action="store_true", help="Skip the viewer playback (default: viewer ON).")
    p.add_argument(
        "--watch-all",
        action="store_true",
        help="Open viewer for EVERY candidate trial (very slow; for debugging the cost surface).",
    )
    p.add_argument("--output", default=None, help="Write the yaml fragment to this path (also printed).")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        import mujoco
    except ImportError:
        print("ERROR: `mujoco` not on the Python path; pip install mujoco.", file=sys.stderr)
        return 2

    if not os.path.isfile(args.mjcf):
        print(f"ERROR: MJCF not found: {args.mjcf!r}", file=sys.stderr)
        return 2

    print(f"[auto-tune] loading {args.mjcf}")
    m = mujoco.MjModel.from_xml_path(args.mjcf)
    d = mujoco.MjData(m)

    # Resolve home pose — prefer a keyframe named 'home', else mj_resetData defaults.
    home_kid = -1
    for i in range(m.nkey):
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) == "home":
            home_kid = i
            break
    if home_kid >= 0:
        mujoco.mj_resetDataKeyframe(m, d, home_kid)
        print(f"[auto-tune] reset to keyframe 'home' (id={home_kid})")
    else:
        mujoco.mj_resetData(m, d)
        print("[auto-tune] no 'home' keyframe; using mj_resetData defaults.")
    mujoco.mj_forward(m, d)
    home_qpos = d.qpos.copy()
    # Default ctrl: current actuator ctrl (zero for fresh-loaded models;
    # for a dump with keyframe, ctrl is whatever the keyframe set).
    home_ctrl = d.ctrl.copy()

    # Build {class: [joint_names]} index from the MJCF's joint list.
    class_to_joints: Dict[str, List[str]] = {}
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not nm:
            continue
        c = classify(nm)
        if c is None:
            continue
        class_to_joints.setdefault(c, []).append(nm)

    print(f"[auto-tune] joint classes found: { {k: len(v) for k, v in class_to_joints.items()} }")

    # Tuning targets — either user-specified single joint or per-class sweep.
    targets: List[Tuple[str, str]] = []
    if args.joint is not None:
        c = classify(args.joint)
        if c is None:
            print(f"ERROR: joint {args.joint!r} doesn't match any known class regex.", file=sys.stderr)
            return 2
        targets = [(c, args.joint)]
    else:
        for cls in args.classes:
            joints = class_to_joints.get(cls)
            if not joints:
                print(f"[auto-tune] WARN: class {cls!r} has no joints in this MJCF; skipping")
                continue
            targets.append((cls, pick_representative(cls, joints)))

    if not targets:
        print("ERROR: nothing to tune.", file=sys.stderr)
        return 1

    step_size_rad = math.radians(args.step_deg)

    # Parse --effort-cap CLASS=N pairs into a dict.
    effort_caps: Dict[str, float] = {}
    for spec in args.effort_cap:
        if "=" not in spec:
            print(f"ERROR: --effort-cap expects CLASS=N, got {spec!r}", file=sys.stderr)
            return 2
        cls_name, val_str = spec.split("=", 1)
        try:
            effort_caps[cls_name.strip()] = float(val_str)
        except ValueError:
            print(f"ERROR: --effort-cap {spec!r}: value not a float", file=sys.stderr)
            return 2

    winners: List[Dict[str, Any]] = []
    for cls, jnt in targets:
        w = tune_class(
            m,
            d,
            cls,
            jnt,
            home_qpos,
            home_ctrl,
            step_size_rad,
            args.duration_s,
            args.tol_frac,
            args.cost_overshoot_weight,
            args.n_kp,
            tuple(args.zetas),
            effort_cap_override=effort_caps.get(cls),
            hold_others_stiff=args.hold_others_stiff,
            stiff_kp=args.stiff_kp,
            saturation_margin=args.saturation_margin,
            zero_gravity=args.zero_gravity,
        )
        if w is not None:
            winners.append(w)

    if not winners:
        print("[auto-tune] no winners produced.", file=sys.stderr)
        return 1

    # YAML output
    fragment = emit_yaml_fragment(winners)
    print("\n[auto-tune] YAML fragment (paste under articulation_view_runtime):\n")
    print(fragment)
    if args.output:
        with open(args.output, "w") as f:
            f.write(fragment)
        print(f"[auto-tune] wrote {args.output}")

    # JSON dump (for tooling) — always to stderr-adjacent path for record.
    summary_path = (args.output or args.mjcf) + ".tune.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "mjcf": args.mjcf,
                    "step_deg": args.step_deg,
                    "duration_s": args.duration_s,
                    "cost_overshoot_weight": args.cost_overshoot_weight,
                    "winners": [{k: v for k, v in w.items() if not k.startswith("_")} for w in winners],
                },
                f,
                indent=2,
            )
        print(f"[auto-tune] full summary: {summary_path}")
    except OSError:
        pass

    # Viewer playback
    if not args.headless:
        try:
            viewer_playback(
                m,
                d,
                winners,
                home_qpos,
                home_ctrl,
                step_size_rad,
                args.duration_s,
                hold_others_stiff=args.hold_others_stiff,
                stiff_kp=args.stiff_kp,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[auto-tune] viewer playback failed: {exc!r}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
