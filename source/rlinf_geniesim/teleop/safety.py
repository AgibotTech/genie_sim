# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Inspect simulator state before passing joint positions to teleoperation IK."""

import numpy as np


def joint_pos_vel(out, env: int = 0):
    """Return this frame's ``(qpos, qvel)``.

    Prefer ``out.joint_states[env]``, shaped (J, 2), because it is the only
    path carrying velocity. Fall back to interleaved ``out.states[env]``.
    """
    js = getattr(out, "joint_states", None)
    if js is not None:
        a = np.asarray(js)[env]
        return (np.asarray(a[:, 0], dtype=np.float64), np.asarray(a[:, 1], dtype=np.float64))
    st = np.asarray(getattr(out, "states"))[env]
    return (np.asarray(st[0::2], dtype=np.float64), np.asarray(st[1::2], dtype=np.float64))


_TRIAGE_FIELDS = ("states", "joint_states", "body_poses", "object_poses", "rewards")


def nonfinite_report(out, env: int = 0, joint_names=None, action=None) -> dict | None:
    """Classify non-finite joint state in one frame, or return ``None``.

    Runs on raw server values, so it can distinguish sim-side corruption from
    client-side processing. ``action_finite`` is ``None`` when no action
    belongs to the frame, rather than claiming that action was finite.
    Classification failures degrade individual fields to ``None``; they never
    suppress the confirmed non-finite verdict.
    """
    qpos, qvel = joint_pos_vel(out, env)
    bad = ~np.isfinite(qpos) | ~np.isfinite(qvel)
    if not bad.any():
        return None
    idx = np.flatnonzero(bad).tolist()
    names = list(joint_names or [])
    counts = {}
    for field in _TRIAGE_FIELDS:
        try:
            arr = getattr(out, field, None)
            if arr is None:
                counts[field] = None
                continue
            value = np.asarray(arr)
            if value.ndim >= 1 and value.shape[0] > env:
                value = value[env]
            counts[field] = [int((~np.isfinite(value)).sum()), int(value.size)]
        except Exception:  # noqa: BLE001 - classification is optional
            counts[field] = None
    return {
        "n_bad_joints": len(idx),
        "n_joints": int(qpos.size),
        "bad_idx": idx[:24],
        "bad_names": [names[i] for i in idx[:24] if i < len(names)],
        "field_counts": counts,
        "action_finite": (None if action is None else bool(np.isfinite(np.asarray(action)).all())),
    }


def format_nonfinite(rep: dict, ctx: dict) -> str:
    """Summarize invalid state without assuming a cause."""
    counts = ", ".join(f"{key}={value[0]}/{value[1]}" for key, value in rep["field_counts"].items() if value)
    context = " ".join(f"{key}={value!r}" for key, value in ctx.items())
    return (
        f"[teleop] non-finite simulator joint state: {rep['n_bad_joints']}/{rep['n_joints']} joints; "
        f"indices={rep['bad_idx']} names={rep['bad_names']}; action_finite={rep.get('action_finite')}; "
        f"{counts}; {context}"
    )
