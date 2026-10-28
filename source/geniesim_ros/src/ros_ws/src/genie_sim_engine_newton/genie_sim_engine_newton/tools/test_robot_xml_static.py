#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Structural diff for two MJCF robots (no stepping).

What this exists for
--------------------

When ``isaac_newton``'s ``robot_runtime.xml`` (the disposable-solver
dump) and a reference MJCF (the newton-standalone dump or a hand-
authored MJCF you trust) disagree at runtime, you usually want to
know FIRST whether the two share the same structure: same joint
list, same actuator gainprms, same equality constraints, same body
inertia.  That's a cheap question — we can answer it without ever
calling ``mj_step``.

This tool is the static-analysis half.  The dynamic / rollout half
lives in ``test_robot_xml_dynamic.py``.  The two are useful in
different parts of the loop:

  * **static** is run-once-per-build.  Loads in <1 s, fails CI when
    the converter regresses a gain or drops an equality edge.
  * **dynamic** is run when something already tracks at runtime;
    structural answers don't help if the question is "this joint
    oscillates at runtime, why".

What it does
------------

Loads BOTH MJCFs through pip-installed ``mujoco``, walks the
resulting ``MjModel`` arrays, matches entities by name (with
optional prefix-strip for wrapper-namespaced names like
``_genie_Physics_idx32_…``), and produces a structured diff:

  * **Options** — timestep, integrator, solver, iterations, gravity,
    impratio, contact / joint defaults.
  * **Joints** — type, axis, range, armature, damping, frictionloss,
    actfrcrange.
  * **Actuators** — gainprm, biasprm, biastype/gaintype, gear,
    forcerange, ctrlrange, trnid (joint target).  Keyed by
    ``"<role>:<joint_name>"`` so position vs motor actuators on
    the same DOF stay distinguishable.
  * **Equality constraints** — type, obj1/obj2, polycoef, solref,
    solimp, active flag.  Lined up by ``(obj1_name, obj2_name)``.
  * **Bodies** — mass, diaginertia, ipos.

Output is a JSON report + a human-readable summary.  Exits non-zero
if the suspicious digest is non-empty (CI-friendly).

Usage
-----

  # Diff two MJCFs, write the JSON report
  python3 scripts/tools/test_robot_xml_static.py \\
      --xml-a /geniesim_assets/scenes/scene_flat_g2_sp/robot_runtime.xml \\
      --xml-b <path-to-reference-mjcf> \\
      --report-out /tmp/static_diff.json

  # Suppress JSON on stdout, keep just the summary
  ... --summary-only

  # Disable the auto-prefix detection
  ... --no-auto-strip-prefix
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
        f"[test_robot_xml_static] ERROR: this tool needs pip-installed `mujoco` "
        f"and `numpy` ({exc}).  Install with `pip install mujoco numpy`.",
        file=sys.stderr,
    )
    sys.exit(2)


# ----------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------


class _Logger:
    def info(self, msg: str) -> None:
        print(f"[test_robot_xml_static] {msg}")

    def warn(self, msg: str) -> None:
        print(f"[test_robot_xml_static] WARN: {msg}")

    def error(self, msg: str) -> None:
        print(f"[test_robot_xml_static] ERROR: {msg}", file=sys.stderr)


# ----------------------------------------------------------------------
# Prefix-strip (for wrapper-namespaced names)
# ----------------------------------------------------------------------


def _strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix) :] if prefix and s.startswith(prefix) else s


def _normalize_keys(
    facts: Dict[str, Dict[str, Any]],
    prefix: str,
) -> Dict[str, Dict[str, Any]]:
    """Replace ALL occurrences of ``prefix`` in keys and in nested
    name-bearing fields (``trnid_joint``, ``parent``, ``obj1``,
    ``obj2``).  Equality keys carry two names joined by ``" --> "``
    so we strip everywhere, not just the leading occurrence.
    """
    if not prefix:
        return facts
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in facts.items():
        new_v = dict(v)
        for field in ("trnid_joint", "parent", "obj1", "obj2"):
            if field in new_v and isinstance(new_v[field], str):
                new_v[field] = new_v[field].replace(prefix, "")
        out[k.replace(prefix, "")] = new_v
    return out


def _detect_common_prefix(facts: Dict[str, Dict[str, Any]]) -> str:
    """Find the longest leading substring shared by ALL keys, trimmed
    back to a sane separator boundary (underscore / slash / dot).
    Returns ``""`` when there's nothing useful to strip.
    """
    keys = list(facts.keys())
    if len(keys) < 2:
        return ""
    p = keys[0]
    for k in keys[1:]:
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
# MjModel introspection
# ----------------------------------------------------------------------


def _name_of(model: mujoco.MjModel, obj_type: int, idx: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, idx)
    return name if name else f"<{int(obj_type)}#{idx}>"


def _f(x: Any) -> float:
    return float(np.asarray(x).reshape(()))


def _vec(arr: Any, n: Optional[int] = None) -> List[float]:
    a = np.asarray(arr).reshape(-1)
    if n is not None:
        a = a[:n]
    return [float(v) for v in a]


def _opt_facts(m: mujoco.MjModel) -> Dict[str, Any]:
    """Snapshot global solver options.  Diff target #1 — a wrong
    timestep or integrator pollutes every downstream behavioral
    check, so flagging this loudest is justified.
    """
    o = m.opt
    return {
        "timestep": _f(o.timestep),
        "integrator": int(o.integrator),
        "solver": int(o.solver),
        "iterations": int(o.iterations),
        "ls_iterations": int(o.ls_iterations),
        "tolerance": _f(o.tolerance),
        "ls_tolerance": _f(o.ls_tolerance),
        "ccd_tolerance": _f(o.ccd_tolerance),
        "gravity": _vec(o.gravity, 3),
        "wind": _vec(o.wind, 3),
        "magnetic": _vec(o.magnetic, 3),
        "density": _f(o.density),
        "viscosity": _f(o.viscosity),
        "impratio": _f(o.impratio),
        "cone": int(o.cone),
        "jacobian": int(o.jacobian),
        "disableflags": int(o.disableflags),
        "enableflags": int(o.enableflags),
    }


def _joint_facts(m: mujoco.MjModel) -> Dict[str, Dict[str, Any]]:
    """Per-joint structural snapshot keyed by joint name.  Captures
    every attribute Newton's converter can author differently across
    runs.  ``qpos_addr`` / ``dof_addr`` / ``body_id`` are surfaced
    so id-shift regressions (joint reordering) are visible.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for j in range(m.njnt):
        name = _name_of(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        dof = int(m.jnt_dofadr[j])
        out[name] = {
            "type": int(m.jnt_type[j]),
            "axis": _vec(m.jnt_axis[j], 3),
            "limited": bool(m.jnt_limited[j]),
            "range": _vec(m.jnt_range[j], 2),
            "actfrclimited": bool(m.jnt_actfrclimited[j]) if hasattr(m, "jnt_actfrclimited") else None,
            "actfrcrange": _vec(m.jnt_actfrcrange[j], 2) if hasattr(m, "jnt_actfrcrange") else None,
            "armature": _f(m.dof_armature[dof]),
            "damping": _f(m.dof_damping[dof]),
            "frictionloss": _f(m.dof_frictionloss[dof]),
            "stiffness": _f(m.jnt_stiffness[j]),
            "qpos_addr": int(m.jnt_qposadr[j]),
            "dof_addr": dof,
            "body_id": int(m.jnt_bodyid[j]),
        }
    return out


def _actuator_facts(m: mujoco.MjModel) -> Dict[str, Dict[str, Any]]:
    """Per-actuator snapshot keyed by ``"<role>:<joint_name>[#order]"``.

    Newton's mjwarp converter doesn't author actuator names, so
    ``mj_id2name`` returns the empty string for every entry — and
    raw position-in-list ``id`` would shift between two MJCFs as
    soon as the actuator list grows (e.g. a new motor block per
    joint).  Keying on (role, joint_name) makes the diff stable
    across converter revisions.

    ``role`` is ``"position"`` when ``biastype == AFFINE`` (the
    converter emits AFFINE for position-PD actuators), else
    ``"motor"`` for the direct-torque actuators with default
    gain/bias.
    """
    out: Dict[str, Dict[str, Any]] = {}
    seen: Dict[Tuple[str, str], int] = {}
    for a in range(m.nu):
        explicit_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        trntype = int(m.actuator_trntype[a])
        trnid_joint: Optional[str] = None
        if trntype == int(mujoco.mjtTrn.mjTRN_JOINT):
            jid = int(m.actuator_trnid[a, 0])
            if 0 <= jid < m.njnt:
                trnid_joint = _name_of(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        biastype = int(m.actuator_biastype[a])
        role = "position" if biastype == int(mujoco.mjtBias.mjBIAS_AFFINE) else "motor"
        if explicit_name:
            key = explicit_name
        else:
            base = trnid_joint or f"<actuator#{a}>"
            key_pair = (role, base)
            bump = seen.get(key_pair, 0)
            seen[key_pair] = bump + 1
            key = f"{role}:{base}" + (f"#{bump}" if bump else "")
        out[key] = {
            "trntype": trntype,
            "trnid_joint": trnid_joint,
            "gear": _vec(m.actuator_gear[a], 6),
            # First 3 prm slots are where Newton's converter writes
            # (kp, 0, 0) / (0, -kp, -kd).  The other 7 are MuJoCo
            # defaults that rarely vary.
            "gainprm": _vec(m.actuator_gainprm[a], 3),
            "biasprm": _vec(m.actuator_biasprm[a], 3),
            "gaintype": int(m.actuator_gaintype[a]),
            "biastype": biastype,
            "dyntype": int(m.actuator_dyntype[a]),
            "forcelimited": bool(m.actuator_forcelimited[a]),
            "forcerange": _vec(m.actuator_forcerange[a], 2) if m.actuator_forcelimited[a] else None,
            "ctrllimited": bool(m.actuator_ctrllimited[a]),
            "ctrlrange": _vec(m.actuator_ctrlrange[a], 2) if m.actuator_ctrllimited[a] else None,
        }
    return out


def _equality_facts(m: mujoco.MjModel) -> Dict[str, Dict[str, Any]]:
    """Per-equality snapshot keyed by ``"<obj1_name> --> <obj2_name>"``.

    Equality constraints rarely have meaningful names, but the
    (obj1, obj2) pair is semantic — that's the mimic edge.
    Append ``" #N"`` to disambiguate the rare case of duplicate
    pairs.
    """
    out: Dict[str, Dict[str, Any]] = {}
    seen: Dict[Tuple[str, str], int] = {}
    for e in range(m.neq):
        eq_type = int(m.eq_type[e])
        if eq_type == int(mujoco.mjtEq.mjEQ_JOINT):
            obj_t = mujoco.mjtObj.mjOBJ_JOINT
        elif eq_type == int(mujoco.mjtEq.mjEQ_TENDON):
            obj_t = mujoco.mjtObj.mjOBJ_TENDON
        else:
            obj_t = mujoco.mjtObj.mjOBJ_BODY
        n1 = _name_of(m, obj_t, int(m.eq_obj1id[e]))
        n2 = _name_of(m, obj_t, int(m.eq_obj2id[e])) if int(m.eq_obj2id[e]) >= 0 else "<world>"
        key_pair = (n1, n2)
        bump = seen.get(key_pair, 0)
        seen[key_pair] = bump + 1
        key = f"{n1} --> {n2}" + (f" #{bump}" if bump else "")
        out[key] = {
            "type": eq_type,
            "obj1": n1,
            "obj2": n2,
            "data": _vec(m.eq_data[e], min(11, m.eq_data.shape[1])),
            "active": bool(m.eq_active0[e]),
            "solref": _vec(m.eq_solref[e], 2),
            "solimp": _vec(m.eq_solimp[e], 5),
        }
    return out


def _body_facts(m: mujoco.MjModel) -> Dict[str, Dict[str, Any]]:
    """Per-body mass/inertia snapshot.  World body excluded."""
    out: Dict[str, Dict[str, Any]] = {}
    for b in range(1, m.nbody):
        name = _name_of(m, mujoco.mjtObj.mjOBJ_BODY, b)
        out[name] = {
            "mass": _f(m.body_mass[b]),
            "diaginertia": _vec(m.body_inertia[b], 3),
            "ipos": _vec(m.body_ipos[b], 3),
            "iquat": _vec(m.body_iquat[b], 4),
            "parent": _name_of(m, mujoco.mjtObj.mjOBJ_BODY, int(m.body_parentid[b])),
            "rootid": int(m.body_rootid[b]),
        }
    return out


# ----------------------------------------------------------------------
# Diff helpers
# ----------------------------------------------------------------------

# Floats round-trip MJCF text at ~1e-6; tighter than that and every
# actuator diff would flag noise.
_REL_TOL = 1e-5
_ABS_TOL = 1e-7


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
    return a == b


def _diff_dict_of_dicts(
    a: Dict[str, Dict[str, Any]],
    b: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    keys_a = set(a)
    keys_b = set(b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    attr_diffs: List[Dict[str, Any]] = []
    for name in sorted(keys_a & keys_b):
        rec_a = a[name]
        rec_b = b[name]
        for attr in sorted(set(rec_a) | set(rec_b)):
            va = rec_a.get(attr, "<missing>")
            vb = rec_b.get(attr, "<missing>")
            if not _values_equal(va, vb):
                attr_diffs.append({"name": name, "attr": attr, "a": va, "b": vb})
    return {
        "only_in_a": only_a,
        "only_in_b": only_b,
        "attr_diffs": attr_diffs,
    }


def _diff_flat(a: Dict[str, Any], b: Dict[str, Any]) -> List[Dict[str, Any]]:
    diffs: List[Dict[str, Any]] = []
    for attr in sorted(set(a) | set(b)):
        va = a.get(attr, "<missing>")
        vb = b.get(attr, "<missing>")
        if not _values_equal(va, vb):
            diffs.append({"attr": attr, "a": va, "b": vb})
    return diffs


# ----------------------------------------------------------------------
# Suspicion heuristics — structural only
# ----------------------------------------------------------------------


def _flag_suspicious(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Distill the structural diff into a short list of highest-
    priority callouts.  Behavioral / dynamic flags live in
    ``test_robot_xml_dynamic.py`` — this function is structural only.
    """
    flags: List[Dict[str, Any]] = []

    # 1. opt.timestep
    for d in report.get("opt", []):
        if d["attr"] == "timestep":
            flags.append(
                {
                    "severity": "critical",
                    "kind": "opt.timestep",
                    "msg": (
                        f"opt.timestep differs: a={d['a']} b={d['b']}.  "
                        f"Every behavioral comparison is meaningless until "
                        f"these match."
                    ),
                }
            )
            break

    # 2. integrator / solver
    for d in report.get("opt", []):
        if d["attr"] in ("integrator", "solver"):
            flags.append(
                {
                    "severity": "critical",
                    "kind": f"opt.{d['attr']}",
                    "msg": (
                        f"opt.{d['attr']} differs: a={d['a']} b={d['b']}.  " f"Solvers will diverge on stiff dynamics."
                    ),
                }
            )

    # 3. Actuator gain class change.
    for d in report.get("actuators", {}).get("attr_diffs", []):
        if d["attr"] not in ("gainprm", "biasprm"):
            continue
        a = d["a"] if isinstance(d["a"], list) else None
        b = d["b"] if isinstance(d["b"], list) else None
        if a is None or b is None:
            continue
        if d["attr"] == "gainprm":
            za = abs(a[0]) < 1e-9
            zb = abs(b[0]) < 1e-9
            if za != zb:
                flags.append(
                    {
                        "severity": "high",
                        "kind": "actuator.gainprm zero↔nonzero",
                        "name": d["name"],
                        "msg": (
                            f"actuator {d['name']!r}: gainprm[0] (kp) flipped "
                            f"between zero and non-zero (a={a[0]} b={b[0]}). "
                            f"This decides whether the actuator applies any "
                            f"position-PD force at all."
                        ),
                    }
                )
                continue  # Don't double-flag the same actuator
            # Both nonzero — flag if they differ by more than 2×.
            # That's a 2-stop change in PD authority and it WILL
            # show up at runtime (10× error rejection at small
            # disturbances; saturation at different effort levels).
            ka = abs(float(a[0]))
            kb = abs(float(b[0]))
            if ka > 1e-9 and kb > 1e-9:
                ratio = max(ka, kb) / min(ka, kb)
                if ratio > 2.0:
                    flags.append(
                        {
                            "severity": "medium",
                            "kind": "actuator.gainprm ratio>2x",
                            "name": d["name"],
                            "msg": (
                                f"actuator {d['name']!r}: gainprm[0] (kp) "
                                f"differs by {ratio:.1f}× (a={ka:g} "
                                f"b={kb:g}).  PD authority diverges; runtime "
                                f"tracking will differ noticeably."
                            ),
                        }
                    )
        elif d["attr"] == "biasprm":
            # biasprm[2] is the kd term for AFFINE bias.  Flag a
            # > 2× divergence the same way as kp — kd controls
            # damping and a 2× swing is the difference between
            # ringing and quiet decay.  We skip biasprm[1] (the
            # -kp term that mirrors gainprm[0]) since gainprm has
            # already covered that.
            if len(a) < 3 or len(b) < 3:
                continue
            kda = abs(float(a[2]))
            kdb = abs(float(b[2]))
            za = kda < 1e-9
            zb = kdb < 1e-9
            if za != zb:
                flags.append(
                    {
                        "severity": "medium",
                        "kind": "actuator.biasprm zero↔nonzero",
                        "name": d["name"],
                        "msg": (
                            f"actuator {d['name']!r}: biasprm[2] (kd) "
                            f"flipped between zero and non-zero "
                            f"(a={a[2]} b={b[2]}).  One side has no "
                            f"velocity feedback term."
                        ),
                    }
                )
                continue
            if kda > 1e-9 and kdb > 1e-9:
                ratio = max(kda, kdb) / min(kda, kdb)
                if ratio > 2.0:
                    flags.append(
                        {
                            "severity": "medium",
                            "kind": "actuator.biasprm ratio>2x",
                            "name": d["name"],
                            "msg": (
                                f"actuator {d['name']!r}: biasprm[2] (kd) "
                                f"differs by {ratio:.1f}× (a={kda:g} "
                                f"b={kdb:g}).  Damping authority diverges; "
                                f"settling / overshoot will differ."
                            ),
                        }
                    )

    # 4. Equality presence
    eq_diff = report.get("equalities", {})
    for k in eq_diff.get("only_in_a", []):
        flags.append(
            {
                "severity": "high",
                "kind": "equality only_in_a",
                "name": k,
                "msg": f"equality {k!r} present in A but missing from B.",
            }
        )
    for k in eq_diff.get("only_in_b", []):
        flags.append(
            {
                "severity": "high",
                "kind": "equality only_in_b",
                "name": k,
                "msg": f"equality {k!r} present in B but missing from A.",
            }
        )

    # 5. Body mass > 1% drift
    for d in report.get("bodies", {}).get("attr_diffs", []):
        if d["attr"] != "mass":
            continue
        a = float(d["a"]) if isinstance(d["a"], (int, float)) else None
        b = float(d["b"]) if isinstance(d["b"], (int, float)) else None
        if a is None or b is None or a <= 0 or b <= 0:
            continue
        rel = abs(a - b) / max(abs(a), abs(b))
        if rel > 0.01:
            flags.append(
                {
                    "severity": "medium",
                    "kind": "body.mass",
                    "name": d["name"],
                    "msg": (
                        f"body {d['name']!r}: mass differs by {rel*100:.2f}% "
                        f"(a={a:g} b={b:g}). Inertia / collision response "
                        f"will diverge."
                    ),
                }
            )
    return flags


# ----------------------------------------------------------------------
# Top-level report assembly
# ----------------------------------------------------------------------


def _build_report(
    xml_a: str,
    xml_b: str,
    logger: _Logger,
    strip_prefix_a: str = "",
    strip_prefix_b: str = "",
    auto_strip_prefix: bool = True,
) -> Dict[str, Any]:
    logger.info(f"loading A: {xml_a}")
    model_a = mujoco.MjModel.from_xml_path(xml_a)
    logger.info(f"loading B: {xml_b}")
    model_b = mujoco.MjModel.from_xml_path(xml_b)

    facts_a = {
        "opt": _opt_facts(model_a),
        "joints": _joint_facts(model_a),
        "actuators": _actuator_facts(model_a),
        "equalities": _equality_facts(model_a),
        "bodies": _body_facts(model_a),
    }
    facts_b = {
        "opt": _opt_facts(model_b),
        "joints": _joint_facts(model_b),
        "actuators": _actuator_facts(model_b),
        "equalities": _equality_facts(model_b),
        "bodies": _body_facts(model_b),
    }

    # Auto-detect prefixes when the joint name sets are completely
    # disjoint (a strong signal that names are namespaced on one side).
    if auto_strip_prefix:
        names_a = set(facts_a["joints"])
        names_b = set(facts_b["joints"])
        if names_a and names_b and not (names_a & names_b):
            cand_a = _detect_common_prefix(facts_a["joints"])
            cand_b = _detect_common_prefix(facts_b["joints"])
            if not strip_prefix_a and cand_a and not any(cand_a in n for n in names_b):
                strip_prefix_a = cand_a
                logger.info(f"auto-detected strip_prefix_a={cand_a!r}")
            if not strip_prefix_b and cand_b and not any(cand_b in n for n in names_a):
                strip_prefix_b = cand_b
                logger.info(f"auto-detected strip_prefix_b={cand_b!r}")

    if strip_prefix_a:
        for sect in ("joints", "actuators", "equalities", "bodies"):
            facts_a[sect] = _normalize_keys(facts_a[sect], strip_prefix_a)
    if strip_prefix_b:
        for sect in ("joints", "actuators", "equalities", "bodies"):
            facts_b[sect] = _normalize_keys(facts_b[sect], strip_prefix_b)

    summary = {
        "xml_a": os.path.abspath(xml_a),
        "xml_b": os.path.abspath(xml_b),
        "strip_prefix_a": strip_prefix_a,
        "strip_prefix_b": strip_prefix_b,
        "nq_a": int(model_a.nq),
        "nq_b": int(model_b.nq),
        "nv_a": int(model_a.nv),
        "nv_b": int(model_b.nv),
        "njnt_a": int(model_a.njnt),
        "njnt_b": int(model_b.njnt),
        "nu_a": int(model_a.nu),
        "nu_b": int(model_b.nu),
        "neq_a": int(model_a.neq),
        "neq_b": int(model_b.neq),
        "nbody_a": int(model_a.nbody),
        "nbody_b": int(model_b.nbody),
    }
    report: Dict[str, Any] = {
        "summary": summary,
        "opt": _diff_flat(facts_a["opt"], facts_b["opt"]),
        "joints": _diff_dict_of_dicts(facts_a["joints"], facts_b["joints"]),
        "actuators": _diff_dict_of_dicts(facts_a["actuators"], facts_b["actuators"]),
        "equalities": _diff_dict_of_dicts(facts_a["equalities"], facts_b["equalities"]),
        "bodies": _diff_dict_of_dicts(facts_a["bodies"], facts_b["bodies"]),
    }
    report["suspicious"] = _flag_suspicious(report)
    return report


# ----------------------------------------------------------------------
# Pretty-printer
# ----------------------------------------------------------------------


def _print_summary(report: Dict[str, Any], logger: _Logger) -> None:
    s = report["summary"]
    logger.info("=" * 72)
    logger.info(f"A: {s['xml_a']}")
    logger.info(f"B: {s['xml_b']}")
    if s.get("strip_prefix_a") or s.get("strip_prefix_b"):
        logger.info(f"prefix-strip:  a={s.get('strip_prefix_a','')!r}  " f"b={s.get('strip_prefix_b','')!r}")
    logger.info("-" * 72)
    logger.info(
        f"counts (a/b):  nq {s['nq_a']}/{s['nq_b']}  nv {s['nv_a']}/{s['nv_b']}  "
        f"njnt {s['njnt_a']}/{s['njnt_b']}  nu {s['nu_a']}/{s['nu_b']}  "
        f"neq {s['neq_a']}/{s['neq_b']}  nbody {s['nbody_a']}/{s['nbody_b']}"
    )
    logger.info(f"opt diffs:        {len(report['opt'])}")
    if report["opt"]:
        for d in report["opt"][:6]:
            logger.info(f"    opt.{d['attr']}:  a={d['a']!r}  b={d['b']!r}")
        if len(report["opt"]) > 6:
            logger.info(f"    ... and {len(report['opt']) - 6} more (see JSON)")
    for sect in ("joints", "actuators", "equalities", "bodies"):
        d = report[sect]
        logger.info(
            f"{sect:12s} only_in_a={len(d['only_in_a'])}  "
            f"only_in_b={len(d['only_in_b'])}  "
            f"attr_diffs={len(d['attr_diffs'])}"
        )

    flags = report.get("suspicious", [])
    logger.info("-" * 72)
    if not flags:
        logger.info("SUSPICIOUS: 0 — nothing stands out.  See JSON for full diff.")
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
    parser.add_argument("--xml-a", required=True, help="Path to MJCF A.")
    parser.add_argument("--xml-b", required=True, help="Path to MJCF B (the reference).")
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
        help=(
            "Strip this leading namespace from every joint / body / "
            "actuator name in A before diffing.  Auto-detected when "
            "the two name sets are disjoint."
        ),
    )
    parser.add_argument(
        "--strip-prefix-b",
        default=None,
        metavar="PREFIX",
        help="Same as --strip-prefix-a, applied to B.",
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
            xml_b=args.xml_b,
            logger=logger,
            strip_prefix_a=args.strip_prefix_a or "",
            strip_prefix_b=args.strip_prefix_b or "",
            auto_strip_prefix=not args.no_auto_strip_prefix,
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
