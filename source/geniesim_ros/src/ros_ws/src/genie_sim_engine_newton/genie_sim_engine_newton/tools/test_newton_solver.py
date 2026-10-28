#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Standalone Newton robot-solver test CLI.

Exercises a :class:`SolverAdapter` without ROS, without
``EngineSession``, without ``simulation_app`` — just ``newton`` +
``warp`` + ``pxr.Usd`` (the last one only if you pass ``--usd``).

Use cases:
  * Validate a new adapter in isolation before wiring it through ROS.
  * Tune PD gains (``--pd-ke`` / ``--pd-kd``) on the mjwarp path
    without rebuilding the ROS launch.
  * Compare mjwarp vs. featherstone stability on the same robot.
  * Sanity-check init poses for new scenes before they hit
    ``app.launch.py``.

Examples:

    # MuJoCo-Warp on a URDF, head-on with the Newton GL viewer
    test_newton_solver.py --urdf /path/to/G2.urdf \\
        --solver mujoco-warp --substeps 5 --physics-hz 100 \\
        --init-pose "arm_l_joint2=-45,arm_l_joint4=-75" \\
        --log-joints "body_joint1,arm_l_joint2" \\
        --viewer

    # Featherstone (vbd path) on a USD scene, headless, 500 frames
    test_newton_solver.py --usd /geniesim_assets/scenes/g2.usda \\
        --solver vbd --steps 500 --substeps 10

The captured CUDA graph is NOT used here — each substep does its
``wp.launch`` / ``solver.step`` ops directly.  Stepping is roughly
10x slower than the production engine but the tool stays single-file
diagnosable.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional

# Make ``engine.newton.adapters`` importable when this script is run
# directly out of the repo without the genie_sim_engine ROS package
# being installed.  ``scripts/`` is the package root.
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import newton  # noqa: E402
import warp as wp  # noqa: E402

from genie_sim_engine_newton.engine.newton.adapters import make_adapter  # noqa: E402


class _StdoutLogger:
    """Logger compatible with the SolverAdapter hook signature.

    Avoids importing ``common.session.SimpleLogger`` (which deferred-imports
    the ``genie_sim_engine_py`` ROS pybind module).
    """

    def __init__(self, prefix: str = "test-newton") -> None:
        self._prefix = prefix

    def info(self, m: str) -> None:
        print(f"[{self._prefix}] {m}", flush=True)

    def warn(self, m: str) -> None:
        print(f"[{self._prefix}] WARN: {m}", flush=True)

    def error(self, m: str) -> None:
        print(f"[{self._prefix}] ERROR: {m}", flush=True)


# ----------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--urdf", help="path to a URDF file (uses Newton's URDF loader)")
    src.add_argument("--usd", help="path to a USD/USDA scene (uses Newton's add_usd via pxr)")
    p.add_argument(
        "--solver",
        default="mujoco-warp",
        choices=["mujoco-warp", "mujoco_warp", "fsvbd", "featherstone", "avbd"],
        help="which adapter to instantiate (default: mujoco-warp)",
    )
    p.add_argument("--steps", type=int, default=200, help="number of frames to simulate (default: 200)")
    p.add_argument("--substeps", type=int, default=5, help="substeps per frame (default: 5)")
    p.add_argument("--physics-hz", type=float, default=100.0, help="frames per simulated second (default: 100)")
    p.add_argument("--viewer", action="store_true", help="open the Newton GL viewer (needs $DISPLAY)")
    p.add_argument(
        "--pd-ke",
        type=float,
        default=0.0,
        help="override mjwarp PD stiffness (N·m/rad).  Ignored by featherstone (which uses selective passive PD).",
    )
    p.add_argument("--pd-kd", type=float, default=0.0, help="override mjwarp PD damping (N·m·s/rad)")
    p.add_argument(
        "--physics-params",
        type=str,
        default="",
        help=(
            "path to physics_params.yaml.  When set, the mjwarp adapter "
            "classifies each DOF via common.joint_classification and "
            "drives kp/kd/max_effort from the per-class tables "
            "(articulation_view_runtime.default / .chassis_drive / "
            ".chassis_steer + usd_drive_api.gripper.master_*) — same "
            "gains the isaac_newton wrapper applies via "
            "ArticulationView.set_dof_stiffnesses at runtime.  Use this "
            "to reproduce production engine behaviour from the CLI.  "
            "When unset, the ``effort × 10`` adapter heuristic "
            "runs."
        ),
    )
    p.add_argument(
        "--save-mjcf",
        default="",
        metavar="PATH",
        help=(
            "mjwarp only: write the compiled mujoco MJCF (joints, actuators "
            "with gainprm/biasprm, contact params, equality constraints, "
            "joint limits) to PATH after build_solver.  Lets you diff "
            "against a reference MJCF, load in `mujoco.viewer.launch_passive("
            "mujoco.MjModel.from_xml_path(PATH))` for independent visual "
            "inspection, or validate with `mjpython` tooling.  Matched to "
            "the production engine's `robot_runtime.xml` sidecar."
        ),
    )
    p.add_argument(
        "--init-pose",
        default="",
        help=(
            "comma-separated joint init positions, e.g. "
            "'arm_l_joint2=-45,arm_l_joint4=-75'.  Revolute joints are degrees, "
            "prismatic joints are metres (same convention as scene YAML)."
        ),
    )
    p.add_argument(
        "--log-joints",
        default="",
        help="comma-separated joint names to print q every 10 frames",
    )
    p.add_argument(
        "--fix-base",
        action="store_true",
        default=True,
        help="treat the robot's root joint as world-fixed (default: True; pass --no-fix-base to disable)",
    )
    p.add_argument("--no-fix-base", action="store_false", dest="fix_base")
    p.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "load the model, run all sanity checks (zero-mass clamp, "
            "material overrides, init-pose vs. joint limits, NaN scan), "
            "log a summary, and exit without stepping the solver"
        ),
    )
    p.add_argument(
        "--validate-pd",
        action="store_true",
        help=(
            "after build_solver, dump every layer of the stiffness / "
            "damping / effort-limit pipeline (Newton model arrays, "
            "mjwarp actuator gainprm/biasprm, per-joint actfrcrange) "
            "and flag suspicious combinations.  Implicit in "
            "--validate-only.  Use this to diagnose 'jelly' (effort "
            "clamp wins) and 'oscillation' (kd=0) symptoms before "
            "spending a step-loop run."
        ),
    )
    p.add_argument(
        "--exit-on-pd-warning",
        action="store_true",
        help=(
            "with --validate-pd, abort (exit code 4) if any DOF is "
            "flagged as suspicious instead of continuing.  Useful in CI "
            "/ regression scripts."
        ),
    )
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the JIT-warmup uncaptured frame (default: warm up)",
    )
    p.add_argument(
        "--no-sanitize",
        action="store_true",
        help=(
            "skip the production-equivalent sanitizers (zero-mass clamp, "
            "shape/soft material overrides) so you can reproduce raw "
            "Newton behavior. Mostly useful for triaging whether a NaN is "
            "caused by something we sanitize."
        ),
    )
    p.add_argument(
        "--capture",
        action="store_true",
        help=(
            "wrap the inner substep block in a CUDA graph capture (matches "
            "engine.py:_capture_graph in production).  All ``args.substeps`` "
            "substeps run as one wp.capture_launch per frame instead of "
            "per-substep dispatch.  Use to reproduce production-only "
            "behavior — the captured graph baked in pointer addresses and "
            "the SolverMuJoCo contact-buffer sizes at capture time, so any "
            "production explosion that disappears here vs. with this flag "
            "implicates the graph (most often a contact-buffer overrun)."
        ),
    )
    # ---- wbc-style command sweep --------------------------------------
    p.add_argument(
        "--cmd-mode",
        choices=["hold", "wbc"],
        default="hold",
        help=(
            "how to drive the target buffer during the step loop. "
            "'hold' (default) keeps the init pose; 'wbc' generates random "
            "per-group goals like genie_sim_bringup wbc_cmds.py does over "
            "ROS — useful to smoke-test /joint_command response without ROS."
        ),
    )
    p.add_argument("--body-deg", type=float, default=3.0, help="wbc body joint excursion ±deg (default 3)")
    p.add_argument("--head-deg", type=float, default=15.0, help="wbc head joint excursion ±deg (default 15)")
    p.add_argument("--arm-deg", type=float, default=20.0, help="wbc arm joint excursion ±deg (default 20)")
    p.add_argument(
        "--chassis-deg",
        type=float,
        default=30.0,
        help="wbc chassis steering excursion ±deg from neutral (default 30)",
    )
    p.add_argument(
        "--with-chassis",
        dest="with_chassis",
        action="store_true",
        default=True,
        help="wbc: include chassis steering joints (default: on)",
    )
    p.add_argument(
        "--no-with-chassis", dest="with_chassis", action="store_false", help="wbc: exclude chassis steering joints"
    )
    p.add_argument(
        "--with-gripper",
        dest="with_gripper",
        action="store_true",
        default=True,
        help="wbc: include gripper master joints (default: on; swiftpicker has closed-loop constraints Featherstone doesn't enforce natively, so excluding may improve stability)",
    )
    p.add_argument("--no-with-gripper", dest="with_gripper", action="store_false", help="wbc: exclude gripper joints")
    p.add_argument("--cmd-tol", type=float, default=0.05, help="wbc goal-reached tolerance rad/m (default 0.05)")
    p.add_argument(
        "--goal-timeout",
        type=float,
        default=5.0,
        help="wbc: sim seconds before advancing to next goal even if not all converge (default 5)",
    )
    p.add_argument(
        "--no-passive-pd",
        action="store_true",
        help=(
            "featherstone only: disable the selective PD applied to passive "
            "body/head joints (FeatherstoneAdapter.post_joint_map).  Useful "
            "when stress-testing /joint_command response — selective PD with "
            "default ke=50000 is rigid enough that arm-chain coupling rings "
            "the body chain to NaN under multi-joint wbc sweeps."
        ),
    )
    p.add_argument("--seed", type=int, default=None, help="wbc random seed for reproducible sequences")
    return p.parse_args()


# ----------------------------------------------------------------------
# Production-equivalent constants (engine.py: NewtonHeadlessEngine)
# ----------------------------------------------------------------------

# Zero-mass body clamp.  USD robots sometimes have massless intermediate
# links (spine segments, structural bridges) that PhysX handles via
# virtual mass injection, but Newton's Featherstone inverts the spatial
# inertia matrix directly — zero mass → singular → NaN on frame 1.
_MIN_MASS = 1e-4  # 0.1 g
_MIN_INERTIA = 1e-8  # kg·m²

# Soft-contact material defaults (cloth-particle ↔ rigid-body contact).
_SOFT_CONTACT_KE = 1e4
_SOFT_CONTACT_KD = 1e-2
_SOFT_CONTACT_MU = 0.25

# Robot shape material defaults (rigid-rigid contact).
_ROBOT_KE = 5e4
_ROBOT_KD = 1e-3
_ROBOT_KU = 1.5


# ----------------------------------------------------------------------
# Joint name → DOF index map (mirrors topology.py _build_joint_map)
# ----------------------------------------------------------------------

_JT_FIXED = 3
_JT_FREE = 4
_JT_REVOLUTE = 1
_SKIP_PREFIXES = ("tn__",)
_SKIP_EXACT = {"root_joint"}


def _build_joint_map(model, logger) -> Dict[str, int]:
    """Return {joint_short_name: dof_index} for command-driveable joints.

    Mirrors ``_TopologyMixin._build_joint_map`` in
    ``scripts/engine/newton/topology.py``.
    """
    labels = getattr(model, "joint_label", []) or []
    types_arr = getattr(model, "joint_type", None)
    q_start_arr = getattr(model, "joint_q_start", None)
    if types_arr is None or q_start_arr is None:
        logger.warn("joint_type / joint_q_start arrays missing — joint map empty")
        return {}
    types_np = types_arr.numpy()
    q_start_np = q_start_arr.numpy()
    name_to_dof: Dict[str, int] = {}
    for ji, label in enumerate(labels):
        if not label:
            continue
        short = label.rsplit("/", 1)[-1] if "/" in label else label
        jtype = int(types_np[ji])
        qstart = int(q_start_np[ji])
        if jtype in (_JT_FIXED, _JT_FREE):
            continue
        if any(short.startswith(p) for p in _SKIP_PREFIXES) or short in _SKIP_EXACT:
            continue
        name_to_dof[short] = qstart
    return name_to_dof


def _joint_type_map(model) -> Dict[str, int]:
    """Return {joint_short_name: jointtype_int}."""
    labels = getattr(model, "joint_label", []) or []
    types_arr = getattr(model, "joint_type", None)
    if types_arr is None:
        return {}
    types_np = types_arr.numpy()
    out: Dict[str, int] = {}
    for ji, label in enumerate(labels):
        if not label:
            continue
        short = label.rsplit("/", 1)[-1] if "/" in label else label
        out[short] = int(types_np[ji])
    return out


# ----------------------------------------------------------------------
# Body map + mimic map (mirrors topology.py)
# ----------------------------------------------------------------------


def _build_body_map(model) -> List[str]:
    """List of body labels, mirrors topology.py _build_body_map."""
    return list(getattr(model, "body_label", []) or [])


def _build_mimic_map(stage, logger) -> Dict[str, List]:
    """Parse mimic relations from a USD stage.

    Returns ``{master_name: [(follower_name, mult, offset), ...]}``.
    Empty dict if ``stage`` is None (URDF-loaded models bypass USD).
    """
    if stage is None:
        return {}
    try:
        # Same parser the production engine uses.
        from genie_sim_engine_core.engine._mimic import parse_mimic

        return parse_mimic(stage, logger)
    except Exception as exc:  # noqa: BLE001
        logger.warn(f"_build_mimic_map: {exc}")
        return {}


def _build_mimic_map_from_urdf(urdf_path: str, logger) -> Dict[str, List]:
    """Parse ``<joint><mimic .../></joint>`` from a URDF and return the
    same shape as :func:`_build_mimic_map`.

    Production ``parse_mimic`` only reads ``NewtonMimicAPI`` from a live
    USD stage; the URDF path has no stage, so the standalone tool needs
    a separate parser when fed ``--urdf``.  Without this, commanding any
    gripper master via ``--cmd-mode wbc`` sends INDEPENDENT random goals
    to the (kinematically-coupled) follower joints, and Featherstone's
    constraint solver instantly produces NaN.
    """
    import xml.etree.ElementTree as ET

    followers: Dict[str, List[tuple]] = {}
    try:
        tree = ET.parse(urdf_path)
    except (ET.ParseError, OSError) as exc:
        logger.warn(f"_build_mimic_map_from_urdf: {exc}")
        return {}
    for joint in tree.iter("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        fname = joint.attrib.get("name", "")
        master = mimic.attrib.get("joint", "")
        if not fname or not master:
            continue
        try:
            mult = float(mimic.attrib.get("multiplier", "1.0"))
            off = float(mimic.attrib.get("offset", "0.0"))
        except ValueError:
            continue
        followers.setdefault(master, []).append((fname, mult, off))
    if followers:
        n_followers = sum(len(v) for v in followers.values())
        logger.info(f"mimic (URDF): {len(followers)} master(s) -> {n_followers} follower(s)")
    return followers


# ----------------------------------------------------------------------
# Production-equivalent sanitizers (mirrors lifecycle.py:161-208)
# ----------------------------------------------------------------------


def _sanitize_model(model, logger) -> None:
    """Clamp zero-mass bodies and write robot-grade material defaults.

    This MUST run before the solver is constructed (Featherstone's
    SolverFeatherstone caches an inverse spatial inertia matrix per body
    at the first step; a zero-mass body is singular and the inversion
    yields NaN, propagating to every state read off that body downstream).

    Mirrors ``lifecycle.py:_build`` lines 161-208.
    """
    import warp as wp
    import numpy as np

    # ----- Zero-mass body clamp (Featherstone NaN protection) -----
    if getattr(model, "body_mass", None) is not None:
        body_mass_np = model.body_mass.numpy().copy()
        zero_idx = (body_mass_np < _MIN_MASS).nonzero()[0]
        if len(zero_idx) > 0:
            labels = getattr(model, "body_label", []) or []
            sample = []
            for i in zero_idx[:8]:
                lbl = labels[int(i)] if int(i) < len(labels) else ""
                short = lbl.rsplit("/", 1)[-1] if "/" in lbl else lbl
                sample.append(f"{int(i)}({short or '?'})")
            logger.info(
                f"sanitize: clamping {len(zero_idx)} zero-mass body/bodies "
                f"to {_MIN_MASS} kg: {sample}" + ("..." if len(zero_idx) > 8 else "")
            )
            body_mass_np[zero_idx] = _MIN_MASS
            model.body_mass = wp.array(body_mass_np, dtype=wp.float32, device=model.device)
            if getattr(model, "body_inertia", None) is not None:
                body_inertia_np = model.body_inertia.numpy().copy()
                for i in zero_idx:
                    if np.max(np.abs(body_inertia_np[i])) < _MIN_INERTIA:
                        # Solid sphere: I = 2/5·m·r², r = 5 mm
                        r = 0.005
                        I_val = float((2.0 / 5.0) * _MIN_MASS * r * r)
                        body_inertia_np[i] = np.eye(3, dtype=np.float32) * I_val
                model.body_inertia = wp.array(body_inertia_np, dtype=wp.mat33, device=model.device)

    # ----- Soft-contact material (cloth ↔ rigid; benign for rigid-only) -----
    if hasattr(model, "soft_contact_ke"):
        model.soft_contact_ke = _SOFT_CONTACT_KE
    if hasattr(model, "soft_contact_kd"):
        model.soft_contact_kd = _SOFT_CONTACT_KD
    if hasattr(model, "soft_contact_mu"):
        model.soft_contact_mu = _SOFT_CONTACT_MU

    # ----- Robot shape material (rigid ↔ rigid contact) -----
    if (
        getattr(model, "shape_material_ke", None) is not None
        and getattr(model, "shape_material_kd", None) is not None
        and getattr(model, "shape_material_mu", None) is not None
    ):
        shape_ke = model.shape_material_ke.numpy().copy()
        shape_kd = model.shape_material_kd.numpy().copy()
        shape_mu = model.shape_material_mu.numpy().copy()
        shape_ke[...] = _ROBOT_KE
        shape_kd[...] = _ROBOT_KD
        shape_mu[...] = _ROBOT_KU
        model.shape_material_ke = wp.array(shape_ke, dtype=model.shape_material_ke.dtype, device=model.device)
        model.shape_material_kd = wp.array(shape_kd, dtype=model.shape_material_kd.dtype, device=model.device)
        model.shape_material_mu = wp.array(shape_mu, dtype=model.shape_material_mu.dtype, device=model.device)

    logger.info(
        f"sanitize: materials set "
        f"(soft ke={_SOFT_CONTACT_KE:g}/kd={_SOFT_CONTACT_KD:g}/mu={_SOFT_CONTACT_MU}, "
        f"shape ke={_ROBOT_KE:g}/kd={_ROBOT_KD:g}/mu={_ROBOT_KU})"
    )


# ----------------------------------------------------------------------
# Joint-limit validation
# ----------------------------------------------------------------------


def _validate_init_pose_against_limits(
    model,
    joint_name_to_dof: Dict[str, int],
    init_pose_str: str,
    logger,
) -> List[str]:
    """Refuse init values that exceed the URDF's joint limits.

    Returns a list of human-readable violation strings (empty = OK).

    Featherstone's joint-limit constraint generates an impulsive
    response when the pose starts outside the limit; mjwarp clamps
    silently.  We surface it here as a hard error so it's caught at
    load time rather than as a frame-1 NaN downstream.
    """
    import math

    if not init_pose_str.strip():
        return []
    if model.joint_limit_lower is None or model.joint_limit_upper is None:
        return []
    type_map = _joint_type_map(model)
    lower = model.joint_limit_lower.numpy()
    upper = model.joint_limit_upper.numpy()
    DEG2RAD = math.pi / 180.0
    violations: List[str] = []
    for entry in init_pose_str.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        name, val_str = entry.split("=", 1)
        name = name.strip()
        try:
            val = float(val_str)
        except ValueError:
            continue
        idx = joint_name_to_dof.get(name)
        if idx is None or idx >= len(lower):
            continue
        if type_map.get(name) == _JT_REVOLUTE:
            val = val * DEG2RAD
        lo = float(lower[idx])
        hi = float(upper[idx])
        if val < lo - 1e-6 or val > hi + 1e-6:
            unit = "rad" if type_map.get(name) == _JT_REVOLUTE else "m"
            violations.append(f"{name}: requested {val:+.4f}{unit} outside limit " f"[{lo:+.4f}, {hi:+.4f}]{unit}")
    return violations


# ----------------------------------------------------------------------
# PD-params validator (catches "robot is jelly" misconfigurations)
# ----------------------------------------------------------------------


def _validate_pd_params(model, adapter, joint_name_to_dof, logger) -> int:
    """Dump every layer of stiffness / damping / effort the model exposes
    so jelly-vs-runaway disagreements can be pinpointed.

    Three layers exist, and they must AGREE.  When they don't, the
    user-visible symptom is "robot is jelly" (effort clamp wins) or
    "robot explodes / oscillates" (gain wins, damping too low):

      1. Newton model arrays — what the adapter's ``prepare_model``
         wrote.  ``joint_target_mode`` / ``joint_target_ke`` /
         ``joint_target_kd`` / ``joint_effort_limit``.

      2. mjwarp ``mjw_model.actuator_gainprm`` / ``actuator_biasprm``
         — what ``_convert_to_mjc`` baked from layer 1.  This is what
         ``solver.step`` actually reads each substep for the actuator
         torque.

      3. mjwarp ``mjw_model.jnt_actfrcrange`` /
         ``mjw_model.jnt_actfrclimited`` — the per-joint clamp on the
         combined P+D output.  When ``actfrclimited=True`` and
         ``range=(-0,+0)`` the joint can produce ZERO torque no matter
         what gainprm says — the canonical jelly footgun.

    Returns the number of suspicious DOFs (those flagged with a `!`)
    so callers can ``return 4`` and exit non-zero in --validate-only
    mode.  Logs are human-readable; the per-DOF table is sorted by
    DOF index so the order matches /joint_command index assignment.
    """
    import numpy as np

    suspicious = 0

    # ---- Layer 1: Newton model arrays ---------------------------------
    if model.joint_target_mode is None:
        logger.warn("[validate-pd] model.joint_target_mode is None — no PD configured")
        return 0
    mode = model.joint_target_mode.numpy()
    ke = model.joint_target_ke.numpy() if model.joint_target_ke is not None else np.zeros_like(mode, dtype=float)
    kd = model.joint_target_kd.numpy() if model.joint_target_kd is not None else np.zeros_like(mode, dtype=float)
    effort = (
        model.joint_effort_limit.numpy()
        if getattr(model, "joint_effort_limit", None) is not None
        else np.zeros_like(mode, dtype=float)
    )

    mode_names = {0: "NONE", 1: "POSITION", 2: "VELOCITY", 5: "POS+VEL"}
    logger.info(f"=== validate-pd  layer 1  Newton model arrays  ({len(joint_name_to_dof)} controllable DOF(s)) ===")
    logger.info(f"  {'joint':<40s} {'dof':>3s} {'mode':<8s} {'ke':>9s} {'kd':>7s} {'effort':>9s}  flags")
    for name, dof_idx in sorted(joint_name_to_dof.items(), key=lambda kv: kv[1]):
        m = int(mode[dof_idx]) if dof_idx < len(mode) else -1
        k = float(ke[dof_idx]) if dof_idx < len(ke) else 0.0
        d = float(kd[dof_idx]) if dof_idx < len(kd) else 0.0
        e = float(effort[dof_idx]) if dof_idx < len(effort) else 0.0
        flags = []
        if m == 0:
            flags.append("mode=NONE → no actuator emitted, joint floppy")
        if k <= 0:
            flags.append("ke=0 → no position spring")
        if d <= 0:
            flags.append("kd=0 → no velocity damping → oscillation")
        if 0 <= e <= 1e-6:
            flags.append("effort=0 → torque clamped to zero → JELLY")
        if flags:
            suspicious += 1
        logger.info(
            f"  {name:<40s} {dof_idx:3d} {mode_names.get(m, str(m)):<8s} "
            f"{k:9.1f} {d:7.1f} {e:9.1f}  {('!  ' + ' ; '.join(flags)) if flags else ''}"
        )

    # ---- Layer 2 + 3: mjwarp's gainprm / biasprm / actfrcrange --------
    solver = getattr(adapter, "_solver", None)
    mjw = getattr(solver, "mjw_model", None) if solver is not None else None
    if mjw is None:
        logger.info("[validate-pd] no mjw_model on adapter (featherstone path?) — layer 2/3 skipped")
        return suspicious

    def _nworld_slice(arr):
        # mjw_model arrays are shaped (nworld, ...) per the mjwarp
        # batched-worlds convention.  We always want world 0.
        np_arr = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
        return np_arr[0] if np_arr.ndim > 1 and np_arr.shape[0] >= 1 else np_arr

    try:
        gainprm = _nworld_slice(mjw.actuator_gainprm)
        biasprm = _nworld_slice(mjw.actuator_biasprm)
    except Exception as exc:
        logger.warn(f"[validate-pd] couldn't read actuator_gainprm/biasprm: {exc}")
        gainprm = biasprm = None

    if gainprm is not None and biasprm is not None:
        # mujoco's POSITION-mode shortcut:
        #   gainprm[0] = kp,   biasprm = (0, -kp, -kd, 0…)
        logger.info(f"=== validate-pd  layer 2  mjwarp actuator gainprm/biasprm  " f"({len(gainprm)} actuator(s)) ===")
        logger.info(f"  {'idx':>3s} {'kp(gain[0])':>11s} {'-kp(bias[1])':>13s} {'-kd(bias[2])':>13s}  flags")
        for ai in range(len(gainprm)):
            kp = float(gainprm[ai][0])
            nkp = float(biasprm[ai][1])
            nkd = float(biasprm[ai][2])
            flags = []
            if abs(kp) < 1e-6:
                flags.append("kp=0 → no position torque")
            if abs(nkd) < 1e-6:
                flags.append("biasprm[2]=0 → no damping (oscillation)")
            if flags:
                suspicious += 1
            logger.info(
                f"  {ai:3d} {kp:11.1f} {nkp:13.1f} {nkd:13.1f}  " f"{('!  ' + ' ; '.join(flags)) if flags else ''}"
            )

    try:
        actfrcrange = _nworld_slice(mjw.jnt_actfrcrange)
        actfrclim = _nworld_slice(mjw.jnt_actfrclimited)
    except Exception as exc:
        logger.warn(f"[validate-pd] couldn't read jnt_actfrcrange: {exc}")
        return suspicious

    logger.info(
        f"=== validate-pd  layer 3  mjwarp per-joint actfrcrange (effort clamp)  " f"({len(actfrcrange)} joint(s)) ==="
    )
    logger.info(f"  {'idx':>3s} {'limited':>7s} {'lo':>10s} {'hi':>10s}  {'kp*0.1rad':>10s}  flags")
    # We also need kp per joint for the saturation check.  The mjwarp
    # gainprm we just read is per-actuator; mapping actuator → joint is
    # 1:1 for POSITION-mode actuators on revolute / prismatic joints
    # (the only kind we emit here).  Treat ji == ai for the check.
    for ji in range(len(actfrcrange)):
        lo = float(actfrcrange[ji][0])
        hi = float(actfrcrange[ji][1])
        lim = bool(actfrclim[ji])
        eff_budget = min(abs(lo), abs(hi)) if lim else float("inf")
        # Saturation check: does kp * 0.1 rad already eat the entire
        # effort budget?  0.1 rad ≈ 6° is a small-motion test — if even
        # that small an error saturates, normal /joint_command goals
        # (often 0.5+ rad) will permanently clamp the actuator and the
        # robot looks soft.
        kp_for_joint = float(gainprm[ji][0]) if (gainprm is not None and ji < len(gainprm)) else 0.0
        demand_at_small_err = kp_for_joint * 0.1
        flags = []
        if lim and (abs(lo) < 1e-6 and abs(hi) < 1e-6):
            flags.append("limited=True, range=(0,0) → ZERO ACTUATOR FORCE → JELLY")
        elif lim and (hi - lo) < 10.0:
            flags.append(f"limited=True, very tight range ({lo:+.2f}, {hi:+.2f})")
        if lim and kp_for_joint > 0 and demand_at_small_err > 2.0 * eff_budget:
            flags.append(
                f"PD SATURATION → kp*0.1rad={demand_at_small_err:.0f} >> "
                f"effort={eff_budget:.0f} → joint clamps at ±{eff_budget:.0f} N·m → "
                f"JELLY; lower kp or raise effort_limit"
            )
        if flags:
            suspicious += 1
        logger.info(
            f"  {ji:3d} {str(lim):>7s} {lo:10.2f} {hi:10.2f}  {demand_at_small_err:10.0f}  "
            f"{('!  ' + ' ; '.join(flags)) if flags else ''}"
        )

    if suspicious:
        logger.warn(
            f"[validate-pd] {suspicious} suspicious entries above; fix in "
            f"adapter.prepare_model or override via --pd-ke / --pd-kd / --pd-effort."
        )
    else:
        logger.info("[validate-pd] all 3 layers look healthy.")
    return suspicious


def _scan_for_nan(state, joint_name_to_dof, body_paths, logger) -> bool:
    """Return True if any DOF or body transform has a NaN.  Logs which."""
    import numpy as np

    bad = False
    if state.joint_q is not None:
        jq = state.joint_q.numpy()
        nan_idx = np.where(~np.isfinite(jq))[0]
        if len(nan_idx) > 0:
            inv = {v: k for k, v in joint_name_to_dof.items()}
            names = [inv.get(int(i), f"DOF#{int(i)}") for i in nan_idx[:8]]
            logger.error(
                f"state_0.joint_q has non-finite values at DOF idx {nan_idx[:8].tolist()}: "
                f"{names}{'...' if len(nan_idx) > 8 else ''}"
            )
            bad = True
    if state.body_q is not None:
        bq = state.body_q.numpy()
        nan_idx = np.where(~np.isfinite(bq).all(axis=1))[0]
        if len(nan_idx) > 0:
            names = [body_paths[int(i)] if int(i) < len(body_paths) else f"body#{int(i)}" for i in nan_idx[:8]]
            logger.error(
                f"state_0.body_q has non-finite values at body idx {nan_idx[:8].tolist()}: "
                f"{names}{'...' if len(nan_idx) > 8 else ''}"
            )
            bad = True
    return bad


# ----------------------------------------------------------------------
# Init-pose parsing
# ----------------------------------------------------------------------


def _apply_init_pose(
    model,
    control,
    joint_name_to_dof: Dict[str, int],
    init_pose_str: str,
    logger,
) -> List[str]:
    """Parse ``init_pose_str`` and write to model.joint_q + control.joint_target_pos.

    Returns the list of joints actually written.
    """
    if not init_pose_str.strip():
        return []
    if model.joint_q is None:
        logger.warn("model.joint_q is None; cannot apply init pose")
        return []
    type_map = _joint_type_map(model)
    DEG2RAD = math.pi / 180.0
    jq = model.joint_q.numpy().copy()
    tgt = None
    if control is not None and getattr(control, "joint_target_pos", None) is not None:
        tgt = control.joint_target_pos.numpy().copy()
    applied: List[str] = []
    unknown: List[str] = []
    for entry in init_pose_str.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        name, val_str = entry.split("=", 1)
        name = name.strip()
        try:
            val = float(val_str)
        except ValueError:
            logger.warn(f"init-pose: cannot parse value for {name!r}: {val_str!r}")
            continue
        idx = joint_name_to_dof.get(name)
        if idx is None or idx >= len(jq):
            unknown.append(name)
            continue
        if type_map.get(name) == _JT_REVOLUTE:
            val = val * DEG2RAD
        jq[idx] = val
        if tgt is not None and idx < len(tgt):
            tgt[idx] = val
        applied.append(f"{name}={val:.4f}")
    model.joint_q.assign(jq)
    if tgt is not None:
        control.joint_target_pos.assign(tgt)
    if applied:
        logger.info(f"init pose applied to {len(applied)} DOF(s): {', '.join(applied)}")
    if unknown:
        logger.warn(f"init pose: unknown joint(s) ignored: {unknown}")
    return applied


# ----------------------------------------------------------------------
# wbc-style random goal driver (mirrors wbc_cmds.py over ROS, but writes
# directly to the adapter's target buffer — no ROS, no rclpy spin loop)
# ----------------------------------------------------------------------

# Name-substring -> group key.  Same order/semantics as wbc_cmds.py:
# "gripper" is handled specially in _classify_joint to keep the master
# in its full range and let mimic expansion drive followers.
_WBC_GROUP_PATTERNS: List[tuple] = [
    ("chassis", "chassis"),
    ("arm", "arm"),
    ("head", "head"),
    ("body", "body"),
]
_WBC_GUARD = 0.005  # rad/m kept inside each joint limit


def _classify_joint(name: str) -> str:
    """Return the group key for a joint name (matches wbc_cmds._classify)."""
    if "gripper" in name:
        return "gripper"
    for pat, group in _WBC_GROUP_PATTERNS:
        if pat in name:
            return group
    return "other"


class _WbcDriver:
    """Random-goal generator + per-frame tick that mirrors wbc_cmds.py.

    Differences from the ROS version:
      * No subscribe/publish — reads ``state_0.joint_q`` and writes to
        ``adapter.target_buffer()`` directly.
      * Time is SIM time (frames × frame_dt), not wall clock.  This keeps
        goal timeouts deterministic when the tool runs slower than real
        time (uncaptured stepping is ~10× the wall budget of production).
      * Mimic followers are filtered OUT of the goal set up-front (using
        the parsed ``mimic_followers`` map); their target buffer entries
        are populated each tick via ``expand_targets`` so a single master
        goal moves the whole gripper sub-chain.

    Construction parameters mirror the wbc_cmds.py CLI defaults so a
    side-by-side comparison reads the same.
    """

    def __init__(
        self,
        *,
        joint_name_to_dof: Dict[str, int],
        joint_limits: Dict[str, tuple],  # name -> (lower, upper)
        joint_types: Dict[str, int],
        mimic_followers: Dict[str, list],  # master -> [(follower, mult, off), ...]
        adapter,
        model,
        body_deg: float,
        head_deg: float,
        arm_deg: float,
        chassis_deg: float,
        with_chassis: bool,
        with_gripper: bool,
        tol: float,
        goal_timeout_s: float,
        logger,
    ) -> None:
        import random
        import math

        self._adapter = adapter
        self._tol = float(tol)
        self._goal_timeout = float(goal_timeout_s)
        self._joint_name_to_dof = joint_name_to_dof
        self._joint_types = joint_types
        self._mimic_followers = mimic_followers
        self._logger = logger
        self._rand = random.Random()  # seeded externally via seed()

        # Anything that is a mimic FOLLOWER must NOT receive an independent
        # goal — followers are derived from master targets via expand_targets.
        follower_set = {f for fl in mimic_followers.values() for (f, _, _) in fl}

        # On the FEATHERSTONE path, a joint with ``joint_target_ke > 0``
        # is a PASSIVE joint that the adapter's selective PD holds at
        # init pose (body/head on G2).  Sending random wbc goals to one
        # would activate PD as a controller; with ke=50000 that's way
        # too stiff for smooth tracking and the underdamped chain rings
        # to NaN within a few frames.  Filter them out so wbc only
        # commands the velocity-injection-driven joints.
        #
        # On the MJWARP path the same ``ke > 0`` test would exclude
        # EVERY joint — mjwarp's adapter sets ke on all DOFs because PD
        # IS the command mechanism (SolverMuJoCo's JOINT_TARGET
        # actuators).  The right policy on mjwarp is "command everything"
        # (subject to the chassis / gripper / mimic-follower filters
        # below); so we only apply the ke filter when adapter.name says
        # we're on Featherstone.
        passive_set: set = set()
        if adapter.name == "featherstone" and model.joint_target_ke is not None:
            ke = model.joint_target_ke.numpy()
            for name, idx in joint_name_to_dof.items():
                if idx < len(ke) and float(ke[idx]) > 0.0:
                    passive_set.add(name)

        # Optional chassis / gripper filters.
        excluded: List[str] = []
        if not with_chassis:
            excluded += [n for n in joint_limits if "chassis" in n]
        if not with_gripper:
            excluded += [n for n in joint_limits if "gripper" in n]

        # Filtered active joint set + per-group excursion (radians for
        # revolute, metres for prismatic — the wbc_cmds default values are
        # in degrees so we convert here).
        self._joints: Dict[str, tuple] = {
            n: lim
            for n, lim in joint_limits.items()
            if n not in follower_set and n not in excluded and n not in passive_set
        }
        self._range: Dict[str, float] = {
            "body": math.radians(body_deg),
            "head": math.radians(head_deg),
            "arm": math.radians(arm_deg),
            "chassis": math.radians(chassis_deg),
            "other": math.radians(body_deg),
        }

        # State updated each tick.
        self._init_q: Dict[str, float] = {}  # snapshot of joint_q at first goal
        self._goals: Dict[str, float] = {}
        self._goal_count: int = 0
        self._goal_t_start: float = 0.0

        # Pre-build group counts for the startup banner.
        counts: Dict[str, int] = {}
        for n in self._joints:
            counts[_classify_joint(n)] = counts.get(_classify_joint(n), 0) + 1
        excluded_summary = ""
        if follower_set:
            excluded_summary += f"; {len(follower_set)} mimic-follower(s) excluded"
        if passive_set:
            excluded_summary += f"; {len(passive_set)} PD-passive joint(s) excluded"
        if excluded:
            excluded_summary += f"; {len(excluded)} chassis-excluded"
        logger.info(
            f"wbc driver: {len(self._joints)} commandable joint(s) "
            f"(groups={counts}); excursions deg "
            f"body={body_deg} head={head_deg} arm={arm_deg} chassis={chassis_deg}"
            f"{excluded_summary}"
        )

    def seed(self, seed: int) -> None:
        self._rand.seed(seed)

    # ------------------------------------------------------------------
    # Goal generation
    # ------------------------------------------------------------------

    def _new_goals(self, current_q: Dict[str, float]) -> Dict[str, float]:
        """Random goal per joint, bounded by group excursion and joint limits.

        Logic mirrors wbc_cmds._new_goals: gripper masters get the full
        [lower, upper] range; chassis steering goals are relative to the
        neutral (0) pose; everything else is bounded relative to the
        anchored init pose.
        """
        goals: Dict[str, float] = {}
        for name, (lower, upper) in self._joints.items():
            group = _classify_joint(name)
            init_val = self._init_q.get(name, (lower + upper) * 0.5)
            lo_lim = lower + _WBC_GUARD
            hi_lim = upper - _WBC_GUARD
            if lo_lim >= hi_lim:
                lo_lim, hi_lim = lower, upper

            if group == "gripper":
                # full range, master only (followers are filtered out at construction)
                goals[name] = self._rand.uniform(lo_lim, hi_lim) if lo_lim < hi_lim else init_val
                continue
            excursion = self._range.get(group, self._range["other"])
            if group == "chassis":
                # absolute range from neutral 0
                lo = max(lo_lim, -excursion)
                hi = min(hi_lim, excursion)
                goals[name] = self._rand.uniform(lo, hi) if lo < hi else 0.0
            else:
                lo = max(lo_lim, init_val - excursion)
                hi = min(hi_lim, init_val + excursion)
                goals[name] = self._rand.uniform(lo, hi) if lo < hi else init_val
        return goals

    def _all_reached(self, current_q: Dict[str, float]) -> bool:
        if not self._goals:
            return False
        return all(n in current_q and abs(current_q[n] - g) < self._tol for n, g in self._goals.items())

    # ------------------------------------------------------------------
    # Per-frame tick — called from the main step loop
    # ------------------------------------------------------------------

    def tick(self, state_0, sim_time: float) -> None:
        """Advance the goal state if needed and (re)apply targets.

        ``state_0`` provides the current joint positions (read via
        ``state_0.joint_q.numpy()``).  Writes target values directly into
        ``self._adapter.target_buffer()`` via a numpy round-trip — same
        as ``control.py:apply_commands``.  Mimic followers are populated
        through ``expand_targets``.
        """
        # Read current state
        jq_np = state_0.joint_q.numpy()
        current_q = {n: float(jq_np[i]) for n, i in self._joint_name_to_dof.items() if i < len(jq_np)}

        advance = False
        if not self._goals:
            advance = True
        elif self._all_reached(current_q):
            advance = True
        elif sim_time - self._goal_t_start >= self._goal_timeout:
            stuck = [n for n, g in self._goals.items() if n in current_q and abs(current_q[n] - g) >= self._tol]
            self._logger.warn(
                f"wbc goal#{self._goal_count} timeout after {self._goal_timeout:.1f}s "
                f"({len(stuck)} stuck): {stuck[:6]}{'...' if len(stuck) > 6 else ''}"
            )
            advance = True

        if advance:
            if not self._init_q:
                # First time we have a current_q reading: anchor it
                self._init_q = dict(current_q)
            self._goals = self._new_goals(current_q)
            self._goal_count += 1
            self._goal_t_start = sim_time
            # Per-group goal summary
            by_group: Dict[str, int] = {}
            for n in self._goals:
                g = _classify_joint(n)
                by_group[g] = by_group.get(g, 0) + 1
            self._logger.info(
                f"wbc goal#{self._goal_count} at t={sim_time:.3f}s: " f"{len(self._goals)} target(s) {by_group}"
            )

        # ---- write to target buffer (mirrors control.py:apply_commands) ----
        tgt = self._adapter.target_buffer()
        if tgt is None:
            return
        try:
            from genie_sim_engine_core.engine._mimic import expand_targets
        except ImportError:
            expand_targets = None  # type: ignore[assignment]
        cmd = dict(self._goals)
        if expand_targets is not None:
            extra = expand_targets(cmd, self._mimic_followers)
            if extra:
                cmd = {**cmd, **extra}
        tgt_np = tgt.numpy().copy()
        for name, val in cmd.items():
            idx = self._joint_name_to_dof.get(name)
            if idx is None or idx >= len(tgt_np):
                continue
            tgt_np[idx] = float(val)
        tgt.assign(tgt_np)


def _extract_joint_limits(model, joint_name_to_dof: Dict[str, int]) -> Dict[str, tuple]:
    """Return {joint_name: (lower, upper)} for every joint in the map."""
    if model.joint_limit_lower is None or model.joint_limit_upper is None:
        return {}
    lo = model.joint_limit_lower.numpy()
    hi = model.joint_limit_upper.numpy()
    out: Dict[str, tuple] = {}
    for name, idx in joint_name_to_dof.items():
        if idx >= len(lo):
            continue
        lo_v, hi_v = float(lo[idx]), float(hi[idx])
        if lo_v < hi_v:  # skip stuck/degenerate joints (mimic followers etc.)
            out[name] = (lo_v, hi_v)
    return out


# ----------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    logger = _StdoutLogger()
    logger.info("=" * 70)
    logger.info(
        f"step 0  solver={args.solver}  substeps={args.substeps}  hz={args.physics_hz}  "
        f"validate_only={args.validate_only}"
    )
    logger.info("=" * 70)

    # === STEP 1: Adapter ===========================================
    if args.no_passive_pd and args.solver in ("fsvbd", "featherstone"):
        # Construct FeatherstoneAdapter directly with empty passive prefixes
        # so post_joint_map does nothing.  The factory's default would set
        # body/head PD which interferes with multi-joint wbc sweeps.
        from genie_sim_engine_newton.engine.newton.adapters.featherstone import FeatherstoneAdapter

        adapter = FeatherstoneAdapter(passive_joint_prefixes=())
        logger.info(
            f"step 1  adapter={adapter.name}  supports_cloth={adapter.supports_cloth} "
            f"(passive PD DISABLED via --no-passive-pd)"
        )
    else:
        # Optional ``--physics-params`` lets the CLI route gains through
        # the same ``PhysicsParams`` the production engine uses, so a
        # smoke run against ``robot.usda`` reproduces the runtime
        # behaviour the wrapper would produce.  When unset the adapter
        # falls back to its ``effort × 10`` heuristic — keeps standalone
        # smoke tests working when the YAML is not on disk.
        physics_params = None
        if getattr(args, "physics_params", "") and args.physics_params:
            try:
                from genie_sim_engine_core.common.params import load_physics_params

                physics_params = load_physics_params(args.physics_params, logger)
            except Exception as exc:  # noqa: BLE001
                logger.warn(
                    f"--physics-params {args.physics_params!r}: load failed ({exc!r}); "
                    f"falling back to effort×10 adapter heuristic."
                )
        adapter = make_adapter(
            args.solver,
            mujoco_pd_ke=args.pd_ke,
            mujoco_pd_kd=args.pd_kd,
            mujoco_save_to_mjcf=args.save_mjcf,
            physics_params=physics_params,
            physics_hz=args.physics_hz,
        )
        logger.info(f"step 1  adapter={adapter.name}  supports_cloth={adapter.supports_cloth}")

    # === STEP 2: ModelBuilder + custom attrs + load ================
    builder = newton.ModelBuilder()
    adapter.register_custom_attributes(builder)

    stage = None  # only set when loading from USD; mimic parser needs it
    if args.urdf:
        logger.info(f"step 2  loading URDF: {args.urdf}  (fix_base={args.fix_base})")
        builder.add_urdf(
            source=args.urdf,
            floating=not args.fix_base,
            collapse_fixed_joints=False,
        )
    else:
        try:
            from pxr import Usd
        except ImportError as exc:
            logger.error(f"cannot import pxr.Usd — install USD bindings or use --urdf instead: {exc}")
            return 2
        logger.info(f"step 2  loading USD: {args.usd}")
        stage = Usd.Stage.Open(args.usd)
        if stage is None:
            logger.error(f"failed to open USD stage: {args.usd}")
            return 2
        # Match production lifecycle.py — pass BOTH resolvers so
        # ``physxJoint:armature`` etc. authored by assemble_robot's USD
        # overlay is read.  Default add_usd() uses only
        # ``SchemaResolverNewton`` which looks for ``newton:armature``;
        # without the Physx resolver our overlay's ``physxJoint:*``
        # attributes are silently ignored and the standalone tool
        # diverges from production behavior.
        from newton.usd import SchemaResolverNewton, SchemaResolverPhysx

        builder.add_usd(
            source=stage,
            verbose=False,
            collapse_fixed_joints=False,
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )

    model = builder.finalize()
    logger.info(
        f"step 2  model finalized: {model.body_count} bodies, "
        f"{model.joint_dof_count} DOFs, {model.particle_count} particles"
    )

    # === STEP 3: Sanitize (zero-mass clamp + materials) ============
    # Skipping this on the G2 makes Featherstone NaN at frame 1 — the body
    # chain has massless intermediate links the USD path inherits from
    # PhysX's virtual-mass-injection convention.  Use --no-sanitize to
    # reproduce the raw Newton failure mode.
    if args.no_sanitize:
        logger.info("step 3  sanitize: SKIPPED (--no-sanitize)")
    else:
        _sanitize_model(model, logger)

    # === STEP 4: State/Control/contacts + gravity arrays ===========
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    # Snapshot the original gravity so we can restore it after each
    # robot step (mirrors lifecycle.py:222-228 — defaults to -9.81 m/s² Z-up
    # if the model didn't get a gravity vector from add_usd/add_urdf).
    if getattr(model, "gravity", None) is not None and len(model.gravity) > 0:
        gravity_earth = wp.clone(model.gravity)
    else:
        gravity_earth = wp.array([wp.vec3(0.0, 0.0, -9.81)], dtype=wp.vec3, device=model.device)
        if hasattr(model, "gravity") and model.gravity is not None:
            model.gravity.assign(gravity_earth)
    gravity_zero = wp.zeros(len(gravity_earth), dtype=wp.vec3, device=gravity_earth.device)
    shape_contact_pair_count_orig = int(model.shape_contact_pair_count)
    logger.info(
        f"step 4  states + control + contacts allocated; gravity earth="
        f"{gravity_earth.numpy()[0]}, shape_contact_pair_count={shape_contact_pair_count_orig}"
    )

    # === STEP 5: Adapter prepare_model + build_solver ==============
    # Parse mimic relationships from the USD stage / URDF BEFORE
    # prepare_model so the adapter can suppress actuators on follower
    # DOFs.  Mirrors lifecycle.py: ``_build_mimic_map`` runs before
    # ``prepare_model`` in production for the same reason — the
    # reference G2 MJCF emits only the master actuator, with followers
    # driven by mjwarp equality constraints.
    if stage is not None:
        mimic_followers = _build_mimic_map(stage, logger)
    elif args.urdf:
        mimic_followers = _build_mimic_map_from_urdf(args.urdf, logger)
    else:
        mimic_followers = {}
    adapter.prepare_model(model, logger, mimic_followers=mimic_followers)
    robot_solver = adapter.build_solver(model, args.substeps, 0, logger)

    # === STEP 6: Joint / body maps =================================
    joint_name_to_dof = _build_joint_map(model, logger)
    body_paths = _build_body_map(model)
    logger.info(
        f"step 6  joint_map={len(joint_name_to_dof)} controllable DOF(s); "
        f"body_map={len(body_paths)} body label(s); "
        f"mimic_map={len(mimic_followers)} master(s) -> "
        f"{sum(len(v) for v in mimic_followers.values())} follower(s)"
    )

    # === STEP 6b: PD-params validation =============================
    # Walks the three layers (Newton arrays → mjwarp actuator
    # gainprm/biasprm → per-joint actfrcrange) and flags any
    # combination that would make the robot jelly or runaway under
    # /joint_command.  Skipped unless --validate-pd is set; always run
    # when --validate-only is set so the dump happens before exit.
    if args.validate_pd or args.validate_only:
        n_susp = _validate_pd_params(model, adapter, joint_name_to_dof, logger)
        if args.validate_pd and args.exit_on_pd_warning and n_susp > 0:
            logger.error(
                f"step 6b ABORTING: {n_susp} suspicious PD entries "
                f"(--exit-on-pd-warning was set).  See table above."
            )
            return 4

    # === STEP 7: Init pose with joint-limit validation =============
    violations = _validate_init_pose_against_limits(model, joint_name_to_dof, args.init_pose, logger)
    if violations:
        for v in violations:
            logger.error(f"step 7  joint-limit violation: {v}")
        logger.error(
            f"step 7  ABORTING: {len(violations)} init-pose value(s) exceed URDF limits. "
            f"Featherstone will impulse to NaN at frame 1; mjwarp would clamp silently. "
            f"Either edit the scene yaml or relax the URDF <limit> tags."
        )
        return 3
    _apply_init_pose(model, control, joint_name_to_dof, args.init_pose, logger)

    # Mirror the engine's lifecycle: capture the post-init-pose qpos into
    # a ``<keyframe name="home">`` in the dumped MJCF so a pure-mujoco
    # ``mj_resetDataKeyframe`` reproduces the runtime starting state.
    if args.save_mjcf:
        try:
            from genie_sim_engine_newton.common.mjcf_postprocess import add_init_pose_keyframe

            add_init_pose_keyframe(
                mjcf_path=args.save_mjcf,
                qpos=model.joint_q.numpy(),
                name="home",
                logger=logger,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warn(f"[tools] init-pose keyframe step failed (continuing): {exc!r}")

    # === STEP 8: Adapter post_joint_map (selective PD) =============
    adapter.post_joint_map(model, joint_name_to_dof, control, logger)

    # === STEP 9: Sync BOTH states with model.joint_q ===============
    # Production lifecycle.py:344-352 seeds state_0 AND state_1 so the
    # very first substep can't read uninitialized state_1 contents.
    for st in (state_0, state_1):
        if st.joint_q is not None and model.joint_q is not None:
            st.joint_q.assign(model.joint_q)
        if st.joint_qd is not None:
            st.joint_qd.zero_()

    # === STEP 10: Adapter init_target_buffer ========================
    adapter.init_target_buffer(model, control, logger)

    # === STEP 11: FK init ===========================================
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # === STEP 12: Post-load summary + NaN scan ======================
    logger.info("-" * 70)
    logger.info("step 12 post-load summary")
    body_mass_np = model.body_mass.numpy() if getattr(model, "body_mass", None) is not None else None
    if body_mass_np is not None:
        logger.info(
            f"        bodies: {len(body_mass_np)}, total mass="
            f"{float(body_mass_np.sum()):.3f} kg, "
            f"min={float(body_mass_np.min()):.4g} kg, max={float(body_mass_np.max()):.4g} kg"
        )
    if model.joint_limit_lower is not None and model.joint_limit_upper is not None:
        lo = model.joint_limit_lower.numpy()
        hi = model.joint_limit_upper.numpy()
        n_at_limit = 0
        jq = model.joint_q.numpy()
        for name, idx in joint_name_to_dof.items():
            if idx < len(lo) and (abs(jq[idx] - lo[idx]) < 1e-4 or abs(jq[idx] - hi[idx]) < 1e-4):
                n_at_limit += 1
        if n_at_limit > 0:
            logger.warn(
                f"        {n_at_limit} controllable DOF(s) start AT a joint limit — "
                f"watch for impulsive constraint response"
            )
    if _scan_for_nan(state_0, joint_name_to_dof, body_paths, logger):
        logger.error("step 12 NaN already present after load — aborting")
        return 4
    logger.info("-" * 70)

    if args.validate_only:
        logger.info("step 12 --validate-only set; skipping warmup + step loop")
        return 0

    # === STEP 13: Warmup ============================================
    # Run one uncaptured frame so Warp's JIT compiles every kernel
    # before the timed loop starts — mirrors lifecycle._warmup.
    if not args.no_warmup:
        logger.info("step 13 warming up kernels (one uncaptured frame)…")
        sub_dt_warmup = (1.0 / args.physics_hz) / args.substeps
        for _ in range(args.substeps):
            state_0.clear_forces()
            model.gravity.assign(gravity_zero)
            model.shape_contact_pair_count = 0
            adapter.substep(model, state_0, state_1, control, sub_dt_warmup)
            model.gravity.assign(gravity_earth)
            model.shape_contact_pair_count = shape_contact_pair_count_orig
            state_0, state_1 = state_1, state_0
        if _scan_for_nan(state_0, joint_name_to_dof, body_paths, logger):
            logger.error("step 13 NaN after warmup — aborting (likely material/inertia issue)")
            return 5

    # === STEP 14: Optional GL viewer ================================
    viewer = None
    if args.viewer:
        try:
            from newton.viewer import ViewerGL
        except ImportError as exc:
            logger.warn(f"--viewer requested but ViewerGL unavailable: {exc}")
        else:
            try:
                viewer = ViewerGL(width=1280, height=720, vsync=False, headless=False)
                viewer.set_model(model)
                logger.info("step 14 GL viewer ready (window opened)")
            except Exception as exc:  # noqa: BLE001
                logger.warn(f"failed to open GL viewer: {exc}")
                viewer = None

    # === STEP 15: Step loop =========================================
    track = [n.strip() for n in args.log_joints.split(",") if n.strip()]
    sim_dt = (1.0 / args.physics_hz) / args.substeps
    frame_dt = 1.0 / args.physics_hz
    sim_time = 0.0
    t_start = time.monotonic()

    # ---- optional wbc-style driver ---------------------------------
    wbc = None
    if args.cmd_mode == "wbc":
        joint_limits = _extract_joint_limits(model, joint_name_to_dof)
        joint_types = _joint_type_map(model)
        wbc = _WbcDriver(
            joint_name_to_dof=joint_name_to_dof,
            joint_limits=joint_limits,
            joint_types=joint_types,
            mimic_followers=mimic_followers,
            adapter=adapter,
            model=model,
            body_deg=args.body_deg,
            head_deg=args.head_deg,
            arm_deg=args.arm_deg,
            chassis_deg=args.chassis_deg,
            with_chassis=args.with_chassis,
            with_gripper=args.with_gripper,
            tol=args.cmd_tol,
            goal_timeout_s=args.goal_timeout,
            logger=logger,
        )
        if args.seed is not None:
            wbc.seed(args.seed)

    logger.info(
        f"step 15 running {args.steps} frame(s) at {args.physics_hz} Hz "
        f"(sim_dt={sim_dt * 1000:.3f} ms, cmd_mode={args.cmd_mode}, capture={args.capture})"
    )

    # When ``--capture`` is on, build a CUDA graph that wraps the entire
    # inner substep block.  Inside the captured graph we CAN'T use the
    # Python ``state_0, state_1 = state_1, state_0`` swap — Warp's graph
    # records GPU ops only, so the Python reference reassignment isn't
    # recorded.  Use .assign() copies instead (matches engine.py:471-474
    # behavior in the captured branch).
    #
    # NOTE: state_0 / state_1 must be the SAME Python objects when the
    # graph replays as they were at capture time, because the captured
    # ops baked in pointer addresses to the wp.array members.  We commit
    # to one fixed pair throughout the run — no Python-side swap even
    # outside the captured loop.
    graph = None
    if args.capture:
        logger.info(f"step 15 capturing CUDA graph for {args.substeps}-substep block…")
        # One warm uncaptured cycle to ensure all kernels are JIT-compiled
        # before capture (capture fails if a kernel triggers compile mid-stream).
        for _ in range(args.substeps):
            state_0.clear_forces()
            state_1.clear_forces()
            model.gravity.assign(gravity_zero)
            model.shape_contact_pair_count = 0
            adapter.substep(model, state_0, state_1, control, sim_dt)
            model.gravity.assign(gravity_earth)
            model.shape_contact_pair_count = shape_contact_pair_count_orig
            # state_0 <- state_1 via assign (NOT Python swap), same pattern we'll
            # use inside the captured graph below.
            if state_0.joint_q is not None and state_1.joint_q is not None:
                state_0.joint_q.assign(state_1.joint_q)
            if state_0.joint_qd is not None and state_1.joint_qd is not None:
                state_0.joint_qd.assign(state_1.joint_qd)
            if state_0.body_q is not None and state_1.body_q is not None:
                state_0.body_q.assign(state_1.body_q)
            if state_0.body_qd is not None and state_1.body_qd is not None:
                state_0.body_qd.assign(state_1.body_qd)
        wp.synchronize_device()

        with wp.ScopedCapture() as cap:
            for _ in range(args.substeps):
                state_0.clear_forces()
                state_1.clear_forces()
                model.gravity.assign(gravity_zero)
                model.shape_contact_pair_count = 0
                adapter.substep(model, state_0, state_1, control, sim_dt)
                model.gravity.assign(gravity_earth)
                model.shape_contact_pair_count = shape_contact_pair_count_orig
                if state_0.joint_q is not None and state_1.joint_q is not None:
                    state_0.joint_q.assign(state_1.joint_q)
                if state_0.joint_qd is not None and state_1.joint_qd is not None:
                    state_0.joint_qd.assign(state_1.joint_qd)
                if state_0.body_q is not None and state_1.body_q is not None:
                    state_0.body_q.assign(state_1.body_q)
                if state_0.body_qd is not None and state_1.body_qd is not None:
                    state_0.body_qd.assign(state_1.body_qd)
        graph = cap.graph
        logger.info(f"step 15 CUDA graph captured.")

    for frame in range(args.steps):
        # wbc driver tick BEFORE the substep block so the freshly
        # written targets are visible to the very next velocity-injection
        # / JOINT_TARGET-PD pass.  Mirrors how /joint_command is processed
        # before the captured graph fires in production engine.py.
        if wbc is not None:
            wbc.tick(state_0, sim_time)

        if graph is not None:
            wp.capture_launch(graph)
        else:
            for _ in range(args.substeps):
                state_0.clear_forces()
                # Robot step: zero gravity AND shape contacts so the
                # kinematic control law (JOINT_TARGET PD on mjwarp /
                # velocity injection on featherstone) tracks the target
                # buffer without fighting a full-gravity load OR
                # self-collision impulses on every substep — same trick
                # the production engine uses in _simulate_substeps.
                model.gravity.assign(gravity_zero)
                model.shape_contact_pair_count = 0
                adapter.substep(model, state_0, state_1, control, sim_dt)
                model.gravity.assign(gravity_earth)
                model.shape_contact_pair_count = shape_contact_pair_count_orig
                state_0, state_1 = state_1, state_0
        sim_time += frame_dt

        if viewer is not None:
            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.end_frame()
            if not viewer.is_running():
                logger.info(f"viewer closed at frame {frame}")
                break

        if track and (frame % 10 == 0 or frame == args.steps - 1):
            jq = state_0.joint_q.numpy()
            parts = []
            for name in track:
                idx = joint_name_to_dof.get(name)
                if idx is None or idx >= len(jq):
                    parts.append(f"{name}=??")
                else:
                    parts.append(f"{name}={float(jq[idx]):+.5f}")
            logger.info(f"frame={frame:4d} t={sim_time:.3f}s  {'  '.join(parts)}")

    wall = time.monotonic() - t_start
    logger.info(
        f"done: {args.steps} frame(s) in {wall:.2f}s wall "
        f"({args.steps / wall:.1f} fps, target {args.physics_hz} fps)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
