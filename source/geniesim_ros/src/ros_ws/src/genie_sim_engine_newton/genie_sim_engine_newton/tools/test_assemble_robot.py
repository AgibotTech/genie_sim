#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Standalone test harness for the ``assemble_robot.py`` pipeline.

Why this exists
---------------
``assemble_robot.py`` is a 9-stage build that turns a scene-yaml +
xacro into the ``robot.usda`` payload set every runtime engine
consumes.  When something goes wrong (a missing joint after AS3, a
material override not applied, a mimic-overlay attribute landing on
the wrong prim, an extra ``loop_joint`` silently dropped) you want to
see the intermediate USD AT THAT POINT IN THE PIPELINE — not the
final composed stage.

This tool runs each stage in isolation (or any contiguous range),
snapshots its output to a numbered file in the working directory,
and runs a per-stage validator that flags structural regressions
without forcing you to spin up Isaac / Newton / a full launcher.

Stages
------
1. ``xacro``          — xacro → ``robot.urdf`` + ``robot_raw.urdf``
2. ``urdf2usd``       — Isaac ``urdf_usd_converter.Converter.convert``
3. ``schemas``        — rigid-body / joint schemas + MJC/PhysX joint attrs
4. ``as3``            — Asset Structure 3.0 transformer (split into payloads)
5. ``material_overrides`` — apply parsed ``<material_override>`` PBR patches
8. ``collision_policy`` — selective collision enable/disable
9. ``mimic_overlay``  — author armature + drive stiffness on hand/arm joints
10. ``gripper``       — read-only structural check of master / follower
                       drives + ``newton:mimicJoint`` edges per gripper
                       side.  Catches the "arms track but gripper swings"
                       failure mode where the overlay missed a follower
                       or a mimic edge dangles.

Stages 1-4 are deeply Isaac-coupled and run in one block (the Isaac
URDF importer is monolithic).  Stages 5-9 are pure pxr stage edits and
can be re-run / inspected independently — these are where iteration
happens.

Usage
-----
Run the full pipeline + validate every stage::

    python3 scripts/tools/test_assemble_robot.py \\
        --scene scene_flat_g2_sp \\
        --workdir /tmp/test_assemble_g2 --validate

Re-run only the post-AS3 stages against an existing ``robot.usda``::

    python3 scripts/tools/test_assemble_robot.py \\
        --workdir /tmp/test_assemble_g2 \\
        --from-stage 5 --to-stage 9 --validate

Inspect a single stage::

    python3 scripts/tools/test_assemble_robot.py \\
        --workdir /tmp/test_assemble_g2 \\
        --only-stage mimic_overlay --inspect

Diagnose a swinging / chattering gripper against an existing build::

    python3 scripts/tools/test_assemble_robot.py \\
        --workdir /tmp/test_assemble_g2 \\
        --only-stage gripper --validate

Validators only (no mutation)::

    python3 scripts/tools/test_assemble_robot.py \\
        --workdir /tmp/test_assemble_g2 --validate-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Optional

# This tool is a HARNESS, not a re-implementation: every stage below calls
# the real pipeline's own helpers.  Those used to live in a top-level
# ``assemble_robot.py`` reachable by putting the parent directory on
# ``sys.path``; the orchestrator has since moved into
# ``genie_sim_engine_isaacsim.assemble._robot_impl`` (which still prints
# ``[assemble_robot]`` in its own diagnostics), so the imports are ordinary
# package imports now and the path insert is gone.
#
# They stay INSIDE each stage function on purpose -- ``_robot_impl`` is
# deeply Isaac-coupled, and stages 5-9 are pure ``pxr`` stage edits that
# must remain runnable without paying for the Isaac import.


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------


class _Logger:
    """Print-style logger compatible with assemble_robot's helpers."""

    def info(self, msg: str) -> None:
        print(f"[test-assemble] {msg}")

    def warn(self, msg: str) -> None:
        print(f"[test-assemble] WARN: {msg}")

    def error(self, msg: str) -> None:
        print(f"[test-assemble] ERROR: {msg}", file=sys.stderr)


# ----------------------------------------------------------------------
# Stage definition
# ----------------------------------------------------------------------


@dataclass
class StageResult:
    """Per-stage output bundle.

    Each stage produces a primary artifact (``output_path``) and a
    set of structured metrics the validator checks.  Snapshot files
    are numbered ``NN_<name>.<ext>`` so an ``ls`` of the workdir gives
    you the pipeline's progression at a glance.
    """

    stage_name: str
    output_path: str
    metrics: dict
    extra_paths: list  # additional artifacts (e.g. material_overrides JSON)


@dataclass
class Stage:
    """One pipeline stage: name, runner, validator."""

    number: int
    name: str
    description: str
    run: Callable[..., StageResult]
    validate: Callable[[StageResult, _Logger], int]  # returns suspicious count


# ----------------------------------------------------------------------
# Stage 1: xacro → URDF
# ----------------------------------------------------------------------


def _stage_xacro(args, workdir: str, logger: _Logger) -> StageResult:
    """Stage the input URDF on disk for downstream URDF->USD conversion.

    Production ``assemble_robot.py`` reads its URDF from the
    ``/robot_description`` topic. To keep this debug harness usable
    offline (no ROS graph, no RSP) it supports two input modes:

      * ``--urdf <path>``  — read the URDF directly from disk; no xacro,
        no scene-yaml lookup, no ament resolution. Mirrors the production
        ``--urdf`` debug switch.
      * ``--scene <name>`` — resolve the scene yaml, locate the xacro
        entry, run xacro inline (replicates what the launch composer's
        ``build_robot_description`` does in normal runs).
    """
    from genie_sim_engine_isaacsim.assemble._robot_impl import _stage_urdf_from_string  # noqa: PLC0415

    urdf_arg = getattr(args, "urdf", None)
    if urdf_arg:
        urdf_file = os.path.abspath(urdf_arg)
        if not os.path.isfile(urdf_file):
            logger.error(f"[xacro] --urdf file not found: {urdf_file}")
            raise FileNotFoundError(urdf_file)
        logger.info(f"[xacro] --urdf mode: reading {urdf_file} directly (skipping xacro)")
        with open(urdf_file) as fh:
            urdf_str = fh.read()
        urdf_out, raw_out, ros_pkgs = _stage_urdf_from_string(urdf_str, dest_dir=workdir)
    else:
        from genie_sim_engine_isaacsim.assemble._robot_impl import _load_scene_yaml  # noqa: PLC0415

        try:
            import xacro  # type: ignore  # noqa: PLC0415
        except ImportError:
            logger.warn("[xacro] 'xacro' Python module not importable. Source your ROS 2 workspace.")
            raise

        try:
            from ament_index_python.packages import (  # noqa: PLC0415
                PackageNotFoundError,
                get_package_share_directory,
            )
        except ImportError:
            logger.warn("[xacro] ament_index_python not importable. Source your ROS 2 workspace.")
            raise

        cfg_path, config = _load_scene_yaml(args.scene)
        robot_section = config.get("robot") or {}
        source = robot_section.get("robot_source") or {}
        urdf_value = source.get("urdf") if isinstance(source.get("urdf"), dict) else {}

        def _pick(key: str) -> str:
            for container in (source, urdf_value):
                v = container.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""

        mappings = {"robot_model": _pick("robot_model")}
        for k in ("arm", "body", "gripper"):
            v = _pick(k)
            if v:
                mappings[k] = v

        package = source.get("package") or "genie_sim_robot_model"
        xacro_entry = (urdf_value.get("xacro_entry") if isinstance(urdf_value, dict) else None) or os.path.join(
            "xacro", "robot.xacro"
        )

        try:
            share = get_package_share_directory(package)
        except PackageNotFoundError:
            logger.warn(f"[xacro] ROS 2 package not found: {package!r}")
            raise
        xacro_path = os.path.join(share, xacro_entry)
        if not os.path.isfile(xacro_path):
            logger.warn(f"[xacro] xacro entry not found: {xacro_path}")
            raise FileNotFoundError(xacro_path)

        doc = xacro.process_file(xacro_path, mappings=mappings)
        urdf_str = doc.toprettyxml(indent="  ")
        urdf_out, raw_out, ros_pkgs = _stage_urdf_from_string(urdf_str, dest_dir=workdir)

    # Snapshot under our numbered naming so the workdir reads as a
    # pipeline transcript.  Originals stay where the helper put them
    # so subsequent stages that take ``urdf_out`` keep working.
    shutil.copy(urdf_out, os.path.join(workdir, "01_xacro.urdf"))
    shutil.copy(raw_out, os.path.join(workdir, "01_xacro.raw.urdf"))

    # Quick metrics from the URDF text — no XML parse needed for the
    # validator's smoke check.  More serious validators (e.g. tree
    # reachability) can parse on demand.
    with open(urdf_out) as f:
        urdf_text = f.read()
    metrics = {
        "joint_total": urdf_text.count("<joint name="),
        "loop_joint_total": urdf_text.count("<loop_joint "),
        "mimic_total": urdf_text.count("<mimic "),
        "dynamics_total": urdf_text.count("<dynamics "),
        "link_total": urdf_text.count("<link name="),
        "ros_packages": list(ros_pkgs.keys()),
    }
    return StageResult(
        stage_name="xacro",
        output_path=urdf_out,
        metrics=metrics,
        extra_paths=[raw_out],
    )


def _validate_xacro(result: StageResult, logger: _Logger) -> int:
    """Smoke-check the URDF: at least one joint, every joint has
    matching closing tag, etc.  Suspicious count.

    Tolerant of generic ``_scan_usd_metrics`` input (used by
    ``--validate-only`` against an existing workdir): if the metrics
    dict doesn't carry our URDF-text counters, we silently skip rather
    than crash, since the only artifact we'd need (``01_xacro.urdf``)
    is regenerated on every fresh run anyway.
    """
    suspicious = 0
    m = result.metrics
    # ``link_total`` is the marker key that the URDF-text scanner
    # produced this dict (vs. ``_scan_usd_metrics`` which has the same
    # ``joint_total`` field but no ``link_total``).  If the marker is
    # absent we're in ``--validate-only`` mode against an existing
    # workdir — skip rather than reinterpret USD metrics as URDF text.
    if "link_total" not in m:
        logger.info("[xacro] (no run-time URDF metrics; skipped — re-run stage 1 to validate URDF text)")
        return 0
    if m["joint_total"] == 0:
        logger.warn("[xacro] URDF has 0 joints — xacro probably failed silently")
        suspicious += 1
    if m.get("link_total", 0) == 0:
        logger.warn("[xacro] URDF has 0 links")
        suspicious += 1
    logger.info(
        f"[xacro] OK: {m['joint_total']} joints, {m.get('link_total', '?')} links, "
        f"{m.get('mimic_total', '?')} mimics, {m.get('loop_joint_total', '?')} loop_joints, "
        f"{m.get('dynamics_total', '?')} dynamics tags"
    )
    return suspicious


# ----------------------------------------------------------------------
# Stages 2-4: Isaac URDF→USD + schemas + AS3
#
# These run inside Isaac Sim as a single monolithic call
# (``_convert_urdf_to_usd``).  We don't try to split them — the Isaac
# pipeline is opaque past Step 1.  We DO snapshot the post-AS3 output
# so subsequent stages can run against it.
# ----------------------------------------------------------------------


def _ros_packages_for(urdf_path: str, logger: _Logger) -> dict[str, str]:
    """Map each ``package://<pkg>/`` in ``urdf_path`` to that package's share dir.

    The 6.x ``urdf_usd_converter`` resolves mesh URIs through this map and
    nothing else.  Handed an empty map it reports success while importing no
    geometry at all: the AS3 transformer then emits no ``geometries.usd`` /
    ``instances.usda`` payload, and the collider and ``<material_override>``
    policies truthfully report zero edits against an empty stage.  That is
    silent, so build the map here rather than letting it default to empty.
    """
    import re  # noqa: PLC0415
    from ament_index_python.packages import (  # noqa: PLC0415
        PackageNotFoundError,
        get_package_share_directory,
    )

    with open(urdf_path) as fh:
        names = sorted(set(re.findall(r"package://([A-Za-z_][A-Za-z0-9_]*)/", fh.read())))
    resolved: dict[str, str] = {}
    for name in names:
        try:
            resolved[name] = get_package_share_directory(name)
        except PackageNotFoundError:
            logger.warn(f"[isaac_pipeline] ROS package {name!r} not found; its meshes cannot resolve")
    logger.info(f"[isaac_pipeline] ros_packages resolved: {sorted(resolved)} (from {len(names)} referenced)")
    return resolved


def _stage_isaac_pipeline(args, workdir: str, logger: _Logger) -> StageResult:
    """Run Isaac's URDF→USD + schemas + AS3 transformer end-to-end.
    Outputs ``robot.usda`` + payloads/ in workdir."""
    from genie_sim_engine_isaacsim.assemble._robot_impl import _convert_urdf_to_usd  # noqa: PLC0415

    urdf_path = os.path.join(workdir, "robot.urdf")
    raw_urdf_path = os.path.join(workdir, "robot_raw.urdf")
    usd_path = os.path.join(workdir, "robot.usda")
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(
            f"[isaac_pipeline] URDF not found at {urdf_path}.  Run stage xacro "
            f"first or supply --workdir with an existing URDF."
        )
    # The 6.x converter reads ``raw_urdf_path`` (package:// URIs intact) and
    # resolves every mesh through ros_packages alone.  Production supplies it
    # from ``_stage_urdf_from_string``; this harness has to rebuild it, and
    # omitting it drops every mesh — see ``_ros_packages_for``.
    ros_packages = _ros_packages_for(
        raw_urdf_path if os.path.isfile(raw_urdf_path) else urdf_path, logger
    )
    _convert_urdf_to_usd(
        urdf_path,
        usd_path,
        ros_packages=ros_packages,
        raw_urdf_path=raw_urdf_path,
    )

    # Note: AS3 emits ``robot.usda`` + ``payloads/*`` + the post-
    # transformer collision policy + mimic overlay.  Because the Isaac
    # call is monolithic, this stage's output IS the final-ish USD —
    # subsequent stages 5-9 are no-ops by default UNLESS the pipeline
    # is split via ``--from-stage`` (developer mode).  In dev mode,
    # those stages re-apply themselves to a baseline ``robot.usda``
    # we'd produce by running Isaac's converter without the post-
    # process hooks.
    snapshot = os.path.join(workdir, "04_isaac_as3.robot.usda")
    if os.path.isfile(usd_path):
        shutil.copy(usd_path, snapshot)

    metrics = _scan_usd_metrics(usd_path)
    return StageResult(
        stage_name="isaac_pipeline",
        output_path=usd_path,
        metrics=metrics,
        extra_paths=[],
    )


def _validate_isaac_pipeline(result: StageResult, logger: _Logger) -> int:
    suspicious = 0
    m = result.metrics
    if m["joint_total"] == 0:
        logger.warn("[isaac] robot.usda has 0 joints — Isaac importer probably failed")
        suspicious += 1
    if m["rigid_body_count"] == 0:
        logger.warn("[isaac] robot.usda has 0 RigidBodyAPI prims")
        suspicious += 1
    if m["payload_dir_exists"] is False:
        logger.warn("[isaac] payloads/ directory missing — AS3 transformer may not have run")
        suspicious += 1
    logger.info(
        f"[isaac_pipeline] OK: {m['joint_total']} joints "
        f"(revolute={m['revolute_count']}, fixed={m['fixed_count']}, prismatic={m['prismatic_count']}), "
        f"{m['rigid_body_count']} rigid bodies, payloads={m['payload_dir_exists']}"
    )
    return suspicious


# ----------------------------------------------------------------------
# Stage 5: material overrides apply
# ----------------------------------------------------------------------


def _stage_material_overrides(args, workdir: str, logger: _Logger) -> StageResult:
    """Apply parsed ``<material_override>`` PBR patches on the AS3 USD.

    Parses the source URDF for inline material-override blocks (those
    are stripped before Isaac sees the URDF) and patches roughness /
    metallic on the named geometry's PBR shaders. Routes through
    ``_apply_post_transformer_material_overrides`` which opens the
    final ``robot.usda``, runs ``_apply_material_overrides`` on the
    open stage, and saves the root layer — same path the production
    converter uses post-transformer (see the docstring on the
    production helper for the silent-failure rationale that drove
    that placement).
    """
    from genie_sim_engine_isaacsim.assemble._robot_impl import (  # noqa: PLC0415
        _apply_post_transformer_material_overrides,
        _parse_material_override_blocks,
    )

    usd_path = os.path.join(workdir, "robot.usda")
    raw_urdf_path = os.path.join(workdir, "robot_raw.urdf")
    snapshot = os.path.join(workdir, "05_material_overrides.usda")

    overrides = []
    if os.path.isfile(raw_urdf_path):
        with open(raw_urdf_path) as f:
            overrides = _parse_material_override_blocks(f.read())

    n_applied = 0
    if overrides and os.path.isfile(usd_path):
        n_applied = _apply_post_transformer_material_overrides(
            usd_path,
            overrides=overrides,
            logger=logger,
        )

    if os.path.isfile(usd_path):
        shutil.copy(usd_path, snapshot)

    metrics = {
        "overrides_parsed": len(overrides),
        "overrides_applied": n_applied,
    }
    return StageResult(
        stage_name="material_overrides",
        output_path=usd_path,
        metrics=metrics,
        extra_paths=[],
    )


def _validate_material_overrides(result: StageResult, logger: _Logger) -> int:
    m = result.metrics
    if "overrides_parsed" not in m:
        logger.info("[material_overrides] (no run-time metrics; skipped)")
        return 0
    if m["overrides_parsed"] > 0 and m["overrides_applied"] != m["overrides_parsed"]:
        logger.warn(
            f"[material_overrides] parsed {m['overrides_parsed']} but only "
            f"applied {m['overrides_applied']} — check geometry-target paths"
        )
        return 1
    logger.info(f"[material_overrides] OK: parsed={m['overrides_parsed']} applied={m['overrides_applied']}")
    return 0


# ----------------------------------------------------------------------
# Stage 8: collision policy (post-AS3)
# ----------------------------------------------------------------------


def _stage_collision_policy(args, workdir: str, logger: _Logger) -> StageResult:
    """Apply the selective collision policy on the final USD.

    Keeps only the prims we WANT colliding (gripper SDF + chassis
    wheel cylinders) and disables ``physics:collisionEnabled`` on
    everything else.  See ``_apply_post_transformer_collision_policy``
    docstring for the full rationale.
    """
    from genie_sim_engine_isaacsim.assemble._robot_impl import _apply_post_transformer_collision_policy  # noqa: PLC0415

    usd_path = os.path.join(workdir, "robot.usda")
    snapshot = os.path.join(workdir, "08_collision_policy.usda")
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"[collision_policy] {usd_path} not found")
    counts = _apply_post_transformer_collision_policy(usd_path, logger=logger)
    shutil.copy(usd_path, snapshot)
    return StageResult(
        stage_name="collision_policy",
        output_path=usd_path,
        metrics=dict(counts),
        extra_paths=[],
    )


def _validate_collision_policy(result: StageResult, logger: _Logger) -> int:
    if not any(k in result.metrics for k in ("kept", "stripped", "no_collision", "skipped_no_visual")):
        logger.info("[collision_policy] (no run-time metrics; skipped)")
        return 0
    logger.info(f"[collision_policy] OK: {result.metrics}")
    return 0


# ----------------------------------------------------------------------
# Stage 9: mimic / joint overlay
# ----------------------------------------------------------------------


def _stage_mimic_overlay(args, workdir: str, logger: _Logger) -> StageResult:
    """Author per-class joint USD attributes (armature + master drive
    on hand joints, armature on arm shoulder/mid/wrist) into the final
    ``robot.usda``'s root layer.

    See ``_apply_mimic_joint_overlay`` for the full per-class table.
    """
    from genie_sim_engine_isaacsim.assemble._robot_impl import _apply_mimic_joint_overlay  # noqa: PLC0415

    usd_path = os.path.join(workdir, "robot.usda")
    snapshot = os.path.join(workdir, "09_mimic_overlay.usda")
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"[mimic_overlay] {usd_path} not found")
    counts = _apply_mimic_joint_overlay(usd_path, logger=logger)
    shutil.copy(usd_path, snapshot)
    return StageResult(
        stage_name="mimic_overlay",
        output_path=usd_path,
        metrics=dict(counts),
        extra_paths=[],
    )


def _validate_mimic_overlay(result: StageResult, logger: _Logger) -> int:
    """Confirm armature + drive made it onto the right joints.  Two
    paths: when invoked right after the overlay run we know the
    expected per-class counts (``masters``, ``arm_shoulder`` etc.);
    when invoked via ``--validate-only`` we only have a generic
    ``_scan_usd_metrics`` dict and fall back to a structural check
    (any armature authored at all + drives only on inner_joint1).
    """
    suspicious = 0
    try:
        from pxr import Usd, UsdPhysics  # noqa: PLC0415
    except ImportError:
        logger.warn("[mimic_overlay] pxr not available — skipping deep validation")
        return 0

    stage = Usd.Stage.Open(result.output_path)
    if stage is None:
        logger.warn(f"[mimic_overlay] couldn't open {result.output_path}")
        return 1

    n_armature = 0
    n_drive_master = 0
    bad_drive_targets: list = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = prim.GetName()
        arm = prim.GetAttribute("physxJoint:armature")
        if arm and arm.IsValid() and arm.HasAuthoredValue():
            n_armature += 1
        drive = prim.GetAttribute("drive:angular:physics:stiffness")
        if drive and drive.IsValid() and drive.HasAuthoredValue():
            if "gripper" in name and name.endswith("inner_joint1"):
                n_drive_master += 1
            else:
                bad_drive_targets.append(name)

    # Per-class counts only available right after the overlay run.
    overlay_counts = {
        k: result.metrics.get(k, 0) for k in ("masters", "followers", "arm_shoulder", "arm_mid", "arm_wrist")
    }
    expected_armature_min = sum(overlay_counts.values())
    expected_masters = overlay_counts["masters"]
    have_overlay_metrics = expected_armature_min > 0

    if have_overlay_metrics:
        if n_armature < expected_armature_min:
            logger.warn(
                f"[mimic_overlay] expected armature on >= {expected_armature_min} "
                f"joints (per overlay counts {overlay_counts}), found {n_armature} authored"
            )
            suspicious += 1
        if n_drive_master != expected_masters:
            logger.warn(f"[mimic_overlay] expected {expected_masters} master drive(s), " f"found {n_drive_master}")
            suspicious += 1
    else:
        # Structural check: at minimum there should be SOME armature
        # authored on hand + arm joints (12 + 14 = 26 for G2) and
        # exactly TWO master drives (one per gripper).
        if n_armature == 0:
            logger.warn(
                "[mimic_overlay] no physxJoint:armature authored anywhere — "
                "overlay may not have run.  Re-bake with stage 9."
            )
            suspicious += 1

    if bad_drive_targets:
        logger.warn(
            f"[mimic_overlay] drive:angular:physics:stiffness authored on "
            f"non-master joint(s): {bad_drive_targets[:6]}"
            f"{'…' if len(bad_drive_targets) > 6 else ''}"
        )
        suspicious += 1

    logger.info(
        f"[mimic_overlay] OK: armature_authored={n_armature}, "
        f"master_drive_authored={n_drive_master}, overlay_counts={overlay_counts or 'n/a'}"
    )
    return suspicious


# ----------------------------------------------------------------------
# Stage 10: gripper inspection (read-only structural check)
#
# Failure mode this catches: arms / body track joint commands fine, but
# the gripper "swings" (droops under gravity, chatters, or doesn't hold
# its commanded position).  That's almost always one of three things on
# the assemble side, all of which a quick eyeball of robot.usda won't
# surface:
#
#   1. The master joint (``*gripper*inner_joint1``) didn't get a non-zero
#      ``drive:angular:physics:stiffness`` authored — the kp=5 (per-deg)
#      master drive ``_apply_mimic_joint_overlay`` is supposed to lay down.
#      Without it the runtime tensor handle has nothing to seed, and on
#      newton-standalone Newton's ``JointTargetMode.from_gains(0, 0,…)``
#      drops the master into EFFORT mode → no position actuator → the
#      finger weight pulls the gripper open at every tick.
#   2. A follower (``*gripper*…`` but NOT ``inner_joint1``) is carrying
#      non-zero stiffness or damping — that fights the mimic equality
#      constraint and the gripper chatters / oscillates.  The overlay is
#      supposed to clear these to 0.
#   3. The ``newton:mimicJoint`` relationship on a follower is missing,
#      pointing at the wrong master, or its ``mimicCoef1`` is not ±1.
#      Followers without a working mimic edge swing free under whatever
#      ``apply_action`` was last told to write for them (usually 0 →
#      finger weight → swing).
#
# We do NOT mutate anything here.  Run as ``--only-stage gripper`` or
# bundle into ``--validate`` to get a per-gripper-side report.
# ----------------------------------------------------------------------


_MASTER_SUFFIX = "inner_joint1"


def _gripper_side_prefix(joint_name: str) -> str:
    """Derive the per-gripper grouping key from a joint name.

    The pipeline's URDF→USD path tags every joint with a unique
    ``idxNN_`` prefix (so isaacsim's converter can keep duplicate
    names distinct on the same articulation), which means the leading
    substring is NOT a side discriminator.  The side identifier is
    ``gripper_<one_token>``: the FIRST token after ``gripper_``.

      ``idx31_gripper_l_inner_joint1``       → ``gripper_l``  (master)
      ``idx41_gripper_l_outer_joint1``       → ``gripper_l``  (follower)
      ``idx33_gripper_l_left_support_joint`` → ``gripper_l``  (follower w/ mult=-1)
      ``gripper_right_finger_joint``         → ``gripper_right``

    If the token immediately after ``gripper_`` is itself a joint role
    keyword (``inner``, ``outer``, ``finger``, ``knuckle``, ``jaw``,
    ``support``, ``joint``), the robot has no side discriminator and
    everything buckets as ``gripper``.
    """
    import re

    m = re.search(r"gripper(?:_([a-z][a-z0-9]*))?", joint_name)
    if not m:
        return joint_name
    side = m.group(1)
    if side is None or side in {"inner", "outer", "finger", "knuckle", "jaw", "support", "joint", "left", "right"}:
        # ``left`` / ``right`` are excluded so a robot whose naming
        # convention puts the side IMMEDIATELY before the role
        # (``gripper_left_finger_joint``) still buckets correctly: the
        # outer regex already captured ``left`` as the side, so we let
        # it pass.  Drop into "no-side" only when the side token
        # really is a role keyword.
        if side in {"left", "right"}:
            return f"gripper_{side}"
        return "gripper"
    return f"gripper_{side}"


def _read_drive(prim, axis: str = "angular") -> dict:
    """Read DriveAPI stiffness / damping / max_force off a joint prim.

    Returns ``{"stiffness": float|None, "damping": float|None,
    "max_force": float|None, "applied": bool}`` — ``None`` means the
    attribute is missing or unauthored (NOT zero — zero is a deliberate
    follower override and we want to distinguish the two cases).
    """
    from pxr import UsdPhysics  # noqa: PLC0415

    out = {"stiffness": None, "damping": None, "max_force": None, "applied": False}
    if prim.HasAPI(UsdPhysics.DriveAPI, axis):
        out["applied"] = True
        for key, attr_name in (
            ("stiffness", f"drive:{axis}:physics:stiffness"),
            ("damping", f"drive:{axis}:physics:damping"),
            ("max_force", f"drive:{axis}:physics:maxForce"),
        ):
            a = prim.GetAttribute(attr_name)
            if a and a.IsValid() and a.HasAuthoredValue():
                try:
                    out[key] = float(a.Get())
                except (TypeError, ValueError):
                    pass
    return out


def _read_mimic(prim) -> dict:
    """Read ``newton:mimicJoint`` rel + coef1/coef0 attrs off a joint prim.

    Returns ``{"target": str|None, "coef1": float, "coef0": float,
    "newton_api": bool, "physx_api": bool}``. ``target`` is the LAST
    PATH SEGMENT of the rel target (i.e. the master joint's name), so
    callers can string-compare against ``prim.GetName()`` without
    re-resolving paths.

    Reads the full applied-API list via ``GetPrimTypeInfo()`` rather
    than ``GetAppliedSchemas()`` so unregistered tokens (``NewtonMimicAPI``
    on a vanilla pxr build that doesn't have Newton's schema bundle
    loaded) still surface — the file does carry them, even if the
    local schema registry can't resolve them.  The PhysX importer
    also authors ``PhysxMimicJointAPI:rotZ`` rather than
    ``PhysxMimicJointAPI:angular`` for revolute joints (the schema is
    instanced per Cartesian axis when the joint axis isn't ``angular``);
    we accept any ``PhysxMimicJointAPI:*`` instance.
    """
    out = {"target": None, "coef1": 1.0, "coef0": 0.0, "newton_api": False, "physx_api": False}
    applied = list(prim.GetPrimTypeInfo().GetAppliedAPISchemas())
    out["newton_api"] = "NewtonMimicAPI" in applied
    out["physx_api"] = any(s.startswith("PhysxMimicJointAPI:") for s in applied)
    rel = prim.GetRelationship("newton:mimicJoint")
    if rel and rel.HasAuthoredTargets():
        targets = rel.GetTargets()
        if targets:
            out["target"] = targets[0].name
    a1 = prim.GetAttribute("newton:mimicCoef1")
    if a1 and a1.HasAuthoredValue():
        try:
            out["coef1"] = float(a1.Get())
        except (TypeError, ValueError):
            pass
    a0 = prim.GetAttribute("newton:mimicCoef0")
    if a0 and a0.HasAuthoredValue():
        try:
            out["coef0"] = float(a0.Get())
        except (TypeError, ValueError):
            pass
    return out


def _read_joint_geometry(prim) -> dict:
    """Read joint axis / limits / armature off a UsdPhysics.Joint prim."""
    out: dict = {"axis": None, "lower": None, "upper": None, "armature": None, "type": None}
    if prim.IsA(__import__("pxr").UsdPhysics.RevoluteJoint):
        out["type"] = "Revolute"
    elif prim.IsA(__import__("pxr").UsdPhysics.PrismaticJoint):
        out["type"] = "Prismatic"
    for key, attr_name in (
        ("axis", "physics:axis"),
        ("lower", "physics:lowerLimit"),
        ("upper", "physics:upperLimit"),
        ("armature", "physxJoint:armature"),
    ):
        a = prim.GetAttribute(attr_name)
        if a and a.IsValid() and a.HasAuthoredValue():
            try:
                v = a.Get()
                out[key] = float(v) if isinstance(v, (int, float)) else str(v)
            except (TypeError, ValueError):
                pass
    return out


def _collect_gripper_bundle(usd_path: str) -> dict:
    """Walk ``robot.usda`` and bundle every gripper joint per side.

    Output shape::

        {
          "gripper_left":  {
              "master":    {"name": ..., "path": ..., "drive": {...}, "geometry": {...}, "mimic": {...}, "mjc": {...}},
              "followers": [ {same shape}, ... ],
          },
          "gripper_right": {...},
          ...
        }

    Side keys are derived from the leading substring of each joint
    name (see ``_gripper_side_prefix``).  Sides with no master under
    them are still emitted so the validator can flag them.
    """
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    bundles: dict = {}
    if not os.path.isfile(usd_path):
        return bundles
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return bundles

    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue
        name = prim.GetName()
        if "gripper" not in name:
            continue
        side = _gripper_side_prefix(name)
        bundle = bundles.setdefault(side, {"master": None, "followers": []})

        info = {
            "name": name,
            "path": str(prim.GetPath()),
            "drive": _read_drive(prim, "angular"),
            "geometry": _read_joint_geometry(prim),
            "mimic": _read_mimic(prim),
            # mjwarp / Newton authoring shows up on a parallel mjc:* attr
            # tree.  Surface presence + raw values so a missing gainPrm
            # on a master can be distinguished from a present-but-zero one.
            "mjc": {
                "gainPrm": None,
                "biasPrm": None,
                "actfrcrange": None,
            },
        }
        for key, attr_name in (
            ("gainPrm", "mjc:joint:gainPrm"),
            ("biasPrm", "mjc:joint:biasPrm"),
            ("actfrcrange", "mjc:joint:actfrcrange"),
        ):
            a = prim.GetAttribute(attr_name)
            if a and a.IsValid() and a.HasAuthoredValue():
                try:
                    v = a.Get()
                    info["mjc"][key] = list(v) if hasattr(v, "__iter__") else float(v)
                except (TypeError, ValueError):
                    pass

        if name.endswith(_MASTER_SUFFIX):
            # If we see two masters under the same prefix (a regression
            # from a rebuild that doubled-authored), keep the first and
            # demote the rest to followers so the validator flags it.
            if bundle["master"] is None:
                bundle["master"] = info
            else:
                bundle["followers"].append(info)
        else:
            bundle["followers"].append(info)

    return bundles


def _stage_gripper_check(args, workdir: str, logger: _Logger) -> StageResult:
    """Collect the per-side gripper bundle from the final ``robot.usda``.

    Pure read — no edits.  The structured bundle moves to the
    validator; this stage just snapshots a JSON copy alongside the USD
    so post-mortems can diff bundles across rebuilds.
    """
    import json

    usd_path = os.path.join(workdir, "robot.usda")
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"[gripper] {usd_path} not found")
    bundles = _collect_gripper_bundle(usd_path)
    out_json = os.path.join(workdir, "10_gripper_bundle.json")
    try:
        with open(out_json, "w") as f:
            json.dump(bundles, f, indent=2, sort_keys=True)
    except Exception as exc:  # noqa: BLE001
        logger.warn(f"[gripper] failed to dump bundle JSON: {exc}")

    return StageResult(
        stage_name="gripper",
        output_path=usd_path,
        metrics={"bundle": bundles, "sides": len(bundles), "json": out_json},
        extra_paths=[out_json] if os.path.isfile(out_json) else [],
    )


def _validate_gripper(result: StageResult, logger: _Logger) -> int:
    """Walk the per-side bundle and flag anything that would explain a
    swinging / chattering / drooping gripper.

    Suspicious conditions (each contributes 1):

      * No gripper sides found at all → either the robot has no
        gripper (fine, returns 0 with an INFO) or the URDF→USD step
        dropped them (alarming, returns 1).
      * A side has no master joint at ``*inner_joint1``.
      * Master's DriveAPI is absent OR stiffness is 0 / None.
      * A follower has non-zero stiffness or damping (fights mimic
        constraint).
      * A follower has no ``newton:mimicJoint`` relationship, or the
        target is not the master we found under its side, or the
        coefficient isn't ±1.
      * Master has a ``newton:mimicJoint`` relationship (would loop
        back into the constraint solver).
      * Master / follower has no ``physxJoint:armature`` authored
        (rotor inertia missing → numerical chatter under PD load).
      * Effort / velocity limits suggest the joint is unactuated
        (effort==0 → mjwarp clamps PD torque to 0 → gripper droops
        regardless of stiffness).

    Each flagged condition prints a one-liner with side + joint name
    + the field value that tripped the check so the assemble-side
    fix is unambiguous.
    """
    import math

    # When invoked via ``--validate-only`` the harness synthesises a
    # StageResult with the generic ``_scan_usd_metrics`` payload, which
    # lacks the per-side bundle our stage runner builds.  Recover by
    # collecting from the USD on demand — the gripper inspector is
    # read-only, so re-running it costs nothing.
    bundles: dict
    if "bundle" in result.metrics and isinstance(result.metrics["bundle"], dict):
        bundles = result.metrics["bundle"]
    else:
        bundles = _collect_gripper_bundle(result.output_path)
    suspicious = 0

    if not bundles:
        # No gripper joints at all.  Could be intentional (chassis-only
        # robot) but on the G2 scenes this is wrong; flag as INFO and
        # let the user decide.  We don't bump suspicious here because we
        # can't tell from inside the validator whether a gripper is
        # expected for this robot model.
        logger.info("[gripper] no gripper joints found on robot.usda (expected for chassis-only robots).")
        return 0

    logger.info(f"[gripper] found {len(bundles)} gripper side(s): {sorted(bundles)}")

    # Expected runtime kp on the master, in Newton/mjwarp units.
    # ``_apply_mimic_joint_overlay`` authors per-degree=5 in USD, which
    # Newton's importer converts to per-radian by × π/180, so the value
    # we want to see in ``info['drive']['stiffness']`` is 5·π/180 ≈
    # 0.0873.  We let some slack — anything below 1e-6 is "missing" for
    # the master and the gripper will droop.
    MASTER_STIFFNESS_AUTHORED_MIN = 1e-6

    for side, bundle in sorted(bundles.items()):
        master = bundle["master"]
        followers = bundle["followers"]

        logger.info(f"[gripper] {side}: master={'YES' if master else 'NO'} " f"followers={len(followers)}")

        # ---------- Master checks ----------
        if master is None:
            logger.warn(
                f"[gripper:{side}] NO master joint (expected name ending in "
                f"{_MASTER_SUFFIX!r}).  The mimic graph has no root — every "
                f"follower will swing free."
            )
            suspicious += 1
        else:
            d = master["drive"]
            geo = master["geometry"]
            mim = master["mimic"]
            if not d["applied"] or d["stiffness"] is None or d["stiffness"] < MASTER_STIFFNESS_AUTHORED_MIN:
                logger.warn(
                    f"[gripper:{side}] master {master['name']} has stiffness="
                    f"{d['stiffness']!r} (applied={d['applied']}).  "
                    f"_apply_mimic_joint_overlay should have authored ~"
                    f"{5.0 * math.pi / 180.0:.4f} rad-units; check stage 9."
                )
                suspicious += 1
            else:
                # Re-derive the per-degree value for legibility — that's
                # what the MJCF reference and physics_params.yaml use.
                deg_val = d["stiffness"] * 180.0 / math.pi
                logger.info(
                    f"[gripper:{side}]   master drive: stiffness={d['stiffness']:.4f} "
                    f"(={deg_val:.2f}/deg)  damping={d['damping']}  "
                    f"max_force={d['max_force']}"
                )
            if geo["armature"] is None:
                logger.warn(
                    f"[gripper:{side}] master {master['name']}: no "
                    f"physxJoint:armature authored.  Reflected rotor "
                    f"inertia missing → expect numerical chatter under PD."
                )
                suspicious += 1
            if mim["target"] is not None:
                logger.warn(
                    f"[gripper:{side}] master {master['name']}: has a "
                    f"newton:mimicJoint relationship → "
                    f"{mim['target']!r}.  A master must not mimic anything "
                    f"or the constraint graph loops."
                )
                suspicious += 1

        # ---------- Follower checks ----------
        master_name = master["name"] if master else None
        for foll in followers:
            d = foll["drive"]
            geo = foll["geometry"]
            mim = foll["mimic"]

            # Drive must be authored AND zero — see
            # ``_apply_mimic_joint_overlay`` for the rationale.
            if d["applied"]:
                if d["stiffness"] is not None and d["stiffness"] > MASTER_STIFFNESS_AUTHORED_MIN:
                    logger.warn(
                        f"[gripper:{side}] follower {foll['name']}: drive "
                        f"stiffness={d['stiffness']!r} should be 0 — fights "
                        f"the mimic constraint → chatter / oscillation."
                    )
                    suspicious += 1
                if d["damping"] is not None and abs(d["damping"]) > MASTER_STIFFNESS_AUTHORED_MIN:
                    logger.warn(
                        f"[gripper:{side}] follower {foll['name']}: drive " f"damping={d['damping']!r} should be 0."
                    )
                    suspicious += 1
            else:
                # On isaac_newton the follower DriveAPI MUST exist (even
                # at zero) for Newton's importer to create a POSITION
                # actuator — without it ``apply_action`` writes go
                # nowhere.  Newton-standalone with mjwarp tolerates the
                # absence, so we warn-not-fail to match both paths.
                logger.warn(
                    f"[gripper:{side}] follower {foll['name']}: NO "
                    f"angular DriveAPI applied.  On isaac_newton, Newton "
                    f"will create an EFFORT actuator → ``apply_action`` "
                    f"writes have no effect → follower swings free."
                )
                suspicious += 1

            if mim["target"] is None:
                logger.warn(
                    f"[gripper:{side}] follower {foll['name']}: no "
                    f"newton:mimicJoint relationship.  Will not track "
                    f"the master under either engine path."
                )
                suspicious += 1
            elif master_name is not None and mim["target"] != master_name:
                logger.warn(
                    f"[gripper:{side}] follower {foll['name']}: "
                    f"mimicJoint → {mim['target']!r} but expected master "
                    f"is {master_name!r}.  Cross-side mimic edge — "
                    f"the wrong gripper will move."
                )
                suspicious += 1
            elif not (mim["newton_api"] or mim["physx_api"]):
                # ``newton:mimicJoint`` rel + ``newton:mimicCoef1/0``
                # attrs are authored, but the prim has neither
                # ``NewtonMimicAPI`` nor ``PhysxMimicJointAPI:<axis>``
                # applied.  Newton's native USD importer keys off the
                # applied API to decide whether to install a mimic
                # equality constraint at finalize time — without the
                # schema, the rel + coefs are inert metadata and the
                # only thing that broadcasts master → follower is the
                # software ``apply_action`` path in ``engine/_mimic.py``.
                # That works on newton-standalone but breaks down on
                # ``isaac_newton`` whenever the wrapper bypasses the
                # software broadcast (e.g. when the controller writes
                # directly to the tensor handle), at which point
                # followers fall back to whatever target_pos was last
                # written → finger weight pulls them open → swing.
                logger.warn(
                    f"[gripper:{side}] follower {foll['name']}: "
                    f"newton:mimicJoint rel authored but neither "
                    f"NewtonMimicAPI nor PhysxMimicJointAPI:<axis> is "
                    f"applied to the prim.  Newton's USD importer "
                    f"won't install a constraint-level mimic — only "
                    f"the software broadcast in engine/_mimic.py does.  "
                    f"On isaac_newton this often manifests as the "
                    f"gripper swinging while arms / body track."
                )
                suspicious += 1

            # mimicCoef1 sanity.  Most parallel-jaw grippers use ±1, but
            # underactuated multi-finger designs (e.g. omnipicker)
            # legitimately specify non-unit ratios per finger
            # (URDF: ``<mimic joint="…" multiplier="0.1"/>``).  So we
            # only flag values that would BREAK the gripper: exactly
            # zero (follower never moves) or non-finite.  For unusual
            # but non-broken values we surface an INFO so you can
            # eyeball them when diagnosing.
            if not math.isfinite(mim["coef1"]):
                logger.warn(
                    f"[gripper:{side}] follower {foll['name']}: "
                    f"mimicCoef1={mim['coef1']!r} is not finite — "
                    f"the importer probably failed to parse the "
                    f"URDF ``<mimic multiplier>`` value."
                )
                suspicious += 1
            elif abs(mim["coef1"]) < 1e-6:
                logger.warn(
                    f"[gripper:{side}] follower {foll['name']}: "
                    f"mimicCoef1={mim['coef1']:.4f} is effectively zero — "
                    f"follower will not track the master regardless of "
                    f"drive / constraint settings."
                )
                suspicious += 1
            elif abs(abs(mim["coef1"]) - 1.0) > 1e-3:
                logger.info(
                    f"[gripper:{side}]   follower {foll['name']} mimicCoef1="
                    f"{mim['coef1']:+.4f}  (non-unit; expected for "
                    f"underactuated multi-finger grippers — verify "
                    f"against URDF ``<mimic multiplier>``)."
                )

            if geo["armature"] is None:
                logger.warn(f"[gripper:{side}] follower {foll['name']}: no " f"physxJoint:armature authored.")
                suspicious += 1

    if suspicious == 0:
        logger.info(
            f"[gripper] OK: all {len(bundles)} gripper side(s) cleanly authored "
            f"(master kp non-zero, followers stiffness=damping=0, mimic edges resolve)."
        )
    return suspicious


# ----------------------------------------------------------------------
# USD scanner used by multiple validators
# ----------------------------------------------------------------------


def _scan_usd_metrics(usd_path: str) -> dict:
    """Top-level structural metrics for a robot.usda.  Used by the
    isaac_pipeline validator and as a sanity check during ``--inspect``.
    """
    metrics = {
        "joint_total": 0,
        "revolute_count": 0,
        "fixed_count": 0,
        "prismatic_count": 0,
        "rigid_body_count": 0,
        "collision_api_count": 0,
        "mimic_api_count": 0,
        "drive_api_count": 0,
        "armature_authored_count": 0,
        "payload_dir_exists": False,
    }
    if not os.path.isfile(usd_path):
        return metrics
    try:
        from pxr import Usd, UsdPhysics  # noqa: PLC0415
    except ImportError:
        return metrics
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return metrics
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            metrics["revolute_count"] += 1
            metrics["joint_total"] += 1
        elif prim.IsA(UsdPhysics.FixedJoint):
            metrics["fixed_count"] += 1
            metrics["joint_total"] += 1
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            metrics["prismatic_count"] += 1
            metrics["joint_total"] += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            metrics["rigid_body_count"] += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            metrics["collision_api_count"] += 1
        if prim.HasAPI(UsdPhysics.DriveAPI):
            metrics["drive_api_count"] += 1
        arm = prim.GetAttribute("physxJoint:armature")
        if arm and arm.IsValid() and arm.HasAuthoredValue():
            metrics["armature_authored_count"] += 1
    payload_dir = os.path.join(os.path.dirname(usd_path), "payloads")
    metrics["payload_dir_exists"] = os.path.isdir(payload_dir)
    return metrics


# ----------------------------------------------------------------------
# Stage registry
# ----------------------------------------------------------------------

STAGES = [
    Stage(1, "xacro", "xacro → URDF (resolves package URIs)", _stage_xacro, _validate_xacro),
    Stage(
        2,
        "isaac_pipeline",
        "Isaac URDF→USD + schemas + AS3 (monolithic)",
        _stage_isaac_pipeline,
        _validate_isaac_pipeline,
    ),
    # Stages 3+4 are folded into isaac_pipeline; we keep them as distinct
    # numbers so future-us can split when the Isaac importer exposes per-
    # step hooks.  Empty-stub entries here would be confusing — don't
    # register them yet; ``--from-stage 3`` resolves to isaac_pipeline.
    Stage(
        5,
        "material_overrides",
        "apply parsed <material_override> PBR patches",
        _stage_material_overrides,
        _validate_material_overrides,
    ),
    Stage(
        8,
        "collision_policy",
        "apply selective collision policy on the final USD",
        _stage_collision_policy,
        _validate_collision_policy,
    ),
    Stage(
        9,
        "mimic_overlay",
        "author per-class joint USD attributes (armature + drive)",
        _stage_mimic_overlay,
        _validate_mimic_overlay,
    ),
    Stage(
        10,
        "gripper",
        "deep-inspect gripper master/follower drive + mimic edges (read-only)",
        _stage_gripper_check,
        _validate_gripper,
    ),
]


def _stages_in_range(from_n: int, to_n: int, only: Optional[str]) -> list:
    if only is not None:
        return [s for s in STAGES if s.name == only]
    return [s for s in STAGES if from_n <= s.number <= to_n]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scene",
        default="",
        help=(
            "scene name (e.g. ``scene_flat_g2_sp``).  One of --scene or --urdf "
            "is required for stage 1.  Stages 5+ can run against an existing "
            "workdir without either."
        ),
    )
    parser.add_argument(
        "--urdf",
        default=None,
        help=(
            "Path to a URDF file to use as stage-1 input. When set, the xacro "
            "step is skipped entirely (no scene-yaml lookup, no ament resolution). "
            "Mirrors the production ``assemble_robot.py --urdf`` debug switch."
        ),
    )
    parser.add_argument(
        "--workdir",
        default="/tmp/test_assemble_robot",
        help="working directory for stage outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        default=1,
        help="first stage to run (1-9; default 1)",
    )
    parser.add_argument(
        "--to-stage",
        type=int,
        default=10,
        help="last stage to run (1-10; default 10)",
    )
    parser.add_argument(
        "--only-stage",
        default=None,
        metavar="NAME",
        help=(
            "run exactly one stage by name (xacro, isaac_pipeline, "
            "material_overrides, collision_policy, "
            "mimic_overlay, gripper).  Overrides --from-stage / --to-stage."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run the per-stage validator after each stage",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "run validators against existing snapshots in --workdir without "
            "re-running any stage.  Useful for iterating on validator code."
        ),
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="print structural metrics of the final USD after the run",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="rm -rf --workdir before running (forces a fresh run)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logger = _Logger()

    if args.clean and os.path.isdir(args.workdir):
        logger.info(f"--clean: rm -rf {args.workdir}")
        shutil.rmtree(args.workdir)
    os.makedirs(args.workdir, exist_ok=True)

    if args.validate_only:
        # Re-run validators against the snapshots that already exist in
        # workdir.  Each validator gets a synthetic StageResult pointing
        # at the LATEST robot.usda so they can re-scan it.
        usd_path = os.path.join(args.workdir, "robot.usda")
        if not os.path.isfile(usd_path):
            logger.error(f"--validate-only: {usd_path} not found")
            return 2
        total_susp = 0
        for s in STAGES:
            try:
                r = StageResult(s.name, usd_path, _scan_usd_metrics(usd_path), [])
                total_susp += s.validate(r, logger)
            except Exception as exc:
                logger.warn(f"validator {s.name} crashed: {exc}")
                total_susp += 1
        logger.info(f"--validate-only: {total_susp} suspicious entries across all validators")
        return 0 if total_susp == 0 else 4

    stages = _stages_in_range(args.from_stage, args.to_stage, args.only_stage)
    if not stages:
        logger.error(f"no stages selected (from={args.from_stage}, to={args.to_stage}, only={args.only_stage})")
        return 2

    # Stage 1 (xacro) requires either --scene or --urdf; later stages can
    # run against an existing workdir.  Fail fast if the user asked for
    # stage 1 without an input source.
    if any(s.name == "xacro" for s in stages) and not args.scene and not args.urdf:
        logger.error(
            "stage 'xacro' requires --scene <name> or --urdf <path>.  Pass "
            "--from-stage 5 to skip xacro+isaac and run only post-AS3 stages "
            "on an existing robot.usda in --workdir."
        )
        return 2

    total_susp = 0
    for s in stages:
        logger.info(f"=== stage {s.number}: {s.name} ({s.description}) ===")
        try:
            result = s.run(args, args.workdir, logger)
        except Exception as exc:
            logger.error(f"stage {s.name} crashed: {exc}")
            import traceback

            traceback.print_exc()
            return 3
        if args.validate:
            total_susp += s.validate(result, logger)

    if args.inspect:
        usd_path = os.path.join(args.workdir, "robot.usda")
        m = _scan_usd_metrics(usd_path)
        logger.info(f"=== final inspect ({usd_path}) ===")
        for k, v in m.items():
            logger.info(f"  {k:30s} {v}")

    if args.validate and total_susp > 0:
        logger.warn(f"validation reported {total_susp} suspicious entries — see above")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
