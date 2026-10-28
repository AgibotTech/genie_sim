# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Post-process the dumped ``robot_runtime.xml`` so pure-MuJoCo tools
can replay it at the same cadence the live engine runs.

What Newton's ``_convert_to_mjc`` doesn't write
-----------------------------------------------

Newton's MJCF emitter populates ``spec.option.disableflags``,
``gravity``, ``solver``, ``integrator``, ``iterations``,
``ls_iterations``, ``cone``, ``jacobian``, ``impratio``, etc. — but
NOT ``spec.option.timestep``.  As a result the dumped MJCF inherits
MuJoCo's built-in default of 0.002 s regardless of what dt the live
solver actually steps at.  When the launcher requests anything other
than (physics_hz=100, sim_substeps=5) → (sim_dt=2 ms), the dump
silently misrepresents the runtime by up to whatever the substep ratio
is.

This module patches that gap so the file on disk is honest:

  * ``<option timestep="…"/>`` rewritten to the live substep dt
    ``= 1 / (physics_hz × sim_substeps)``.
  * A ``<custom>`` block carries ``physics_hz`` and ``sim_substeps``
    as ``<numeric>`` entries — MuJoCo loads these into
    ``mjModel.numeric_*`` so any CPU-MuJoCo tool
    (``test_robot_xml_dynamic.py``, ``mujoco.viewer``, hand-written
    notebooks) can recover the outer-loop cadence and match the
    runtime control rate.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Optional


def augment_mjcf_timing(
    mjcf_path: str,
    sim_substeps: int,
    physics_hz: float,
    logger: Optional[Any] = None,
) -> None:
    """Rewrite ``<option timestep>`` and append a ``<custom>`` cadence
    block on the MJCF at ``mjcf_path``.

    Parameters
    ----------
    mjcf_path:
        File to mutate in place.  Must exist; this is called AFTER
        Newton has dumped the file.  Skipped silently when the path
        is empty or the file is missing — Newton's MJCF dump is best-
        effort and we don't want a missing-file edge case to abort
        startup.
    sim_substeps:
        Per-frame substep count the engine is running.  Used both to
        compute ``sim_dt`` and to seed the recoverable
        ``sim_substeps`` numeric attribute.
    physics_hz:
        Outer-frame rate of the engine loop.  Must be > 0 — if zero,
        we cannot compute sim_dt and the augment is skipped with a
        warning.
    logger:
        Optional logger for diagnostic info.  When absent we fall
        back to ``print``.

    Behaviour
    ---------
    * If ``<option>`` exists, its ``timestep`` attribute is set (or
      added).  Other attributes on the element are preserved.
    * If ``<option>`` doesn't exist (shouldn't happen with a Newton-
      authored MJCF but defensive), one is inserted at the start of
      the ``<mujoco>`` root.
    * A new ``<custom>`` block is appended (or merged with an
      existing one) carrying::

          <numeric name="physics_hz"   data="<physics_hz>"/>
          <numeric name="sim_substeps" data="<sim_substeps>"/>
          <numeric name="sim_dt"       data="<sim_dt>"/>

      ``sim_dt`` is redundant with ``opt.timestep`` but cheap to
      include, and saves tools a divide.
    * Pre-existing ``<numeric>`` entries with the same name are
      replaced (so re-running the augment on an already-augmented
      file is idempotent — useful when isaac_newton's ``startup()``
      runs the dump once, the wrapper re-runs after a reload, etc.).
    """
    import os

    def _log_info(msg: str) -> None:
        if logger is not None and hasattr(logger, "info"):
            logger.info(msg)
        else:
            print(msg, flush=True)

    def _log_warn(msg: str) -> None:
        if logger is not None and hasattr(logger, "warn"):
            logger.warn(msg)
        else:
            print(f"WARN: {msg}", flush=True)

    if not mjcf_path:
        return
    if not os.path.isfile(mjcf_path):
        _log_warn(f"[mjcf-postprocess] file {mjcf_path!r} does not exist; " f"timing augment skipped.")
        return
    if physics_hz <= 0.0:
        _log_warn(
            f"[mjcf-postprocess] physics_hz={physics_hz!r} is not > 0; "
            f"cannot compute sim_dt for {mjcf_path}.  Leaving "
            f"<option timestep> at MuJoCo's default and skipping the "
            f"<custom> block.  Pass physics_hz to the adapter / "
            f"engine call site to enable this augment."
        )
        return
    if sim_substeps <= 0:
        _log_warn(
            f"[mjcf-postprocess] sim_substeps={sim_substeps!r} is not "
            f"> 0; cannot compute sim_dt for {mjcf_path}.  Skipping."
        )
        return

    sim_dt = 1.0 / (float(physics_hz) * float(sim_substeps))

    try:
        tree = ET.parse(mjcf_path)
    except ET.ParseError as exc:
        _log_warn(f"[mjcf-postprocess] could not parse {mjcf_path}: {exc!r}.  " f"File left as-is.")
        return
    root = tree.getroot()
    if root.tag != "mujoco":
        _log_warn(
            f"[mjcf-postprocess] {mjcf_path} root tag is {root.tag!r} " f"(expected 'mujoco').  Aborting augment."
        )
        return

    # ----- <option timestep="..."/> ------------------------------------
    option_elem = root.find("option")
    if option_elem is None:
        # Insert at start of root so it visually sits with the other
        # top-level config blocks Newton emits.
        option_elem = ET.Element("option")
        root.insert(0, option_elem)
    option_elem.set("timestep", f"{sim_dt:.10g}")

    # ----- <custom><numeric .../></custom> ------------------------------
    custom_elem = root.find("custom")
    if custom_elem is None:
        custom_elem = ET.SubElement(root, "custom")

    # Replace any existing numeric with the same name; idempotent.
    desired = {
        "physics_hz": f"{float(physics_hz):.10g}",
        "sim_substeps": f"{int(sim_substeps)}",
        "sim_dt": f"{sim_dt:.10g}",
    }
    existing_by_name = {n.get("name"): n for n in custom_elem.findall("numeric")}
    for name, data in desired.items():
        node = existing_by_name.get(name)
        if node is None:
            node = ET.SubElement(custom_elem, "numeric")
            node.set("name", name)
        node.set("data", data)

    # Indent for readability (Python 3.9+).
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)

    _log_info(
        f"[mjcf-postprocess] {mjcf_path}: opt.timestep={sim_dt:.6g} s "
        f"(physics_hz={physics_hz:g}, sim_substeps={int(sim_substeps)}); "
        f"<custom> block carries physics_hz / sim_substeps / sim_dt "
        f"as recoverable numerics."
    )


def normalize_mjcf_names(
    mjcf_path: str,
    *,
    joint_prefix_strip: str = "",
    body_prefix_strip: str = "",
    actuator_position_to_motor: bool = True,
    logger: Optional[Any] = None,
) -> None:
    """Rewrite body / joint / actuator names so they match the
    convention used by the hand-authored G2 reference MJCF.

    The static / dynamic XML diff tools index by exact name, so the
    dumped MJCF's wrapper-side conventions
    (``_<prefix>_Geometry_<full ancestor path>``,
    ``_<prefix>_Physics_<joint>``, ``position:<joint>``) prevent any
    match against the reference's flat ``base_link`` / ``idx01_*`` /
    ``position_*`` / ``velocity_*`` / ``motor_*`` naming.  The
    behavioural data is identical; only the names differ.

    Prefix auto-detection
    ---------------------
    The actual prim-scope prefix varies by scene
    (``_genie_Physics_`` for production, ``_robot_Physics_`` for
    ``test_newton_solver.py``).  When ``joint_prefix_strip`` /
    ``body_prefix_strip`` are empty (default), we auto-detect via the
    regex ``^_[A-Za-z0-9]+_(Physics|Geometry)_`` on the first joint /
    body name encountered.  Pass explicit values to override.

    Transformations applied
    -----------------------

    1. **Bodies.**  Each ``<body>`` carries a name like
       ``_genie_Geometry_base_link_body_link1_..._arm_l_link7`` —
       Newton concatenates every ancestor's short name onto each
       descendant.  We walk parent → child and compute each body's
       short name as
       ``body.name.removeprefix(parent.name + '_')`` (the part NOT in
       the parent's full name).  The root body (no parent under
       ``<worldbody>``) is short-named by stripping the auto-
       detected ``_<prefix>_Geometry_``.
    2. **Joints.**  Strip the auto-detected ``_<prefix>_Physics_``
       from every ``<joint name>`` attribute.  Equality constraints
       reference joints by name so we update ``<joint joint1=
       joint2=>`` in the ``<equality>`` block too.
    3. **Actuators.**  Each actuator's ``joint=`` is updated to the
       short joint name.  When ``actuator_position_to_motor=True``
       (default), the actuator's own ``name=`` is rewritten from
       ``position:<joint>`` (Newton's emit convention for
       POSITION-mode actuators) to ``<prefix>_<short_joint>`` where
       ``<prefix>`` is picked by control mode from the MJCF
       ``biasprm`` shape: ``position_`` (PD spring),
       ``velocity_`` (velocity damper) or ``motor_`` (raw torque
       pass-through).  Matches the hand-authored G2 convention
       documented in
       ``genie_robot_description/docs/naming_convention.mjcf.actuator.md``.

    Idempotent.  Re-running on an already-normalised file is a
    no-op.

    Parameters
    ----------
    mjcf_path:
        File to mutate in place.  Skipped silently when empty /
        missing.
    joint_prefix_strip:
        Prefix to strip from joint names.  Empty (default) →
        auto-detect via the regex above.  Pass an explicit prefix to
        force, or pass ``"<no-strip>"`` (any non-empty string that
        doesn't appear in the file) to disable joint name editing.
    body_prefix_strip:
        Prefix to strip from the ROOT body's name.  Empty (default)
        → auto-detect.
    actuator_position_to_motor:
        When True, ``position:<joint>`` actuator names are rewritten
        to ``motor_<short_joint>``.  Disable to keep Newton's
        original naming.
    logger:
        Optional logger.

    What we don't touch
    -------------------

    * ``<geom>`` / ``<site>`` / ``<camera>`` references — those
      reference bodies via parent containment, not by attribute
      name, so no edits needed.
    * ``<sensor>`` blocks (``objname=``, ``site=``, etc.) — none
      authored by Newton's converter.
    * ``<contact>`` / ``<exclude>`` pair lists — Newton doesn't
      emit named-pair contact rules.
    """
    import os
    import re

    def _log_info(msg: str) -> None:
        if logger is not None and hasattr(logger, "info"):
            logger.info(msg)
        else:
            print(msg, flush=True)

    def _log_warn(msg: str) -> None:
        if logger is not None and hasattr(logger, "warn"):
            logger.warn(msg)
        else:
            print(f"WARN: {msg}", flush=True)

    if not mjcf_path:
        return
    if not os.path.isfile(mjcf_path):
        _log_warn(f"[mjcf-postprocess] file {mjcf_path!r} does not exist; " f"naming normalisation skipped.")
        return

    try:
        tree = ET.parse(mjcf_path)
    except ET.ParseError as exc:
        _log_warn(f"[mjcf-postprocess] could not parse {mjcf_path}: {exc!r}.  " f"File left as-is.")
        return
    root = tree.getroot()
    if root.tag != "mujoco":
        _log_warn(f"[mjcf-postprocess] {mjcf_path} root tag is {root.tag!r}; " f"aborting naming normalisation.")
        return

    # ----- Auto-detect prefix when not explicitly specified ----------
    _RE_PREFIX_BODY = re.compile(r"^(_[A-Za-z0-9]+_Geometry_)")
    _RE_PREFIX_JOINT = re.compile(r"^(_[A-Za-z0-9]+_Physics_)")

    if not body_prefix_strip:
        worldbody_tmp = root.find("worldbody")
        if worldbody_tmp is not None:
            for b in worldbody_tmp.iter("body"):
                bn = b.get("name", "")
                m = _RE_PREFIX_BODY.match(bn)
                if m:
                    body_prefix_strip = m.group(1)
                    break
    if not joint_prefix_strip:
        for j in root.iter("joint"):
            jn = j.get("name", "")
            m = _RE_PREFIX_JOINT.match(jn)
            if m:
                joint_prefix_strip = m.group(1)
                break

    # ----- Bodies ----------------------------------------------------
    # Walk every <body> in document order under <worldbody>.  Parents
    # are visited before children (XML order), so we can record each
    # body's stripped (short) name and use it as the prefix to strip
    # for its descendants.  Map full→short for later use.
    body_full_to_short: dict[str, str] = {}

    def _strip_root(name: str) -> str:
        return name[len(body_prefix_strip) :] if body_prefix_strip and name.startswith(body_prefix_strip) else name

    def _recurse(elem: ET.Element, parent_full: Optional[str]) -> None:
        for body in elem.findall("body"):
            full = body.get("name", "")
            if not full:
                _recurse(body, parent_full)
                continue
            if parent_full and full.startswith(parent_full + "_"):
                short = full[len(parent_full) + 1 :]
            else:
                short = _strip_root(full)
            body.set("name", short)
            body_full_to_short[full] = short
            # Children should be diffed against THIS body's ORIGINAL
            # full name (since their concatenated names include it).
            _recurse(body, full)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        _recurse(worldbody, parent_full=None)
    n_bodies = len(body_full_to_short)

    # ----- Joints ----------------------------------------------------
    joint_full_to_short: dict[str, str] = {}

    def _strip_joint(name: str) -> str:
        return name[len(joint_prefix_strip) :] if joint_prefix_strip and name.startswith(joint_prefix_strip) else name

    # Joints live inside <body> elements; iterate everywhere because
    # Newton's emit varies a bit by joint type.
    for joint in root.iter("joint"):
        # Skip <joint joint1= joint2=> inside <equality> — different
        # schema (the element is also tagged "joint" inside equality
        # but its attributes are joint1/joint2/polycoef, not name).
        if joint.get("name") is None:
            continue
        old = joint.get("name", "")
        new = _strip_joint(old)
        if new != old:
            joint.set("name", new)
        joint_full_to_short[old] = new

    # ----- Equality joint references ---------------------------------
    eq_block = root.find("equality")
    n_eq_updated = 0
    if eq_block is not None:
        for eq in eq_block:
            for attr in ("joint1", "joint2"):
                v = eq.get(attr)
                if v is None:
                    continue
                new = joint_full_to_short.get(v, _strip_joint(v))
                if new != v:
                    eq.set(attr, new)
                    n_eq_updated += 1

    # ----- Contact <exclude body1= body2=> references ----------------
    # Newton emits self-collision exclusion pairs in
    # ``<contact><exclude body1=... body2=...>``.  Those reference
    # the FULL concatenated body names; rewrite to the short forms
    # the bodies now carry so the file stays internally consistent
    # (MuJoCo would reject references to bodies it can't resolve).
    contact_block = root.find("contact")
    n_contact_updated = 0
    if contact_block is not None:
        for excl in contact_block.findall("exclude"):
            for attr in ("body1", "body2"):
                v = excl.get(attr)
                if v is None:
                    continue
                new = body_full_to_short.get(v, v)
                if new != v:
                    excl.set(attr, new)
                    n_contact_updated += 1

    # ----- Actuators -------------------------------------------------
    actuator_block = root.find("actuator")
    n_act = 0
    n_act_renamed = 0
    if actuator_block is not None:
        # Naming rule (drives the prefix): we name actuators by what
        # they DO, not by which robot region they live on.  Newton's
        # mjwarp converter emits ``<general>`` elements in three
        # shapes, all distinguishable from their ``gainprm`` /
        # ``biasprm`` attributes alone — no joint-class lookup needed:
        #
        #   POSITION-PD spring  →  gainprm="<kp>"  biasprm="0 -<kp> -<kd>"
        #     (biasprm[1] != 0 — has a position-error term)
        #     → name: ``position_<joint>``
        #
        #   VELOCITY damper      →  gainprm="<kd>"  biasprm="0 0 -<kd>"
        #     (biasprm[0]=0, biasprm[1]=0, biasprm[2]=-kd — no position term,
        #     has a velocity-error term)
        #     → name: ``velocity_<joint>``
        #
        #   Raw-torque pass-through  →  no gainprm, no biasprm
        #     (sibling actuator added by add_raw_torque_motor_actuators)
        #     → name: ``motor_<joint>``
        #
        # That gives every chassis_steer joint (POSITION-mode) the
        # ``position_`` prefix, every chassis_drive wheel (VELOCITY-mode)
        # the ``velocity_`` prefix, and every body/head/arm joint two
        # actuators: ``position_<joint>`` (the PD spring) plus a sibling
        # ``motor_<joint>`` (raw torque).  Classifying by control mode
        # (rather than by joint name) keeps chassis_steer correctly
        # tagged as a position controller.  The convention matches the
        # hand-authored G2 reference MJCF; see
        # ``genie_robot_description/docs/naming_convention.mjcf.actuator.md``.
        def _act_prefix_from_attrs(act_el) -> str:
            biasprm = act_el.get("biasprm")
            if biasprm is None:
                # Raw-torque pass-through — sibling motor actuator
                return "motor_"
            try:
                parts = [float(x) for x in biasprm.split()]
            except ValueError:
                # Malformed biasprm — fall back to the safe default
                return "position_"
            # POSITION-emit has biasprm[1] = -kp != 0.
            # VELOCITY-emit has biasprm[0] = biasprm[1] = 0 with
            #   biasprm[2] = -kd (velocity-error term only).
            # Anything else falls back to motor_ (raw torque).
            has_position_term = len(parts) >= 2 and parts[1] != 0.0
            if has_position_term:
                return "position_"
            has_velocity_term = len(parts) >= 3 and parts[2] != 0.0
            if has_velocity_term:
                return "velocity_"
            return "motor_"

        for act in actuator_block:
            n_act += 1
            jv = act.get("joint")
            if jv is not None:
                act.set("joint", joint_full_to_short.get(jv, _strip_joint(jv)))
            cur_name = act.get("name")
            joint_short = act.get("joint")
            # Strategy when ``actuator_position_to_motor`` is on:
            #   * ``position:<joint>`` (Newton POSITION-mode emit)
            #     → ``<prefix><short_joint>`` where ``<prefix>`` is
            #     decided by control-mode (see ``_act_prefix_from_attrs``).
            #   * Anonymous actuator (no ``name=``) — Newton-standalone's
            #     ``_convert_to_mjc`` emits unnamed ``<general/>``
            #     elements; we name them ourselves with the same
            #     prefix logic.
            #   * Other authored names — strip the
            #     ``_<prefix>_Physics_`` joint prefix for consistency.
            if actuator_position_to_motor and cur_name and cur_name.startswith("position:"):
                inner = cur_name[len("position:") :]
                inner_short = joint_full_to_short.get(inner, _strip_joint(inner))
                act.set("name", f"{_act_prefix_from_attrs(act)}{inner_short}")
                n_act_renamed += 1
            elif not cur_name and joint_short and actuator_position_to_motor:
                act.set("name", f"{_act_prefix_from_attrs(act)}{joint_short}")
                n_act_renamed += 1
            elif cur_name:
                short = joint_full_to_short.get(cur_name, _strip_joint(cur_name))
                if short != cur_name:
                    act.set("name", short)

    # Indent and write back.
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)

    _log_info(
        f"[mjcf-postprocess] {mjcf_path}: renamed {n_bodies} body name(s), "
        f"{len(joint_full_to_short)} joint name(s) (prefix "
        f"{joint_prefix_strip!r} stripped), {n_act_renamed}/{n_act} "
        f"actuator(s) named position_* / velocity_* / motor_* by control mode, "
        f"{n_eq_updated} equality joint "
        f"reference(s) + {n_contact_updated} contact-exclude reference(s) "
        f"updated.  File now matches the reference G2 MJCF convention."
    )


def add_raw_torque_motor_actuators(
    mjcf_path: str,
    *,
    logger: Optional[Any] = None,
) -> None:
    """Add sibling raw-torque ``motor_<joint>`` actuators alongside
    the existing ``position_<joint>`` PD-spring actuators.

    Why this matches the reference convention
    -----------------------------------------
    The hand-authored reference G2 MJCF exposes TWO actuators per
    body/head/arm joint:

      * ``position_<joint>`` — affine PD spring (gainprm/biasprm),
        used for set-point control.  Newton emits this on its own.
      * ``motor_<joint>`` — raw torque pass-through
        (``<general/>`` with only ``joint=`` + ``ctrlrange=`` +
        ``forcerange=``, no gainprm/biasprm).  Used for direct
        torque commands.

    Downstream policies pick which one to drive via ``ctrl[i]`` on
    the matching actuator.  Newton's converter only emits the
    position-PD path; this function fills in the motor sibling so
    the dumped MJCF carries the full reference contract.

    Scope
    -----
    Only emits motor siblings for body / head / arm joints.  Chassis
    joints already use ``motor_*`` naming as their primary actuator
    (and don't get a position sibling in the reference), so we leave
    them alone.  Gripper masters are also skipped — the reference
    MJCF only has the position actuator for them too.

    Idempotent.  Re-running on a file that already has motor sibling
    actuators is a no-op (we detect by name lookup).

    Parameters
    ----------
    mjcf_path:
        File to mutate in place.  Skipped silently when empty /
        missing.
    logger:
        Optional logger.

    Author convention for the new motor actuators
    ---------------------------------------------
      ``<general name="motor_<joint>" joint="<joint>"
                 ctrlrange="<lo> <hi>" forcerange="<lo> <hi>" />``

    where ``<lo>``/``<hi>`` come from the joint's own
    ``actuatorfrcrange`` if present, else the actuator's own
    ``forcerange`` if present, else are omitted (raw motor with no
    cap — MuJoCo's default).
    """
    import os

    def _log_info(msg: str) -> None:
        if logger is not None and hasattr(logger, "info"):
            logger.info(msg)
        else:
            print(msg, flush=True)

    def _log_warn(msg: str) -> None:
        if logger is not None and hasattr(logger, "warn"):
            logger.warn(msg)
        else:
            print(f"WARN: {msg}", flush=True)

    if not mjcf_path:
        return
    if not os.path.isfile(mjcf_path):
        _log_warn(f"[mjcf-postprocess] file {mjcf_path!r} does not exist; " f"motor-sibling emit skipped.")
        return

    try:
        tree = ET.parse(mjcf_path)
    except ET.ParseError as exc:
        _log_warn(f"[mjcf-postprocess] could not parse {mjcf_path}: {exc!r}.  " f"File left as-is.")
        return
    root = tree.getroot()
    actuator_block = root.find("actuator")
    if actuator_block is None:
        _log_warn(f"[mjcf-postprocess] {mjcf_path} has no <actuator> block; " f"nothing to add motor siblings to.")
        return

    # Build a quick joint-name → actuatorfrcrange lookup from the
    # <joint> elements so we can give each new motor actuator a
    # matching ctrlrange / forcerange.
    joint_actfrcrange: dict[str, str] = {}
    for j in root.iter("joint"):
        name = j.get("name")
        if not name:
            continue
        rng = j.get("actuatorfrcrange") or j.get("actfrcrange")
        if rng:
            joint_actfrcrange[name] = rng

    from genie_sim_engine_newton.common.joint_classification import (  # noqa: PLC0415
        JK_BODY,
        JK_HEAD,
        JK_ARM,
        JK_ARM_SHOULDER,
        JK_ARM_MID,
        JK_ARM_WRIST,
        classify_joint_by_name,
    )

    eligible_classes = {JK_BODY, JK_HEAD, JK_ARM, JK_ARM_SHOULDER, JK_ARM_MID, JK_ARM_WRIST}

    # Existing actuator names so we can skip joints that already have
    # a motor sibling (idempotency).
    existing_names = {a.get("name") for a in actuator_block if a.get("name")}

    # Iterate over a snapshot of children (we'll mutate the block).
    n_added = 0
    for act in list(actuator_block):
        jv = act.get("joint")
        if jv is None:
            continue
        kind = classify_joint_by_name(jv)
        if kind not in eligible_classes:
            continue
        motor_name = f"motor_{jv}"
        if motor_name in existing_names:
            continue
        # Pull a sensible cap.  Prefer the joint's actuatorfrcrange
        # over the position actuator's forcerange — the reference
        # uses the joint's effort limit verbatim.
        rng = joint_actfrcrange.get(jv) or act.get("forcerange") or act.get("actuatorfrcrange")
        new_act = ET.SubElement(actuator_block, "general")
        new_act.set("name", motor_name)
        new_act.set("joint", jv)
        if rng:
            new_act.set("ctrlrange", rng)
            new_act.set("forcerange", rng)
        existing_names.add(motor_name)
        n_added += 1

    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)

    _log_info(
        f"[mjcf-postprocess] {mjcf_path}: added {n_added} raw-torque "
        f"motor_* actuator(s) (sibling to position_*) for body / head / "
        f"arm joints.  File now exposes both PD and torque control "
        f"contracts per controlled joint, matching the reference G2 "
        f"MJCF convention."
    )


def add_init_pose_keyframe(
    mjcf_path: str,
    qpos: Any,
    *,
    name: str = "home",
    logger: Optional[Any] = None,
) -> None:
    """Add a ``<keyframe key="<name>" qpos="..."/>`` block to the dumped
    MJCF so pure MuJoCo can restore the runtime's init joint pose.

    Why this matters
    ----------------
    Newton's MJCF emit doesn't author initial qpos — every joint gets
    ``qpos0 = 0``.  But the live runtime applies the scene yaml's
    ``robot.init_joint_pos`` (typical SP pose folds the arms at
    joint2=-45°, joint4=-75°, joint6=-55°) AFTER the model is built.
    So loading ``robot_runtime.xml`` in pure MuJoCo lands the robot
    with arms extended straight — visually wrong and behaviourally
    different from frame 1.

    This function captures the post-init-pose state into a named
    MuJoCo keyframe so the user can:

      >>> m = mujoco.MjModel.from_xml_path(mjcf_path)
      >>> d = mujoco.MjData(m)
      >>> mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)

    and start stepping from the same state the live engine starts
    from.  ``mj_resetDataKeyframe`` also seeds ``ctrl`` from the
    keyframe's ``ctrl`` field (we leave it zero — the live engine
    drives ctrl per step anyway, this just gets the resting state
    right).

    Parameters
    ----------
    mjcf_path:
        File to mutate in place.  Skipped silently when missing.
    qpos:
        Initial joint position vector — either a numpy array, a
        plain Python list, or anything with ``.tolist()`` /
        iteration.  Length must equal the dumped MJCF's ``nq``.
        For our typical fixed-base, single-DOF-per-joint robots the
        Newton-side ``model.joint_q.numpy()`` array maps directly to
        this layout; pass it verbatim.
    name:
        Keyframe name (default ``"home"``).  Used by callers as
        ``m.key(name).id``.
    logger:
        Optional logger.

    Idempotent — re-running on a file that already has a keyframe of
    the same name overwrites the existing qpos.
    """
    import os

    def _log_info(msg: str) -> None:
        if logger is not None and hasattr(logger, "info"):
            logger.info(msg)
        else:
            print(msg, flush=True)

    def _log_warn(msg: str) -> None:
        if logger is not None and hasattr(logger, "warn"):
            logger.warn(msg)
        else:
            print(f"WARN: {msg}", flush=True)

    if not mjcf_path:
        return
    if not os.path.isfile(mjcf_path):
        _log_warn(f"[mjcf-postprocess] file {mjcf_path!r} does not exist; " f"init-pose keyframe skipped.")
        return

    if hasattr(qpos, "tolist"):
        qpos_list = qpos.tolist()
    else:
        qpos_list = list(qpos)
    if not qpos_list:
        _log_warn(f"[mjcf-postprocess] empty qpos passed to " f"add_init_pose_keyframe; skipping.")
        return

    # Newton's ``model.joint_q`` stores FREE-joint orientation as
    # (x, y, z, w) — Warp native, same convention reset.py's
    # ``_apply_instance_poses`` documents — while a MuJoCo keyframe's
    # qpos expects (w, x, y, z).  Writing the vector verbatim garbles
    # every free body's initial orientation: the benchmark table's
    # 90°-about-X import quat became 90°-about-Z, spawning it in deep
    # penetration so any pure-MuJoCo replay exploded at t=0 (found
    # 2026-07-11 while building physics-fidelity control experiments).
    # Reorder each free joint's quat slice before writing.
    try:
        import mujoco  # noqa: PLC0415

        _m = mujoco.MjModel.from_xml_path(mjcf_path)
        if _m.nq == len(qpos_list):
            n_fixed = 0
            for _j in range(_m.njnt):
                if _m.jnt_type[_j] == mujoco.mjtJoint.mjJNT_FREE:
                    a = int(_m.jnt_qposadr[_j])
                    x, y, z, w = qpos_list[a + 3 : a + 7]
                    qpos_list[a + 3 : a + 7] = [w, x, y, z]
                    n_fixed += 1
            if n_fixed:
                _log_info(
                    f"[mjcf-postprocess] keyframe quat order: reordered "
                    f"{n_fixed} free-joint quat(s) xyzw→wxyz for MuJoCo."
                )
        else:
            _log_warn(
                f"[mjcf-postprocess] keyframe quat reorder skipped: " f"model nq={_m.nq} != len(qpos)={len(qpos_list)}."
            )
    except Exception as exc:  # noqa: BLE001
        _log_warn(f"[mjcf-postprocess] keyframe free-joint quat reorder failed: {exc!r}")

    try:
        tree = ET.parse(mjcf_path)
    except ET.ParseError as exc:
        _log_warn(f"[mjcf-postprocess] could not parse {mjcf_path}: {exc!r}.  " f"File left as-is.")
        return
    root = tree.getroot()
    if root.tag != "mujoco":
        _log_warn(f"[mjcf-postprocess] {mjcf_path} root tag is {root.tag!r}; " f"aborting keyframe insert.")
        return

    keyframe_block = root.find("keyframe")
    if keyframe_block is None:
        keyframe_block = ET.SubElement(root, "keyframe")

    # Replace existing same-named key (idempotent).
    existing = None
    for k in keyframe_block.findall("key"):
        if k.get("name") == name:
            existing = k
            break
    if existing is None:
        existing = ET.SubElement(keyframe_block, "key")
        existing.set("name", name)
    existing.set("qpos", " ".join(f"{q:.10g}" for q in qpos_list))

    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)

    n_nonzero = sum(1 for q in qpos_list if abs(q) > 1e-9)
    _log_info(
        f"[mjcf-postprocess] {mjcf_path}: keyframe {name!r} written "
        f"with {len(qpos_list)} qpos value(s), {n_nonzero} non-zero.  "
        f"Pure MuJoCo can restore via ``mj_resetDataKeyframe(m, d, "
        f"m.key({name!r}).id)``."
    )


def augment_mjcf_gripper_from_params(
    mjcf_path: str,
    *,
    gripper_params: Any,
    logger: Optional[Any] = None,
) -> None:
    """Mirror ``MuJoCoWarpAdapter._apply_mimic_eq_solref`` +
    master-actuator ``ctrlrange`` / ``forcerange`` authoring into the
    dumped MJCF.

    The live mjw_model already has the YAML's values pushed into it by
    the adapter; this function brings the dumped file into the same
    state so pure MuJoCo reproduces the live runtime exactly.

    Three edits per call:

    1. **Mimic equality solref / solimp** — for every ``<equality>
       <joint joint1="..." joint2="..." polycoef=".../>"`` whose
       ``joint1`` name contains ``gripper`` (matches the adapter's
       mimic_jids gating), write
       ``solref="<gripper_params.mimic_eq_solref>"`` and
       ``solimp="<gripper_params.mimic_eq_solimp>"``.

    2. **Master position actuator ``ctrlrange``** — when
       ``master_ctrl_range_from_joint_limit`` is True, copy the
       master joint's ``range="lo hi"`` attribute onto the
       corresponding ``<general>`` actuator as ``ctrlrange="lo hi"``.

    3. **Master position actuator ``forcerange``** — when
       ``master_max_force`` > 0, author ``forcerange="-max +max"``
       on the master actuator.  When 0 we leave the existing emit
       alone (lets URDF effort propagate through Newton's
       ``actfrcrange``).

    Parameters
    ----------
    mjcf_path:
        File to mutate in place.  Skipped silently when missing.
    gripper_params:
        ``GripperDriveParams`` instance — typically
        ``physics_params.drive_gripper`` from the loaded YAML.
    logger:
        Optional logger.

    Idempotent.  Re-running just overwrites the same attributes.
    """
    import os

    def _log_info(msg: str) -> None:
        if logger is not None and hasattr(logger, "info"):
            logger.info(msg)
        else:
            print(msg, flush=True)

    def _log_warn(msg: str) -> None:
        if logger is not None and hasattr(logger, "warn"):
            logger.warn(msg)
        else:
            print(f"WARN: {msg}", flush=True)

    if not mjcf_path:
        return
    if not os.path.isfile(mjcf_path):
        _log_warn(f"[mjcf-postprocess] file {mjcf_path!r} does not exist; " f"gripper params apply skipped.")
        return
    if gripper_params is None:
        return

    try:
        tree = ET.parse(mjcf_path)
    except ET.ParseError as exc:
        _log_warn(f"[mjcf-postprocess] could not parse {mjcf_path}: {exc!r}.  " f"File left as-is.")
        return
    root = tree.getroot()
    if root.tag != "mujoco":
        return

    # ----- (1) Mimic equality solref / solimp -----------------------
    solref_str = " ".join(f"{float(v):.6g}" for v in gripper_params.mimic_eq_solref)
    solimp_str = " ".join(f"{float(v):.6g}" for v in gripper_params.mimic_eq_solimp)
    eq_block = root.find("equality")
    n_eq_patched = 0
    if eq_block is not None:
        for eq in eq_block.findall("joint"):
            j1 = eq.get("joint1", "")
            if "gripper" not in j1:
                continue
            eq.set("solref", solref_str)
            eq.set("solimp", solimp_str)
            n_eq_patched += 1

    # ----- (2) + (3) Master ctrlrange / forcerange ------------------
    # Walk joints to build a name → range map.
    joint_range: dict[str, str] = {}
    for j in root.iter("joint"):
        nm = j.get("name")
        rg = j.get("range")
        if nm is not None and rg is not None:
            joint_range[nm] = rg

    actuator_block = root.find("actuator")
    n_ctrl_set = 0
    n_force_set = 0
    if actuator_block is not None:
        for act in actuator_block:
            jv = act.get("joint", "")
            # Only the gripper master gets the per-YAML treatment.
            # We identify it by name: ``inner_joint1`` is the
            # universal master suffix across Robotiq-like grippers in
            # this codebase.  Followers stay alone.
            if "gripper" not in jv or "inner_joint1" not in jv:
                continue
            # Skip the raw-torque sibling (motor_*) — only edit the
            # position_* PD actuator.
            cur_name = act.get("name", "")
            if cur_name.startswith("motor_"):
                continue
            if gripper_params.master_ctrl_range_from_joint_limit:
                rg = joint_range.get(jv)
                if rg:
                    act.set("ctrlrange", rg)
                    act.set("ctrllimited", "true")
                    n_ctrl_set += 1
            if gripper_params.master_max_force > 0.0:
                mx = float(gripper_params.master_max_force)
                act.set("forcerange", f"{-mx:.6g} {mx:.6g}")
                act.set("forcelimited", "true")
                n_force_set += 1

    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)

    _log_info(
        f"[mjcf-postprocess] {mjcf_path}: gripper params applied — "
        f"{n_eq_patched} equality(s) given solref={solref_str!r} "
        f"solimp={solimp_str!r}; {n_ctrl_set} master ctrlrange / "
        f"{n_force_set} master forcerange authored.  Sourced from "
        f"physics_params.yaml::usd_drive_api.gripper."
    )


# Map an ``imu.sensors`` entry to the MJCF sensor element it emits per site.
# ``framequat`` reports the site's world orientation (objtype/objname form);
# ``gyro`` / ``accelerometer`` read the site frame (site= form).
_IMU_SENSOR_SPEC = {
    "framequat": ("framequat", "imu_quat_", "objname"),
    "gyro": ("gyro", "imu_gyro_", "site"),
    "accelerometer": ("accelerometer", "imu_accel_", "site"),
}


def add_imu_sites_and_sensors(
    mjcf_path: str,
    imu_cfg: Any,
    *,
    logger: Optional[Any] = None,
) -> None:
    """Author IMU ``<site>`` + ``<sensor>`` blocks into the dumped MJCF.

    Newton's ``_convert_to_mjc`` emits neither sites nor a ``<sensor>``
    section, so the raw ``robot_runtime.xml`` has no IMU.  This step adds
    them from the scene yaml's ``imu:`` block so the file, when loaded in
    pure MuJoCo, exposes the same IMU sensors the reference G2.chassis MJCF
    does (front / back Livox IMUs + chassis IMU).

    ``imu_cfg`` shape (from the scene yaml)::

        parent_body: base_link            # body the sites attach to
        sites:
          - {name: imu_site_front, pos: [...], quat: [...], size: 0.01}
          - ...
        sensors: [framequat, gyro, accelerometer]   # emitted per site

    Each site name of the form ``imu_site_<tag>`` yields sensors
    ``imu_quat_<tag>`` / ``imu_gyro_<tag>`` / ``imu_accel_<tag>`` (falls
    back to the full site name as the tag when the prefix is absent).

    Idempotent — sites / sensors whose ``name`` already exists are skipped,
    so a re-run (or a dump that already carried them) never duplicates.
    No-op when ``imu_cfg`` is falsy or declares no sites.
    """
    import os

    def _log_info(msg: str) -> None:
        if logger is not None and hasattr(logger, "info"):
            logger.info(msg)
        else:
            print(msg, flush=True)

    def _log_warn(msg: str) -> None:
        if logger is not None and hasattr(logger, "warn"):
            logger.warn(msg)
        else:
            print(f"WARN: {msg}", flush=True)

    if not mjcf_path or not imu_cfg:
        return
    sites = imu_cfg.get("sites") or []
    if not sites:
        return
    if not os.path.isfile(mjcf_path):
        _log_warn(f"[mjcf-postprocess] file {mjcf_path!r} does not exist; " f"IMU injection skipped.")
        return

    parent_body = imu_cfg.get("parent_body") or "base_link"
    sensor_kinds = imu_cfg.get("sensors") or ["framequat", "gyro", "accelerometer"]

    try:
        tree = ET.parse(mjcf_path)
    except ET.ParseError as exc:
        _log_warn(f"[mjcf-postprocess] could not parse {mjcf_path}: {exc!r}.  " f"File left as-is.")
        return
    root = tree.getroot()
    if root.tag != "mujoco":
        _log_warn(f"[mjcf-postprocess] {mjcf_path} root tag is {root.tag!r}; " f"aborting IMU insert.")
        return

    # Locate the parent body anywhere in the tree (ET.find only searches
    # direct children; the robot bodies are nested deep under worldbody).
    target_body = None
    for b in root.iter("body"):
        if b.get("name") == parent_body:
            target_body = b
            break
    if target_body is None:
        _log_warn(f"[mjcf-postprocess] parent_body {parent_body!r} not found in " f"{mjcf_path}; IMU sites skipped.")
        return

    # Existing names (MuJoCo requires site / sensor names unique) for the
    # idempotency guard.
    existing_sites = {s.get("name") for s in root.iter("site")}
    existing_sensor_names = set()
    sensor_block = root.find("sensor")
    if sensor_block is not None:
        for s in list(sensor_block):
            existing_sensor_names.add(s.get("name"))

    def _fmt(vals) -> str:
        return " ".join(f"{float(v):g}" for v in vals)

    n_sites = 0
    for site in sites:
        name = site.get("name")
        if not name or name in existing_sites:
            continue
        el = ET.SubElement(target_body, "site")
        el.set("name", name)
        if site.get("pos") is not None:
            el.set("pos", _fmt(site["pos"]))
        if site.get("quat") is not None:
            el.set("quat", _fmt(site["quat"]))
        el.set("size", f"{float(site.get('size', 0.01)):g}")
        existing_sites.add(name)
        n_sites += 1

    # Author the sensors.  MuJoCo puts ``<sensor>`` at the top level; create
    # it if the dump has none (Newton never writes one).
    if sensor_block is None:
        sensor_block = ET.SubElement(root, "sensor")

    n_sensors = 0
    for site in sites:
        site_name = site.get("name")
        if not site_name:
            continue
        tag = site_name[len("imu_site_") :] if site_name.startswith("imu_site_") else site_name
        for kind in sensor_kinds:
            spec = _IMU_SENSOR_SPEC.get(kind)
            if spec is None:
                _log_warn(f"[mjcf-postprocess] unknown imu sensor kind {kind!r}; skipped.")
                continue
            elem_tag, name_prefix, ref_attr = spec
            sname = f"{name_prefix}{tag}"
            if sname in existing_sensor_names:
                continue
            sensor = ET.SubElement(sensor_block, elem_tag)
            sensor.set("name", sname)
            if ref_attr == "objname":
                sensor.set("objtype", "site")
                sensor.set("objname", site_name)
            else:
                sensor.set(ref_attr, site_name)
            existing_sensor_names.add(sname)
            n_sensors += 1

    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)

    _log_info(
        f"[mjcf-postprocess] {mjcf_path}: injected {n_sites} IMU site(s) under "
        f"{parent_body!r} + {n_sensors} sensor(s)."
    )


def apply_mjcf_postprocess_pipeline(
    *,
    mjcf_path: str,
    sim_substeps: int,
    physics_hz: float,
    physics_params: Any = None,
    qpos: Any = None,
    imu_cfg: Any = None,
    logger: Optional[Any] = None,
) -> None:
    """Run every standard MJCF post-process step on ``mjcf_path``,
    each in its own ``try / except`` so one step's failure does NOT
    block the rest.

    Why this exists
    ---------------
    A single outer ``try:`` wrapping the whole post-process chain
    silently swallows later steps when an intermediate one raises —
    e.g. the equality text-patch (step 1) lands but the
    naming-normalisation + cadence-numerics + ctrlrange (steps 2-5)
    are skipped.  This helper isolates each step so a failure logs a
    warning with the step name and exception text, then the next
    step runs.  Partial post-processing is far better than no
    post-processing, and the warning surfaces the broken step so it
    can be fixed deliberately.

    Both engine paths (``isaac_newton`` wrapper and newton-standalone
    ``MuJoCoWarpAdapter``) call this helper.  The wrapper additionally
    mutates the live ``mjw_model`` for equality stiffening BEFORE
    calling this helper — that's wrapper-specific because it requires
    the live solver handle.

    Step order
    ----------
    1. ``augment_mjcf_timing``       — writes ``<option timestep>``
       and the ``<custom>`` cadence block (physics_hz / sim_substeps
       / sim_dt) so pure-MuJoCo tools can replay the runtime cadence.
    2. ``normalize_mjcf_names``      — strips prim-path prefixes from
       bodies / joints / actuators; renames ``position:<joint>``
       actuators to ``position_<short_joint>``.
    3. ``add_raw_torque_motor_actuators`` — adds sibling
       ``motor_<joint>`` raw-torque actuators on body/head/arm
       joints, matching the reference G2 MJCF dual-actuator topology.
    4. ``augment_mjcf_gripper_from_params`` (skipped when
       ``physics_params=None``) — authors gripper-master
       ``ctrlrange`` / ``forcerange`` from URDF effort + YAML, and
       rewrites mimic equality ``solref`` / ``solimp`` per
       ``physics_params.yaml::usd_drive_api.gripper``.
    5. ``add_init_pose_keyframe`` (skipped when ``qpos=None``) —
       writes a ``<keyframe name="home" qpos="...">`` so pure
       MuJoCo can restore the runtime's starting state via
       ``mj_resetDataKeyframe``.  Only the newton-standalone path
       provides a ``qpos`` (it runs after ``_apply_init_joint_pos``);
       the wrapper has no equivalent lifecycle hook, so it passes
       ``qpos=None``.

    Parameters
    ----------
    mjcf_path:
        Path to the freshly-dumped MJCF.  Must exist; individual
        steps tolerate missing files but the pipeline doesn't
        re-verify per step.
    sim_substeps, physics_hz:
        Passed verbatim to ``augment_mjcf_timing``.
    physics_params:
        ``PhysicsParams`` from the loaded YAML.  Skipped when
        ``None`` (test paths that don't load the YAML).
    qpos:
        Init-pose joint positions (numpy array or list).  Skipped
        when ``None``.
    logger:
        Optional logger; falls back to ``print``.
    """

    def _safe(step_name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            if logger is not None and hasattr(logger, "warn"):
                logger.warn(f"[mjcf-postprocess] step {step_name!r} failed " f"(continuing pipeline): {exc!r}")
            else:
                print(
                    f"WARN: [mjcf-postprocess] step {step_name!r} failed " f"(continuing pipeline): {exc!r}",
                    flush=True,
                )

    _safe(
        "augment_mjcf_timing",
        lambda: augment_mjcf_timing(
            mjcf_path=mjcf_path,
            sim_substeps=int(sim_substeps) if sim_substeps else 1,
            physics_hz=float(physics_hz),
            logger=logger,
        ),
    )
    _safe(
        "normalize_mjcf_names",
        lambda: normalize_mjcf_names(mjcf_path=mjcf_path, logger=logger),
    )
    _safe(
        "add_raw_torque_motor_actuators",
        lambda: add_raw_torque_motor_actuators(mjcf_path=mjcf_path, logger=logger),
    )
    if physics_params is not None:
        _safe(
            "augment_mjcf_gripper_from_params",
            lambda: augment_mjcf_gripper_from_params(
                mjcf_path=mjcf_path,
                gripper_params=physics_params.drive_gripper,
                logger=logger,
            ),
        )
    if qpos is not None:
        _safe(
            "add_init_pose_keyframe",
            lambda: add_init_pose_keyframe(mjcf_path=mjcf_path, qpos=qpos, logger=logger),
        )
    if imu_cfg:
        _safe(
            "add_imu_sites_and_sensors",
            lambda: add_imu_sites_and_sensors(mjcf_path=mjcf_path, imu_cfg=imu_cfg, logger=logger),
        )
    # Defensive last step.  MuJoCo refuses to load an MJCF with two
    # actuators sharing a ``name=``.  In rare cases (stale launcher
    # process state, old code cached in memory, an upstream step bug)
    # one of the steps above can produce duplicates.  Strip them at the
    # tail so the worst case is a loud WARN, not an unloadable file.
    _safe(
        "dedupe_actuator_names",
        lambda: dedupe_actuator_names(mjcf_path=mjcf_path, logger=logger),
    )


def dedupe_actuator_names(mjcf_path: str, *, logger: Optional[Any] = None) -> None:
    """Defensive last step in the pipeline: enforce unique actuator
    names within the ``<actuator>`` block.

    Why
    ---
    MuJoCo rejects an MJCF whose ``<actuator>`` block has two
    elements with the same ``name=`` — fails to load with
    ``XML Error: Error: repeated name '<n>' in actuator``.

    Duplicates can appear when:

    * The pipeline runs partially on an old file before
      ``SolverMuJoCo.__init__`` re-overwrites it (rare but observed
      with stale launcher process state).
    * A stale code version is still loaded in memory and produces
      actuators that another step also adds.
    * A post-process step has a bug.

    Rather than try to detect every root cause, this deduplicator
    runs as the FINAL pipeline step so the worst case is
    "post-process complete but file unloadable" → "post-process
    complete and file loads".  When duplicates are found, they are
    logged loudly so the operator can investigate the root cause.

    Strategy
    --------
    Keep the FIRST occurrence of each name.  Newton's original
    emit produces actuators with full ``gainprm``/``biasprm`` (the
    behavioural actuator); duplicates from a follow-up step
    typically carry partial attributes (only ``forcerange`` or
    similar).  Keeping the first preserves the runtime contract.
    Anonymous actuators (no ``name=``) are untouched.
    """
    import os

    def _log_info(msg: str) -> None:
        if logger is not None and hasattr(logger, "info"):
            logger.info(msg)
        else:
            print(msg, flush=True)

    def _log_warn(msg: str) -> None:
        if logger is not None and hasattr(logger, "warn"):
            logger.warn(msg)
        else:
            print(f"WARN: {msg}", flush=True)

    if not mjcf_path:
        return
    if not os.path.isfile(mjcf_path):
        return
    try:
        tree = ET.parse(mjcf_path)
    except ET.ParseError as exc:
        _log_warn(f"[mjcf-postprocess] dedupe: parse failed ({exc!r}); leaving file as-is.")
        return
    root = tree.getroot()
    actuator_block = root.find("actuator")
    if actuator_block is None:
        return

    seen_names: set = set()
    to_remove: list = []
    duplicate_names: list = []
    for act in list(actuator_block):
        nm = act.get("name")
        if nm is None:
            continue
        if nm in seen_names:
            to_remove.append(act)
            duplicate_names.append(nm)
            continue
        seen_names.add(nm)

    if not to_remove:
        return

    for act in to_remove:
        actuator_block.remove(act)

    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)

    # Loud warn — duplicates are always a symptom of something
    # going wrong upstream, even if we successfully clean them up.
    sample = duplicate_names[:5]
    suffix = "" if len(duplicate_names) <= 5 else f" (… and {len(duplicate_names) - 5} more)"
    _log_warn(
        f"[mjcf-postprocess] dedupe_actuator_names: dropped "
        f"{len(to_remove)} duplicate <general> element(s) from "
        f"{mjcf_path}.  First occurrence of each name was kept "
        f"(usually Newton's emit with gainprm/biasprm).  Sample names: "
        f"{sample!r}{suffix}.  This is a SAFETY NET — investigate why "
        f"the pipeline produced duplicates upstream (stale launcher "
        f"process? old code cached in memory? buggy step?)."
    )
