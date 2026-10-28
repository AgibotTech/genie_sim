# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Physics parameter loader for genie_sim_engine.

The node passes ``physics_params_file`` (a path to a YAML file) at construction
time. This module returns a fully-populated :class:`PhysicsParams` dataclass.
On any failure (missing file, parse error, missing key) it falls back to the
built-in defaults defined below.

In addition to ``PhysicsParams`` (user-tuning, sourced from
``physics_params.yaml``), this module also defines :class:`JointInitSpec` —
the typed shape used to push **scene-specific** initial joint poses
into the articulation. The pose dict is sourced from the scene YAML's
``robot.init_joint_pos`` block at launch time and forwarded as a
JSON-encoded node parameter (``init_joint_pos_json``); see
``parse_init_joint_pos``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class DriveParams:
    """Per-joint USD ``DriveAPI`` + passive-joint authoring values.

    Two unrelated families of fields live here together because both
    flow through the same YAML block (``usd_drive_api.<class>:``):

    Actuator / DriveAPI (Layer 1 PD on the joint actuator):
      * ``stiffness`` — kp written to ``UsdPhysics.DriveAPI.stiffness``
      * ``damping``   — kd written to ``UsdPhysics.DriveAPI.damping``
      * ``max_force`` — written to ``UsdPhysics.DriveAPI.maxForce``

    Passive joint (no actuator involvement — physics dissipation
    intrinsic to the joint, applied every step regardless of any
    drive):
      * ``dof_damping``     — joint viscous damping (N·m·s/rad);
                              written to ``mjc:damping`` on USD
                              prims, lands as MuJoCo ``dof_damping``
                              after Newton's converter.  Default 0
                              (kept off so scenes that don't author
                              passive damping aren't unexpectedly
                              stiffer).
      * ``dof_frictionloss`` — Coulomb-style joint friction (N·m);
                               written to ``mjc:frictionloss`` on
                               USD prims, lands as MuJoCo
                               ``dof_frictionloss``.  Default 0.

    The passive-joint fields exist on every class because the
    reference G2 MJCF authors per-joint damping/frictionloss values
    that are well above zero (arm shoulder: damping=3.0,
    frictionloss=1.0; arm mid: 1.5 / 0.5; arm wrist: 0.8 / 0.3).
    Without these the dumped MJCF has 0 across the board and the
    simulation feels less settled than the reference under the same
    control — verified empirically via the static / dynamic XML diff
    tools against the reference G2 MJCF.
    """

    stiffness: float
    damping: float
    max_force: float
    free_limits: bool = False
    # Passive joint damping (mjc:damping) — see class docstring.
    dof_damping: float = 0.0
    # Joint frictionloss (mjc:frictionloss) — see class docstring.
    dof_frictionloss: float = 0.0
    # Reflected rotor inertia (N·m·s²/rad).  Lands as MuJoCo
    # ``dof_armature`` via Newton's ``model.joint_armature`` array.
    # Default 0 = leave whatever the URDF importer wrote (usually 0
    # for non-gripper joints).  Per-class values matter for mjwarp
    # because the gripper-only authoring path skips body/head/arm
    # and the dumped MJCF then omits the ``armature`` attribute,
    # which makes the model under-damped in pure-MuJoCo replay and
    # less stable at larger outer timesteps.  No angular quirk on
    # this field — author the physical value verbatim.
    armature: float = 0.0


@dataclass(frozen=True)
class ArticulationGains:
    kp: float
    kd: float
    max_effort: float


@dataclass(frozen=True)
class GripperDriveParams:
    """URDF-route gripper tuning — single source of truth.

    This dataclass is the **only** place gripper tuning lives.  The
    mjwarp adapter
    (``engine/newton/adapters/mujoco_warp.py:MuJoCoWarpAdapter``)
    reads every gripper field from here.  The isaac_newton wrapper
    (``kit/stage.py:_configure_drives`` plus
    ``kit/isaac_newton.py:_stiffen_*_mimic_equalities``) reads the
    same fields, so PhysX and mjwarp see the same tuning.

    Applied only on the URDF route (``from_urdf=True``); hand-authored
    USDs are left as-authored.  The master joint is identified by
    ``HasAPI(DriveAPI)`` (Isaac path) or by name classification
    (``classify_joint_by_name`` returning ``JK_GRIPPER`` and the joint
    matching the master via ``mimic_followers`` set membership;
    standalone path).

    Field families (call out who reads what):

    Master actuator
    ---------------
    Used by BOTH the PhysX path (DriveAPI on the joint USD prim,
    pushed through ``UsdPhysics.DriveAPI``) and the mjwarp path
    (``model.joint_target_ke/kd`` → ``actuator gainprm/biasprm``
    in the dumped MJCF).

      * ``master_stiffness`` / ``master_damping`` — kp / kd of the
        master position actuator.  Reference G2 MJCF uses (5, 0);
        a stiffer master fights the equality constraint when the
        equality is left at MuJoCo's soft default solref.
      * ``master_max_force`` — actuator effort cap, MJCF
        ``forcerange="-max +max"``.  ``0`` means "preserve the
        URDF ``<limit effort>`` value" (recommended — matches
        hardware spec).  Set non-zero to override.
      * ``master_ctrl_range_from_joint_limit`` — when True, the
        master's actuator ``ctrlrange`` is authored from the
        joint's ``range`` attribute, clamping policy ctrl writes
        to within the hardware limits.  Default True.

    Joint passive physics (applied to master + every follower)
    ----------------------------------------------------------
    Both paths consume; mjwarp reads via ``model.mujoco
    .dof_passive_damping`` / ``model.joint_friction`` /
    ``model.joint_armature``; PhysX reads via ``physxJoint:armature``
    + ``physxJoint:jointDamping`` schemas.

      * ``armature`` — per-joint rotor inertia (N·m·s²/rad).
        Improves solver stability without changing the
        dynamics.  Reference: 0.001.
      * ``dof_damping`` — passive viscous damping (N·m·s/rad).
        Reference: 0.05.  Critical for damping out follower
        oscillations under a soft mimic equality.
      * ``dof_frictionloss`` — Coulomb-style joint friction
        (N·m).  Reference: 0.01.  Tames residual creep.

    Mimic equality solver tuning (mjwarp ONLY)
    ------------------------------------------
    PhysX uses ``PhysxMimicJointAPI`` (see ``mimic_natural_frequency``
    / ``mimic_damping_ratio`` below) and ignores these.  Mjwarp uses
    these to set ``mjw_model.eq_solref`` / ``eq_solimp`` for every
    gripper mimic equality.

      * ``mimic_eq_solref`` — equality solver impedance time-constant
        / damping-ratio pair.  MuJoCo's default ``(0.02, 1.0)`` is a
        soft spring with 20 ms time constant — works but lets
        followers swing under aggressive arm motion.  For stiff
        direct mode set negative values: ``(-stiffness, -damping)``
        e.g. ``(-10000, -100)``.  See MuJoCo docs on ``solref``.
      * ``mimic_eq_solimp`` — equality impedance schedule
        ``(d_min, d_max, width, midpoint, power)``.

    PhysX mimic API (Isaac path only — ignored by mjwarp)
    -----------------------------------------------------
      * ``mimic_natural_frequency`` / ``mimic_damping_ratio`` —
        ``PhysxMimicJoint:<axis>:naturalFrequency`` /
        ``dampingRatio``.  Both MUST be 0.0 for a rigid kinematic
        constraint on the PhysX side.  Non-zero turns the
        constraint into a soft spring/damper.
    """

    master_stiffness: float
    master_damping: float
    armature: float
    mimic_natural_frequency: float
    mimic_damping_ratio: float
    # New mjwarp-relevant fields with defaults matching the prior
    # adapter constants.  These propagate through ``_gripper()``
    # (the YAML loader merger) so a YAML without these keys still
    # loads at the documented defaults.
    master_max_force: float = 0.0
    master_ctrl_range_from_joint_limit: bool = True
    dof_damping: float = 0.05
    dof_frictionloss: float = 0.01
    mimic_eq_solref: tuple = (0.02, 1.0)
    mimic_eq_solimp: tuple = (0.9, 0.95, 0.001, 0.5, 2.0)


@dataclass(frozen=True)
class GripperNaming:
    """Joint/link name patterns used to find gripper parts when the USD
    carries no ``newton:isGripperPad`` / ``newton:isLoopCoupler``
    annotations.

    Annotated assets never consult these.  The defaults describe the G2
    omnipicker; a differently-named gripper overrides them in
    ``physics_params.yaml`` under ``gripper_naming:`` instead of touching
    code.  Every value is a Python regular expression.
    """

    #: Marks a joint/link as belonging to a gripper at all.
    family: str = r"gripper|finger|jaw|knuckle"
    #: Passive four-bar joints closing the finger loop.
    loop_coupler: str = r"gripper_[lr]_(inner|outer)_joint[034]$"
    #: Driven joints, consulted only when no mimic map was parsed.
    driven: str = r"gripper_[lr]_(inner|outer)_joint1$"
    #: Master joint, consulted only when no mimic map was parsed.
    master: str = r"gripper_[lr]_outer_joint1$"


@dataclass(frozen=True)
class SceneCollidersParams:
    """SCENE collider geometry: CoACD decomposition + the shell box cover.

    Scene props (rooms, shelves, sorter boxes) arrive as single meshes that
    mjwarp would convex-hull into solid blocks — a shell becomes a brick that
    swallows whatever should drop through it. Two passes fix that: CoACD
    decomposition, and a voxel box cover for shell-like colliders.

    These are SCENE-side; the gripper's own pad decomposition lives on
    :class:`GripperContactParams`, and scene-wide solver knobs on
    :class:`SolverParams`. Do not mix them.

    Shipped values follow the same rule as ``GripperContactParams``: every
    capability here is off, so an untouched profile behaves like an engine
    without these passes. Tuned numbers are config, not code defaults —
    a profile that wants them turns them on by name.

    ``box_cover_pitch`` / ``box_cover_max_boxes`` are 0 meaning "use the
    implementation's own measured constants" — those two carry calibration
    that belongs with the algorithm, not with a per-scene profile.
    """

    decomposition: bool = False
    decomposition_threshold: float = 0.05
    decomposition_max_hulls: int = 32
    decomposition_max_solidity: float = 0.5
    # Scan-phase gates (stage.py picks WHICH colliders the pass above will
    # split). ``min_void`` is the real "does this shell swallow things"
    # criterion; ``max_verts`` only guards against pathological meshes —
    # the largest real collider (a shopping trolley) is 101682 vertices.
    decomposition_min_void: float = 0.001
    decomposition_max_verts: int = 200000
    # Path substrings to skip BEFORE the solidity/void measurement runs. A
    # plain four-leg table measures like a shopping cart (big hull, little
    # mesh, the space between the legs reads as void), but unlike a cart it is
    # the surface objects rest on -- and CoACD's slightly-inflated pieces left
    # a seam that let a block sink ~36-54mm through the table top.
    # ``None`` (the default) means "use stage.py's own
    # ``_SCENE_DECOMP_EXCLUDE_NAME_SUBSTRINGS``"; an explicit empty list
    # clears the exclusion entirely.
    decomposition_exclude_names: Optional[tuple] = None
    box_cover: bool = False
    box_cover_pitch: float = 0.0
    box_cover_max_boxes: int = 0
    # Demote free-floating decor under ``/Background`` that no joint drives
    # to a static collider. Off (shipped) keeps it dynamic, which is the
    # behaviour of an engine without this pass — each such prop then costs
    # 6 DOF and a contact set.
    background_decor_static: bool = False


@dataclass(frozen=True)
class GripperContactParams:
    """Gripper-specific contact / grasp parameters.

    Every field here narrows to the gripper: forcerange cap on the master
    actuator, and CoACD decomposition of the gripper pads. Scene-wide
    mjwarp solver knobs live on :class:`SolverParams` instead — do not
    add global settings here.

    All fields are NEW capabilities with no upstream counterpart, so the
    shipped value of every one of them is "off" — with them off, these code
    paths do not execute at all and behaviour matches an engine without these
    changes. Whoever wants them turns them on explicitly in their own profile
    file.

    Shipped values never carry tuning results: a tuned number is config.
    Writing one as a code default changes physics for every user who has not
    read this file, without them ever knowing.
    """

    # forcerange cap on the gripper master actuator (N*m). <=0 (shipped) leaves
    # forcerange untouched, keeping the effort limit already in the MJCF/URDF.
    # Note that ``drive_gripper.master_max_force`` > 0 takes priority over this
    # one — that is the explicit override in URDF terms.
    gripper_master_force_cap: float = 0.0
    # CoACD convex decomposition of the gripper pads. Off (shipped) = pads keep
    # their original collision geometry, i.e. a single hull, and contact is a
    # point; on = decomposed into several hulls and contact becomes a face.
    pad_decomposition: bool = False
    # The next two only matter while ``pad_decomposition`` is on.
    pad_decomposition_threshold: float = 0.01
    pad_decomposition_max_hulls: int = 16


@dataclass(frozen=True)
class ArmContactParams:
    """Arm-specific contact / actuator-force parameters.

    Every field here narrows to the arm — same one-purpose-per-class rule as
    :class:`GripperContactParams` (gripper) and :class:`SolverParams`
    (scene-wide): do not add gripper or scene-wide settings here.

    This is a POST-HOC cap on the compiled model, independent of the
    per-joint-class drive table (``art_arm_shoulder`` / ``art_arm_mid`` /
    ``art_arm_wrist``): that table's ``max_effort`` can only RAISE a joint's
    effort limit above whatever the URDF authored, never lower it (see
    ``MuJoCoWarpAdapter``'s per-class gain lookup) — it exists to backfill a
    class that needs MORE torque than a stingy URDF gives it, not to cap one
    that has too much. ``arm_max_torque`` here is the opposite: it forces
    ``actuator_forcerange`` down to this value on every arm actuator,
    regardless of what the URDF or the class table authored.

    All fields are NEW capabilities with no upstream counterpart, so the
    shipped value is "off" — with it off, this code path does not execute at
    all and behaviour matches an engine without this change.
    """

    # forcerange cap on the arm's POSITION-servo actuators (N*m). <=0
    # (shipped) leaves forcerange untouched, keeping whatever effort limit
    # the compiled model already carries (which may be unlimited — MuJoCo's
    # [0,0] forcerange means "no limit", not "zero"). Caps all arm joints
    # uniformly (shoulder/mid/wrist/generic JK_ARM) — the URDF's own
    # per-joint effort limits already carry any finer-grained tiering.
    arm_max_torque: float = 0.0


@dataclass(frozen=True)
class SolverParams:
    """Scene-wide physics-behaviour toggles applied by the mjwarp adapter.

    Every field here is scene-wide — it changes solver behaviour for all
    contacts / all bodies, not for any one part of the robot. Add new
    scene-wide toggles here as they get exposed (mjwarp ``mjOption`` fields
    like cone type / disable flags / iterations mode / warmstart, or
    adapter-level pre/post-solve knobs).

    Shipped values match upstream behaviour so an engine that never reads
    its ``solver:`` block matches an engine without these changes.

    Fields are set from ``config/physics_params.yaml`` (or a sibling
    profile) and from nowhere else. There is deliberately no env-var
    back door: a stale ``RLINF_*`` in a shell would change simulation
    results with no trace of it in the config the run was launched with.
    Tune by editing the yaml — ``--symlink-install`` dev workspaces pick
    up an edit without a rebuild.
    """

    # mjwarp ``opt.impratio`` + elliptic friction cone. >0 turns on the
    # elliptic cone (real static friction); 0 (shipped) leaves the friction
    # cone setting alone, keeping the solver's own pyramidal cone.
    impratio: float = 0.0
    # SolverMuJoCo constraint-solver iteration count. 0 (shipped) leaves
    # SolverMuJoCo's own default alone, which is MuJoCo's 100.
    #
    # This exists because MuJoCo's rigid constraint solver must NOT share an
    # iteration count with the particle solver. The launcher's
    # ``physics_solver_iterations`` is tuned for the particle side -- FEM tet
    # scenes run 4 because substeps=24 carries the stiffness -- and the combo
    # solvers (``mjxpbd``, ``mjvbd``) drive BOTH from one adapter. Forwarding
    # that 4 to SolverMuJoCo ran the rigid solve at 4 against a default of 100,
    # and a frictional contact block cannot converge in 4: the grasped object
    # creeps out of the jaws, which reads as a friction problem and sends you
    # tuning friction coefficients that were never wrong.
    #
    # Read by both the single-env and multi-env mjwarp adapters, from here and
    # from nowhere else -- see geniesim_ros/AGENTS.md, "one knob, one route".
    mjc_iterations: int = 0
    # SolverMuJoCo write-back cadence: how many substeps to wait before
    # writing mjw state back into Newton state. 1 (shipped) = every substep,
    # upstream behaviour. Setting to 0 removes the microscopic tumbling of
    # resting free bodies (the per-substep round trip accumulates numerical
    # error into visible jitter) but ONLY holds with its companion:
    #   * an explicit push after the engine writes state
    #     (``sync_state_into_solver`` in the mjwarp adapter).
    # The two go together; do NOT change one alone.
    update_data_interval: int = 1


@dataclass(frozen=True)
class PhysicsParams:
    drive_chassis_drive_joint: DriveParams
    drive_chassis_steer_joint: DriveParams
    drive_default_revolute: DriveParams
    drive_default_prismatic: DriveParams
    # Articulated SCENE objects (door hinges, drawer slides).
    # Classified by body OWNERSHIP (parent + child both outside the robot
    # prefix), NOT by joint name; routed to ``JK_SCENE_ARTICULATION`` in
    # the mjwarp adapter. The actuator-gain fields on this block are NOT
    # read (scene joints get zero PD — they are passive); only
    # ``dof_damping`` / ``dof_frictionloss`` are applied, so a passive
    # hinge settles under damping instead of swinging frictionlessly.
    # See ``MuJoCoWarpAdapter._per_class_passive_joint``.
    drive_scene_articulation: DriveParams
    # Per-sub-class passive-joint authoring.  The actuator gains
    # (``stiffness`` / ``damping`` / ``max_force``) on these blocks
    # are NOT applied — they fall back to ``drive_default_revolute``
    # for the adapter's PD lookup.  Only ``dof_damping`` and
    # ``dof_frictionloss`` are read per sub-class, matching the
    # reference G2 MJCF's per-joint-position profile (shoulder
    # 3.0/1.0, mid 1.5/0.5, wrist 0.8/0.3, body/head 0/0).  See
    # ``MuJoCoWarpAdapter._per_class_passive_joint`` for the route.
    drive_body: DriveParams
    drive_head: DriveParams
    drive_arm_shoulder: DriveParams
    drive_arm_mid: DriveParams
    drive_arm_wrist: DriveParams
    drive_gripper: GripperDriveParams
    art_default: ArticulationGains
    art_chassis_drive: ArticulationGains
    art_chassis_steer: ArticulationGains
    # Per-sub-class articulation-PD gains (Layer 2 actuator authority,
    # written via ArticulationView.set_dof_stiffnesses /
    # set_dof_dampings / set_dof_max_forces).  The reference G2 MJCF
    # tiers these by joint
    # inertia: heavy proximal joints carry high kp (40000) with
    # commensurate kd (600); light wrist joints run far lower
    # (8000 / 100); body uses 1e5 / 1e3; head uses 1e3 / 20.
    # Falling back to ``art_default`` (5e4 / 5e3) across body / head
    # / arm gives > 2× divergence on every joint vs the reference,
    # which the static-diff tool surfaces as suspicious flags.
    # Each sub-class block is independent — when a YAML omits a
    # block, the parser falls back to ``art_default`` so operator
    # configs without sub-class overrides keep working unchanged.
    art_body: ArticulationGains
    art_head: ArticulationGains
    art_arm_shoulder: ArticulationGains
    art_arm_mid: ArticulationGains
    art_arm_wrist: ArticulationGains
    cmd_4ws_timeout_s: float
    render_target_hz: float
    render_safety_ms: float
    gripper_contact: GripperContactParams
    arm_contact: ArmContactParams
    scene_colliders: SceneCollidersParams
    gripper_naming: GripperNaming
    solver: SolverParams


_DEFAULTS = PhysicsParams(
    drive_chassis_drive_joint=DriveParams(stiffness=0.0, damping=1.0e6, max_force=1.0e7, free_limits=True),
    # Steering joints: same shape as the regular position drive but stiffer
    # so the steering rack tracks setpoints fast and accurately. Mirrors the
    # ``art_chassis_steer`` articulation-level kp = 1e5.
    drive_chassis_steer_joint=DriveParams(stiffness=1.0e5, damping=5.0e3, max_force=5.0e3, armature=0.05),
    # Passive damping/frictionloss defaults (``dof_damping``, ``dof_frictionloss``)
    # for body / arm / head are the midpoints of the reference G2 MJCF's
    # per-joint-position tiers: arm shoulder uses 3.0 / 1.0, mid uses
    # 1.5 / 0.5, wrist uses 0.8 / 0.3.  We collapse those into a single
    # per-class value here (1.5 / 0.5) because the classifier in
    # ``common/joint_classification.py`` doesn't split arm by joint
    # number — if you need the tiered profile, add sub-class regexes
    # there.  Lifted from the reference G2 MJCF's arm joints.
    drive_default_revolute=DriveParams(
        stiffness=5.0e4,
        damping=5.0e3,
        max_force=5.0e3,
        dof_damping=1.5,
        dof_frictionloss=0.5,
        armature=0.05,
    ),
    drive_default_prismatic=DriveParams(stiffness=5.0e4, damping=5.0e3, max_force=5.0e4),
    # Scene-articulation passive defaults. Actuator gains are placeholders
    # (never read — scene joints are passive / zero PD); the
    # ``dof_damping=0.5`` / ``dof_frictionloss=0.1`` are the starting values
    # that let a T1 door hinge settle instead of swinging near-
    # frictionlessly (measured a barely-damped oscillation under the old
    # JK_PASSIVE "leave damping untouched" path). ``armature=0`` ⇒ leave
    # whatever the importer wrote.
    drive_scene_articulation=DriveParams(
        stiffness=0.0,
        damping=0.0,
        max_force=0.0,
        dof_damping=0.5,
        dof_frictionloss=0.1,
        armature=0.0,
    ),
    # Per-sub-class passive-joint defaults — see ``PhysicsParams`` docstring.
    # The stiffness / damping / max_force fields are placeholder copies
    # of ``drive_default_revolute``'s actuator gains (the adapter never
    # reads those off the sub-class blocks; per-class gains stay
    # centralised in ``art_default``).  Only ``dof_damping`` and
    # ``dof_frictionloss`` matter here.
    drive_body=DriveParams(
        stiffness=5.0e4,
        damping=5.0e3,
        max_force=5.0e3,
        dof_damping=0.0,
        dof_frictionloss=0.0,
        armature=0.1,
    ),
    drive_head=DriveParams(
        stiffness=5.0e4,
        damping=5.0e3,
        max_force=5.0e3,
        dof_damping=0.0,
        dof_frictionloss=0.0,
        armature=0.05,
    ),
    drive_arm_shoulder=DriveParams(
        stiffness=5.0e4,
        damping=5.0e3,
        max_force=5.0e3,
        dof_damping=3.0,
        dof_frictionloss=1.0,
        armature=0.05,
    ),
    drive_arm_mid=DriveParams(
        stiffness=5.0e4,
        damping=5.0e3,
        max_force=5.0e3,
        dof_damping=1.5,
        dof_frictionloss=0.5,
        armature=0.05,
    ),
    drive_arm_wrist=DriveParams(
        stiffness=5.0e4,
        damping=5.0e3,
        max_force=5.0e3,
        dof_damping=0.8,
        dof_frictionloss=0.3,
        armature=0.02,
    ),
    # Gripper master/mimic tuning — see GripperDriveParams docstring. The
    # 1e4 / 10 / 0.001 / 0 / 0 defaults are hand-tuned values that
    # avoid gravity-induced droop on the Robotiq 2F-140 linkage while
    # keeping the 2F-85 within a working envelope.
    drive_gripper=GripperDriveParams(
        master_stiffness=1.0e4,
        master_damping=10.0,
        armature=0.001,
        mimic_natural_frequency=0.0,
        mimic_damping_ratio=0.0,
    ),
    art_default=ArticulationGains(kp=5.0e4, kd=5.0e3, max_effort=5.0e3),
    art_chassis_drive=ArticulationGains(kp=0.0, kd=1.0e4, max_effort=1.0e7),
    art_chassis_steer=ArticulationGains(kp=1.0e5, kd=5.0e3, max_effort=5.0e3),
    # Per-sub-class actuator-PD gains, lifted from the reference G2
    # MJCF's position-actuator gainprm /
    # biasprm[2].  ``max_effort`` for body/head matches the
    # reference's motor-actuator ``ctrlrange=±1200`` (the reference
    # leaves position-actuator forcerange unlimited and gates
    # input via ctrlrange — ours expresses the cap as max_effort
    # since the runtime PD path doesn't author ctrlrange).  Arm
    # tiers use the URDF effort caps (108 / 35 / 18 N·m).
    art_body=ArticulationGains(kp=1.0e5, kd=1.0e3, max_effort=1.2e3),
    art_head=ArticulationGains(kp=1.0e3, kd=20.0, max_effort=1.2e3),
    art_arm_shoulder=ArticulationGains(kp=4.0e4, kd=600.0, max_effort=108.0),
    art_arm_mid=ArticulationGains(kp=1.5e4, kd=220.0, max_effort=35.0),
    art_arm_wrist=ArticulationGains(kp=8.0e3, kd=100.0, max_effort=18.0),
    cmd_4ws_timeout_s=0.1,
    render_target_hz=30.0,
    render_safety_ms=2.0,
    gripper_contact=GripperContactParams(),
    arm_contact=ArmContactParams(),
    scene_colliders=SceneCollidersParams(),
    gripper_naming=GripperNaming(),
    solver=SolverParams(),
)


def default_physics_params() -> PhysicsParams:
    return _DEFAULTS


def _drive(d: Dict[str, Any], fallback: DriveParams) -> DriveParams:
    return DriveParams(
        stiffness=float(d.get("stiffness", fallback.stiffness)),
        damping=float(d.get("damping", fallback.damping)),
        max_force=float(d.get("max_force", fallback.max_force)),
        free_limits=bool(d.get("free_limits", fallback.free_limits)),
        dof_damping=float(d.get("dof_damping", fallback.dof_damping)),
        dof_frictionloss=float(d.get("dof_frictionloss", fallback.dof_frictionloss)),
        armature=float(d.get("armature", fallback.armature)),
    )


def _art(d: Dict[str, Any], fallback: ArticulationGains) -> ArticulationGains:
    return ArticulationGains(
        kp=float(d.get("kp", fallback.kp)),
        kd=float(d.get("kd", fallback.kd)),
        max_effort=float(d.get("max_effort", fallback.max_effort)),
    )


def _gripper(d: Dict[str, Any], fallback: GripperDriveParams) -> GripperDriveParams:
    # Experiment knob (doc §7.16 vibrational-walking fix): a high kp
    # drives the master into its forcerange cap when squeezing, turning
    # the position-controlled gripper into a CONSTANT-FORCE one — force
    # no longer collapses as the object walks (force = kp * error
    # otherwise shrinks exactly when the grasp is being lost).
    return GripperDriveParams(
        master_stiffness=float(d.get("master_stiffness", fallback.master_stiffness)),
        master_damping=float(d.get("master_damping", fallback.master_damping)),
        armature=float(d.get("armature", fallback.armature)),
        mimic_natural_frequency=float(d.get("mimic_natural_frequency", fallback.mimic_natural_frequency)),
        mimic_damping_ratio=float(d.get("mimic_damping_ratio", fallback.mimic_damping_ratio)),
        master_max_force=float(d.get("master_max_force", fallback.master_max_force)),
        master_ctrl_range_from_joint_limit=bool(
            d.get("master_ctrl_range_from_joint_limit", fallback.master_ctrl_range_from_joint_limit)
        ),
        dof_damping=float(d.get("dof_damping", fallback.dof_damping)),
        dof_frictionloss=float(d.get("dof_frictionloss", fallback.dof_frictionloss)),
        mimic_eq_solref=tuple(d.get("mimic_eq_solref", fallback.mimic_eq_solref)),
        mimic_eq_solimp=tuple(d.get("mimic_eq_solimp", fallback.mimic_eq_solimp)),
    )


def _gripper_naming(d: Dict[str, Any], fallback: GripperNaming) -> GripperNaming:
    return GripperNaming(
        family=str(d.get("family", fallback.family)),
        loop_coupler=str(d.get("loop_coupler", fallback.loop_coupler)),
        driven=str(d.get("driven", fallback.driven)),
        master=str(d.get("master", fallback.master)),
    )


def _gripper_contact(d: Dict[str, Any], fallback: GripperContactParams) -> GripperContactParams:
    _pad = d.get("pad_decomposition", {}) or {}
    if isinstance(_pad, bool):  # allow the ``pad_decomposition: false`` shorthand
        _pad = {"enabled": _pad}
    return GripperContactParams(
        gripper_master_force_cap=float(d.get("gripper_master_force_cap", fallback.gripper_master_force_cap)),
        pad_decomposition=bool(_pad.get("enabled", fallback.pad_decomposition)),
        pad_decomposition_threshold=float(_pad.get("threshold", fallback.pad_decomposition_threshold)),
        pad_decomposition_max_hulls=int(_pad.get("max_hulls", fallback.pad_decomposition_max_hulls)),
    )


def _arm_contact(d: Dict[str, Any], fallback: ArmContactParams) -> ArmContactParams:
    return ArmContactParams(
        arm_max_torque=float(d.get("arm_max_torque", fallback.arm_max_torque)),
    )


def _scene_colliders(d: Dict[str, Any],
                     fallback: SceneCollidersParams) -> SceneCollidersParams:
    _dec = d.get("decomposition", {}) or {}
    if isinstance(_dec, bool):   # ``decomposition: false`` shorthand
        _dec = {"enabled": _dec}
    _box = d.get("box_cover", {}) or {}
    if isinstance(_box, bool):
        _box = {"enabled": _box}
    return SceneCollidersParams(
        decomposition=bool(_dec.get("enabled", fallback.decomposition)),
        decomposition_threshold=float(
            _dec.get("threshold", fallback.decomposition_threshold)),
        decomposition_max_hulls=int(
            _dec.get("max_hulls", fallback.decomposition_max_hulls)),
        decomposition_max_solidity=float(
            _dec.get("max_solidity", fallback.decomposition_max_solidity)),
        decomposition_min_void=float(
            _dec.get("min_void", fallback.decomposition_min_void)),
        decomposition_max_verts=int(
            _dec.get("max_verts", fallback.decomposition_max_verts)),
        # Absent -> None -> stage.py's own default (tables). A present-but-empty
        # list is meaningful ("exclude nothing"), so this distinguishes
        # "unset" from "explicitly cleared" the same way the other knobs'
        # fallbacks do -- ``_dec.get`` returns None only when the key is
        # missing, and an explicit ``exclude_names: []`` yields an empty tuple.
        decomposition_exclude_names=(
            None if _dec.get("exclude_names") is None
            else tuple(str(s) for s in _dec.get("exclude_names"))),
        box_cover=bool(_box.get("enabled", fallback.box_cover)),
        box_cover_pitch=float(_box.get("pitch", fallback.box_cover_pitch)),
        box_cover_max_boxes=int(
            _box.get("max_boxes", fallback.box_cover_max_boxes)),
        background_decor_static=bool(
            d.get("background_decor_static", fallback.background_decor_static)),
    )


def _solver(d: Dict[str, Any], fallback: SolverParams) -> SolverParams:
    return SolverParams(
        impratio=float(d.get("impratio", fallback.impratio)),
        mjc_iterations=int(d.get("mjc_iterations", fallback.mjc_iterations)),
        update_data_interval=int(d.get("update_data_interval", fallback.update_data_interval)),
    )


def load_physics_params(path: Optional[str], logger=None) -> PhysicsParams:
    """Load physics tuning from ``path`` (YAML).

    Empty / missing path, missing file, parse error, or bad shape all log a
    warning (when ``logger`` is provided) and return :func:`default_physics_params`.
    """
    if not path:
        if logger is not None:
            logger.info("physics_params_file empty — using built-in defaults")
        return _DEFAULTS

    try:
        import yaml  # ros2 already pulls PyYAML
    except ImportError:
        if logger is not None:
            logger.warn(f"PyYAML unavailable — using defaults, ignoring {path}")
        return _DEFAULTS

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except OSError as exc:
        if logger is not None:
            logger.warn(f"could not read {path}: {exc} — using defaults")
        return _DEFAULTS
    except yaml.YAMLError as exc:
        if logger is not None:
            logger.warn(f"could not parse {path}: {exc} — using defaults")
        return _DEFAULTS

    # Two-layer drive authoring (see top-of-file block comment in
    # ``config/physics_params.yaml`` for the full explanation).  The YAML
    # canonical block names are ``usd_drive_api:`` (Layer 1: USD-authored,
    # cross-engine) and ``articulation_view_runtime:`` (Layer 2:
    # ArticulationView tensor API, Isaac-only).  We also accept the
    # shorter ``drive:`` / ``articulation:`` block names for back-compat
    # and log a nudge so the rename eventually propagates.
    if "usd_drive_api" in data:
        drive = data.get("usd_drive_api", {}) or {}
    elif "drive" in data:
        if logger is not None:
            logger.warn(
                f"{path}: ``drive:`` block name accepted for back-compat — "
                f"rename to ``usd_drive_api:`` to match the canonical schema."
            )
        drive = data.get("drive", {}) or {}
    else:
        drive = {}
    if "articulation_view_runtime" in data:
        art = data.get("articulation_view_runtime", {}) or {}
    elif "articulation" in data:
        if logger is not None:
            logger.warn(
                f"{path}: ``articulation:`` block name accepted for "
                f"back-compat — rename to ``articulation_view_runtime:`` to "
                f"match the canonical schema."
            )
        art = data.get("articulation", {}) or {}
    else:
        art = {}
    cmd = data.get("command", {}) or {}
    step = data.get("stepping", {}) or {}
    gripper_contact = data.get("gripper_contact", {}) or {}
    arm_contact = data.get("arm_contact", {}) or {}
    solver = data.get("solver", {}) or {}
    scene_colliders = data.get("scene_colliders", {}) or {}

    params = PhysicsParams(
        drive_chassis_drive_joint=_drive(
            drive.get("chassis_drive_joint", {}) or {}, _DEFAULTS.drive_chassis_drive_joint
        ),
        drive_chassis_steer_joint=_drive(
            drive.get("chassis_steer_joint", {}) or {}, _DEFAULTS.drive_chassis_steer_joint
        ),
        drive_default_revolute=_drive(drive.get("default_revolute", {}) or {}, _DEFAULTS.drive_default_revolute),
        drive_default_prismatic=_drive(drive.get("default_prismatic", {}) or {}, _DEFAULTS.drive_default_prismatic),
        # Scene-articulation passive block. Falls back to the built-in
        # 0.5 / 0.1 starting values when the YAML omits it.
        drive_scene_articulation=_drive(drive.get("scene_articulation", {}) or {}, _DEFAULTS.drive_scene_articulation),
        # Per-sub-class passive-joint blocks default the
        # actuator-gain fields from ``default_revolute`` (or from the
        # built-in sub-class default, whichever the YAML doesn't
        # author).  Means an operator only has to write
        # ``dof_damping`` / ``dof_frictionloss`` per sub-class and
        # the unrelated DriveAPI gains stay consistent.
        drive_body=_drive(drive.get("body", {}) or {}, _DEFAULTS.drive_body),
        drive_head=_drive(drive.get("head", {}) or {}, _DEFAULTS.drive_head),
        drive_arm_shoulder=_drive(drive.get("arm_shoulder", {}) or {}, _DEFAULTS.drive_arm_shoulder),
        drive_arm_mid=_drive(drive.get("arm_mid", {}) or {}, _DEFAULTS.drive_arm_mid),
        drive_arm_wrist=_drive(drive.get("arm_wrist", {}) or {}, _DEFAULTS.drive_arm_wrist),
        drive_gripper=_gripper(drive.get("gripper", {}) or {}, _DEFAULTS.drive_gripper),
        art_default=_art(art.get("default", {}) or {}, _DEFAULTS.art_default),
        art_chassis_drive=_art(art.get("chassis_drive", {}) or {}, _DEFAULTS.art_chassis_drive),
        art_chassis_steer=_art(art.get("chassis_steer", {}) or {}, _DEFAULTS.art_chassis_steer),
        # Per-sub-class actuator gains.  Each sub-class falls back to
        # its built-in tiered default (NOT to ``art_default``) — same
        # pattern as ``chassis_drive`` / ``chassis_steer`` above.
        # The tiered defaults match the reference G2 MJCF; an
        # operator who wants a uniform 5e4 / 5e3 across body / arm /
        # head must spell it out explicitly per sub-class in the
        # YAML.
        art_body=_art(art.get("body", {}) or {}, _DEFAULTS.art_body),
        art_head=_art(art.get("head", {}) or {}, _DEFAULTS.art_head),
        art_arm_shoulder=_art(art.get("arm_shoulder", {}) or {}, _DEFAULTS.art_arm_shoulder),
        art_arm_mid=_art(art.get("arm_mid", {}) or {}, _DEFAULTS.art_arm_mid),
        art_arm_wrist=_art(art.get("arm_wrist", {}) or {}, _DEFAULTS.art_arm_wrist),
        cmd_4ws_timeout_s=float(cmd.get("cmd_4ws_timeout_s", _DEFAULTS.cmd_4ws_timeout_s)),
        render_target_hz=float(step.get("render_target_hz", _DEFAULTS.render_target_hz)),
        render_safety_ms=float(step.get("render_safety_ms", _DEFAULTS.render_safety_ms)),
        gripper_contact=_gripper_contact(gripper_contact, _DEFAULTS.gripper_contact),
        arm_contact=_arm_contact(arm_contact, _DEFAULTS.arm_contact),
        gripper_naming=_gripper_naming(data.get("gripper_naming", {}) or {}, _DEFAULTS.gripper_naming),
        solver=_solver(solver, _DEFAULTS.solver),
        scene_colliders=_scene_colliders(
            scene_colliders, _DEFAULTS.scene_colliders),
    )
    # A ``newton:`` block in the YAML is silently ignored — there is no
    # runtime consumer for it.  Issue a one-line nudge so an operator
    # with such a block knows it is dead weight.
    if logger is not None and data.get("newton"):
        logger.warn(
            f"{path}: ``newton:`` block found — no runtime consumer "
            f"exists.  Drop the block to silence this warning."
        )

    if logger is not None:
        logger.info(f"loaded physics params from {path}")
    return params


# ---------------------------------------------------------------------------
# Engine node parameters (ROS parameter dict → typed dataclass)
#
# Mirrors the IsaacLab @configclass pattern: one place owns all field
# definitions, types, defaults, and coercion logic.  EngineSession constructs
# this from the raw {str: str} dict that _parse_args() returns so that the
# rest of EngineSession.__init__ only touches typed attributes.
# ---------------------------------------------------------------------------


@dataclass
class EngineNodeParams:
    """Typed representation of the flat ROS-parameter dict for engine entry points.

    Construct with :meth:`from_dict`.  All coercion and sentinel semantics live here.

    ``render_hz == 0.0`` is the "not provided" sentinel; callers resolve it
    against ``PhysicsParams.render_target_hz`` after loading the YAML config.

    ``physics_solver == ""`` is the "engine picks its own default" sentinel;
    each engine normalises it internally.

    ``physics_solver_substep / physics_solver_iterations == 0`` is the
    "engine picks its own default" sentinel — see the comment in
    ``EngineSession.__init__`` for per-engine semantics.
    """

    physics_hz: float = 100.0
    render_hz: float = 0.0  # 0.0 → use PhysicsParams.render_target_hz
    dds_hz: float = 0.0  # 0.0 → publish every physics tick (legacy behaviour)
    realtime_factor: float = 1.0  # 1.0 = realtime; 0.1 = 10× slower; >1 = faster-than-real
    num_envs: int = 1  # native multi-world replicate count (1 = single env)
    fake_slam: bool = False
    physics_params_file: str = ""
    physics_solver: str = ""  # "" → engine default
    render_mode: str = "raster"
    physics_solver_substep: int = 0  # 0 → engine default
    physics_solver_iterations: int = 0  # 0 → engine default
    physics_solver_mass_matrix_interval: int = (
        0  # 0 → engine default (= sim_substeps; large value disables auto rebuild)
    )
    stage_manifest: str = ""
    init_joint_pos_json: str = ""
    mujoco_pd_ke: float = 0.0  # 0 → engine default (50000 N·m/rad)
    mujoco_pd_kd: float = 0.0  # 0 → engine default (500 N·m·s/rad)
    mujoco_njmax: int = 0  # 0 → auto-scale with scene body count
    mujoco_nconmax: int = 0  # 0 → auto-scale with scene body count
    # /contact_states publish policy. Both fields are tri-state strings
    # — ``""`` means "not set at the launcher level; fall back to the
    # scene YAML's ``contact_publish:`` block (or its manipulation
    # defaults when the scene omits the block)". Backends merge with
    # launcher-wins semantics, matching how ``dds_hz`` overrides
    # anything the scene might otherwise pin.
    #
    # ``contact_publish_scope`` accepts ``"all"``, ``"robot_scene"``,
    # or ``"robot_only"``. ``contact_publish_include_self_contacts``
    # accepts the same truthy/falsy strings ``fake_slam`` does
    # (``true``/``false``/``1``/``0``/``yes``/``no``/``on``/``off``).
    # Invalid values are treated as unset and logged by the adapter.
    contact_publish_scope: str = ""
    contact_publish_include_self_contacts: str = ""

    # ------------------------------------------------------------------
    # Loop pacing policy — WHEN a physics tick is allowed to run.
    #
    # The tick body is identical in every mode; only the gate in front of
    # it changes.  See ``docs/step_modes.md`` for the full treatment and
    # ``common/pacing.py`` for the registry these strings index.
    #
    #   realtime  (default)  wall clock, ``realtime_factor``-scaled.  What
    #                        rolling has always done.  Unchanged.
    #   lockstep             one command == exactly ``step_lockstep_ticks``
    #                        physics ticks, publish on the last one.  For RL
    #                        training, where wall time is not the unit of
    #                        work.  REQUIRES a backend that reports command
    #                        arrivals; without one the loop free-runs and
    #                        warns loudly at bring-up.
    #   freerun              no pacing at all — step as fast as the hardware
    #                        allows.  Benchmarking / offline replay.
    #
    # Unlike the tri-state ``contact_publish_*`` strings above, an
    # unrecognised value here RAISES rather than falling back to the
    # default.  A mis-typed contact scope publishes slightly wrong topics;
    # a mis-typed step mode silently redefines what "one training step"
    # means, and every number the run produces is then wrong in a way no
    # log line would reveal.
    step_mode: str = ""  # "" → pacing.DEFAULT_STEP_MODE ("realtime")

    # Pause the loop after this many seconds with no command, in
    # ``realtime`` mode only.  ``0.0`` (default) = never pause, i.e.
    # rolling's behaviour.  Costs nothing when unset; when set it stops a
    # headless GPU burning a full core between episodes.  Arms only after
    # the FIRST command arrives, so bring-up is never gated.
    step_idle_timeout: float = 0.0

    # Physics ticks bought by one command in ``lockstep`` mode.  ``0``
    # (default) → the pacer's own default.  Ignored by other modes.
    step_lockstep_ticks: int = 0

    # ------------------------------------------------------------------
    # [DEPRECATED — v1 protocol]
    #
    # Publish every body pose on ONE topic (``/tf_render``) instead of the
    # v2 split, and do not publish ``/tf_scene`` at all.
    #
    # **Remaining v1 consumer:** ``genie_sim_render/scripts/isaacsim_render.py``
    # (single-camera Isaac renderer, no-RSP deployments only).  All other
    # render and teleop paths have migrated to v2.  New work must not take a
    # dependency on ``/tf_render``.
    #
    # **Why v1 exists:** a deployment that runs NO ``robot_state_publisher``
    # has no way to reconstruct link poses from the v2 split, so it needs
    # every body on one atomic topic.  Turning this on renames the publisher
    # to ``/tf_render`` and stops ``/tf_scene`` entirely, so no consumer reads
    # a channel whose v2 meaning ("scene bodies only") no longer holds.
    #
    # **v2 protocol (preferred for all new work):** leave this ``False``.
    # The bridge publishes scene-body poses on ``/tf_scene``; robot-link poses
    # reach consumers via ``/tf`` (``map → base_link``) + ``robot_state_publisher``
    # URDF FK.  All bringup-launch deployments start an RSP and use v2.
    #
    # Off by default; rolling users are unaffected.
    use_tf_render: bool = False

    # ------------------------------------------------------------------
    # SysID recording — off by default.  Adapters that don't implement
    # recording silently ignore these fields, so the launcher YAML
    # stays valid across all backends.
    #
    # ``sysid_mode``: ``""`` = off, ``"record"`` = record.  Left as a
    # string (not bool) so future modes can land without another param.
    # ``sysid_label``: free-form tag appended to the run id.  The
    # adapter validates it against ``[A-Za-z0-9][A-Za-z0-9_.-]{0,63}``
    # and falls back to ``"unlabeled"`` for illegal values.
    # ``sysid_max_ticks``: safety cap; ``0`` = no cap.  Recording
    # closes when reached (physics keeps running unless
    # ``sysid_auto_stop`` is on).  Protects against a forgotten
    # ``sysid_mode=record`` filling disk.
    # ``sysid_chunk_size``: flush cadence in ticks.  ``0`` → adapter
    # default.
    # ``sysid_auto_stop``: when ``True`` (default), the run loop stops
    # via ``exit_check`` as soon as recording completes (either
    # ``sysid_max_ticks`` reached or a writer error).  Set to
    # ``False`` to keep the physics loop running after the recorder
    # closes — useful for interactive inspection of the sim after a
    # bounded record.  No effect when ``sysid_mode`` is off.
    sysid_mode: str = ""
    sysid_label: str = ""
    sysid_max_ticks: int = 0
    sysid_chunk_size: int = 0
    sysid_auto_stop: bool = True
    # ``sysid_stimulus``: name of an active-ctrl stimulus generator
    # in ``geniesim_sysid.stimuli.STIMULUS_REGISTRY`` — one of
    # ``""`` / ``"gravity_drop"`` (both = no ctrl writes),
    # ``"joint_step"``, ``"joint_chirp"``, ``"arm_hold"``,
    # ``"grip_close"``.  When non-empty the recorder loads the
    # stimulus and pushes its per-tick targets into the engine via
    # ``apply_commands`` before snapshotting — ``ctrl.parquet``
    # then reflects what was actually applied.
    # ``sysid_stimulus_config``: either inline JSON (``{"joints": …}``
    # style) or a path to a YAML file with the same schema.  Each
    # stimulus has sensible defaults so the empty string is legal.
    sysid_stimulus: str = ""
    sysid_stimulus_config: str = ""

    #: Node params that were renamed. Passing the old name still works — see
    #: the ``use_tf_render`` fallback below — but it prints a one-off notice so
    #: the caller finds out at startup rather than from wrong data later.
    _RENAMED_PARAMS = {"tf_scene_include_robot": "use_tf_render"}

    @classmethod
    def from_dict(cls, params: dict) -> "EngineNodeParams":
        """Coerce a flat ``{str: str}`` params dict into typed fields."""
        for _old, _new in cls._RENAMED_PARAMS.items():
            if _old in params and _new not in params:
                print(
                    f"[engine-params] ⚠ node param {_old!r} was renamed to "
                    f"{_new!r}; the old name still works but will be removed. "
                    f"Note the semantics also changed: {_new} publishes every "
                    f"body on /tf_render and stops publishing /tf_scene.",
                    flush=True,
                )

        def _bool(key: str, default: bool = False) -> bool:
            s = str(params.get(key, "")).strip().lower()
            if s in ("true", "1", "yes", "on"):
                return True
            if s in ("false", "0", "no", "off"):
                return False
            return default

        def _float(key: str, default: float) -> float:
            raw = params.get(key, "")
            try:
                return float(raw) if raw not in ("", None) else default
            except (TypeError, ValueError):
                return default

        def _int_nonneg(key: str) -> int:
            try:
                return max(0, int(params.get(key, 0) or 0))
            except (TypeError, ValueError):
                return 0

        return cls(
            physics_hz=_float("physics_hz", 100.0),
            render_hz=_float("render_hz", 0.0),
            dds_hz=_float("dds_hz", 0.0),
            realtime_factor=max(1e-6, _float("realtime_factor", 1.0)),
            num_envs=max(1, int(params.get("num_envs", 1) or 1)),
            fake_slam=_bool("fake_slam"),
            physics_params_file=str(params.get("physics_params_file", "") or ""),
            physics_solver=str(params.get("physics_solver", "") or "").strip().lower(),
            render_mode=str(params.get("render_mode", "raster") or "raster").strip().lower(),
            physics_solver_substep=_int_nonneg("physics_solver_substep"),
            physics_solver_iterations=_int_nonneg("physics_solver_iterations"),
            physics_solver_mass_matrix_interval=_int_nonneg("physics_solver_mass_matrix_interval"),
            stage_manifest=str(params.get("stage_manifest", "") or ""),
            init_joint_pos_json=str(params.get("init_joint_pos_json", "") or ""),
            mujoco_pd_ke=_float("mujoco_pd_ke", 0.0),
            mujoco_pd_kd=_float("mujoco_pd_kd", 0.0),
            mujoco_njmax=_int_nonneg("mujoco_njmax"),
            mujoco_nconmax=_int_nonneg("mujoco_nconmax"),
            contact_publish_scope=str(params.get("contact_publish_scope", "") or "").strip().lower(),
            contact_publish_include_self_contacts=str(params.get("contact_publish_include_self_contacts", "") or "")
            .strip()
            .lower(),
            # Loop pacing.  Normalised here (trim + lowercase) so a YAML
            # ``Lockstep`` still hits the registry key; VALIDATED in
            # ``pacing.make_pacer``, which raises on anything unknown.
            # Deliberately not validated twice — one owner for the mode
            # vocabulary, and it is the module that defines it.
            step_mode=str(params.get("step_mode", "") or "").strip().lower(),
            step_idle_timeout=max(0.0, _float("step_idle_timeout", 0.0)),
            step_lockstep_ticks=_int_nonneg("step_lockstep_ticks"),
            # ``tf_scene_include_robot`` was the previous name. Dropping it
            # silently would be the worst outcome: the caller keeps passing the
            # old key, the engine keeps its default, ``/tf_scene`` reverts to
            # scene-only, and the caller's robot link poses just vanish — no
            # error, no log, only a downstream consumer reading empty arrays.
            # Honour the old key, and say so loudly.
            use_tf_render=_bool("use_tf_render", _bool("tf_scene_include_robot", False)),
            # ------------------------------------------------------------------
            # SysID recording knobs — see the class-level comment block.
            # ``sysid_mode`` lowercased so YAML capitalization variance
            # doesn't turn recording on/off surprisingly.  ``sysid_label``
            # kept as-authored (case + punctuation preserved for the
            # run_id tag; adapter re-validates against the run_id regex).
            sysid_mode=str(params.get("sysid_mode", "") or "").strip().lower(),
            sysid_label=str(params.get("sysid_label", "") or "").strip(),
            sysid_max_ticks=_int_nonneg("sysid_max_ticks"),
            sysid_chunk_size=_int_nonneg("sysid_chunk_size"),
            sysid_auto_stop=_bool("sysid_auto_stop", True),
            # Stimulus name: normalise case + trim so a launcher YAML
            # ``JointStep`` still hits the registry key ``joint_step``.
            sysid_stimulus=str(params.get("sysid_stimulus", "") or "").strip().lower(),
            # Config: preserve authoring (case + whitespace inside JSON /
            # path) but strip surrounding whitespace.  Empty legal.
            sysid_stimulus_config=str(params.get("sysid_stimulus_config", "") or "").strip(),
        )


# ---------------------------------------------------------------------------
# Scene-sourced initial joint pose
#
# Mirrors the typed-dataclass + override-logging style of DriveParams /
# ArticulationGains, but sourced from the **scene YAML** at launch time
# (forwarded as a JSON-encoded node parameter so it survives launch's
# flat-string parameter pipeline). Kept out of the manifest deliberately
# so re-launching with a tweaked ``init_joint_pos`` does NOT require a
# re-assemble step.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointInitSpec:
    """Requested initial pose for one joint, sourced from scene YAML.

    The numeric ``value`` is interpreted by the consumer based on the
    joint's USD type at apply time:

    * ``UsdPhysics.RevoluteJoint`` → degrees (converted to radians for
      ``set_joint_positions``).
    * ``UsdPhysics.PrismaticJoint`` → metres (passed through verbatim).

    This matches the convention already used in existing scene YAMLs
    (e.g. ``arm_joint2: -90`` is degrees, not radians), and avoids the
    error-prone hand-editing of radian values in YAML.
    """

    value: float


def parse_init_joint_pos(
    payload: Optional[Union[str, Dict[str, Any]]],
    logger=None,
) -> Dict[str, JointInitSpec]:
    """Parse the ``init_joint_pos`` payload into ``{joint_name: JointInitSpec}``.

    Accepts either:

    * a JSON-encoded string (the launch-file → node parameter path), or
    * a Python dict (programmatic / test path).

    Empty / missing payloads return an empty dict; malformed payloads
    log a warning (when ``logger`` is provided) and return empty.
    """
    if not payload:
        return {}

    data: Any
    if isinstance(payload, str):
        s = payload.strip()
        if not s or s in ("{}", "null"):
            return {}
        try:
            data = json.loads(s)
        except json.JSONDecodeError as exc:
            if logger is not None:
                logger.warn(f"init_joint_pos_json: invalid JSON ({exc}) — ignoring")
            return {}
    else:
        data = payload

    if not isinstance(data, dict):
        if logger is not None:
            logger.warn(f"init_joint_pos: expected mapping, got {type(data).__name__} — ignoring")
        return {}

    out: Dict[str, JointInitSpec] = {}
    for jname, raw in data.items():
        try:
            out[str(jname)] = JointInitSpec(value=float(raw))
        except (TypeError, ValueError):
            if logger is not None:
                logger.warn(f"init_joint_pos[{jname}]: non-numeric value {raw!r} — skipping")
    return out
