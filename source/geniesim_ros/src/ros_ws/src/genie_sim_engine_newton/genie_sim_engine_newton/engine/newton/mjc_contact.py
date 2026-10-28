# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""MuJoCo-style per-class contact compliance + friction injection.

Single source of truth: write per-shape ``solref`` / ``solimp`` / friction
into the ``ModelBuilder``'s shape arrays + custom-attribute store BEFORE
``finalize()``.  After finalize, Newton propagates these values into:

  * ``model.shape_material_{ke,kd,mu,mu_torsional,mu_rolling}`` (runtime
    contact kernel + MJW MJCF compile path both read these)
  * the custom-attribute store ``mujoco:geom_solimp`` (frequency=SHAPE),
    which both ``SolverMuJoCo._compile`` (MJCF emit) AND ``mjw_model.
    geom_solimp`` (runtime contact kernel) sample.

That means a single pre-finalize write makes the MJCF dump and the live
simulation agree.  The previous post-finalize approach wrote only to
the runtime buffers and missed the custom-attribute store, producing
the misleading "MJCF shows defaults, runtime uses overrides" split.

Source of override values: the active scene's ``newton.mjc_contact:``
yaml block, merged onto the defaults in
``common/object_classification.py::MJC_CONTACT_DEFAULTS``.

The vocabulary is the five shape classes ``classify_shape`` produces —
``robot`` / ``floor`` / ``passive`` / ``static_prop`` / ``other`` — plus
one REFINEMENT, ``gripper_pad``.  A ``gripper_pad:`` block inherits every
value ``robot`` resolved to and overrides only the keys it names, and it
applies to the robot shapes whose BODY carries the assemble-time
``newton:isGripperPad`` annotation (``common/gripper_annotations.py``;
gripper-classified ``link4`` leaves when the asset predates it).  That
split is deliberate: the yaml supplies the parameters, the asset supplies
which bodies are pads, so a differently-named gripper needs no yaml edit.

It exists because ``condim: 6`` is worth roughly 27% of real-time when it
covers the whole robot class and is only useful where the robot actually
grips.  With no ``gripper_pad:`` block present the pads stay in ``robot``
and nothing about the write changes.

Adapters that don't honour these (Featherstone, AVBD) skip the helper
without modifying any builder state.

Public surface
--------------
* :func:`apply_pre_finalize` — call right before ``builder.finalize()``.
* :func:`readback_post_finalize` — optional diagnostic; logs the
  per-class values that actually landed on the finalized model.  Useful
  for verifying overrides reached the GPU buffers without re-running
  the whole MJCF dump comparison.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Adapter gate
# ---------------------------------------------------------------------------

_MJW_ADAPTER_NAMES = frozenset({"mujoco-warp", "mujoco_warp", "mujoco-warp-multi-env"})


def _is_mjw(adapter_name: str) -> bool:
    """Return True when the active rigid adapter routes through MuJoCo-Warp.

    Featherstone / AVBD use different contact compliance APIs and the
    custom-attribute store this module writes to doesn't apply there —
    skip cleanly rather than corrupt unrelated arrays.
    """
    return (adapter_name or "").strip().lower() in _MJW_ADAPTER_NAMES


# ---------------------------------------------------------------------------
# solref ↔ ke/kd inversion
# ---------------------------------------------------------------------------
#
# MJW's MJCF compile step + the runtime contact kernel both IGNORE any
# direct ``geom_solref`` write — they recompute solref each step from
# ``shape_material_ke / shape_material_kd`` via
# ``convert_solref(ke, kd, d_width=1, d_r=1)``.  Concretely (from
# ``solver_mujoco`` kernels):
#
#     timeconst = 2 / kd
#     dampratio = (kd / 2) * sqrt(1 / ke)
#
# Inverting:
#
#     kd = 2 / timeconst
#     ke = (kd / (2 * dampratio))^2 = (1 / (timeconst * dampratio))^2
#
# So we author solref by writing ke + kd onto ``shape_material_*``.  All
# call sites use the inverted form — never write ``geom_solref`` directly.


def _solref_to_ke_kd(solref: Tuple[float, float]) -> Tuple[float, float]:
    """Invert MJW's ``convert_solref`` so we can write ke/kd that produce
    the requested (timeconst, dampratio).

    Falls back to MJW's stock defaults (timeconst=0.02, dampratio=1.0 →
    ke=2500, kd=100) when the requested values are non-positive — same
    rule MJW's ``convert_solref`` else-branch uses internally.
    """
    timeconst, dampratio = float(solref[0]), float(solref[1])
    if timeconst <= 0.0 or dampratio <= 0.0:
        return 2500.0, 100.0
    kd = 2.0 / timeconst
    ke = (kd / (2.0 * dampratio)) ** 2
    return ke, kd


# ---------------------------------------------------------------------------
# Per-class param resolution
# ---------------------------------------------------------------------------


#: Refinement of ``robot`` rather than a sixth class: it inherits everything
#: ``robot`` resolved to and overrides only the keys it names.  Not a member of
#: ``ALL_OBJECT_KINDS`` — ``classify_shape`` never produces it; membership comes
#: from the ``newton:isGripperPad`` body annotation instead.
PAD_KIND = "gripper_pad"

#: The per-class keys and their required arity.  ``condim`` is scalar and
#: handled separately.
_VECTOR_KEYS = (("solref", 2), ("solimp", 5), ("friction", 3))
_VALID_CONDIM = (1, 3, 4, 6)
_BLOCK_KEYS = frozenset({k for k, _ in _VECTOR_KEYS} | {"condim"})


def _apply_override(dst: Dict[str, Any], override: Any, where: str, logger: Any) -> bool:
    """Fold one yaml block onto ``dst`` in place.  ``False`` if unusable.

    Every rejection warns and keeps the value already in ``dst`` — a scene
    author gets a named line at build time rather than a sim quietly built
    from the defaults they thought they had replaced.
    """
    if not isinstance(override, dict):
        logger.warn(f"[mjc_contact] {where}: expected mapping, got {type(override).__name__}; ignoring")
        return False

    for key in override:
        if key not in _BLOCK_KEYS:
            logger.warn(
                f"[mjc_contact] {where}.{key}: unknown key; ignoring " f"(expected one of {sorted(_BLOCK_KEYS)})"
            )

    for key, expected_n in _VECTOR_KEYS:
        if key not in override:
            continue
        v = override[key]
        if not isinstance(v, (list, tuple)) or len(v) != expected_n:
            logger.warn(
                f"[mjc_contact] {where}.{key}: expected {expected_n}-tuple, " f"got {v!r}; keeping default {dst[key]}"
            )
            continue
        dst[key] = tuple(float(x) for x in v)

    # condim is optional and scalar
    if "condim" in override:
        v = override["condim"]
        try:
            if isinstance(v, float) and not v.is_integer():
                # ``int(6.5)`` is 6, so without this the author gets a
                # working sim built from a number they did not write.
                raise ValueError(v)
            condim = int(v)
            if condim not in _VALID_CONDIM:
                logger.warn(f"[mjc_contact] {where}.condim: expected 1/3/4/6, got {v!r}; not writing condim")
            else:
                dst["condim"] = condim
        except (TypeError, ValueError):
            logger.warn(f"[mjc_contact] {where}.condim: expected int, got {v!r}; not writing condim")
    return True


def _resolve_params(scene_cfg: dict, logger: Any) -> Dict[str, Dict[str, Any]]:
    """Merge ``newton.mjc_contact:`` yaml overrides onto defaults.

    Returns ``{kind: {solref, solimp, friction, condim}}`` for every kind in
    ``ALL_OBJECT_KINDS`` plus :data:`PAD_KIND`.  Validates lengths so a typo in
    the yaml fails loud at build time instead of silently keeping the default.

    ``condim`` is optional and scalar (int 1/3/4/6). When absent, no condim
    is written and MuJoCo uses its stock default (3).

    :data:`PAD_KIND` is always present in the result — it is ``robot``'s
    resolved values when the scene names no ``gripper_pad:`` block.  Whether
    any SHAPE lands in it is decided in :func:`apply_pre_finalize`, which only
    splits the pads out when the scene asked for it.
    """
    from genie_sim_engine_newton.common.object_classification import (
        ALL_OBJECT_KINDS,
        MJC_CONTACT_DEFAULTS,
        OK_ROBOT,
    )  # noqa: PLC0415

    scene_newton = (scene_cfg or {}).get("newton") or {}
    user = scene_newton.get("mjc_contact") or {}
    if not isinstance(user, dict):
        logger.warn(f"[mjc_contact] newton.mjc_contact: expected mapping, got {type(user).__name__}; ignoring")
        user = {}

    known = set(ALL_OBJECT_KINDS) | {PAD_KIND}
    for name in user:
        if name not in known:
            # A migration typo (``gripper_pads:``, ``static-prop:``) is
            # otherwise indistinguishable from "I chose not to override it".
            logger.warn(
                f"[mjc_contact] newton.mjc_contact.{name}: unknown class; ignoring "
                f"(expected one of {sorted(known)})"
            )

    params: Dict[str, Dict[str, Any]] = {}
    for kind in ALL_OBJECT_KINDS:
        params[kind] = dict(MJC_CONTACT_DEFAULTS[kind])
        _apply_override(params[kind], user.get(kind) or {}, kind, logger)

    # gripper_pad REFINES robot: start from whatever robot resolved to —
    # overrides included — so a scene that retunes ``robot.friction`` does not
    # have to restate it here just to add ``condim``.
    params[PAD_KIND] = dict(params[OK_ROBOT])
    _apply_override(params[PAD_KIND], user.get(PAD_KIND) or {}, PAD_KIND, logger)
    return params


# ---------------------------------------------------------------------------
# Stability check
# ---------------------------------------------------------------------------


def _stability_warn(
    params: Dict[str, Dict[str, Tuple[float, ...]]],
    class_counts: Dict[str, int],
    sub_dt: float,
) -> Optional[str]:
    """Return a warn string when ``sub_dt > min(timeconst)/2`` across the
    classes actually present in the scene, else ``None``.

    MJW's contact integrator is stable when the substep dt is below half
    the tightest solref timeconst it has to honour.  Above that the
    Baumgarte stabiliser silently softens contact every other substep,
    producing the "gripper walks through the box" / penetration symptoms
    no amount of yaml tuning can fix.
    """
    if sub_dt <= 0.0:
        return None
    present = [k for k, n in class_counts.items() if n > 0]
    if not present:
        return None
    tightest = min(params[k]["solref"][0] for k in present)
    threshold = tightest / 2.0
    if sub_dt > threshold and threshold > 0.0:
        needed_subs = max(1, int(round(1.0 / threshold)) + 1)
        return (
            f"sub_dt={sub_dt*1000:.2f} ms > min(timeconst)/2={threshold*1000:.2f} ms; "
            f"contact will silently soften.  Raise ``physics_solver_substep`` "
            f"to >= {needed_subs} or relax solref timeconsts."
        )
    return None


# ---------------------------------------------------------------------------
# Pre-finalize injection (THE entry point)
# ---------------------------------------------------------------------------


def apply_pre_finalize(
    builder: Any,
    *,
    scene_cfg: dict,
    robot_prefix: str,
    adapter_name: str,
    physics_hz: float,
    sim_substeps: int,
    logger: Any,
    pad_body_names: Optional[Set[str]] = None,
) -> None:
    """Inject per-class MJW contact compliance + friction into ``builder``.

    Call this **after** ``builder.add_usd(...)`` (so shape arrays are
    populated) and **before** ``builder.finalize()`` (so the custom-
    attribute writes reach both consumers).  No-op on non-MJW adapters.

    ``pad_body_indices`` — the set of BODY indices (into ``builder.
    body_label``) identified as fingertip pads by the assemble-time
    ``newton:isGripperPad`` annotation reader (see
    ``common/gripper_annotations.py``; computed by the caller in
    ``setup/model.py`` which holds the live USD stage).  When provided, a
    shape is a pad iff its owning body is in this set — the physical-truth
    4-pad rule.  When ``None`` (no reader available) the site FALLS BACK to
    the pre-fix ``"link4" in shape_label`` substring test, which also
    (legacy latent bug) broadened to ``arm_l/r_link4`` / ``body_link4``;
    kept only so an un-wired caller still gets the old behaviour.

    Writes per-shape:
      * ``shape_material_ke`` and ``shape_material_kd`` — drive ``solref``
        via MJW's compile-time inversion.
      * ``shape_material_mu`` / ``shape_material_mu_torsional`` /
        ``shape_material_mu_rolling`` — the three friction slots.
      * ``mujoco:geom_solimp`` custom attribute (frequency=SHAPE) — drives
        ``solimp`` for both MJCF emit and runtime kernel.
      * ``mujoco:condim`` custom attribute — only for classes whose yaml
        block names a ``condim``.

    ``pad_body_names`` is the fingertip-pad body NAME set from
    ``common/gripper_annotations.read_pad_body_names`` — the caller has
    already read it off the stage for the CoACD decomposition, so it is passed
    in rather than re-traversed here.  ``None`` resolves the stage-free
    gripper-class ``link4`` fallback, matching ``_gripper_pad_bodies``.  It is
    consulted ONLY when the scene names a ``gripper_pad:`` block; without one
    the pads stay in ``robot`` and every write is byte-identical to before.

    Emits one summary log line per class with the chosen values, the
    source (override vs default), and an MJW stability check warning if
    ``sub_dt > min(timeconst)/2``.
    """
    if not _is_mjw(adapter_name):
        logger.info(f"[mjc_contact] adapter={adapter_name!r}: skipping (non-MJW path)")
        return

    import newton  # noqa: PLC0415
    from genie_sim_engine_newton.common.object_classification import (  # noqa: PLC0415
        ALL_OBJECT_KINDS,
        OK_ROBOT,
        classify_shape,
    )

    n_shapes = int(getattr(builder, "shape_count", 0))
    if n_shapes == 0:
        logger.info("[mjc_contact] no shapes registered yet; skipping")
        return

    params = _resolve_params(scene_cfg, logger)

    # Builder-side snapshots (lists, not warp arrays — pre-finalize).
    shape_types = list(builder.shape_type)
    shape_bodies = list(builder.shape_body)
    shape_labels = list(getattr(builder, "shape_label", []) or [])
    body_labels = list(getattr(builder, "body_label", []) or [])
    JT_PLANE = int(newton.GeoType.PLANE)
    SHAPE_FREQ = newton.Model.AttributeFrequency.SHAPE
    _user = ((scene_cfg or {}).get("newton") or {}).get("mjc_contact")
    user_set = set(_user) if isinstance(_user, dict) else set()

    # ``gripper_pad`` is opt-in.  Without the block the pads stay in ``robot``
    # and this whole path costs one set lookup that never runs.
    pad_bodies: Set[int] = set()
    pad_active = PAD_KIND in user_set
    if pad_active:
        from genie_sim_engine_newton.common.gripper_annotations import (  # noqa: PLC0415
            pad_body_indices,
            read_pad_body_names,
        )

        names = set(pad_body_names) if pad_body_names is not None else read_pad_body_names(labels=body_labels)
        pad_bodies = pad_body_indices(body_labels, names)
        if not pad_bodies:
            # Silently falling back to ``robot`` here is the failure that costs
            # a training run: the scene asked for pad-only condim 6, the grasp
            # behaves like condim 3, and no line in the log says why.
            logger.warn(
                "[mjc_contact] gripper_pad: block present but NO pad body was identified "
                "(the robot USD carries no ``newton:isGripperPad`` annotation and the "
                "gripper-class ``link4`` fallback matched nothing) -- every robot shape "
                "keeps the ``robot`` values. Re-assemble robot.usda to author the "
                "annotation, or move these values under ``robot:``."
            )
            pad_active = False
        else:
            logger.info(f"[mjc_contact] gripper_pad: refining {len(pad_bodies)} pad body(ies) out of ``robot``")

    kinds = tuple(ALL_OBJECT_KINDS) + (PAD_KIND,)
    # Precompute ke/kd per class so we only invert once per kind.
    class_ke_kd: Dict[str, Tuple[float, float]] = {k: _solref_to_ke_kd(params[k]["solref"]) for k in kinds}
    class_counts: Dict[str, int] = {k: 0 for k in kinds}

    has_custom_solimp = hasattr(builder, "custom_attributes") and "mujoco:geom_solimp" in (
        builder.custom_attributes or {}
    )

    for i in range(n_shapes):
        b = int(shape_bodies[i]) if i < len(shape_bodies) else -1
        kind = classify_shape(
            shape_label=shape_labels[i] if i < len(shape_labels) else "",
            body_index=b,
            body_label=body_labels[b] if 0 <= b < len(body_labels) else None,
            shape_type_int=int(shape_types[i]),
            robot_prefix=robot_prefix,
            plane_geo_type=JT_PLANE,
        )
        if pad_active and kind == OK_ROBOT and b in pad_bodies:
            # Body-index selection, not shape-label: ``_decompose_gripper_pads``
            # has already replaced each pad's single hull with N CoACD parts,
            # and all of them hang off the same body.
            kind = PAD_KIND
        p = params[kind]
        ke, kd = class_ke_kd[kind]

        # 1. solref via material ke/kd inversion.
        builder.shape_material_ke[i] = float(ke)
        builder.shape_material_kd[i] = float(kd)

        # 2. friction triplet — three independent slots.
        mu_t, mu_tor, mu_rol = p["friction"]
        builder.shape_material_mu[i] = float(mu_t)
        if hasattr(builder, "shape_material_mu_torsional"):
            builder.shape_material_mu_torsional[i] = float(mu_tor)
        if hasattr(builder, "shape_material_mu_rolling"):
            builder.shape_material_mu_rolling[i] = float(mu_rol)

        # 3. solimp via the custom-attribute store (the asymmetric one
        #    that the post-finalize ``model.geom_solimp`` write doesn't
        #    reach — see this module's docstring).
        if has_custom_solimp:
            builder._process_custom_attributes(
                i,
                {"mujoco:geom_solimp": list(p["solimp"])},
                SHAPE_FREQ,
            )

        # 4. condim — contact dimensionality, gates which friction slots are active.
        #    Only written when the scene yaml provides a condim override for this class.
        #    When absent, MuJoCo uses its stock default (3 = tangential only).
        if "condim" in p:
            has_custom_condim = hasattr(builder, "custom_attributes") and "mujoco:condim" in (
                builder.custom_attributes or {}
            )
            if has_custom_condim:
                builder._process_custom_attributes(
                    i,
                    {"mujoco:condim": int(p["condim"])},
                    SHAPE_FREQ,
                )

        class_counts[kind] += 1

    # Summary log.
    sub_dt = 0.0
    if physics_hz > 0 and sim_substeps > 0:
        sub_dt = (1.0 / float(physics_hz)) / int(sim_substeps)
    warn = _stability_warn(params, class_counts, sub_dt)
    stab = "OK" if warn is None else f"WARN {warn}"

    applied_lines = []
    for kind in kinds:
        if class_counts[kind] == 0:
            continue
        p = params[kind]
        ke, kd = class_ke_kd[kind]
        if kind in user_set:
            src = "override"
        elif kind == PAD_KIND:
            src = "= robot"
        else:
            src = "default"
        condim_str = f", condim={p['condim']}" if "condim" in p else ""
        applied_lines.append(
            f"  {kind:<12} [{class_counts[kind]:>3} shape(s), {src}] "
            f"solref={p['solref']} -> ke={ke:.1f} kd={kd:.1f}, "
            f"solimp={p['solimp']}, friction={p['friction']}{condim_str}"
        )
    logger.info(
        f"[mjc_contact] pre-finalize injection complete -- "
        f"sub_dt={sub_dt*1000:.2f} ms, "
        f"min(timeconst)/2 stability check: {stab}\n" + "\n".join(applied_lines)
    )


# ---------------------------------------------------------------------------
# Optional post-finalize diagnostic
# ---------------------------------------------------------------------------


def readback_post_finalize(
    model: Any,
    *,
    scene_cfg: dict,
    robot_prefix: str,
    adapter_name: str,
    logger: Any,
) -> None:
    """Diagnostic-only: confirm the pre-finalize writes reached the GPU
    buffers + custom-attribute store after ``builder.finalize()``.

    Walks one representative shape per class and prints
    ``shape_material_ke / kd`` (drives solref) and the corresponding
    ``model.geom_solimp[i]`` (drives solimp).  Use this when you want to
    sanity-check the override path without diffing MJCF dumps.

    No state mutation -- safe to call repeatedly.
    """
    if not _is_mjw(adapter_name):
        return

    import newton  # noqa: PLC0415
    from genie_sim_engine_newton.common.object_classification import ALL_OBJECT_KINDS, classify_shape  # noqa: PLC0415

    try:
        n_shapes = int(getattr(model, "shape_count", 0))
        if n_shapes == 0:
            return
        ke = (
            model.shape_material_ke.numpy()
            if hasattr(model.shape_material_ke, "numpy")
            else np.asarray(model.shape_material_ke)
        )
        kd = (
            model.shape_material_kd.numpy()
            if hasattr(model.shape_material_kd, "numpy")
            else np.asarray(model.shape_material_kd)
        )
        mu = (
            model.shape_material_mu.numpy()
            if hasattr(model.shape_material_mu, "numpy")
            else np.asarray(model.shape_material_mu)
        )
        solimp = None
        if hasattr(model, "geom_solimp") and model.geom_solimp is not None:
            solimp = model.geom_solimp.numpy()
            if solimp.ndim == 3:
                solimp = solimp[0]
        shape_types = model.shape_type.numpy() if hasattr(model.shape_type, "numpy") else np.asarray(model.shape_type)
        shape_bodies = model.shape_body.numpy() if hasattr(model.shape_body, "numpy") else np.asarray(model.shape_body)
        shape_labels = list(getattr(model, "shape_label", []) or [])
        body_labels = list(getattr(model, "body_label", []) or [])
        JT_PLANE = int(newton.GeoType.PLANE)

        first_of: Dict[str, int] = {}
        for i in range(n_shapes):
            b = int(shape_bodies[i]) if i < len(shape_bodies) else -1
            kind = classify_shape(
                shape_label=shape_labels[i] if i < len(shape_labels) else "",
                body_index=b,
                body_label=body_labels[b] if 0 <= b < len(body_labels) else None,
                shape_type_int=int(shape_types[i]),
                robot_prefix=robot_prefix,
                plane_geo_type=JT_PLANE,
            )
            if kind not in first_of:
                first_of[kind] = i

        lines = []
        for kind in ALL_OBJECT_KINDS:
            if kind not in first_of:
                continue
            i = first_of[kind]
            solimp_str = ", ".join(f"{v:.4g}" for v in solimp[i]) if solimp is not None else "n/a"
            lines.append(
                f"  {kind:<12} shape[{i}] ke={float(ke[i]):.1f} kd={float(kd[i]):.2f} "
                f"mu={float(mu[i]):.3f}  geom_solimp=[{solimp_str}]"
            )
        if lines:
            logger.info("[mjc_contact] post-finalize readback:\n" + "\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        logger.warn(f"[mjc_contact] post-finalize readback failed: {exc!r}")
