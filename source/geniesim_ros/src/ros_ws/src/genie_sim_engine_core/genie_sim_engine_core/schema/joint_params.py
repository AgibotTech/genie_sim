# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``joint_params`` overlay schema — sparse per-joint / per-link parameter sets.

The URDF is the **baseline** (named ``sim-defaults``: a documented mixture
of vendor/CAD facts and sim-chosen values). An *overlay* is a sparse YAML
file carrying only the parameters some identification run actually
measured. Overlays stack in order, last-wins-per-field.

Sparseness is the whole point: a field is present if and only if it was
identified, so "was this measured or copied to make the file complete?"
stops being a question answerable only by reading prose. The predecessor
mechanism — a one-of-N ``param_source`` selector choosing between full
161-property xacro clones — could not express that, nor combine two runs,
nor carry ``armature`` (which has no URDF representation at all).

Design document: ``source/geniesim_ros/docs/JOINT_PARAMS.md``. The
invariants referenced below (I1..I8) are numbered there.

This module is the **method**; the overlay files are **data** and live in
``genie_sim_robot_model/robots/<vendor>/<model>/overlays/``. Core is handed
paths and never resolves where an overlay lives — the same contract
:func:`~genie_sim_engine_core.common.params.load_physics_params` already
follows, and what keeps core free of a dependency on the asset package.

Phase 1 scope (this module): schema, loader, validator, resolve/diff
queries. Channel A (the URDF->URDF patch) and Channel B
(``JointParamResolver`` at the adapter boundary) are Phase 2/3 and live in
``common/joint_params.py`` when they land.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ._common import _get_str, _list_or_empty, _require_mapping, check_unknown_keys
from .errors import SchemaError, SchemaVersionError

#: Schema identity string every overlay must carry verbatim. Unlike
#: ``MANIFEST_VERSION`` this is not an integer to compare ranges against:
#: overlays are human-authored records of physical measurements, so a
#: version this build does not know is refused outright rather than
#: best-effort upgraded. Bumping means a new ``/2`` reader alongside.
JOINT_PARAMS_SCHEMA_ID = "genie_sim.joint_params/1"

#: ``meta.kind`` vocabulary. ``sysid`` = output of an identification run
#: (a record of a measurement; do not hand-edit). ``correction`` = a
#: deliberate fix to the baseline that is not a measurement.
#: ``operator`` = a scratch overlay for an experiment.
OVERLAY_KINDS = ("sysid", "correction", "operator")

#: ``meta.identified_on_engine`` vocabulary -- the *control law* a drive
#: gain was fitted against, which is not the same question as which
#: process ran the solver. A ``kp`` is only meaningful on the plane it
#: was identified on: MuJoCo's position actuator and PhysX's
#: articulation drive are different laws, so an untagged gain is a
#: number with no units of behaviour. ``validate`` makes the tag
#: mandatory for any ``sysid`` overlay that carries one.
PLANE_MUJOCO_PD = "mujoco_pd"
PLANE_ISAACSIM_PHYSX = "isaacsim_physx"
PLANE_UNRECORDED = "unrecorded"
OVERLAY_ACTUATOR_PLANES = {
    PLANE_MUJOCO_PD: (
        "MuJoCo position actuator: kp is the actuator gainprm[0], kd is "
        "-biasprm[2], and damping/frictionloss are passive <joint> "
        "attributes outside the servo. mujoco-warp and Newton's "
        "SolverMuJoCo share this law, so a fit on either reads here"
    ),
    PLANE_ISAACSIM_PHYSX: (
        "Isaac Sim PhysX articulation drive: kp/kd are DriveAPI "
        "stiffness/damping against a position *and* a velocity target, "
        "summed and then clamped to maxForce. There is no frictionloss "
        "term on this plane at all"
    ),
    PLANE_UNRECORDED: (
        "the run did not record which law it was fitted against. The "
        "gains are reportable but not transferable -- moving them to "
        "another engine needs a refit, not arithmetic"
    ),
}

#: Fields that only exist as part of a servo law, and therefore cannot be
#: read without knowing which law. Presence of one of these in a
#: ``sysid`` overlay is what makes ``identified_on_engine`` mandatory.
PLANE_BOUND_FIELDS = ("stiffness", "damping_drive")

# ---------------------------------------------------------------------------
# Field taxonomy — which channel carries each family (design doc section 4)
# ---------------------------------------------------------------------------

CHANNEL_A = "A"  # URDF -> URDF patch, before the importer
CHANNEL_B = "B"  # JointParamResolver, at the adapter boundary
CHANNEL_AB = "A+B"  # both: A writes it, B defends it from the class table

#: Joint fields and the channel that must carry each one.
#:
#: The ``A+B`` rows are the subtle ones. ``damping`` / ``frictionloss`` ARE
#: URDF-expressible, so Channel A writes them and the importer reads them
#: — and then the mjwarp adapter overwrites them from its per-joint-class
#: table unconditionally. ``effort`` / ``velocity`` survive import but the
#: adapter only ever *raises* a cap, never lowers one. So for these four
#: Channel B's job is not to introduce a value but to **defend** the one
#: Channel A already wrote.
JOINT_FIELD_CHANNEL: Dict[str, str] = {
    "damping": CHANNEL_AB,
    "frictionloss": CHANNEL_AB,
    "effort": CHANNEL_AB,
    "velocity": CHANNEL_AB,
    # No URDF representation whatsoever -> resolver only.
    "armature": CHANNEL_B,
    "stiffness": CHANNEL_B,
    "damping_drive": CHANNEL_B,
    "max_effort": CHANNEL_B,
    "command_delay": CHANNEL_B,
    "torque_constant": CHANNEL_B,
}

#: Backends an overlay may name, as engine package names. ``mujoco`` and
#: ``newton`` drive rigid bodies through the same mujoco-warp adapter and
#: will usually carry the same number; they are kept separate so an engine
#: key maps one-to-one onto a launcher family.
ENGINES = ("mujoco", "newton", "physx")

#: The only fields that may carry a per-engine value. These four are
#: consumed by the backends' *control law*, and the laws differ: the
#: mujoco-warp adapter keeps drive gains ``(ke, kd, effort)`` and passive
#: ``(dof_damping, dof_frictionloss, armature)`` as two independent
#: channels, while PhysX folds passive damping into drive damping
#: (``kd + damping``) and maps ``frictionloss`` onto
#: ``physxJoint:jointFriction``, whose semantics are only approximate.
#:
#: Everything outside this set is the SAME physical quantity in the SAME
#: units on every backend -- ``armature`` is rotor inertia, ``max_effort``
#: is a torque cap, inertials are inertials. An engine key on one of those
#: is a hard error, not a warning: it would almost always mean someone is
#: encoding a backend bug as a measurement (see I9).
ENGINE_SCOPED_FIELDS = frozenset({"stiffness", "damping_drive", "damping", "frictionloss"})

#: Link fields. Inertials are owned by the URDF importer end-to-end and no
#: class table touches them, so Channel A alone is sufficient.
LINK_SCALAR_FIELDS = ("mass",)
LINK_VEC3_FIELDS = ("com", "com_rpy")
INERTIA_FIELDS = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")

#: Fields the schema accepts but for which **no consumer exists yet**.
#: ``validate`` reports these as a warning rather than an error: recording
#: a measured number is useful even before anything reads it, but a caller
#: must not believe it is in effect. ``command_delay`` is the live case —
#: ``git grep command_delay`` is empty, and the arx run attributes ~90% of
#: its reported tracking improvement to it.
UNCONSUMED_FIELDS = frozenset({"command_delay", "torque_constant"})

#: Backends an identified set can be mechanically mapped onto. Only
#: ``physx`` today; ``mujoco`` and ``newton`` are where the sets are
#: identified, so there is nothing to convert.
CONVERSION_TARGETS = ("physx",)

#: Where each joint actuator field lands on PhysX, and how exactly.
#:
#: ``(sink, layer, kind)``. ``layer`` is 2 for the runtime tensor handle
#: (``set_dof_stiffnesses`` / ``set_dof_dampings`` / ``set_dof_max_forces``
#: -- the solve-time authority on both Isaac backends) and 1 for the
#: model-build seed (USD ``DriveAPI`` / ``physxJoint:*``). ``kind`` is
#: ``identity``, ``fold`` or ``no_sink``.
#:
#: The identity rows are not an assumption. One SI class table
#: (``config/physics_params_*.yaml``, section ``articulation_view_runtime``)
#: feeds the mujoco-warp adapter's drive gains AND the Isaac Sim stage,
#: which builds a single ``kps``/``kds``/``max_efforts`` array and ships
#: the same numbers to ``isaac_newton`` and ``isaac_physx``; the branch
#: between them only works around a cpu/cuda pin in
#: ``Articulation.set_gains``. The per-degree convention of the USD
#: angular drive attributes (``5 deg -> 0.0873``) applies to the Layer-1
#: seed alone, which Layer 2 then overwrites -- so it never multiplies an
#: identified gain, and under I7 it is the adapter's business, not an
#: overlay author's.
#:
#: ``damping`` is a ``fold`` rather than an identity because PhysX has one
#: damping channel where mujoco-warp has two. MuJoCo applies
#: ``-kd*qd`` from the actuator and ``-c*qd`` from ``dof_damping``, both
#: against *absolute* velocity; PhysX applies ``d*(qd_target - qd)``. The
#: velocity target is pinned to zero -- once on the authoring path and
#: once at articulation init, the only two velocity-target writes that
#: exist -- so ``d = damping_drive + damping`` is exact, not approximate.
#: See ``tests/api/test_velocity_target_is_zero.py``, which is what keeps
#: that true.
#:
#: ``frictionloss`` has NO sink. MuJoCo's is a constraint-based stick-slip
#: Coulomb torque; ``isaac_physx`` writes no joint-friction attribute at
#: all, so the term is simply absent from the PhysX plant. It cannot be
#: converted into a viscous gain (a Coulomb torque is constant in
#: magnitude, a viscous one is proportional to speed -- they agree at one
#: speed only), and folding it into ``damping`` would be exactly the
#: pre-compensation I7 forbids. A PhysX value for it is a **refit**: a
#: separate measurement on the PhysX plane, written as a ``physx`` branch
#: with its provenance, per I9.
PHYSX_FIELD_MAP: Dict[str, Tuple[Optional[str], int, str]] = {
    "stiffness": ("kp", 2, "identity"),
    "damping_drive": ("kd", 2, "identity"),
    "damping": ("kd", 2, "fold"),
    "max_effort": ("max_effort", 2, "identity"),
    "armature": ("physxJoint:armature", 1, "identity"),
    "frictionloss": (None, 0, "no_sink"),
}

#: Per-joint conversion verdicts, worst first. ``_cmd_convert`` exits
#: nonzero when any joint is ``blocked``.
CONVERT_BLOCKED = "blocked"
CONVERT_LOSSY = "lossy"
CONVERT_CLAMP_RISK = "clamp_risk"
CONVERT_EXACT = "exact"
CONVERT_STATUS_RANK = (CONVERT_BLOCKED, CONVERT_LOSSY, CONVERT_CLAMP_RISK, CONVERT_EXACT)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scalar:
    """One identified number, optionally annotated.

    YAML accepts either form::

        mass: 2.17000000002
        mass: {value: 2.17000000002, frozen_to_cad: true}

    ``frozen_to_cad`` marks a value the run reported but did not actually
    recover — the G2 crsB set freezes link 1-3 masses because the L2/L3
    torque derivative is ~7e-10 N*m/kg. It is annotation, not behaviour:
    a frozen value still applies. Flagging it keeps a reader from citing
    it as evidence the run identified mass.
    """

    value: float
    frozen_to_cad: bool = False
    confidence: Optional[float] = None
    note: str = ""

    @classmethod
    def parse(cls, raw: Any, *, where: str, path: object = None) -> "Scalar":
        if isinstance(raw, Mapping):
            check_unknown_keys(
                raw,
                known=("value", "frozen_to_cad", "confidence", "note"),
                where=where,
                strict=True,
            )
            if "value" not in raw:
                raise SchemaError(f"{where}: mapping form requires a 'value' key", path=path)
            conf = raw.get("confidence")
            return cls(
                value=_as_float(raw["value"], where=f"{where}.value", path=path),
                frozen_to_cad=bool(raw.get("frozen_to_cad", False)),
                confidence=None if conf is None else _as_float(conf, where=f"{where}.confidence", path=path),
                note=_get_str(raw, "note"),
            )
        return cls(value=_as_float(raw, where=where, path=path))

    def to_dict(self) -> Any:
        """Round-trip back to the terse form when nothing is annotated."""
        if not self.frozen_to_cad and self.confidence is None and not self.note:
            return self.value
        out: Dict[str, Any] = {"value": self.value}
        if self.frozen_to_cad:
            out["frozen_to_cad"] = True
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.note:
            out["note"] = self.note
        return out


@dataclass(frozen=True)
class FieldValue:
    """One joint field: either one value for every engine, or one per engine.

    The flat form is unchanged and still the normal case -- it means
    *this number applies to every backend*::

        stiffness: 0.65

    The per-engine form exists for the four :data:`ENGINE_SCOPED_FIELDS`,
    whose backends do not consume the number the same way::

        stiffness:
          mujoco: 0.65
          newton: 0.65
          physx: {value: 0.71, note: "refitted on physx, run 2026-10-20"}

    **I9: a per-engine branch is a separate measurement, never a unit or
    bug fix.** If PhysX folds passive damping into drive damping, that is
    adapter arithmetic and belongs at the adapter boundary (I7) -- writing
    a smaller ``physx`` number to pre-compensate silently breaks the day
    the adapter is fixed. A legitimate branch is one a run actually
    identified against that engine, and it says so in its ``note``.
    """

    shared: Optional[Scalar] = None
    per_engine: Tuple[Tuple[str, Scalar], ...] = ()

    @property
    def is_per_engine(self) -> bool:
        return self.shared is None

    def for_engine(self, engine: str) -> Optional[Scalar]:
        """The Scalar that applies to ``engine``, or None if undecidable.

        A shared value applies to every engine, including the empty one.
        A per-engine value with no engine selected returns None rather
        than guessing a branch -- a caller that does not know its backend
        must not silently receive one backend's number.
        """
        if self.shared is not None:
            return self.shared
        if not engine:
            return None
        return dict(self.per_engine).get(engine)

    def engines(self) -> Tuple[str, ...]:
        return tuple(e for e, _ in self.per_engine)

    def scalars(self) -> Tuple[Scalar, ...]:
        """Every Scalar this field holds, for counting and validation."""
        if self.shared is not None:
            return (self.shared,)
        return tuple(sc for _, sc in self.per_engine)

    # -- parse ----------------------------------------------------------
    @classmethod
    def parse(cls, raw: Any, *, field_name: str, where: str, path: object = None) -> "FieldValue":
        if not isinstance(raw, Mapping):
            return cls(shared=Scalar.parse(raw, where=where, path=path))

        engine_keys = [k for k in raw if k in ENGINES]
        if "value" in raw and engine_keys:
            raise SchemaError(
                f"{where}: cannot mix the annotated scalar form with per-engine "
                f"keys -- found 'value' alongside {sorted(engine_keys)}. Use one "
                f"shared {{value: .., note: ..}} or one branch per engine.",
                path=path,
            )
        if "value" in raw:
            return cls(shared=Scalar.parse(raw, where=where, path=path))
        if not engine_keys:
            # Neither form. Report against whichever one the field can take,
            # so the message names the keys that would actually be accepted.
            legal = f"{sorted(('value', 'frozen_to_cad', 'confidence', 'note'))}"
            if field_name in ENGINE_SCOPED_FIELDS:
                legal += f" or one of {list(ENGINES)}"
            raise SchemaError(f"{where}: mapping form requires {legal}, got {sorted(raw)}", path=path)

        if field_name not in ENGINE_SCOPED_FIELDS:
            raise SchemaError(
                f"{where}: {field_name!r} cannot take a per-engine value. Only "
                f"{sorted(ENGINE_SCOPED_FIELDS)} are consumed by a backend's control "
                f"law; {field_name!r} is the same physical quantity in the same units "
                f"on every engine, so one value applies to all of them (I9).",
                path=path,
            )
        check_unknown_keys(raw, known=ENGINES, where=where, strict=True)
        return cls(
            per_engine=tuple(
                (e, Scalar.parse(raw[e], where=f"{where}.{e}", path=path)) for e in ENGINES if e in raw
            )
        )

    def to_dict(self) -> Any:
        if self.shared is not None:
            return self.shared.to_dict()
        return {e: sc.to_dict() for e, sc in self.per_engine}


def _as_float(raw: Any, *, where: str, path: object = None) -> float:
    # Rejects bool explicitly: YAML `true` would otherwise coerce to 1.0,
    # which for a physical parameter is never what the author meant.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise SchemaError(f"{where}: expected a number, got {raw!r}", path=path)
    return float(raw)


def _vec3(raw: Any, *, where: str, path: object = None) -> Tuple[float, float, float]:
    seq = _list_or_empty(raw)
    if len(seq) != 3:
        raise SchemaError(f"{where}: expected 3 numbers, got {raw!r}", path=path)
    out = [_as_float(v, where=f"{where}[{i}]", path=path) for i, v in enumerate(seq)]
    return (out[0], out[1], out[2])


# ---------------------------------------------------------------------------
# Overlay entries
# ---------------------------------------------------------------------------


@dataclass
class LinkOverlay:
    """Sparse inertial override for one link. Every field is optional."""

    mass: Optional[Scalar] = None
    com: Optional[Tuple[float, float, float]] = None
    com_rpy: Optional[Tuple[float, float, float]] = None
    inertia: Dict[str, Scalar] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, src: Any, *, where: str, path: object = None) -> "LinkOverlay":
        src = _require_mapping(src, where=where, path=path)
        check_unknown_keys(
            src,
            known=LINK_SCALAR_FIELDS + LINK_VEC3_FIELDS + ("inertia",),
            where=where,
            strict=True,
        )
        inertia_src = _require_mapping(src.get("inertia"), where=f"{where}.inertia", path=path)
        check_unknown_keys(inertia_src, known=INERTIA_FIELDS, where=f"{where}.inertia", strict=True)
        return cls(
            mass=Scalar.parse(src["mass"], where=f"{where}.mass", path=path) if "mass" in src else None,
            com=_vec3(src["com"], where=f"{where}.com", path=path) if "com" in src else None,
            com_rpy=(_vec3(src["com_rpy"], where=f"{where}.com_rpy", path=path) if "com_rpy" in src else None),
            inertia={k: Scalar.parse(v, where=f"{where}.inertia.{k}", path=path) for k, v in inertia_src.items()},
        )

    def fields(self) -> Dict[str, "FieldValue"]:
        """Flatten to ``{field_name: FieldValue}`` for uniform resolution.

        ``com`` / ``com_rpy`` flatten to ``com.x`` .. ``com_rpy.z`` so a
        vector and a scalar resolve through the same code path. Inertials
        are never per-engine -- a mass is a mass on every backend -- so
        every value here is a shared :class:`FieldValue`. Returning the
        same type as :meth:`JointOverlay.fields` is what keeps
        ``OverlayStack._resolve`` one loop for both kinds.
        """
        out: Dict[str, "FieldValue"] = {}
        if self.mass is not None:
            out["mass"] = FieldValue(shared=self.mass)
        for name, vec in (("com", self.com), ("com_rpy", self.com_rpy)):
            if vec is not None:
                for axis, v in zip("xyz", vec):
                    out[f"{name}.{axis}"] = FieldValue(shared=Scalar(value=v))
        for k, s in self.inertia.items():
            out[f"inertia.{k}"] = FieldValue(shared=s)
        return out

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.mass is not None:
            out["mass"] = self.mass.to_dict()
        if self.com is not None:
            out["com"] = list(self.com)
        if self.com_rpy is not None:
            out["com_rpy"] = list(self.com_rpy)
        if self.inertia:
            out["inertia"] = {k: v.to_dict() for k, v in self.inertia.items()}
        return out


@dataclass
class JointOverlay:
    """Sparse parameter override for one joint. Keys are :data:`JOINT_FIELD_CHANNEL`.

    Each value is a :class:`FieldValue`: one number for every engine, or
    -- for the four :data:`ENGINE_SCOPED_FIELDS` only -- one per engine.
    """

    values: Dict[str, FieldValue] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, src: Any, *, where: str, path: object = None) -> "JointOverlay":
        src = _require_mapping(src, where=where, path=path)
        check_unknown_keys(src, known=tuple(JOINT_FIELD_CHANNEL), where=where, strict=True)
        return cls(
            values={
                k: FieldValue.parse(v, field_name=k, where=f"{where}.{k}", path=path) for k, v in src.items()
            }
        )

    def fields(self) -> Dict[str, FieldValue]:
        return dict(self.values)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self.values.items()}


@dataclass
class OverlayMeta:
    """Provenance block. ``kind`` is the only required field."""

    kind: str
    identified_on: str = ""
    source_run: str = ""
    #: Which control law the drive gains were fitted against, from
    #: :data:`OVERLAY_ACTUATOR_PLANES`. Mandatory once the overlay
    #: carries a :data:`PLANE_BOUND_FIELDS` value -- see :func:`validate`.
    identified_on_engine: str = ""
    #: Machine-readable list of parameters this run did NOT produce. Exists
    #: so an automated consumer does not mine the file for values that were
    #: never identified — the G2 sidecar currently defends this in prose.
    not_identified: Tuple[str, ...] = ()
    caveats: str = ""

    @classmethod
    def from_dict(cls, src: Any, *, path: object = None) -> "OverlayMeta":
        src = _require_mapping(src, where="meta", path=path)
        check_unknown_keys(
            src,
            known=(
                "kind",
                "identified_on",
                "identified_on_engine",
                "source_run",
                "not_identified",
                "caveats",
            ),
            where="meta",
            strict=True,
        )
        kind = _get_str(src, "kind")
        if kind not in OVERLAY_KINDS:
            raise SchemaError(f"meta.kind: expected one of {list(OVERLAY_KINDS)}, got {kind!r}", path=path)
        plane = _get_str(src, "identified_on_engine")
        if plane and plane not in OVERLAY_ACTUATOR_PLANES:
            raise SchemaError(
                f"meta.identified_on_engine: expected one of "
                f"{list(OVERLAY_ACTUATOR_PLANES)}, got {plane!r}",
                path=path,
            )
        return cls(
            kind=kind,
            identified_on=_get_str(src, "identified_on"),
            identified_on_engine=plane,
            source_run=_get_str(src, "source_run"),
            not_identified=tuple(str(x) for x in _list_or_empty(src.get("not_identified"))),
            caveats=_get_str(src, "caveats"),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind}
        for k in ("identified_on", "identified_on_engine", "source_run", "caveats"):
            v = getattr(self, k)
            if v:
                out[k] = v
        if self.not_identified:
            out["not_identified"] = list(self.not_identified)
        return out


@dataclass
class JointParamOverlay:
    """One overlay file."""

    id: str
    applies_to: Dict[str, str]
    meta: OverlayMeta
    links: Dict[str, LinkOverlay] = field(default_factory=dict)
    joints: Dict[str, JointOverlay] = field(default_factory=dict)
    source_path: Optional[Path] = None

    # -- parse ----------------------------------------------------------
    @classmethod
    def from_dict(cls, src: Any, *, path: object = None) -> "JointParamOverlay":
        src = _require_mapping(src, where="<root>", path=path)
        check_unknown_keys(
            src,
            known=("schema", "id", "applies_to", "meta", "links", "joints"),
            where="<root>",
            strict=True,
        )
        declared = _get_str(src, "schema")
        if declared != JOINT_PARAMS_SCHEMA_ID:
            raise SchemaVersionError(
                f"schema: expected {JOINT_PARAMS_SCHEMA_ID!r}, got {declared!r}",
                path=path,
            )
        overlay_id = _get_str(src, "id")
        if not overlay_id:
            raise SchemaError("id: required and must be non-empty", path=path)
        applies = _require_mapping(src.get("applies_to"), where="applies_to", path=path)
        if not applies:
            raise SchemaError(
                "applies_to: required and must name at least one selector "
                "(e.g. robot_model/body/arm), so an overlay cannot be "
                "silently applied to the wrong robot",
                path=path,
            )
        links_src = _require_mapping(src.get("links"), where="links", path=path)
        joints_src = _require_mapping(src.get("joints"), where="joints", path=path)
        return cls(
            id=overlay_id,
            applies_to={str(k): str(v) for k, v in applies.items()},
            meta=OverlayMeta.from_dict(src.get("meta"), path=path),
            links={str(k): LinkOverlay.from_dict(v, where=f"links.{k}", path=path) for k, v in links_src.items()},
            joints={str(k): JointOverlay.from_dict(v, where=f"joints.{k}", path=path) for k, v in joints_src.items()},
            source_path=Path(path) if isinstance(path, (str, Path)) else None,
        )

    @classmethod
    def read(cls, path: Any) -> "JointParamOverlay":
        import yaml  # deferred — keeps `schema.errors` importable without yaml

        p = Path(path)
        try:
            raw = yaml.safe_load(p.read_text()) or {}
        except FileNotFoundError as err:
            raise SchemaError(f"overlay not found: {p}", path=p) from err
        except yaml.YAMLError as err:
            raise SchemaError(f"overlay {p}: not valid YAML: {err}", path=p) from err
        return cls.from_dict(raw, path=p)

    # -- emit -----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema": JOINT_PARAMS_SCHEMA_ID,
            "id": self.id,
            "applies_to": dict(self.applies_to),
            "meta": self.meta.to_dict(),
        }
        if self.links:
            out["links"] = {k: v.to_dict() for k, v in self.links.items()}
        if self.joints:
            out["joints"] = {k: v.to_dict() for k, v in self.joints.items()}
        return out

    def write(self, path: Any) -> None:
        import yaml

        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False))

    # -- queries --------------------------------------------------------
    def field_count(self) -> int:
        """Total identified scalars. The honest size of the overlay.

        Counts *scalars*, not fields, so a ``stiffness`` split across
        three engines counts three -- three numbers were identified. A
        file with no per-engine value counts exactly as it did before the
        per-engine form existed.
        """
        return sum(
            len(fv.scalars())
            for entries in (self.links.values(), self.joints.values())
            for entry in entries
            for fv in entry.fields().values()
        )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

#: ``source`` value for a field no overlay carries.
BASELINE = "baseline sim-defaults"


@dataclass(frozen=True)
class Resolution:
    """Where one field's winning value came from.

    This is the contract behind ``resolve``, the command that makes an
    edit verifiable instead of assumed. ``channel`` says which mechanism
    carries it: a field in the ``A+B`` family reporting only ``A`` on a
    newton run means the class table will clobber it (design doc 10.1).
    """

    name: str
    value: Optional[float]
    source: str
    channel: str
    frozen_to_cad: bool = False
    unconsumed: bool = False
    #: Which engine branch supplied the value. Empty means the overlay
    #: carries one number for every engine, which is the normal case.
    engine: str = ""
    #: The overlay carries this field per engine and no engine was
    #: selected, so ``value`` is None. Not an error -- a report with no
    #: ``--engine`` is a legitimate way to look at an overlay -- but a
    #: caller about to apply the value must select an engine first.
    needs_engine: bool = False

    @property
    def from_overlay(self) -> bool:
        return self.source != BASELINE


@dataclass
class OverlayStack:
    """An ordered overlay stack. Last overlay wins per field (not per entry).

    Per-*field* precedence is what lets ``[sysid-2026-06-01,
    sysid-2026-09-22]`` express "the newer inertials over the older
    friction" — which is what the G2 crsB set actually is, currently
    achieved by hand-carrying numbers between two files.
    """

    overlays: Tuple[JointParamOverlay, ...] = ()
    #: Which backend this stack is being resolved for, one of
    #: :data:`ENGINES`. Empty is valid and means "not selected": shared
    #: values still resolve, per-engine ones report ``needs_engine``
    #: instead of picking a branch. Channel A and Channel B both know
    #: their engine; a report does not have to.
    engine: str = ""

    def __bool__(self) -> bool:
        return bool(self.overlays)

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(o.id for o in self.overlays)

    def for_engine(self, engine: str) -> "OverlayStack":
        """The same stack bound to an engine. Overlays are not re-read."""
        if engine and engine not in ENGINES:
            raise SchemaError(f"engine: expected one of {list(ENGINES)}, got {engine!r}")
        return OverlayStack(overlays=self.overlays, engine=engine)

    def has_per_engine_fields(self) -> bool:
        """True if any overlay in the stack splits a field by engine."""
        return any(
            fv.is_per_engine
            for ov in self.overlays
            for entry in ov.joints.values()
            for fv in entry.fields().values()
        )

    def _resolve(self, kind: str, name: str, engine: Optional[str] = None) -> Dict[str, Resolution]:
        eng = self.engine if engine is None else engine
        winners: Dict[str, Resolution] = {}
        for ov in self.overlays:
            entry = (ov.links if kind == "link" else ov.joints).get(name)
            if entry is None:
                continue
            for fname, fv in entry.fields().items():
                scalar = fv.for_engine(eng)
                # I6: presence decides, never magnitude. A `damping: 0.0`
                # in a later overlay overrides a nonzero earlier one. That
                # holds per field, so a later overlay carrying this field
                # per engine replaces an earlier shared one outright --
                # the two forms do not merge.
                winners[fname] = Resolution(
                    name=fname,
                    value=None if scalar is None else scalar.value,
                    source=ov.id,
                    channel=_channel_for(kind, fname, per_engine=fv.is_per_engine),
                    frozen_to_cad=bool(scalar is not None and scalar.frozen_to_cad),
                    unconsumed=_base_field(fname) in UNCONSUMED_FIELDS,
                    engine=eng if (fv.is_per_engine and scalar is not None) else "",
                    needs_engine=fv.is_per_engine and scalar is None,
                )
        return winners

    def resolve_joint(self, name: str, engine: Optional[str] = None) -> Dict[str, Resolution]:
        return self._resolve("joint", name, engine)

    def resolve_link(self, name: str, engine: Optional[str] = None) -> Dict[str, Resolution]:
        return self._resolve("link", name, engine)

    def joint_names(self) -> Tuple[str, ...]:
        seen: Dict[str, None] = {}
        for ov in self.overlays:
            for n in ov.joints:
                seen[n] = None
        return tuple(seen)

    def link_names(self) -> Tuple[str, ...]:
        seen: Dict[str, None] = {}
        for ov in self.overlays:
            for n in ov.links:
                seen[n] = None
        return tuple(seen)


#: Mirrors ``genie_sim_engine_newton.common.joint_classification.strip_idx_prefix``.
#: Duplicated rather than imported because core must not depend on a backend.
_RE_IDX_PREFIX = __import__("re").compile(r"^idx\d+_")


def _strip_idx(name: str) -> str:
    """Strip the ``idxNN_`` prefix that URDF joint names carry.

    Overlay keys are URDF names verbatim (I4), and on G2 that means
    ``idx21_arm_l_joint1`` for joints but plain ``arm_l_link1`` for links —
    the prefix is authored into the URDF, not added by the engine. This
    helper exists only for *reporting* (pairing left with right); it is
    never used to rewrite a key.
    """
    return _RE_IDX_PREFIX.sub("", name)


def _base_field(flat_name: str) -> str:
    """``inertia.ixx`` -> ``inertia``; ``damping`` -> ``damping``."""
    return flat_name.split(".", 1)[0]


def _channel_for(kind: str, flat_name: str, *, per_engine: bool = False) -> str:
    """Which mechanism carries this field.

    A per-engine value demotes an ``A+B`` field to ``B`` only. Channel A
    patches the URDF, and there is one URDF shared by every backend, so a
    URDF attribute cannot hold three engine-specific numbers. Only
    ``damping`` and ``frictionloss`` can hit this -- ``stiffness`` and
    ``damping_drive`` are already ``B``. The demotion is reported rather
    than silent so an author sees it here instead of discovering a patch
    that never arrived.
    """
    if kind == "link":
        return CHANNEL_A
    channel = JOINT_FIELD_CHANNEL.get(flat_name, CHANNEL_B)
    if per_engine and channel == CHANNEL_AB:
        return CHANNEL_B
    return channel


def load_joint_overlays(paths: Optional[Sequence[Any]], logger: Any = None) -> OverlayStack:
    """Load an ordered overlay stack from paths.

    Signature deliberately mirrors
    :func:`~genie_sim_engine_core.common.params.load_physics_params`: the
    caller supplies paths, an empty/``None`` argument yields a no-op, and
    core never resolves *where* an overlay lives. That is what keeps this
    package free of a dependency on ``genie_sim_robot_model`` (I4's sibling
    concern — one key space, one ownership boundary).

    An empty stack must be behaviourally identical to today (I1).
    """
    if not paths:
        if logger is not None:
            logger.debug("[joint_params] no overlays — baseline sim-defaults only")
        return OverlayStack()
    loaded = tuple(JointParamOverlay.read(p) for p in paths)
    if logger is not None:
        for ov in loaded:
            logger.info(
                f"[joint_params] overlay {ov.id!r} kind={ov.meta.kind} "
                f"links={len(ov.links)} joints={len(ov.joints)} fields={ov.field_count()}"
            )
    return OverlayStack(loaded)


# ---------------------------------------------------------------------------
# URDF-side validation (I2)
# ---------------------------------------------------------------------------


@dataclass
class UrdfNames:
    """Link and joint names present in a URDF, for name checking."""

    links: Tuple[str, ...]
    joints: Tuple[str, ...]


def read_urdf_names(urdf_path: Any) -> UrdfNames:
    """Collect link/joint names from a URDF.

    ElementTree is correct for *reading*. It must never be used to
    **write** a URDF: a round-trip of the real generated
    ``genie_g2_crsB_wuji.urdf`` loses 8.5% of the file (121 comments,
    including the autogenerated "EDITING THIS FILE BY HAND IS NOT
    RECOMMENDED" provenance header) and rewrites the XML declaration.
    I1 therefore forces Channel A to be a surgical text patch, not an
    XML round-trip. See the design doc.
    """
    import xml.etree.ElementTree as ET

    p = Path(urdf_path)
    try:
        root = ET.parse(str(p)).getroot()
    except FileNotFoundError as err:
        raise SchemaError(f"urdf not found: {p}", path=p) from err
    except ET.ParseError as err:
        raise SchemaError(f"urdf {p}: not valid XML: {err}", path=p) from err
    return UrdfNames(
        links=tuple(e.get("name", "") for e in root.findall("link")),
        joints=tuple(e.get("name", "") for e in root.findall("joint")),
    )


def validate_against_urdf(stack: OverlayStack, urdf_path: Any) -> None:
    """Raise if any overlay names a joint or link the URDF does not have.

    I2: an unknown name is a **hard error**, never a skipped entry and
    never a warning. A typo must not be a silent no-op — that failure
    mode is how a confident wrong answer gets produced, and several
    current code paths already have it (design doc 10.1).

    Overlay keys are URDF names (I4), never engine-side ``idx21_``
    prefixed names; Channel B reaches them via ``strip_idx_prefix``.
    """
    names = read_urdf_names(urdf_path)
    known_links, known_joints = set(names.links), set(names.joints)
    problems: List[str] = []
    for ov in stack.overlays:
        for n in sorted(ov.links):
            if n not in known_links:
                problems.append(f"{ov.id}: links.{n} — no such link in {Path(urdf_path).name}{_hint(n, known_links)}")
        for n in sorted(ov.joints):
            if n not in known_joints:
                problems.append(
                    f"{ov.id}: joints.{n} — no such joint in {Path(urdf_path).name}{_hint(n, known_joints)}"
                )
    if problems:
        raise SchemaError(
            "overlay names not present in the URDF (I2 — an unknown name is "
            "never skipped):\n  " + "\n  ".join(problems),
            path=urdf_path,
        )


def _hint(name: str, candidates: Iterable[str]) -> str:
    """Suggest a near-miss, because a typo is the overwhelmingly likely cause."""
    import difflib

    close = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.7)
    return f"; did you mean {close[0]!r}?" if close else ""


def validate(stack: OverlayStack, urdf_path: Optional[Any] = None) -> List[str]:
    """Full validation. Returns advisory warnings; raises on real errors.

    Warnings are things a caller should know but that do not make the
    overlay wrong: a field with no consumer yet, or a value the run
    reported without actually recovering.
    """
    warnings: List[str] = []
    if urdf_path is not None:
        validate_against_urdf(stack, urdf_path)
    for ov in stack.overlays:
        # A drive gain without its control law is unusable: the same number
        # means different behaviour under MuJoCo's position actuator than
        # under PhysX's articulation drive, and the difference is not a unit
        # conversion. This is an error rather than a warning because the
        # downstream failure is silent -- a kp read onto the wrong plane
        # produces a plausible-looking run, not a crash.
        if ov.meta.kind == "sysid" and not ov.meta.identified_on_engine:
            carried = sorted(
                f
                for entry in ov.joints.values()
                for f in entry.fields()
                if _base_field(f) in PLANE_BOUND_FIELDS
            )
            if carried:
                raise SchemaError(
                    f"{ov.id}: carries {', '.join(sorted(set(carried)))} but "
                    f"meta.identified_on_engine is unset. A servo gain is only "
                    f"meaningful on the law it was fitted against; declare one of "
                    f"{list(OVERLAY_ACTUATOR_PLANES)} "
                    f"({PLANE_UNRECORDED!r} if the run genuinely did not record it)",
                    path=ov.id,
                )
        declared_absent = set(ov.meta.not_identified)
        for scope, entries in (("joints", ov.joints), ("links", ov.links)):
            for name, entry in entries.items():
                for fname, fv in entry.fields().items():
                    base = _base_field(fname)
                    where = f"{ov.id}: {scope}.{name}.{fname}"
                    if base in UNCONSUMED_FIELDS:
                        warnings.append(
                            f"{where} is recorded but NOTHING CONSUMES IT yet — "
                            f"do not assume it is in effect"
                        )
                    if base in declared_absent:
                        warnings.append(
                            f"{where} is present but meta.not_identified lists "
                            f"{base!r} — contradictory"
                        )
                    for eng, scalar in (fv.per_engine or ((("", fv.shared),) if fv.shared else ())):
                        if scalar is not None and scalar.frozen_to_cad:
                            suffix = f" (engine {eng})" if eng else ""
                            warnings.append(
                                f"{where}{suffix} is frozen_to_cad — reported but "
                                f"not recovered by the run"
                            )
                    warnings.extend(_per_engine_warnings(fv, field_name=fname, where=where))
    return warnings


def _per_engine_warnings(fv: FieldValue, *, field_name: str, where: str) -> List[str]:
    """I9 checks on a per-engine field. All advisory; none make it wrong.

    A per-engine branch claims a *separate measurement* against that
    backend. Three things make that claim doubtful, and each is something
    a reader would otherwise have to notice by eye.
    """
    if not fv.is_per_engine:
        return []
    out: List[str] = []
    named = fv.engines()

    missing = [e for e in ENGINES if e not in named]
    if missing:
        # The sharp one: an unnamed engine gets NO value, not the baseline
        # of one of its siblings. Easy to write by accident when only one
        # backend was refitted.
        out.append(
            f"{where} names only {list(named)} — {missing} resolve to no value "
            f"at all. Name every engine, or carry the field flat and add one "
            f"branch only where a run actually measured a different number"
        )

    values = [sc.value for _, sc in fv.per_engine]
    if len(named) > 1 and len(set(values)) == 1:
        out.append(
            f"{where} carries the same value {values[0]:.12g} on {list(named)} — "
            f"that is a flat value written {len(named)} times. Write it flat "
            f"unless the branches are genuinely separate measurements (I9)"
        )

    unannotated = [e for e, sc in fv.per_engine if not sc.note]
    if unannotated:
        out.append(
            f"{where} splits by engine but {unannotated} carry no note — a "
            f"per-engine value is a separate measurement and must say what "
            f"identified it, never a fix for a backend quirk (I7, I9)"
        )

    if JOINT_FIELD_CHANNEL.get(_base_field(field_name), CHANNEL_B) == CHANNEL_AB:
        out.append(
            f"{where} is normally {CHANNEL_AB} but a per-engine value demotes "
            f"it to Channel {CHANNEL_B}: one URDF is shared by every backend, "
            f"so Channel A cannot carry three engine-specific numbers"
        )
    return out


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------
#
# Defined here as a dict rather than shipped as a sibling .json file: the
# python package is installed by `ament_python_install_package`, which
# copies .py only (no data files exist anywhere in the package tree today;
# `config/` installs separately into share/). Keeping it as a module
# constant means one source of truth, always importable, and zero new
# install plumbing. `emit-schema` writes it out for editors and agents.


def json_schema() -> Dict[str, Any]:
    """The JSON Schema for a v1 overlay, for pre-write validation."""
    scalar = {
        "oneOf": [
            {"type": "number"},
            {
                "type": "object",
                "required": ["value"],
                "additionalProperties": False,
                "properties": {
                    "value": {"type": "number"},
                    "frozen_to_cad": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "note": {"type": "string"},
                },
            },
        ]
    }
    # Per-engine form, for ENGINE_SCOPED_FIELDS only. The two branches of
    # the oneOf are mutually exclusive by construction: the scalar object
    # branch requires `value` with additionalProperties false, so an
    # engine key fails it; this branch allows engine keys only, so `value`
    # fails it. Keeping the other six joint fields on the bare `scalar`
    # is what makes `armature: {physx: ..}` fail JSON-Schema validation
    # too, not only the parser.
    engine_scoped = {
        "oneOf": [
            scalar,
            {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": False,
                "properties": {e: scalar for e in ENGINES},
            },
        ]
    }
    vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": JOINT_PARAMS_SCHEMA_ID,
        "title": "Genie Sim joint parameter overlay",
        "type": "object",
        "required": ["schema", "id", "applies_to", "meta"],
        "additionalProperties": False,
        "properties": {
            "schema": {"const": JOINT_PARAMS_SCHEMA_ID},
            "id": {"type": "string", "minLength": 1, "pattern": r"^[A-Za-z0-9_.-]+$"},
            "applies_to": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": {"type": "string"},
            },
            "meta": {
                "type": "object",
                "required": ["kind"],
                "additionalProperties": False,
                "properties": {
                    "kind": {"enum": list(OVERLAY_KINDS)},
                    "identified_on": {"type": "string"},
                    "identified_on_engine": {"enum": list(OVERLAY_ACTUATOR_PLANES)},
                    "source_run": {"type": "string"},
                    "not_identified": {"type": "array", "items": {"type": "string"}},
                    "caveats": {"type": "string"},
                },
            },
            "links": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "mass": scalar,
                        "com": vec3,
                        "com_rpy": vec3,
                        "inertia": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {k: scalar for k in INERTIA_FIELDS},
                        },
                    },
                },
            },
            "joints": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        k: (engine_scoped if k in ENGINE_SCOPED_FIELDS else scalar)
                        for k in JOINT_FIELD_CHANNEL
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Baseline extraction (read-only)
# ---------------------------------------------------------------------------

#: URDF element/attribute -> overlay field name. The one non-obvious row is
#: ``drive@damping`` -> ``damping_drive``: a joint has BOTH a passive
#: ``<dynamics damping>`` (resisting motion) and a drive-loop derivative
#: gain ``<drive damping>``, which are different physical quantities with
#: the same attribute name. The overlay namespace separates them so an
#: author cannot write one meaning the other.
_URDF_JOINT_FIELDS = (
    ("dynamics", "damping", "damping"),
    ("dynamics", "friction", "frictionloss"),
    ("limit", "effort", "effort"),
    ("limit", "velocity", "velocity"),
    ("drive", "stiffness", "stiffness"),
    ("drive", "damping", "damping_drive"),
)


def read_urdf_baseline(urdf_path: Any) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Extract today's values from a URDF, keyed ``(kind, name) -> {field: value}``.

    Read-only; see :func:`read_urdf_names` on why ElementTree must never
    write a URDF here.
    """
    import xml.etree.ElementTree as ET

    p = Path(urdf_path)
    try:
        root = ET.parse(str(p)).getroot()
    except FileNotFoundError as err:
        raise SchemaError(f"urdf not found: {p}", path=p) from err
    except ET.ParseError as err:
        raise SchemaError(f"urdf {p}: not valid XML: {err}", path=p) from err

    out: Dict[Tuple[str, str], Dict[str, float]] = {}

    for link in root.findall("link"):
        vals: Dict[str, float] = {}
        inertial = link.find("inertial")
        if inertial is not None:
            mass = inertial.find("mass")
            if mass is not None and mass.get("value") is not None:
                vals["mass"] = float(mass.get("value", "0"))
            origin = inertial.find("origin")
            if origin is not None:
                for attr, fname in (("xyz", "com"), ("rpy", "com_rpy")):
                    raw = origin.get(attr)
                    if raw:
                        parts = raw.split()
                        if len(parts) == 3:
                            for axis, v in zip("xyz", parts):
                                vals[f"{fname}.{axis}"] = float(v)
            inertia = inertial.find("inertia")
            if inertia is not None:
                for k in INERTIA_FIELDS:
                    raw = inertia.get(k)
                    if raw is not None:
                        vals[f"inertia.{k}"] = float(raw)
        out[("link", link.get("name", ""))] = vals

    for joint in root.findall("joint"):
        vals = {}
        for tag, attr, fname in _URDF_JOINT_FIELDS:
            el = joint.find(tag)
            if el is None:
                continue
            raw = el.get(attr)
            if raw is not None:
                vals[fname] = float(raw)
        out[("joint", joint.get("name", ""))] = vals

    return out


def read_urdf_mimic_followers(urdf_path: Any) -> "frozenset[str]":
    """Joint names carrying a ``<mimic>`` tag, i.e. kinematic followers.

    Load-bearing for the PhysX conversion, and for nothing else. A mimic
    follower is constrained by ``PhysxMimicJointAPI`` on the PhysX path and
    is given **no DriveAPI** -- deliberately, because a force drive in
    parallel with a kinematic mimic constraint fights it. So a follower has
    no ``kp``/``kd``/``maxForce`` to receive a converted gain at all, and
    the usual fold has nowhere to land.

    ``isaac_newton`` is the opposite case: it seeds a follower DriveAPI on
    purpose, because Newton's importer will not allocate a POSITION
    actuator for a drive left at ``(0, 0)``. Same overlay, same USD, two
    different sinks -- which is why this cannot be inferred from the
    numbers and has to be read off the kinematics.
    """
    import xml.etree.ElementTree as ET

    p = Path(urdf_path)
    try:
        root = ET.parse(str(p)).getroot()
    except FileNotFoundError as err:
        raise SchemaError(f"urdf not found: {p}", path=p) from err
    except ET.ParseError as err:
        raise SchemaError(f"urdf {p}: not valid XML: {err}", path=p) from err

    return frozenset(
        j.get("name", "") for j in root.findall("joint") if j.find("mimic") is not None and j.get("name")
    )


def resolve_all(
    stack: OverlayStack,
    kind: str,
    name: str,
    baseline: Optional[Mapping[str, float]] = None,
) -> Dict[str, Resolution]:
    """Overlay winners merged over baseline values, as a full field picture.

    Baseline-sourced fields are reported too, so ``resolve`` can show a
    field you expected to be overridden sitting at its baseline value —
    which is the signal that the overlay does not carry it.
    """
    resolved = stack.resolve_joint(name) if kind == "joint" else stack.resolve_link(name)
    out: Dict[str, Resolution] = {}
    for fname, value in (baseline or {}).items():
        out[fname] = Resolution(
            name=fname,
            value=value,
            source=BASELINE,
            channel=_channel_for(kind, fname),
        )
    out.update(resolved)
    return out


# ---------------------------------------------------------------------------
# Cross-backend conversion
# ---------------------------------------------------------------------------
#
# The sets in the tree are identified on mujoco / newton. This maps one
# onto PhysX and -- the part that matters -- says per joint what the map
# loses. See PHYSX_FIELD_MAP for why each field converts the way it does.


@dataclass(frozen=True)
class PhysxJointGains:
    """One joint's PhysX-side actuator values, and what converting cost.

    ``kd_terms`` keeps the fold visible: ``(("damping_drive", 6.853),
    ("damping", 1.800))`` rather than just ``8.653``, so a reader can see
    that a passive channel was absorbed and check it against the source.
    """

    joint: str
    kp: Optional[float] = None
    kd: Optional[float] = None
    kd_terms: Tuple[Tuple[str, float], ...] = ()
    max_effort: Optional[float] = None
    armature: Optional[float] = None
    #: Coulomb torque present in the source set and absent from the PhysX
    #: plant, in N*m. Never folded into ``kd`` -- see PHYSX_FIELD_MAP.
    frictionloss_dropped: float = 0.0
    #: Torque cap the dropped Coulomb is measured against: the overlay's
    #: own ``max_effort`` when it has one, else the URDF effort limit from
    #: ``--urdf``. None means neither was available.
    effort_ref: Optional[float] = None
    #: Which of those two supplied ``effort_ref`` -- ``"max_effort"`` or
    #: ``"urdf-effort"``. They are different quantities and the report has to
    #: name the one it used: ``max_effort`` is the runtime drive clamp, a
    #: condition of the run, while ``<limit effort>`` is the vendor rating.
    #: No sysid overlay in the tree publishes ``max_effort``, so in practice
    #: this reads ``"urdf-effort"`` or the fraction is None.
    effort_ref_source: Optional[str] = None
    status: str = CONVERT_EXACT
    #: ``(kind, text)`` pairs, kind in ``blocked`` / ``frictionloss`` /
    #: ``clamp``. Tagged rather than free text so a caller can group them:
    #: the clamp note is the same sentence on every folded joint.
    notes: Tuple[Tuple[str, str], ...] = ()

    @property
    def dropped_fraction(self) -> Optional[float]:
        """Dropped Coulomb as a fraction of the torque cap, or None.

        This is the number that decides whether the loss is ignorable: a
        0.07 N*m Coulomb term under an 18 N*m rating is noise, 0.53 N*m
        under a 15 N*m rating is visible droop in a static hold.

        Read it with ``effort_ref_source``: the same torque against a vendor
        rating and against a tighter run clamp gives two different
        percentages, and only the second is headroom the run actually had.
        """
        if not self.effort_ref:
            return None
        return self.frictionloss_dropped / abs(self.effort_ref)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint": self.joint,
            "kp": self.kp,
            "kd": self.kd,
            "kd_terms": [{"field": f, "value": v} for f, v in self.kd_terms],
            "max_effort": self.max_effort,
            "armature": self.armature,
            "frictionloss_dropped": self.frictionloss_dropped,
            "effort_ref": self.effort_ref,
            "effort_ref_source": self.effort_ref_source,
            "dropped_fraction": self.dropped_fraction,
            "status": self.status,
            "notes": [{"kind": k, "text": t} for k, t in self.notes],
        }


@dataclass(frozen=True)
class PhysxConversion:
    """A whole set mapped onto PhysX, per joint."""

    rows: Tuple[PhysxJointGains, ...] = ()
    from_engine: str = ""
    ids: Tuple[str, ...] = ()
    #: ``meta.identified_on_engine`` of every overlay that contributed a
    #: servo gain, deduplicated. This is the control law the numbers on
    #: the left of the conversion were fitted against -- distinct from
    #: ``from_engine``, which only selects a per-engine *branch* within
    #: one overlay.
    source_planes: Tuple[str, ...] = ()

    def counts(self) -> Dict[str, int]:
        out = {s: 0 for s in CONVERT_STATUS_RANK}
        for r in self.rows:
            out[r.status] = out.get(r.status, 0) + 1
        return out

    @property
    def blocked(self) -> Tuple[PhysxJointGains, ...]:
        return tuple(r for r in self.rows if r.status == CONVERT_BLOCKED)

    @property
    def total_dropped(self) -> float:
        return sum(r.frictionloss_dropped for r in self.rows)

    @property
    def untransferable(self) -> bool:
        """True when the source plane is unknown, so no row can be trusted.

        Separate from :attr:`blocked`, which is a per-joint verdict. This
        one condemns the whole table: the arithmetic below is only a
        conversion if the numbers above it were PD gains on the MuJoCo
        law. If nobody recorded that, the output is a reformatting.
        """
        return PLANE_UNRECORDED in self.source_planes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": "physx",
            "from_engine": self.from_engine,
            "source_planes": list(self.source_planes),
            "untransferable": self.untransferable,
            "overlays": list(self.ids),
            "counts": self.counts(),
            "total_frictionloss_dropped": self.total_dropped,
            "joints": [r.to_dict() for r in self.rows],
        }


def convert_to_physx(
    stack: OverlayStack,
    *,
    from_engine: str = "",
    baseline: Optional[Mapping[Tuple[str, str], Mapping[str, float]]] = None,
    mimic_followers: "frozenset[str]" = frozenset(),
) -> PhysxConversion:
    """Map a mujoco/newton-identified set onto PhysX, per joint.

    Resolution goes through :meth:`OverlayStack.resolve_joint`, so layer
    precedence and per-engine branches behave exactly as ``resolve``
    reports them. ``baseline`` is a :func:`read_urdf_baseline` result,
    used only for the ``effort`` denominator; ``mimic_followers`` is a
    :func:`read_urdf_mimic_followers` result, and a joint named there has
    no PhysX drive to receive anything (see that function).

    This returns a **report**. It deliberately does not emit a ``physx``
    overlay branch for anything it converts exactly: baking the fold into
    the data is the pre-compensation I7 forbids, and it would
    double-apply the day the adapter does the fold itself.

    Raises :class:`SchemaError` if the stack splits a field by engine and
    ``from_engine`` is empty -- converting from an unselected branch would
    silently pick one backend's measurement for another backend -- or if a
    contributing overlay was itself identified on PhysX, where the whole
    conversion is a category error (see :data:`OVERLAY_ACTUATOR_PLANES`).
    """
    if from_engine and from_engine not in ENGINES:
        raise SchemaError(f"unknown engine {from_engine!r}; expected one of {', '.join(ENGINES)}")
    # The source plane of the *gains*, not of the branch selection. An
    # overlay that carries no servo gain contributes no plane: crsB's
    # passive damping has nothing to fold onto and nothing to mis-read.
    planes: List[str] = []
    for ov in stack.overlays:
        if not any(
            _base_field(f) in PLANE_BOUND_FIELDS for entry in ov.joints.values() for f in entry.fields()
        ):
            continue
        plane = ov.meta.identified_on_engine or PLANE_UNRECORDED
        if plane not in planes:
            planes.append(plane)
    if PLANE_ISAACSIM_PHYSX in planes:
        raise SchemaError(
            f"this set was identified on {PLANE_ISAACSIM_PHYSX!r} and is already on the "
            f"target plane; converting it would fold passive damping into a kd that "
            f"already contains it. Read the overlay as-is"
        )
    bound = stack.for_engine(from_engine)
    rows: List[PhysxJointGains] = []
    for name in bound.joint_names():
        winners = bound.resolve_joint(name)
        pending = sorted(f for f, r in winners.items() if r.needs_engine)
        if pending:
            raise SchemaError(
                f"joints.{name}: {', '.join(pending)} is carried per engine and no source "
                f"engine was selected; pass --from ({'|'.join(ENGINES)}) so the conversion "
                f"reads a named measurement instead of guessing a branch"
            )

        def _val(field: str) -> Optional[float]:
            r = winners.get(field)
            return None if r is None else r.value

        stiffness = _val("stiffness")
        damping_drive = _val("damping_drive")
        damping = _val("damping")
        frictionloss = _val("frictionloss")

        kd_terms: List[Tuple[str, float]] = []
        if damping_drive is not None:
            kd_terms.append(("damping_drive", damping_drive))
        if damping is not None:
            kd_terms.append(("damping", damping))
        kd = sum(v for _, v in kd_terms) if kd_terms else None

        max_effort = _val("max_effort")
        effort_ref = max_effort
        effort_ref_source = None if max_effort is None else "max_effort"
        if effort_ref is None and baseline is not None:
            effort_ref = (baseline.get(("joint", name)) or {}).get("effort")
            effort_ref_source = None if effort_ref is None else "urdf-effort"

        dropped = float(frictionloss or 0.0)
        notes: List[Tuple[str, str]] = []
        status = CONVERT_EXACT

        folded = damping is not None and damping != 0.0
        is_follower = name in mimic_followers
        if is_follower and (folded or kd is not None or stiffness is not None):
            status = CONVERT_BLOCKED
            notes.append(
                (
                    "blocked-no-drive",
                    "this joint is a kinematic mimic follower, which on PhysX is constrained "
                    "by PhysxMimicJointAPI and carries no DriveAPI at all -- there is no kp, "
                    "kd or maxForce to receive these gains, and authoring one would fight the "
                    "mimic constraint",
                )
            )
        elif folded and stiffness is None:
            status = CONVERT_BLOCKED
            notes.append(
                (
                    "blocked",
                    f"damping {_fmt(damping)} folds into PhysX drive damping but this joint "
                    "has no identified stiffness, so the folded term would act against a kp "
                    "this set never measured",
                )
            )
        elif dropped:
            status = CONVERT_LOSSY
        elif folded:
            status = CONVERT_CLAMP_RISK

        if dropped:
            frac = ""
            if effort_ref:
                ref_name = "max_effort" if effort_ref_source == "max_effort" else "the URDF effort limit"
                frac = f" ({dropped / abs(effort_ref) * 100.0:.1f}% of {ref_name} {_fmt(effort_ref)})"
            else:
                frac = " (no max_effort or URDF effort limit available to scale it against)"
            notes.append(("frictionloss", f"frictionloss {_fmt(dropped)} N*m has no PhysX sink and is dropped{frac}"))
        if folded and status != CONVERT_BLOCKED:
            notes.append(
                (
                    "clamp",
                    f"kd folds passive damping {_fmt(damping)} into the drive, which puts it "
                    "under the maxForce clamp; MuJoCo applies it unclamped, so the two differ "
                    "only while the drive saturates",
                )
            )
        rows.append(
            PhysxJointGains(
                joint=name,
                kp=stiffness,
                kd=kd,
                kd_terms=tuple(kd_terms),
                max_effort=max_effort,
                armature=_val("armature"),
                frictionloss_dropped=dropped,
                effort_ref=effort_ref,
                effort_ref_source=effort_ref_source,
                status=status,
                notes=tuple(notes),
            )
        )
    return PhysxConversion(
        rows=tuple(rows),
        from_engine=from_engine,
        ids=bound.ids,
        source_planes=tuple(planes),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#
# Invoked as `python -m genie_sim_engine_core.schema.joint_params <cmd>`.
# No console_scripts exist anywhere in this package (CMakeLists uses
# `ament_python_install_package`), so `python -m` is the zero-plumbing
# entry point rather than the first `install(PROGRAMS ...)` rule.


def _fmt(v: Optional[float]) -> str:
    return "(none)" if v is None else f"{v:.12g}"


def _cmd_validate(args: Any) -> int:
    stack = load_joint_overlays(args.overlay)
    warnings = validate(stack, args.urdf)
    if args.json:
        print(json.dumps({"ok": True, "ids": list(stack.ids), "warnings": warnings}, indent=2))
    else:
        for ov in stack.overlays:
            print(
                f"ok  {ov.id}  kind={ov.meta.kind}  links={len(ov.links)}  "
                f"joints={len(ov.joints)}  fields={ov.field_count()}"
            )
        for w in warnings:
            print(f"warn  {w}")
        if args.urdf:
            print(f"names checked against {Path(args.urdf).name}: all present (I2)")
    return 0


def _bind_engine(stack: OverlayStack, args: Any) -> OverlayStack:
    """Bind ``--engine`` onto the stack, warning when it was needed."""
    engine = getattr(args, "engine", "") or ""
    if not engine and stack.has_per_engine_fields():
        print(
            f"warn  this stack splits at least one field by engine; without "
            f"--engine ({'|'.join(ENGINES)}) those fields have no single value",
            file=__import__("sys").stderr,
        )
    return stack.for_engine(engine)


def _resolve_rows(
    stack: OverlayStack,
    kind: str,
    name: str,
    baseline: Optional[Mapping[str, float]],
    engine: str,
) -> List[Tuple[str, Resolution]]:
    """``(field, Resolution)`` rows for one entry.

    With an engine selected this is one row per field. With none, a
    per-engine field expands to one row per branch, so divergence between
    backends is visible in a single command instead of three.
    """
    if engine:
        return sorted(resolve_all(stack.for_engine(engine), kind, name, baseline).items())
    rows: List[Tuple[str, Resolution]] = []
    for fname, r in sorted(resolve_all(stack, kind, name, baseline).items()):
        if not r.needs_engine:
            rows.append((fname, r))
            continue
        for eng in ENGINES:
            bound = stack.for_engine(eng)
            per = (bound.resolve_joint(name) if kind == "joint" else bound.resolve_link(name)).get(fname)
            if per is not None and per.value is not None:
                rows.append((fname, per))
    return rows


def _cmd_resolve(args: Any) -> int:
    stack = load_joint_overlays(args.overlay)
    engine = args.engine or ""
    if engine and engine not in ENGINES:
        raise SchemaError(f"--engine: expected one of {list(ENGINES)}, got {engine!r}")
    baseline = read_urdf_baseline(args.urdf) if args.urdf else {}
    targets: List[Tuple[str, str]] = []
    if args.joint:
        targets += [("joint", n) for n in args.joint]
    if args.link:
        targets += [("link", n) for n in args.link]
    if not targets:
        targets = [("joint", n) for n in stack.joint_names()] + [("link", n) for n in stack.link_names()]

    payload: Dict[str, Any] = {}
    for kind, name in targets:
        rows = _resolve_rows(stack, kind, name, baseline.get((kind, name)), engine)
        if args.json:
            # A per-engine field expands to several rows under one field
            # name, so the JSON value is a list, not a single object.
            entry: Dict[str, Any] = {}
            for f, r in rows:
                entry.setdefault(f, []).append(
                    {
                        "value": r.value,
                        "source": r.source,
                        "channel": r.channel,
                        "engine": r.engine,
                        "frozen_to_cad": r.frozen_to_cad,
                        "unconsumed": r.unconsumed,
                        "needs_engine": r.needs_engine,
                    }
                )
            payload[f"{kind}:{name}"] = {f: v[0] if len(v) == 1 else v for f, v in entry.items()}
        else:
            print(f"{kind} {name}" + (f"  [engine {engine}]" if engine else ""))
            if not rows:
                print("  (no fields — not in any overlay, and no baseline URDF given)")
            for f, r in rows:
                flags = "".join(
                    (
                        f"  [{r.engine}]" if r.engine else "",
                        # ``needs_engine`` covers two different situations
                        # and the label has to tell them apart: no engine
                        # was selected, or one was and the overlay carries
                        # no branch for it (never falls back — I9).
                        (
                            ""
                            if not r.needs_engine
                            else f"  [NO {engine} BRANCH]" if engine else "  [NEEDS --engine]"
                        ),
                        "  [frozen_to_cad]" if r.frozen_to_cad else "",
                        "  [NO CONSUMER]" if r.unconsumed else "",
                    )
                )
                print(f"  {f:<16} {_fmt(r.value):>18}  <- {r.source:<28} [{r.channel}]{flags}")
            print()
    if args.json:
        print(json.dumps(payload, indent=2))
    return 0


def _cmd_diff(args: Any) -> int:
    """Baseline vs overlaid, per field. Measures drift; needs no patch."""
    stack = _bind_engine(load_joint_overlays(args.overlay), args)
    baseline = read_urdf_baseline(args.urdf)
    rows: List[Tuple[str, str, str, float, float, float]] = []
    for (kind, name), base_vals in sorted(baseline.items()):
        winners = stack.resolve_joint(name) if kind == "joint" else stack.resolve_link(name)
        for fname, r in sorted(winners.items()):
            if r.value is None:
                continue
            old = base_vals.get(fname)
            if old is None:
                rows.append((kind, name, fname, float("nan"), r.value, float("nan")))
                continue
            pct = float("inf") if old == 0.0 else (r.value - old) / abs(old) * 100.0
            rows.append((kind, name, fname, old, r.value, pct))
    if args.json:
        print(
            json.dumps(
                [
                    {"kind": k, "name": n, "field": f, "baseline": b, "overlaid": o, "pct": p}
                    for k, n, f, b, o, p in rows
                ],
                indent=2,
            )
        )
        return 0
    print(f"{'kind':<5} {'name':<22} {'field':<16} {'baseline':>16} {'overlaid':>16} {'change':>10}")
    for k, n, f, b, o, p in rows:
        pct = "new" if b != b else ("inf" if p == float("inf") else f"{p:+.2f}%")
        base = "(absent)" if b != b else _fmt(b)
        print(f"{k:<5} {n:<22} {f:<16} {base:>16} {_fmt(o):>16} {pct:>10}")
    print(f"\n{len(rows)} field(s) changed by overlay stack {list(stack.ids)}")
    return 0


def _pct(frac: float) -> str:
    """A fraction as a percentage, never rounded to a misleading ``0.0%``."""
    pct = frac * 100.0
    if pct and pct < 0.05:
        return "<0.1%"
    return f"{pct:.1f}%"


def _cmd_convert(args: Any) -> int:
    """Map a mujoco/newton set onto another backend, and price the map.

    Exits nonzero when any joint is ``blocked``, or when the source set
    never recorded which control law it was fitted against, so a re-test
    run can gate on "this set cannot be transferred" instead of reading it
    off a table.

    Per-joint notes are grouped rather than repeated: the clamp caveat is
    the same sentence on every folded joint, and twelve copies of it bury
    the one line that differs.
    """
    stack = load_joint_overlays(args.overlay)
    baseline = read_urdf_baseline(args.urdf) if args.urdf else None
    followers = read_urdf_mimic_followers(args.urdf) if args.urdf else frozenset()
    conv = convert_to_physx(
        stack, from_engine=args.from_engine, baseline=baseline, mimic_followers=followers
    )
    if args.json:
        print(json.dumps(conv.to_dict(), indent=2))
        return 1 if (conv.blocked or conv.untransferable) else 0

    src_engine = args.from_engine or "flat (one value for every engine)"
    plane = ", ".join(conv.source_planes) or "n/a (this set carries no servo gain)"
    print(f"{' '.join(conv.ids)}  ->  physx   branch: {src_engine}")
    print(f"identified on: {plane}")
    print("kp/kd/max_effort are the Layer-2 runtime tensor values; armature is physxJoint:armature.")
    if conv.untransferable:
        print(
            "\nUNTRANSFERABLE: no contributing overlay records which control law its gains\n"
            "  were fitted against. Everything below is arithmetic on numbers of unknown\n"
            "  provenance -- set meta.identified_on_engine, or refit on PhysX."
        )
    if not args.urdf:
        print(
            "NOTE: without --urdf, kinematic mimic followers cannot be identified. Those\n"
            "  joints have no PhysX DriveAPI to receive a gain, so the statuses below are an\n"
            "  upper bound on what transfers -- pass --urdf for the verdict."
        )
    print()

    head = ("joint", "kp", "kd", "max_effort", "armature", "status")
    body = []
    for r in conv.rows:
        # A mimic follower has no DriveAPI, so printing its kp/kd/maxForce --
        # even correctly summed -- would name three sinks that do not exist.
        # ``armature`` is a joint attribute and does still land.
        if any(k == "blocked-no-drive" for k, _ in r.notes):
            body.append((r.joint, "(no drive)", "(no drive)", "(no drive)", _fmt(r.armature), r.status))
            continue
        kd = _fmt(r.kd)
        if len(r.kd_terms) > 1:
            kd = " + ".join(_fmt(v) for _, v in r.kd_terms) + f" = {kd}"
        body.append((r.joint, _fmt(r.kp), kd, _fmt(r.max_effort), _fmt(r.armature), r.status))
    w = [max(len(row[i]) for row in (head,) + tuple(body)) for i in range(len(head))]
    print("  ".join(h.ljust(w[i]) if i == 0 else h.rjust(w[i]) for i, h in enumerate(head)))
    for row in body:
        print("  ".join(c.ljust(w[i]) if i in (0, 5) else c.rjust(w[i]) for i, c in enumerate(row)))

    counts = conv.counts()
    print(f"\n{len(conv.rows)} joint(s): " + ", ".join(f"{counts[s]} {s}" for s in CONVERT_STATUS_RANK if counts[s]))

    dropped = [r for r in conv.rows if r.frictionloss_dropped]
    if dropped:
        print(
            f"\nfrictionloss has no PhysX sink: {_fmt(conv.total_dropped)} N*m dropped "
            f"across {len(dropped)} joint(s)"
        )
        by_value: Dict[float, List[PhysxJointGains]] = {}
        for r in dropped:
            by_value.setdefault(r.frictionloss_dropped, []).append(r)
        for value, group in sorted(by_value.items(), reverse=True):
            fracs = [r.dropped_fraction for r in group if r.dropped_fraction is not None]
            if fracs:
                span = _pct(min(fracs)) if min(fracs) == max(fracs) else f"{_pct(min(fracs))}..{_pct(max(fracs))}"
                srcs = {r.effort_ref_source for r in group if r.dropped_fraction is not None}
                if srcs == {"max_effort"}:
                    ref_name = "max_effort"
                elif srcs == {"urdf-effort"}:
                    ref_name = "the URDF effort limit"
                else:
                    ref_name = "the per-joint torque cap"
                scale = f"{span} of {ref_name}"
                if len(fracs) < len(group):
                    scale += f" on {len(fracs)} of them; no cap known for the rest"
            else:
                scale = "no max_effort or URDF effort limit to scale it against"
            # One joint per value is the common shape on a per-joint set, and a
            # count column plus a wrapped name line reads as padding there.
            if len(group) == 1:
                print(f"  {_fmt(value):>18} N*m  {group[0].joint:<22} {scale}")
                continue
            names = ", ".join(r.joint for r in group[:3]) + (", ..." if len(group) > 3 else "")
            print(f"  {_fmt(value):>18} N*m  x{len(group):<3} {scale}")
            print(f"  {'':>18}       {names}")
        print(
            "  A PhysX value for frictionloss is a refit on the PhysX plane (I9), never\n"
            "  arithmetic on these numbers: Coulomb torque is constant in magnitude and a\n"
            "  viscous gain is proportional to speed, so they agree at one speed only."
        )

    folded = [r for r in conv.rows if any(k == "clamp" for k, _ in r.notes)]
    if folded:
        print(
            f"\nkd folds passive damping into drive damping on {len(folded)} joint(s). Exact: "
            "PhysX damps\n  velocity error and the velocity target is pinned to zero, so "
            "d*(0 - qd) is the -kd*qd\n  and -c*qd that MuJoCo applies separately. The folded "
            "term does come under the\n  maxForce clamp, where MuJoCo leaves dof_damping "
            "unclamped -- they differ only\n  while the drive saturates."
        )

    if conv.blocked:
        print(f"\nBLOCKED -- {len(conv.blocked)} joint(s) have no PhysX sink for what was identified:")
        jw = max(len(r.joint) for r in conv.blocked)
        # Group by reason: the two blocking shapes need different fixes, and a
        # single merged list reads as one problem.
        by_reason: Dict[str, List[PhysxJointGains]] = {}
        for r in conv.blocked:
            reason = next((t for k, t in r.notes if k.startswith("blocked")), "unknown")
            by_reason.setdefault(reason, []).append(r)
        for reason, group in by_reason.items():
            wrapped = textwrap.fill(reason, width=76, initial_indent="  ", subsequent_indent="    ")
            print(f"\n  {len(group)} joint(s):\n{wrapped}")
            for r in group:
                kd = f"  damping {_fmt(r.kd)}" if r.kd is not None else ""
                fl = f"  frictionloss {_fmt(r.frictionloss_dropped)}" if r.frictionloss_dropped else ""
                print(f"    {r.joint:<{jw}}{kd}{fl}")
        print("\n  This set needs a PhysX-side identification run.")
        return 1
    return 1 if conv.untransferable else 0


def _cmd_bilateral(args: Any) -> int:
    """Left/right comparison.

    This is what buys back the ergonomics of an ``arms: {l: .., r: ..}``
    nesting without a second key space — I4 requires one flat URDF-name
    key space, so bilateral symmetry is a *report*, not a file layout.
    """
    stack = _bind_engine(load_joint_overlays(args.overlay), args)
    pairs: List[Tuple[str, str, str, Optional[float], Optional[float]]] = []
    for kind, names in (("joint", stack.joint_names()), ("link", stack.link_names())):
        # Mirror through the idx-stripped name: URDF joint names carry a
        # per-side ``idxNN_`` prefix (left 21..27, right 61..67 on G2), so a
        # literal ``_l_`` -> ``_r_`` substitution yields a name that does not
        # exist. Link names carry no prefix. Build the lookup from the
        # overlay's own keys so this needs no URDF.
        by_stripped = {_strip_idx(n).replace("_l_", "_*_", 1).replace("_r_", "_*_", 1): n for n in names}
        for name in names:
            if "_l_" not in name:
                continue
            mirror_key = _strip_idx(name).replace("_l_", "_*_", 1)
            mirror = by_stripped.get(mirror_key.replace("_*_", "_r_", 1))
            if mirror is None:
                # the overlay's own keys, stripped, mirrored
                cand = _strip_idx(name).replace("_l_", "_r_", 1)
                mirror = next((n for n in names if _strip_idx(n) == cand), None)
            if mirror is None:
                mirror = name.replace("_l_", "_r_", 1)
            left = stack.resolve_joint(name) if kind == "joint" else stack.resolve_link(name)
            right = stack.resolve_joint(mirror) if kind == "joint" else stack.resolve_link(mirror)
            for f in sorted(set(left) | set(right)):
                lv = left[f].value if f in left else None
                rv = right[f].value if f in right else None
                pairs.append((kind, _strip_idx(name).replace("_l_", "_*_", 1), f, lv, rv))
    if args.json:
        print(
            json.dumps(
                [{"kind": k, "name": n, "field": f, "left": l, "right": r} for k, n, f, l, r in pairs],
                indent=2,
            )
        )
        return 0
    print(f"{'name':<24} {'field':<16} {'left':>16} {'right':>16} {'L/R':>9}")
    for _k, n, f, lv, rv in pairs:
        if lv is None or rv is None:
            ratio = "MISSING"
        elif rv == 0.0:
            ratio = "inf" if lv != 0.0 else "1.000"
        else:
            ratio = f"{lv / rv:.3f}"
        print(f"{n:<24} {f:<16} {_fmt(lv):>16} {_fmt(rv):>16} {ratio:>9}")
    missing = sum(1 for *_x, lv, rv in pairs if lv is None or rv is None)
    print(f"\n{len(pairs)} paired field(s); {missing} present on one side only")
    return 0


def _cmd_emit_schema(args: Any) -> int:
    text = json.dumps(json_schema(), indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m genie_sim_engine_core.schema.joint_params",
        description="Inspect and validate joint parameter overlays.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _add(name: str, help_: str, fn: Any) -> Any:
        s = sub.add_parser(name, help=help_)
        s.add_argument("--json", action="store_true", help="machine-readable output")
        s.set_defaults(fn=fn)
        return s

    s = _add("validate", "schema check; with --urdf also check every name (I2)", _cmd_validate)
    s.add_argument("overlay", nargs="+")
    s.add_argument("--urdf")

    engine_help = (
        f"resolve for one backend ({'|'.join(ENGINES)}); only affects the four "
        f"control-law fields {sorted(ENGINE_SCOPED_FIELDS)}"
    )

    s = _add("resolve", "print every field with the layer that won", _cmd_resolve)
    s.add_argument("--overlay", action="append", default=[])
    s.add_argument("--urdf")
    s.add_argument("--joint", action="append", default=[])
    s.add_argument("--link", action="append", default=[])
    s.add_argument("--engine", default="", choices=("",) + ENGINES, help=engine_help)

    s = _add("diff", "baseline vs overlaid, per field", _cmd_diff)
    s.add_argument("--overlay", action="append", default=[], required=True)
    s.add_argument("--urdf", required=True)
    s.add_argument("--engine", default="", choices=("",) + ENGINES, help=engine_help)

    s = _add("bilateral", "left/right symmetry report", _cmd_bilateral)
    s.add_argument("--overlay", action="append", default=[], required=True)
    s.add_argument("--engine", default="", choices=("",) + ENGINES, help=engine_help)

    s = _add("convert", "map a mujoco/newton set onto another backend", _cmd_convert)
    s.add_argument("--overlay", action="append", default=[], required=True)
    s.add_argument(
        "--to",
        required=True,
        choices=CONVERSION_TARGETS,
        help="target backend; only physx needs a conversion",
    )
    s.add_argument(
        "--from",
        dest="from_engine",
        default="",
        choices=("",) + ENGINES,
        help="engine the set was identified on; required if any field is split by engine",
    )
    s.add_argument("--urdf", help="supplies the effort limit the dropped frictionloss is scaled against")

    s = _add("emit-schema", "write the JSON Schema", _cmd_emit_schema)
    s.add_argument("--out")

    args = ap.parse_args(argv)
    try:
        return int(args.fn(args))
    except SchemaError as err:
        print(f"error: {err}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "BASELINE",
    "CHANNEL_A",
    "CHANNEL_AB",
    "CHANNEL_B",
    "ENGINES",
    "CONVERSION_TARGETS",
    "CONVERT_STATUS_RANK",
    "ENGINE_SCOPED_FIELDS",
    "INERTIA_FIELDS",
    "JOINT_FIELD_CHANNEL",
    "JOINT_PARAMS_SCHEMA_ID",
    "LINK_SCALAR_FIELDS",
    "LINK_VEC3_FIELDS",
    "OVERLAY_KINDS",
    "PHYSX_FIELD_MAP",
    "UNCONSUMED_FIELDS",
    "FieldValue",
    "JointOverlay",
    "JointParamOverlay",
    "LinkOverlay",
    "OverlayMeta",
    "OverlayStack",
    "PhysxConversion",
    "PhysxJointGains",
    "Resolution",
    "Scalar",
    "UrdfNames",
    "convert_to_physx",
    "json_schema",
    "load_joint_overlays",
    "main",
    "read_urdf_baseline",
    "read_urdf_mimic_followers",
    "resolve_all",
    "read_urdf_names",
    "validate",
    "validate_against_urdf",
]
