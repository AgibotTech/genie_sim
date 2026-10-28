# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``scene_*.yaml`` schema — the human-authored scene contract.

Every downstream package (bringup launchers, ``genie_sim_engine_core``
runtime session, mujoco/isaac/newton assemblers, render nodes) reads the
same YAML file. Before this module existed, each of them re-parsed the
same nested dicts with ``get(...)`` chains and silently disagreed on
defaults / key spellings — the two field-drift bugs that motivated this
schema (``_write_manifest`` shipping the nested camera shape instead of
the flat one; ``render_layer_usda`` hardcoded to empty) were both
manifestations of that drift.

The parser is deliberately **tolerant**:

* Missing optional blocks yield empty typed sub-objects, not exceptions
  — humans forget to author sections that don't apply.
* Unknown top-level keys are collected in ``extra`` so callers who
  own an engine-specific sub-block (``newton``, ``imu``,
  ``characters``) can pull their raw slice out and parse it themselves.
* Bool coercion accepts YAML strings (``"false"``, ``"no"``, ``"off"``)
  — matches ``common/scene_config._parse_bool`` behaviour so we don't
  regress anyone's yaml that spells booleans as strings.

The strict counterpart is :class:`~genie_sim_engine_core.schema.manifest.Manifest`:
manifests are machine-emitted and drift there means a schema-version
mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from ._common import (
    _get_bool,
    _get_float,
    _get_int,
    _get_opt_path,
    _get_str,
    _list_or_empty,
    _mapping_or_empty,
    _require_mapping,
    _wxyz,
    _xyz,
)
from .errors import SchemaError

# -----------------------------------------------------------------------
# Camera / lidar / pose blocks
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """Local pose written as YAML ``xyz`` + ``wxyz`` (unit quaternion, w-first).

    ``xyz`` in metres, ``wxyz`` normalized. USD / Isaac graphics convention
    (camera looks -Z, +Y up). Missing / malformed values fall back to
    identity — same behaviour every downstream helper used inline before.
    """

    xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "Pose":
        raw = _mapping_or_empty(raw)
        xyz = _xyz(raw) or (0.0, 0.0, 0.0)
        wxyz = _wxyz(raw) or (1.0, 0.0, 0.0, 0.0)
        return cls(xyz=xyz, wxyz=wxyz)

    def to_dict(self) -> Dict[str, Any]:
        return {"xyz": list(self.xyz), "wxyz": list(self.wxyz)}

    def is_identity(self) -> bool:
        return self.xyz == (0.0, 0.0, 0.0) and self.wxyz == (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CameraSensor:
    """``cameras[].sensor`` block."""

    width: int = 1280
    height: int = 800
    min_range: float = 0.01
    max_range: float = 10000.0
    freq: Optional[float] = None  # optional; some scenes throttle per-cam

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "CameraSensor":
        raw = _mapping_or_empty(raw)
        freq = raw.get("freq")
        return cls(
            width=_get_int(raw, "width", 1280),
            height=_get_int(raw, "height", 800),
            min_range=_get_float(raw, "min_range", 0.01),
            max_range=_get_float(raw, "max_range", 10000.0),
            freq=float(freq) if freq is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "width": self.width,
            "height": self.height,
            "min_range": self.min_range,
            "max_range": self.max_range,
        }
        if self.freq is not None:
            out["freq"] = self.freq
        return out


@dataclass(frozen=True)
class CameraIntrinsic:
    """``cameras[].intrinsic`` block.

    ``p1``/``p2`` are only meaningful for plumb_bob. The full set is
    carried in the flat manifest because the render node's
    :class:`CameraManifest` mirrors this exactly and needs every knob
    per camera.

    ``k4``/``k5``/``k6`` are the DENOMINATOR of OpenCV's rational model,
        r_d = r_u (1 + k1 u + k2 u^2 + k3 u^3) / (1 + k4 u + k5 u^2 + k6 u^3)
    which is what the head stereo lens calibrates as. They must all round-trip
    through :meth:`to_dict` -- the assemblers hand that dict straight to
    ``render_layer.author_opencv_pinhole``, and a missing denominator term is
    not a small error: for the head stereo lens, truncating after ``k4`` turns
    the ratio at the image corner from 0.28 (barrel) into 7.4 (pincushion), so
    the rendered distortion comes out bowing the wrong way. ``k4`` alone also
    doubles as OpenCV *fisheye*'s fourth radial term, where ``k5``/``k6``
    are unused and stay 0.
    """

    fx: float = 610.0
    fy: float = 610.0
    cx: float = 640.0
    cy: float = 400.0
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "CameraIntrinsic":
        raw = _mapping_or_empty(raw)
        return cls(
            fx=_get_float(raw, "fx", 610.0),
            fy=_get_float(raw, "fy", 610.0),
            cx=_get_float(raw, "cx", 640.0),
            cy=_get_float(raw, "cy", 400.0),
            k1=_get_float(raw, "k1", 0.0),
            k2=_get_float(raw, "k2", 0.0),
            p1=_get_float(raw, "p1", 0.0),
            p2=_get_float(raw, "p2", 0.0),
            k3=_get_float(raw, "k3", 0.0),
            k4=_get_float(raw, "k4", 0.0),
            k5=_get_float(raw, "k5", 0.0),
            k6=_get_float(raw, "k6", 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,
            "k3": self.k3,
            "k4": self.k4,
            "k5": self.k5,
            "k6": self.k6,
        }


@dataclass(frozen=True)
class CameraTopic:
    """``cameras[].topic`` block — ROS side of the pub graph.

    Both fields are strings; empty ``depth`` means the camera doesn't
    publish depth. DDS-side mappings live on :class:`CameraEntry`
    directly (they're flat scalars in the YAML, not nested under
    ``topic:``).
    """

    rgb: str = ""
    depth: str = ""

    @classmethod
    def from_dict(cls, raw: Optional[Any]) -> "CameraTopic":
        if isinstance(raw, str):
            return cls(rgb=raw)
        raw = _mapping_or_empty(raw)
        return cls(rgb=_get_str(raw, "rgb"), depth=_get_str(raw, "depth"))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"rgb": self.rgb}
        if self.depth:
            out["depth"] = self.depth
        return out


@dataclass(frozen=True)
class CameraEntry:
    """One ``cameras[]`` entry from the scene YAML.

    Mirrors the shape both the Isaac and MuJoCo assemblers consume.
    Extrinsic is optional: cameras without an explicit local pose track
    the parent body's TF at runtime (see
    ``render_node.cpp::cam_body_to_render_path_``).

    ``camera_group`` gates the whole entry against the global
    ``camera_mode`` tier
    (``off``/``cam3``/``cam4``/``cam6``/``full``, ranked in that
    order). ``depth_group`` optionally gates just the
    ``topic.depth``/``dds_depth_topic`` sub-stream at a *higher* tier
    than the entry's own ``camera_group`` — for an RGBD entry whose
    color feed should stay on at a low tier while its depth feed only
    turns on at a higher one. Empty ``depth_group`` means "inherit
    ``camera_group``".
    """

    prim_path: str
    frame_id: str = ""
    sensor: CameraSensor = field(default_factory=CameraSensor)
    intrinsic: CameraIntrinsic = field(default_factory=CameraIntrinsic)
    topic: CameraTopic = field(default_factory=CameraTopic)
    extrinsic: Pose = field(default_factory=Pose)
    enabled: bool = True
    camera_group: str = "cam3"
    depth_group: str = ""
    model: str = ""
    dds_topic: str = ""
    dds_depth_topic: str = ""
    depth_scale: Optional[float] = None
    max_fov_deg: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CameraEntry":
        raw = _require_mapping(raw, where="cameras[]")
        prim_path = _get_str(raw, "prim_path") or _get_str(raw, "path")
        if not prim_path:
            raise SchemaError("cameras[]: missing 'prim_path'")
        depth_scale = raw.get("depth_scale")
        # Total field of view of a >180 deg fisheye, in degrees. ``mask_fov_deg``
        # is the legacy spelling: it used to drive only the rmagine post-process
        # mask, but it is the same physical quantity the OVRTX lens schema needs
        # for ``maxFov``, so both spellings are accepted.
        max_fov_deg = raw.get("max_fov_deg", raw.get("mask_fov_deg"))
        return cls(
            prim_path=prim_path,
            frame_id=_get_str(raw, "frame_id"),
            sensor=CameraSensor.from_dict(raw.get("sensor")),
            intrinsic=CameraIntrinsic.from_dict(raw.get("intrinsic")),
            topic=CameraTopic.from_dict(raw.get("topic")),
            extrinsic=Pose.from_dict(raw.get("extrinsic")),
            enabled=bool(raw.get("enabled", True)),
            camera_group=(_get_str(raw, "camera_group") or "cam3"),
            depth_group=_get_str(raw, "depth_group"),
            model=_get_str(raw, "model").strip(),
            dds_topic=_get_str(raw, "dds_topic"),
            dds_depth_topic=_get_str(raw, "dds_depth_topic"),
            depth_scale=float(depth_scale) if depth_scale is not None else None,
            max_fov_deg=float(max_fov_deg) if max_fov_deg is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "prim_path": self.prim_path,
            "frame_id": self.frame_id,
            "sensor": self.sensor.to_dict(),
            "intrinsic": self.intrinsic.to_dict(),
            "topic": self.topic.to_dict(),
        }
        if not self.extrinsic.is_identity():
            out["extrinsic"] = self.extrinsic.to_dict()
        if not self.enabled:
            out["enabled"] = False
        if self.camera_group != "cam3":
            out["camera_group"] = self.camera_group
        if self.depth_group:
            out["depth_group"] = self.depth_group
        if self.model:
            out["model"] = self.model
        if self.dds_topic:
            out["dds_topic"] = self.dds_topic
        if self.dds_depth_topic:
            out["dds_depth_topic"] = self.dds_depth_topic
        if self.depth_scale is not None:
            out["depth_scale"] = self.depth_scale
        if self.max_fov_deg is not None:
            out["max_fov_deg"] = self.max_fov_deg
        return out


# Global ``camera_mode`` tiers, ranked low → high. Both
# assemblers (mujoco's ``_assemble.py``, isaacsim's ``_scene_impl.py``)
# duplicate this table rather than importing it — the isaacsim side
# filters plain dicts read straight from YAML, before any CameraEntry
# is constructed, so there is no shared object to hang a method off.
# Keep the two tables in sync by hand when a tier is added/renamed.
CAMERA_MODE_RANK: Dict[str, int] = {
    "off": 0,
    "cam3": 1,
    "cam4": 2,
    "cam6": 3,
    "full": 4,
}


def camera_mode_rank(camera_mode: str) -> int:
    """Rank of a ``camera_mode``/``camera_group`` tier name.

    Unknown names rank as ``full`` (the highest defined tier). That
    default is deliberately asymmetric depending on which side of the
    comparison ``entry_included = camera_mode_rank(mode) >=
    camera_mode_rank(entry.camera_group)`` it lands on: a malformed
    *global* mode (e.g. a typo'd ``camera_mode`` scene-YAML field)
    ranks high and so shows every entry (fails open — better to render
    an unexpected camera than silently drop one), while a malformed
    *entry* tier ranks high and so only shows under the ``full`` mode
    (fails closed — a typo'd ``camera_group`` shouldn't leak the
    camera into a leaner tier by accident).
    """
    return CAMERA_MODE_RANK.get(camera_mode, CAMERA_MODE_RANK["full"])


@dataclass(frozen=True)
class LidarSensor:
    """``lidars[].sensor`` — livox-style rotating lidar params."""

    num_cols: int = 400
    num_rows: int = 96
    min_elevation: float = -7.0
    max_elevation: float = 52.0
    min_range: float = 0.05
    max_range: float = 10000.0
    vert_offset: float = 0.0
    inner_vert: bool = False
    frequency: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "LidarSensor":
        raw = _mapping_or_empty(raw)
        frequency = raw.get("frequency")
        return cls(
            num_cols=_get_int(raw, "num_cols", 400),
            num_rows=_get_int(raw, "num_rows", 96),
            min_elevation=_get_float(raw, "min_elevation", -7.0),
            max_elevation=_get_float(raw, "max_elevation", 52.0),
            min_range=_get_float(raw, "min_range", 0.05),
            max_range=_get_float(raw, "max_range", 10000.0),
            vert_offset=_get_float(raw, "vert_offset", 0.0),
            inner_vert=_get_bool(raw, "inner_vert", False),
            frequency=float(frequency) if frequency is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "num_cols": self.num_cols,
            "num_rows": self.num_rows,
            "min_elevation": self.min_elevation,
            "max_elevation": self.max_elevation,
            "min_range": self.min_range,
            "max_range": self.max_range,
            "vert_offset": self.vert_offset,
            "inner_vert": self.inner_vert,
        }
        if self.frequency is not None:
            out["frequency"] = self.frequency
        return out


@dataclass(frozen=True)
class LidarEntry:
    """One ``lidars[]`` entry from the scene YAML."""

    prim_path: str
    topic: str = ""
    frame_id: str = "base_link"
    sensor: LidarSensor = field(default_factory=LidarSensor)
    extrinsic: Pose = field(default_factory=Pose)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LidarEntry":
        raw = _require_mapping(raw, where="lidars[]")
        prim_path = _get_str(raw, "prim_path") or _get_str(raw, "path")
        if not prim_path:
            raise SchemaError("lidars[]: missing 'prim_path'")
        return cls(
            prim_path=prim_path,
            topic=_get_str(raw, "topic"),
            frame_id=_get_str(raw, "frame_id", "base_link"),
            sensor=LidarSensor.from_dict(raw.get("sensor")),
            extrinsic=Pose.from_dict(raw.get("extrinsic")),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "prim_path": self.prim_path,
            "topic": self.topic,
            "frame_id": self.frame_id,
            "sensor": self.sensor.to_dict(),
        }
        if not self.extrinsic.is_identity():
            out["extrinsic"] = self.extrinsic.to_dict()
        return out


# -----------------------------------------------------------------------
# Robot source (discriminated union: urdf | prebaked | usd)
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class UrdfSource:
    """``robot.robot_source.urdf`` block.

    Fields are open-ended (per-scene xacro args land here as arbitrary
    keys); we keep the raw mapping in ``extra`` so downstream xacro
    invocations still see everything.
    """

    xacro_entry: str = ""
    lgripper: str = ""
    rgripper: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "UrdfSource":
        raw = _mapping_or_empty(raw)
        known = {"xacro_entry", "lgripper", "rgripper"}
        extra = {k: v for k, v in raw.items() if k not in known}
        return cls(
            xacro_entry=_get_str(raw, "xacro_entry"),
            lgripper=_get_str(raw, "lgripper"),
            rgripper=_get_str(raw, "rgripper"),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.xacro_entry:
            out["xacro_entry"] = self.xacro_entry
        if self.lgripper:
            out["lgripper"] = self.lgripper
        if self.rgripper:
            out["rgripper"] = self.rgripper
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class PrebakedSource:
    """``robot.robot_source.prebaked`` — presence opts out of URDF→USD assemble.

    Paths are stored as raw strings (resolved against the asset root
    downstream); ``pipeline`` is the launcher's ``prebaked.pipeline``
    hook — same shape as ``launcher.physics.engines.<engine>``.
    """

    usda: str = ""
    mjcf_robot: str = ""
    mjcf_scene: str = ""
    pipeline: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "PrebakedSource":
        raw = _mapping_or_empty(raw)
        return cls(
            usda=_get_str(raw, "usda"),
            mjcf_robot=_get_str(raw, "mjcf_robot"),
            mjcf_scene=_get_str(raw, "mjcf_scene"),
            pipeline=dict(_mapping_or_empty(raw.get("pipeline"))),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.usda:
            out["usda"] = self.usda
        if self.mjcf_robot:
            out["mjcf_robot"] = self.mjcf_robot
        if self.mjcf_scene:
            out["mjcf_scene"] = self.mjcf_scene
        if self.pipeline:
            out["pipeline"] = dict(self.pipeline)
        return out

    def is_empty(self) -> bool:
        return not (self.usda or self.mjcf_robot or self.mjcf_scene or self.pipeline)


# -----------------------------------------------------------------------
# Base-pin mode
# -----------------------------------------------------------------------


#: Accepted ``robot_source.pin_base_mode`` spellings (``""`` = use the default).
PIN_BASE_MODE_VALUES = frozenset({"", "weld", "mocap"})


class PinBaseMode(str, Enum):
    """How the robot base is held — the resolved pairing of
    ``robot_source.pin_base_to_world`` and ``robot_source.pin_base_mode``.

    * ``weld``  — compile-time world weld; the default when pinned.
    * ``mocap`` — keep the root FREE joint and clamp it per substep to a
      settable reference, so a reset can re-place the base.  Needs an
      adapter that implements the clamp (``SolverAdapter.supports_base_pin``).
    * ``free``  — unpinned; the base is under ordinary physics.

    ``pin_base_to_world: false`` outranks any mode request, and ``mocap``
    is only ever selected by an explicit request — nothing else switches it
    on.  Worth knowing when reading a scene yaml: under ``mocap`` a true
    ``pin_base_to_world`` does NOT mean a FixedJoint exists — "pinned"
    means "held", not "welded".

    Subclasses ``str`` so ``mode == "weld"``, ``f"{mode}"`` and
    ``json.dumps`` keep behaving exactly like the bare strings this
    replaced.  ``enum.StrEnum`` would say the same thing but needs Python
    3.11; the ``__str__`` assignment below is the 3.10-compatible
    equivalent from the enum HOWTO, and without it ``f"{mode}"`` prints
    ``PinBaseMode.WELD``.
    """

    WELD = "weld"
    MOCAP = "mocap"
    FREE = "free"

    __str__ = str.__str__

    # -- construction ---------------------------------------------------

    @classmethod
    def resolve(cls, pin_base_to_world: bool, explicit_mode: str = "") -> "PinBaseMode":
        """The one rule that turns the two config fields into a mode.

        An unrecognised ``explicit_mode`` falls back to ``weld`` rather
        than raising: the config boundary (``RobotSource.from_dict``)
        rejects junk, but engines can also be constructed with direct
        kwargs that never pass through it, and the pre-mocap behaviour is
        the safe thing to land on.
        """
        if not pin_base_to_world:
            return cls.FREE
        m = (explicit_mode or "").strip().lower()
        return cls(m) if m in ("weld", "mocap") else cls.WELD

    @classmethod
    def of(cls, owner: Any) -> "PinBaseMode":
        """The resolved mode of an engine-like object.

        The single home for the defensive fallback that construction
        paths predating ``_pin_base_mode`` rely on — consumers ask here
        instead of each re-deriving it.
        """
        m = getattr(owner, "_pin_base_mode", None)
        if m is not None:
            try:
                return cls(str(m))
            except ValueError:
                pass
        return cls.resolve(bool(getattr(owner, "_pin_base_to_world", False)))

    # -- what each consumer actually needs to know ----------------------

    @property
    def keeps_root_weld(self) -> bool:
        """The URDF ``root_joint`` weld stays active.

        One question, three consumers: the stage keeps the joint instead
        of deactivating it, its anchor is worth repairing, and the substep
        can use kinematic control instead of real gravity — all because
        something rigid is holding the base.
        """
        return self is PinBaseMode.WELD

    @property
    def needs_base_clamp(self) -> bool:
        """The adapter must clamp the root FREE joint every substep."""
        return self is PinBaseMode.MOCAP

    @property
    def base_replaceable_at_reset(self) -> bool:
        """A reset can move the base (free-joint ``q``, or mocap offsets)."""
        return self is not PinBaseMode.WELD


@dataclass(frozen=True)
class RobotSource:
    """``robot.robot_source`` — resolved discriminant + raw config.

    The discriminant is the historical rule "presence of the ``urdf``
    key selects the URDF route" documented at
    ``_core_launch_utils.py::uses_genie_sim_robot_model`` — kept
    identical so no scene YAML has to change. ``kind`` values:

    * ``"urdf"``     — ``urdf`` key present (may be empty mapping).
    * ``"prebaked"`` — ``prebaked`` key present, ``urdf`` absent.
    * ``"usd"``      — ``usd`` key present, neither of the above.
    * ``""``         — none of the above; some caller must decide.
    """

    kind: str
    robot_model: str = ""
    body: str = ""
    arm: str = ""
    gripper: str = ""
    package: str = ""
    chassis: str = ""
    pin_base_to_world: bool = False
    #: "" | "weld" | "mocap" — see :class:`PinBaseMode`.  ``""`` takes the
    #: default; only meaningful when ``pin_base_to_world``.
    pin_base_mode: str = ""
    convert_joints_to_fixed: List[str] = field(default_factory=list)
    urdf: Optional[UrdfSource] = None
    prebaked: Optional[PrebakedSource] = None
    #: Every remaining key (usd, custom launcher extensions) — round-tripped verbatim.
    extra: Dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS = frozenset(
        {
            "robot_model",
            "body",
            "arm",
            "gripper",
            "package",
            "chassis",
            "pin_base_to_world",
            "pin_base_mode",
            "convert_joints_to_fixed",
            "urdf",
            "prebaked",
        }
    )

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "RobotSource":
        raw = _mapping_or_empty(raw)
        if "urdf" in raw:
            kind = "urdf"
        elif "prebaked" in raw:
            kind = "prebaked"
        elif "usd" in raw:
            kind = "usd"
        else:
            kind = ""
        # convert_joints_to_fixed accepts str or list; normalize to list[str]
        cj = raw.get("convert_joints_to_fixed")
        if cj is None:
            convert: List[str] = []
        elif isinstance(cj, str):
            convert = [cj.strip().lower()] if cj.strip() else []
        elif isinstance(cj, (list, tuple)):
            convert = [str(x).strip().lower() for x in cj if str(x).strip()]
        else:
            convert = []

        # pin_base_mode is an enum at the config boundary: normalize once
        # here and refuse junk, so no downstream consumer has to.  A
        # mode alongside ``pin_base_to_world: false`` is NOT refused —
        # the boolean outranks it (``PinBaseMode.resolve``) and the
        # engine logs the override; raising would break a config that
        # flips the boolean for a quick test with the mode line left in.
        pin_base_mode = _get_str(raw, "pin_base_mode").strip().lower()
        if pin_base_mode not in PIN_BASE_MODE_VALUES:
            raise SchemaError(
                f"robot_source.pin_base_mode must be one of "
                f"{sorted(v for v in PIN_BASE_MODE_VALUES if v)} or omitted, "
                f"got {_get_str(raw, 'pin_base_mode')!r}"
            )

        extra = {k: v for k, v in raw.items() if k not in cls._KNOWN_FIELDS}

        return cls(
            kind=kind,
            robot_model=_get_str(raw, "robot_model"),
            body=_get_str(raw, "body"),
            arm=_get_str(raw, "arm"),
            gripper=_get_str(raw, "gripper"),
            package=_get_str(raw, "package"),
            chassis=_get_str(raw, "chassis"),
            pin_base_to_world=_get_bool(raw, "pin_base_to_world", False),
            pin_base_mode=pin_base_mode,
            convert_joints_to_fixed=convert,
            urdf=UrdfSource.from_dict(raw.get("urdf")) if "urdf" in raw else None,
            prebaked=(PrebakedSource.from_dict(raw.get("prebaked")) if "prebaked" in raw else None),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.robot_model:
            out["robot_model"] = self.robot_model
        if self.body:
            out["body"] = self.body
        if self.arm:
            out["arm"] = self.arm
        if self.gripper:
            out["gripper"] = self.gripper
        if self.package:
            out["package"] = self.package
        if self.chassis:
            out["chassis"] = self.chassis
        if self.pin_base_to_world:
            out["pin_base_to_world"] = True
        if self.pin_base_mode:
            out["pin_base_mode"] = self.pin_base_mode
        if self.convert_joints_to_fixed:
            out["convert_joints_to_fixed"] = list(self.convert_joints_to_fixed)
        if self.urdf is not None:
            out["urdf"] = self.urdf.to_dict()
        if self.prebaked is not None:
            out["prebaked"] = self.prebaked.to_dict()
        out.update(self.extra)
        return out


# -----------------------------------------------------------------------
# Robot section
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class RobotSection:
    """``robot:`` block — the largest section of the scene YAML.

    ``init_joint_pos`` and ``init_joint_pos`` values are kept as raw
    dicts because each backend handles them differently (Isaac writes
    directly, MuJoCo forwards JSON-encoded to node params, Newton
    fills a builder). The typed fields cover the launcher / engine-
    startup path; the per-joint dict is opaque.

    ``init_base_pose2d`` / ``init_base_z`` used to be typed here and had
    zero consumers — every backend reads ``init_base_pose`` (x/y/z/theta).
    They were dropped rather than kept as dead API; a yaml that still
    sets them now lands them in ``extra`` and round-trips them verbatim,
    so nothing breaks and nothing pretends they are honoured.
    """

    robot_name: str = ""
    robot_prefix: str = ""
    robot_usd: str = ""
    robot_source: RobotSource = field(default_factory=lambda: RobotSource(kind=""))
    init_joint_pos: Dict[str, Any] = field(default_factory=dict)
    init_base_pose: Dict[str, Any] = field(default_factory=dict)
    #: Everything else — launcher-specific extensions. Round-tripped.
    extra: Dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS = frozenset(
        {
            "robot_name",
            "robot_prefix",
            "robot_usd",
            "robot_source",
            "init_joint_pos",
            "init_base_pose",
        }
    )

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "RobotSection":
        raw = _mapping_or_empty(raw)
        extra = {k: v for k, v in raw.items() if k not in cls._KNOWN_FIELDS}
        return cls(
            robot_name=_get_str(raw, "robot_name"),
            robot_prefix=_get_str(raw, "robot_prefix"),
            robot_usd=_get_str(raw, "robot_usd"),
            robot_source=RobotSource.from_dict(raw.get("robot_source")),
            init_joint_pos=dict(_mapping_or_empty(raw.get("init_joint_pos"))),
            init_base_pose=dict(_mapping_or_empty(raw.get("init_base_pose"))),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.robot_name:
            out["robot_name"] = self.robot_name
        if self.robot_prefix:
            out["robot_prefix"] = self.robot_prefix
        if self.robot_usd:
            out["robot_usd"] = self.robot_usd
        if self.robot_source.kind or self.robot_source.to_dict():
            out["robot_source"] = self.robot_source.to_dict()
        if self.init_joint_pos:
            out["init_joint_pos"] = dict(self.init_joint_pos)
        if self.init_base_pose:
            out["init_base_pose"] = dict(self.init_base_pose)
        out.update(self.extra)
        return out


# -----------------------------------------------------------------------
# Contact publish section
# -----------------------------------------------------------------------


CONTACT_SCOPE_VALUES = frozenset({"all", "robot_scene", "robot_only"})


@dataclass(frozen=True)
class ContactPublishSection:
    """``contact_publish:`` block — knobs for ``/contact_states``.

    Backends producing ``gazebo_msgs/ContactsState`` on the DDS-throttled
    publish tick honour these to keep the topic focused on what a
    manipulation harness actually reads.

    ``scope``
        Which contact pairs are emitted. Three values:

        * ``"all"`` — every solver contact this step, matching the
          Gazebo/Isaac contact-sensor convention.
        * ``"robot_scene"`` (default) — pairs where at least one body
          belongs to the robot. Membership uses ``manifest.robot``
          body paths (same source the ``/tf_scene`` split reads).
        * ``"robot_only"`` — pairs where *both* bodies are robot
          links. Only useful for self-collision debugging.

    ``include_self_contacts``
        When ``False`` (default), robot-vs-robot pairs are stripped
        even inside ``robot_scene`` — finger↔palm / arm↔torso contacts
        that MuJoCo emits at loose margins swamp the topic without
        adding signal for grasp checkers. ``robot_only`` scope
        implicitly overrides this; the caller has explicitly asked
        for self-contacts.

    Defaults are tuned for manipulation benchmarks (gripper↔object,
    arm↔shelf); flip to ``scope="all"`` for parity with Gazebo /
    Isaac topics.
    """

    scope: str = "robot_scene"
    include_self_contacts: bool = False

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "ContactPublishSection":
        raw = _mapping_or_empty(raw)
        scope = _get_str(raw, "scope", "robot_scene") or "robot_scene"
        if scope not in CONTACT_SCOPE_VALUES:
            raise SchemaError(f"contact_publish.scope must be one of {sorted(CONTACT_SCOPE_VALUES)}, " f"got {scope!r}")
        return cls(
            scope=scope,
            include_self_contacts=_get_bool(raw, "include_self_contacts", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "include_self_contacts": self.include_self_contacts,
        }


# -----------------------------------------------------------------------
# Root SceneConfig
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class SceneConfig:
    """Typed view of a ``scene_*.yaml`` file.

    Top-level structure:

    ::

      robot:            RobotSection
      scene:            "path/to/world.usda"   # optional; empty → default world
      cameras:          [CameraEntry, ...]
      lidars:           [LidarEntry, ...]
      viewer_camera:    dict   # GL viewer / FreeCam preset (opaque)
      free_cam_width:   int
      free_cam_height:  int
      newton:           dict   # engine-specific block; parsed by newton package
      imu:              dict   # engine-specific block; parsed by newton package
      characters:       list   # pedestrian sim; parsed by rmagine renderer
      assemble:         dict   # assemble-time overrides (isaacsim urdf_to_usd)

    Blocks that only one downstream engine cares about (``newton``,
    ``imu``, ``characters``, ``assemble``) are round-tripped as raw
    dicts / lists — that keeps this module's dep footprint down to
    ``pyyaml`` and pushes engine-specific parsing to the engine that
    owns it.
    """

    robot: RobotSection = field(default_factory=RobotSection)
    scene_usda: str = ""
    cameras: List[CameraEntry] = field(default_factory=list)
    lidars: List[LidarEntry] = field(default_factory=list)
    viewer_camera: Dict[str, Any] = field(default_factory=dict)
    free_cam_width: int = 1280
    free_cam_height: int = 720
    #: Global camera tier filter (``off``/``cam3``/``cam4``/``cam6``/``full``).
    #: The scene YAML is the single source for which tier a run enables;
    #: leave unset for the ``cam3`` default.
    camera_mode: str = "cam3"
    contact_publish: ContactPublishSection = field(default_factory=ContactPublishSection)
    #: Engine-owned raw blocks — round-tripped verbatim so downstream
    #: consumers can parse them with their own module-local schema.
    newton: Dict[str, Any] = field(default_factory=dict)
    imu: Dict[str, Any] = field(default_factory=dict)
    characters: List[Any] = field(default_factory=list)
    assemble: Dict[str, Any] = field(default_factory=dict)
    #: Source path (``None`` for from_dict without a path).
    source_path: Optional[Path] = None
    #: Unknown top-level keys — round-tripped, callers may inspect.
    extra: Dict[str, Any] = field(default_factory=dict)

    _KNOWN_TOP_LEVEL = frozenset(
        {
            "robot",
            "scene",
            "cameras",
            "lidars",
            "viewer_camera",
            "free_cam_width",
            "free_cam_height",
            "camera_mode",
            "contact_publish",
            "newton",
            "imu",
            "characters",
            "assemble",
        }
    )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, source_path: Optional[Path] = None) -> "SceneConfig":
        raw = _require_mapping(raw, where="scene root", path=source_path)
        camera_raw = _list_or_empty(raw.get("cameras"))
        lidar_raw = _list_or_empty(raw.get("lidars"))
        try:
            cameras = [CameraEntry.from_dict(c) for c in camera_raw if isinstance(c, Mapping)]
        except SchemaError as exc:
            exc.path = source_path
            raise
        try:
            lidars = [LidarEntry.from_dict(l) for l in lidar_raw if isinstance(l, Mapping)]
        except SchemaError as exc:
            exc.path = source_path
            raise

        extra = {k: v for k, v in raw.items() if k not in cls._KNOWN_TOP_LEVEL}

        return cls(
            robot=RobotSection.from_dict(raw.get("robot")),
            scene_usda=_get_str(raw, "scene"),
            cameras=cameras,
            lidars=lidars,
            viewer_camera=dict(_mapping_or_empty(raw.get("viewer_camera"))),
            free_cam_width=_get_int(raw, "free_cam_width", 1280),
            free_cam_height=_get_int(raw, "free_cam_height", 720),
            camera_mode=(_get_str(raw, "camera_mode", "cam3") or "cam3"),
            contact_publish=ContactPublishSection.from_dict(raw.get("contact_publish")),
            newton=dict(_mapping_or_empty(raw.get("newton"))),
            imu=dict(_mapping_or_empty(raw.get("imu"))),
            characters=_list_or_empty(raw.get("characters")),
            assemble=dict(_mapping_or_empty(raw.get("assemble"))),
            source_path=source_path,
            extra=extra,
        )

    @classmethod
    def read(cls, path: Path) -> "SceneConfig":
        p = Path(path)
        try:
            with p.open() as fh:
                raw = yaml.safe_load(fh)
        except yaml.YAMLError as err:
            raise SchemaError(f"scene YAML at {p} failed to parse: {err}", path=p) from err
        except OSError as err:
            raise SchemaError(f"scene YAML at {p} unreadable: {err}", path=p) from err
        if raw is None:
            raw = {}
        return cls.from_dict(raw, source_path=p)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        robot_dict = self.robot.to_dict()
        if robot_dict:
            out["robot"] = robot_dict
        if self.scene_usda:
            out["scene"] = self.scene_usda
        if self.cameras:
            out["cameras"] = [c.to_dict() for c in self.cameras]
        if self.lidars:
            out["lidars"] = [l.to_dict() for l in self.lidars]
        if self.viewer_camera:
            out["viewer_camera"] = dict(self.viewer_camera)
        if self.free_cam_width != 1280:
            out["free_cam_width"] = self.free_cam_width
        if self.free_cam_height != 720:
            out["free_cam_height"] = self.free_cam_height
        if self.camera_mode != "cam3":
            out["camera_mode"] = self.camera_mode
        # Only round-trip contact_publish when it deviates from the
        # manipulation default — matches how other optional blocks
        # only serialize when non-default, keeping the emitted YAML
        # readable.
        if self.contact_publish != ContactPublishSection():
            out["contact_publish"] = self.contact_publish.to_dict()
        if self.newton:
            out["newton"] = dict(self.newton)
        if self.imu:
            out["imu"] = dict(self.imu)
        if self.characters:
            out["characters"] = list(self.characters)
        if self.assemble:
            out["assemble"] = dict(self.assemble)
        out.update(self.extra)
        return out

    def write(self, path: Path) -> None:
        """Round-trip write — used by launchers that snapshot the scene into stage_dir."""
        p = Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        tmp.replace(p)

    # ------------------------------------------------------------------
    # Launcher helper: resolve robot_model/body/arm/gripper with CLI fallback
    # ------------------------------------------------------------------

    def resolve_robot_params(
        self,
        *,
        robot_model: str = "",
        body: str = "",
        arm: str = "",
        gripper: str = "",
        lgripper: str = "",
        rgripper: str = "",
    ) -> Dict[str, str]:
        """Scene YAML wins where non-empty; CLI defaults fill the rest.

        Consolidates ``resolve_robot_params_from_scene`` and the three
        duplicated ``resolve_scene_yaml_robot_params`` copies in
        ``genie_sim_bringup`` / ``genie_sim_bringup_industrial``.

        For each of ``robot_model / body / arm / gripper``: checks the
        flat position on ``robot_source`` first, then the nested
        ``robot_source.urdf`` block, then falls back to the CLI default.
        Matches ``_pick_robot_source_str`` in the legacy helpers so no
        scene YAML has to move fields.

        ``lgripper`` / ``rgripper`` are returned alongside so callers
        that need the split (grippers of two different types on the
        two arms) get them without a second lookup. Precedence for each
        side: ``robot_source.urdf.{lgripper,rgripper}`` → resolved
        ``gripper`` (simple mirror) → CLI-passed ``lgripper`` /
        ``rgripper``.  Simple configs still Just Work: setting only
        ``robot_source.gripper: scc`` yields ``lgripper=rgripper=scc``.
        """
        rs = self.robot.robot_source
        # Nested urdf sub-block (round-tripped into extra + explicit fields).
        urdf_extras: Dict[str, Any] = dict(rs.urdf.extra) if rs.urdf is not None else {}
        if rs.urdf is not None:
            for name in ("xacro_entry", "lgripper", "rgripper"):
                value = getattr(rs.urdf, name, "") or ""
                if value:
                    urdf_extras[name] = value

        def _flat(name: str) -> str:
            flat = getattr(rs, name, "") or ""
            return flat.strip() if isinstance(flat, str) else ""

        def _nested(name: str) -> str:
            raw = urdf_extras.get(name)
            return str(raw).strip() if raw is not None else ""

        launch_defaults = {
            "robot_model": robot_model,
            "body": body,
            "arm": arm,
            "gripper": gripper,
        }
        out: Dict[str, str] = {}
        for key in ("robot_model", "body", "arm", "gripper"):
            scene_val = _flat(key) or _nested(key)
            out[key] = scene_val if scene_val else launch_defaults[key]

        # Split-gripper resolution. ``urdf.lgripper`` / ``urdf.rgripper``
        # win when the scene has them (two-gripper robots); otherwise
        # both sides mirror the resolved simple ``gripper``. The CLI
        # ``lgripper`` / ``rgripper`` args (if the launcher forwards
        # them) are only used when neither the scene split nor
        # ``gripper`` is set — that way ``gripper: scc`` on a scene
        # keeps overriding a CLI ``lgripper=jd``.
        resolved_gripper = out["gripper"]
        out["lgripper"] = _nested("lgripper") or resolved_gripper or lgripper
        out["rgripper"] = _nested("rgripper") or resolved_gripper or rgripper
        return out


__all__ = [
    "CameraEntry",
    "CameraIntrinsic",
    "CameraSensor",
    "CameraTopic",
    "LidarEntry",
    "LidarSensor",
    "PIN_BASE_MODE_VALUES",
    "PinBaseMode",
    "Pose",
    "PrebakedSource",
    "RobotSection",
    "RobotSource",
    "SceneConfig",
    "UrdfSource",
]
