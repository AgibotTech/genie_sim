# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``manifest.json`` schema — the cross-language stage-dir contract.

Every producer (assemble stages / MuJoCo tail hook) writes this shape;
every reader (session bootstrap, render nodes, engine adapters) parses
it. The C++ side has a mirror header (``manifest_schema.hpp``) that
uses hand-written ``to_json``/``from_json`` overloads on the same
field names, so a rename that lands in Python without the corresponding
C++ edit fails fast rather than silently no-op'ing (the class of bug
that motivated adding this schema).

Historically the flat manifest had all fields at the top level and the
initial schema-v3 landing placed cameras/robot/physics under nested
buckets. This module unifies: top-level flat fields (``base_path``,
``scene_usda``, ``robot_usda``, ``background_usda``,
``render_layer_usda``, ``robot_prefix``, ``cameras``, ``lidars``,
``free_cam_prim_path``) are canonical AND typed; the ``robot`` /
``scene`` / ``physics`` dicts are opaque core-owned buckets. v4 (the v2
wire protocol) promoted ``body_paths`` + ``joint_names`` into ``robot``
and ``scene`` respectively — that protocol is specified in
``DESIGN.DATA_EXCHANGE_PROTOCOL.md``, which rolling moved out of this
repo in ``5cc27ba67b``; read it with
``git show 5cc27ba67b^:source/geniesim_ros/docs/DESIGN.DATA_EXCHANGE_PROTOCOL.md``.

Bumping ``MANIFEST_VERSION`` is a coordinated change:

* update the schema below,
* update the C++ ``kExpectedVersion`` constant in ``manifest_schema.hpp``,
* teach ``Manifest.read`` to handle the older version (or refuse it),
* update every backend that writes the manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ._common import (
    _get_bool,
    _get_float,
    _get_int,
    _get_str,
    _list_or_empty,
    _mapping_or_empty,
    _require_mapping,
    relativize_under,
    resolve_under,
)
from .errors import SchemaError, SchemaVersionError

#: Bump on any change to the top-level (core-owned) shape that the C++ side
#: reads or that alters the meaning of existing keys. When bumping, also bump
#: ``manifest_schema.hpp::kExpectedVersion`` in lockstep.
#:
#: Carve-out: adding an OPTIONAL field with a default that old manifests can
#: omit, and that the C++ mirror does not reference, is backward- and
#: forward-compatible and does NOT require a bump (precedent:
#: ``background_usda``). Bumping in that case would itself create a real
#: incompatibility (old C++ expecting the previous version vs. new manifests).
#:
#: 3 -> 4 was NOT such a case: v4 moved ``body_paths`` / ``joint_names`` into
#: the ``robot`` / ``scene`` buckets, which changes the meaning of existing
#: keys, so ``read`` refuses v3 outright rather than upgrading it. Every
#: manifest already written to disk is a per-task temp artifact
#: (``/tmp/rlinf_stage_p<port>/``) — rebuild the task, don't migrate the file.
MANIFEST_VERSION = 4


# -----------------------------------------------------------------------
# Cameras / lidars — the render node's flat contract
# -----------------------------------------------------------------------


@dataclass
class CameraManifest:
    """One ``cameras[]`` entry in ``manifest.json``.

    Flat field layout mirrors ``render_node.cpp::CameraConfig`` — the
    C++ ``struct CameraManifest`` in ``manifest_schema.hpp`` maps
    field-for-field via hand-written ``to_json``/``from_json`` overloads.
    Every knob the render node's ``load_manifest`` reads lives here;
    no nesting, no ``sensor.*`` / ``intrinsic.*`` sub-dicts. Producers
    that read a nested :class:`~genie_sim_engine_core.schema.scene.CameraEntry`
    from scene YAML flatten it to this shape via :meth:`from_scene_entry`.
    """

    path: str
    topic: str
    depth_topic: str = ""
    dds_topic: str = ""
    dds_depth_topic: str = ""
    render_product_path: str = ""
    frame_id: str = ""
    width: int = 1280
    height: int = 800
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
    model: str = ""
    # Total field of view of a >180 deg fisheye, in degrees, carried through
    # from the scene yaml ``mask_fov_deg`` / ``max_fov_deg``. The render node
    # uses it only to size the circular black mask it paints over the pixels
    # the lens traces no ray for; 0 means "not a masked fisheye".
    max_fov_deg: float = 0.0
    min_range: float = 0.01
    max_range: float = 10000.0
    is_free_cam: bool = False
    depth_scale: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CameraManifest":
        raw = _require_mapping(raw, where="manifest.cameras[]")
        path = _get_str(raw, "path")
        if not path:
            raise SchemaError("manifest.cameras[]: missing 'path'")
        topic = _get_str(raw, "topic")
        if not topic and not _get_bool(raw, "is_free_cam", False):
            raise SchemaError(f"manifest.cameras[{path!r}]: missing 'topic'")
        depth_scale = raw.get("depth_scale")
        return cls(
            path=path,
            topic=topic,
            depth_topic=_get_str(raw, "depth_topic"),
            dds_topic=_get_str(raw, "dds_topic"),
            dds_depth_topic=_get_str(raw, "dds_depth_topic"),
            render_product_path=_get_str(raw, "render_product_path"),
            frame_id=_get_str(raw, "frame_id"),
            width=_get_int(raw, "width", 1280),
            height=_get_int(raw, "height", 800),
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
            model=_get_str(raw, "model"),
            max_fov_deg=_get_float(raw, "max_fov_deg", 0.0),
            min_range=_get_float(raw, "min_range", 0.01),
            max_range=_get_float(raw, "max_range", 10000.0),
            is_free_cam=_get_bool(raw, "is_free_cam", False),
            depth_scale=float(depth_scale) if depth_scale is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "path": self.path,
            "topic": self.topic,
            "depth_topic": self.depth_topic,
            "render_product_path": self.render_product_path,
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
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
            "model": self.model,
            "max_fov_deg": self.max_fov_deg,
            "min_range": self.min_range,
            "max_range": self.max_range,
            "is_free_cam": self.is_free_cam,
        }
        # dds_* only appear when the scene routes this cam through DDS.
        # Emit them unconditionally so C++ round-trip sees the empty string
        # (the header default) instead of key-missing.
        out["dds_topic"] = self.dds_topic
        out["dds_depth_topic"] = self.dds_depth_topic
        if self.depth_scale is not None:
            out["depth_scale"] = self.depth_scale
        return out

    @classmethod
    def from_scene_entry(
        cls,
        entry: Any,
        *,
        index: int,
        render_product_path: Optional[str] = None,
    ) -> "CameraManifest":
        """Flatten a :class:`~genie_sim_engine_core.schema.scene.CameraEntry`.

        The mujoco assemble hook and the isaacsim assembler both call
        this — one authoritative shape, no more "cam_json.at('path')
        threw because the writer shipped a nested block" surprises.

        ``render_product_path`` defaults to ``/RenderOVRTX/Cam_<i>``,
        matching what ``render_node.cpp`` fabricates when the field is
        missing.
        """
        rp = render_product_path or f"/RenderOVRTX/Cam_{index}"
        return cls(
            path=entry.prim_path,
            topic=entry.topic.rgb,
            depth_topic=entry.topic.depth,
            dds_topic=entry.dds_topic,
            dds_depth_topic=entry.dds_depth_topic,
            render_product_path=rp,
            frame_id=entry.frame_id,
            width=entry.sensor.width,
            height=entry.sensor.height,
            fx=entry.intrinsic.fx,
            fy=entry.intrinsic.fy,
            cx=entry.intrinsic.cx,
            cy=entry.intrinsic.cy,
            k1=entry.intrinsic.k1,
            k2=entry.intrinsic.k2,
            p1=entry.intrinsic.p1,
            p2=entry.intrinsic.p2,
            k3=entry.intrinsic.k3,
            k4=entry.intrinsic.k4,
            model=entry.model,
            max_fov_deg=float(entry.max_fov_deg or 0.0),
            min_range=entry.sensor.min_range,
            max_range=entry.sensor.max_range,
            depth_scale=entry.depth_scale,
        )


@dataclass
class LidarBaseLinkT:
    """``base_link_T_sensor`` xform on ``lidars[]`` in the manifest."""

    xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "LidarBaseLinkT":
        raw = _mapping_or_empty(raw)
        xyz_raw = raw.get("xyz")
        wxyz_raw = raw.get("wxyz")
        try:
            xyz = (float(xyz_raw[0]), float(xyz_raw[1]), float(xyz_raw[2])) if xyz_raw else (0.0, 0.0, 0.0)
        except (TypeError, ValueError, IndexError):
            xyz = (0.0, 0.0, 0.0)
        try:
            wxyz = (
                (
                    float(wxyz_raw[0]),
                    float(wxyz_raw[1]),
                    float(wxyz_raw[2]),
                    float(wxyz_raw[3]),
                )
                if wxyz_raw
                else (1.0, 0.0, 0.0, 0.0)
            )
        except (TypeError, ValueError, IndexError):
            wxyz = (1.0, 0.0, 0.0, 0.0)
        return cls(xyz=xyz, wxyz=wxyz)

    def to_dict(self) -> Dict[str, Any]:
        return {"xyz": list(self.xyz), "wxyz": list(self.wxyz)}


@dataclass
class LidarManifest:
    """One ``lidars[]`` entry in ``manifest.json``."""

    render_product_path: str
    topic: str
    frame_id: str = "base_link"
    parent_body: str = ""
    base_link_T_sensor: LidarBaseLinkT = field(default_factory=LidarBaseLinkT)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LidarManifest":
        raw = _require_mapping(raw, where="manifest.lidars[]")
        return cls(
            render_product_path=_get_str(raw, "render_product_path"),
            topic=_get_str(raw, "topic"),
            frame_id=_get_str(raw, "frame_id", "base_link"),
            parent_body=_get_str(raw, "parent_body"),
            base_link_T_sensor=LidarBaseLinkT.from_dict(raw.get("base_link_T_sensor")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "render_product_path": self.render_product_path,
            "topic": self.topic,
            "frame_id": self.frame_id,
            "parent_body": self.parent_body,
            "base_link_T_sensor": self.base_link_T_sensor.to_dict(),
        }


# -----------------------------------------------------------------------
# Root Manifest
# -----------------------------------------------------------------------


@dataclass
class Manifest:
    """Structured view of ``manifest.json`` — cross-language ABI.

    Path fields (``base_path``, ``scene_usda``, ``robot_usda``,
    ``background_usda``, ``render_layer_usda``, ``robot_visual_usda``,
    ``scene_yaml``) are stored on disk **relative to ``base_path``**
    (the convention every prior writer followed). :meth:`read` resolves
    them into absolute :class:`Path` objects; :meth:`write` relativizes
    back. Fields whose absolute form is not under ``base_path`` (e.g. a
    scene YAML on a sibling drive) are stored absolute.

    ``background_usda`` is optional (empty/``None`` by default): when a
    task opts into a background room, the newton-standalone engine
    references it into its stage *before* ``add_usd`` so the room's
    collision (shelf/counter/table) physically supports authored
    objects. Isaac (Kit) backends ignore it. This is an additive,
    backward-compatible field — it does not require a
    ``MANIFEST_VERSION`` bump (see module docstring): an old manifest
    that lacks the key simply resolves it to ``None``, matching
    pre-existing behaviour exactly.

    ``physics`` is a nested dict keyed by backend engine name; each
    backend contributes its own slice there via
    :meth:`merge_slice`. Core does not validate its contents.

    ``version`` is checked strictly on read. ``SchemaVersionError`` is
    the diagnostic path for stage-dir staleness — the launcher can then
    prompt the operator to re-run the assemble CLI.
    """

    version: int = MANIFEST_VERSION
    physics_engine: str = ""
    #: Launcher YAML name / path passed via ``launcher_config:=…`` at
    #: bringup time. The bringup pre-check compares this against the
    #: launch's current value: on mismatch, cache-gated stages
    #: (URDF→USD) are re-run so a launcher swap always produces a
    #: consistent stage dir. Empty string is treated as "mismatch"
    #: (older bakes that predate this field trigger a one-time
    #: regeneration).
    launcher_config: str = ""
    base_path: Optional[Path] = None
    stage_dir: Optional[Path] = None
    scene_yaml: Optional[Path] = None
    robot_prefix: str = "genie"
    robot_from_urdf: bool = False
    scene_usda: Optional[Path] = None
    robot_usda: Optional[Path] = None
    #: Optional background room (shelf/counter/table). Empty unless a
    #: task opts in. See the class docstring for the physics-loading
    #: contract; the render-side consumer lives in genie_sim_render.
    background_usda: Optional[Path] = None
    robot_visual_usda: Optional[Path] = None
    render_layer_usda: Optional[Path] = None
    free_cam_prim_path: str = ""
    cameras: List[CameraManifest] = field(default_factory=list)
    lidars: List[LidarManifest] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)
    #: Backend-specific slice; keys are engine names.
    physics: Dict[str, Any] = field(default_factory=dict)
    #: Cross-backend robot metadata. Under v2 wire protocol this carries
    #: ``robot.body_paths`` (short link name → absolute USD prim path)
    #: alongside the existing URDF hash stamp.
    robot: Dict[str, Any] = field(default_factory=dict)
    #: Cross-backend scene metadata. Under v2 wire protocol this carries
    #: ``scene.body_paths`` (scene body name → absolute USD prim path)
    #: and ``scene.joint_names`` (verbatim MJCF joint names, list) that
    #: the renderer / benchmark consume to resolve ``/tf_scene`` +
    #: ``/joint_states_scene`` traffic. Opaque top-level bucket to
    #: match the shape of ``robot`` and ``physics``.
    scene: Dict[str, Any] = field(default_factory=dict)

    _CORE_TOP_LEVEL = frozenset(
        {
            "version",
            "physics_engine",
            "launcher_config",
            "base_path",
            "stage_dir",
            "scene_yaml",
            "robot_prefix",
            "robot_from_urdf",
            "scene_usda",
            "robot_usda",
            "background_usda",
            "robot_visual_usda",
            "render_layer_usda",
            "free_cam_prim_path",
            "cameras",
            "lidars",
            "files",
        }
    )
    _STAGE_WRITABLE_KEYS = frozenset(
        {
            # keys stages MAY contribute via manifest_slice
            "physics",
            "cameras",
            "lidars",
            "robot",
            "scene",
            "files",
            "scene_usda",
            "robot_usda",
            "background_usda",
            "robot_visual_usda",
            "render_layer_usda",
            "robot_prefix",
            "robot_from_urdf",
            "free_cam_prim_path",
            "scene_yaml",
            "launcher_config",
        }
    )

    # ------------------------------------------------------------------
    # Reader
    # ------------------------------------------------------------------

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        """Read ``manifest.json`` and validate the top-level shape.

        Refuses older versions rather than silently upgrading — the
        launcher should stage a fresh manifest instead.
        """
        p = Path(path)
        try:
            with p.open() as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as err:
            raise SchemaError(f"manifest at {p} unreadable: {err}", path=p) from err
        return cls.from_dict(raw, source_path=p)

    @classmethod
    def from_dict(cls, raw: Any, *, source_path: Optional[Path] = None) -> "Manifest":
        if not isinstance(raw, Mapping):
            raise SchemaError(f"manifest is not a JSON object", path=source_path)
        version = raw.get("version")
        if version != MANIFEST_VERSION:
            raise SchemaVersionError(
                f"manifest at {source_path} has version={version!r}; "
                f"core expects {MANIFEST_VERSION}. Re-run the assemble CLI.",
                path=source_path,
            )
        base_path_raw = _get_str(raw, "base_path")
        base_path = Path(base_path_raw) if base_path_raw else None
        # Resolve every path field against base_path (falling back to the
        # manifest file's directory when base_path is unset — same
        # behaviour as the legacy render_node.cpp::load_manifest).
        resolve_root = (
            base_path if base_path is not None else (source_path.parent if source_path is not None else Path.cwd())
        )
        cameras_raw = _list_or_empty(raw.get("cameras"))
        lidars_raw = _list_or_empty(raw.get("lidars"))
        return cls(
            version=version,
            physics_engine=_get_str(raw, "physics_engine"),
            launcher_config=_get_str(raw, "launcher_config"),
            base_path=base_path,
            stage_dir=resolve_under(resolve_root, _get_str(raw, "stage_dir") or None),
            scene_yaml=resolve_under(resolve_root, _get_str(raw, "scene_yaml") or None),
            robot_prefix=_get_str(raw, "robot_prefix", "genie") or "genie",
            robot_from_urdf=_get_bool(raw, "robot_from_urdf", False),
            scene_usda=resolve_under(resolve_root, raw.get("scene_usda") or raw.get("usd_path")),
            robot_usda=resolve_under(resolve_root, raw.get("robot_usda")),
            background_usda=resolve_under(resolve_root, raw.get("background_usda")),
            robot_visual_usda=resolve_under(resolve_root, raw.get("robot_visual_usda")),
            render_layer_usda=resolve_under(resolve_root, raw.get("render_layer_usda")),
            free_cam_prim_path=_get_str(raw, "free_cam_prim_path"),
            cameras=[CameraManifest.from_dict(c) for c in cameras_raw if isinstance(c, Mapping)],
            lidars=[LidarManifest.from_dict(l) for l in lidars_raw if isinstance(l, Mapping)],
            files=dict(_mapping_or_empty(raw.get("files"))),
            physics=dict(_mapping_or_empty(raw.get("physics"))),
            robot=dict(_mapping_or_empty(raw.get("robot"))),
            scene=dict(_mapping_or_empty(raw.get("scene"))),
        )

    # ------------------------------------------------------------------
    # Writer
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the on-disk shape.

        Paths are relativized against :attr:`base_path` when set so the
        stage dir survives being moved (rsync'd to a container, etc.).
        Fields left as ``None`` become empty strings on disk — matches
        the legacy convention every existing reader assumes.
        """
        base = self.base_path or Path.cwd()
        return {
            "version": self.version,
            "physics_engine": self.physics_engine,
            "launcher_config": self.launcher_config,
            "base_path": str(self.base_path) if self.base_path is not None else "",
            "stage_dir": relativize_under(base, self.stage_dir),
            "scene_yaml": str(self.scene_yaml) if self.scene_yaml is not None else "",
            "robot_prefix": self.robot_prefix,
            "robot_from_urdf": self.robot_from_urdf,
            "scene_usda": relativize_under(base, self.scene_usda),
            "robot_usda": relativize_under(base, self.robot_usda),
            "background_usda": relativize_under(base, self.background_usda),
            "robot_visual_usda": relativize_under(base, self.robot_visual_usda),
            "render_layer_usda": relativize_under(base, self.render_layer_usda),
            "free_cam_prim_path": self.free_cam_prim_path,
            "cameras": [c.to_dict() for c in self.cameras],
            "lidars": [l.to_dict() for l in self.lidars],
            "files": dict(self.files),
            "robot": dict(self.robot),
            "scene": dict(self.scene),
            "physics": dict(self.physics),
        }

    def write(self, path: Path) -> None:
        """Write ``manifest.json`` atomically (temp file + rename).

        Stage-dir consumers may see the file mid-write otherwise, which
        has caused mysterious "empty manifest" launcher failures. Keys
        are NOT sorted so the on-disk layout follows :meth:`to_dict`
        order and diffs stay readable when producers change.
        """
        p = Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(p)

    # ------------------------------------------------------------------
    # Stage integration — merge_slice
    # ------------------------------------------------------------------

    def merge_slice(self, slice_: Mapping[str, Any], stage_name: str) -> None:
        """Merge one stage's ``manifest_slice`` into the manifest.

        Stages MAY write to any key in :attr:`_STAGE_WRITABLE_KEYS`.
        Scalar fields overwrite; dict fields deep-merge; list fields
        (``cameras``, ``lidars``) replace (a stage that touches cameras
        owns the full list, matching current behaviour).

        Any other key is a schema violation. This catches "stage tried
        to overwrite ``version``" bugs at merge time instead of at
        manifest-read time.
        """
        if not isinstance(slice_, Mapping):
            raise TypeError(f"stage {stage_name!r} returned non-dict manifest_slice")
        for key, value in slice_.items():
            if key not in self._STAGE_WRITABLE_KEYS:
                raise ValueError(
                    f"stage {stage_name!r} attempted to write core-owned key "
                    f"{key!r}; stages may only write under: "
                    f"{sorted(self._STAGE_WRITABLE_KEYS)!r}"
                )
            if key == "physics":
                if not isinstance(value, Mapping):
                    raise TypeError(f"stage {stage_name!r}: 'physics' slice must be dict")
                for engine_key, engine_slice in value.items():
                    bucket = self.physics.setdefault(engine_key, {})
                    _deep_update(bucket, engine_slice)
            elif key == "robot":
                _deep_update(self.robot, dict(_mapping_or_empty(value)))
            elif key == "scene":
                _deep_update(self.scene, dict(_mapping_or_empty(value)))
            elif key == "files":
                if not isinstance(value, Mapping):
                    raise TypeError(f"stage {stage_name!r}: 'files' slice must be dict")
                self.files.update({k: str(v) for k, v in value.items()})
            elif key == "cameras":
                if not isinstance(value, list):
                    raise TypeError(f"stage {stage_name!r}: 'cameras' slice must be list")
                self.cameras = [c if isinstance(c, CameraManifest) else CameraManifest.from_dict(c) for c in value]
            elif key == "lidars":
                if not isinstance(value, list):
                    raise TypeError(f"stage {stage_name!r}: 'lidars' slice must be list")
                self.lidars = [l if isinstance(l, LidarManifest) else LidarManifest.from_dict(l) for l in value]
            elif key in (
                "scene_usda",
                "robot_usda",
                "background_usda",
                "robot_visual_usda",
                "render_layer_usda",
                "scene_yaml",
            ):
                setattr(self, key, Path(value) if value else None)
            elif key == "robot_prefix":
                self.robot_prefix = str(value) or "genie"
            elif key == "robot_from_urdf":
                self.robot_from_urdf = bool(value)
            elif key == "free_cam_prim_path":
                self.free_cam_prim_path = str(value)
            elif key == "launcher_config":
                self.launcher_config = str(value)


def _deep_update(dst: Dict[str, Any], src: Any) -> None:
    """Merge ``src`` into ``dst`` in place; dicts recurse, leaves overwrite."""
    if not isinstance(src, Mapping):
        raise TypeError(f"cannot deep-update: expected mapping, got {type(src).__name__}")
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value


__all__ = [
    "CameraManifest",
    "LidarBaseLinkT",
    "LidarManifest",
    "MANIFEST_VERSION",
    "Manifest",
]
