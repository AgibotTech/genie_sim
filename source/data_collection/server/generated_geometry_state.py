"""Recording and checkpoint primitives for append-only generated meshes.

The data model in this module deliberately has no Isaac Sim imports.  Live USD
access is supplied by a small backend so validation, timestamp alignment, and
checkpoint state transitions can be tested in a normal Python environment.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import numbers
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


GENERATED_GEOMETRY_SCHEMA = "geniesim.generated_geometry/v1"
GENERATED_GEOMETRY_DIR = "generated_geometry"
GENERATED_GEOMETRY_MANIFEST_SIDECAR = "_manifest.json"
GENERATED_GEOMETRY_FRAMES_SIDECAR = "_frames.json"
GENERATED_GEOMETRY_RECORDABLE_ATTR = "generatedGeometry:recordable"
GENERATED_GEOMETRY_TARGET_REL = "generatedGeometry:target"
GENERATED_GEOMETRY_MODE_ATTR = "generatedGeometry:mode"
GENERATED_GEOMETRY_REVISION_ATTR = "generatedGeometry:revision"
GENERATED_GEOMETRY_POINTS_END_ATTR = "generatedGeometry:pointsEnd"
GENERATED_GEOMETRY_FACES_END_ATTR = "generatedGeometry:facesEnd"
GENERATED_GEOMETRY_INDICES_END_ATTR = "generatedGeometry:indicesEnd"
GENERATED_GEOMETRY_STATE_ATTRIBUTES_ATTR = "generatedGeometry:stateAttributes"


def generated_geometry_key(prim_path: str) -> str:
    if (
        not isinstance(prim_path, str)
        or not prim_path.startswith("/")
        or prim_path == "/"
    ):
        raise ValueError(
            "generated geometry prim_path must be an absolute USD prim path"
        )
    return prim_path[1:].replace("/", "_")


def generated_geometry_artifact_path(prim_path: str) -> str:
    source_id = generated_geometry_key(prim_path)
    return f"{GENERATED_GEOMETRY_DIR}/{source_id}/mesh.npz"


@dataclass(frozen=True)
class GeneratedGeometryPrefix:
    points_end: int
    faces_end: int
    indices_end: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GeneratedGeometryPrefix":
        try:
            fields = (
                value["points_end"],
                value["faces_end"],
                value["indices_end"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "generated geometry prefix requires points_end, faces_end, and indices_end"
            ) from exc
        if any(isinstance(item, bool) or not isinstance(item, numbers.Integral) for item in fields):
            raise ValueError("generated geometry prefix values must be integers")
        return cls(*(int(item) for item in fields))

    def to_dict(self) -> dict[str, int]:
        return {
            "points_end": self.points_end,
            "faces_end": self.faces_end,
            "indices_end": self.indices_end,
        }


@dataclass(frozen=True)
class GeneratedGeometryMeshData:
    points: np.ndarray
    face_vertex_counts: np.ndarray
    face_vertex_indices: np.ndarray
    display_color: np.ndarray


@dataclass(frozen=True)
class GeneratedGeometryCheckpoint:
    control_exists: bool
    mesh_exists: bool
    prefix: GeneratedGeometryPrefix
    revision: int
    control_state: dict[str, Any]


def _normalized_mesh(
    data: GeneratedGeometryMeshData,
) -> GeneratedGeometryMeshData:
    return GeneratedGeometryMeshData(
        points=np.ascontiguousarray(data.points, dtype=np.float32),
        face_vertex_counts=np.ascontiguousarray(data.face_vertex_counts, dtype=np.int32),
        face_vertex_indices=np.ascontiguousarray(data.face_vertex_indices, dtype=np.int32),
        display_color=np.ascontiguousarray(data.display_color, dtype=np.float32),
    )


def validate_generated_geometry_mesh(data: GeneratedGeometryMeshData) -> None:
    points = np.asarray(data.points)
    counts = np.asarray(data.face_vertex_counts)
    indices = np.asarray(data.face_vertex_indices)
    color = np.asarray(data.display_color)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("generated geometry points must have shape (N, 3)")
    if counts.ndim != 1:
        raise ValueError("generated geometry face counts must have shape (F,)")
    if indices.ndim != 1:
        raise ValueError("generated geometry face indices must have shape (I,)")
    if color.ndim != 2 or color.shape[1:] != (3,):
        raise ValueError("generated geometry display color must have shape (C, 3)")
    if not np.isfinite(points).all() or not np.isfinite(color).all():
        raise ValueError("generated geometry points and display color must be finite")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError(
            "generated geometry face counts must be non-negative integers"
        )
    if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0):
        raise ValueError(
            "generated geometry face index values must be non-negative integers"
        )
    if int(np.sum(counts, dtype=np.int64)) != len(indices):
        raise ValueError(
            "generated geometry face counts do not match face indices length"
        )
    if len(indices) and int(np.max(indices)) >= len(points):
        raise ValueError("generated geometry face index is outside points array")


def validate_generated_geometry_prefix(
    data: GeneratedGeometryMeshData,
    prefix: GeneratedGeometryPrefix,
) -> None:
    values = (prefix.points_end, prefix.faces_end, prefix.indices_end)
    if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) for value in values):
        raise ValueError("generated geometry prefix values must be integers")
    if any(value < 0 for value in values):
        raise ValueError("generated geometry prefix values must be non-negative")
    if prefix.points_end > len(data.points):
        raise ValueError("generated geometry points prefix exceeds artifact")
    if prefix.faces_end > len(data.face_vertex_counts):
        raise ValueError("generated geometry faces prefix exceeds artifact")
    if prefix.indices_end > len(data.face_vertex_indices):
        raise ValueError("generated geometry indices prefix exceeds artifact")
    expected_indices_end = int(
        np.sum(data.face_vertex_counts[: prefix.faces_end], dtype=np.int64)
    )
    if expected_indices_end != prefix.indices_end:
        raise ValueError(
            "generated geometry prefix does not end on a face boundary"
        )
    visible_indices = data.face_vertex_indices[: prefix.indices_end]
    if len(visible_indices) and int(np.max(visible_indices)) >= prefix.points_end:
        raise ValueError(
            "generated geometry face index is outside visible points prefix"
        )


def write_generated_geometry_artifact(
    path: str | os.PathLike[str],
    data: GeneratedGeometryMeshData,
) -> Path:
    normalized = _normalized_mesh(data)
    validate_generated_geometry_mesh(normalized)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    try:
        np.savez_compressed(
            temporary,
            points=normalized.points,
            face_vertex_counts=normalized.face_vertex_counts,
            face_vertex_indices=normalized.face_vertex_indices,
            display_color=normalized.display_color,
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_generated_geometry_artifact(
    path: str | os.PathLike[str],
) -> GeneratedGeometryMeshData:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"generated geometry artifact not found: {artifact_path}"
        )
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            required = {
                "points",
                "face_vertex_counts",
                "face_vertex_indices",
                "display_color",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    f"generated geometry artifact missing keys: {sorted(missing)}"
                )
            data = GeneratedGeometryMeshData(
                points=np.asarray(archive["points"], dtype=np.float32),
                face_vertex_counts=np.asarray(archive["face_vertex_counts"], dtype=np.int32),
                face_vertex_indices=np.asarray(archive["face_vertex_indices"], dtype=np.int32),
                display_color=np.asarray(archive["display_color"], dtype=np.float32),
            )
    except (OSError, KeyError) as exc:
        raise ValueError(
            f"failed to load generated geometry artifact {artifact_path}: {exc}"
        ) from exc
    validate_generated_geometry_mesh(data)
    return data


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class GeneratedGeometryRecorder:
    def __init__(self, backend):
        self.backend = backend
        self.control_prim_path = str(backend.control_prim_path)
        self.prim_path = str(backend.mesh_prim_path)
        self.source_id = generated_geometry_key(self.prim_path)
        self.artifact_path = generated_geometry_artifact_path(self.prim_path)
        self.samples: list[dict[str, Any]] = []
        self._output_dir: Path | None = None
        self._active = False

    def start(self, stage, output_dir: str | os.PathLike[str], timestamp: float) -> bool:
        self.samples = []
        self._output_dir = Path(output_dir)
        self._active = bool(self.backend.is_recordable(stage))
        if self._active:
            self.sample(stage, timestamp)
        return self._active

    def sample(self, stage, timestamp: float) -> None:
        if not self._active:
            return
        prefix = self.backend.read_prefix(stage)
        revision = int(self.backend.read_revision(stage))
        signature = (prefix, revision)
        if self.samples:
            previous = self.samples[-1]
            previous_signature = (
                GeneratedGeometryPrefix.from_mapping(previous["prefix"]),
                int(previous["revision"]),
            )
            if signature == previous_signature:
                return
        self.samples.append(
            {
                "time_stamp": float(timestamp),
                "revision": revision,
                "prefix": prefix.to_dict(),
            }
        )

    def stop(self, stage, timestamp: float) -> dict[str, Any] | None:
        if not self._active or self._output_dir is None:
            self._active = False
            return None
        self.sample(stage, timestamp)
        data = _normalized_mesh(self.backend.read_mesh(stage))
        validate_generated_geometry_mesh(data)
        final_prefix = self.backend.read_prefix(stage)
        validate_generated_geometry_prefix(data, final_prefix)
        complete_prefix = GeneratedGeometryPrefix(
            len(data.points),
            len(data.face_vertex_counts),
            len(data.face_vertex_indices),
        )
        if final_prefix != complete_prefix:
            raise ValueError(
                "generated geometry final prefix does not match final Mesh arrays"
            )

        artifact = self._output_dir / self.artifact_path
        write_generated_geometry_artifact(artifact, data)
        manifest = {
            "schema": GENERATED_GEOMETRY_SCHEMA,
            "file": self.artifact_path,
            "control_prim_path": self.control_prim_path,
            "prim_path": self.prim_path,
            "append_only": True,
            "coordinate_space": "record_world",
        }
        _atomic_write_json(
            artifact.parent / GENERATED_GEOMETRY_FRAMES_SIDECAR,
            {"source_id": self.source_id, "frames": self.samples},
        )
        _atomic_write_json(
            artifact.parent / GENERATED_GEOMETRY_MANIFEST_SIDECAR,
            {"source_id": self.source_id, "manifest": manifest},
        )
        self._active = False
        return manifest


class GeneratedGeometryStateManager:
    def __init__(self, backend):
        self.backend = backend

    def capture_checkpoint(self, stage) -> GeneratedGeometryCheckpoint | None:
        control_exists = bool(self.backend.control_exists(stage))
        if not control_exists:
            return GeneratedGeometryCheckpoint(
                control_exists=False,
                mesh_exists=False,
                prefix=GeneratedGeometryPrefix(0, 0, 0),
                revision=0,
                control_state={},
            )
        if not self.backend.is_recordable(stage):
            return None
        return GeneratedGeometryCheckpoint(
            control_exists=True,
            mesh_exists=bool(self.backend.mesh_exists(stage)),
            prefix=self.backend.read_prefix(stage),
            revision=int(self.backend.read_revision(stage)),
            control_state=self.backend.read_control_state(stage),
        )

    def begin_restore(
        self,
        stage,
        checkpoint: GeneratedGeometryCheckpoint | None,
    ) -> None:
        if checkpoint is None:
            return
        if not checkpoint.control_exists:
            if (
                not self.backend.control_exists(stage)
                or not self.backend.is_recordable(stage)
            ):
                return
            self.backend.set_mode(stage, "paused")
            revision = int(self.backend.read_revision(stage))
            self.backend.clear(stage)
            self.backend.set_revision(stage, revision + 1)
            return
        if not self.backend.is_recordable(stage):
            return
        self.backend.set_mode(stage, "paused")
        self.backend.truncate_mesh(stage, checkpoint.prefix)
        self.backend.write_control_state(stage, checkpoint.control_state)
        self.backend.set_revision(stage, int(self.backend.read_revision(stage)) + 1)

    def finish_restore(self, stage) -> None:
        if self.backend.is_recordable(stage):
            self.backend.set_mode(stage, "live")

    def reset_live(self, stage) -> None:
        if not self.backend.is_recordable(stage):
            return
        self.backend.set_mode(stage, "paused")
        revision = int(self.backend.read_revision(stage))
        self.backend.clear(stage)
        self.backend.set_revision(stage, revision + 1)
        self.backend.set_mode(stage, "live")


class UsdGeneratedGeometryBackend:
    """USD adapter for registered append-only generated meshes.

    Imports from ``pxr`` stay inside methods so the pure data helpers above
    remain usable in extraction and unit-test environments without Isaac Sim.
    """

    def __init__(self, control_prim_path: str, mesh_prim_path: str):
        generated_geometry_key(control_prim_path)
        generated_geometry_key(mesh_prim_path)
        self.control_prim_path = str(control_prim_path)
        self.mesh_prim_path = str(mesh_prim_path)

    @classmethod
    def discover(cls, stage) -> list["UsdGeneratedGeometryBackend"]:
        discovered = []
        source_keys: set[str] = set()
        for prim in stage.Traverse():
            attr = prim.GetAttribute(GENERATED_GEOMETRY_RECORDABLE_ATTR)
            if not attr or not attr.IsValid() or attr.Get() is not True:
                continue
            relationship = prim.GetRelationship(GENERATED_GEOMETRY_TARGET_REL)
            targets = (
                list(relationship.GetTargets())
                if relationship and relationship.IsValid()
                else []
            )
            if len(targets) != 1 or not targets[0].IsAbsolutePath():
                raise ValueError(
                    f"generated geometry control prim {prim.GetPath()} must declare "
                    "exactly one target relationship"
                )
            mesh_prim_path = str(targets[0])
            source_key = generated_geometry_key(mesh_prim_path)
            if source_key in source_keys:
                raise ValueError(
                    f"duplicate generated geometry source key: {source_key}"
                )
            source_keys.add(source_key)
            discovered.append(cls(str(prim.GetPath()), mesh_prim_path))
        return discovered

    @staticmethod
    def _imports():
        from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

        return Gf, Sdf, UsdGeom, Vt

    def _control(self, stage, *, create: bool = False):
        _gf, _sdf, UsdGeom, _vt = self._imports()
        prim = stage.GetPrimAtPath(self.control_prim_path)
        if (not prim or not prim.IsValid()) and create:
            prim = UsdGeom.Xform.Define(stage, self.control_prim_path).GetPrim()
        return prim

    def _mesh(self, stage, *, create: bool = False):
        _gf, _sdf, UsdGeom, _vt = self._imports()
        prim = stage.GetPrimAtPath(self.mesh_prim_path)
        if (not prim or not prim.IsValid()) and create:
            self._control(stage, create=True)
            return UsdGeom.Mesh.Define(stage, self.mesh_prim_path)
        if not prim or not prim.IsValid():
            return None
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(
                "generated geometry target must be a UsdGeom.Mesh when it exists: "
                f"{self.mesh_prim_path}"
            )
        return UsdGeom.Mesh(prim)

    @staticmethod
    def _get_attr(prim, name: str, default):
        if not prim or not prim.IsValid():
            return default
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid():
            return default
        value = attr.Get()
        return default if value is None else value

    @staticmethod
    def _set_attr(prim, name: str, type_name, value) -> None:
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(name, type_name, custom=True)
        attr.Set(value)

    def control_exists(self, stage) -> bool:
        control = self._control(stage)
        return bool(control and control.IsValid())

    def is_recordable(self, stage) -> bool:
        control = self._control(stage)
        return bool(
            self._get_attr(control, GENERATED_GEOMETRY_RECORDABLE_ATTR, False)
        )

    def mesh_exists(self, stage) -> bool:
        return self._mesh(stage) is not None

    def read_revision(self, stage) -> int:
        return int(
            self._get_attr(
                self._control(stage), GENERATED_GEOMETRY_REVISION_ATTR, 0
            )
        )

    def read_prefix(self, stage) -> GeneratedGeometryPrefix:
        control = self._control(stage)
        counter_values = [
            self._get_attr(control, GENERATED_GEOMETRY_POINTS_END_ATTR, None),
            self._get_attr(control, GENERATED_GEOMETRY_FACES_END_ATTR, None),
            self._get_attr(control, GENERATED_GEOMETRY_INDICES_END_ATTR, None),
        ]
        if all(
            isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 0
            for value in counter_values
        ):
            return GeneratedGeometryPrefix(
                *(int(value) for value in counter_values)
            )
        mesh = self.read_mesh(stage)
        return GeneratedGeometryPrefix(
            len(mesh.points),
            len(mesh.face_vertex_counts),
            len(mesh.face_vertex_indices),
        )

    def read_mesh(self, stage) -> GeneratedGeometryMeshData:
        mesh = self._mesh(stage)
        if mesh is None:
            return GeneratedGeometryMeshData(
                points=np.empty((0, 3), dtype=np.float32),
                face_vertex_counts=np.empty((0,), dtype=np.int32),
                face_vertex_indices=np.empty((0,), dtype=np.int32),
                display_color=np.array([[0.005, 0.004, 0.003]], dtype=np.float32),
            )
        points_value = mesh.GetPointsAttr().Get() or []
        counts_value = mesh.GetFaceVertexCountsAttr().Get() or []
        indices_value = mesh.GetFaceVertexIndicesAttr().Get() or []
        color_value = mesh.GetDisplayColorAttr().Get() or [(0.005, 0.004, 0.003)]
        data = GeneratedGeometryMeshData(
            points=np.asarray(points_value, dtype=np.float32).reshape(-1, 3),
            face_vertex_counts=np.asarray(counts_value, dtype=np.int32).reshape(-1),
            face_vertex_indices=np.asarray(indices_value, dtype=np.int32).reshape(-1),
            display_color=np.asarray(color_value, dtype=np.float32).reshape(-1, 3),
        )
        validate_generated_geometry_mesh(data)
        return data

    def _write_mesh(self, stage, data: GeneratedGeometryMeshData) -> None:
        Gf, _sdf, _UsdGeom, Vt = self._imports()
        normalized = _normalized_mesh(data)
        validate_generated_geometry_mesh(normalized)
        mesh = self._mesh(stage, create=True)
        mesh.CreatePointsAttr().Set(
            Vt.Vec3fArray(
                [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in normalized.points]
            )
        )
        mesh.CreateFaceVertexCountsAttr().Set(
            Vt.IntArray([int(value) for value in normalized.face_vertex_counts])
        )
        mesh.CreateFaceVertexIndicesAttr().Set(
            Vt.IntArray([int(value) for value in normalized.face_vertex_indices])
        )
        mesh.CreateDisplayColorAttr().Set(
            Vt.Vec3fArray(
                [Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in normalized.display_color]
            )
        )
        self._write_prefix(
            stage,
            GeneratedGeometryPrefix(
                len(normalized.points),
                len(normalized.face_vertex_counts),
                len(normalized.face_vertex_indices),
            ),
        )

    def _write_prefix(self, stage, prefix: GeneratedGeometryPrefix) -> None:
        _gf, Sdf, _UsdGeom, _vt = self._imports()
        control = self._control(stage, create=True)
        self._set_attr(
            control,
            GENERATED_GEOMETRY_POINTS_END_ATTR,
            Sdf.ValueTypeNames.Int,
            prefix.points_end,
        )
        self._set_attr(
            control,
            GENERATED_GEOMETRY_FACES_END_ATTR,
            Sdf.ValueTypeNames.Int,
            prefix.faces_end,
        )
        self._set_attr(
            control,
            GENERATED_GEOMETRY_INDICES_END_ATTR,
            Sdf.ValueTypeNames.Int,
            prefix.indices_end,
        )

    def truncate_mesh(self, stage, prefix: GeneratedGeometryPrefix) -> None:
        data = self.read_mesh(stage)
        validate_generated_geometry_prefix(data, prefix)
        self._write_mesh(
            stage,
            GeneratedGeometryMeshData(
                points=data.points[: prefix.points_end],
                face_vertex_counts=data.face_vertex_counts[: prefix.faces_end],
                face_vertex_indices=data.face_vertex_indices[: prefix.indices_end],
                display_color=data.display_color,
            ),
        )

    def set_mode(self, stage, mode: str) -> None:
        _gf, Sdf, _UsdGeom, _vt = self._imports()
        self._set_attr(
            self._control(stage, create=True),
            GENERATED_GEOMETRY_MODE_ATTR,
            Sdf.ValueTypeNames.Token,
            str(mode),
        )

    def set_revision(self, stage, revision: int) -> None:
        _gf, Sdf, _UsdGeom, _vt = self._imports()
        self._set_attr(
            self._control(stage, create=True),
            GENERATED_GEOMETRY_REVISION_ATTR,
            Sdf.ValueTypeNames.Int,
            int(revision),
        )

    def read_control_state(self, stage) -> dict[str, Any]:
        control = self._control(stage)
        state = {}
        for attr_name in self._state_attribute_names(control):
            attr = control.GetAttribute(attr_name)
            if not attr or not attr.IsValid():
                raise ValueError(
                    f"generated geometry state attribute does not exist: {attr_name}"
                )
            state[attr_name] = attr.Get()
        return state

    def write_control_state(self, stage, state: Mapping[str, Any]) -> None:
        control = self._control(stage, create=True)
        declared = set(self._state_attribute_names(control))
        undeclared = set(state).difference(declared)
        if undeclared:
            raise ValueError(
                "generated geometry checkpoint contains undeclared state attributes: "
                f"{sorted(undeclared)}"
            )
        for attr_name, value in state.items():
            attr = control.GetAttribute(attr_name)
            if not attr or not attr.IsValid():
                raise ValueError(
                    f"generated geometry state attribute does not exist: {attr_name}"
                )
            attr.Set(value)

    def _state_attribute_names(self, control) -> list[str]:
        value = self._get_attr(
            control,
            GENERATED_GEOMETRY_STATE_ATTRIBUTES_ATTR,
            [],
        )
        names = [str(item) for item in (value or [])]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(
                "generatedGeometry:stateAttributes must contain unique non-empty names"
            )
        return names

    def clear(self, stage) -> None:
        if self.mesh_exists(stage):
            self._write_mesh(
                stage,
                GeneratedGeometryMeshData(
                    points=np.empty((0, 3), dtype=np.float32),
                    face_vertex_counts=np.empty((0,), dtype=np.int32),
                    face_vertex_indices=np.empty((0,), dtype=np.int32),
                    display_color=np.array([[0.005, 0.004, 0.003]], dtype=np.float32),
                ),
            )
        else:
            self._write_prefix(stage, GeneratedGeometryPrefix(0, 0, 0))


class GeneratedGeometryRuntime:
    """Discover and coordinate every registered generated-geometry source."""

    def __init__(self, backend_type=UsdGeneratedGeometryBackend):
        self.backend_type = backend_type
        self._recorders: dict[str, GeneratedGeometryRecorder] = {}
        self._restore_managers: dict[str, GeneratedGeometryStateManager] = {}

    def discover(self, stage) -> dict[str, Any]:
        discovered: dict[str, Any] = {}
        for backend in self.backend_type.discover(stage):
            source_key = generated_geometry_key(backend.mesh_prim_path)
            if source_key in discovered:
                raise ValueError(
                    f"duplicate generated geometry source key: {source_key}"
                )
            discovered[source_key] = backend
        return discovered

    def capture_checkpoints(
        self,
        stage,
    ) -> dict[str, GeneratedGeometryCheckpoint]:
        checkpoints: dict[str, GeneratedGeometryCheckpoint] = {}
        for source_key, backend in self.discover(stage).items():
            checkpoint = GeneratedGeometryStateManager(backend).capture_checkpoint(
                stage
            )
            if checkpoint is not None:
                checkpoints[source_key] = checkpoint
        return checkpoints

    @staticmethod
    def _missing_checkpoint() -> GeneratedGeometryCheckpoint:
        return GeneratedGeometryCheckpoint(
            control_exists=False,
            mesh_exists=False,
            prefix=GeneratedGeometryPrefix(0, 0, 0),
            revision=0,
            control_state={},
        )

    def begin_restore(
        self,
        stage,
        checkpoints: Mapping[str, GeneratedGeometryCheckpoint] | None,
    ) -> None:
        checkpoint_map = checkpoints or {}
        self._restore_managers = {}
        for source_key, backend in self.discover(stage).items():
            manager = GeneratedGeometryStateManager(backend)
            manager.begin_restore(
                stage,
                checkpoint_map.get(source_key, self._missing_checkpoint()),
            )
            self._restore_managers[source_key] = manager

    def finish_restore(self, stage) -> None:
        managers = self._restore_managers
        if not managers:
            managers = {
                source_key: GeneratedGeometryStateManager(backend)
                for source_key, backend in self.discover(stage).items()
            }
        for manager in managers.values():
            manager.finish_restore(stage)
        self._restore_managers = {}

    def reset_live(self, stage) -> None:
        for backend in self.discover(stage).values():
            GeneratedGeometryStateManager(backend).reset_live(stage)

    def start_recording(
        self,
        stage,
        output_dir: str | os.PathLike[str],
        timestamp: float,
    ) -> int:
        self._recorders = {}
        for source_key, backend in self.discover(stage).items():
            recorder = GeneratedGeometryRecorder(backend)
            if recorder.start(stage, output_dir, timestamp):
                self._recorders[source_key] = recorder
        return len(self._recorders)

    def sample(self, stage, timestamp: float) -> None:
        for recorder in self._recorders.values():
            recorder.sample(stage, timestamp)

    def stop_recording(
        self,
        stage,
        timestamp: float,
    ) -> dict[str, dict[str, Any]]:
        manifests: dict[str, dict[str, Any]] = {}
        try:
            for source_key, recorder in self._recorders.items():
                manifest = recorder.stop(stage, timestamp)
                if manifest is not None:
                    manifests[source_key] = manifest
        finally:
            self._recorders = {}
        return manifests


def _is_monotonic(
    previous: GeneratedGeometryPrefix,
    current: GeneratedGeometryPrefix,
) -> bool:
    return (
        previous.points_end <= current.points_end
        and previous.faces_end <= current.faces_end
        and previous.indices_end <= current.indices_end
    )


def merge_generated_geometry_samples_into_frames(
    frames: list[dict[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    source_id: str,
    artifact: GeneratedGeometryMeshData,
) -> None:
    if not samples:
        raise ValueError("generated geometry recording has no baseline sample")
    ordered = sorted(samples, key=lambda item: float(item["time_stamp"]))
    parsed = [
        (
            float(item["time_stamp"]),
            GeneratedGeometryPrefix.from_mapping(item["prefix"]),
        )
        for item in ordered
    ]
    sample_index = 0
    previous_prefix: GeneratedGeometryPrefix | None = None
    for frame in frames:
        if "time_stamp" not in frame:
            raise ValueError(
                "state frame is missing time_stamp required for generated geometry merge"
            )
        frame_timestamp = float(frame["time_stamp"])
        while sample_index + 1 < len(parsed) and parsed[sample_index + 1][0] <= frame_timestamp:
            sample_index += 1
        prefix = parsed[sample_index][1]
        validate_generated_geometry_prefix(artifact, prefix)
        if previous_prefix is not None and not _is_monotonic(previous_prefix, prefix):
            raise ValueError(
                "retained generated geometry frame prefixes must be monotonic"
            )
        frame.setdefault("generated_geometry", {})[source_id] = prefix.to_dict()
        previous_prefix = prefix


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _resolve_artifact(root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError(
            f"generated geometry artifact path must be relative: {relative_path}"
        )
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"generated geometry artifact path escapes episode root: {relative_path}"
        ) from exc
    return candidate


def _generated_geometry_sidecar_dirs(episode_root: Path) -> list[Path]:
    generated_geometry_root = episode_root / GENERATED_GEOMETRY_DIR
    if not generated_geometry_root.is_dir():
        return []
    sidecar_dirs = []
    for source_dir in sorted(generated_geometry_root.iterdir()):
        if not source_dir.is_dir():
            continue
        manifest_path = source_dir / GENERATED_GEOMETRY_MANIFEST_SIDECAR
        frames_path = source_dir / GENERATED_GEOMETRY_FRAMES_SIDECAR
        if not manifest_path.exists() and not frames_path.exists():
            continue
        if not manifest_path.is_file() or not frames_path.is_file():
            raise FileNotFoundError(
                "generated geometry recording requires both manifest and frames "
                f"sidecars in {source_dir}"
            )
        sidecar_dirs.append(source_dir)
    return sidecar_dirs


def merge_generated_geometry_sidecars(
    root: str | os.PathLike[str],
    state: dict[str, Any],
) -> bool:
    episode_root = Path(root)
    source_dirs = _generated_geometry_sidecar_dirs(episode_root)
    if not source_dirs:
        return False

    merged_state = deepcopy(state)
    frames = merged_state.get("frames")
    if not isinstance(frames, list):
        raise ValueError("state.json frames must be a list")
    sidecars_to_delete: list[Path] = []
    for source_dir in source_dirs:
        manifest_path = source_dir / GENERATED_GEOMETRY_MANIFEST_SIDECAR
        frames_path = source_dir / GENERATED_GEOMETRY_FRAMES_SIDECAR
        manifest_doc = _load_json(manifest_path)
        frames_doc = _load_json(frames_path)
        source_id = manifest_doc.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(
                "generated geometry manifest source_id must be a non-empty string"
            )
        if source_id != source_dir.name:
            raise ValueError(
                "generated geometry source key does not match its directory name"
            )
        if frames_doc.get("source_id") != source_id:
            raise ValueError("generated geometry sidecar source_id mismatch")
        manifest = manifest_doc.get("manifest")
        if not isinstance(manifest, dict):
            raise ValueError("generated geometry manifest payload must be an object")
        if manifest.get("schema") != GENERATED_GEOMETRY_SCHEMA:
            raise ValueError(
                "unsupported generated geometry schema: "
                f"{manifest.get('schema')!r}"
            )
        if manifest.get("append_only") is not True:
            raise ValueError("generated geometry v1 manifest must be append_only")
        if manifest.get("coordinate_space") != "record_world":
            raise ValueError(
                "generated geometry v1 coordinate_space must be record_world"
            )
        prim_path = manifest.get("prim_path")
        expected_source_id = generated_geometry_key(prim_path)
        if expected_source_id != source_id:
            raise ValueError(
                "generated geometry source key does not match manifest prim_path"
            )
        generated_geometry_key(manifest.get("control_prim_path"))
        artifact_rel = manifest.get("file")
        if not isinstance(artifact_rel, str) or not artifact_rel:
            raise ValueError(
                "generated geometry manifest file must be a non-empty relative path"
            )
        artifact = load_generated_geometry_artifact(
            _resolve_artifact(episode_root, artifact_rel)
        )
        samples = frames_doc.get("frames")
        if not isinstance(samples, list):
            raise ValueError("generated geometry frames sidecar frames must be a list")
        merge_generated_geometry_samples_into_frames(
            frames, samples, source_id, artifact
        )
        merged_state.setdefault("generated_geometry", {})[source_id] = manifest
        sidecars_to_delete.extend([manifest_path, frames_path])

    state.clear()
    state.update(merged_state)
    for sidecar in sidecars_to_delete:
        sidecar.unlink()
    return True
