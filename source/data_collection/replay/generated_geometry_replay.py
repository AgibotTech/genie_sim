"""Frame-exact replay for registered append-only generated meshes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from server.generated_geometry_state import (
    GENERATED_GEOMETRY_FACES_END_ATTR,
    GENERATED_GEOMETRY_INDICES_END_ATTR,
    GENERATED_GEOMETRY_MODE_ATTR,
    GENERATED_GEOMETRY_POINTS_END_ATTR,
    GENERATED_GEOMETRY_RECORDABLE_ATTR,
    GENERATED_GEOMETRY_SCHEMA,
    GENERATED_GEOMETRY_TARGET_REL,
    GeneratedGeometryMeshData,
    GeneratedGeometryPrefix,
    generated_geometry_key,
    load_generated_geometry_artifact,
    validate_generated_geometry_prefix,
)


def resolve_generated_geometry_paths(
    recorded_prim_path: str,
    env_paths: Sequence[str],
    *,
    base_record_path: str = "/World",
) -> list[str]:
    generated_geometry_key(recorded_prim_path)
    if not recorded_prim_path.startswith(base_record_path + "/"):
        return [recorded_prim_path] * max(1, len(env_paths))
    suffix = recorded_prim_path[len(base_record_path) :].lstrip("/")
    return [env.rstrip("/") + "/" + suffix for env in env_paths] or [
        recorded_prim_path
    ]


def _quat_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape != (4,):
        raise ValueError("clone orientation must be a wxyz quaternion")
    norm_squared = float(np.dot(q, q))
    if norm_squared < 1e-12:
        raise ValueError("clone orientation quaternion must be non-zero")
    w, x, y, z = q
    scale = 2.0 / norm_squared
    return np.array(
        [
            [
                1.0 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
            ],
            [
                scale * (x * y + z * w),
                1.0 - scale * (x * x + z * z),
                scale * (y * z - x * w),
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1.0 - scale * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def transform_record_world_points(
    points: np.ndarray,
    clone_position: Sequence[float],
    clone_orientation_wxyz: Sequence[float],
) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    position = np.asarray(clone_position, dtype=np.float64).reshape(-1)
    if position.shape != (3,):
        raise ValueError("clone position must have shape (3,)")
    rotation = _quat_wxyz_to_matrix(clone_orientation_wxyz)
    return source @ rotation.T + position


def world_to_local_points(
    points_world: np.ndarray,
    local_to_world: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(local_to_world, dtype=np.float64).reshape(4, 4)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "generated geometry local-to-world transform is singular"
        ) from exc
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    # Gf matrices transform row vectors: world = local @ M.
    return (homogeneous @ inverse)[:, :3].astype(np.float32)


class _GeneratedGeometrySourceReplayer:
    def __init__(
        self,
        input_dir: Path,
        source_key: str,
        manifest: Mapping[str, Any],
        frames: list[dict[str, Any]],
        *,
        base_record_path: str,
    ):
        self.input_dir = input_dir
        self.source_key = source_key
        self.manifest = dict(manifest)
        self.base_record_path = base_record_path
        self._validate_manifest()
        artifact_path = self._resolve_artifact_path(self.manifest["file"])
        self.artifact = load_generated_geometry_artifact(artifact_path)
        self.prefixes = self._load_prefixes(frames)
        self._mesh_cache: dict[tuple[str, str], tuple[Any, Any]] = {}

    def _validate_manifest(self) -> None:
        if self.manifest.get("schema") != GENERATED_GEOMETRY_SCHEMA:
            raise ValueError(
                "unsupported generated geometry schema: "
                f"{self.manifest.get('schema')!r}"
            )
        if self.manifest.get("append_only") is not True:
            raise ValueError("generated geometry v1 manifest must be append_only")
        if self.manifest.get("coordinate_space") != "record_world":
            raise ValueError(
                "generated geometry v1 coordinate_space must be record_world"
            )
        file_value = self.manifest.get("file")
        if not isinstance(file_value, str) or not file_value:
            raise ValueError(
                "generated geometry manifest file must be a non-empty relative path"
            )
        if Path(file_value).is_absolute():
            raise ValueError(
                "generated geometry manifest file must be a relative path"
            )
        prim_path = self.manifest.get("prim_path")
        if generated_geometry_key(prim_path) != self.source_key:
            raise ValueError(
                "generated geometry source key does not match manifest prim_path"
            )
        generated_geometry_key(self.manifest.get("control_prim_path"))

    def _resolve_artifact_path(self, relative_path: str) -> Path:
        candidate = (self.input_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.input_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                "generated geometry artifact path escapes episode root: "
                f"{relative_path}"
            ) from exc
        return candidate

    def _load_prefixes(
        self,
        frames: list[dict[str, Any]],
    ) -> list[GeneratedGeometryPrefix]:
        prefixes: list[GeneratedGeometryPrefix] = []
        previous: GeneratedGeometryPrefix | None = None
        for index, frame in enumerate(frames):
            geometry = (
                frame.get("generated_geometry")
                if isinstance(frame, Mapping)
                else None
            )
            value = (
                geometry.get(self.source_key)
                if isinstance(geometry, Mapping)
                else None
            )
            if not isinstance(value, Mapping):
                raise ValueError(
                    "generated geometry prefix missing from frame "
                    f"{index} for source {self.source_key}"
                )
            prefix = GeneratedGeometryPrefix.from_mapping(value)
            validate_generated_geometry_prefix(self.artifact, prefix)
            if previous is not None and (
                prefix.points_end < previous.points_end
                or prefix.faces_end < previous.faces_end
                or prefix.indices_end < previous.indices_end
            ):
                raise ValueError(
                    "generated geometry prefixes must be monotonic at frame "
                    f"{index} for source {self.source_key}"
                )
            prefixes.append(prefix)
            previous = prefix
        return prefixes

    def frame_data(self, frame_index: int) -> GeneratedGeometryMeshData:
        prefix = self.prefixes[frame_index]
        return GeneratedGeometryMeshData(
            points=self.artifact.points[: prefix.points_end],
            face_vertex_counts=self.artifact.face_vertex_counts[
                : prefix.faces_end
            ],
            face_vertex_indices=self.artifact.face_vertex_indices[
                : prefix.indices_end
            ],
            display_color=self.artifact.display_color,
        )

    @staticmethod
    def _set_custom_attr(prim, name: str, type_name, value) -> None:
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(name, type_name, custom=True)
        attr.Set(value)

    def _define_pair(self, stage, control_path: str, mesh_path: str):
        from pxr import Sdf, UsdGeom  # noqa: PLC0415

        cache_key = (control_path, mesh_path)
        cached = self._mesh_cache.get(cache_key)
        if cached is not None:
            return cached
        control = stage.GetPrimAtPath(control_path)
        if not control or not control.IsValid():
            control = UsdGeom.Xform.Define(stage, control_path).GetPrim()
        self._set_custom_attr(
            control,
            GENERATED_GEOMETRY_MODE_ATTR,
            Sdf.ValueTypeNames.Token,
            "replay",
        )
        self._set_custom_attr(
            control,
            GENERATED_GEOMETRY_RECORDABLE_ATTR,
            Sdf.ValueTypeNames.Bool,
            True,
        )
        target = control.GetRelationship(GENERATED_GEOMETRY_TARGET_REL)
        if not target or not target.IsValid():
            target = control.CreateRelationship(
                GENERATED_GEOMETRY_TARGET_REL,
                custom=True,
            )
        target.SetTargets([Sdf.Path(mesh_path)])
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        self._mesh_cache[cache_key] = (control, mesh)
        return control, mesh

    @staticmethod
    def _to_vec3f_array(points: np.ndarray):
        from pxr import Gf, Vt  # noqa: PLC0415

        return Vt.Vec3fArray(
            [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points]
        )

    def _write_mesh(
        self,
        control,
        mesh,
        points: np.ndarray,
        data: GeneratedGeometryMeshData,
    ) -> None:
        from pxr import Sdf, UsdGeom, Vt  # noqa: PLC0415

        points_array = self._to_vec3f_array(points)
        mesh.CreatePointsAttr().Set(points_array)
        mesh.CreateFaceVertexCountsAttr().Set(
            Vt.IntArray([int(value) for value in data.face_vertex_counts])
        )
        mesh.CreateFaceVertexIndicesAttr().Set(
            Vt.IntArray([int(value) for value in data.face_vertex_indices])
        )
        mesh.CreateDisplayColorAttr().Set(
            self._to_vec3f_array(data.display_color)
        )
        extent_attr = mesh.CreateExtentAttr()
        if len(points_array):
            extent_attr.Set(UsdGeom.PointBased.ComputeExtent(points_array))
        else:
            extent_attr.Clear()
        self._set_custom_attr(
            control,
            GENERATED_GEOMETRY_POINTS_END_ATTR,
            Sdf.ValueTypeNames.Int,
            len(points),
        )
        self._set_custom_attr(
            control,
            GENERATED_GEOMETRY_FACES_END_ATTR,
            Sdf.ValueTypeNames.Int,
            len(data.face_vertex_counts),
        )
        self._set_custom_attr(
            control,
            GENERATED_GEOMETRY_INDICES_END_ATTR,
            Sdf.ValueTypeNames.Int,
            len(data.face_vertex_indices),
        )

    def _resolved_pairs(
        self,
        env_paths: Sequence[str],
    ) -> list[tuple[str, str]]:
        control_paths = resolve_generated_geometry_paths(
            self.manifest["control_prim_path"],
            env_paths,
            base_record_path=self.base_record_path,
        )
        mesh_paths = resolve_generated_geometry_paths(
            self.manifest["prim_path"],
            env_paths,
            base_record_path=self.base_record_path,
        )
        if len(control_paths) != len(mesh_paths):
            raise ValueError(
                "generated geometry control and mesh clone counts do not match"
            )
        return list(zip(control_paths, mesh_paths))

    def setup(self, stage, env_paths: Sequence[str]) -> None:
        empty = GeneratedGeometryMeshData(
            points=self.artifact.points[:0],
            face_vertex_counts=self.artifact.face_vertex_counts[:0],
            face_vertex_indices=self.artifact.face_vertex_indices[:0],
            display_color=self.artifact.display_color,
        )
        for control_path, mesh_path in self._resolved_pairs(env_paths):
            control, mesh = self._define_pair(stage, control_path, mesh_path)
            self._write_mesh(control, mesh, empty.points, empty)

    def apply_frame(
        self,
        stage,
        frame_index: int,
        env_paths: Sequence[str],
        *,
        env_clone_positions: np.ndarray | None = None,
        env_clone_orientations: np.ndarray | None = None,
    ) -> None:
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        data = self.frame_data(frame_index)
        pairs = self._resolved_pairs(env_paths)
        count = len(pairs)
        positions = self._clone_positions(env_clone_positions, count)
        orientations = self._clone_orientations(env_clone_orientations, count)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for index, (control_path, mesh_path) in enumerate(pairs):
            control, mesh = self._define_pair(stage, control_path, mesh_path)
            points_world = transform_record_world_points(
                data.points,
                positions[index],
                orientations[index],
            )
            matrix = np.array(
                xform_cache.GetLocalToWorldTransform(mesh.GetPrim()),
                dtype=np.float64,
            )
            points_local = world_to_local_points(points_world, matrix)
            self._write_mesh(control, mesh, points_local, data)

    @staticmethod
    def _clone_positions(values, count: int) -> np.ndarray:
        if values is None:
            return np.zeros((count, 3), dtype=np.float64)
        result = np.asarray(values, dtype=np.float64)
        if result.ndim == 1:
            result = result.reshape(1, 3)
        if result.shape != (count, 3):
            raise ValueError(
                f"expected {count} clone positions, got {result.shape}"
            )
        return result

    @staticmethod
    def _clone_orientations(values, count: int) -> np.ndarray:
        if values is None:
            result = np.zeros((count, 4), dtype=np.float64)
            result[:, 0] = 1.0
            return result
        result = np.asarray(values, dtype=np.float64)
        if result.ndim == 1:
            result = result.reshape(1, 4)
        if result.shape != (count, 4):
            raise ValueError(
                f"expected {count} clone orientations, got {result.shape}"
            )
        return result


class GeneratedGeometryReplayer:
    def __init__(
        self,
        input_dir: str | os.PathLike[str],
        state: Mapping[str, Any],
        *,
        base_record_path: str = "/World",
    ):
        self.input_dir = Path(input_dir)
        geometry = state.get("generated_geometry")
        if not isinstance(geometry, Mapping) or not geometry:
            raise ValueError("state has no generated geometry manifests")
        frames = state.get("frames")
        if not isinstance(frames, list):
            raise ValueError("state frames must be a list")
        self.sources: dict[str, _GeneratedGeometrySourceReplayer] = {}
        control_paths: set[str] = set()
        for source_key in sorted(geometry):
            manifest = geometry[source_key]
            if not isinstance(source_key, str) or not source_key:
                raise ValueError(
                    "generated geometry source key must be a non-empty string"
                )
            if not isinstance(manifest, Mapping):
                raise ValueError("generated geometry manifest must be an object")
            control_prim_path = manifest.get("control_prim_path")
            if control_prim_path in control_paths:
                raise ValueError(
                    "generated geometry sources must use distinct control prims"
                )
            control_paths.add(control_prim_path)
            self.sources[source_key] = _GeneratedGeometrySourceReplayer(
                self.input_dir,
                source_key,
                manifest,
                frames,
                base_record_path=base_record_path,
            )

    @classmethod
    def from_state(
        cls,
        input_dir: str | os.PathLike[str],
        state: Mapping[str, Any],
        *,
        base_record_path: str = "/World",
    ) -> "GeneratedGeometryReplayer | None":
        geometry = state.get("generated_geometry")
        if geometry is None or geometry == {}:
            return None
        if not isinstance(geometry, Mapping):
            raise ValueError("state generated_geometry must be an object")
        return cls(
            input_dir,
            state,
            base_record_path=base_record_path,
        )

    def setup(self, stage, env_paths: Sequence[str]) -> None:
        for source in self.sources.values():
            source.setup(stage, env_paths)

    def apply_frame(
        self,
        stage,
        frame_index: int,
        env_paths: Sequence[str],
        *,
        env_clone_positions: np.ndarray | None = None,
        env_clone_orientations: np.ndarray | None = None,
    ) -> None:
        for source in self.sources.values():
            source.apply_frame(
                stage,
                frame_index,
                env_paths,
                env_clone_positions=env_clone_positions,
                env_clone_orientations=env_clone_orientations,
            )
