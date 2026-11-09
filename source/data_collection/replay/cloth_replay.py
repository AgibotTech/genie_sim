"""Replay deformable mesh points without running cloth physics.

Each frame reads world-space points from a deformable NPZ file. Clone offsets
are applied per environment, then points are transformed into the mesh's local
coordinates before updating the USD mesh.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from common.deformable_compat import (
    COLLISION_SCHEMA_NAMES, DEFORMABLE_SCHEMA_NAMES, paths_in_deformable_subtrees,
)
from common.base_utils.logger import logger


class ClothReplayer:
    """Update USD cloth meshes from each frame's deformable entries.

    A state.json frame has entries like::
        {"time_stamp", "objects":..., "deformable":[
            {"name", "prim_path", "file": "<name>/<idx>.npz", "n_nodes"}, ...]}
    NPZ files are located below ``{input_dir}/deformable``.
    """

    def __init__(self, input_dir: str, base_record_path: str = "/World", subdivision_levels: int = 2):
        """
        Args:
            input_dir: Recording root containing state.json and deformable/.
            base_record_path: Recorded prim prefix used to derive clone paths.
        """
        self.input_dir = input_dir.rstrip("/")
        self.deformable_dir = os.path.join(self.input_dir, "deformable")
        self._base_record_path = base_record_path
        # Cache USD meshes by prim path.
        self._mesh_cache: Dict[str, Any] = {}
        # Replay-only refinement, enabled by default. Input NPZ and source USD stay untouched.
        self._subdivision_levels = subdivision_levels
        if type(subdivision_levels) is not int or subdivision_levels not in (0, 1, 2, 3):
            raise ValueError("cloth_subdivision_levels must be an integer 0..3")
        self._subdivided_meshes: Dict[str, Any] = {}

    @staticmethod
    def _to_vec3f_array(P: np.ndarray):
        """Convert an (N, 3) array to Vt.Vec3fArray with a scalar fallback."""
        from pxr import Gf, Vt  # noqa: PLC0415
        try:
            return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(P, dtype=np.float32))
        except Exception:
            return Vt.Vec3fArray([Gf.Vec3f(float(a), float(b), float(c)) for a, b, c in P])

    def _resolve_mesh_paths(self, prim_path: str, env_paths: List[str]) -> List[str]:
        """Map a recorded mesh prim path into each cloned environment.
        """
        if not prim_path.startswith(self._base_record_path + "/"):
            # Keep absolute paths outside the recorded base unchanged.
            return [prim_path] * max(1, len(env_paths))
        suffix = prim_path[len(self._base_record_path):].lstrip("/")
        out = []
        for ep in env_paths:
            out.append(ep.rstrip("/") + "/" + suffix)
        return out or [prim_path]

    def setup(self, stage, deformable_entries: List[Dict[str, Any]], env_paths: List[str]):
        """Remove deformable physics schemas while keeping the render topology."""
        for entry in deformable_entries:
            for mesh_path in self._resolve_mesh_paths(entry["prim_path"], env_paths):
                self._strip_deformable_schema_once(stage, mesh_path)

    def apply_frame(self, stage, deformable_entries: Optional[List[Dict[str, Any]]],
                    env_paths: List[str],
                    env_clone_positions: Optional[np.ndarray] = None,
                    env_clone_orientations: Optional[np.ndarray] = None):
        """Apply one frame of cloth points to every cloned environment.

        NPZ points are world-space positions for the source environment.
        First apply each clone's rotation and translation, then transform the
        result into that mesh's local frame. This respects both the clone
        offset and object poses applied by kinematic replay.
        """
        if not deformable_entries:
            return
        n_env = max(1, len(env_paths))
        cps = self._normalize_clone_positions(env_clone_positions, n_env)
        Rs = self._normalize_clone_rotations(env_clone_orientations, n_env)
        for entry in deformable_entries:
            file_rel = entry.get("file")
            if not file_rel:
                continue
            npz_path = os.path.join(self.deformable_dir, file_rel)
            if not os.path.isfile(npz_path):
                logger.warning(f"[cloth_replay] npz not found: {npz_path}")
                continue
            try:
                data = np.load(npz_path)
                points_world = np.asarray(data["points"], dtype=np.float64)
            except Exception as exc:
                logger.warning(f"[cloth_replay] load {npz_path} failed: {exc}")
                continue
            mesh_paths = self._resolve_mesh_paths(entry["prim_path"], env_paths)
            for env_i, mesh_path in enumerate(mesh_paths):
                self._strip_deformable_schema_once(stage, mesh_path)
                cp_i = cps[env_i] if env_i < len(cps) else np.zeros(3)
                R_i = Rs[env_i] if env_i < len(Rs) else np.eye(3)
                # Apply the clone's SE(3) transform to source-world points.
                points_world_env_i = points_world @ R_i.T + cp_i
                points_local = self._world_to_mesh_local(stage, mesh_path, points_world_env_i)
                if points_local is None:
                    continue
                self._set_mesh_points(stage, mesh_path, points_local)

    @staticmethod
    def _normalize_clone_positions(cps, n_env: int) -> np.ndarray:
        if cps is None:
            return np.zeros((n_env, 3), dtype=np.float64)
        arr = np.asarray(cps, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 3:
            arr = arr.reshape(1, 3)
        return arr

    @staticmethod
    def _normalize_clone_rotations(cqs, n_env: int) -> np.ndarray:
        """Convert wxyz quaternions to rotation matrices.

        Missing or invalid input returns identity matrices.
        """
        if cqs is None:
            return np.tile(np.eye(3, dtype=np.float64), (n_env, 1, 1))
        arr = np.asarray(cqs, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 4:
            arr = arr.reshape(1, 4)
        if arr.ndim != 2 or arr.shape[-1] != 4:
            return np.tile(np.eye(3, dtype=np.float64), (n_env, 1, 1))
        out = np.empty((arr.shape[0], 3, 3), dtype=np.float64)
        for i, q in enumerate(arr):
            w, x, y, z = q
            n = w * w + x * x + y * y + z * z
            if n < 1e-12:
                out[i] = np.eye(3)
                continue
            s = 2.0 / n
            out[i] = np.array([
                [1.0 - s * (y * y + z * z), s * (x * y - z * w),       s * (x * z + y * w)],
                [s * (x * y + z * w),       1.0 - s * (x * x + z * z), s * (y * z - x * w)],
                [s * (x * z - y * w),       s * (y * z + x * w),       1.0 - s * (x * x + y * y)],
            ])
        return out

    def _world_to_mesh_local(self, stage, mesh_prim_path: str, points_world: np.ndarray):
        """Transform world-space points into the mesh's local frame.

        USD uses row vectors here: ``world = local @ M``. Read M each frame
        because kinematic replay updates the parent pose.
        """
        try:
            from pxr import Usd, UsdGeom  # noqa: PLC0415
        except Exception:
            return None
        prim = stage.GetPrimAtPath(mesh_prim_path)
        if not prim or not prim.IsValid():
            logger.warning(f"[cloth_replay] mesh prim missing: {mesh_prim_path}")
            return None
        xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        gf_M = xf_cache.GetLocalToWorldTransform(prim)
        M = np.array(gf_M, dtype=np.float64).reshape(4, 4)
        try:
            invM = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            logger.warning(f"[cloth_replay] singular xform for {mesh_prim_path}, fallback identity")
            invM = np.eye(4, dtype=np.float64)
        N = points_world.shape[0]
        wp_h = np.concatenate(
            [points_world, np.ones((N, 1), dtype=np.float64)], axis=1
        )   # (N, 4)
        local_h = wp_h @ invM   # (N, 4) row-vector × inv(M)
        return local_h[:, :3].astype(np.float32)

    def _strip_deformable_schema_once(self, stage, mesh_prim_path: str):
        """Strip the entry's entire deformable subtree, never adjacent rigid bodies."""
        if not hasattr(self, "_schema_stripped"):
            self._schema_stripped = set()
        if mesh_prim_path in self._schema_stripped:
            return
        self._schema_stripped.add(mesh_prim_path)
        try:
            records = [
                (str(prim.GetPath()), tuple(prim.GetAppliedSchemas()))
                for prim in stage.Traverse()
            ]
            cleanup_paths = paths_in_deformable_subtrees(
                records, entry_paths=[mesh_prim_path]
            )
            removed = []
            for path in sorted(cleanup_paths):
                prim = stage.GetPrimAtPath(path)
                applied = list(prim.GetAppliedSchemas())
                for name in DEFORMABLE_SCHEMA_NAMES + COLLISION_SCHEMA_NAMES:
                    if name in applied:
                        try:
                            prim.RemoveAppliedSchema(name)
                            removed.append((path, name))
                        except Exception as exc:
                            logger.warning(
                                f"[cloth_replay] remove {name} on {path} failed: {exc}"
                            )
            if removed:
                logger.info(
                    f"[cloth_replay] removed {len(removed)} schemas from "
                    f"deformable subtree for {mesh_prim_path}"
                )
        except Exception as exc:
            logger.warning(f"[cloth_replay] strip schema failed for {mesh_prim_path}: {exc}")

    def _set_mesh_points(self, stage, mesh_prim_path: str, points: np.ndarray):
        try:
            from pxr import UsdGeom  # noqa: PLC0415
        except Exception:
            return
        mesh = self._mesh_cache.get(mesh_prim_path)
        if mesh is None:
            prim = stage.GetPrimAtPath(mesh_prim_path)
            if not prim or not prim.IsValid():
                logger.warning(f"[cloth_replay] mesh prim missing: {mesh_prim_path}")
                self._mesh_cache[mesh_prim_path] = False    # negative cache
                return
            mesh = UsdGeom.Mesh(prim) if prim.IsA(UsdGeom.Mesh) else UsdGeom.PointBased(prim)
            if not mesh:
                logger.warning(f"[cloth_replay] prim has no points: {mesh_prim_path}")
                self._mesh_cache[mesh_prim_path] = False
                return
            self._mesh_cache[mesh_prim_path] = mesh
        elif mesh is False:
            return
        try:
            # TetMesh simulation nodes keep their tetrahedral and surface connectivity.
            # Triangle-only cloth refinement applies only to USD Mesh geometry.
            if self._subdivision_levels and isinstance(mesh, UsdGeom.Mesh):
                from replay.cloth_subdivision import SubdividedUsdMesh
                refined = self._subdivided_meshes.get(mesh_prim_path)
                if refined is None:
                    if len(mesh.GetPointsAttr().Get()) != len(points):
                        # USD clones can inherit an already-refined env_0 mesh.
                        # Reuse its original-node mapping instead of subdividing again.
                        suffix = mesh_prim_path.split('/Objects/', 1)[-1]
                        source = next((v for p, v in self._subdivided_meshes.items()
                                       if p.split('/Objects/', 1)[-1] == suffix
                                       and v.mapping.num_vertices == len(points)), None)
                        if source is None or not np.array_equal(
                                np.asarray(mesh.GetFaceVertexIndicesAttr().Get()),
                                source.mapping.faces.ravel()):
                            raise ValueError("Recorded cloth vertices do not match source mesh")
                        refined = SubdividedUsdMesh.__new__(SubdividedUsdMesh)
                        refined.mesh = mesh
                        refined.mapping = source.mapping
                    else:
                        refined = SubdividedUsdMesh(mesh, self._subdivision_levels)
                    self._subdivided_meshes[mesh_prim_path] = refined
                    logger.info("[cloth_replay] subdivided %s: %d -> %d vertices, %d faces",
                                mesh_prim_path, len(points), refined.mapping.weights.shape[0],
                                len(refined.mapping.faces))
                refined.update(points)
                return
            mesh.GetPointsAttr().Set(self._to_vec3f_array(points))
        except Exception as exc:
            logger.warning(f"[cloth_replay] set points failed for {mesh_prim_path}: {exc}")
