# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Collect Newton's actual viewer geometry without constructing a renderer.

Newton's ``ViewerBase.set_model`` creates the same primitive meshes and
shape batches used by its viewers. Its ``log_state`` supplies world transforms;
the scene plugin's original ``on_render`` supplies any custom deforming meshes.
This sink only copies those outputs to host arrays. It never steps the engine,
changes its model, or creates an OpenGL/Kit context.

The first capture contains every mesh. Later captures contain changed meshes
and the complete instance table; consumers retain previously received meshes.
Arrays belong to the returned snapshot and may be sent through a private,
trusted multiprocessing connection. A new consumer needs a new SceneSnapshot
so it receives the initial mesh table.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


def _load_viewer_base() -> type:
    # Lazy import keeps CPU-only consumers/tests independent of Newton/Warp.
    # Newton itself eagerly imports all viewer modules, but no ViewerGL is
    # constructed here and no GL renderer/window is used.
    from newton.viewer import ViewerBase

    return ViewerBase


def _array(value: Any, dtype: Any, width: int | None = None) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if width is not None:
        result = result.reshape(-1, width)
    if not np.isfinite(result).all():
        raise ValueError("scene snapshot contains non-finite geometry")
    return result


def _same(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(
        (
            np.array_equal(left[key], value)
            if isinstance(left[key], np.ndarray) or isinstance(value, np.ndarray)
            else left[key] == value
        )
        for key, value in right.items()
    )


class _SnapshotSink:
    """Concrete logging methods mixed with the lazily imported ViewerBase."""

    def __init__(self) -> None:
        self._snapshot_meshes: dict[str, dict[str, Any]] = {}
        self._snapshot_changed: dict[str, dict[str, Any]] = {}
        self._snapshot_instances: dict[str, dict[str, Any]] = {}
        self._snapshot_appearance: dict[str, dict[str, np.ndarray]] = {}
        self._snapshot_seen: set[str] = set()
        super().__init__()

    def begin_frame(self, time: float) -> None:
        super().begin_frame(time)
        self._snapshot_instances = {}
        self._snapshot_seen = set()

    def end_frame(self) -> None:
        # Chef changes these viewer options inside on_render, AFTER log_state.
        # Respect that change in the first capture too, without re-running the
        # plugin or exporting a visible duplicate of the raw FEM surface.
        triangles = self._snapshot_meshes.get("/model/triangles")
        if triangles is not None and triangles["hidden"] != (not self.show_triangles):
            triangles = {**triangles, "hidden": not self.show_triangles}
            self._snapshot_meshes["/model/triangles"] = triangles
            self._snapshot_changed["/model/triangles"] = triangles

    def log_mesh(
        self,
        name: str,
        points: Any,
        indices: Any,
        normals: Any = None,
        uvs: Any = None,
        texture: Any = None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: Any = None,
        roughness: float | None = None,
        metallic: float | None = None,
        dynamic: bool = False,
        opacity: float | None = None,
    ) -> None:
        vertices = _array(points, np.float32, 3)
        faces = _array(indices, np.int32).reshape(-1)
        if len(faces) % 3 or (len(faces) and (faces.min() < 0 or faces.max() >= len(vertices))):
            raise ValueError(f"invalid triangle indices for scene mesh {name!r}")
        entry: dict[str, Any] = {
            "points": vertices,
            "indices": faces,
            "hidden": bool(hidden),
            "backface_culling": bool(backface_culling),
        }
        for key, value, width in (("normals", normals, 3), ("uvs", uvs, 2), ("color", color, None)):
            if value is not None:
                entry[key] = _array(value, np.float32, width)
        if texture is not None:
            entry["texture"] = texture if isinstance(texture, str) else _array(texture, np.uint8)
        for key, value in (("roughness", roughness), ("metallic", metallic), ("opacity", opacity)):
            if value is not None:
                if not np.isfinite(value):
                    raise ValueError(f"non-finite {key} for scene mesh {name!r}")
                if key == "opacity" and not 0 <= value <= 1:
                    raise ValueError("scene mesh opacity must be in [0,1]")
                entry[key] = float(value)
        previous = self._snapshot_meshes.get(name)
        if previous is None or not _same(previous, entry):
            self._snapshot_meshes[name] = entry
            self._snapshot_changed[name] = entry
        self._snapshot_seen.add(name)

    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms: Any,
        scales: Any,
        colors: Any,
        materials: Any,
        hidden: bool = False,
        opacities: Any = None,
    ) -> None:
        if mesh not in self._snapshot_meshes:
            raise ValueError(f"scene instance {name!r} references unknown mesh {mesh!r}")
        transforms = np.empty((0, 7), dtype=np.float32) if xforms is None else _array(xforms, np.float32, 7)
        sizes = np.ones((len(transforms), 3), dtype=np.float32) if scales is None else _array(scales, np.float32, 3)
        if len(sizes) != len(transforms):
            raise ValueError(f"scene instance {name!r} has mismatched scales/transforms")
        entry: dict[str, Any] = {
            "mesh": mesh,
            "xforms": transforms,
            "scales": sizes,
            "hidden": bool(hidden),
        }
        # ViewerBase omits materials (and sometimes colors) after their first
        # submission. Preserve them so EVERY frame's instance table is complete.
        appearance = self._snapshot_appearance.setdefault(name, {})
        for key, value, width in (("colors", colors, 3), ("materials", materials, 4), ("opacities", opacities, 1)):
            if value is not None:
                array = _array(value, np.float32, width)
                if key == "opacities" and np.any((array < 0) | (array > 1)):
                    raise ValueError("scene instance opacities must be in [0,1]")
                appearance[key] = array
            cached = appearance.get(key)
            if cached is not None and len(cached) == len(transforms):
                entry[key] = cached.copy()
            elif value is not None:
                raise ValueError(f"scene instance {name!r} has mismatched {key}/transforms")
        self._snapshot_instances[name] = entry

    def log_points(self, name: str, points: Any, radii: Any = None, colors: Any = None, hidden: bool = False) -> None:
        if not hidden and points is not None and len(points):
            raise ValueError(f"visible point cloud {name!r} cannot be represented by a mesh snapshot")

    def log_lines(
        self, name: str, starts: Any, ends: Any, colors: Any, width: float = 0.01, hidden: bool = False
    ) -> None:
        if not hidden and starts is not None and len(starts):
            raise ValueError(f"visible debug lines {name!r} cannot be represented by a mesh snapshot")

    def log_array(self, name: str, array: Any) -> None:
        pass

    def log_scalar(self, name: str, value: Any, *, clear: bool = False, smoothing: int = 1) -> None:
        pass

    def apply_forces(self, state: Any) -> None:
        pass

    def close(self) -> None:
        self.clear_model()
        self._snapshot_meshes.clear()
        self._snapshot_changed.clear()
        self._snapshot_instances.clear()
        self._snapshot_appearance.clear()
        self._snapshot_seen.clear()


class SceneSnapshot:
    """Capture the built engine's viewer output, without advancing physics."""

    @staticmethod
    def authored_visuals_from_config(
        config: Mapping[str, Any], *, assets_root: Path | None
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Resolve the scene YAML's visual-only descriptor against its asset root.

        Shape prefixes belong to the parent snapshot filter; they are never
        forwarded to the worker or applied to the physical Newton model.
        """
        if not isinstance(config, Mapping):
            raise ValueError("newton.authored_visuals must be a mapping")
        allowed = {"asset_path", "root_prim", "hide_prims", "use_asset_lights", "body_visuals", "hide_shape_prefixes"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown authored_visuals fields: {', '.join(sorted(unknown))}")
        asset = config.get("asset_path")
        if not isinstance(asset, str) or not asset:
            raise ValueError("authored_visuals.asset_path must be a nonempty local asset path")
        path = Path(asset)
        if not path.is_absolute():
            if assets_root is None:
                raise ValueError("relative authored_visuals.asset_path requires manifest.base_path")
            path = assets_root / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"authored visual asset does not exist: {path}")

        def prim(value: Any, field: str) -> str:
            if (
                not isinstance(value, str)
                or not value.startswith("/")
                or any(part in {"", ".", ".."} for part in value.split("/")[1:])
            ):
                raise ValueError(f"authored_visuals.{field} must be an absolute prim path")
            return value

        def names(value: Any, field: str) -> tuple[str, ...]:
            if not isinstance(value, (list, tuple)) or any(not isinstance(name, str) or not name for name in value):
                raise ValueError(f"authored_visuals.{field} must be a list of nonempty strings")
            return tuple(value)

        hide = [prim(value, "hide_prims") for value in names(config.get("hide_prims", []), "hide_prims")]
        lights = config.get("use_asset_lights", False)
        if not isinstance(lights, bool):
            raise ValueError("authored_visuals.use_asset_lights must be a boolean")
        visuals = config.get("body_visuals", [])
        if not isinstance(visuals, (list, tuple)):
            raise ValueError("authored_visuals.body_visuals must be a list")
        body_visuals, bodies = [], set()
        for visual in visuals:
            if not isinstance(visual, Mapping) or set(visual) != {"body", "source_prim", "reference_pose"}:
                raise ValueError("each body_visual requires body, source_prim and reference_pose only")
            body = visual["body"]
            if not isinstance(body, str) or not body or "/" in body or body in bodies:
                raise ValueError("authored_visuals body names must be unique nonempty Newton body names")
            pose = _array(visual["reference_pose"], np.float64).reshape(-1)
            if len(pose) != 7 or np.linalg.norm(pose[3:]) < 1e-8:
                raise ValueError("authored_visuals.reference_pose requires finite xyz + nonzero quaternion xyzw")
            body_visuals.append(
                {
                    "body": body,
                    "source_prim": prim(visual["source_prim"], "source_prim"),
                    "reference_pose": pose.tolist(),
                }
            )
            bodies.add(body)
        descriptor = {
            "asset_path": str(path),
            "root_prim": prim(config.get("root_prim", "/World"), "root_prim"),
            "hide_prims": hide,
            "use_asset_lights": lights,
            "body_visuals": body_visuals,
        }
        return descriptor, names(config.get("hide_shape_prefixes", []), "hide_shape_prefixes")

    def __init__(
        self,
        sim: Any,
        *,
        required_meshes: Iterable[str] = (),
        authored_robot: dict[str, str] | None = None,
        robot_bodies: Iterable[str] = (),
        authored_scene: dict[str, Any] | None = None,
        hide_shape_prefixes: Iterable[str] = (),
    ) -> None:
        self._sim: Any = sim
        self._viewer: Any = None
        self._authored_robot = dict(authored_robot) if authored_robot is not None else None
        self._authored_scene = deepcopy(authored_scene)
        self._robot_indices: dict[str, int] = {}
        self._visual_indices: dict[str, int] = {}
        self._batch_rows: dict[str, np.ndarray] = {}
        if isinstance(required_meshes, str):
            raise ValueError("required_meshes must be a collection of mesh names")
        self._required_meshes = frozenset(required_meshes)
        if any(not isinstance(name, str) or not name for name in self._required_meshes):
            raise ValueError("required_meshes must contain nonempty strings")
        base = _load_viewer_base()
        viewer_class = type("_SceneSnapshotViewer", (_SnapshotSink, base), {})
        viewer = viewer_class()
        try:
            viewer.set_model(sim._model)
            excluded_bodies: set[int] = set()
            if self._authored_robot is not None:
                names = set(robot_bodies)
                if not names or any(not isinstance(name, str) or not name for name in names):
                    raise ValueError("authored robot requires nonempty body names")
                for index, path in enumerate(sim._body_paths):
                    leaf = path.rsplit("/", 1)[-1]
                    if leaf in names:
                        if leaf in self._robot_indices:
                            raise ValueError(f"ambiguous Newton robot body: {leaf}")
                        self._robot_indices[leaf] = index
                missing = names - self._robot_indices.keys()
                if missing:
                    raise ValueError(f"authored robot has no Newton body: {', '.join(sorted(missing))}")
                excluded_bodies.update(self._robot_indices.values())
            if self._authored_scene is not None:
                names = {entry["body"] for entry in self._authored_scene["body_visuals"]}
                for index, path in enumerate(sim._body_paths):
                    leaf = path.rsplit("/", 1)[-1]
                    if leaf in names:
                        if leaf in self._visual_indices:
                            raise ValueError(f"ambiguous Newton visual body: {leaf}")
                        self._visual_indices[leaf] = index
                missing = names - self._visual_indices.keys()
                if missing:
                    raise ValueError(f"authored scene has no Newton body: {', '.join(sorted(missing))}")
                excluded_bodies.update(self._visual_indices.values())
            if isinstance(hide_shape_prefixes, str):
                raise ValueError("hide_shape_prefixes must be a collection of shape-name prefixes")
            prefixes = tuple(hide_shape_prefixes)
            if any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
                raise ValueError("hide_shape_prefixes must contain nonempty strings")
            excluded_shapes: set[int] = set()
            if prefixes:
                labels = [str(label).rsplit("/", 1)[-1] for label in sim._model.shape_label]
                for prefix in prefixes:
                    matches = {index for index, label in enumerate(labels) if label.startswith(prefix)}
                    if not matches:
                        raise ValueError(f"authored visual shape prefix matches no Newton shape: {prefix}")
                    excluded_shapes.update(matches)
            if excluded_bodies or excluded_shapes:
                shape_bodies = _array(sim._model.shape_body, np.int32).reshape(-1)
                # A ViewerBase batch can mix robot and scene shapes. Remove
                # only selected shape rows; never hide a shared mesh prototype.
                for batch in viewer._shape_instances.values():
                    self._batch_rows[batch.name] = np.array(
                        [
                            shape_bodies[index] not in excluded_bodies and index not in excluded_shapes
                            for index in batch.model_shapes
                        ],
                        dtype=bool,
                    )
        except BaseException:
            viewer.close()
            self._sim = None
            raise
        self._viewer = viewer

    def capture(self, sim_time: float) -> dict[str, Any]:
        """Return changed meshes and all world-space instances for this state.

        Instance transforms are Newton's ``xyz + xyzw`` convention. Capture
        must run on the engine thread at a completed step/pause boundary.
        ``.numpy()`` reads synchronize the generated Warp geometry before the
        arrays are returned; no GPU array or engine-owned view crosses IPC.
        """
        if self._viewer is None:
            raise RuntimeError("scene snapshot is closed")
        viewer, sim = self._viewer, self._sim
        viewer.begin_frame(float(sim_time))
        viewer.log_state(sim._state_0)
        sim._call_plugin("on_render", viewer, sim._state_0, float(sim_time), sim._plugin_ctx())
        viewer.end_frame()
        missing = self._required_meshes - viewer._snapshot_seen
        if missing:
            # Scene plugins may swallow their own render failures. Cached
            # geometry from an earlier frame is not evidence of this capture.
            raise RuntimeError(f"scene snapshot missing required meshes this frame: {', '.join(sorted(missing))}")
        # Copy mesh arrays again at the boundary: the retained comparison cache
        # and a caller/Connection recipient must never share writable storage.
        meshes = {
            name: {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in entry.items()}
            for name, entry in viewer._snapshot_changed.items()
        }
        instances = viewer._snapshot_instances
        if self._batch_rows:
            for name, keep in self._batch_rows.items():
                entry = instances.get(name)
                if entry is None:
                    continue
                if len(keep) != len(entry["xforms"]):
                    raise ValueError(f"authored visual shape batch changed topology: {name}")
                if not keep.any():
                    del instances[name]
                else:
                    instances[name] = {
                        key: value[keep].copy() if isinstance(value, np.ndarray) else value
                        for key, value in entry.items()
                    }
        if self._authored_robot is not None or self._authored_scene is not None:
            poses = _array(sim._state_0.body_q, np.float32, 7)
        viewer._snapshot_changed = {}
        viewer._snapshot_instances = {}
        result = {"meshes": meshes, "instances": instances}
        if self._authored_robot is not None:
            result.update(
                authored_robot=dict(self._authored_robot),
                body_poses={name: poses[index].copy() for name, index in self._robot_indices.items()},
            )
        if self._authored_scene is not None:
            result.update(
                authored_scene=deepcopy(self._authored_scene),
                visual_body_poses={name: poses[index].copy() for name, index in self._visual_indices.items()},
            )
        return result

    def close(self) -> None:
        """Release only viewer caches; never close or mutate the engine."""
        viewer, self._viewer = self._viewer, None
        self._sim = None
        if viewer is not None:
            viewer.close()
