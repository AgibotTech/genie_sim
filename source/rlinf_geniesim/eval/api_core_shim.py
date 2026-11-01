# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Expose benchmark state getters through a physics backend for one environment.

World poses use (xyz, quaternion_wxyz), pose matrices are 4x4 arrays, AABBs
use [min_x, min_y, min_z, max_x, max_y, max_z], and articulated-object state
contains joint_names and joint_positions. These formats match benchmark
checker expectations across physics backends."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _rotmat_from_quat_wxyz(q) -> np.ndarray:
    """3x3 rotation matrix from a (w, x, y, z) quaternion (matches the
    benchmark's ``get_rotation_matrix_from_quaternion`` convention)."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


class ApiCoreShim:
    """Drop-in ``api_core`` over a ``VecPhysicsBackend``, scoped to one env.

    ``backend`` must implement the generic interface: ``get_scene_schema()``
    (with ``body_paths``), ``get_body_poses()`` ``(N, B, 7)`` as
    ``[x,y,z,qw,qx,qy,qz]``, and ``get_body_aabbs()`` ``(N, B, 6)``.

    Prim-path resolution mirrors the checkers' ``_analyze_obj_name`` output
    (e.g. ``/Workspace/Objects/<id>`` or robot links like
    ``/G2/gripper_r_center_link``): the path is looked up in
    ``SceneSchema.body_paths`` to an index shared by poses and AABBs.
    """

    def __init__(
        self,
        backend,
        env_id: int = 0,
        obj_prefix: str = "",
        alias: Optional[dict] = None,
        kinematic_alias_keys: Optional[set] = None,
    ) -> None:
        # ``obj_prefix`` is the older superset object-prim prefix. Asset
        # de-duplication uses slot naming instead, so it matches nothing in a
        # current scene and the alias table is the only mapping; it is kept
        # for scenes still using the old naming.
        #
        # ``alias`` maps the checker's names to engine object ids. Instance
        # switching builds one topology while problems.json names objects by
        # each member instance's own prims, so without the mapping nothing
        # resolves and the checker silently reads (0,0,0).
        #
        # ``kinematic_alias_keys`` are alias keys pointing at kinematic
        # objects such as tables. The engine demotes kinematic prims to static
        # colliders before rigid-body indices are assigned, so those aliases
        # can never resolve. ``refresh()`` skips them in its warning: a warning
        # that fires on every reset only trains people to ignore it.
        self._b = backend
        self._env = int(env_id)
        self._obj_prefix = str(obj_prefix or "")
        self._alias: dict = dict(alias or {})
        self._kinematic_alias_keys: set = set(kinematic_alias_keys or ())
        # reverse map for _norm_obj_id (engine label -> checker dialect)
        self._alias_rev: dict = {v: k for k, v in self._alias.items()}
        self._path2idx: dict[str, int] = {}
        self._obj2idx: dict[str, int] = {}
        # Static scene-prop boxes (see _static_prop_box) and the dedup set for
        # the "nothing knows this path" warning in get_obj_aabb.
        self._prop_aabb_cache: "dict | None" = None
        self._aabb_miss_warned: set = set()
        self.refresh()

    @property
    def env_id(self) -> int:
        return self._env

    @env_id.setter
    def env_id(self, value: int) -> None:
        self._env = int(value)

    def refresh(self) -> None:
        """(Re)build the prim-path -> body index map from the scene schema.
        Call after the backend's scene structure changes.

        We index by ``body_link_names`` — the engine's REAL per-env labels
        (e.g. ``genie/gripper_l_center_link`` for robot links,
        ``World/Objects/<id>/entity`` for manipulable objects) — NOT the
        generic ``body_paths`` (``/World/body_i``), which carry no semantics.
        Both leading-slash and bare forms are registered, plus an object-id
        index so the checkers' resolved paths map onto engine bodies.
        """
        self._prop_aabb_cache = None  # a reload can replace the entire background
        schema = self._b.get_scene_schema()
        if isinstance(schema, dict):
            labels = schema.get("body_link_names") or schema.get("body_paths") or []
        else:
            labels = getattr(schema, "body_link_names", None) or getattr(schema, "body_paths", None) or []
        self._path2idx = {}
        self._obj2idx = {}
        for i, lbl in enumerate(labels):
            lbl = str(lbl)
            self._path2idx[lbl] = i
            self._path2idx["/" + lbl.lstrip("/")] = i
            if "/Objects/" in lbl:
                obj_id = lbl.split("/Objects/", 1)[1].split("/", 1)[0]
                self._obj2idx[obj_id] = i
                if self._obj_prefix and obj_id.startswith(self._obj_prefix):
                    # active-instance superset prim also answers to its raw
                    # benchmark name (the dialect the checkers emit)
                    self._obj2idx[obj_id[len(self._obj_prefix) :]] = i
        # Explicit alias table (in-group instance switching): every member-prim
        # name the checkers may ask for points at the build prim's body index.
        # Applied AFTER the label loop so it can reference prefix-stripped ids
        # registered above. Resolved against a SNAPSHOT of ``_obj2idx`` taken
        # before this loop, not the live dict: resolving in place let one
        # alias entry's write feed another entry's read within this SAME loop
        # whenever an alias VALUE (engine_name) happened to equal another
        # alias's KEY (checker_name) — e.g. ``alias={"B": "A", "A": "B"}`` on
        # a 2-body scene silently collapsed to ``{'A': 0, 'B': 0}`` (both
        # names pointing at the SAME body) instead of the correct swap. A
        # target that still can't be resolved against the snapshot now warns
        # instead of being dropped silently: an unresolvable name_alias means
        # the checker sees no such object at all — score never lights up, and
        # nothing raises.
        base = dict(self._obj2idx)  # aliases don't see each other's writes
        resolved = 0
        unresolved: list = []
        for checker_name, engine_name in self._alias.items():
            j = base.get(engine_name)
            if j is None:
                unresolved.append(checker_name)
                continue
            self._obj2idx[checker_name] = j
            resolved += 1
        # Kinematic misses (see kinematic_alias_keys above) are EXPECTED —
        # exclude them before deciding whether to warn. A miss that ISN'T a
        # known kinematic object is still a genuine problem and still warns.
        surprising = [n for n in unresolved if n not in self._kinematic_alias_keys]
        if surprising:
            print(
                f"[shim] WARN: {resolved}/{len(self._alias)} name_alias "
                f"entries resolved; the rest reference build prims absent "
                f"from the engine's object table, so the checker cannot see "
                f"those objects",
                flush=True,
            )

    def reset(self) -> None:
        """No-op state reset (the benchmark ``AderEnv.reset`` calls
        ``api_core.reset()``). The shim is stateless beyond the body-index
        map, so we just rebuild it in case the scene changed."""
        self.refresh()

    def _idx(self, prim_path: str) -> Optional[int]:
        """Resolve a checker prim path to a body index.

        Handles the two naming dialects the checkers emit:
          * robot links  ``/genie/<link>``           -> engine ``genie/<link>``
          * objects      ``/Workspace|/World/Objects/<id>`` -> engine
            ``World/Objects/<id>/entity`` (matched by the ``<id>`` segment).
        """
        idx = self._path2idx.get(prim_path)
        if idx is not None:
            return idx
        if "/Objects/" in prim_path:
            obj_id = prim_path.split("/Objects/", 1)[1].split("/", 1)[0]
            idx = self._obj2idx.get(obj_id)
            if idx is not None:
                return idx
        return self._path2idx.get(prim_path.lstrip("/"))

    @staticmethod
    def _obj_id_of(path: Optional[str]) -> Optional[str]:
        """Extract the object-id segment from a prim path / engine joint label.

        The checker emits ``/World/objects/benchmark_door_000`` (the
        background authors ``objects`` lowercase); the engine's scene-joint
        labels carry a ``/Background/...`` prefix and the object id one level
        under an ``objects`` / ``Objects`` segment
        (``/Background/.../objects/benchmark_door_000/.../RevoluteJoint_00``).
        Both reduce to the same id via the segment right after
        ``/objects/`` or ``/Objects/`` — mirrors ``_idx``'s ``/Objects/``
        handling, tolerant of the lowercase form the background uses.
        Returns ``None`` when neither segment is present (e.g. a robot link
        path like ``/genie/gripper_r_center_link``), so such prims match no
        object and get an empty joint list.
        """
        if not path:
            return None
        for sep in ("/objects/", "/Objects/"):
            if sep in path:
                return path.split(sep, 1)[1].split("/", 1)[0]
        return None

    def _norm_obj_id(self, path: Optional[str]) -> Optional[str]:
        """``_obj_id_of`` + active-instance superset prefix strip + reverse
        alias, so an ENGINE label normalizes into the dialect the checker
        emits: the prefix strip handles superset (``i{id}_x`` -> ``x``), the
        reverse alias handles in-group switching (build prim -> the member
        instance's prim name problems.json actually uses). Parked donors keep
        their prefix and thus never match a raw checker id."""
        seg = self._obj_id_of(path)
        if seg is not None and self._obj_prefix and seg.startswith(self._obj_prefix):
            seg = seg[len(self._obj_prefix) :]
        if seg is not None:
            seg = self._alias_rev.get(seg, seg)
        return seg

    # ------------------------------------------------------------------
    # Checker getter surface (mirrors benchmark api_core return formats)
    # ------------------------------------------------------------------

    def get_obj_world_pose(self, prim_path: str):
        idx = self._idx(prim_path)
        if idx is None:
            return (0, 0, 0), (0, 0, 0, 0)
        p = self._b.get_body_poses()[self._env, idx]  # [x,y,z,qw,qx,qy,qz]
        return (
            (float(p[0]), float(p[1]), float(p[2])),
            (float(p[3]), float(p[4]), float(p[5]), float(p[6])),
        )

    def get_obj_world_pose_matrix(self, prim_path: str, camera: bool = False) -> np.ndarray:
        position, rotation = self.get_obj_world_pose(prim_path)
        pose = np.eye(4)
        pose[:3, :3] = _rotmat_from_quat_wxyz(rotation)
        pose[:3, 3] = np.asarray(position, dtype=float)
        if camera:
            # benchmark applies a 180deg flip about X for camera frames
            pose = pose @ np.diag([1.0, -1.0, -1.0, 1.0])
        return pose

    def _static_prop_box(self, prim_path: str) -> "np.ndarray | None":
        """World AABB for a STATIC scene prop, or None if it isn't one.

        Props are published once (latched) and never move, so the mapping is
        cached on first non-empty read. An empty read is NOT cached: the
        latched message can land after the first checker tick, and a task
        that names no props pays only a lock + None check per call.
        """
        table = self._prop_aabb_cache
        if table is None:
            getter = getattr(self._b, "get_static_prop_aabbs", None)
            if getter is None:
                return None
            try:
                table = getter() or {}
            except Exception:  # noqa: BLE001
                return None
            if table:
                self._prop_aabb_cache = table
        box = table.get(prim_path)
        if box is None:
            # Tolerate a leading-slash mismatch between what the task
            # allowlist spells and what the checker emits; anything else is
            # a genuine miss and should surface as the warning below.
            box = table.get("/" + prim_path.lstrip("/"))
        if box is None:
            return None
        # box is now a list[ndarray], one per env. Index by self._env.
        array = np.asarray(box, dtype=np.float32)
        if array.shape == (6,):
            # Backward compat: if schema predates per-env boxes, treat as
            # a single box shared across all envs (the old behavior).
            return array
        if array.ndim != 2 or array.shape[1] != 6 or not 0 <= self._env < len(array):
            # Schema mismatch (fewer envs published than this shim expects).
            return None
        return array[self._env]

    def get_obj_aabb(self, prim_path: str) -> np.ndarray:
        idx = self._idx(prim_path)
        if idx is None:
            # Not a rigid body — but it may be a STATIC scene prop whose
            # world box the engine published for exactly this purpose (see
            # the backend's get_static_prop_aabbs). Background props such as
            # a storage bin have no rigid body, so an ``Inside`` checker
            # naming one lands here.
            box = self._static_prop_box(prim_path)
            if box is not None:
                return box
            # Nothing knows this path. A zero box contains no point, so the
            # checker referencing it can NEVER pass and its task scores 0%
            # forever -- a failure mode indistinguishable from a bad policy
            # unless this is said out loud. Warn once per path, since
            # checkers ask every tick. Fix by adding the prop to the task's
            # ``eval_static_props`` allowlist (eval_tasks/<scene>.json).
            if prim_path not in self._aabb_miss_warned:
                self._aabb_miss_warned.add(prim_path)
                print(
                    f"[shim] WARN: no body index and no static-prop box for "
                    f"{prim_path!r}; returning a zero AABB — any checker "
                    f"using it (Inside/InBBox) can never pass",
                    flush=True,
                )
            return np.zeros(6, dtype=np.float32)
        # Backend AABBs are optional (only some checkers use api_core.get_obj_aabb;
        # Follow/PickUpOnGripper compute bboxes from the pose matrix instead).
        get_aabbs = getattr(self._b, "get_body_aabbs", None)
        if get_aabbs is None:
            return np.zeros(6, dtype=np.float32)
        try:
            return np.asarray(get_aabbs()[self._env, idx])
        except NotImplementedError:
            return np.zeros(6, dtype=np.float32)

    def get_obj_joint(self, prim_path: str) -> dict:
        """Return an articulated scene object's joint angles for THIS env.

        Mirrors the benchmark Isaac ``api_core.get_obj_joint`` return shape:
        ``{"joint_names": [...], "joint_positions": [...]}``.  The checker
        (open_door's ``PushPull``, via ``EvaluateAction.get_prismatic_joint``)
        reads ``["joint_positions"][joint_index]`` — for the door
        ``joint_index=0`` is the panel hinge (``RevoluteJoint_00``, declared
        first in the asset).

        Ground truth comes from the engine's ``/rlinf/scene_joint_states``
        cache, exposed by the backend as ``get_scene_joint_states()`` — a
        per-env, insertion-ordered ``{joint_label: position}`` map whose order
        IS the engine's published order (= asset joint-declaration order).
        We keep only this object's joints (matched by object-id segment — see
        ``_obj_id_of``) and preserve that order.

        Miss semantics mirror ``get_obj_world_pose``: an unknown object, a
        robot prim, no cached data, or a backend that doesn't publish scene
        joints all yield an empty (but well-formed) result rather than
        raising — the checker's ``list(info["joint_positions"])`` then reads
        ``[]`` instead of hitting a ``KeyError``.
        """
        empty = {"joint_names": [], "joint_positions": []}
        obj_id = self._norm_obj_id(prim_path)
        if obj_id is None:
            return empty
        getter = getattr(self._b, "get_scene_joint_states", None)
        if getter is None:
            return empty
        try:
            per_env = getter()
        except NotImplementedError:
            return empty
        if not per_env or self._env >= len(per_env):
            return empty
        env_map = per_env[self._env] or {}
        names: list[str] = []
        positions: list[float] = []
        for label, pos in env_map.items():  # insertion order == published order
            if self._norm_obj_id(label) == obj_id:
                names.append(str(label))
                positions.append(float(pos))
        # joint_velocities: present for exact shape parity with the benchmark
        # Isaac api_core.get_obj_joint (3 keys); the scene-joint transport
        # carries positions only, and no vendored checker reads velocities
        # (PushPull et al. consume joint_positions) — empty list, not None,
        # so a future list() consumer degrades to zero-length not TypeError.
        return {"joint_names": names, "joint_positions": positions, "joint_velocities": []}


__all__ = ["ApiCoreShim"]
