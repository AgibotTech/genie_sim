# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Debug visualizer — sibling rclpy publishers for engine state.

Enabled per-topic via the ``newton.debug:`` block in the scene yaml
(every knob defaults to off).  The engine's primary ROS topics —
``/clock``, ``/joint_states``, ``/tf``, ``/odom`` — go through the C++
``gsi::RosBridge`` exposed by ``pybinding.cpp``.  That bridge has no
``visualization_msgs/Marker[Array]`` publishers, and adding C++ bindings
for every new debug topic forces a rebuild on every iteration.  This
module hosts the Python-side debug publishers instead — they share a
single lightweight rclpy node that coexists with the C++ bridge in the
same process (the same pattern the urdf_to_usd stage already uses).

Each publisher uses ``rclpy.qos.qos_profile_sensor_data`` (best-effort,
KEEP_LAST 5, VOLATILE) — standard for high-rate, lossy-OK telemetry
that RViz/sensors expect.  RViz's Marker / MarkerArray displays default
to sensor QoS too, so subscribers connect without manual QoS overrides.

Current publishers:

  * :class:`DeformableMarkerPublisher` — ``visualization_msgs/Marker``
    ``TRIANGLE_LIST`` of the deformable / cloth surface (particles
    indexed by ``model.tri_indices``).  Works for VBD cloth, XPBD cloth,
    and FEM tet bars / soft grids — anything that emits surface tris
    in the model.

  * :class:`DeformablePointCloudPublisher` — ``sensor_msgs/PointCloud2``
    of every deformable particle (no triangle indexing).  Useful when
    ``tri_count == 0`` or when an RViz PointCloud2 display fits the
    debugging task better than a flat-shaded triangle marker.

  * :class:`ObjectMarkerPublisher` — ``visualization_msgs/MarkerArray``
    of every free-joint rigid body.  Each body's collider mesh is
    baked into an OBJ file on ``/tmp`` at startup and rendered via a
    ``Marker.MESH_RESOURCE`` referencing that file.  Per-tick wire cost
    is just the body pose; RViz loads the OBJ once and caches it by URL.

Add new publishers as small classes alongside these two; each one calls
``_get_or_create_node()`` to share the rclpy node.

When every flag is off, this module is never imported and rclpy stays
untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# rclpy / sensor_msgs / std_msgs are imported lazily so this module's
# mere presence on the engine path doesn't drag in rclpy when every
# debug flag is off.
_rclpy = None
_SENSOR_QOS = None
_Marker = None
_MarkerArray = None
_ColorRGBA = None
_Point = None
_PointCloud2 = None
_PointField = None
_Header = None
_node = None  # singleton — one rclpy node per process for all debug pubs


def _lazy_import() -> None:
    """Import rclpy + sensor_msgs once.  Raises if ROS 2 Python is unavailable."""
    global _rclpy, _SENSOR_QOS, _Marker, _MarkerArray, _ColorRGBA, _Point
    global _PointCloud2, _PointField, _Header
    if _rclpy is not None:
        return
    import rclpy  # noqa: PLC0415
    from rclpy.qos import qos_profile_sensor_data  # noqa: PLC0415
    from std_msgs.msg import ColorRGBA, Header  # noqa: PLC0415
    from geometry_msgs.msg import Point  # noqa: PLC0415
    from sensor_msgs.msg import PointCloud2, PointField  # noqa: PLC0415
    from visualization_msgs.msg import Marker, MarkerArray  # noqa: PLC0415

    _rclpy = rclpy
    _SENSOR_QOS = qos_profile_sensor_data
    _Marker = Marker
    _MarkerArray = MarkerArray
    _ColorRGBA = ColorRGBA
    _Point = Point
    _PointCloud2 = PointCloud2
    _PointField = PointField
    _Header = Header


def _get_or_create_node() -> Any:
    """Return the shared rclpy debug node, creating it on first call.

    Node namespace is ``/genie_sim_engine`` — matches the C++ engine's
    ROS namespace so all debug publishers land under
    ``/genie_sim_engine/<topic>`` (RViz keeps the engine's topics
    grouped together).  Publishers in this module use RELATIVE topic
    names (``"deformable_marker"`` not ``"/deformable_marker"``) so
    the namespace prefix applies automatically.

    Initializes ``rclpy.init`` with an EMPTY args list (NOT ``args=None``).
    ``args=None`` falls back to ``sys.argv``, which under
    ``ros2 launch`` contains tokens like ``--gui`` that rclcpp's CLI
    parser rejects with ``UnknownROSArgsError`` — the C++ engine
    consumes those flags before main() and never strips them from
    ``sys.argv``, so any rclpy client started inside the same process
    inherits the polluted argv.  Idempotent — if the C++ bridge
    already initialized its own rclcpp context, that's independent of
    rclpy's; calling ``rclpy.init`` here is still safe.
    """
    global _node
    if _node is not None:
        return _node
    _lazy_import()
    if not _rclpy.ok():
        _rclpy.init(args=[])
    _node = _rclpy.create_node("debug_visualizer", namespace="/genie_sim_engine")
    return _node


class DeformableMarkerPublisher:
    """Per-topic MarkerArray publisher for deformable surfaces.

    Covers cloth (VBD / XPBD sheets) and FEM tet / soft-grid surfaces —
    anything that emits surface triangles into ``model.tri_indices``.
    Snapshots ``model.tri_indices`` at construction (topology is
    immutable after ``builder.finalize``) so the per-tick path is just
    a ``particle_q.numpy()`` read + gather + ``Marker.points`` build.

    Each tick emits one ``visualization_msgs/MarkerArray`` carrying a
    single ``Marker`` of type ``TRIANGLE_LIST``.  Why MarkerArray and
    not a bare Marker (same as ``ObjectMarkerPublisher``):

      * Consistency — every debug publisher in this module uses the
        same wire type so RViz "MarkerArray" displays subscribe to
        every topic with no per-topic display switching.
      * Forward room — adding a label / per-shell separation / a
        secondary wireframe overlay later is one extra ``arr.markers
        .append(...)`` instead of needing a second publisher.
      * No DELETEALL race — when the publisher is disarmed mid-run
        we can emit a single MarkerArray with a DELETEALL action to
        clear RViz.

    Single-color marker by default (``color`` arg, RGBA in 0..1).
    Per-triangle coloring would require populating ``marker.colors``
    and is skipped here — single color keeps the message small and is
    enough for debug.

    QoS is ``qos_profile_sensor_data``.
    """

    def __init__(
        self,
        model: Any,
        topic: str,
        frame_id: str = "map",
        ns: str = "deformable",
        color: tuple = (0.85, 0.85, 0.90, 1.0),
    ) -> None:
        node = _get_or_create_node()
        self._topic = topic
        self._frame_id = frame_id
        self._ns = ns
        self._color = tuple(float(c) for c in color)
        self._publisher = node.create_publisher(_MarkerArray, topic, _SENSOR_QOS)
        # Snapshot tri_indices as a flat int64 array of length 3 * n_tri.
        # The per-tick path indexes particle_q by this array to build the
        # vertex stream.  Topology is immutable post-finalize so the
        # snapshot is reused every frame.
        try:
            ti = model.tri_indices.numpy() if hasattr(model.tri_indices, "numpy") else np.asarray(model.tri_indices)
        except Exception:
            ti = np.zeros((0, 3), dtype=np.int32)
        self._tri_flat = np.ascontiguousarray(ti.reshape(-1), dtype=np.int64)
        self._n_tri = int(self._tri_flat.size // 3)

    def publish(self, particle_q: Any, sim_time: float) -> None:
        """Publish a MarkerArray containing a TRIANGLE_LIST marker built from ``particle_q``."""
        if self._publisher is None or self._n_tri == 0:
            return
        try:
            if hasattr(particle_q, "numpy"):
                pq = particle_q.numpy()
            else:
                pq = np.asarray(particle_q)
        except Exception:
            return
        if pq.size == 0:
            return
        # Gather: (3 * n_tri, 3) flat float64 vertex stream
        # (geometry_msgs/Point uses float64).
        verts = np.ascontiguousarray(pq[self._tri_flat], dtype=np.float64)

        m = _Marker()
        sec = int(sim_time)
        nsec = int((sim_time - sec) * 1.0e9) & 0x7FFFFFFF
        m.header.stamp.sec = sec
        m.header.stamp.nanosec = nsec
        m.header.frame_id = self._frame_id
        m.ns = self._ns
        m.id = 0
        m.type = _Marker.TRIANGLE_LIST
        m.action = _Marker.ADD
        m.pose.orientation.w = 1.0  # identity (vertex positions are in frame_id)
        # ``scale.x/y/z = 1`` for TRIANGLE_LIST — the field is a uniform
        # multiplier; setting < 1 would shrink the visible mesh.
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0
        cr, cg, cb, ca = self._color
        m.color = _ColorRGBA(r=cr, g=cg, b=cb, a=ca)
        # Build the ``points`` list once via list comprehension — this is
        # the per-tick hot path.  Profile note: for ~10 000 vertices the
        # comprehension dominates publish cost; if needed, swap to a
        # Cython helper or numpy_msg-style serializer.
        m.points = [_Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in verts]
        m.frame_locked = False

        arr = _MarkerArray()
        arr.markers.append(m)
        self._publisher.publish(arr)


class DeformablePointCloudPublisher:
    """``sensor_msgs/PointCloud2`` of every deformable particle.

    Same source data as :class:`DeformableMarkerPublisher` (Newton's
    ``state.particle_q``) but no triangle indexing — just the raw
    particle positions as an unstructured point cloud.  Useful when:

      * The cloth/FEM has no surface tris (e.g. particle-only soft
        body, or a model with ``tri_count == 0``) — TRIANGLE_LIST
        renders nothing, but the PointCloud2 still shows every particle.
      * You want an RViz "PointCloud2" display for shading by intensity
        / per-point colour later — denser feedback than the flat-shaded
        triangle marker.
      * The host is rendering many cloth instances and the per-tick
        triangle build cost (one ``geometry_msgs/Point`` per vertex)
        is a bottleneck — the PointCloud2 path is a single contiguous
        ``np.float32`` blob, ~3× cheaper to wire-encode.

    Wire format: ``height=1``, ``width=N``, three ``FLOAT32`` fields
    ``x``/``y``/``z``, ``point_step=12``, ``row_step=12*N``,
    ``is_dense=True``.  Standard PCL layout — every RViz/PCL consumer
    handles it without conversion.

    QoS is ``qos_profile_sensor_data``.
    """

    def __init__(
        self,
        model: Any,
        topic: str,
        frame_id: str = "map",
    ) -> None:
        node = _get_or_create_node()
        self._topic = topic
        self._frame_id = frame_id
        self._publisher = node.create_publisher(_PointCloud2, topic, _SENSOR_QOS)
        # Cache the static PointField layout so we don't rebuild it per tick.
        self._fields = [
            _PointField(name="x", offset=0, datatype=_PointField.FLOAT32, count=1),
            _PointField(name="y", offset=4, datatype=_PointField.FLOAT32, count=1),
            _PointField(name="z", offset=8, datatype=_PointField.FLOAT32, count=1),
        ]
        # Snapshot the particle count for diagnostics; the publish path
        # rereads ``particle_q`` every tick (count may grow on rebuild).
        try:
            self._n_particles_init = int(getattr(model, "particle_count", 0) or 0)
        except Exception:
            self._n_particles_init = 0

    def publish(self, particle_q: Any, sim_time: float) -> None:
        """Publish a PointCloud2 carrying ``particle_q`` as XYZ float32."""
        if self._publisher is None:
            return
        try:
            if hasattr(particle_q, "numpy"):
                pq = particle_q.numpy()
            else:
                pq = np.asarray(particle_q)
        except Exception:
            return
        if pq.size == 0:
            return
        # Force float32 + contiguity in one allocation.  ``particle_q``
        # is float32 already in standalone Newton, so this is a copy
        # only for the rare wp.array-with-different-dtype path.
        pts = np.ascontiguousarray(pq.reshape(-1, 3), dtype=np.float32)
        n = pts.shape[0]

        sec = int(sim_time)
        nsec = int((sim_time - sec) * 1.0e9) & 0x7FFFFFFF
        header = _Header()
        header.stamp.sec = sec
        header.stamp.nanosec = nsec
        header.frame_id = self._frame_id

        msg = _PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = n
        msg.fields = self._fields
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * n
        # ``data`` must be ``bytes``; ``tobytes()`` allocates once and
        # transfers ownership to the message.
        msg.data = pts.tobytes()
        msg.is_dense = True
        self._publisher.publish(msg)


# Distinct hue cycle for marker colors — keeps adjacent body indices
# visually different (sequential rainbow is harder to tell apart).
_MARKER_PALETTE = (
    (0.95, 0.30, 0.30, 0.80),  # red
    (0.30, 0.65, 0.95, 0.80),  # blue
    (0.30, 0.90, 0.40, 0.80),  # green
    (0.95, 0.75, 0.20, 0.80),  # amber
    (0.75, 0.40, 0.95, 0.80),  # purple
    (0.20, 0.85, 0.85, 0.80),  # teal
    (0.95, 0.55, 0.40, 0.80),  # coral
    (0.55, 0.75, 0.30, 0.80),  # olive
)


def _classify_free_bodies(model: Any, robot_prefix: str = "", logger: Any = None) -> list[int]:
    """Return body indices whose parent joint is ``JointType.FREE``,
    EXCLUDING any body that lives under the robot's prim path.

    Why the robot filter: when ``pin_base_to_world: false`` (e.g. the WBC
    scene), Newton attaches the robot's base body to the world via a
    FREE joint so the controller can drive locomotion.  Without the
    filter ``_classify_free_bodies`` would return the robot base, and
    :class:`ObjectMarkerPublisher` would then render the robot's base
    collider mesh as a TRIANGLE_LIST every render tick — thousands of
    ``geometry_msgs/Point`` allocations per body per tick, dragging the
    sim to a crawl.  Robot links already publish TF via the C++
    ``gsi::RosBridge``; they don't need to appear in the debug marker
    array.

    Pass ``robot_prefix`` to enable the filter (empty string disables
    it).  Match rule is identical to
    :func:`common.object_classification.classify_shape`: a body whose
    ``body_label`` starts with ``/<robot_prefix>/`` (or equals
    ``/<robot_prefix>``) is robot.  ``logger`` (optional) gets a one-shot
    debug listing of every FREE body and whether it was kept or filtered
    — useful when the empirical filter behaviour disagrees with what the
    yaml seems to ask for.
    """
    import newton  # noqa: PLC0415

    free_kind = int(newton.JointType.FREE)
    try:
        jt = model.joint_type.numpy() if hasattr(model.joint_type, "numpy") else np.asarray(model.joint_type)
        jc = model.joint_child.numpy() if hasattr(model.joint_child, "numpy") else np.asarray(model.joint_child)
    except Exception:
        return []
    try:
        body_label = list(model.body_label) if hasattr(model, "body_label") and model.body_label is not None else []
    except Exception:
        body_label = []
    prefix = f"/{robot_prefix}/" if robot_prefix else ""
    exact = f"/{robot_prefix}" if robot_prefix else ""
    free_bodies: list[int] = []
    diag_lines: list[str] = []
    for j, kind in enumerate(jt):
        if int(kind) != free_kind:
            continue
        child = int(jc[j])
        if child < 0:
            continue
        lbl = body_label[child] if 0 <= child < len(body_label) else ""
        filtered = False
        if prefix and lbl and (lbl.startswith(prefix) or lbl == exact):
            filtered = True
        if logger is not None:
            diag_lines.append(
                f"    joint[{j}] -> body[{child}] label={lbl!r} " f"{'FILTERED (robot)' if filtered else 'kept'}"
            )
        if not filtered:
            free_bodies.append(child)
    if logger is not None and diag_lines:
        logger.info(
            f"[debug_visualizer] _classify_free_bodies(robot_prefix={robot_prefix!r}): "
            f"{len(free_bodies)} kept / {len(diag_lines)} FREE joints scanned\n" + "\n".join(diag_lines)
        )
    return free_bodies


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` (..., 3) by quaternion ``q`` (qx, qy, qz, qw).

    Vectorized form of ``v + 2 * cross(qxyz, cross(qxyz, v) + qw * v)``
    — broadcasts a single quaternion across an (N, 3) vertex array.
    """
    qxyz = q[:3]
    qw = float(q[3])
    t = 2.0 * np.cross(qxyz, v)
    return v + qw * t + np.cross(qxyz, t)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``(qx,qy,qz,qw)`` quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _bake_body_collider_obj(model: Any, body_idx: int, out_path: str) -> int:
    """Concatenate the body's collider meshes into ``out_path`` (OBJ).

    Picks all MESH / CONVEX_MESH shapes attached to ``body_idx``.  When
    multiple are present (typical for COACD-decomposed assets that
    register both a high-poly visual AND a set of convex hulls as
    CollisionAPI'd prims), prefers the smaller hulls and DROPS outliers
    whose tri count exceeds ``2 ×`` the smallest hull — that's where
    the "use collider not visual" intent lands without requiring the
    asset author to remove ``CollisionAPI`` from the visual.  Falls
    back to "use everything" when shape counts are similar (single mesh,
    or all-hulls bodies).

    Each accepted shape's vertices are pre-baked with its
    ``shape_transform`` (translation + quaternion) and ``shape_scale``,
    so the resulting OBJ is in **body-local** space.  Per-tick markers
    only set ``marker.pose`` to the body's world pose; RViz transforms
    on the GPU.

    Returns the total triangle count written.  0 if no mesh data was
    available (caller falls back to a primitive AABB marker).
    """
    import newton  # noqa: PLC0415

    try:
        sb = model.shape_body.numpy() if hasattr(model.shape_body, "numpy") else np.asarray(model.shape_body)
        st = model.shape_type.numpy() if hasattr(model.shape_type, "numpy") else np.asarray(model.shape_type)
        ss = model.shape_scale.numpy() if hasattr(model.shape_scale, "numpy") else np.asarray(model.shape_scale)
        sxf = (
            model.shape_transform.numpy()
            if hasattr(model.shape_transform, "numpy")
            else np.asarray(model.shape_transform)
        )
        ssrc = list(model.shape_source) if hasattr(model, "shape_source") else []
    except Exception:
        return 0

    hits = np.where(sb == body_idx)[0]
    if len(hits) == 0:
        return 0

    mesh_kind = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))
    # Collect (n_tri, shape_idx, verts, indices) for every mesh shape on
    # this body.  Sort ascending by tri count so we can apply the
    # collider-preference heuristic before baking.
    candidates: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for h in hits:
        si = int(h)
        if int(st[si]) not in mesh_kind:
            continue
        src = ssrc[si] if 0 <= si < len(ssrc) else None
        if src is None or not hasattr(src, "vertices") or not hasattr(src, "indices"):
            continue
        try:
            v = np.ascontiguousarray(src.vertices, dtype=np.float64).reshape(-1, 3)
            ii = np.ascontiguousarray(src.indices, dtype=np.int64).reshape(-1, 3)
        except Exception:
            continue
        if v.size == 0 or ii.size == 0:
            continue
        candidates.append((int(ii.shape[0]), si, v, ii))

    if not candidates:
        return 0

    candidates.sort(key=lambda t: t[0])
    smallest = candidates[0][0]
    # Outlier rule: drop any shape whose tri count exceeds 2 × the
    # smallest mesh on this body.  Hits the COACD case (small hulls
    # 1.8k tris + visual 90k tris → drop visual) without affecting
    # single-mesh bodies (wok, hanger — only candidate, kept).
    keep = [c for c in candidates if c[0] <= max(smallest * 2, smallest + 1)]

    # Pre-bake each kept shape's xform + scale into world-of-body verts,
    # then accumulate verts + faces with running vertex-index offset.
    verts_chunks: list[np.ndarray] = []
    faces_chunks: list[np.ndarray] = []
    vert_offset = 0
    total_tris = 0
    for n_tri, si, v, ii in keep:
        shape_xform = np.asarray(sxf[si], dtype=np.float64)
        shape_t = shape_xform[:3]
        shape_q = shape_xform[3:7]
        shape_scale = np.asarray(ss[si], dtype=np.float64)
        v_scaled = v * shape_scale  # (Nv, 3) * (3,)
        v_local = _quat_rotate(shape_q, v_scaled) + shape_t
        verts_chunks.append(v_local)
        faces_chunks.append(ii + vert_offset)
        vert_offset += v_local.shape[0]
        total_tris += n_tri

    verts = np.concatenate(verts_chunks, axis=0)
    faces = np.concatenate(faces_chunks, axis=0) + 1  # OBJ is 1-indexed

    # Write OBJ via numpy.savetxt — much faster than Python f-string
    # iteration at 100k+ lines.  RViz parses OBJ on its own loader.
    import io  # noqa: PLC0415

    buf = io.StringIO()
    np.savetxt(buf, verts, fmt="v %.6f %.6f %.6f")
    np.savetxt(buf, faces, fmt="f %d %d %d")
    with open(out_path, "w") as fh:
        fh.write(buf.getvalue())
    return total_tris


def _body_marker_template(model: Any, body_idx: int, obj_out_dir: str) -> dict:
    """Pick a marker spec for ``body_idx`` and return a per-template dict.

    Returns one of two shapes:

    Mesh-resource template (the body's collider mesh, baked to OBJ
    on disk):
        {
            "kind": "mesh_resource",
            "mesh_resource": "file:///tmp/.../body_<N>.obj",
            "n_tri": int,
        }

    Primitive template (no collider mesh available — fallback):
        {
            "kind": "primitive",
            "type": int (Marker.CUBE | SPHERE | CYLINDER),
            "scale": (sx, sy, sz),       # world-axis scale fed to marker.scale
        }

    Why MESH_RESOURCE over inline TRIANGLE_LIST: at 14 bottle bodies × a
    few thousand triangles each, the inline TRIANGLE_LIST path allocates
    ~50 000 ``geometry_msgs/Point`` objects every render tick — Python's
    per-second allocation budget caps wall-clock cost at ~1.5 s/tick and
    the extras phase stalls.  MESH_RESOURCE moves the geometry off the
    wire entirely: RViz fetches the OBJ ONCE (cached by URL) and only
    the body's 7-float pose travels per tick.  Wire cost per tick drops
    from ~3 × n_tri × 24 B to ~80 B regardless of mesh complexity.

    OBJ path is unique per process (``/tmp/genie_sim_engine/markers/<pid>``
    by default — see :class:`ObjectMarkerPublisher`) so concurrent runs
    don't share files and RViz doesn't cache a stale mesh across launches.
    """
    import newton  # noqa: PLC0415
    import os  # noqa: PLC0415

    Marker = _Marker
    try:
        sb = model.shape_body.numpy() if hasattr(model.shape_body, "numpy") else np.asarray(model.shape_body)
        st = model.shape_type.numpy() if hasattr(model.shape_type, "numpy") else np.asarray(model.shape_type)
        ss = model.shape_scale.numpy() if hasattr(model.shape_scale, "numpy") else np.asarray(model.shape_scale)
        aabb_lo = (
            model.shape_collision_aabb_lower.numpy()
            if hasattr(model.shape_collision_aabb_lower, "numpy")
            else np.asarray(model.shape_collision_aabb_lower)
        )
        aabb_hi = (
            model.shape_collision_aabb_upper.numpy()
            if hasattr(model.shape_collision_aabb_upper, "numpy")
            else np.asarray(model.shape_collision_aabb_upper)
        )
    except Exception:
        return {"kind": "primitive", "type": Marker.CUBE, "scale": (0.05, 0.05, 0.05)}

    hits = np.where(sb == body_idx)[0]
    if len(hits) == 0:
        return {"kind": "primitive", "type": Marker.CUBE, "scale": (0.05, 0.05, 0.05)}
    shape_idx = int(hits[0])
    gt = int(st[shape_idx])
    s = ss[shape_idx]

    mesh_kind = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))
    if gt in mesh_kind:
        obj_path = os.path.join(obj_out_dir, f"body_{int(body_idx)}.obj")
        n_tri = _bake_body_collider_obj(model, int(body_idx), obj_path)
        if n_tri > 0:
            return {
                "kind": "mesh_resource",
                "mesh_resource": "file://" + os.path.abspath(obj_path),
                "n_tri": int(n_tri),
            }
        # Bake failed (e.g. no usable shape_source) — fall back to AABB.
        lo = aabb_lo[hits].min(axis=0)
        hi = aabb_hi[hits].max(axis=0)
        ext = np.maximum(hi - lo, 1.0e-3)
        return {"kind": "primitive", "type": Marker.CUBE, "scale": (float(ext[0]), float(ext[1]), float(ext[2]))}

    if gt == int(newton.GeoType.BOX):
        return {
            "kind": "primitive",
            "type": Marker.CUBE,
            "scale": (float(2.0 * s[0]), float(2.0 * s[1]), float(2.0 * s[2])),
        }
    if gt == int(newton.GeoType.SPHERE):
        d = float(2.0 * s[0])
        return {"kind": "primitive", "type": Marker.SPHERE, "scale": (d, d, d)}
    if gt == int(newton.GeoType.CYLINDER):
        d = float(2.0 * s[0])
        h = float(s[1])
        return {"kind": "primitive", "type": Marker.CYLINDER, "scale": (d, d, h)}
    if gt == int(newton.GeoType.CAPSULE):
        d = float(2.0 * s[0])
        h = float(s[1])
        return {"kind": "primitive", "type": Marker.CYLINDER, "scale": (d, d, h + d)}
    # ELLIPSOID / PLANE / HFIELD / CONE / GAUSSIAN — AABB cube fallback.
    lo = aabb_lo[shape_idx]
    hi = aabb_hi[shape_idx]
    ext = np.maximum(hi - lo, 1.0e-3)
    return {"kind": "primitive", "type": Marker.CUBE, "scale": (float(ext[0]), float(ext[1]), float(ext[2]))}


class ObjectMarkerPublisher:
    """Per-topic MarkerArray publisher for free-joint rigid objects.

    Bakes each free body's collider mesh to an OBJ file on disk at
    startup, then publishes a ``visualization_msgs/Marker`` of type
    ``MESH_RESOURCE`` referencing that file via ``file://`` URL.  RViz
    loads each OBJ once (cached by URL) and per-tick wire traffic is
    just the body's 7-float pose.  No vertex math in the hot path, no
    per-vertex allocations, and bandwidth stays bounded regardless of
    mesh complexity.

    Per-body workflow:
      * :func:`_bake_body_collider_obj` writes ``body_<N>.obj`` to the
        publisher's per-process OBJ directory, picking the body's
        smallest mesh shapes (collider hulls preferred over visual via
        a 2 × tri-count outlier rule) and pre-baking each shape's
        ``shape_transform + shape_scale`` so verts are body-local.
      * The cached ``Marker`` is built once: type=MESH_RESOURCE,
        mesh_resource=file://..., color from palette, scale=1, etc.
      * Bodies whose colliders aren't representable as a mesh (or the
        bake failed) fall back to a primitive marker — same render
        path as :class:`DeformableMarkerPublisher`.

    Output dir defaults to ``/tmp/genie_sim_engine/markers/<pid>`` —
    unique per process so concurrent runs don't clobber each other and
    RViz never serves a stale mesh from a previous launch.  Override
    by passing ``obj_out_dir`` to the constructor.

    Two cached markers per body:
      * The shape marker (mesh-resource or primitive).
      * A small TEXT_VIEW_FACING with the body label, floating 8 cm
        above the body origin so overlapping objects stay distinguishable.

    QoS is ``qos_profile_sensor_data``; RViz subscribes with no override.
    """

    def __init__(
        self,
        model: Any,
        topic: str,
        frame_id: str = "map",
        ns: str = "objects",
        robot_prefix: str = "",
        obj_out_dir: str = "",
        logger: Any = None,
    ) -> None:
        import os  # noqa: PLC0415

        node = _get_or_create_node()
        self._topic = topic
        self._frame_id = frame_id
        self._ns = ns
        self._publisher = node.create_publisher(_MarkerArray, topic, _SENSOR_QOS)

        # Per-process OBJ output directory.  Default keyed by PID so
        # concurrent simulator processes don't share files and RViz
        # never serves stale geometry across a re-launch.
        if not obj_out_dir:
            obj_out_dir = f"/tmp/genie_sim_engine/markers/{os.getpid()}"
        os.makedirs(obj_out_dir, exist_ok=True)
        self._obj_out_dir = obj_out_dir

        # Snapshot the per-body marker templates once — topology is fixed
        # after builder.finalize.  ``robot_prefix`` keeps the robot's own
        # FREE-jointed base (when ``pin_base_to_world: false``) out of
        # the marker array.
        self._free_bodies = _classify_free_bodies(model, robot_prefix=robot_prefix, logger=logger)
        self._templates: list[dict[str, Any]] = []
        # Pair of CACHED markers per body: (shape_marker, label_marker).
        # Pre-built once; per-tick publish() just stamps them and
        # assigns body pose.  No allocations in the hot path.
        self._cached_markers: list[tuple[Any, Any]] = []
        try:
            body_label = list(model.body_label) if hasattr(model, "body_label") and model.body_label is not None else []
        except Exception:
            body_label = []

        for idx, b in enumerate(self._free_bodies):
            tpl = _body_marker_template(model, b, obj_out_dir=self._obj_out_dir)
            label = body_label[b] if 0 <= b < len(body_label) else f"body_{b}"
            r, g, bc, a = _MARKER_PALETTE[idx % len(_MARKER_PALETTE)]
            tpl["body"] = int(b)
            tpl["label"] = str(label)
            tpl["color"] = (float(r), float(g), float(bc), float(a))
            self._templates.append(tpl)

            shape_marker = self._build_shape_marker(tpl)
            label_marker = self._build_label_marker(tpl)
            self._cached_markers.append((shape_marker, label_marker))

    def _build_shape_marker(self, tpl: dict) -> Any:
        """Build the per-body shape marker once.  Pose + stamp are set
        per-tick in :meth:`publish`; everything else is baked here."""
        m = _Marker()
        m.header.frame_id = self._frame_id
        m.ns = self._ns
        m.id = int(tpl["body"]) * 2
        m.action = _Marker.ADD
        m.frame_locked = False
        cr, cg, cb, ca = tpl["color"]
        m.color = _ColorRGBA(r=cr, g=cg, b=cb, a=ca)
        if tpl["kind"] == "mesh_resource":
            m.type = _Marker.MESH_RESOURCE
            m.mesh_resource = str(tpl["mesh_resource"])
            # mesh_use_embedded_materials=False lets ``marker.color``
            # tint the mesh; True would use whatever materials the OBJ
            # references (we don't write materials, so the OBJ has none
            # anyway, but explicit-false matches RViz behaviour).
            m.mesh_use_embedded_materials = False
            m.scale.x = 1.0
            m.scale.y = 1.0
            m.scale.z = 1.0
        else:
            m.type = int(tpl["type"])
            sx, sy, sz = tpl["scale"]
            m.scale.x = float(sx)
            m.scale.y = float(sy)
            m.scale.z = float(sz)
        return m

    def _build_label_marker(self, tpl: dict) -> Any:
        """Build the per-body TEXT_VIEW_FACING label marker once."""
        t = _Marker()
        t.header.frame_id = self._frame_id
        t.ns = self._ns + "_labels"
        t.id = int(tpl["body"]) * 2 + 1
        t.type = _Marker.TEXT_VIEW_FACING
        t.action = _Marker.ADD
        t.pose.orientation.w = 1.0
        t.scale.z = 0.04  # text height in metres
        t.color = _ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
        t.text = str(tpl["label"])
        return t

    def publish(self, body_q: Any, sim_time: float) -> None:
        """Publish a MarkerArray from ``body_q`` (shape (N,7) px,py,pz,qx,qy,qz,qw).

        Hot path: for each cached marker, update ``header.stamp`` +
        ``pose`` (shape marker) or ``pose.position`` (label marker) and
        append to the outgoing array.  No vertex math, no allocations.
        """
        if self._publisher is None or not self._cached_markers:
            return
        try:
            if hasattr(body_q, "numpy"):
                bq = body_q.numpy()
            else:
                bq = np.asarray(body_q)
        except Exception:
            return
        if bq.size == 0:
            return

        sec = int(sim_time)
        nsec = int((sim_time - sec) * 1.0e9) & 0x7FFFFFFF
        arr = _MarkerArray()
        for tpl, (shape_marker, label_marker) in zip(self._templates, self._cached_markers):
            b = int(tpl["body"])
            if b >= bq.shape[0]:
                continue
            body_pose = bq[b]
            bx, by, bz = float(body_pose[0]), float(body_pose[1]), float(body_pose[2])
            qx, qy, qz, qw = (
                float(body_pose[3]),
                float(body_pose[4]),
                float(body_pose[5]),
                float(body_pose[6]),
            )

            # Shape marker — body pose, RViz applies it to verts on GPU.
            shape_marker.header.stamp.sec = sec
            shape_marker.header.stamp.nanosec = nsec
            shape_marker.pose.position.x = bx
            shape_marker.pose.position.y = by
            shape_marker.pose.position.z = bz
            shape_marker.pose.orientation.x = qx
            shape_marker.pose.orientation.y = qy
            shape_marker.pose.orientation.z = qz
            shape_marker.pose.orientation.w = qw
            arr.markers.append(shape_marker)

            # Label marker — text only follows body translation; the
            # TEXT_VIEW_FACING marker auto-billboards, so its orientation
            # stays identity (set once in _build_label_marker).
            label_marker.header.stamp.sec = sec
            label_marker.header.stamp.nanosec = nsec
            label_marker.pose.position.x = bx
            label_marker.pose.position.y = by
            label_marker.pose.position.z = bz + 0.08
            arr.markers.append(label_marker)

        self._publisher.publish(arr)


def shutdown() -> None:
    """Best-effort teardown — called from the engine on shutdown.

    Safe to call multiple times.  Leaves the rclpy context alone if other
    parts of the process still rely on it.
    """
    global _node
    if _node is not None:
        try:
            _node.destroy_node()
        except Exception:
            pass
        _node = None
    if _rclpy is not None and _rclpy.ok():
        try:
            _rclpy.shutdown()
        except Exception:
            pass
