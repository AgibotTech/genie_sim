# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Scene plugin: G2 + wok-on-a-stove demo (teleop-oriented).

Companion to ``scene_flat_g2_sp_chef_wuji.yaml``.  Loaded by the
newton-standalone engine via ``newton.scene_plugin: scene_flat_g2_sp_chef.py``
(see ``engine/newton/plugin.py``).

Ported from the standalone ``chow_mein_flip_fem.py`` demo (FEM tet noodles,
ham dice, julienne strips, smooth render tubes, spherical-cap wok mesh).  The
procedural fallback authors that wok and stove.  The Wuji USDA scene binds
the imported pot body, fits the particle-only thick shell
from its authored bowl, binds all authored noodles to FEM proxies, and binds
the authored meat, pepper-strip, and carrot-strip families.

Three wok modes (see ``WokGeometry.wok_dynamic`` + ``Config.auto_flip``):

  * DYNAMIC   (wok_dynamic=True)                    — DEFAULT.  Real free rigid
                body: falls under gravity, rests on the ring via thin-shell
                rigid contact, and can be pushed / GRASPED by the gripper.
                Thin-shell ``has_shape_collision`` turns ON only here; the
                thick collider stays particle-only (food-in-bowl) as before.
                Required for the gripper-grasps-the-handle workflow — a
                kinematic wok is a MuJoCo mocap body with no DOFs and
                infinite effective mass, so the fingers just bounce off it.
  * PARKED    (wok_dynamic=False, auto_flip=False) — Kinematic wok parked at
                its home pose in the ring; ignores gravity/contact, stays put.
                Zero blow-up risk, stove is cosmetic.  Use for render work or
                noodle-settling tuning, NOT for grasping.
  * AUTO-FLIP (wok_dynamic=False, auto_flip=True)   — legacy autonomous flip:
                kinematic free-joint wok driven by the controller writing
                ``joint_q`` each frame.  If both flags are set, wok_dynamic
                wins and auto_flip is disabled with a warning.

Plugin hooks:

  * ``on_stage``      — USDA mode only: validates the pot binding and authors
                        an anonymous session-layer override that disables and
                        hides the rigid food source meshes, captures all 16
                        noodles' authored geometry, and audits all retained
                        side dishes.
  * ``on_build``      — adds the wok body (kinematic or dynamic), soft-grid
                        noodles / ham / strips, the wok handle, and the stove
                        in legacy mode; binds the imported pot, its hidden
                        particle proxy, 16 USDA-derived noodles, and selected
                        USDA side-dish families in USDA mode.
  * ``on_model_ready``— writes ``model.particle_*`` / ``soft_contact_*`` for
                        noodle-vs-noodle and noodle-vs-wok contact, prepares
                        the optional startup hand trajectory, and in AUTO-FLIP
                        mode locates the wok free-joint slot.
  * ``on_post_step``  — publishes the optional startup hand trajectory; in
                        AUTO-FLIP mode also writes the controller's wok pose
                        into ``state.joint_q``.
  * ``on_record_snapshot`` — publishes the delayed particle snapshot paired
                        with the rigid state carried by ``/tf_scene``.
  * ``on_render``     — builds Catmull-Rom smooth tubes on the GPU from each
                        FEM bar's ring centroids (sleek strands, not the
                        faceted tet bars).

newton 1.4.0 note: rigid shapes inherit ``ModelBuilder.rigid_gap`` (0.1 m)
into MuJoCo's contact gap, which would make the wok hover ~10 cm over the
ring.  Every rigid ShapeConfig here passes ``gap=0.0`` for a crisp seat.

Geometry / parameter knobs live in the ``Config`` dataclass tree below —
edit and relaunch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

# ----------------------------------------------------------------------------
# Constants used inside @wp.kernel device code only — they can't read the
# Config dataclass at runtime since wp.kernels are compiled.  Geometry /
# controller knobs all flow through the ``CONFIG`` dataclass below.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Quaternion helpers (xyzw)
# ----------------------------------------------------------------------------
def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_rotate(q, v):
    x, y, z, w = q
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _quat_pitch_roll(pitch, roll):
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    q_pitch = np.array([0.0, sp, 0.0, cp])
    q_roll = np.array([sr, 0.0, 0.0, cr])
    return _quat_mul(q_pitch, q_roll)


def _quat_about_z(theta):
    return np.array([0.0, 0.0, math.sin(0.5 * theta), math.cos(0.5 * theta)])


def _smoothstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def _quat_z_to(axis):
    """xyzw quaternion rotating local +Z onto ``axis`` (capsules are Z-aligned)."""
    z = np.array([0.0, 0.0, 1.0])
    a = axis / np.linalg.norm(axis)
    v = np.cross(z, a)
    s = np.linalg.norm(v)
    c = float(np.dot(z, a))
    if s < 1e-9:
        return wp.quat(0.0, 0.0, 0.0, 1.0) if c > 0 else wp.quat(1.0, 0.0, 0.0, 0.0)
    n = v / s
    half = 0.5 * np.arctan2(s, c)
    sh = np.sin(half)
    return wp.quat(float(n[0] * sh), float(n[1] * sh), float(n[2] * sh), float(np.cos(half)))


def _quat_x_to(axis):
    """xyzw quaternion rotating local +X onto ``axis`` (boxes are X/Y/Z-aligned).

    Sibling of :func:`_quat_z_to` for box shapes, whose half-extents are
    authored per-axis rather than along a capsule's implicit +Z.
    """
    x = np.array([1.0, 0.0, 0.0])
    a = axis / np.linalg.norm(axis)
    v = np.cross(x, a)
    s = np.linalg.norm(v)
    c = float(np.dot(x, a))
    if s < 1e-9:
        # Parallel (c>0) or antiparallel (c<0); the latter is a 180° flip
        # about any axis ⟂ X — pick Z.
        return wp.quat(0.0, 0.0, 0.0, 1.0) if c > 0 else wp.quat(0.0, 0.0, 1.0, 0.0)
    n = v / s
    half = 0.5 * np.arctan2(s, c)
    sh = np.sin(half)
    return wp.quat(float(n[0] * sh), float(n[1] * sh), float(n[2] * sh), float(np.cos(half)))


def _capsule_fromto(a, b):
    """Convert an MJCF-style fromto capsule to (xform, half_height), body-local."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    length = float(np.linalg.norm(d))
    center = 0.5 * (a + b)
    xform = wp.transform(wp.vec3(*center), _quat_z_to(d))
    return xform, 0.5 * length


# --- rigid-transform helpers (numpy; pose = (pos[3], quat[4] xyzw)) ----------
def _quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _xform_mul(a, b):
    """Compose two (pos, quat) transforms: result applies ``b`` then ``a``."""
    ap, aq = a
    bp, bq = b
    return (ap + _quat_rotate(aq, bp), _quat_mul(aq, bq))


def _xform_inv(a):
    ap, aq = a
    qi = _quat_conj(aq)
    return (-_quat_rotate(qi, ap), qi)


def _bind_points_to_tets(points, particle_positions, tet_indices, tolerance=1.0e-4):
    """Bind ordered surface points to a tetrahedral simulation proxy.

    Returns one four-particle index row and one barycentric-weight row per
    input point.  Boundary points may belong to multiple tetrahedra; choosing
    the lowest tet index makes the binding deterministic.  Indices are local
    to ``particle_positions`` so the same mapping can be applied to a
    contiguous runtime particle slice.
    """
    points = np.asarray(points, dtype=np.float64)
    particles = np.asarray(particle_positions, dtype=np.float64)
    tets = np.asarray(tet_indices, dtype=np.int64).reshape(-1, 4)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("authored points must be a non-empty Nx3 array")
    if particles.ndim != 2 or particles.shape[1] != 3 or not len(particles):
        raise ValueError("proxy particles must be a non-empty Nx3 array")
    if not len(tets) or np.any(tets < 0) or np.any(tets >= len(particles)):
        raise ValueError("proxy tetrahedra are empty or contain out-of-range particle indices")
    if not np.isfinite(points).all() or not np.isfinite(particles).all():
        raise ValueError("authored points and proxy particles must be finite")

    tet_points = particles[tets]
    origins = tet_points[:, 3]
    matrices = np.stack(
        (
            tet_points[:, 0] - origins,
            tet_points[:, 1] - origins,
            tet_points[:, 2] - origins,
        ),
        axis=2,
    )
    determinants = np.linalg.det(matrices)
    if np.any(np.abs(determinants) < 1.0e-15):
        raise ValueError("proxy contains a degenerate tetrahedron")
    inverse = np.linalg.inv(matrices)

    bound_tets = np.empty((len(points), 4), dtype=np.int32)
    bound_weights = np.empty((len(points), 4), dtype=np.float32)
    for point_index, point in enumerate(points):
        first_three = np.einsum("tij,tj->ti", inverse, point - origins)
        weights = np.column_stack((first_three, 1.0 - np.sum(first_three, axis=1)))
        containing = np.flatnonzero(
            np.logical_and(
                np.min(weights, axis=1) >= -float(tolerance),
                np.max(weights, axis=1) <= 1.0 + float(tolerance),
            )
        )
        if not len(containing):
            raise ValueError(f"authored vertex {point_index} lies outside the soft proxy")
        tet_index = int(containing[0])
        selected = np.clip(weights[tet_index], 0.0, 1.0)
        selected /= np.sum(selected)
        bound_tets[point_index] = tets[tet_index]
        bound_weights[point_index] = selected.astype(np.float32)
    return bound_tets, bound_weights


def _reconstruct_tet_bound_points(particle_positions, tet_particle_indices, barycentric_weights):
    particles = np.asarray(particle_positions, dtype=np.float64)
    indices = np.asarray(tet_particle_indices, dtype=np.int64).reshape(-1, 4)
    weights = np.asarray(barycentric_weights, dtype=np.float64).reshape(-1, 4)
    if len(indices) != len(weights):
        raise ValueError("tet index and barycentric-weight row counts differ")
    return np.sum(particles[indices] * weights[:, :, None], axis=1)


def _tet_binding_sha256(tet_particle_indices, barycentric_weights):
    digest = hashlib.sha256(b"tet_barycentric_v1\0")
    digest.update(np.asarray(tet_particle_indices, dtype="<i4").tobytes())
    digest.update(np.asarray(barycentric_weights, dtype="<f4").tobytes())
    return digest.hexdigest()


def _mesh_topology_sha256(vertex_count, face_vertex_counts, face_vertex_indices):
    topology = {
        "vertex_count": int(vertex_count),
        "face_vertex_counts": [int(value) for value in face_vertex_counts],
        "face_vertex_indices": [int(value) for value in face_vertex_indices],
    }
    payload = json.dumps(topology, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sparse_voxel_tet_mesh(points, face_vertex_indices, bbox_min, bbox_size, resolution):
    """Build a coarse tet proxy from only voxels touched by an authored mesh.

    This is used for ring-shaped meshes: unlike a full bounding-box grid,
    occupied voxels retain the large authored openings. Vertices, triangle
    edge midpoints, and triangle centroids seed occupancy so every authored
    vertex remains inside at least one tetrahedron while adjacent surface
    samples keep the sparse shell connected.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(face_vertex_indices, dtype=np.int64).reshape(-1, 3)
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_size = np.asarray(bbox_size, dtype=np.float64)
    resolution = np.asarray(resolution, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("voxel proxy points must be a non-empty Nx3 array")
    if np.any(resolution <= 0) or np.any(bbox_size <= 0.0):
        raise ValueError("voxel proxy resolution and bbox size must be positive")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError("voxel proxy triangle indices are out of range")

    tri_points = points[triangles]
    samples = np.concatenate(
        (
            points,
            np.mean(tri_points, axis=1),
            0.5 * (tri_points[:, 0] + tri_points[:, 1]),
            0.5 * (tri_points[:, 1] + tri_points[:, 2]),
            0.5 * (tri_points[:, 2] + tri_points[:, 0]),
        ),
        axis=0,
    )
    cell_size = bbox_size / resolution
    cell_indices = np.floor((samples - bbox_min) / cell_size).astype(np.int64)
    cell_indices = np.clip(cell_indices, 0, resolution - 1)
    occupied_set = {tuple(int(value) for value in row) for row in cell_indices}
    thin_axes = np.flatnonzero(resolution == 1)
    if len(thin_axes) == 1:
        thin_axis = int(thin_axes[0])
        plane_axes = [axis for axis in range(3) if axis != thin_axis]
        # A pair of voxels touching only at a corner creates a non-manifold
        # edge after tetrahedral surface extraction. Fill both missing cells
        # in each checkerboard 2x2 block; this preserves macroscopic holes but
        # makes the occupied sheet a face-connected polyomino.
        changed = True
        while changed:
            changed = False
            for first in range(int(resolution[plane_axes[0]]) - 1):
                for second in range(int(resolution[plane_axes[1]]) - 1):
                    cells = []
                    for d_first, d_second in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        index = [0, 0, 0]
                        index[plane_axes[0]] = first + d_first
                        index[plane_axes[1]] = second + d_second
                        cells.append(tuple(index))
                    present = [cell in occupied_set for cell in cells]
                    if present in ([True, False, False, True], [False, True, True, False]):
                        occupied_set.update(cells)
                        changed = True
    occupied = sorted(occupied_set)
    if not occupied:
        raise ValueError("voxel proxy has no occupied cells")

    lattice_vertices = {}
    vertices = []

    def vertex_index(x, y, z):
        key = (int(x), int(y), int(z))
        if key not in lattice_vertices:
            lattice_vertices[key] = len(vertices)
            vertices.append(bbox_min + np.asarray(key, dtype=np.float64) * cell_size)
        return lattice_vertices[key]

    tets = []
    for x, y, z in occupied:
        v0 = vertex_index(x, y, z)
        v1 = vertex_index(x + 1, y, z)
        v2 = vertex_index(x + 1, y, z + 1)
        v3 = vertex_index(x, y, z + 1)
        v4 = vertex_index(x, y + 1, z)
        v5 = vertex_index(x + 1, y + 1, z)
        v6 = vertex_index(x + 1, y + 1, z + 1)
        v7 = vertex_index(x, y + 1, z + 1)
        if (x & 1) ^ (y & 1) ^ (z & 1):
            cell_tets = (
                (v0, v1, v4, v3),
                (v2, v3, v6, v1),
                (v5, v4, v1, v6),
                (v7, v6, v3, v4),
                (v4, v1, v6, v3),
            )
        else:
            cell_tets = (
                (v1, v2, v5, v0),
                (v3, v0, v7, v2),
                (v4, v7, v0, v5),
                (v6, v5, v2, v7),
                (v5, v2, v7, v0),
            )
        tets.extend(cell_tets)

    vertices = np.asarray(vertices, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int32)
    for tet_index, tet in enumerate(tets):
        p, q, r, s = vertices[tet]
        if np.linalg.det(np.column_stack((q - p, r - p, s - p))) <= 0.0:
            tets[tet_index, 1], tets[tet_index, 2] = tets[tet_index, 2], tets[tet_index, 1]
    return vertices, tets, occupied


# ----------------------------------------------------------------------------
# Smooth render tube kernels (GPU-side; per-frame Catmull-Rom resample).
# ----------------------------------------------------------------------------
def _tube_indices(n_noodles, n_rings, n_theta):
    V = n_rings * n_theta + 2
    tris: list[int] = []
    for k in range(n_noodles):
        b = k * V
        for i in range(n_rings - 1):
            r0, r1 = b + i * n_theta, b + (i + 1) * n_theta
            for j in range(n_theta):
                j1 = (j + 1) % n_theta
                tris += [r0 + j, r1 + j, r1 + j1, r0 + j, r1 + j1, r0 + j1]
        head, tail = b + n_rings * n_theta, b + n_rings * n_theta + 1
        r0, rN = b, b + (n_rings - 1) * n_theta
        for j in range(n_theta):
            j1 = (j + 1) % n_theta
            tris += [head, r0 + j1, r0 + j]
            tris += [tail, rN + j, rN + j1]
    return np.array(tris, dtype=np.int32)


@wp.func
def _safe_norm(v: wp.vec3, fallback: wp.vec3) -> wp.vec3:
    L = wp.length(v)
    if L < 1.0e-9:
        return fallback
    return v / L


@wp.kernel
def _ring_centroids(
    particle_q: wp.array(dtype=wp.vec3),
    part_start: int,
    M: int,
    centroids: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    k = tid / M
    m = tid % M
    base = part_start + k * 4 * M + m
    c = particle_q[base] + particle_q[base + M] + particle_q[base + 2 * M] + particle_q[base + 3 * M]
    centroids[k * M + m] = c * 0.25


@wp.kernel
def _catmull_smooth(
    centroids: wp.array(dtype=wp.vec3),
    M: int,
    cr_W: wp.array2d(dtype=float),
    out_pts: int,
    smooth: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    k = tid / out_pts
    o = tid % out_pts
    acc = wp.vec3(0.0, 0.0, 0.0)
    for m in range(M):
        acc = acc + cr_W[o, m] * centroids[k * M + m]
    smooth[k * out_pts + o] = acc


@wp.kernel
def _tube_kernel(
    smooth: wp.array(dtype=wp.vec3),
    n_rings: int,
    n_theta: int,
    radius: float,
    verts: wp.array(dtype=wp.vec3),
    norms: wp.array(dtype=wp.vec3),
):
    k = wp.tid()
    V = n_rings * n_theta + 2
    bp = k * n_rings
    bv = k * V
    tau = 6.2831853071795864

    t0 = _safe_norm(smooth[bp + 1] - smooth[bp + 0], wp.vec3(1.0, 0.0, 0.0))
    ref = wp.vec3(0.0, 0.0, 1.0)
    if wp.abs(wp.dot(t0, ref)) > 0.9:
        ref = wp.vec3(1.0, 0.0, 0.0)
    n_prev = _safe_norm(ref - wp.dot(ref, t0) * t0, wp.vec3(1.0, 0.0, 0.0))

    for i in range(n_rings):
        if i == 0:
            t = smooth[bp + 1] - smooth[bp + 0]
        elif i == n_rings - 1:
            t = smooth[bp + n_rings - 1] - smooth[bp + n_rings - 2]
        else:
            t = smooth[bp + i + 1] - smooth[bp + i - 1]
        t = _safe_norm(t, wp.vec3(1.0, 0.0, 0.0))
        nn = _safe_norm(n_prev - wp.dot(n_prev, t) * t, n_prev)
        b = wp.cross(t, nn)
        c = smooth[bp + i]
        for j in range(n_theta):
            th = tau * float(j) / float(n_theta)
            radial = wp.cos(th) * nn + wp.sin(th) * b
            idx = bv + i * n_theta + j
            verts[idx] = c + radius * radial
            norms[idx] = radial
        n_prev = nn

    th0 = _safe_norm(smooth[bp + 1] - smooth[bp + 0], wp.vec3(1.0, 0.0, 0.0))
    thN = _safe_norm(smooth[bp + n_rings - 1] - smooth[bp + n_rings - 2], wp.vec3(1.0, 0.0, 0.0))
    hv = bv + n_rings * n_theta
    verts[hv] = smooth[bp + 0] - radius * th0
    norms[hv] = -th0
    verts[hv + 1] = smooth[bp + n_rings - 1] + radius * thN
    norms[hv + 1] = thN


def _catmull_rom_weights(n_pts, sub):
    out_pts = (n_pts - 1) * sub + 1
    W = np.zeros((out_pts, n_pts), dtype=np.float32)
    row = 0
    for i in range(n_pts - 1):
        i0, i1, i2, i3 = max(i - 1, 0), i, i + 1, min(i + 2, n_pts - 1)
        for j in range(sub):
            t = j / sub
            t2, t3 = t * t, t * t * t
            W[row, i0] += 0.5 * (-t3 + 2.0 * t2 - t)
            W[row, i1] += 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
            W[row, i2] += 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
            W[row, i3] += 0.5 * (t3 - t2)
            row += 1
    W[row, n_pts - 1] = 1.0
    return W


# ----------------------------------------------------------------------------
# Spherical-cap wok mesh.
# ----------------------------------------------------------------------------
def _make_wok(rim_r=0.160, depth=0.090, thickness=0.003, n_az=96, n_radial=32):
    R = (rim_r * rim_r + depth * depth) / (2.0 * depth)
    center_z = R - depth
    theta_max = math.acos((R - depth) / R)

    az = np.linspace(0.0, 2.0 * math.pi, n_az, endpoint=False)
    theta = np.linspace(0.0, theta_max, n_radial + 1)

    def sphere_pts(radius):
        cosT, sinT = np.cos(theta), np.sin(theta)
        cosA, sinA = np.cos(az), np.sin(az)
        x = radius * np.outer(sinT, cosA)
        y = radius * np.outer(sinT, sinA)
        z = center_z - radius * np.outer(cosT, np.ones_like(az))
        return np.stack([x, y, z], axis=-1)

    inside = sphere_pts(R).reshape(-1, 3)
    outside = sphere_pts(R + thickness).reshape(-1, 3)
    verts = np.concatenate([inside, outside], axis=0)

    n_in = inside.shape[0]
    cols = n_az

    def idx_in(r, a):
        return r * cols + (a % cols)

    def idx_out(r, a):
        return n_in + r * cols + (a % cols)

    tris = []
    for r in range(n_radial):
        for a in range(cols):
            v00, v01 = idx_in(r, a), idx_in(r, a + 1)
            v10, v11 = idx_in(r + 1, a), idx_in(r + 1, a + 1)
            tris.append((v00, v10, v11))
            tris.append((v00, v11, v01))
    for r in range(n_radial):
        for a in range(cols):
            v00, v01 = idx_out(r, a), idx_out(r, a + 1)
            v10, v11 = idx_out(r + 1, a), idx_out(r + 1, a + 1)
            tris.append((v00, v11, v10))
            tris.append((v00, v01, v11))
    r = n_radial
    for a in range(cols):
        vi0, vi1 = idx_in(r, a), idx_in(r, a + 1)
        vo0, vo1 = idx_out(r, a), idx_out(r, a + 1)
        tris.append((vi0, vi1, vo1))
        tris.append((vi0, vo1, vo0))

    return verts, np.asarray(tris, dtype=np.int64)


def _connected_vertex_components(vertices, faces):
    """Return mesh vertex-index components, largest face count first."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    parent = np.arange(len(vertices), dtype=np.int64)

    def find(index):
        index = int(index)
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, c in faces:
        union(a, b)
        union(a, c)

    grouped: dict[int, dict[str, Any]] = {}
    for face in faces:
        root = find(face[0])
        item = grouped.setdefault(root, {"vertices": set(), "face_count": 0})
        item["vertices"].update(int(index) for index in face)
        item["face_count"] += 1
    return sorted(
        (
            {
                "indices": np.asarray(sorted(item["vertices"]), dtype=np.int64),
                "face_count": int(item["face_count"]),
            }
            for item in grouped.values()
        ),
        key=lambda item: item["face_count"],
        reverse=True,
    )


def _fit_usda_wok_bowl(vertices, faces, body_axes_world):
    """Fit the bowl cap from the dominant bowl-shaped mesh component(s).

    Older authored meshes keep inner and outer bowl surfaces disconnected;
    decimated meshes may weld them together at the rim.  In both cases the
    dominant component identifies the bowl's symmetry axis.  Other components
    are accepted as bowl surfaces only when their two radial spans match the
    validated wok scale, preventing a large handle component from becoming the
    supposed outer surface.  The axis sign is selected by whichever body-local
    direction points toward world +Z, so ``up`` always points out of the bowl.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    components = _connected_vertex_components(vertices, faces)
    if not components:
        raise ValueError("USDA wok mesh has no connected surface components")

    dominant = vertices[components[0]["indices"]]
    symmetry_axis = int(np.argmin(np.ptp(dominant, axis=0)))
    axes_world = np.asarray(body_axes_world, dtype=np.float64).reshape(3, 3)
    sign = 1.0 if axes_world[symmetry_axis, 2] >= 0.0 else -1.0
    up = np.eye(3, dtype=np.float64)[symmetry_axis] * sign

    radial_indices = [index for index in range(3) if index != symmetry_axis]
    radial_0 = np.eye(3, dtype=np.float64)[radial_indices[0]]
    radial_1 = np.cross(up, radial_0)
    radial_1 /= np.linalg.norm(radial_1)

    bowl_surfaces = []
    for component in components:
        points = vertices[component["indices"]]
        radial_spans = np.ptp(points[:, radial_indices], axis=0)
        if (
            0.24 <= float(radial_spans[0]) <= 0.44
            and 0.24 <= float(radial_spans[1]) <= 0.44
            and 0.75 <= float(radial_spans[0] / radial_spans[1]) <= 1.25
        ):
            bowl_surfaces.append(points)
    if not bowl_surfaces:
        raise ValueError("USDA wok mesh has no bowl-scale surface component")

    # With disconnected surfaces, the inner one has the shallower bottom and
    # therefore the larger minimum projection along ``up``.  A welded shell is
    # already the only bowl-scale component, so fitting all of it differs only
    # by the physical wall thickness and remains inside the scale guard below.
    inner = max(bowl_surfaces, key=lambda points: float(np.min(points @ up)))
    p0 = inner @ radial_0
    p1 = inner @ radial_1
    pu = inner @ up
    center_0 = 0.5 * (float(np.min(p0)) + float(np.max(p0)))
    center_1 = 0.5 * (float(np.min(p1)) + float(np.max(p1)))
    rim_up = float(np.max(pu))
    rim_radius = 0.25 * (float(np.ptp(p0)) + float(np.ptp(p1)))
    depth = rim_up - float(np.min(pu))
    center = center_0 * radial_0 + center_1 * radial_1 + rim_up * up

    if not (0.12 <= rim_radius <= 0.22 and 0.05 <= depth <= 0.14):
        raise ValueError(f"USDA wok fit outside validated scale: rim_radius={rim_radius:g} depth={depth:g}")

    # Find the radial direction whose full-mesh extent protrudes farthest
    # beyond the fitted bowl.  That is the handle direction; later we use it
    # to apply Wuji's condim only to handle convex pieces.
    candidates = []
    for axis in (radial_0, radial_1):
        projected = vertices @ axis
        center_axis = float(center @ axis)
        candidates.extend(
            (
                (float(np.max(projected)) - center_axis - rim_radius, axis),
                (center_axis - float(np.min(projected)) - rim_radius, -axis),
            )
        )
    _, handle_axis = max(candidates, key=lambda item: item[0])

    # Estimate the grasp shaft radius from the separate long handle component.
    # The old Newton proxy used a fixed 9.5 mm radius, which is smaller than
    # the authored USDA wooden shaft.  Select a non-bowl component that is
    # long along the detected handle axis and narrow in both transverse axes,
    # then fit its middle 80% so ferrules/knobs do not inflate the radius.
    handle_radius = 0.0095
    axis_x = -np.asarray(handle_axis, dtype=np.float64)
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(up, axis_x)
    axis_y /= np.linalg.norm(axis_y)
    axis_z = up
    candidate_components = []
    for component in components[1:]:
        points = vertices[component["indices"]]
        local = (points - center) @ np.column_stack((axis_x, axis_y, axis_z))
        span_x = float(np.ptp(local[:, 0]))
        span_y = float(np.ptp(local[:, 1]))
        span_z = float(np.ptp(local[:, 2]))
        if span_x >= 0.08 and span_y <= 0.08 and span_z <= 0.08:
            candidate_components.append((span_x, local))
    if candidate_components:
        _, local = max(candidate_components, key=lambda item: item[0])
        x_min, x_max = float(np.min(local[:, 0])), float(np.max(local[:, 0]))
        shaft = local[(local[:, 0] >= x_min + 0.10 * (x_max - x_min)) & (local[:, 0] <= x_max - 0.10 * (x_max - x_min))]
        if len(shaft) >= 8:
            z_fit = np.polyfit(shaft[:, 0], shaft[:, 2], 1)
            z_center = np.polyval(z_fit, shaft[:, 0])
            y_center = float(np.median(shaft[:, 1]))
            radii = np.sqrt((shaft[:, 1] - y_center) ** 2 + (shaft[:, 2] - z_center) ** 2)
            handle_radius = max(0.0095, float(np.percentile(radii, 90)))

    return {
        "center": center,
        "up": up,
        "radial_0": radial_0,
        "radial_1": radial_1,
        "rim_radius": rim_radius,
        "depth": depth,
        "handle_axis": handle_axis,
        "handle_start": float(center @ handle_axis) + rim_radius,
        "handle_radius": handle_radius,
        "component_count": len(components),
    }


def _usda_wok_proxy_frame(fit):
    """Return the canonical-Newton-to-USDA wok frame in body coordinates."""
    center = np.asarray(fit["center"], dtype=np.float64)
    axis_z = np.asarray(fit["up"], dtype=np.float64)
    axis_z /= np.linalg.norm(axis_z)
    axis_x = -np.asarray(fit["handle_axis"], dtype=np.float64)
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(axis_z, axis_x)
    axis_y /= np.linalg.norm(axis_y)
    axis_x = np.cross(axis_y, axis_z)
    axis_x /= np.linalg.norm(axis_x)
    return center, np.column_stack((axis_x, axis_y, axis_z))


def _transform_wok_proxy_points(points, center, basis):
    points = np.asarray(points, dtype=np.float64)
    return np.asarray(center, dtype=np.float64)[None, :] + points @ np.asarray(basis, dtype=np.float64).T


# ----------------------------------------------------------------------------
# Particle / FEM-bar parameters. The plugin dataclass is the source of truth;
# scene YAML supplies robot, solver and per-scene overrides.
# ----------------------------------------------------------------------------
@dataclass
class WokGeometry:
    rim_r: float = 0.160
    depth: float = 0.090
    # World-frame home of the wok's rim plane.  The bowl extends ``depth``
    # below this; the stove ring (see StoveGeometry, computed from these
    # numbers) is placed to cradle the round bottom at this height.  home_x
    # sits in front of the robot; home_z is chosen so the arm's init pose
    # (scene yaml init_joint_pos) starts the gripper near the handle.
    home_x: float = 0.9949
    home_y: float = 0.0
    home_z: float = 1.090
    az: int = 96  # collider mesh azimuthal resolution
    radial: int = 32  # collider mesh radial resolution
    # Free-body inertial properties.  Mass is heavy enough that particle-vs-
    # mesh (food) contact can't shove the wok around, but light enough that a
    # teleoperated wrist can still lift/track it.
    mass: float = 1.3  # kg
    inertia_diag: tuple = (4.0e-3, 4.0e-3, 6.0e-3)  # I_xx, I_yy, I_zz
    # Wok physics mode:
    #   False = KINEMATIC: the wok is a kinematic free body parked at home in
    #     the ring.  It ignores gravity + contact and stays put unless the
    #     AUTO-FLIP controller drives it (Config.auto_flip).  Zero blow-up
    #     risk; the stove ring is cosmetic in this mode.  NOTE a kinematic
    #     body compiles to a MuJoCo *mocap* body — no DOFs, infinite effective
    #     mass — so the gripper can never move it no matter how good the
    #     contact is.  Use this mode only for render / noodle-settling work.
    #   True (default) = DYNAMIC: a real free rigid body under gravity.  The
    #     thin-shell rigid collider turns ON so the wok rests on the stove ring
    #     and the arm can push / GRASP it.  (The thick particle collider stays
    #     shape-collision-OFF regardless — see on_build.)  Required for the
    #     gripper-grasps-the-handle workflow.
    wok_dynamic: bool = True
    # DYNAMIC only: start the wok this far above its ring-seat height so it
    # settles onto the ring instead of spawning interpenetrating (a first-frame
    # overlap with gap=0 rigid contact would kick it).
    drop_gap: float = 0.004


@dataclass
class UsdObjects:
    """Binding contract for the art-authored fried-noodles scene.

    Disabled by default so the sibling legacy chef scene keeps its procedural
    wok/stove path.  The Wuji scene enables it explicitly in scene YAML.
    """

    enabled: bool = False
    objects_root: str = "/World/objects"
    pot_body_prim_path: str = "/World/objects/benchmark_pot_013/entity"
    pot_mesh_prim_path: str = "/World/objects/benchmark_pot_013/entity/body/visual"
    expected_food_count: int = 42
    expected_pot_mass: float = 1.3
    particle_proxy_thickness: float = 0.080
    # When enabled, imported food is hidden and stripped of rigid physics in
    # an anonymous session layer so it can be replaced by soft proxies.
    procedural_food_transition: bool = False


@dataclass
class UsdNoodles:
    """Authored noodle family compiled to Newton soft proxies."""

    enabled: bool = False
    object_name_prefix: str = "benchmark_noodles_001_"
    expected_instance_count: int = 16
    expected_vertex_count: int = 1100
    expected_topology_sha256: str = "cc12e7d653fb61af0178055723c0cb2644f4fb997a0776d1073950b292253049"


@dataclass
class UsdSideDishes:
    """Audited USDA side-dish family classification.

    ``families`` is supplied by scene YAML because the asset's generic
    ``semantic_name=side_dish`` is insufficient to select a physics profile.
    ``migrated_families`` selects complete families for soft-proxy creation;
    unselected families remain visual-only data in the source USDA.
    """

    enabled: bool = False
    expected_instance_count: int = 26
    migrated_families: tuple = ()
    procedural_fallback: bool = True
    families: dict = field(default_factory=dict)


@dataclass
class GraspGeometry:
    """Form-closure grasp interface: a square grip section on the wok handle
    plus flush pads on the gripper's two support links.

    WHY A BOX AND NOT THE ROUND STICK.  In mujoco-warp a (CAPSULE, MESH) pair
    routes through GJK/EPA, which emits exactly ONE contact point per pair;
    ``SolverMuJoCo(enable_multiccd=...)`` defaults to False in newton 1.4.0 and
    a capsule can never be a multi-CCD operand anyway.  With one point per pad,
    the two contacts sit at opposite ends of a diameter — and the normal forces
    then produce *mathematically zero* net moment about the jaw-closing axis,
    no matter how high μ goes.  The wok would pivot out of the jaws on the
    first toss.  BOX-BOX is the one pair type that multicontacts
    unconditionally, giving a real 4-point patch per pad and non-zero capacity
    about all three axes.  A square-section wooden wok-spatula grip is also
    perfectly plausible physically.

    Grip-section extent is chosen to sit between the ferrule brackets (which
    end at wok-local x = -0.2029) and the end knob (starts ≈ -0.3139), leaving
    round stubs at both ends so the handle still reads as a wooden stick.

    TWO MODES, picked by ``enable``:

    * ``enable=True``  — PARALLEL-JAW gripper (swiftpicker).  Square grip box +
      flush pads on the two support links + explicit ``mujoco:pair`` rows, all
      isolated in a private ``collision_group`` so the interface only ever
      touches itself.  Requires the ``pad_link_names`` links to exist.
    * ``enable=False`` — MULTI-FINGER hand (wuji).  The handle stays a ROUND
      capsule but becomes a real collider in the DEFAULT collision group, so the
      hand's own finger meshes contact it directly.  No box, no pads, no pairs.
      A five-finger envelope gets its contact manifold from many finger links at
      once, so it does not need the box's form closure — and a round stick is
      what a hand actually wraps.

    Either way the graspable shape carries ``mujoco:condim`` (see ``condim``) and
    ``gap=0.0``, and friction must come from the scene yaml rather than from a
    ShapeConfig here — ``mjc_contact.apply_pre_finalize`` runs after this hook
    and overwrites every shape's friction, but never touches condim or pairs.
    """

    # False for a multi-finger hand, True for a parallel-jaw gripper.
    enable: bool = True
    # Optional robot-hand collision-gap override.  Newton's USD import gives
    # robot geoms builder.rigid_gap=0.1 m; in MuJoCo-Warp that creates a large
    # inactive-contact shell which can consume the finite contact buffer around
    # a dense five-finger grasp.  Empty prefix disables the override (the
    # parallel-jaw scene keeps its imported geometry unchanged).  The Wuji
    # scene selects ``hand_r_`` and sets the gap to zero.
    hand_body_prefix: str = ""
    hand_geom_gap: float = 0.0
    # ROUND-handle mode only.  Radius of the collar added at the INBOARD end of
    # the grip section, forming a shoulder over the 0.0095 stick.  With the end
    # knob (r=0.0124) bounding the outboard end, the graspable span then has a
    # mechanical stop at both ends — the only way to resist AXIAL load, since a
    # smooth cylinder has no form closure along its own axis and axial force can
    # otherwise only be carried by mu*N.  Set <= 0.0095 to disable.
    collar_radius: float = 0.0124
    # Grip section, along the handle in wok-body-local coords.  The wooden
    # stick runs (-0.2029, 0, -0.0226) → (-0.3139, 0, -0.0131); the box spans
    # the middle of it and inherits the stick's slight rise in z.
    grip_x0: float = -0.2100
    grip_x1: float = -0.2820
    # Half-extents ⟂ the handle.  0.0095 matches the stick radius, so the
    # square section circumscribes the same 19 mm envelope.
    grip_half: float = 0.0095
    # Gripper pad boxes.  Measured directly off right_support_link.STL by
    # isolating the coplanar -x-normal triangles: the dominant flat face is
    # 541 triangles at link-local x = -0.02550 spanning y ∈ [0.0240, 0.0635],
    # z ∈ [-0.0115, 0.0115] — a real 39.5 × 23 mm / 603 mm² jaw face.  (A
    # second 234-triangle sliver sits 0.15 mm proud at -0.02565 but is only
    # 11 mm², so the seating plane is -0.02550.)  The pad box sits flush:
    # its outer face lands exactly on the mesh face.
    pad_link_names: tuple = ("gripper_r_left_support_link", "gripper_r_right_support_link")
    pad_face_x: float = -0.02550
    pad_half_thick: float = 0.002
    pad_half_len: float = 0.019
    pad_half_wid: float = 0.0115
    pad_center_y: float = 0.04375
    # Shallow retaining lips on the gripper-side pad. The flat pad alone lets
    # the square handle escape in either tangential direction during a fast
    # toss; these four edge stops provide form closure without a welded wok.
    lip_enable: bool = True
    lip_half_x: float = 0.004
    lip_half_y: float = 0.003
    lip_half_z: float = 0.003
    # Contact dimensionality for the grasp.  >=4 adds torsional friction about
    # the contact normal, the only thing that resists spin about the closing
    # axis; 6 would add rolling.  Authored as the ``mujoco:condim`` custom
    # attribute, which ``mjc_contact.apply_pre_finalize`` does NOT overwrite.
    # Newton's parallel-jaw grasp uses condim=4; Wuji overrides this to 6.
    condim: int = 4
    # Explicit mujoco:pair parameters for grip-box-vs-pad.  Friction MUST be
    # set here rather than on the ShapeConfig: apply_pre_finalize runs after
    # on_build and unconditionally rewrites every shape's
    # shape_material_mu/mu_torsional/mu_rolling, but never touches pairs.
    # Order is MuJoCo's vec5: (slide1, slide2, torsional, roll1, roll2).
    pair_friction: tuple = (2.0, 2.0, 0.005, 0.0001, 0.0001)
    # Pair rows take all contact parameters from these arrays, not the geoms.
    pair_solref: tuple = (0.02, 1.0)
    pair_solimp: tuple = (0.9, 0.95, 0.001, 0.5, 2.0)
    # Private collision group shared by the grip box and both pads.
    #
    # These shapes MUST have has_shape_collision=True: newton 1.4.0 cross-checks
    # ``len(spec.geoms) != colliding_shapes_per_world + tendon_extra_geoms`` and
    # raises "The number of geoms in the MuJoCo model does not match the number
    # of colliding shapes" if a mujoco:pair references a non-colliding shape —
    # the keep-non-colliding-geom exemption covers spatial tendons only, not
    # pairs.
    #
    # But we don't want the grasp geometry colliding with everything else, so
    # it goes in its own collision group: broad-phase then only ever considers
    # grip-vs-pad, and the explicit pair supplies that contact's parameters.
    # Anything outside the group (stove, noodles, robot links, the wok's own
    # bowl shell) is filtered out before narrowphase.
    collision_group: int = 99


@dataclass
class StoveGeometry:
    """Static stove under the wok: a counter box + a wok-support ring.

    The ring (a circle of beads) cradles the wok's spherical-cap bottom so a
    DYNAMIC wok rests stably on a circle of contacts (a round bottom on a flat
    top would wobble).  All placement is COMPUTED from ``WokGeometry`` in
    ``ScenePlugin._add_stove`` so it tracks home_z / rim_r / depth
    automatically.  In PARKED/AUTO-FLIP modes the stove is cosmetic (the
    kinematic wok doesn't rest on it), but the ring still frames the wok.
    """

    enable: bool = True
    # Counter base box (static, collidable).  ``base_top_clearance`` sets how
    # far the box top sits BELOW the wok bowl's lowest point, so the round
    # bottom pokes down into the ring without hitting the box.
    base_half_x: float = 0.22
    base_half_y: float = 0.22
    base_top_clearance: float = 0.010
    base_color: tuple = (0.18, 0.19, 0.21)
    # Wok-support ring: ``ring_radius`` is the radius (from the wok axis) at
    # which the beads grip the bowl; keep it < rim_r so the beads catch the
    # rounded flank, not the rim lip.
    ring_radius: float = 0.11
    ring_beads: int = 16
    bead_radius: float = 0.010
    ring_color: tuple = (0.34, 0.35, 0.37)
    # Rigid friction for stove + ring (helps a DYNAMIC wok not slide out).
    mu: float = 0.6
    # newton 1.4.0: pass gap=0 so MuJoCo doesn't inherit rigid_gap=0.1 and
    # float the wok ~10 cm above the ring.
    gap: float = 0.0


@dataclass
class XPBDTuning:
    # XPBD-internal knobs not surfaced through the engine constructor
    # (the engine reads model.particle_max_velocity + soft_contact_*).
    # The two solver-relax values are set on the SolverXPBD and read
    # back by the engine at build-time via ``newton.xpbd:`` in the yaml.
    soft_body_relaxation: float = 0.9
    soft_contact_relaxation: float = 0.3  # 0.9 default is jittery; lower = calmer rest
    max_vel: float = 12.0  # particle-velocity clamp (m/s)


@dataclass
class WokContact:
    # Wok ↔ noodle contact (write-through to model.soft_contact_*).
    soft_ke: float = 1.0e3
    soft_kd: float = 1.0
    soft_kf: float = 1.0e3
    soft_radius: float = 0.0017
    soft_margin: float = 0.01


@dataclass
class NoodleParams:
    n_noodles: int = 16
    n_len: int = 80
    length: float = 0.26
    cross: float = 0.0026  # square cross-section side (m)
    density: float = 1100.0
    k_mu: float = 2.0e3  # FEM shear Lame (firmness)
    k_lambda: float = 1.0e4  # FEM volume Lame (resists squashing)
    k_damp: float = 0.75
    self_radius: float = 0.0013  # noodle-vs-noodle XPBD self-contact radius
    particle_mu: float = 0.5  # noodle-vs-noodle friction
    wok_mu: float = 0.3  # noodle-vs-wok friction
    drop_z0_offset: float = (
        0.015  # just above rim — small drop so the oversized noodles spread as they enter (cramming them at the bowl bottom self-intersects → ejected)
    )
    drop_dz: float = 0.006
    spawn_disk_r: float = 0.020  # keeps each 0.26 m strand inside the 0.16 m rim
    color: tuple = (0.93, 0.83, 0.46)
    vis_radius_scale: float = 1.15


@dataclass
class HamParams:
    n_cubes: int = 8
    dim: int = 2  # cells per side; (dim+1)^3 particles
    size: float = 0.009  # cube side (m)
    density: float = 1050.0
    k_mu: float = 3.0e3
    k_lambda: float = 1.5e4
    k_damp: float = 1.0
    radius_scale: float = 0.5
    color: tuple = (0.89, 0.49, 0.47)
    drop_z0_offset: float = 0.025  # just above rim, gentle drop
    drop_dz: float = 0.006
    spawn_disk_r: float = 0.045


@dataclass
class StripParams:
    """Short FEM-bar julienne (青椒丝 / 胡萝卜丝)."""

    n: int = 12
    n_len: int = 7
    length: float = 0.050
    cross: float = 0.0022
    density: float = 900.0
    k_mu: float = 1.0e3
    k_lambda: float = 5.0e3
    k_damp: float = 0.8
    color: tuple = (0.30, 0.52, 0.18)
    vis_radius_scale: float = 1.0
    drop_z0_offset: float = 0.020  # just above rim, gentle drop
    drop_dz: float = 0.005
    spawn_disk_r: float = 0.045


@dataclass
class Controller:
    settle_time: float = 1.2
    toss_t: float = 0.40
    dwell_t: float = 0.35
    toss_dip: float = 0.010
    toss_lift: float = 0.075
    toss_fwd: float = 0.010
    toss_back: float = 0.015
    toss_pitch: float = 0.38
    gather_t: float = 0.7
    gather_pitch: float = 0.34
    gather_edge: float = 0.030
    gather_settle: float = 0.05
    gather_lift: float = 0.012
    catch_lead: float = 0.7
    catch_vmax: float = 1.2
    catch_reach: float = 0.14
    catch_return: float = 0.08
    catch_level: float = 0.15
    catch_tilt: float = 1.6
    catch_tilt_max: float = 0.20
    catch_recenter_vmax: float = 0.30
    centered_radius: float = 0.05
    settled_speed: float = 0.06
    catch_timeout: float = 2.5


@dataclass
class Render:
    vis_sub: int = 3  # capsules per tet cell (lengthwise smoothness)
    n_theta: int = 10  # cross-section roundness
    vis_radius: float = 1.15  # tube radius as multiple of cross/2


@dataclass
class FoodStatePublish:
    """ROS recording streams for chef-food particles and static bindings."""

    enabled: bool = False
    state_topic: str = "chef_food/state"
    info_topic: str = "chef_food/info"
    mapping_topic: str = "chef_food/mapping"
    rigid_state_topic: str = "chef_food/rigid_state"
    frame_id: str = "map"
    hz: float = 30.0


@dataclass
class StartupHandMotion:
    """Delayed right-hand target ramp for collision-safe scene startup."""

    enabled: bool = False
    delay_s: float = 1.0
    duration_s: float = 2.0
    hold_s: float = 0.5
    topic: str = "/joint_command"
    target_joint_pos: dict = field(default_factory=dict)


@dataclass
class Config:
    wok: WokGeometry = field(default_factory=WokGeometry)
    usd_objects: UsdObjects = field(default_factory=UsdObjects)
    usd_noodles: UsdNoodles = field(default_factory=UsdNoodles)
    usd_side_dishes: UsdSideDishes = field(default_factory=UsdSideDishes)
    stove: StoveGeometry = field(default_factory=StoveGeometry)
    grasp: GraspGeometry = field(default_factory=GraspGeometry)
    xpbd: XPBDTuning = field(default_factory=XPBDTuning)
    contact: WokContact = field(default_factory=WokContact)
    noodle: NoodleParams = field(default_factory=NoodleParams)
    ham: HamParams = field(default_factory=HamParams)
    pepper: StripParams = field(
        default_factory=lambda: StripParams(
            n=12,
            n_len=5,
            length=0.036,
            color=(0.24, 0.52, 0.15),
        )
    )
    carrot: StripParams = field(
        default_factory=lambda: StripParams(
            n=10,
            cross=0.0024,
            k_mu=1.2e3,
            k_lambda=6.0e3,
            color=(0.94, 0.48, 0.10),
            drop_z0_offset=0.018,  # just above rim, gentle drop
        )
    )
    controller: Controller = field(default_factory=Controller)
    render: Render = field(default_factory=Render)
    food_state: FoodStatePublish = field(default_factory=FoodStatePublish)
    startup_hand_motion: StartupHandMotion = field(default_factory=StartupHandMotion)
    # Default OFF — the demo is driven by TELEOP now.  Honored only in the
    # KINEMATIC wok mode; if ``wok.wok_dynamic`` is True this is ignored
    # (dynamic wins) with a warning in on_build.
    auto_flip: bool = False


CONFIG = Config()
G_ACC = 9.81


def _startup_joint_targets_radians(
    start_degrees: dict,
    target_degrees: dict,
    delay_s: float,
    duration_s: float,
    sim_time: float,
):
    """Return a smoothstep-interpolated ROS joint target, or None while waiting."""
    if float(duration_s) <= 0.0:
        raise ValueError("startup hand motion duration_s must be positive")
    if float(sim_time) < float(delay_s):
        return None
    u = min(1.0, max(0.0, (float(sim_time) - float(delay_s)) / float(duration_s)))
    blend = u * u * (3.0 - 2.0 * u)
    return {
        name: math.radians(float(start_degrees[name]) + blend * (float(value) - float(start_degrees[name])))
        for name, value in target_degrees.items()
    }


class _StartupHandCommandPublisher:
    """Publish one finite startup trajectory through the normal command path."""

    def __init__(self, cfg: Config, scene_cfg: dict, logger):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState

        motion = cfg.startup_hand_motion
        initial = dict(((scene_cfg.get("robot") or {}).get("init_joint_pos")) or {})
        target = dict(motion.target_joint_pos or {})
        if float(motion.delay_s) < 0.0:
            raise ValueError("startup_hand_motion.delay_s must be non-negative")
        if float(motion.duration_s) <= 0.0:
            raise ValueError("startup_hand_motion.duration_s must be positive")
        if float(motion.hold_s) < 0.0:
            raise ValueError("startup_hand_motion.hold_s must be non-negative")
        if not target or any(not str(name).startswith("hand_r_") for name in target):
            raise ValueError("startup_hand_motion.target_joint_pos must contain right-hand joints")
        missing = sorted(set(target) - set(initial))
        if missing:
            raise ValueError("startup_hand_motion target joints are missing from robot.init_joint_pos: " f"{missing}")

        self._motion = motion
        self._start = {name: float(initial[name]) for name in target}
        self._target = {name: float(value) for name, value in target.items()}
        self._names = list(target)
        self._JointState = JointState
        self._started = False
        self._finished = False
        self._logger = logger

        if not rclpy.ok():
            rclpy.init(args=[])
        self._node = rclpy.create_node("chef_startup_hand_motion")
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._publisher = self._node.create_publisher(JointState, str(motion.topic), qos)
        logger.info(
            f"[chef] startup hand motion armed: joints={len(self._names)} "
            f"delay={motion.delay_s:g}s duration={motion.duration_s:g}s "
            f"hold={motion.hold_s:g}s topic={motion.topic}"
        )

    def publish(self, sim_time: float) -> None:
        if self._finished:
            return
        values = _startup_joint_targets_radians(
            self._start,
            self._target,
            self._motion.delay_s,
            self._motion.duration_s,
            sim_time,
        )
        if values is None:
            return
        if not self._started:
            self._started = True
            self._logger.info("[chef] startup hand motion started")

        message = self._JointState()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.name = self._names
        message.position = [values[name] for name in self._names]
        self._publisher.publish(message)

        end_time = float(self._motion.delay_s) + float(self._motion.duration_s) + float(self._motion.hold_s)
        if float(sim_time) >= end_time:
            self._finished = True
            self._logger.info("[chef] startup hand motion reached configured grasp")


class _ChefFoodStatePublisher:
    """Publish replayable, instance-labelled raw Newton particles.

    Dynamic state uses a standard PointCloud2 so rosbag, PCL and Python tools
    can consume it without a custom interface package. Static instance
    metadata is JSON on a transient-local String topic. The authored-vertex
    map is a second transient-local PointCloud2 published once, so the large
    static mapping is not repeated in every dynamic frame.
    """

    _POINT_DTYPE = np.dtype(
        {
            "names": (
                "x",
                "y",
                "z",
                "food_type",
                "instance_id",
                "local_index",
                "global_index",
            ),
            "formats": ("<f4", "<f4", "<f4", "u1", "<u2", "<u4", "<u4"),
            "offsets": (0, 4, 8, 12, 14, 16, 20),
            "itemsize": 24,
        }
    )

    _MAPPING_DTYPE = np.dtype(
        {
            "names": (
                "binding_index",
                "vertex_index",
                "particle_0",
                "particle_1",
                "particle_2",
                "particle_3",
                "weight_0",
                "weight_1",
                "weight_2",
                "weight_3",
            ),
            "formats": (
                "<u2",
                "<u4",
                "<u4",
                "<u4",
                "<u4",
                "<u4",
                "<f4",
                "<f4",
                "<f4",
                "<f4",
            ),
            "offsets": (0, 4, 8, 12, 16, 20, 24, 28, 32, 36),
            "itemsize": 40,
        }
    )

    _RIGID_DTYPE = np.dtype(
        {
            "names": ("body_index", "x", "y", "z", "qw", "qx", "qy", "qz"),
            "formats": ("<u4", "<f4", "<f4", "<f4", "<f4", "<f4", "<f4", "<f4"),
            "offsets": (0, 4, 8, 12, 16, 20, 24, 28),
            "itemsize": 32,
        }
    )

    def __init__(
        self,
        cfg: Config,
        instances: list[dict],
        deformable_bindings: list[dict],
        particle_count: int,
        physics_hz: float,
        logger,
    ):
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import String

        if not rclpy.ok():
            # The C++ bridge owns a separate rclcpp context.  Empty args avoid
            # inheriting the engine entry point's non-ROS flags (e.g. --gui).
            rclpy.init(args=[])
        self._node = rclpy.create_node("chef_food_state_publisher", namespace="/genie_sim_engine")
        self._PointCloud2 = PointCloud2
        self._String = String
        self._frame_id = str(cfg.food_state.frame_id)
        self._publish_period_ticks = max(
            1,
            int(round(float(physics_hz) / max(float(cfg.food_state.hz), 1.0e-6))),
        )
        self._ticks_since_state = 0
        self._next_info_time = 0.0
        self._last_sim_time = -math.inf

        info_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._info_pub = self._node.create_publisher(String, str(cfg.food_state.info_topic), info_qos)
        self._mapping_pub = self._node.create_publisher(
            PointCloud2,
            str(cfg.food_state.mapping_topic),
            info_qos,
        )
        self._rigid_pub = self._node.create_publisher(
            PointCloud2,
            str(cfg.food_state.rigid_state_topic),
            qos_profile_sensor_data,
        )
        self._state_pub = self._node.create_publisher(
            PointCloud2,
            str(cfg.food_state.state_topic),
            qos_profile_sensor_data,
        )
        self._fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="food_type", offset=12, datatype=PointField.UINT8, count=1),
            PointField(name="instance_id", offset=14, datatype=PointField.UINT16, count=1),
            PointField(name="local_index", offset=16, datatype=PointField.UINT32, count=1),
            PointField(name="global_index", offset=20, datatype=PointField.UINT32, count=1),
        ]
        self._mapping_fields = [
            PointField(name="binding_index", offset=0, datatype=PointField.UINT16, count=1),
            PointField(name="vertex_index", offset=4, datatype=PointField.UINT32, count=1),
            PointField(name="particle_0", offset=8, datatype=PointField.UINT32, count=1),
            PointField(name="particle_1", offset=12, datatype=PointField.UINT32, count=1),
            PointField(name="particle_2", offset=16, datatype=PointField.UINT32, count=1),
            PointField(name="particle_3", offset=20, datatype=PointField.UINT32, count=1),
            PointField(name="weight_0", offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name="weight_1", offset=28, datatype=PointField.FLOAT32, count=1),
            PointField(name="weight_2", offset=32, datatype=PointField.FLOAT32, count=1),
            PointField(name="weight_3", offset=36, datatype=PointField.FLOAT32, count=1),
        ]
        self._rigid_fields = [
            PointField(name="body_index", offset=0, datatype=PointField.UINT32, count=1),
            PointField(name="x", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="qw", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="qx", offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name="qy", offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name="qz", offset=28, datatype=PointField.FLOAT32, count=1),
        ]
        self._body_paths: tuple[str, ...] | None = None

        ordered = sorted(instances, key=lambda item: int(item["particle_start"]))
        bindings_by_path = {str(item["mesh_prim_path"]): item for item in deformable_bindings}
        if len(bindings_by_path) != len(deformable_bindings):
            raise ValueError("chef-food deformable Mesh paths must be unique")
        global_indices = []
        food_types = []
        instance_ids = []
        local_indices = []
        mapping_rows = []
        serializable_instances = []
        mapping_offset = 0
        for binding_index, item in enumerate(ordered):
            start = int(item["particle_start"])
            count = int(item["particle_count"])
            if start < 0 or count <= 0 or start + count > particle_count:
                raise ValueError(
                    f"invalid chef-food particle range {start}:{start + count} "
                    f"for model.particle_count={particle_count}"
                )
            global_indices.append(np.arange(start, start + count, dtype=np.uint32))
            food_types.append(np.full(count, int(item["food_type"]), dtype=np.uint8))
            instance_ids.append(np.full(count, int(item["instance_id"]), dtype=np.uint16))
            local_indices.append(np.arange(count, dtype=np.uint32))

            mesh_path = str(item.get("mesh_prim_path", ""))
            binding = bindings_by_path.get(mesh_path)
            if binding is None:
                raise ValueError(f"missing authored deformable mapping for {mesh_path!r}")
            vertex_count = int(item.get("vertex_count", -1))
            tet_indices = np.asarray(binding.get("tet_particle_indices"), dtype=np.int64)
            weights = np.asarray(binding.get("barycentric_weights"), dtype=np.float32)
            expected_shape = (vertex_count, 4)
            if tet_indices.shape != expected_shape or weights.shape != expected_shape:
                raise ValueError(
                    f"chef-food mapping {mesh_path} must have shape {expected_shape}; "
                    f"got {tet_indices.shape} and {weights.shape}"
                )
            if np.any(tet_indices < 0) or np.any(tet_indices >= count):
                raise ValueError(f"chef-food mapping {mesh_path} escapes its particle slice")
            if not np.isfinite(weights).all() or not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-5):
                raise ValueError(f"chef-food mapping {mesh_path} has invalid weights")

            rows = np.zeros(vertex_count, dtype=self._MAPPING_DTYPE)
            rows["binding_index"] = binding_index
            rows["vertex_index"] = np.arange(vertex_count, dtype=np.uint32)
            global_tets = tet_indices.astype(np.uint32) + np.uint32(start)
            for column in range(4):
                rows[f"particle_{column}"] = global_tets[:, column]
                rows[f"weight_{column}"] = weights[:, column]
            mapping_rows.append(rows)

            parent_path = str(item.get("parent_prim_path", ""))
            if not parent_path.startswith("/") or not mesh_path.startswith("/"):
                raise ValueError(f"chef-food binding paths must be absolute: {parent_path!r}, {mesh_path!r}")
            metadata = dict(item)
            metadata.update(
                {
                    "binding_index": binding_index,
                    "name": str(item["instance_prim_path"]).rsplit("/", 1)[-1],
                    "object_key": parent_path.lstrip("/").replace("/", "_"),
                    "mapping_offset": mapping_offset,
                    "mapping_count": vertex_count,
                }
            )
            serializable_instances.append(metadata)
            mapping_offset += vertex_count

        self._global_indices = np.concatenate(global_indices) if global_indices else np.empty(0, dtype=np.uint32)
        self._points = np.zeros(len(self._global_indices), dtype=self._POINT_DTYPE)
        if len(self._points):
            self._points["food_type"] = np.concatenate(food_types)
            self._points["instance_id"] = np.concatenate(instance_ids)
            self._points["local_index"] = np.concatenate(local_indices)
            self._points["global_index"] = self._global_indices
        # ``np.concatenate`` repacks structured dtypes and would silently
        # remove the two padding bytes after ``binding_index`` (38-byte rows
        # despite the advertised 40-byte PointCloud2 stride). Copy into an
        # explicitly typed destination to preserve the wire offsets.
        self._mapping = np.empty(mapping_offset, dtype=self._MAPPING_DTYPE)
        cursor = 0
        for rows in mapping_rows:
            self._mapping[cursor : cursor + len(rows)] = rows
            cursor += len(rows)

        type_defs = {
            1: {
                "name": "noodle",
                "proposed_usd_root": "/World/ChefFood/noodles",
                "color_rgb": list(cfg.noodle.color),
            },
            2: {
                "name": "ham",
                "proposed_usd_root": "/World/ChefFood/ham",
                "color_rgb": list(cfg.ham.color),
            },
            3: {
                "name": "pepper",
                "proposed_usd_root": "/World/ChefFood/pepper",
                "color_rgb": list(cfg.pepper.color),
            },
            4: {
                "name": "carrot",
                "proposed_usd_root": "/World/ChefFood/carrot",
                "color_rgb": list(cfg.carrot.color),
            },
        }
        topology_payload = {
            "schema_version": 2,
            "frame_id": self._frame_id,
            "state_topic": str(cfg.food_state.state_topic),
            "mapping_topic": str(cfg.food_state.mapping_topic),
            "rigid_state_topic": str(cfg.food_state.rigid_state_topic),
            "point_step": int(self._POINT_DTYPE.itemsize),
            "point_count": int(len(self._points)),
            "model_particle_count": int(particle_count),
            "mapping_point_step": int(self._MAPPING_DTYPE.itemsize),
            "mapping_point_count": int(len(self._mapping)),
            "rigid_point_step": int(self._RIGID_DTYPE.itemsize),
            "body_paths": [],
            "points_coordinate_space": "world",
            "fields": [
                {"name": f.name, "offset": int(f.offset), "datatype": int(f.datatype), "count": int(f.count)}
                for f in self._fields
            ],
            "food_types": {str(key): value for key, value in type_defs.items()},
            "instances": serializable_instances,
        }
        self._topology_payload = topology_payload
        self._refresh_info_json()
        self._publish_info()
        self._publish_mapping()
        logger.info(
            f"[chef] food state publisher: points={len(self._points)} "
            f"instances={len(ordered)} hz={cfg.food_state.hz:g} "
            f"topics=/{self._node.get_namespace().strip('/')}/{cfg.food_state.state_topic}, "
            f"/{self._node.get_namespace().strip('/')}/{cfg.food_state.info_topic}, "
            f"/{self._node.get_namespace().strip('/')}/{cfg.food_state.mapping_topic}, "
            f"/{self._node.get_namespace().strip('/')}/{cfg.food_state.rigid_state_topic} "
            f"topology={self._topology_payload['topology_sha256'][:12]}"
        )

    def _refresh_info_json(self) -> None:
        payload = dict(self._topology_payload)
        payload.pop("topology_sha256", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["topology_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self._topology_payload = payload
        self._info_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _publish_info(self) -> None:
        msg = self._String()
        msg.data = self._info_json
        self._info_pub.publish(msg)

    def _publish_mapping(self) -> None:
        msg = self._PointCloud2()
        msg.header.frame_id = self._frame_id
        msg.height = 1
        msg.width = len(self._mapping)
        msg.fields = self._mapping_fields
        msg.is_bigendian = False
        msg.point_step = int(self._MAPPING_DTYPE.itemsize)
        msg.row_step = msg.point_step * msg.width
        msg.data = self._mapping.tobytes()
        msg.is_dense = True
        self._mapping_pub.publish(msg)

    def publish(self, particle_positions, body_xyzwxyz, body_paths, sim_time: float) -> None:
        if sim_time < self._last_sim_time:
            # Simulation reset: restart the tick gate from the new epoch.
            self._ticks_since_state = 0
            self._next_info_time = sim_time
        self._last_sim_time = sim_time
        self._ticks_since_state += 1
        if self._ticks_since_state < self._publish_period_ticks:
            return
        self._ticks_since_state = 0
        if particle_positions is None or body_xyzwxyz is None:
            return

        body_paths = tuple(str(path) for path in body_paths)
        if self._body_paths is None:
            if len(set(body_paths)) != len(body_paths):
                raise ValueError("chef-food rigid snapshot paths must be unique")
            self._body_paths = body_paths
            self._topology_payload["body_paths"] = list(body_paths)
            self._refresh_info_json()
            self._publish_info()
        elif body_paths != self._body_paths:
            raise ValueError("chef-food rigid snapshot topology changed at runtime")

        positions = np.asarray(particle_positions)
        rigid = np.asarray(body_xyzwxyz, dtype=np.float32)
        if rigid.shape != (len(body_paths), 7) or not np.isfinite(rigid).all():
            raise ValueError(f"chef-food rigid snapshot must have shape ({len(body_paths)}, 7) and be finite")
        selected = np.asarray(positions)[self._global_indices]
        self._points["x"] = selected[:, 0]
        self._points["y"] = selected[:, 1]
        self._points["z"] = selected[:, 2]

        sec = int(sim_time)
        nsec = int((sim_time - sec) * 1.0e9)
        msg = self._PointCloud2()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nsec
        msg.header.frame_id = self._frame_id
        msg.height = 1
        msg.width = len(self._points)
        msg.fields = self._fields
        msg.is_bigendian = False
        msg.point_step = int(self._POINT_DTYPE.itemsize)
        msg.row_step = msg.point_step * msg.width
        msg.data = self._points.tobytes()
        msg.is_dense = bool(np.isfinite(selected).all())
        self._state_pub.publish(msg)

        rigid_rows = np.empty(len(body_paths), dtype=self._RIGID_DTYPE)
        rigid_rows["body_index"] = np.arange(len(body_paths), dtype=np.uint32)
        for column, field in enumerate(("x", "y", "z", "qw", "qx", "qy", "qz")):
            rigid_rows[field] = rigid[:, column]
        rigid_msg = self._PointCloud2()
        rigid_msg.header.stamp.sec = sec
        rigid_msg.header.stamp.nanosec = nsec
        rigid_msg.header.frame_id = self._frame_id
        rigid_msg.height = 1
        rigid_msg.width = len(rigid_rows)
        rigid_msg.fields = self._rigid_fields
        rigid_msg.is_bigendian = False
        rigid_msg.point_step = int(self._RIGID_DTYPE.itemsize)
        rigid_msg.row_step = rigid_msg.point_step * rigid_msg.width
        rigid_msg.data = rigid_rows.tobytes()
        rigid_msg.is_dense = True
        self._rigid_pub.publish(rigid_msg)

        if sim_time + 1.0e-12 >= self._next_info_time:
            self._publish_info()
            self._next_info_time = sim_time + 1.0


# ----------------------------------------------------------------------------
# Emit helpers — return start indices and topology metadata for the render
# stage to find the particles.
# ----------------------------------------------------------------------------
def _emit_noodles(
    builder,
    p: NoodleParams,
    drop_z0: float,
    spawn_xy: tuple[float, float],
    instance_ids=None,
):
    rng = np.random.default_rng(0)
    cell_len = p.length / p.n_len
    local_center = np.array([0.5 * p.length, 0.5 * p.cross, 0.5 * p.cross])
    selected = set(range(p.n_noodles) if instance_ids is None else (int(value) for value in instance_ids))
    if any(index < 0 or index >= p.n_noodles for index in selected):
        raise ValueError(f"noodle instance ids outside [0,{p.n_noodles}): {sorted(selected)}")
    starts: list[int] = []
    for k in range(p.n_noodles):
        theta = float(rng.uniform(0.0, math.pi))
        qz = _quat_about_z(theta)
        cx, cy = rng.uniform(-p.spawn_disk_r, p.spawn_disk_r, size=2)
        cz = drop_z0 + k * p.drop_dz
        if k not in selected:
            continue
        center = np.array([spawn_xy[0] + cx, spawn_xy[1] + cy, cz])
        corner = center - _quat_rotate(qz, local_center)
        starts.append(builder.particle_count)
        builder.add_soft_grid(
            pos=wp.vec3(*corner),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), theta),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=p.n_len,
            dim_y=1,
            dim_z=1,
            cell_x=cell_len,
            cell_y=p.cross,
            cell_z=p.cross,
            density=p.density,
            k_mu=p.k_mu,
            k_lambda=p.k_lambda,
            k_damp=p.k_damp,
            particle_radius=0.5 * p.cross,
        )
    return np.array(starts, dtype=np.int32), p.n_len


def _emit_ham(builder, p: HamParams, drop_z0: float, spawn_xy, instance_ids=None):
    rng = np.random.default_rng(2)
    cell = p.size / p.dim
    local_center = np.array([0.5 * p.size, 0.5 * p.size, 0.5 * p.size])
    selected = set(range(p.n_cubes) if instance_ids is None else (int(value) for value in instance_ids))
    if any(index < 0 or index >= p.n_cubes for index in selected):
        raise ValueError(f"ham instance ids outside [0,{p.n_cubes}): {sorted(selected)}")
    tri0 = len(builder.tri_indices)
    starts = []
    for k in range(p.n_cubes):
        cx, cy = rng.uniform(-p.spawn_disk_r, p.spawn_disk_r, size=2)
        cz = drop_z0 + k * p.drop_dz
        q = wp.quat_rpy(*(float(a) for a in rng.uniform(-math.pi, math.pi, size=3)))
        if k not in selected:
            continue
        corner = np.array([spawn_xy[0] + cx, spawn_xy[1] + cy, cz]) - _quat_rotate(
            np.array([q[0], q[1], q[2], q[3]]), local_center
        )
        starts.append(builder.particle_count)
        builder.add_soft_grid(
            pos=wp.vec3(*corner),
            rot=q,
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=p.dim,
            dim_y=p.dim,
            dim_z=p.dim,
            cell_x=cell,
            cell_y=cell,
            cell_z=cell,
            density=p.density,
            k_mu=p.k_mu,
            k_lambda=p.k_lambda,
            k_damp=p.k_damp,
            particle_radius=p.radius_scale * cell,
        )
    return (tri0, len(builder.tri_indices)), np.asarray(starts, dtype=np.int32)


def _emit_strips(builder, p: StripParams, drop_z0: float, spawn_xy, seed, instance_ids=None):
    rng = np.random.default_rng(seed)
    cell_len = p.length / p.n_len
    local_center = np.array([0.5 * p.length, 0.5 * p.cross, 0.5 * p.cross])
    selected = set(range(p.n) if instance_ids is None else (int(value) for value in instance_ids))
    if any(index < 0 or index >= p.n for index in selected):
        raise ValueError(f"strip instance ids outside [0,{p.n}): {sorted(selected)}")
    tri0 = len(builder.tri_indices)
    starts = []
    for k in range(p.n):
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        tilt = float(rng.uniform(-0.5, 0.5))
        q = wp.quat_rpy(tilt, 0.0, theta)
        cx, cy = rng.uniform(-p.spawn_disk_r, p.spawn_disk_r, size=2)
        cz = drop_z0 + k * p.drop_dz
        if k not in selected:
            continue
        corner = np.array([spawn_xy[0] + cx, spawn_xy[1] + cy, cz]) - _quat_rotate(
            np.array([q[0], q[1], q[2], q[3]]), local_center
        )
        starts.append(builder.particle_count)
        builder.add_soft_grid(
            pos=wp.vec3(*corner),
            rot=q,
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=p.n_len,
            dim_y=1,
            dim_z=1,
            cell_x=cell_len,
            cell_y=p.cross,
            cell_z=p.cross,
            density=p.density,
            k_mu=p.k_mu,
            k_lambda=p.k_lambda,
            k_damp=p.k_damp,
            particle_radius=0.5 * p.cross,
        )
    return (tri0, len(builder.tri_indices)), np.asarray(starts, dtype=np.int32)


# ----------------------------------------------------------------------------
# Food sensor (median centroid + low-passed velocity).
# ----------------------------------------------------------------------------
class _NoodleSensor:
    def __init__(self):
        self.centroid = None
        self.vel = np.zeros(3)
        self.bottom = 0.0
        self.top = 0.0

    def update(self, particle_q, dt: float, alpha: float = 0.12):
        pos = particle_q.numpy()
        c = np.median(pos, axis=0)
        if self.centroid is None:
            self.centroid = c.copy()
        v = (c - self.centroid) / dt if dt > 0 else np.zeros(3)
        self.vel = (1.0 - alpha) * self.vel + alpha * v
        self.centroid = c.copy()
        self.bottom = float(np.percentile(pos[:, 2], 3.0))
        self.top = float(np.percentile(pos[:, 2], 97.0))

    def speed(self):
        return float(np.linalg.norm(self.vel))


# ----------------------------------------------------------------------------
# Wok pose driver (geometry helper) + state-machine controller.
# ----------------------------------------------------------------------------
class _WokDriver:
    def __init__(self, wok: WokGeometry):
        self.wok = wok
        self.home_pan_pos = np.array([wok.home_x, wok.home_y, wok.home_z], dtype=float)
        # The "bowl" frame the controller phases relative to is the
        # cooking-surface centre (one ``wok.depth`` below the body origin).
        self.bowl_center_home = self.home_pan_pos.copy()
        self.bowl_center_home[2] -= wok.depth

    def target_for_pose(self, center, pitch, roll=0.0):
        q = _quat_pitch_roll(pitch, roll)
        offset_body = np.array([0.0, 0.0, self.wok.depth])
        pan_world = np.asarray(center, dtype=float) + _quat_rotate(q, offset_body)
        return pan_world, q


class _Controller:
    def __init__(self, driver: _WokDriver, knobs: Controller, depth: float):
        self.driver = driver
        self.k = knobs
        self.depth = depth
        self.state = "settle"
        self.t_state = 0.0
        self.floor_z = driver.bowl_center_home[2]
        self.center_xy = driver.bowl_center_home[:2].copy()
        self.pitch = 0.0
        self.roll = 0.0
        self.cycles = 0

    def _toss_pose(self, p):
        k = self.k
        if p < 0.40:
            s = _smoothstep(p / 0.40)
            dx = -k.toss_fwd * s
            dz = k.gather_lift - k.toss_dip * s
            pitch = -k.gather_pitch
        elif p < 0.70:
            s = _smoothstep((p - 0.40) / 0.30)
            dx = -k.toss_fwd + (k.toss_fwd + k.toss_back) * s
            dz = (k.gather_lift - k.toss_dip) + (k.toss_dip - k.gather_lift + k.toss_lift) * s
            pitch = -k.gather_pitch + (k.gather_pitch + k.toss_pitch) * s
        else:
            s = _smoothstep((p - 0.70) / 0.30)
            dx = k.toss_back * (1.0 - s)
            dz = k.toss_lift * (1.0 - s)
            pitch = k.toss_pitch * (1.0 - s)
        return dx, dz, pitch

    def update(self, food, dt):
        drv = self.driver
        k = self.k
        self.t_state += dt
        pos, quat = drv.target_for_pose(drv.bowl_center_home, 0.0)

        if self.state == "settle":
            if self.t_state >= k.settle_time:
                self._enter("gather")

        elif self.state == "gather":
            self.pitch += (-k.gather_pitch - self.pitch) * 0.06
            self.roll += (0.0 - self.roll) * 0.1
            self.center_xy += (drv.bowl_center_home[:2] - self.center_xy) * 0.1
            target_floor = drv.bowl_center_home[2] + k.gather_lift
            self.floor_z += (target_floor - self.floor_z) * 0.1
            center = np.array([self.center_xy[0], self.center_xy[1], self.floor_z])
            pos, quat = drv.target_for_pose(center, self.pitch, self.roll)
            off_x = (food.centroid[0] - drv.bowl_center_home[0]) if food.centroid is not None else 0.0
            gathered = off_x < -k.gather_edge and food.speed() < k.gather_settle and self.t_state > 0.25
            if gathered or self.t_state > k.gather_t:
                self._enter("toss")

        elif self.state == "toss":
            p = self.t_state / k.toss_t
            dx, dz, pitch = self._toss_pose(p)
            center = drv.bowl_center_home + np.array([dx, 0.0, dz])
            self.center_xy = center[:2].copy()
            self.pitch = pitch
            self.floor_z = float(center[2])
            pos, quat = drv.target_for_pose(center, pitch)
            if p >= 1.0:
                self._enter("catch")

        elif self.state == "catch":
            airborne = self._catch(food, dt)
            if food.centroid is None:
                off = np.zeros(2)
            else:
                off = food.centroid[:2] - self.center_xy
            if airborne:
                pitch_t = roll_t = 0.0
            else:
                pitch_t = float(np.clip(-k.catch_tilt * off[0], -k.catch_tilt_max, k.catch_tilt_max))
                roll_t = float(np.clip(k.catch_tilt * off[1], -k.catch_tilt_max, k.catch_tilt_max))
            self.pitch += (pitch_t - self.pitch) * k.catch_level
            self.roll += (roll_t - self.roll) * k.catch_level
            center = np.array([self.center_xy[0], self.center_xy[1], self.floor_z])
            pos, quat = drv.target_for_pose(center, self.pitch, self.roll)
            if food.centroid is None:
                centered = False
            else:
                centered = float(np.linalg.norm(food.centroid[:2] - drv.bowl_center_home[:2])) < k.centered_radius
            settled = (
                food.top < drv.bowl_center_home[2] + self.depth + 0.005
                and food.speed() < k.settled_speed
                and centered
                and self.t_state > 0.40
            )
            if settled or self.t_state > k.catch_timeout:
                self.cycles += 1
                self._enter("dwell")

        elif self.state == "dwell":
            home_xy = drv.bowl_center_home[:2]
            self.center_xy += (home_xy - self.center_xy) * 0.1
            self.floor_z += (drv.bowl_center_home[2] - self.floor_z) * 0.1
            self.pitch += (0.0 - self.pitch) * 0.1
            self.roll += (0.0 - self.roll) * 0.1
            center = np.array([self.center_xy[0], self.center_xy[1], self.floor_z])
            pos, quat = drv.target_for_pose(center, self.pitch, self.roll)
            if self.t_state >= k.dwell_t:
                self._enter("gather")

        return pos, quat

    def _catch(self, food, dt):
        drv = self.driver
        k = self.k
        if food.centroid is None:
            return False
        c = food.centroid
        v = food.vel
        home_xy = drv.bowl_center_home[:2]
        floor_w = self.floor_z
        airborne = food.bottom > floor_w + 0.5 * self.depth
        if airborne:
            drop = max(food.bottom - (floor_w + self.depth), 0.0)
            disc = v[2] * v[2] + 2.0 * G_ACC * drop
            t_fall = (v[2] + np.sqrt(disc)) / G_ACC if disc > 0 else 0.0
            target_xy = c[:2] + v[:2] * (k.catch_lead * t_fall)
            off = target_xy - home_xy
            r = float(np.linalg.norm(off))
            if r > k.catch_reach:
                target_xy = home_xy + off * (k.catch_reach / r)
            goal_xy = target_xy
            vmax = k.catch_vmax
        else:
            goal_xy = home_xy
            vmax = k.catch_recenter_vmax
        step = goal_xy - self.center_xy
        max_step = vmax * dt
        n = float(np.linalg.norm(step))
        if n > max_step:
            step *= max_step / n
        self.center_xy += step
        self.floor_z += (drv.bowl_center_home[2] - self.floor_z) * k.catch_return
        return airborne

    def _enter(self, state):
        self.state = state
        self.t_state = 0.0


# ----------------------------------------------------------------------------
# Plugin entry point — bound by engine.newton.plugin._load_plugin.
# ----------------------------------------------------------------------------
class ScenePlugin:
    def __init__(self):
        self.cfg = CONFIG  # one source of truth
        self._scene_overrides_applied = False
        self._usd_wok_fit: dict[str, Any] | None = None
        self._usd_food_object_paths: tuple[str, ...] = ()
        self._usd_noodle_sources: list[dict[str, Any]] = []
        self._usd_side_dish_sources: list[dict[str, Any]] = []
        self.usd_deformable_bindings: list[dict[str, Any]] = []
        # Resolved wok mode (set in on_build): dynamic wins over auto_flip.
        self.wok_dynamic: bool = bool(CONFIG.wok.wok_dynamic)
        self.auto_flip: bool = bool(CONFIG.auto_flip) and not self.wok_dynamic
        self.pan_id: int = -1
        self.collider_shape: int = -1
        self.visual_shape: int = -1
        self.imported_pot_shapes: list[int] = []
        self.imported_handle_shapes: list[int] = []
        self._food_spawn_xy: tuple[float, float] | None = None
        self._food_rim_z: float | None = None
        # Grasp interface (see GraspGeometry): the square grip section on the
        # handle, and the flush pad boxes on the gripper's support links.
        # Both stay -1 / empty when grasp.enable is False or the robot has no
        # matching links, and _add_grasp_pairs then no-ops.
        self.grip_shape: int = -1
        # Round-handle mode only: the two axial stops bounding the grip section.
        self.knob_shape: int = -1
        self.collar_shape: int = -1
        self.pad_shapes: list = []
        self.stove_shapes: list[int] = []
        self.driver: _WokDriver | None = None
        self.ctrl: _Controller | None = None
        self.food: _NoodleSensor | None = None
        # AUTO-FLIP path: joint_q / joint_qd slots for the wok body's
        # auto-created FREE joint.  on_post_step writes the controller pose
        # there each frame so MJW integrates the wok along the flip path.
        self._wok_q_start: int = -1
        self._wok_qd_start: int = -1

        # Particle-emit bookkeeping.
        self.starts_np: np.ndarray | None = None
        self.n_len_noodle: int = 0
        self.ham_tri_range = (0, 0)
        self.pepper_pstart = 0
        self.carrot_pstart = 0
        self.pepper_render_count = int(CONFIG.pepper.n)
        self.carrot_render_count = int(CONFIG.carrot.n)
        self.food_instances: list[dict] = []
        self._food_state_pub: _ChefFoodStatePublisher | None = None
        self._startup_hand_pub: _StartupHandCommandPublisher | None = None

        # Render-side state (built lazily on first on_render call).
        self.tube_groups: list[dict] = []
        self.ham_tri_idx: wp.array | None = None
        self._render_ready = False
        self._native_render_hidden = False

    # Sugar so call sites read like ``self.noodle.k_mu``.
    @property
    def noodle(self):
        return self.cfg.noodle

    @property
    def ham(self):
        return self.cfg.ham

    @property
    def pepper(self):
        return self.cfg.pepper

    @property
    def carrot(self):
        return self.cfg.carrot

    @property
    def wok(self):
        return self.cfg.wok

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------
    def _apply_scene_overrides(self, ctx):
        """Let the scene yaml override this plugin's dataclass knobs.

        The Wuji scene selects round-handle grasp geometry for its five-finger
        hand rather than the plugin's procedural parallel-jaw defaults. The
        scene YAML supplies these task-specific settings:

            newton:
              chef:
                grasp:
                  enable: false      # round graspable handle, for a hand
                  condim: 4

        Only keys present in the yaml are touched, and only names that already
        exist on the dataclass, so a typo is reported rather than silently
        ignored.  Extend ``_OVERRIDABLE`` when another sub-config needs it.
        """
        if self._scene_overrides_applied:
            return
        self._scene_overrides_applied = True

        _SECTIONS = {
            "grasp": self.cfg.grasp,
            "wok": self.cfg.wok,
            "usd_objects": self.cfg.usd_objects,
            "usd_noodles": self.cfg.usd_noodles,
            "usd_side_dishes": self.cfg.usd_side_dishes,
            "noodle": self.cfg.noodle,
            "stove": self.cfg.stove,
            "food_state": self.cfg.food_state,
            "startup_hand_motion": self.cfg.startup_hand_motion,
        }

        def _set(target, key, val, where):
            if not hasattr(target, key):
                ctx.logger.warn(
                    f"[chef] scene yaml newton.chef.{where}{key} is not a field of "
                    f"{type(target).__name__} — ignored (typo?)"
                )
                return
            old = getattr(target, key)
            setattr(target, key, type(old)(val) if old is not None else val)
            new = getattr(target, key)
            old_log = f"<dict entries={len(old)}>" if isinstance(old, dict) else repr(old)
            new_log = f"<dict entries={len(new)}>" if isinstance(new, dict) else repr(new)
            ctx.logger.info(f"[chef] scene yaml override: {where}{key} {old_log} -> {new_log}")

        try:
            scene_cfg = getattr(ctx, "scene_cfg", None) or {}
            chef = ((scene_cfg.get("newton") or {}).get("chef")) or {}
            if not chef:
                return
            for key, val in chef.items():
                if key in _SECTIONS:
                    # a nested block -> the matching sub-dataclass
                    for k2, v2 in (val or {}).items():
                        _set(_SECTIONS[key], k2, v2, f"{key}.")
                else:
                    # a bare scalar -> a top-level Config field
                    _set(self.cfg, key, val, "")
        except Exception as exc:
            ctx.logger.warn(f"[chef] scene override parse failed: {exc!r}; using plugin defaults")

    def _prepare_usda_noodle_sources(self, food_objects, xforms):
        """Discover and validate every authored noodle before hiding it."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        cfg = self.cfg.usd_noodles
        self._usd_noodle_sources = []
        if not cfg.enabled:
            return

        name_pattern = re.compile(rf"^{re.escape(str(cfg.object_name_prefix))}(\d{{2}})$")
        indexed_objects = []
        for object_prim in food_objects:
            match = name_pattern.fullmatch(object_prim.GetName())
            if match:
                indexed_objects.append((int(match.group(1)), object_prim))
        indexed_objects.sort(key=lambda item: item[0])
        actual_ids = [instance_id for instance_id, _ in indexed_objects]
        expected_ids = list(range(int(cfg.expected_instance_count)))
        if actual_ids != expected_ids:
            raise RuntimeError(f"USDA noodle instance ids are {actual_ids}; expected exactly {expected_ids}")
        if int(cfg.expected_instance_count) != int(self.cfg.noodle.n_noodles):
            raise RuntimeError(
                f"USDA noodle count {cfg.expected_instance_count} does not match "
                f"newton.chef.noodle.n_noodles={self.cfg.noodle.n_noodles}"
            )

        reference_points = None
        reference_counts = None
        reference_indices = None
        for instance_id, object_prim in indexed_objects:
            rigid_prims = [prim for prim in Usd.PrimRange(object_prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            mesh_prims = [prim for prim in Usd.PrimRange(object_prim) if prim.IsA(UsdGeom.Mesh)]
            if len(rigid_prims) != 1 or len(mesh_prims) != 1:
                raise RuntimeError(
                    f"{object_prim.GetPath()} expected one rigid body and one Mesh, "
                    f"found {len(rigid_prims)} and {len(mesh_prims)}"
                )
            mesh_prim = mesh_prims[0]
            mesh = UsdGeom.Mesh(mesh_prim)
            points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
            if len(points) != int(cfg.expected_vertex_count):
                raise RuntimeError(
                    f"USDA noodle {instance_id:02d} has {len(points)} vertices; "
                    f"expected {cfg.expected_vertex_count}"
                )
            if not len(counts) or np.any(counts != 3) or len(indices) != 3 * len(counts):
                raise RuntimeError(f"USDA noodle {instance_id:02d} must be a non-empty triangulated Mesh")
            if np.any(indices < 0) or np.any(indices >= len(points)) or not np.isfinite(points).all():
                raise RuntimeError(f"USDA noodle {instance_id:02d} has invalid indices or non-finite points")
            topology_sha256 = _mesh_topology_sha256(len(points), counts, indices)
            if topology_sha256 != str(cfg.expected_topology_sha256):
                raise RuntimeError(
                    f"USDA noodle {instance_id:02d} topology changed: {topology_sha256}; "
                    f"expected {cfg.expected_topology_sha256}"
                )
            if reference_points is None:
                reference_points = points.copy()
                reference_counts = counts.copy()
                reference_indices = indices.copy()
            elif not (
                np.array_equal(points, reference_points)
                and np.array_equal(counts, reference_counts)
                and np.array_equal(indices, reference_indices)
            ):
                raise RuntimeError(f"USDA noodle {instance_id:02d} local points/topology differ from instance 00")

            bbox_min = points.min(axis=0)
            bbox_size = np.ptp(points, axis=0)
            if int(np.argmax(bbox_size)) != 0 or bbox_size[0] < 20.0 * max(bbox_size[1:]):
                raise RuntimeError(f"USDA noodle {instance_id:02d} is not a local-X slender body: bbox={bbox_size}")

            parent_prim = rigid_prims[0]
            parent_to_world = xforms.GetLocalToWorldTransform(parent_prim)
            mesh_to_parent = xforms.ComputeRelativeTransform(mesh_prim, parent_prim)[0]
            mesh_to_world = xforms.GetLocalToWorldTransform(mesh_prim)
            parent_pose_world = np.asarray(parent_to_world, dtype=np.float64).T
            mesh_pose_parent = np.asarray(mesh_to_parent, dtype=np.float64).T
            mesh_pose_world = np.asarray(mesh_to_world, dtype=np.float64).T
            if not np.allclose(
                parent_pose_world @ mesh_pose_parent,
                mesh_pose_world,
                atol=1.0e-9,
                rtol=0.0,
            ):
                raise RuntimeError(f"USDA noodle {instance_id:02d} parent/Mesh transform chain is inconsistent")
            decomposed = Gf.Transform(mesh_to_world)
            scale = np.asarray(decomposed.GetScale(), dtype=np.float64)
            if not np.allclose(scale, np.ones(3), atol=1.0e-6):
                raise RuntimeError(f"USDA noodle {instance_id:02d} requires unit composed scale, got {scale}")
            rotation = decomposed.GetRotation().GetQuat()
            imaginary = rotation.GetImaginary()
            quat_xyzw = np.asarray(
                [float(imaginary[0]), float(imaginary[1]), float(imaginary[2]), float(rotation.GetReal())],
                dtype=np.float64,
            )
            corner_world = np.asarray(mesh_to_world.Transform(Gf.Vec3d(*bbox_min)), dtype=np.float64)
            world_points = np.asarray(
                [mesh_to_world.Transform(Gf.Vec3d(*point)) for point in points],
                dtype=np.float64,
            )
            predicted = np.asarray(
                [corner_world + _quat_rotate(quat_xyzw, point - bbox_min) for point in points],
                dtype=np.float64,
            )
            transform_error = float(np.max(np.linalg.norm(predicted - world_points, axis=1)))
            if transform_error > 1.0e-6:
                raise RuntimeError(
                    f"USDA noodle {instance_id:02d} transform is not rigid-decomposable; "
                    f"max error={transform_error:g}m"
                )

            mesh_path = str(mesh_prim.GetPath())
            self._usd_noodle_sources.append(
                {
                    "instance_id": instance_id,
                    "instance_prim_path": str(object_prim.GetPath()),
                    "object_prim_path": str(parent_prim.GetPath()),
                    "parent_prim_path": str(parent_prim.GetPath()),
                    "mesh_prim_path": mesh_path,
                    "recording_prim_path": mesh_path,
                    # Gf matrices act on row vectors and keep translation in
                    # the last row.  Dataset poses use the conventional
                    # column-vector form, hence the transpose at this boundary.
                    "parent_pose_world": parent_pose_world.tolist(),
                    "mesh_pose_parent": mesh_pose_parent.tolist(),
                    "points_coordinate_space": "mesh_local",
                    "authored_points_local": points,
                    "authored_points_world": world_points,
                    "proxy_corner_world": corner_world,
                    "proxy_quat_xyzw": quat_xyzw,
                    "proxy_size": bbox_size,
                    "vertex_count": int(len(points)),
                    "topology_sha256": topology_sha256,
                }
            )

    def _prepare_usda_side_dish_sources(self, food_objects, xforms):
        """Validate the scene-YAML classification of all 45 side dishes."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: PLC0415

        cfg = self.cfg.usd_side_dishes
        self._usd_side_dish_sources = []
        if not cfg.enabled:
            return
        profiles = dict(cfg.families or {})
        required_fields = {
            "semantic_kind",
            "physics_profile",
            "proxy_geometry",
            "expected_instance_count",
            "expected_vertex_count",
            "expected_topology_sha256",
            "expected_albedo_texture",
            "proxy_resolution",
        }
        allowed_semantics = {"meat", "pepper", "carrot", "green_garnish"}
        allowed_physics = {"ham", "pepper", "carrot", "pending"}
        allowed_proxies = {"solid_grid", "strip_grid", "ring_sheet"}
        name_pattern = re.compile(r"^(benchmark_side_dish_(\d{3}))(?:_(\d{2}))?$")

        grouped = {}
        for object_prim in food_objects:
            match = name_pattern.fullmatch(object_prim.GetName())
            if match:
                grouped.setdefault(match.group(1), []).append(
                    (None if match.group(3) is None else int(match.group(3)), object_prim)
                )
        if set(grouped) != set(profiles):
            raise RuntimeError(
                "USDA side-dish family keys differ from newton.chef.usd_side_dishes.families: "
                f"stage_only={sorted(set(grouped) - set(profiles))} "
                f"yaml_only={sorted(set(profiles) - set(grouped))}"
            )
        if sum(len(members) for members in grouped.values()) != int(cfg.expected_instance_count):
            raise RuntimeError(
                f"USDA side-dish count is {sum(len(members) for members in grouped.values())}; "
                f"expected {cfg.expected_instance_count}"
            )

        for family in sorted(grouped):
            profile = profiles[family]
            if not isinstance(profile, dict):
                raise RuntimeError(f"USDA side-dish profile {family} must be a mapping")
            missing = required_fields - set(profile)
            unknown = set(profile) - required_fields
            if missing or unknown:
                raise RuntimeError(
                    f"USDA side-dish profile {family} fields invalid: "
                    f"missing={sorted(missing)} unknown={sorted(unknown)}"
                )
            if profile["semantic_kind"] not in allowed_semantics:
                raise RuntimeError(f"USDA side-dish profile {family} has invalid semantic_kind")
            if profile["physics_profile"] not in allowed_physics:
                raise RuntimeError(f"USDA side-dish profile {family} has invalid physics_profile")
            if profile["proxy_geometry"] not in allowed_proxies:
                raise RuntimeError(f"USDA side-dish profile {family} has invalid proxy_geometry")
            resolution = tuple(int(value) for value in profile["proxy_resolution"])
            if len(resolution) != 3 or any(value <= 0 for value in resolution):
                raise RuntimeError(f"USDA side-dish profile {family} has invalid proxy_resolution {resolution}")

            members = sorted(grouped[family], key=lambda item: -1 if item[0] is None else item[0])
            expected_count = int(profile["expected_instance_count"])
            if len(members) != expected_count:
                raise RuntimeError(
                    f"USDA side-dish family {family} has {len(members)} instances; " f"expected {expected_count}"
                )
            actual_suffixes = [suffix for suffix, _ in members]
            expected_suffixes = [None] if expected_count == 1 else list(range(expected_count))
            if actual_suffixes != expected_suffixes:
                raise RuntimeError(
                    f"USDA side-dish family {family} suffixes are {actual_suffixes}; " f"expected {expected_suffixes}"
                )

            reference_points = None
            reference_counts = None
            reference_indices = None
            for instance_id, (_, object_prim) in enumerate(members):
                rigid_prims = [prim for prim in Usd.PrimRange(object_prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
                mesh_prims = [prim for prim in Usd.PrimRange(object_prim) if prim.IsA(UsdGeom.Mesh)]
                if len(rigid_prims) != 1 or len(mesh_prims) != 1:
                    raise RuntimeError(
                        f"{object_prim.GetPath()} expected one rigid body and one Mesh, "
                        f"found {len(rigid_prims)} and {len(mesh_prims)}"
                    )
                mesh_prim = mesh_prims[0]
                mesh = UsdGeom.Mesh(mesh_prim)
                points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
                if len(points) != int(profile["expected_vertex_count"]):
                    raise RuntimeError(
                        f"USDA side dish {object_prim.GetPath()} has {len(points)} vertices; "
                        f"expected {profile['expected_vertex_count']}"
                    )
                if not len(counts) or np.any(counts != 3) or len(indices) != 3 * len(counts):
                    raise RuntimeError(f"USDA side dish {object_prim.GetPath()} is not triangulated")
                if np.any(indices < 0) or np.any(indices >= len(points)) or not np.isfinite(points).all():
                    raise RuntimeError(f"USDA side dish {object_prim.GetPath()} has invalid indices or points")
                topology_sha256 = _mesh_topology_sha256(len(points), counts, indices)
                if topology_sha256 != str(profile["expected_topology_sha256"]):
                    raise RuntimeError(
                        f"USDA side dish {object_prim.GetPath()} topology changed: "
                        f"{topology_sha256}; expected {profile['expected_topology_sha256']}"
                    )
                if reference_points is None:
                    reference_points = points.copy()
                    reference_counts = counts.copy()
                    reference_indices = indices.copy()
                elif not (
                    np.array_equal(points, reference_points)
                    and np.array_equal(counts, reference_counts)
                    and np.array_equal(indices, reference_indices)
                ):
                    raise RuntimeError(f"USDA side-dish family {family} has inconsistent local points/topology")

                material = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()[0]
                texture_names = set()
                if material:
                    for prim in Usd.PrimRange(material.GetPrim()):
                        if not prim.IsA(UsdShade.Shader):
                            continue
                        for shader_input in UsdShade.Shader(prim).GetInputs():
                            value = shader_input.Get()
                            if isinstance(value, Sdf.AssetPath) and value.path:
                                texture_names.add(value.path.rsplit("/", 1)[-1])
                expected_albedo = str(profile["expected_albedo_texture"])
                if expected_albedo not in texture_names:
                    raise RuntimeError(
                        f"USDA side dish {object_prim.GetPath()} does not bind expected albedo "
                        f"{expected_albedo!r}; textures={sorted(texture_names)}"
                    )

                parent_prim = rigid_prims[0]
                parent_to_world = xforms.GetLocalToWorldTransform(parent_prim)
                mesh_to_parent = xforms.ComputeRelativeTransform(mesh_prim, parent_prim)[0]
                mesh_path = str(mesh_prim.GetPath())
                mesh_to_world = xforms.GetLocalToWorldTransform(mesh_prim)
                parent_pose_world = np.asarray(parent_to_world, dtype=np.float64).T
                mesh_pose_parent = np.asarray(mesh_to_parent, dtype=np.float64).T
                mesh_pose_world = np.asarray(mesh_to_world, dtype=np.float64).T
                if not np.allclose(
                    parent_pose_world @ mesh_pose_parent,
                    mesh_pose_world,
                    atol=1.0e-9,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        f"USDA side dish {object_prim.GetPath()} parent/Mesh transform chain is inconsistent"
                    )
                decomposed = Gf.Transform(mesh_to_world)
                scale = np.asarray(decomposed.GetScale(), dtype=np.float64)
                if not np.allclose(scale, np.ones(3), atol=1.0e-6):
                    raise RuntimeError(
                        f"USDA side dish {object_prim.GetPath()} requires unit composed scale, got {scale}"
                    )
                rotation = decomposed.GetRotation().GetQuat()
                imaginary = rotation.GetImaginary()
                quat_xyzw = np.asarray(
                    [
                        float(imaginary[0]),
                        float(imaginary[1]),
                        float(imaginary[2]),
                        float(rotation.GetReal()),
                    ],
                    dtype=np.float64,
                )
                bbox_min = points.min(axis=0)
                bbox_size = np.ptp(points, axis=0)
                corner_world = np.asarray(mesh_to_world.Transform(Gf.Vec3d(*bbox_min)), dtype=np.float64)
                world_points = np.asarray(
                    [mesh_to_world.Transform(Gf.Vec3d(*point)) for point in points],
                    dtype=np.float64,
                )
                predicted = np.asarray(
                    [corner_world + _quat_rotate(quat_xyzw, point - bbox_min) for point in points],
                    dtype=np.float64,
                )
                transform_error = float(np.max(np.linalg.norm(predicted - world_points, axis=1)))
                if transform_error > 1.0e-6:
                    raise RuntimeError(
                        f"USDA side dish {object_prim.GetPath()} transform is not rigid-decomposable; "
                        f"max error={transform_error:g}m"
                    )
                self._usd_side_dish_sources.append(
                    {
                        "family": family,
                        "family_instance_id": instance_id,
                        "semantic_kind": str(profile["semantic_kind"]),
                        "physics_profile": str(profile["physics_profile"]),
                        "proxy_geometry": str(profile["proxy_geometry"]),
                        "proxy_resolution": resolution,
                        "instance_prim_path": str(object_prim.GetPath()),
                        "object_prim_path": str(parent_prim.GetPath()),
                        "parent_prim_path": str(parent_prim.GetPath()),
                        "mesh_prim_path": mesh_path,
                        "recording_prim_path": mesh_path,
                        "parent_pose_world": parent_pose_world.tolist(),
                        "mesh_pose_parent": mesh_pose_parent.tolist(),
                        "points_coordinate_space": "mesh_local",
                        "authored_points_local": points,
                        "authored_points_world": world_points,
                        "face_vertex_indices": indices,
                        "proxy_corner_world": corner_world,
                        "proxy_quat_xyzw": quat_xyzw,
                        "proxy_bbox_min": bbox_min,
                        "proxy_size": bbox_size,
                        "vertex_count": int(len(points)),
                        "topology_sha256": topology_sha256,
                    }
                )

        semantic_next = {}
        for source in self._usd_side_dish_sources:
            semantic_kind = source["semantic_kind"]
            source["instance_id"] = semantic_next.get(semantic_kind, 0)
            semantic_next[semantic_kind] = source["instance_id"] + 1

        migrated_families = tuple(str(family) for family in cfg.migrated_families)
        if len(set(migrated_families)) != len(migrated_families) or not set(migrated_families) <= set(profiles):
            raise RuntimeError(
                "newton.chef.usd_side_dishes.migrated_families contains "
                f"duplicates or unknown families: {migrated_families}"
            )
        migrated_sources = [
            source for source in self._usd_side_dish_sources if source["family"] in set(migrated_families)
        ]
        if any(source["physics_profile"] == "pending" for source in migrated_sources):
            raise RuntimeError("USDA side-dish binding selected a pending physics profile")

    def on_stage(self, stage, ctx):
        """Prepare the USDA scene in an anonymous session layer.

        Migrated soft proxies replace selected imported rigid food. All exact
        prim paths and authored geometry are captured first for recording
        identity; visual selection is authored directly in the source USDA.
        """
        self._apply_scene_overrides(ctx)
        usd_cfg = self.cfg.usd_objects
        if not usd_cfg.enabled:
            return
        if not usd_cfg.procedural_food_transition:
            raise RuntimeError(
                "newton.chef.usd_objects currently requires "
                "procedural_food_transition=true until the USDA soft-body compiler lands"
            )

        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        root = stage.GetPrimAtPath(usd_cfg.objects_root)
        if not root or not root.IsValid():
            raise RuntimeError(f"fried-noodles stage is missing {usd_cfg.objects_root}")
        pot_body = stage.GetPrimAtPath(usd_cfg.pot_body_prim_path)
        pot_mesh_prim = stage.GetPrimAtPath(usd_cfg.pot_mesh_prim_path)
        if not pot_body or not pot_body.IsValid() or not pot_mesh_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(
                f"fried-noodles pot binding is invalid: body={usd_cfg.pot_body_prim_path!r} "
                f"mesh={usd_cfg.pot_mesh_prim_path!r}"
            )

        mass = UsdPhysics.MassAPI(pot_body).GetMassAttr().Get()
        if mass is None or abs(float(mass) - float(usd_cfg.expected_pot_mass)) > 1.0e-6:
            raise RuntimeError(f"composed USDA pot mass is {mass!r} kg; expected {usd_cfg.expected_pot_mass:g} kg")

        mesh = UsdGeom.Mesh(pot_mesh_prim)
        points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        if not len(points) or not len(counts) or np.any(counts != 3):
            raise RuntimeError("fried-noodles pot must be a non-empty triangulated Mesh")

        xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        mesh_to_body = xforms.ComputeRelativeTransform(pot_mesh_prim, pot_body)[0]
        points_body = np.asarray(
            [mesh_to_body.Transform(Gf.Vec3d(*point)) for point in points],
            dtype=np.float64,
        )
        body_to_world = xforms.GetLocalToWorldTransform(pot_body)
        body_axes_world = np.asarray(
            [body_to_world.TransformDir(Gf.Vec3d(*axis)) for axis in np.eye(3)],
            dtype=np.float64,
        )
        self._usd_wok_fit = _fit_usda_wok_bowl(points_body, indices, body_axes_world)

        food_objects = []
        for object_prim in root.GetChildren():
            name = object_prim.GetName()
            if name.startswith("benchmark_noodles_") or name.startswith("benchmark_side_dish_"):
                food_objects.append(object_prim)
        if len(food_objects) != int(usd_cfg.expected_food_count):
            raise RuntimeError(
                f"fried-noodles food count is {len(food_objects)}; " f"expected {usd_cfg.expected_food_count}"
            )
        self._usd_food_object_paths = tuple(str(prim.GetPath()) for prim in food_objects)

        self._prepare_usda_noodle_sources(food_objects, xforms)
        self._prepare_usda_side_dish_sources(food_objects, xforms)

        with Usd.EditContext(stage, stage.GetSessionLayer()):
            # USDA owns only the visible pot surface. Newton-specific rigid and
            # particle collision proxies are added to the imported body later.
            UsdPhysics.CollisionAPI(pot_mesh_prim).CreateCollisionEnabledAttr(False)
            for object_prim in food_objects:
                rigid_prims = [prim for prim in Usd.PrimRange(object_prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
                mesh_prims = [prim for prim in Usd.PrimRange(object_prim) if prim.IsA(UsdGeom.Mesh)]
                if len(rigid_prims) != 1 or len(mesh_prims) != 1:
                    raise RuntimeError(
                        f"{object_prim.GetPath()} expected one rigid body and one Mesh, "
                        f"found {len(rigid_prims)} and {len(mesh_prims)}"
                    )
                UsdPhysics.RigidBodyAPI(rigid_prims[0]).CreateRigidBodyEnabledAttr(False)
                UsdPhysics.CollisionAPI(mesh_prims[0]).CreateCollisionEnabledAttr(False)
                UsdGeom.Imageable(mesh_prims[0]).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

        fit = self._usd_wok_fit
        ctx.logger.info(
            f"[chef] USDA stage overlay: pot={usd_cfg.pot_body_prim_path} mass={float(mass):g}kg "
            f"food_rigid_disabled={len(food_objects)} food_hidden={len(food_objects)} "
            f"bowl_r={fit['rim_radius']:.6f} depth={fit['depth']:.6f} "
            f"components={fit['component_count']}"
        )
        if self._usd_noodle_sources:
            first = self._usd_noodle_sources[0]
            ctx.logger.info(
                f"[chef] USDA noodle sources: count={len(self._usd_noodle_sources)} "
                f"first={first['mesh_prim_path']} vertices_each={first['vertex_count']} "
                f"topology={first['topology_sha256'][:12]} bbox={first['proxy_size'].tolist()}"
            )
        if self._usd_side_dish_sources:
            counts = {}
            for source in self._usd_side_dish_sources:
                counts[source["semantic_kind"]] = counts.get(source["semantic_kind"], 0) + 1
            ctx.logger.info(
                f"[chef] USDA side-dish classification: count={len(self._usd_side_dish_sources)} "
                f"semantic_counts={dict(sorted(counts.items()))}"
            )

    @staticmethod
    def _shape_vertices_in_body(builder, shape_index):
        source = builder.shape_source[shape_index]
        vertices = np.asarray(getattr(source, "vertices", []), dtype=np.float64)
        if not len(vertices):
            return vertices.reshape(0, 3)
        vertices = vertices * np.asarray(builder.shape_scale[shape_index], dtype=np.float64)
        shape_xform = builder.shape_transform[shape_index]
        quat = np.asarray(
            [shape_xform.q[0], shape_xform.q[1], shape_xform.q[2], shape_xform.q[3]],
            dtype=np.float64,
        )
        pos = np.asarray([shape_xform.p[0], shape_xform.p[1], shape_xform.p[2]], dtype=np.float64)
        return np.asarray([_quat_rotate(quat, vertex) + pos for vertex in vertices])

    @staticmethod
    def _set_shape_custom_value(builder, key, shape_index, value):
        attribute = getattr(builder, "custom_attributes", {}).get(key)
        if attribute is None:
            return False
        if attribute.values is None:
            attribute.values = {}
        if not isinstance(attribute.values, dict):
            raise TypeError(f"shape custom attribute {key!r} does not use sparse per-shape storage")
        attribute.values[int(shape_index)] = value
        return True

    @staticmethod
    def _disable_usda_background_particle_collisions(builder, ctx):
        """Keep authored background rigid contact without testing food particles.

        Newton assigns ``COLLIDE_PARTICLES`` to imported collision meshes by
        default.  The fried-noodles background contains convex-decomposed table
        and stove meshes, so leaving that bit set puts hundreds of distant
        convex pieces into every XPBD particle-collision pass.  Robot and wok
        contact still needs those shapes, therefore preserve ``VISIBLE`` and
        ``COLLIDE_SHAPES`` and clear only the particle-collision bit.
        """
        import newton  # noqa: PLC0415

        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        changed = 0
        for shape_index, label in enumerate(list(getattr(builder, "shape_label", []) or [])):
            if not str(label).startswith("/World/background/"):
                continue
            if int(builder.shape_flags[shape_index]) & collide_particles:
                builder.shape_flags[shape_index] &= ~collide_particles
                changed += 1
        ctx.logger.info(
            f"[chef] USDA background particle collision disabled: shapes={changed} "
            f"(rigid collision and visibility preserved)"
        )
        return changed

    @staticmethod
    def _disable_usda_stove_rigid_collisions(builder, ctx):
        """Disable only the imported gas-stove grater collision pieces.

        The authored stove keeps its visual mesh in USDA, but its four
        optimized ``collider_*``/``visual_convex_*`` pieces sit inside the
        fitted Newton wok shell.  Keeping those pieces as rigid colliders
        makes the dynamic wok start in penetration and lets contact impulses
        throw it off the stove when the arm moves.  Other background rigid
        surfaces (table, floor and walls) remain untouched.
        """
        import newton  # noqa: PLC0415

        collision = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        stove_prefix = "/World/background/benchmark_gas_stove_001/"
        body_prefix = stove_prefix + "entity/body/"
        changed = 0
        for shape_index, label in enumerate(list(getattr(builder, "shape_label", []) or [])):
            name = str(label)
            if not name.startswith(body_prefix):
                continue
            leaf = name.rsplit("/", 1)[-1]
            if not (leaf.startswith("collider_") or leaf.startswith("visual_convex_")):
                continue
            if int(builder.shape_flags[shape_index]) & collision:
                builder.shape_flags[shape_index] &= ~collision
                changed += 1
        ctx.logger.info(
            f"[chef] USDA gas-stove grater rigid collision disabled: shapes={changed} " "(USDA visual mesh preserved)"
        )
        return changed

    def _add_usda_stove_support(self, builder, ctx):
        """Add hidden stove supports fitted to the current USDA geometry.

        The imported gas-stove ``collider_*`` meshes are convex grater pieces
        whose broad hulls overlap the curved wok when used verbatim.  We keep
        each piece's authored XY footprint and world placement, but cap its
        hidden Newton proxy immediately below the fitted wok shell.  Thus the
        support follows the actual stove mesh instead of reintroducing the
        old fixed procedural ring; the USDA stove remains the sole render
        asset.
        """
        import newton  # noqa: PLC0415

        st = self.cfg.stove
        if not st.enable:
            self.stove_shapes = []
            return []
        if self.pan_id < 0 or self._usd_wok_fit is None:
            self.stove_shapes = []
            return []

        fit = self._usd_wok_fit
        body_pose = builder.body_q[self.pan_id]
        body_pos = np.asarray([body_pose[0], body_pose[1], body_pose[2]], dtype=np.float64)
        body_quat = np.asarray(
            [body_pose[3], body_pose[4], body_pose[5], body_pose[6]],
            dtype=np.float64,
        )
        rim_center = body_pos + _quat_rotate(body_quat, np.asarray(fit["center"]))
        up = _quat_rotate(body_quat, np.asarray(fit["up"], dtype=np.float64))
        up /= np.linalg.norm(up)
        radial_0 = _quat_rotate(body_quat, np.asarray(fit["radial_0"], dtype=np.float64))
        radial_0 -= up * float(np.dot(radial_0, up))
        radial_0 /= np.linalg.norm(radial_0)
        radial_1 = np.cross(up, radial_0)
        radial_1 /= np.linalg.norm(radial_1)

        cfg_static = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=st.mu,
            has_shape_collision=True,
            has_particle_collision=False,
            gap=st.gap,
        )
        visible = int(newton.ShapeFlags.VISIBLE)
        shapes = []

        # The thin proxy is generated in the imported wok body's frame.  Its
        # world vertices give a measured lower surface for every stove pad,
        # including any authored rotation/translation in the USDA stage.
        rigid_vertices, _ = _make_wok(
            rim_r=float(self.cfg.wok.rim_r),
            depth=float(self.cfg.wok.depth),
            thickness=0.003,
            n_az=48,
            n_radial=16,
        )
        proxy_vertices = _transform_wok_proxy_points(
            rigid_vertices, np.asarray(fit["center"]), _usda_wok_proxy_frame(fit)[1]
        )
        proxy_world = np.asarray(
            [_quat_rotate(body_quat, point) + body_pos for point in proxy_vertices],
            dtype=np.float64,
        )

        body_labels = list(getattr(builder, "body_label", []) or [])
        shape_labels = list(getattr(builder, "shape_label", []) or [])
        shape_bodies = list(getattr(builder, "shape_body", []) or [])
        stove_prefix = "/World/background/benchmark_gas_stove_001/entity/body/"

        def shape_world_vertices(shape_index):
            vertices = self._shape_vertices_in_body(builder, shape_index)
            if not len(vertices):
                return vertices
            body_index = int(shape_bodies[shape_index]) if shape_index < len(shape_bodies) else -1
            if body_index < 0 or body_index >= len(body_labels):
                return vertices
            body_pose = builder.body_q[body_index]
            pose_pos = np.asarray([body_pose[0], body_pose[1], body_pose[2]], dtype=np.float64)
            pose_quat = np.asarray([body_pose[3], body_pose[4], body_pose[5], body_pose[6]], dtype=np.float64)
            return np.asarray(
                [_quat_rotate(pose_quat, point) + pose_pos for point in vertices],
                dtype=np.float64,
            )

        stove_shape_indices = []
        for index, label in enumerate(shape_labels):
            name = str(label)
            if not name.startswith(stove_prefix):
                continue
            leaf = name.rsplit("/", 1)[-1]
            if leaf.startswith("collider_") or leaf.startswith("visual_convex_"):
                stove_shape_indices.append(index)

        # Preserve the authored support footprints.  A 1 mm inward margin
        # prevents the replacement box from recreating the convex hull's
        # corner overlap while remaining visually co-located with the burner.
        for ordinal, source_index in enumerate(stove_shape_indices):
            vertices = shape_world_vertices(source_index)
            if len(vertices) < 4:
                continue
            bounds_min = vertices.min(axis=0)
            bounds_max = vertices.max(axis=0)
            xy_min = bounds_min[:2] + 0.001
            xy_max = bounds_max[:2] - 0.001
            if np.any(xy_max <= xy_min):
                xy_min, xy_max = bounds_min[:2], bounds_max[:2]

            in_footprint = proxy_world[
                (proxy_world[:, 0] >= xy_min[0])
                & (proxy_world[:, 0] <= xy_max[0])
                & (proxy_world[:, 1] >= xy_min[1])
                & (proxy_world[:, 1] <= xy_max[1])
            ]
            if len(in_footprint):
                support_top = float(np.min(in_footprint[:, 2])) - 0.001
            else:
                distance = np.linalg.norm(proxy_world[:, :2] - vertices[:, :2].mean(axis=0), axis=1)
                support_top = float(np.min(proxy_world[np.argsort(distance)[: min(16, len(proxy_world))], 2])) - 0.001

            support_bottom = float(bounds_min[2])
            if support_top <= support_bottom + 0.001:
                support_bottom = support_top - 0.002
            centre = np.array(
                [
                    0.5 * (xy_min[0] + xy_max[0]),
                    0.5 * (xy_min[1] + xy_max[1]),
                    0.5 * (support_bottom + support_top),
                ],
                dtype=np.float64,
            )
            shape = builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(*centre), wp.quat_identity()),
                hx=0.5 * float(xy_max[0] - xy_min[0]),
                hy=0.5 * float(xy_max[1] - xy_min[1]),
                hz=0.5 * float(support_top - support_bottom),
                cfg=cfg_static,
                label=f"chef_usda_stove_support_{ordinal:02d}",
            )
            builder.shape_flags[shape] &= ~visible
            shapes.append(shape)

        # A source USDA without explicit stove collider pieces still gets a
        # conservative ring fallback, keeping old fixtures usable.
        if not shapes:
            rim_r = float(self.cfg.wok.rim_r)
            depth = float(self.cfg.wok.depth)
            radius = (rim_r * rim_r + depth * depth) / (2.0 * depth)
            center_z = radius - depth
            radius_out = radius + 0.003
            ring_radius = float(min(st.ring_radius, 0.98 * radius_out))
            surface_offset = center_z - math.sqrt(max(radius_out * radius_out - ring_radius * ring_radius, 0.0))
            bead_center = rim_center + up * (surface_offset - float(st.bead_radius))
            count = max(int(st.ring_beads), 3)
            for i in range(count):
                theta = 2.0 * math.pi * i / count
                point = (
                    bead_center + ring_radius * math.cos(theta) * radial_0 + ring_radius * math.sin(theta) * radial_1
                )
                shape = builder.add_shape_sphere(
                    -1,
                    xform=wp.transform(wp.vec3(*point), wp.quat_identity()),
                    radius=float(st.bead_radius),
                    cfg=cfg_static,
                    label=f"chef_usda_stove_support_{i:02d}",
                )
                builder.shape_flags[shape] &= ~visible
                shapes.append(shape)

        self.stove_shapes = shapes
        ctx.logger.info(
            f"[chef] USDA stove support fitted: centre=({rim_center[0]:.6f},"
            f"{rim_center[1]:.6f},{rim_center[2]:.6f}) source_shapes="
            f"{len(stove_shape_indices)} replacement_shapes={len(shapes)}"
        )
        return shapes

    def _disable_usda_food_shapes(self, builder):
        """Validate food identities and disable all imported shapes per object."""
        import newton  # noqa: PLC0415

        usd_cfg = self.cfg.usd_objects
        expected = set(self._usd_food_object_paths)
        if len(expected) != int(usd_cfg.expected_food_count):
            raise RuntimeError("USDA food inventory is unavailable or incomplete; on_stage must run before on_build")

        # One food object can import several shapes: Newton 1.6.1 also loads
        # hidden static visuals such as lowpoly/bbox; material subsets can add
        # further shapes. Validate the exact source-object inventory rather
        # than assuming one shape per object, then disable every descendant.
        root = usd_cfg.objects_root.rstrip("/") + "/"
        imported_food_shapes = []
        imported_objects = set()
        for index, label in enumerate(list(getattr(builder, "shape_label", []) or [])):
            path = str(label)
            if not path.startswith(root):
                continue
            object_name = path[len(root) :].partition("/")[0]
            if object_name.startswith(("benchmark_noodles_", "benchmark_side_dish_")):
                imported_objects.add(root + object_name)
                imported_food_shapes.append(index)

        if imported_objects != expected:
            raise RuntimeError(
                "imported food objects do not match the USDA inventory: "
                f"missing={sorted(expected - imported_objects)} "
                f"unexpected={sorted(imported_objects - expected)}"
            )

        all_shape_flags = int(
            newton.ShapeFlags.VISIBLE | newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES
        )
        for shape_index in imported_food_shapes:
            builder.shape_flags[shape_index] &= ~all_shape_flags

        return imported_food_shapes

    def _bind_usda_wok(self, builder, ctx):
        """Reuse the imported pot body with USDA visual and Newton proxies."""
        import newton  # noqa: PLC0415

        cfg = self.cfg
        usd_cfg = cfg.usd_objects
        fit = self._usd_wok_fit
        if fit is None:
            raise RuntimeError("USDA wok fit is unavailable; on_stage must run before on_build")
        if not cfg.wok.wok_dynamic or cfg.auto_flip:
            raise RuntimeError("the imported USDA wok currently supports dynamic teleoperation only")

        self._disable_usda_background_particle_collisions(builder, ctx)

        labels = list(getattr(builder, "body_label", []) or [])
        matches = [index for index, label in enumerate(labels) if str(label) == usd_cfg.pot_body_prim_path]
        if len(matches) != 1:
            raise RuntimeError(f"expected one imported pot body {usd_cfg.pot_body_prim_path!r}, found {len(matches)}")
        self.pan_id = matches[0]
        self.wok_dynamic = True
        self.auto_flip = False

        imported_food_shapes = self._disable_usda_food_shapes(builder)

        mass = float(usd_cfg.expected_pot_mass)
        builder.body_mass[self.pan_id] = mass
        builder.body_inv_mass[self.pan_id] = 1.0 / mass

        up = np.asarray(fit["up"], dtype=np.float64)
        radial_0 = np.asarray(fit["radial_0"], dtype=np.float64)
        radial_1 = np.asarray(fit["radial_1"], dtype=np.float64)
        principal = np.asarray(
            [cfg.wok.inertia_diag[0], cfg.wok.inertia_diag[1], cfg.wok.inertia_diag[2]],
            dtype=np.float64,
        )
        basis = np.column_stack((radial_0, radial_1, up))
        inertia = basis @ np.diag(principal) @ basis.T
        inverse_inertia = np.linalg.inv(inertia)
        builder.body_inertia[self.pan_id] = wp.mat33(*inertia.reshape(-1))
        builder.body_inv_inertia[self.pan_id] = wp.mat33(*inverse_inertia.reshape(-1))
        center_of_mass = np.asarray(fit["center"]) - 0.5 * float(cfg.wok.depth) * up
        builder.body_com[self.pan_id] = wp.vec3(*center_of_mass)

        visible = int(newton.ShapeFlags.VISIBLE)
        all_collision = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        self.imported_pot_shapes = [int(index) for index in builder.body_shapes[self.pan_id]]
        for shape_index in self.imported_pot_shapes:
            builder.shape_flags[shape_index] = (int(builder.shape_flags[shape_index]) & ~all_collision) | visible

        center, proxy_basis = _usda_wok_proxy_frame(fit)
        proxy_vertices, proxy_triangles = _make_wok(
            rim_r=float(cfg.wok.rim_r),
            depth=float(cfg.wok.depth),
            thickness=float(usd_cfg.particle_proxy_thickness),
            n_az=cfg.wok.az,
            n_radial=cfg.wok.radial,
        )
        proxy_vertices = _transform_wok_proxy_points(proxy_vertices, center, proxy_basis)
        proxy_mesh = newton.Mesh(proxy_vertices, proxy_triangles.flatten(), compute_inertia=False)
        proxy_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=cfg.noodle.wok_mu,
            has_shape_collision=False,
            has_particle_collision=True,
            gap=0.0,
        )
        self.collider_shape = builder.add_shape_mesh(
            self.pan_id,
            xform=wp.transform_identity(),
            mesh=proxy_mesh,
            cfg=proxy_cfg,
            label="chef_usda_wok_particle_proxy",
        )

        rigid_vertices, rigid_triangles = _make_wok(
            rim_r=float(cfg.wok.rim_r),
            depth=float(cfg.wok.depth),
            thickness=0.003,
            n_az=48,
            n_radial=16,
        )
        rigid_vertices = _transform_wok_proxy_points(rigid_vertices, center, proxy_basis)
        rigid_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=cfg.stove.mu,
            has_shape_collision=True,
            has_particle_collision=False,
            gap=cfg.stove.gap,
        )
        self.visual_shape = builder.add_shape_mesh(
            self.pan_id,
            xform=wp.transform_identity(),
            mesh=newton.Mesh(rigid_vertices, rigid_triangles.flatten(), compute_inertia=False),
            cfg=rigid_cfg,
            label="chef_usda_wok_rigid_proxy",
        )
        self._add_usda_wok_handle_colliders(builder, ctx, center, proxy_basis)

        # Physics proxies must never render over the authoritative USDA pot.
        for shape_index in (
            self.collider_shape,
            self.visual_shape,
            self.grip_shape,
            self.collar_shape,
            self.knob_shape,
        ):
            if shape_index >= 0:
                builder.shape_flags[shape_index] &= ~visible

        body_pose = builder.body_q[self.pan_id]
        body_pos = np.asarray([body_pose[0], body_pose[1], body_pose[2]], dtype=np.float64)
        body_quat = np.asarray([body_pose[3], body_pose[4], body_pose[5], body_pose[6]], dtype=np.float64)
        rim_world = body_pos + _quat_rotate(body_quat, np.asarray(fit["center"]))
        self._food_spawn_xy = (float(rim_world[0]), float(rim_world[1]))
        self._food_rim_z = float(rim_world[2])

        ctx.logger.info(
            f"[chef] bound USDA wok body={self.pan_id} mass={mass:g}kg "
            f"inertia_diag={cfg.wok.inertia_diag} visual_shapes={len(self.imported_pot_shapes)} "
            f"food_shapes_disabled={len(imported_food_shapes)} "
            f"particle_proxy={self.collider_shape} thickness={usd_cfg.particle_proxy_thickness:g} "
            f"rigid_proxy={self.visual_shape} handle_proxy={self.grip_shape} "
            f"rim_world=({rim_world[0]:.6f},{rim_world[1]:.6f},{rim_world[2]:.6f})"
        )

    def _add_usda_wok_handle_colliders(self, builder, ctx, center, basis):
        """Add the original Newton round-handle collision set in USDA frame."""
        import newton  # noqa: PLC0415

        g = self.cfg.grasp
        if g.enable:
            raise RuntimeError("USDA Wuji wok requires the round-handle grasp mode")

        def point(value):
            return _transform_wok_proxy_points([value], center, basis)[0]

        grip_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=True,
            has_particle_collision=False,
            gap=0.0,
        )
        handle_radius = float(self._usd_wok_fit.get("handle_radius", 0.0095))
        stick_a = point((-0.2029, 0.0, -0.0226))
        stick_b = point((-0.3139, 0.0, -0.0131))
        xform, half_height = _capsule_fromto(stick_a, stick_b)
        self.grip_shape = builder.add_shape_capsule(
            self.pan_id,
            xform=xform,
            radius=handle_radius,
            half_height=half_height,
            cfg=grip_cfg,
            label="chef_usda_wok_grip_round_proxy",
            custom_attributes={"mujoco:condim": int(g.condim)},
        )
        if g.collar_radius > handle_radius:
            self.collar_shape = builder.add_shape_sphere(
                self.pan_id,
                xform=wp.transform(wp.vec3(*stick_a), wp.quat_identity()),
                radius=g.collar_radius,
                cfg=grip_cfg,
                label="chef_usda_wok_grip_collar_proxy",
                custom_attributes={"mujoco:condim": int(g.condim)},
            )
        knob = point((-0.3227, 0.0, -0.0124))
        self.knob_shape = builder.add_shape_sphere(
            self.pan_id,
            xform=wp.transform(wp.vec3(*knob), wp.quat_identity()),
            radius=0.0124,
            cfg=grip_cfg,
            label="chef_usda_wok_knob_proxy",
            custom_attributes={"mujoco:condim": int(g.condim)},
        )
        ctx.logger.info(
            f"[chef] USDA wok Newton handle proxies: grip={self.grip_shape} "
            f"collar={self.collar_shape} knob={self.knob_shape} "
            f"radius={handle_radius * 1e3:.1f}mm condim={g.condim}"
        )

    def _add_usda_noodle_soft_proxies(self, builder, ctx):
        """Replace all discovered authored noodles with aligned FEM grids."""
        if not self._usd_noodle_sources:
            return []
        p = self.cfg.noodle
        expected_particles = 4 * (int(p.n_len) + 1)
        shared_tet_indices = None
        shared_barycentric_weights = None
        shared_proxy_tets = None
        shared_mapping_sha256 = None
        bindings = []

        for source in self._usd_noodle_sources:
            instance_id = int(source["instance_id"])
            size = np.asarray(source["proxy_size"], dtype=np.float64)
            if np.any(size <= 0.0):
                raise RuntimeError(f"USDA noodle {instance_id:02d} has non-positive proxy size {size}")

            particle_start = int(builder.particle_count)
            tet_start = len(builder.tet_indices)
            builder.add_soft_grid(
                pos=wp.vec3(*source["proxy_corner_world"]),
                rot=wp.quat(*source["proxy_quat_xyzw"]),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=int(p.n_len),
                dim_y=1,
                dim_z=1,
                cell_x=float(size[0]) / int(p.n_len),
                cell_y=float(size[1]),
                cell_z=float(size[2]),
                density=float(p.density),
                k_mu=float(p.k_mu),
                k_lambda=float(p.k_lambda),
                k_damp=float(p.k_damp),
                particle_radius=float(p.self_radius),
                label=str(source["mesh_prim_path"]),
            )
            particle_end = int(builder.particle_count)
            tet_end = len(builder.tet_indices)
            if particle_end - particle_start != expected_particles:
                raise RuntimeError(
                    f"USDA noodle {instance_id:02d} proxy has "
                    f"{particle_end - particle_start} particles; expected {expected_particles}"
                )

            proxy_particles = np.asarray(
                [
                    [float(point[0]), float(point[1]), float(point[2])]
                    for point in builder.particle_q[particle_start:particle_end]
                ],
                dtype=np.float64,
            )
            proxy_tets_global = np.asarray(builder.tet_indices[tet_start:tet_end], dtype=np.int64).reshape(-1, 4)
            proxy_tets_local = proxy_tets_global - particle_start
            if np.any(proxy_tets_local < 0) or np.any(proxy_tets_local >= len(proxy_particles)):
                raise RuntimeError(f"USDA noodle {instance_id:02d} proxy tetrahedra escaped its particle slice")

            if shared_tet_indices is None:
                shared_proxy_tets = proxy_tets_local.copy()
                shared_tet_indices, shared_barycentric_weights = _bind_points_to_tets(
                    source["authored_points_world"],
                    proxy_particles,
                    proxy_tets_local,
                )
                shared_mapping_sha256 = _tet_binding_sha256(shared_tet_indices, shared_barycentric_weights)
            elif not np.array_equal(proxy_tets_local, shared_proxy_tets):
                raise RuntimeError(f"USDA noodle {instance_id:02d} proxy topology differs from instance 00")

            reconstructed = _reconstruct_tet_bound_points(
                proxy_particles,
                shared_tet_indices,
                shared_barycentric_weights,
            )
            reconstruction_error = float(
                np.max(np.linalg.norm(reconstructed - source["authored_points_world"], axis=1))
            )
            if reconstruction_error > 1.0e-6:
                raise RuntimeError(
                    f"USDA noodle {instance_id:02d} rest reconstruction error "
                    f"{reconstruction_error:g}m exceeds 1e-6m"
                )

            binding = {
                "food_type": 1,
                "kind": "noodle",
                "instance_id": instance_id,
                "instance_prim_path": str(source["instance_prim_path"]),
                "object_prim_path": str(source["object_prim_path"]),
                "mesh_prim_path": str(source["mesh_prim_path"]),
                "recording_prim_path": str(source["recording_prim_path"]),
                "parent_prim_path": str(source["parent_prim_path"]),
                "parent_pose_world": source["parent_pose_world"],
                "mesh_pose_parent": source["mesh_pose_parent"],
                "points_coordinate_space": str(source["points_coordinate_space"]),
                "particle_start": particle_start,
                "particle_count": particle_end - particle_start,
                "vertex_count": int(source["vertex_count"]),
                "mapping_kind": "tet_barycentric_v1",
                "topology_sha256": str(source["topology_sha256"]),
                "mapping_sha256": shared_mapping_sha256,
                "rest_reconstruction_max_error_m": reconstruction_error,
                # Indices are local to this binding's particle slice, so the
                # identical authored meshes can intentionally share one map.
                "tet_particle_indices": shared_tet_indices,
                "barycentric_weights": shared_barycentric_weights,
            }
            self.usd_deformable_bindings.append(binding)
            bindings.append(binding)
            ctx.logger.info(
                f"[chef] USDA noodle bound: id={instance_id:02d} prim={binding['mesh_prim_path']} "
                f"particles={binding['particle_count']} tets={tet_end - tet_start} "
                f"authored_vertices={binding['vertex_count']} "
                f"mapping={binding['mapping_sha256'][:12]} "
                f"rest_error={reconstruction_error:.3e}m"
            )
        return bindings

    def _add_usda_side_dish_soft_proxies(self, builder, ctx):
        """Build soft proxies for every instance in the selected USDA families."""
        migrated_families = set(str(family) for family in self.cfg.usd_side_dishes.migrated_families)
        sources = [source for source in self._usd_side_dish_sources if source["family"] in migrated_families]
        if not sources:
            return []

        profile_params = {
            "ham": self.cfg.ham,
            "pepper": self.cfg.pepper,
            "carrot": self.cfg.carrot,
        }
        food_types = {"meat": 2, "pepper": 3, "carrot": 4}
        procedural_kinds = {"meat": "ham", "pepper": "pepper", "carrot": "carrot"}
        family_mappings = {}
        bindings = []

        for source in sources:
            physics_profile = source["physics_profile"]
            if physics_profile not in profile_params:
                raise RuntimeError(
                    f"USDA side dish {source['instance_prim_path']} has unavailable "
                    f"physics profile {physics_profile!r}"
                )
            params = profile_params[physics_profile]
            resolution = np.asarray(source["proxy_resolution"], dtype=np.int64)
            size = np.asarray(source["proxy_size"], dtype=np.float64)
            cell_size = size / resolution
            particle_start = int(builder.particle_count)
            tet_start = len(builder.tet_indices)
            occupied_cell_count = int(np.prod(resolution))

            if source["proxy_geometry"] in {"solid_grid", "strip_grid"}:
                if physics_profile == "ham":
                    particle_radius = float(params.radius_scale) * float(np.min(cell_size))
                else:
                    particle_radius = 0.5 * float(params.cross)
                builder.add_soft_grid(
                    pos=wp.vec3(*source["proxy_corner_world"]),
                    rot=wp.quat(*source["proxy_quat_xyzw"]),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    dim_x=int(resolution[0]),
                    dim_y=int(resolution[1]),
                    dim_z=int(resolution[2]),
                    cell_x=float(cell_size[0]),
                    cell_y=float(cell_size[1]),
                    cell_z=float(cell_size[2]),
                    density=float(params.density),
                    k_mu=float(params.k_mu),
                    k_lambda=float(params.k_lambda),
                    k_damp=float(params.k_damp),
                    particle_radius=particle_radius,
                    label=str(source["mesh_prim_path"]),
                )
            elif source["proxy_geometry"] == "ring_sheet":
                local_vertices, local_tets, occupied = _sparse_voxel_tet_mesh(
                    source["authored_points_local"],
                    source["face_vertex_indices"],
                    source["proxy_bbox_min"],
                    size,
                    resolution,
                )
                occupied_cell_count = len(occupied)
                if occupied_cell_count >= int(np.prod(resolution)):
                    raise RuntimeError(f"USDA ring proxy {source['instance_prim_path']} filled its entire bbox")
                local_from_corner = local_vertices - source["proxy_bbox_min"]
                world_vertices = np.asarray(
                    [
                        source["proxy_corner_world"] + _quat_rotate(source["proxy_quat_xyzw"], point)
                        for point in local_from_corner
                    ],
                    dtype=np.float64,
                )
                builder.add_soft_mesh(
                    pos=wp.vec3(0.0, 0.0, 0.0),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    vertices=world_vertices,
                    indices=local_tets.reshape(-1),
                    density=float(params.density),
                    k_mu=float(params.k_mu),
                    k_lambda=float(params.k_lambda),
                    k_damp=float(params.k_damp),
                    particle_radius=0.5 * float(params.cross),
                    validate_mesh=True,
                    label=str(source["mesh_prim_path"]),
                )
            else:
                raise RuntimeError(f"unsupported USDA side-dish proxy {source['proxy_geometry']!r}")

            particle_end = int(builder.particle_count)
            tet_end = len(builder.tet_indices)
            proxy_particles = np.asarray(
                [
                    [float(point[0]), float(point[1]), float(point[2])]
                    for point in builder.particle_q[particle_start:particle_end]
                ],
                dtype=np.float64,
            )
            proxy_tets_global = np.asarray(builder.tet_indices[tet_start:tet_end], dtype=np.int64).reshape(-1, 4)
            proxy_tets_local = proxy_tets_global - particle_start
            if np.any(proxy_tets_local < 0) or np.any(proxy_tets_local >= len(proxy_particles)):
                raise RuntimeError(f"USDA side-dish proxy {source['instance_prim_path']} escaped its particle slice")

            family = source["family"]
            cached = family_mappings.get(family)
            if cached is None:
                tet_particle_indices, barycentric_weights = _bind_points_to_tets(
                    source["authored_points_world"], proxy_particles, proxy_tets_local
                )
                mapping_sha256 = _tet_binding_sha256(tet_particle_indices, barycentric_weights)
                family_mappings[family] = (
                    proxy_tets_local.copy(),
                    tet_particle_indices,
                    barycentric_weights,
                    mapping_sha256,
                )
            else:
                cached_tets, tet_particle_indices, barycentric_weights, mapping_sha256 = cached
                if not np.array_equal(proxy_tets_local, cached_tets):
                    raise RuntimeError(f"USDA side-dish family {family} proxy topology is inconsistent")

            reconstructed = _reconstruct_tet_bound_points(proxy_particles, tet_particle_indices, barycentric_weights)
            reconstruction_error = float(
                np.max(np.linalg.norm(reconstructed - source["authored_points_world"], axis=1))
            )
            if reconstruction_error > 1.0e-6:
                raise RuntimeError(
                    f"USDA side dish {source['instance_prim_path']} rest reconstruction error "
                    f"{reconstruction_error:g}m exceeds 1e-6m"
                )

            semantic_kind = source["semantic_kind"]
            binding = {
                "food_type": food_types[semantic_kind],
                "kind": procedural_kinds[semantic_kind],
                "semantic_kind": semantic_kind,
                "instance_id": int(source["instance_id"]),
                "family": family,
                "family_instance_id": int(source["family_instance_id"]),
                "instance_prim_path": str(source["instance_prim_path"]),
                "object_prim_path": str(source["object_prim_path"]),
                "mesh_prim_path": str(source["mesh_prim_path"]),
                "recording_prim_path": str(source["recording_prim_path"]),
                "parent_prim_path": str(source["parent_prim_path"]),
                "parent_pose_world": source["parent_pose_world"],
                "mesh_pose_parent": source["mesh_pose_parent"],
                "points_coordinate_space": str(source["points_coordinate_space"]),
                "particle_start": particle_start,
                "particle_count": particle_end - particle_start,
                "vertex_count": int(source["vertex_count"]),
                "proxy_geometry": str(source["proxy_geometry"]),
                "proxy_resolution": [int(value) for value in resolution],
                "proxy_occupied_cell_count": occupied_cell_count,
                "mapping_kind": "tet_barycentric_v1",
                "topology_sha256": str(source["topology_sha256"]),
                "mapping_sha256": mapping_sha256,
                "rest_reconstruction_max_error_m": reconstruction_error,
                "tet_particle_indices": tet_particle_indices,
                "barycentric_weights": barycentric_weights,
            }
            self.usd_deformable_bindings.append(binding)
            bindings.append(binding)
            ctx.logger.info(
                f"[chef] USDA side dish bound: prim={binding['mesh_prim_path']} "
                f"semantic={semantic_kind} proxy={binding['proxy_geometry']} "
                f"particles={binding['particle_count']} tets={tet_end - tet_start} "
                f"cells={occupied_cell_count}/{int(np.prod(resolution))} "
                f"mapping={mapping_sha256[:12]} rest_error={reconstruction_error:.3e}m"
            )
        return bindings

    def _apply_hand_contact_gap(self, builder, ctx):
        """Remove the imported rigid-gap shell from the selected robot hand."""
        prefix = str(self.cfg.grasp.hand_body_prefix or "")
        if not prefix:
            return
        gaps = getattr(builder, "shape_gap", None)
        shape_bodies = list(getattr(builder, "shape_body", []) or [])
        body_labels = list(getattr(builder, "body_label", []) or [])
        if gaps is None or not shape_bodies or not body_labels:
            ctx.logger.warn("[chef] hand geom gap override skipped: builder topology unavailable")
            return

        target = float(self.cfg.grasp.hand_geom_gap)
        changed = 0
        for shape_idx, body_idx_raw in enumerate(shape_bodies):
            body_idx = int(body_idx_raw)
            if not (0 <= body_idx < len(body_labels)):
                continue
            body_name = str(body_labels[body_idx]).rsplit("/", 1)[-1]
            if not body_name.startswith(prefix):
                continue
            gaps[shape_idx] = target
            changed += 1
        ctx.logger.info(f"[chef] hand geom gap override: prefix={prefix!r} " f"gap={target:g} shapes={changed}")

    def get_deformable_render_bindings(self):
        """Expose exact authored-Mesh maps to the inline OVRtx visualizer."""
        return list(self.usd_deformable_bindings)

    def on_build(self, builder, ctx):
        cfg = self.cfg
        self._apply_scene_overrides(ctx)
        self._apply_hand_contact_gap(builder, ctx)
        self.food_instances = []
        self.usd_deformable_bindings = []
        ctx.logger.info(
            f"[chef] build: noodles={cfg.noodle.n_noodles} ham={cfg.ham.n_cubes} "
            f"peppers={cfg.pepper.n} carrots={cfg.carrot.n}  "
            f"wok=({cfg.wok.home_x:g},{cfg.wok.home_y:g},{cfg.wok.home_z:g}) "
            f"rim={cfg.wok.rim_r:g} depth={cfg.wok.depth:g}"
        )

        if cfg.usd_objects.enabled:
            self._bind_usda_wok(builder, ctx)
            self._disable_usda_stove_rigid_collisions(builder, ctx)
            self._add_usda_stove_support(builder, ctx)
            assert self._food_spawn_xy is not None and self._food_rim_z is not None
            authored_noodles = self._add_usda_noodle_soft_proxies(builder, ctx)
            authored_side_dishes = self._add_usda_side_dish_soft_proxies(builder, ctx)
            self._add_procedural_food(
                builder,
                cfg,
                self._food_spawn_xy,
                self._food_rim_z,
                authored_noodles=authored_noodles,
                authored_side_dishes=authored_side_dishes,
            )
            return

        # --- wok body: KINEMATIC (parked / auto-flip) or DYNAMIC (free rigid) ---
        # The wok is an independent body. Both modes use add_body, which
        # auto-creates a FREE joint + its own articulation; is_kinematic picks:
        #   * KINEMATIC (wok_dynamic=False): ignores gravity/contact; stays at
        #     home unless the AUTO-FLIP controller writes joint_q each frame.
        #   * DYNAMIC   (wok_dynamic=True):  real free body under gravity that
        #     rests on the stove ring via the thin-shell rigid collider, and
        #     that a teleop gripper can push / grab.
        self.wok_dynamic = bool(cfg.wok.wok_dynamic)
        self.auto_flip = bool(cfg.auto_flip) and not self.wok_dynamic
        if bool(cfg.auto_flip) and self.wok_dynamic:
            ctx.logger.warn(
                "[chef] auto_flip ignored: wok_dynamic=True (a dynamic free body "
                "can't be driven by the kinematic joint_q controller)."
            )
        _mode = "DYNAMIC" if self.wok_dynamic else ("AUTO-FLIP" if self.auto_flip else "PARKED")
        self.pan_id = -1
        # DYNAMIC starts a hair above the ring seat so it settles instead of
        # spawning interpenetrating (gap=0 rigid contact would kick an overlap).
        spawn_z = cfg.wok.home_z + (cfg.wok.drop_gap if self.wok_dynamic else 0.0)
        try:
            import newton

            inertia = wp.mat33(
                cfg.wok.inertia_diag[0],
                0.0,
                0.0,
                0.0,
                cfg.wok.inertia_diag[1],
                0.0,
                0.0,
                0.0,
                cfg.wok.inertia_diag[2],
            )
            self.pan_id = builder.add_body(
                xform=wp.transform(
                    wp.vec3(cfg.wok.home_x, cfg.wok.home_y, spawn_z),
                    wp.quat_identity(),
                ),
                label="chef_wok",
                mass=cfg.wok.mass,
                inertia=inertia,
                com=wp.vec3(0.0, 0.0, -cfg.wok.depth * 0.5),
                is_kinematic=not self.wok_dynamic,
            )
            ctx.logger.info(
                f"[chef] wok body={self.pan_id} mode={_mode} "
                f"is_kinematic={not self.wok_dynamic} spawn_z={spawn_z:g}"
            )
        except Exception as exc:
            ctx.logger.warn(f"[chef] wok body setup failed: {exc!r}; falling back to static body=-1")
            self.pan_id = -1

        wok_xform_local = wp.transform_identity()  # shapes are body-local now

        # Thick collider shell — PARTICLE-ONLY contact (noodles vs wok).
        # ``has_shape_collision`` MUST stay False here: ``_make_wok`` inflates
        # the outer surface by ``thickness`` (8 cm), and that inflated shell
        # extends ~17 cm below the rim plane (vs the actual bowl which is
        # only ``depth=9 cm`` below).  If shape collision were enabled on
        # this thick shell, with the wok at ``home_z=1.090`` the outer
        # shell bottom would sit at z≈0.920, 3.7 cm INSIDE the box top
        # (0.9573) — and the wok-vs-box penetration impulse blows the wok
        # off into orbit on substep 1.  The thin visual shell below carries
        # the shape-collision flag instead, using the true bowl shape.
        cverts, ctris = _make_wok(
            rim_r=cfg.wok.rim_r,
            depth=cfg.wok.depth,
            thickness=0.080,
            n_az=cfg.wok.az,
            n_radial=cfg.wok.radial,
        )
        try:
            import newton

            collider_mesh = newton.Mesh(cverts, ctris.flatten(), compute_inertia=False)
            collider_cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                mu=cfg.noodle.wok_mu,
                has_shape_collision=False,
                has_particle_collision=True,
            )
            self.collider_shape = builder.add_shape_mesh(
                self.pan_id, xform=wok_xform_local, mesh=collider_mesh, cfg=collider_cfg, label="chef_wok_collider"
            )
            # Thin visual + shape-collider shell.  Thin (3 mm) so its outer
            # surface is ≈ the actual bowl shape — that's what the wok's
            # rigid contact against the box / floor / gripper uses.  Visible
            # (stays True; the THICK collider gets hidden in on_model_ready).
            vverts, vtris = _make_wok(
                rim_r=cfg.wok.rim_r,
                depth=cfg.wok.depth,
                thickness=0.003,
                n_az=48,
                n_radial=16,
            )
            visual_mesh = newton.Mesh(vverts, vtris.flatten(), compute_inertia=False)
            visual_cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                mu=cfg.stove.mu,
                # Shape collision turns ON only in DYNAMIC mode — the wok rests
                # on the stove ring and a teleop gripper can push it.  PARKED /
                # AUTO-FLIP keep it visual-only (the kinematic wok doesn't rest
                # on anything).  Particle contact always lives on the THICK
                # collider (above), never here.
                has_shape_collision=self.wok_dynamic,
                has_particle_collision=False,
                gap=cfg.stove.gap,  # 1.4.0: gap=0 so MuJoCo doesn't inherit rigid_gap=0.1
            )
            self.visual_shape = builder.add_shape_mesh(
                self.pan_id,
                xform=wok_xform_local,
                mesh=visual_mesh,
                cfg=visual_cfg,
                color=wp.vec3(0.28, 0.29, 0.31),
                label="chef_wok_visual",
            )
        except Exception as exc:
            ctx.logger.warn(f"[chef] wok mesh shape add failed: {exc!r}; continuing without colliders")

        # Wok handle (锅把手) — cosmetic capsules + end knob + the square grip
        # section the gripper grasps, all parented to the pan body so they ride
        # with the wok.  See GraspGeometry / _add_wok_handle.
        self._add_wok_handle(builder, ctx)

        # Grasp interface: pad boxes on the gripper's support links, then the
        # explicit mujoco:pair rows binding them to the grip box.  Both are
        # no-ops when grasp.enable is False.  Ordering matters — the pairs
        # reference shape indices produced by the two steps above.
        self._add_gripper_pads(builder, ctx)
        self._add_grasp_pairs(builder, ctx)

        # Stove: static counter box + wok-support ring, placed under the wok
        # home so the round bottom seats in the ring.  (The old comment claimed
        # the G2 scene supplies a stove — it doesn't; we build one here.)
        self._add_stove(builder, ctx)

        self._add_procedural_food(
            builder,
            cfg,
            (cfg.wok.home_x, cfg.wok.home_y),
            cfg.wok.home_z,
        )

    def _add_procedural_food(
        self,
        builder,
        cfg,
        spawn_xy,
        rim_z,
        authored_noodles=None,
        authored_side_dishes=None,
    ):
        """Emit procedural food and replace selected instances with USDA bindings."""
        # Noodles first so the tube-render slice is contiguous.
        authored_by_id = {int(binding["instance_id"]): binding for binding in (authored_noodles or [])}
        if len(authored_by_id) != len(authored_noodles or []):
            raise RuntimeError("duplicate USDA noodle instance ids")
        if any(instance_id < 0 or instance_id >= cfg.noodle.n_noodles for instance_id in authored_by_id):
            raise RuntimeError(
                f"USDA noodle instance ids outside [0,{cfg.noodle.n_noodles}): " f"{sorted(authored_by_id)}"
            )
        procedural_ids = [
            instance_id for instance_id in range(cfg.noodle.n_noodles) if instance_id not in authored_by_id
        ]
        starts, n_len = _emit_noodles(
            builder,
            cfg.noodle,
            drop_z0=rim_z + cfg.noodle.drop_z0_offset,
            spawn_xy=spawn_xy,
            instance_ids=procedural_ids,
        )
        noodle_ranges = [
            (instance_id, int(binding["particle_start"]), binding) for instance_id, binding in authored_by_id.items()
        ]
        procedural_particle_count = 4 * (int(cfg.noodle.n_len) + 1)
        noodle_ranges.extend((instance_id, int(start), None) for instance_id, start in zip(procedural_ids, starts))
        noodle_ranges.sort(key=lambda item: item[1])
        if len(noodle_ranges) != int(cfg.noodle.n_noodles):
            raise RuntimeError(
                f"chef noodle binding produced {len(noodle_ranges)} instances; " f"expected {cfg.noodle.n_noodles}"
            )
        self.starts_np = np.asarray([start for _, start, _ in noodle_ranges], dtype=np.int32)
        self.n_len_noodle = n_len
        for instance_id, start, authored_noodle in noodle_ranges:
            item = {
                "food_type": 1,
                "kind": "noodle",
                "instance_id": int(instance_id),
                "particle_start": int(start),
                "particle_count": (
                    procedural_particle_count if authored_noodle is None else int(authored_noodle["particle_count"])
                ),
                "proposed_usd_prim": f"/World/ChefFood/noodles/noodle_{instance_id:02d}",
            }
            if authored_noodle is not None:
                item.pop("proposed_usd_prim")
                for key in (
                    "instance_prim_path",
                    "object_prim_path",
                    "parent_prim_path",
                    "mesh_prim_path",
                    "recording_prim_path",
                    "parent_pose_world",
                    "mesh_pose_parent",
                    "points_coordinate_space",
                    "vertex_count",
                    "mapping_kind",
                    "topology_sha256",
                    "mapping_sha256",
                    "rest_reconstruction_max_error_m",
                ):
                    item[key] = authored_noodle[key]
            self.food_instances.append(item)

        authored_side_dishes = list(authored_side_dishes or [])
        authored_by_kind = {"ham": {}, "pepper": {}, "carrot": {}}
        for binding in authored_side_dishes:
            kind = str(binding["kind"])
            if kind not in authored_by_kind:
                raise RuntimeError(f"unsupported authored side-dish kind {kind!r}")
            instance_id = int(binding["instance_id"])
            if instance_id in authored_by_kind[kind]:
                raise RuntimeError(f"duplicate authored {kind} instance id {instance_id}")
            authored_by_kind[kind][instance_id] = binding

        def append_authored(binding):
            item = {
                "food_type": int(binding["food_type"]),
                "kind": str(binding["kind"]),
                "instance_id": int(binding["instance_id"]),
                "particle_start": int(binding["particle_start"]),
                "particle_count": int(binding["particle_count"]),
            }
            for key in (
                "semantic_kind",
                "family",
                "family_instance_id",
                "instance_prim_path",
                "object_prim_path",
                "parent_prim_path",
                "mesh_prim_path",
                "recording_prim_path",
                "parent_pose_world",
                "mesh_pose_parent",
                "points_coordinate_space",
                "vertex_count",
                "proxy_geometry",
                "proxy_resolution",
                "proxy_occupied_cell_count",
                "mapping_kind",
                "topology_sha256",
                "mapping_sha256",
                "rest_reconstruction_max_error_m",
            ):
                item[key] = binding[key]
            self.food_instances.append(item)

        for binding in authored_side_dishes:
            append_authored(binding)

        if cfg.usd_side_dishes.procedural_fallback and cfg.ham.n_cubes > 0:
            authored_ham = authored_by_kind["ham"]
            if any(instance_id >= cfg.ham.n_cubes for instance_id in authored_ham):
                raise RuntimeError(f"authored meat ids exceed procedural ham range: {sorted(authored_ham)}")
            ham_ids = [instance_id for instance_id in range(cfg.ham.n_cubes) if instance_id not in authored_ham]
            self.ham_tri_range, ham_starts = _emit_ham(
                builder,
                cfg.ham,
                drop_z0=rim_z + cfg.ham.drop_z0_offset,
                spawn_xy=spawn_xy,
                instance_ids=ham_ids,
            )
            count = (cfg.ham.dim + 1) ** 3
            for instance_id, start in zip(ham_ids, ham_starts):
                self.food_instances.append(
                    {
                        "food_type": 2,
                        "kind": "ham",
                        "instance_id": instance_id,
                        "particle_start": int(start),
                        "particle_count": int(count),
                        "proposed_usd_prim": f"/World/ChefFood/ham/ham_{instance_id:02d}",
                    }
                )
        if cfg.usd_side_dishes.procedural_fallback and cfg.pepper.n > 0:
            authored_pepper = authored_by_kind["pepper"]
            if any(instance_id >= cfg.pepper.n for instance_id in authored_pepper):
                raise RuntimeError(f"authored pepper ids exceed procedural pepper range: {sorted(authored_pepper)}")
            pepper_ids = [instance_id for instance_id in range(cfg.pepper.n) if instance_id not in authored_pepper]
            _, pepper_starts = _emit_strips(
                builder,
                cfg.pepper,
                drop_z0=rim_z + cfg.pepper.drop_z0_offset,
                spawn_xy=spawn_xy,
                seed=3,
                instance_ids=pepper_ids,
            )
            self.pepper_pstart = int(pepper_starts[0]) if len(pepper_starts) else 0
            self.pepper_render_count = len(pepper_starts)
            count = 4 * (cfg.pepper.n_len + 1)
            for instance_id, start in zip(pepper_ids, pepper_starts):
                self.food_instances.append(
                    {
                        "food_type": 3,
                        "kind": "pepper",
                        "instance_id": instance_id,
                        "particle_start": int(start),
                        "particle_count": int(count),
                        "proposed_usd_prim": f"/World/ChefFood/pepper/pepper_{instance_id:02d}",
                    }
                )
        if cfg.usd_side_dishes.procedural_fallback and cfg.carrot.n > 0:
            authored_carrot = authored_by_kind["carrot"]
            if any(instance_id >= cfg.carrot.n for instance_id in authored_carrot):
                raise RuntimeError(f"authored carrot ids exceed procedural carrot range: {sorted(authored_carrot)}")
            carrot_ids = [instance_id for instance_id in range(cfg.carrot.n) if instance_id not in authored_carrot]
            _, carrot_starts = _emit_strips(
                builder,
                cfg.carrot,
                drop_z0=rim_z + cfg.carrot.drop_z0_offset,
                spawn_xy=spawn_xy,
                seed=4,
                instance_ids=carrot_ids,
            )
            self.carrot_pstart = int(carrot_starts[0]) if len(carrot_starts) else 0
            self.carrot_render_count = len(carrot_starts)
            count = 4 * (cfg.carrot.n_len + 1)
            for instance_id, start in zip(carrot_ids, carrot_starts):
                self.food_instances.append(
                    {
                        "food_type": 4,
                        "kind": "carrot",
                        "instance_id": instance_id,
                        "particle_start": int(start),
                        "particle_count": int(count),
                        "proposed_usd_prim": f"/World/ChefFood/carrot/carrot_{instance_id:02d}",
                    }
                )

        if not cfg.usd_side_dishes.procedural_fallback:
            self.ham_tri_range = (0, 0)
            self.pepper_pstart = 0
            self.carrot_pstart = 0
            self.pepper_render_count = 0
            self.carrot_render_count = 0

        # Particle velocity clamp (avoid FEM blow-ups during settling).
        builder.particle_max_velocity = cfg.xpbd.max_vel

        # Autonomous wok controller — drives the kinematic free-joint body
        # through the SETTLE→GATHER→TOSS→CATCH→DWELL cycle from the
        # standalone chow_mein_flip_fem demo.  on_post_step writes joint_q
        # each frame; on_model_ready locates the joint_q slot.
        self.driver = _WokDriver(cfg.wok)
        self.ctrl = _Controller(self.driver, cfg.controller, cfg.wok.depth)
        self.food = _NoodleSensor()

    def _add_wok_handle(self, builder, ctx):
        """Wok handle, parented to the pan body (rides the flip).

        Coordinates are body-local, taken verbatim from
        chow_mein_flip_fem.py:_add_cosmetics — the pan body origin is the
        rim plane in both the demo and this plugin (``_make_wok`` puts the
        rim at z=0), so the geometry transfers 1:1 regardless of the wok's
        world ``home_z``.

        Cosmetic shapes use ``density=0.0`` and no collision so they add no
        mass/inertia and never perturb the noodles or the wok-vs-stove
        contact.  The one exception is the square GRIP SECTION in the middle
        of the wooden stick, which is the gripper's grasp target — see
        :class:`GraspGeometry` for why it is a box and not the round stick.
        It does carry ``has_shape_collision=True`` (required for its
        ``mujoco:pair`` to compile), but sits in a private
        ``collision_group`` so broad-phase only ever pairs it with the
        gripper pads — never the stove, the noodles, or the wok's own shell.
        """
        if self.pan_id < 0:
            return
        try:
            import newton

            g = self.cfg.grasp
            vis = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=False,
                has_particle_collision=False,
            )
            steel = wp.vec3(0.58, 0.56, 0.50)
            wood = wp.vec3(0.45, 0.27, 0.13)
            # Handle is mirrored to the wok's local -X (every x-coord negated vs
            # the demo's +X layout). newton 1.3.0 drops the kinematic free-body
            # rotation, so base_yaw can't turn the handle toward the robot;
            # mirroring the geometry achieves the same "handle faces the robot"
            # result purely in body-local space, independent of body_q rotation.
            # Steel ferrule: two side brackets + a tapered base socket.
            for a, b, r in (
                ((-0.1372, 0.0190, -0.0299), (-0.1840, 0.0146, -0.0248), 0.0044),
                ((-0.1372, -0.0190, -0.0299), (-0.1840, -0.0146, -0.0248), 0.0044),
                ((-0.1723, 0.0, -0.0248), (-0.2029, 0.0, -0.0226), 0.0095),
            ):
                xform, hh = _capsule_fromto(a, b)
                builder.add_shape_capsule(self.pan_id, xform=xform, radius=r, half_height=hh, cfg=vis, color=steel)

            # Wooden grip stick.  Runs from the ferrule socket to the knob;
            # linear in both x and z, so any point along it is a lerp.
            stick_a = np.array([-0.2029, 0.0, -0.0226])
            stick_b = np.array([-0.3139, 0.0, -0.0131])

            def _on_stick(x):
                """Point on the stick centreline at wok-local ``x``."""
                t = (x - stick_a[0]) / (stick_b[0] - stick_a[0])
                return stick_a + t * (stick_b - stick_a)

            if g.enable:
                # Round stubs either side of the square grip section, so the
                # handle still reads as a wooden stick end-to-end.
                p0, p1 = _on_stick(g.grip_x0), _on_stick(g.grip_x1)
                for a, b in ((stick_a, p0), (p1, stick_b)):
                    if abs(b[0] - a[0]) > 1e-6:
                        xform, hh = _capsule_fromto(a, b)
                        builder.add_shape_capsule(
                            self.pan_id, xform=xform, radius=0.0095, half_height=hh, cfg=vis, color=wood
                        )
                # The square grip section itself.
                d = p1 - p0
                center = 0.5 * (p0 + p1)
                grip_cfg = newton.ModelBuilder.ShapeConfig(
                    density=0.0,
                    # Must be True — a mujoco:pair referencing a non-colliding
                    # shape trips newton's geom-count cross-check.  Isolation
                    # comes from the private collision_group instead, so this
                    # box still never touches the stove / noodles / robot.
                    has_shape_collision=True,
                    has_particle_collision=False,
                    collision_group=g.collision_group,
                    # Never inherit builder.rigid_gap (0.1 m): a 10 cm detection
                    # shell would write contact records against every geom in
                    # reach and can overflow nconmax, silently dropping the real
                    # contacts.
                    gap=0.0,
                )
                self.grip_shape = builder.add_shape_box(
                    self.pan_id,
                    xform=wp.transform(wp.vec3(*center), _quat_x_to(d)),
                    hx=0.5 * float(np.linalg.norm(d)),
                    hy=g.grip_half,
                    hz=g.grip_half,
                    cfg=grip_cfg,
                    color=wood,
                    label="chef_wok_grip",
                    custom_attributes={"mujoco:condim": int(g.condim)},
                )
                ctx.logger.info(
                    f"[chef] wok grip box shape={self.grip_shape} "
                    f"x=[{g.grip_x0:.4f},{g.grip_x1:.4f}] len={np.linalg.norm(d) * 1e3:.1f}mm "
                    f"section={2 * g.grip_half * 1e3:.1f}mm condim={g.condim}"
                )
            else:
                # ROUND graspable handle — the mode for a multi-finger hand.
                #
                # The square grip box above exists for a PARALLEL-JAW gripper,
                # where two single-point contacts on a cylinder have zero moment
                # capacity about the closing axis.  A five-finger hand does not
                # have that problem: the envelope comes from many finger links
                # at once, so a round stick is both fine and more natural, and
                # the fingers can wrap it the way a hand actually holds a wok.
                #
                # Unlike the cosmetic capsules this one COLLIDES, and it stays
                # in the DEFAULT collision group so the hand's own finger meshes
                # can touch it — no explicit mujoco:pair, no private group.
                # gap=0.0 is mandatory (see the class docstring): without it the
                # shape inherits builder.rigid_gap = 0.1 and writes contact
                # records against everything within 10 cm.
                grip_cfg = newton.ModelBuilder.ShapeConfig(
                    density=0.0,
                    has_shape_collision=True,
                    has_particle_collision=False,
                    gap=0.0,
                )
                xform, hh = _capsule_fromto(stick_a, stick_b)
                self.grip_shape = builder.add_shape_capsule(
                    self.pan_id,
                    xform=xform,
                    radius=0.0095,
                    half_height=hh,
                    cfg=grip_cfg,
                    color=wood,
                    label="chef_wok_grip_round",
                    # condim >= 4 adds torsional friction about the contact
                    # normal, which is what keeps a round handle from spinning
                    # inside the fingers.  mjc_contact.apply_pre_finalize
                    # rewrites shape friction but never touches condim, so this
                    # survives to the MJCF compile — whereas friction authored
                    # here would not (set it via the scene yaml's
                    # newton.mjc_contact.passive.friction instead).
                    custom_attributes={"mujoco:condim": int(g.condim)},
                )
                ctx.logger.info(
                    f"[chef] wok ROUND graspable handle shape={self.grip_shape} "
                    f"r=9.5mm len={2 * hh * 1e3:.0f}mm condim={g.condim} "
                    f"(default collision group — for a multi-finger hand)"
                )

                # INBOARD axial stop.  The end knob bounds the outboard end, but
                # the steel ferrule socket is the same 9.5 mm radius as the stick
                # so it forms no shoulder — the hand could slide inboard toward
                # the bowl unchecked.  Add a collar at the inboard end of the
                # grip so the graspable section is bounded by a shoulder at BOTH
                # ends, the way a real tool handle is.  Together with the knob
                # this is genuine form closure along the handle axis, which
                # friction alone cannot provide on a smooth cylinder.
                if g.collar_radius > 0.0095:
                    self.collar_shape = builder.add_shape_sphere(
                        self.pan_id,
                        xform=wp.transform(wp.vec3(*stick_a), wp.quat_identity()),
                        radius=g.collar_radius,
                        cfg=grip_cfg,
                        color=steel,
                        label="chef_wok_grip_collar",
                        custom_attributes={"mujoco:condim": int(g.condim)},
                    )
                    ctx.logger.info(
                        f"[chef] wok grip collar shape={self.collar_shape} "
                        f"r={g.collar_radius * 1e3:.1f}mm at the inboard end "
                        f"({(g.collar_radius - 0.0095) * 1e3:.1f} mm shoulder) — axial stop "
                        f"toward the bowl"
                    )

            # Rounded end knob.  In the round-handle (multi-finger) mode this is
            # a REAL COLLIDER, not decoration: it is r=0.0124 against the stick's
            # r=0.0095, i.e. a 2.9 mm radial shoulder, which is the only thing
            # that mechanically stops the hand sliding off the end of the
            # handle.  A smooth cylinder has ZERO form closure along its own
            # axis — axial load can only ever be carried by mu*N — so without
            # this the handle slides fore-and-aft through the fingers whenever
            # the arm accelerates along the handle direction.  Real wok handles
            # have exactly this shoulder for exactly this reason.
            # In parallel-jaw mode the jaws clamp a flat, so the knob stays
            # cosmetic and out of the contact budget.
            knob_cfg = vis
            if not g.enable:
                knob_cfg = newton.ModelBuilder.ShapeConfig(
                    density=0.0,
                    has_shape_collision=True,
                    has_particle_collision=False,
                    gap=0.0,
                )
            self.knob_shape = builder.add_shape_sphere(
                self.pan_id,
                xform=wp.transform(wp.vec3(-0.3227, 0.0, -0.0124), wp.quat_identity()),
                radius=0.0124,
                cfg=knob_cfg,
                color=wood,
                label="chef_wok_knob",
                custom_attributes={"mujoco:condim": int(g.condim)} if not g.enable else None,
            )
            if not g.enable:
                ctx.logger.info(
                    f"[chef] wok end knob shape={self.knob_shape} r=12.4mm is a COLLIDER "
                    f"(2.9 mm shoulder over the 9.5 mm stick) — axial stop against the hand "
                    f"sliding off the end; condim={g.condim}"
                )
            ctx.logger.info("[chef] wok handle added (parented to pan)")
        except Exception as exc:
            ctx.logger.warn(f"[chef] wok handle add failed: {exc!r}; continuing without handle")

    def _add_gripper_pads(self, builder, ctx):
        """Add a flush box pad to each of the gripper's two support links.

        The support links already carry convex-mesh colliders from the URDF,
        but a (BOX, MESH) pair still resolves through GJK/EPA to a single
        contact point.  Adding an explicit box on each jaw makes the grasp a
        BOX-BOX pair, which is the one combination mujoco-warp multicontacts
        unconditionally — four points per pad instead of one.

        The robot's bodies are already in the builder by the time ``on_build``
        runs (``setup/model.py`` calls ``add_usd`` first, then the plugin), so
        we can look the links up by label.  Nothing here touches the existing
        mesh colliders; the pads are additive, and their private
        ``collision_group`` keeps them from generating any contact except
        against the grip box.
        """
        g = self.cfg.grasp
        if not g.enable:
            return
        try:
            import newton

            labels = list(getattr(builder, "body_label", []) or [])
            # Body labels are USD prim paths (/genie/.../gripper_r_left_support_link);
            # match on the trailing component so the robot prefix is irrelevant.
            found = {}
            for want in g.pad_link_names:
                for i, lab in enumerate(labels):
                    if lab and str(lab).rsplit("/", 1)[-1] == want:
                        found[want] = i
                        break
            missing = [w for w in g.pad_link_names if w not in found]
            if missing:
                ctx.logger.warn(
                    f"[chef] grasp pads: link(s) {missing} not found among {len(labels)} bodies — "
                    f"no pads added, the gripper will NOT be able to hold the wok.  "
                    f"Check the robot's gripper naming in the scene yaml."
                )
                return

            pad_cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                # See grip_cfg: True is required for the pair to compile; the
                # private collision group is what keeps these pads from
                # colliding with anything except the grip box.
                has_shape_collision=True,
                has_particle_collision=False,
                collision_group=g.collision_group,
                gap=0.0,
            )
            # Pad sits just inboard of the mesh face so its OUTER surface is
            # coplanar with it: face_x is the jaw plane, and the box centre is
            # half a thickness behind (less negative x = into the link).
            cx = g.pad_face_x + g.pad_half_thick
            for want, body_idx in found.items():
                idx = builder.add_shape_box(
                    body_idx,
                    xform=wp.transform(wp.vec3(cx, g.pad_center_y, 0.0), wp.quat_identity()),
                    hx=g.pad_half_thick,
                    hy=g.pad_half_len,
                    hz=g.pad_half_wid,
                    cfg=pad_cfg,
                    color=wp.vec3(0.15, 0.15, 0.17),
                    label=f"chef_grip_pad_{want}",
                    custom_attributes={"mujoco:condim": int(g.condim)},
                )
                self.pad_shapes.append(idx)
                ctx.logger.info(f"[chef] grasp pad shape={idx} on body={body_idx} ({want})")

                if g.lip_enable:
                    # The lip outer face is coplanar with the pad face. Its
                    # inner edge is exactly at the pad boundary, so the
                    # centered handle has clearance and only contacts a lip
                    # after it starts to slide tangentially.
                    lip_x = g.pad_face_x + g.lip_half_x
                    lip_specs = (
                        (
                            "y_neg",
                            g.pad_center_y - g.pad_half_len - g.lip_half_y,
                            g.lip_half_x,
                            g.lip_half_y,
                            g.pad_half_wid,
                        ),
                        (
                            "y_pos",
                            g.pad_center_y + g.pad_half_len + g.lip_half_y,
                            g.lip_half_x,
                            g.lip_half_y,
                            g.pad_half_wid,
                        ),
                        ("z_neg", g.pad_center_y, g.lip_half_x, g.pad_half_len, g.lip_half_z),
                        ("z_pos", g.pad_center_y, g.lip_half_x, g.pad_half_len, g.lip_half_z),
                    )
                    for lip_name, lip_y, hx, hy, hz in lip_specs:
                        lip_z = 0.0
                        if lip_name == "z_neg":
                            lip_z = -g.pad_half_wid - g.lip_half_z
                        elif lip_name == "z_pos":
                            lip_z = g.pad_half_wid + g.lip_half_z
                        lip_idx = builder.add_shape_box(
                            body_idx,
                            xform=wp.transform(wp.vec3(lip_x, lip_y, lip_z), wp.quat_identity()),
                            hx=hx,
                            hy=hy,
                            hz=hz,
                            cfg=pad_cfg,
                            color=wp.vec3(0.15, 0.15, 0.17),
                            label=f"chef_grip_lip_{want}_{lip_name}",
                            custom_attributes={"mujoco:condim": int(g.condim)},
                        )
                        self.pad_shapes.append(lip_idx)
                    ctx.logger.info(
                        f"[chef] grasp lips enabled on {want}: "
                        f"y={g.pad_center_y - g.pad_half_len:.4f}..{g.pad_center_y + g.pad_half_len:.4f} "
                        f"z=+/-{g.pad_half_wid:.4f}"
                    )
        except Exception as exc:
            ctx.logger.warn(f"[chef] grasp pad add failed: {exc!r}; gripper will not hold the wok")

    def _add_grasp_pairs(self, builder, ctx):
        """Bind the grip box to each pad with an explicit ``mujoco:pair`` row.

        Why pairs rather than plain shape collision:

        * ``mjc_contact.apply_pre_finalize`` runs AFTER this hook and
          unconditionally rewrites every shape's ``shape_material_mu`` /
          ``mu_torsional`` / ``mu_rolling`` / ``ke`` / ``kd`` from its
          per-object-class table.  Friction authored on the ShapeConfig here
          would simply be discarded.  It never touches ``mujoco:pair_*``.
        * A pair bypasses contype/conaffinity for the two shapes it names, so
          the grasp contact is fully specified here.  The grip box and pads
          additionally sit in a private ``collision_group``, so broad-phase
          never pairs them with the stove, the noodles, or the robot's own
          links.

        Caveat: once a pair exists, EVERY contact parameter for it comes from
        the pair arrays, including ``solref``, whose pair-side default is a
        20 ms timeconst — 10x softer than this scene's tuned 2 ms.  All of
        them are therefore written explicitly.

        The ``mujoco:pair_*`` attributes only exist after
        ``SolverMuJoCo.register_custom_attributes`` has run, so on the
        featherstone / avbd paths this degrades to a warning.
        """
        g = self.cfg.grasp
        if not g.enable or self.grip_shape < 0 or not self.pad_shapes:
            return
        try:
            world = int(getattr(builder, "current_world", -1))
            if world < 0:
                world = 0
            for pad in self.pad_shapes:
                builder.add_custom_values(
                    **{
                        "mujoco:pair_geom1": int(self.grip_shape),
                        "mujoco:pair_geom2": int(pad),
                        "mujoco:pair_world": world,
                        "mujoco:pair_condim": int(g.condim),
                        "mujoco:pair_friction": tuple(float(v) for v in g.pair_friction),
                        "mujoco:pair_solref": tuple(float(v) for v in g.pair_solref),
                        "mujoco:pair_solimp": tuple(float(v) for v in g.pair_solimp),
                        "mujoco:pair_margin": 0.0,
                        "mujoco:pair_gap": 0.0,
                    }
                )
            ctx.logger.info(
                f"[chef] grasp pairs: grip={self.grip_shape} vs pads={self.pad_shapes} "
                f"condim={g.condim} friction={g.pair_friction} solref={g.pair_solref}"
            )
        except Exception as exc:
            ctx.logger.warn(
                f"[chef] grasp pair add failed: {exc!r}.  mujoco:pair_* is registered by "
                f"SolverMuJoCo — expected on the featherstone/avbd paths.  The gripper will "
                f"NOT hold the wok on this solver."
            )

    def _add_stove(self, builder, ctx):
        """Static stove: a counter box + a wok-support ring under the wok home.

        Placed in WORLD coordinates on body -1 (static) and COMPUTED from the
        wok's spherical-cap geometry so the ring cradles the round bottom at
        ``home_z``.  A round bottom on a flat top would wobble, so the ring is a
        circle of beads — the bowl seats on a ring of contacts (stable, since
        the CoM sits below the ring).  Every shape passes ``gap=0`` so MuJoCo
        doesn't inherit ``rigid_gap=0.1`` and float the wok above the ring.
        """
        st = self.cfg.stove
        if not st.enable:
            self.stove_shapes = []
            return
        wok = self.cfg.wok
        try:
            import newton

            # Spherical-cap geometry (matches _make_wok): rim plane at local
            # z=0, bowl bulges DOWN to z=-depth; outer thin-shell radius R_out.
            R = (wok.rim_r * wok.rim_r + wok.depth * wok.depth) / (2.0 * wok.depth)
            center_z = R - wok.depth
            R_out = R + 0.003  # thin visual-shell thickness (see on_build)
            hx, hy = wok.home_x, wok.home_y

            # Ring grip height: where the outer sphere sits at radius r_ring.
            r_ring = float(min(st.ring_radius, 0.98 * R_out))
            surf_local = center_z - math.sqrt(max(R_out * R_out - r_ring * r_ring, 0.0))
            ring_surface_z = wok.home_z + surf_local  # world z of bowl surface at r_ring
            bead_center_z = ring_surface_z - st.bead_radius  # bead top touches the bowl

            # Bowl's lowest point + counter box top just below it.
            bowl_bottom_z = wok.home_z + (center_z - R_out)
            box_top_z = bowl_bottom_z - st.base_top_clearance

            cfg_static = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                mu=st.mu,
                has_shape_collision=True,
                has_particle_collision=False,
                gap=st.gap,
            )

            shapes: list[int] = []

            # Counter box (from floor z=0 up to box_top_z).
            if box_top_z > 0.02:
                hz = 0.5 * box_top_z
                shapes.append(
                    builder.add_shape_box(
                        -1,
                        xform=wp.transform(wp.vec3(hx, hy, hz), wp.quat_identity()),
                        hx=st.base_half_x,
                        hy=st.base_half_y,
                        hz=hz,
                        cfg=cfg_static,
                        color=wp.vec3(*st.base_color),
                        label="chef_stove_base",
                    )
                )
            else:
                ctx.logger.warn(
                    f"[chef] stove base skipped: computed box top {box_top_z:g} m too low " f"(home_z={wok.home_z:g})."
                )

            # Wok-support ring: beads on a circle of radius r_ring.
            n = max(int(st.ring_beads), 3)
            for i in range(n):
                th = 2.0 * math.pi * i / n
                bx = hx + r_ring * math.cos(th)
                by = hy + r_ring * math.sin(th)
                shapes.append(
                    builder.add_shape_sphere(
                        -1,
                        xform=wp.transform(wp.vec3(bx, by, bead_center_z), wp.quat_identity()),
                        radius=st.bead_radius,
                        cfg=cfg_static,
                        color=wp.vec3(*st.ring_color),
                        label=f"chef_stove_ring_{i:02d}",
                    )
                )

            self.stove_shapes = shapes
            ctx.logger.info(
                f"[chef] stove: base_top={box_top_z:g} ring r={r_ring:g} "
                f"beads={n}@z={bead_center_z:g} bowl_bottom={bowl_bottom_z:g} shapes={len(shapes)}"
            )
        except Exception as exc:
            ctx.logger.warn(f"[chef] stove add failed: {exc!r}; continuing without stove")
            self.stove_shapes = []

    def on_model_ready(self, model, ctx):
        # Noodle-vs-noodle (particle hash grid) + noodle-vs-wok (soft contact).
        c = self.cfg
        if hasattr(model, "particle_mu"):
            model.particle_mu = c.noodle.particle_mu
        model.soft_contact_mu = c.noodle.wok_mu
        model.soft_contact_ke = c.contact.soft_ke
        model.soft_contact_kd = c.contact.soft_kd
        model.soft_contact_kf = c.contact.soft_kf
        model.soft_contact_radius = c.contact.soft_radius
        if hasattr(model, "soft_contact_margin"):
            model.soft_contact_margin = c.contact.soft_margin
        if hasattr(model, "particle_max_velocity"):
            model.particle_max_velocity = c.xpbd.max_vel

        if c.food_state.enabled:
            try:
                self._food_state_pub = _ChefFoodStatePublisher(
                    c,
                    self.food_instances,
                    self.usd_deformable_bindings,
                    int(model.particle_count),
                    float(ctx.physics_hz),
                    ctx.logger,
                )
            except Exception as exc:
                self._food_state_pub = None
                ctx.logger.warn(f"[chef] food state publisher init failed: {exc!r}")

        if c.startup_hand_motion.enabled:
            self._startup_hand_pub = _StartupHandCommandPublisher(c, ctx.scene_cfg, ctx.logger)

        # Hide the visual collider shell (cooking surface stays visible).
        try:
            import newton

            flags = model.shape_flags.numpy()
            vis = int(newton.ShapeFlags.VISIBLE)
            if 0 <= self.collider_shape < len(flags):
                flags[self.collider_shape] &= ~vis
                model.shape_flags.assign(flags)
        except Exception as exc:
            ctx.logger.warn(f"[chef] hide collider failed: {exc!r}")

        # Cache ham triangle indices for render (the global tet surface
        # stays hidden; we re-log just the ham slice each frame).
        try:
            tri_a, tri_b = self.ham_tri_range
            if tri_b > tri_a and model.tri_count:
                tri_np = model.tri_indices.numpy()[tri_a:tri_b]
                self.ham_tri_idx = wp.array(
                    tri_np.reshape(-1).astype(np.int32),
                    dtype=wp.int32,
                    device=model.device,
                )
        except Exception:
            self.ham_tri_idx = None

        # Wok free-joint slot lookup — AUTO-FLIP only (the controller in
        # on_post_step writes joint_q here).  PARKED / DYNAMIC don't need it:
        # parked stays at its home pose, dynamic is driven by the solver.
        self._wok_q_start = -1
        self._wok_qd_start = -1
        if self.auto_flip and self.pan_id >= 0:
            # The wok's FREE joint is the unique one whose child == self.pan_id.
            try:
                jc = model.joint_child.numpy() if hasattr(model.joint_child, "numpy") else np.asarray(model.joint_child)
                jq_start = (
                    model.joint_q_start.numpy()
                    if hasattr(model.joint_q_start, "numpy")
                    else np.asarray(model.joint_q_start)
                )
                jqd_start = (
                    model.joint_qd_start.numpy()
                    if hasattr(model.joint_qd_start, "numpy")
                    else np.asarray(model.joint_qd_start)
                )
                hits = np.where(jc == int(self.pan_id))[0]
                if len(hits) > 0:
                    j = int(hits[0])
                    self._wok_q_start = int(jq_start[j])
                    self._wok_qd_start = int(jqd_start[j])
                    ctx.logger.info(
                        f"[chef] wok free-joint body={self.pan_id} joint={j} "
                        f"q_start={self._wok_q_start} qd_start={self._wok_qd_start}"
                    )
                else:
                    ctx.logger.warn(f"[chef] couldn't find wok joint (no joint_child=={self.pan_id})")
            except Exception as exc:
                ctx.logger.warn(f"[chef] wok joint slot lookup failed: {exc!r}")

    def _write_wok_pose(self, state, pos, quat):
        """Splice a (pos, quat) world pose into the wok free-joint slot.

        Newton's free-joint convention is [px, py, pz, qx, qy, qz, qw]; MJW
        propagates joint_q → body_q via FK.  joint_qd is zeroed so a stale
        finalize velocity can't leak into MJW's actuator equations.
        """
        jq = state.joint_q.numpy()
        s = self._wok_q_start
        jq[s + 0] = float(pos[0])
        jq[s + 1] = float(pos[1])
        jq[s + 2] = float(pos[2])
        jq[s + 3] = float(quat[0])
        jq[s + 4] = float(quat[1])
        jq[s + 5] = float(quat[2])
        jq[s + 6] = float(quat[3])
        state.joint_q.assign(jq)
        if self._wok_qd_start >= 0:
            jqd = state.joint_qd.numpy()
            qd_s = self._wok_qd_start
            jqd[qd_s + 0 : qd_s + 6] = 0.0
            state.joint_qd.assign(jqd)

    def on_post_step(self, state, sim_time, dt, ctx):
        """Per-frame startup-hand and optional wok-controller hook.

        * STARTUP HAND: publishes a finite smooth target ramp, then stops.
        * PARKED / DYNAMIC wok: no pose write; physics owns the wok body.
        * AUTO-FLIP: the autonomous controller writes the wok free-joint
          ``joint_q`` each frame to drive the SETTLE→GATHER→TOSS→CATCH→DWELL cycle.
        """
        if self._startup_hand_pub is not None:
            self._startup_hand_pub.publish(float(sim_time))

        if not self.auto_flip:
            return  # PARKED / DYNAMIC — no wok controller update
        if self.pan_id < 0 or self._wok_q_start < 0 or self.ctrl is None:
            return
        try:
            if self.food is not None and getattr(state, "particle_q", None) is not None:
                if state.particle_q.shape[0] > 0:
                    self.food.update(state.particle_q, dt)
            pos, quat = self.ctrl.update(self.food, dt)
            self._write_wok_pose(state, pos, quat)
        except Exception as exc:
            ctx.logger.warn(f"[chef] on_post_step failed: {exc!r}")

    def record_snapshot_enabled(self):
        """Copy particles only while the configured food publisher is live."""
        return self._food_state_pub is not None

    def on_record_snapshot(self, particle_positions, body_xyzwxyz, body_paths, sim_time, ctx):
        """Publish the same delayed physics snapshot used by ``/tf_scene``."""
        if self._food_state_pub is None:
            return
        try:
            robot_root = f"/{ctx.robot_prefix}/"
            keep = [
                index
                for index, path in enumerate(body_paths)
                if path != f"/{ctx.robot_prefix}" and not str(path).startswith(robot_root)
            ]
            scene_poses = np.asarray(body_xyzwxyz)[keep]
            scene_paths = [body_paths[index] for index in keep]
            self._food_state_pub.publish(
                particle_positions,
                scene_poses,
                scene_paths,
                float(sim_time),
            )
        except Exception as exc:
            ctx.logger.warn(f"[chef] food state publish failed: {exc!r}; disabling publisher")
            self._food_state_pub = None

    # ------------------------------------------------------------------
    # Render hook
    # ------------------------------------------------------------------
    def _build_tube_groups(self, model):
        if self._render_ready:
            return
        dev = model.device
        groups: list[dict] = []
        r = self.cfg.render

        def add_group(name, start, K, n_len, length, cross, color, rscale):
            if K <= 0:
                return
            M = n_len + 1
            cr_W = _catmull_rom_weights(M, r.vis_sub)
            out_pts = cr_W.shape[0]
            V = out_pts * r.n_theta + 2
            groups.append(
                {
                    "name": name,
                    "start": int(start),
                    "K": int(K),
                    "M": M,
                    "out_pts": out_pts,
                    "radius": rscale * 0.5 * cross,
                    "color": color,
                    "cr_W_dev": wp.array(cr_W, dtype=float, device=dev),
                    "centroids_dev": wp.empty(K * M, dtype=wp.vec3, device=dev),
                    "tube_idx": wp.array(_tube_indices(K, out_pts, r.n_theta), dtype=wp.int32, device=dev),
                    "smooth_dev": wp.empty(K * out_pts, dtype=wp.vec3, device=dev),
                    "verts": wp.empty(K * V, dtype=wp.vec3, device=dev),
                    "norms": wp.empty(K * V, dtype=wp.vec3, device=dev),
                }
            )

        add_group(
            "chef_noodles",
            0,
            self.noodle.n_noodles,
            self.n_len_noodle,
            self.noodle.length,
            self.noodle.cross,
            self.noodle.color,
            # NOTE: rscale is multiplied ONCE here (matches the demo's noodle
            # group, which passes vis_radius=1.15 directly).  Was
            # ``vis_radius_scale * r.vis_radius`` (1.15*1.15=1.3225) — that
            # double-count gave a 0.00172 m render tube, 32% fatter than the
            # 0.0013 m physical contact radius, so noodles that physically only
            # touch rendered as interpenetrating.  pepper/carrot already pass
            # the bare vis_radius_scale below, so only noodles had the bug.
            self.noodle.vis_radius_scale,
        )
        add_group(
            "chef_pepper",
            self.pepper_pstart,
            self.pepper_render_count,
            self.pepper.n_len,
            self.pepper.length,
            self.pepper.cross,
            self.pepper.color,
            self.pepper.vis_radius_scale,
        )
        add_group(
            "chef_carrot",
            self.carrot_pstart,
            self.carrot_render_count,
            self.carrot.n_len,
            self.carrot.length,
            self.carrot.cross,
            self.carrot.color,
            self.carrot.vis_radius_scale,
        )
        self.tube_groups = groups
        self._render_ready = True

    def on_render(self, viewer, state, sim_time, ctx):
        # Lazy-build tube groups on the first render call so we have a
        # live model device handle.
        try:
            model = getattr(state, "model", None)
            # state doesn't always carry .model; pull device from a known
            # array on state instead.
            pq = state.particle_q
            if pq is None:
                return
            dev = pq.device
            if not self._render_ready:
                # Synthesize a minimal model-shim with .device for builder.
                class _D:
                    pass

                d = _D()
                d.device = dev
                self._build_tube_groups(d)

            for g in self.tube_groups:
                K, M = g["K"], g["M"]
                wp.launch(
                    _ring_centroids, dim=K * M, inputs=[pq, g["start"], M], outputs=[g["centroids_dev"]], device=dev
                )
                wp.launch(
                    _catmull_smooth,
                    dim=K * g["out_pts"],
                    inputs=[g["centroids_dev"], M, g["cr_W_dev"], g["out_pts"]],
                    outputs=[g["smooth_dev"]],
                    device=dev,
                )
                wp.launch(
                    _tube_kernel,
                    dim=K,
                    inputs=[g["smooth_dev"], g["out_pts"], self.cfg.render.n_theta, g["radius"]],
                    outputs=[g["verts"], g["norms"]],
                    device=dev,
                )
                viewer.log_mesh(
                    "/" + g["name"],
                    g["verts"],
                    g["tube_idx"],
                    normals=g["norms"],
                    color=g["color"],
                    backface_culling=False,
                )
            if self.ham_tri_idx is not None:
                viewer.log_mesh(
                    "/chef_ham",
                    pq,
                    self.ham_tri_idx,
                    color=self.ham.color,
                    backface_culling=True,
                )
            # Hide the native FEM surface only after every custom draw above
            # succeeds. If Warp compilation/rendering fails, native particles
            # remain visible instead of leaving an empty pan.
            if not getattr(self, "_native_render_hidden", False):
                for _attr in ("show_triangles", "show_particles"):
                    if hasattr(viewer, _attr):
                        setattr(viewer, _attr, False)
                self._native_render_hidden = True
        except Exception as exc:
            # Render failures shouldn't kill the engine; warn once.
            if not getattr(self, "_render_warned", False):
                ctx.logger.warn(f"[chef] on_render failed: {exc!r}")
                self._render_warned = True
