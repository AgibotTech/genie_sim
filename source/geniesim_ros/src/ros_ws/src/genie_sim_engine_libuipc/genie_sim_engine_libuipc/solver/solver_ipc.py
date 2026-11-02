# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import tempfile
import weakref
from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp
from newton import (
    Axis,
    BodyFlags,
    Contacts,
    Control,
    GeoType,
    JointType,
    Mesh,
    Model,
    ModelFlags,
    ShapeFlags,
    State,
)
from newton.solvers import SolverBase

_SUPPORTED_RIGID_GEO_TYPES = (
    GeoType.BOX,
    GeoType.SPHERE,
    GeoType.CAPSULE,
    GeoType.MESH,
    GeoType.CONVEX_MESH,
)


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert an ``[x, y, z, w]`` quaternion to a 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _matrix_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an ``[x, y, z, w]`` quaternion (Shepperd's method)."""
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w])
    return quat / np.linalg.norm(quat)


def _weld_vertices(vertices: np.ndarray, faces: np.ndarray, eps: float = 1.0e-6) -> tuple[np.ndarray, np.ndarray]:
    """Merge duplicate vertices (within ``eps``) so ``libuipc`` sees a closed, two-manifold trimesh.

    Newton's tessellated primitives (:meth:`~newton.Mesh.create_sphere`, etc.) duplicate
    vertices at UV seams for correct texturing/normals; ``libuipc``'s ``AffineBodyConstitution``
    requires a watertight mesh to compute enclosed volume.
    """
    key = np.round(vertices / eps).astype(np.int64)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    welded = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(welded, inverse, vertices)
    welded /= counts[:, None]
    welded_faces = inverse[faces.reshape(-1)].reshape(faces.shape)
    degenerate = (
        (welded_faces[:, 0] == welded_faces[:, 1])
        | (welded_faces[:, 1] == welded_faces[:, 2])
        | (welded_faces[:, 0] == welded_faces[:, 2])
    )
    return welded, welded_faces[~degenerate].astype(np.int32)


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)


def _tet_boundary_faces(tet_indices: np.ndarray) -> np.ndarray:
    """Return the tet mesh's boundary faces: triangles that belong to exactly one tet.

    Mirrors the face-cancellation logic in :meth:`ModelBuilder.add_soft_grid`: each tet
    contributes four outward-oriented triangular faces, and a face shared by two adjacent
    tets (i.e. appearing twice) is interior and excluded from the boundary.

    Args:
        tet_indices: Tetrahedron vertex indices, shape ``[tet_count, 4]``.

    Returns:
        Boundary face vertex indices, shape ``[boundary_face_count, 3]``.
    """
    i, j, k, l = (
        tet_indices[:, 0],
        tet_indices[:, 1],
        tet_indices[:, 2],
        tet_indices[:, 3],
    )
    faces = np.concatenate(
        [
            np.stack([i, k, j], axis=1),
            np.stack([j, k, l], axis=1),
            np.stack([i, j, l], axis=1),
            np.stack([i, l, k], axis=1),
        ],
        axis=0,
    )
    keys = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    is_boundary = counts[inverse.reshape(-1)] == 1
    return faces[is_boundary]


def _classify_tet_surface_triangles(tri_indices: np.ndarray, boundary_faces: np.ndarray) -> np.ndarray:
    """Match ``tri_indices`` rows against the tet mesh's boundary faces (see :func:`_tet_boundary_faces`).

    Every boundary face implied by ``tet_indices`` must appear in ``tri_indices`` exactly
    once (as the same three vertices, in any order); a missing or duplicated boundary face
    is a malformed surface mesh and raises ``ValueError``. Rows of ``tri_indices`` unrelated
    to any tet boundary face (e.g. cloth triangles in a mixed model) are ignored.

    Args:
        tri_indices: Surface triangle vertex indices, shape ``[tri_count, 3]``.
        boundary_faces: Tet boundary face vertex indices, shape ``[boundary_face_count, 3]``.

    Returns:
        Boolean mask, shape ``[tri_count]``, marking the rows of ``tri_indices`` that are
        tet surface triangles.
    """
    boundary_keys = [tuple(row) for row in np.sort(boundary_faces, axis=1)]
    boundary_key_set = set(boundary_keys)
    tri_keys = [tuple(row) for row in np.sort(tri_indices, axis=1)] if tri_indices.shape[0] > 0 else []

    matches: dict[tuple[int, int, int], list[int]] = {}
    for row, key in enumerate(tri_keys):
        matches.setdefault(key, []).append(row)

    missing = [key for key in boundary_keys if key not in matches]
    if missing:
        raise ValueError(
            f"Model.tri_indices is missing {len(missing)} tet surface triangle(s) implied by "
            f"Model.tet_indices (e.g. vertices {missing[0]})."
        )

    duplicated = {key: rows for key, rows in matches.items() if key in boundary_key_set and len(rows) > 1}
    if duplicated:
        key, rows = next(iter(duplicated.items()))
        raise ValueError(f"Model.tri_indices duplicates tet surface triangle {key} at rows {rows}.")

    mask = np.zeros(len(tri_keys), dtype=bool)
    for key in boundary_keys:
        mask[matches[key][0]] = True
    return mask


def _tet_connected_components(tet_indices: np.ndarray, particle_count: int) -> np.ndarray:
    """Label each particle referenced by ``tet_indices`` with a stable connected-component id.

    Particles transitively connected through shared tet vertices receive the same id.
    Components are numbered in ascending order of their smallest member particle index,
    so the labeling is deterministic and independent of tet iteration order. Implemented
    as a plain union-find (no ``scipy`` dependency).

    Args:
        tet_indices: Tetrahedron vertex indices, shape ``[tet_count, 4]``.
        particle_count: Total number of particles in the model.

    Returns:
        Component id per particle, shape ``[particle_count]``; ``-1`` for particles not
        referenced by any tet.
    """
    parent = np.arange(particle_count, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Always attach the larger root to the smaller one: the surviving root ends up
            # equal to the smallest particle index in the component, which is what makes
            # the final labeling below stable/deterministic.
            parent[max(ra, rb)] = min(ra, rb)

    referenced = np.zeros(particle_count, dtype=bool)
    for tet in tet_indices:
        i, j, k, l = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
        referenced[i] = referenced[j] = referenced[k] = referenced[l] = True
        union(i, j)
        union(i, k)
        union(i, l)

    component_id = np.full(particle_count, -1, dtype=np.int64)
    root_to_label: dict[int, int] = {}
    for p in np.flatnonzero(referenced):
        root = find(int(p))
        label = root_to_label.setdefault(root, len(root_to_label))
        component_id[p] = label
    return component_id


def _animate_kinematic_body(solver_ref: weakref.ReferenceType, body: int, info: Any) -> None:
    """``libuipc`` ``Animator`` callback: pin an externally-driven ABD body to its per-step target pose.

    Bound via a weak reference to the owning :class:`SolverIPC`, not a direct closure: `libuipc`'s
    native ``Animator`` cannot be traversed by Python's cyclic garbage collector, so a strong
    reference here would keep the solver alive indefinitely.
    """
    solver = solver_ref()
    if solver is None:
        return
    uipc = solver._uipc
    geom = info.geo_slots()[0].geometry()
    uipc.view(geom.instances().find(uipc.builtin.is_constrained))[0] = 1
    uipc.view(geom.instances().find(uipc.builtin.aim_transform))[:] = solver._kinematic_targets[body]


def _animate_articulation_joint(solver_ref: weakref.ReferenceType, slot_idx: int, info: Any) -> None:
    """``libuipc`` ``Animator`` callback: for a revolute/prismatic driving joint,
    write ``aim_angle`` and ``driving/is_constrained=1`` on the joint's edge each
    step. Attribute names verified in Task 2 POC (``driving/is_constrained`` is on
    the edge, not the instances, and namespaced under ``driving/``). Bound via weak
    reference for the same GC reason as :func:`_animate_kinematic_body`."""
    solver = solver_ref()
    if solver is None:
        return
    slot = solver._articulation_slots[slot_idx]
    if slot["kind"] == "fixed":
        return  # No drive on a fixed joint.
    uipc = solver._uipc
    geom = info.geo_slots()[0].geometry()
    aim = solver._joint_aim_values[slot_idx]
    is_constrained = geom.edges().find("driving/is_constrained")
    if is_constrained is not None:
        uipc.view(is_constrained)[:] = 1
    aim_attr = geom.edges().find("aim_angle")
    if aim_attr is not None:
        uipc.view(aim_attr)[:] = aim


def _quat_rotate_vec(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector ``vec`` by a quaternion ``(qx, qy, qz, qw)``.

    Newton stores :class:`wp.transform` orientations as ``(qx, qy, qz, qw)``.
    Used to lift joint-frame axis vectors (as they appear in ``Model.joint_axis``)
    into the parent- or child-body-local frames that libuipc's
    ``AffineBodyRevoluteJoint.create_geometry`` local API expects for
    ``l_position1 - l_position0`` and ``r_position1 - r_position0``."""
    x, y, z, w = (
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
        float(quat_xyzw[3]),
    )
    v = np.asarray(vec, dtype=np.float64)
    return np.array(
        [
            (1.0 - 2.0 * (y * y + z * z)) * v[0] + 2.0 * (x * y - z * w) * v[1] + 2.0 * (x * z + y * w) * v[2],
            2.0 * (x * y + z * w) * v[0] + (1.0 - 2.0 * (x * x + z * z)) * v[1] + 2.0 * (y * z - x * w) * v[2],
            2.0 * (x * z - y * w) * v[0] + 2.0 * (y * z + x * w) * v[1] + (1.0 - 2.0 * (x * x + y * y)) * v[2],
        ],
        dtype=np.float64,
    )


def _animate_ref_dof_prev(solver_ref: weakref.ReferenceType, body: int, info: Any) -> None:
    """``libuipc`` ``Animator`` callback (EAC mode only): refresh an ABD instance's
    ``ref_dof_prev`` attribute from its current transform, so ExternalArticulation-
    Constraint's kinetic term sees a valid prior configuration each step.

    Follows the pattern in ``libuipc/python/examples/eac_probe.py``'s
    ``refresh_arm_ref_dof_prev``: read ``geom.transforms()``, convert to the
    12-vector affine-body q via ``affine_body.transform_to_q``, and write into
    ``ref_dof_prev`` (shape ``[num_instances, 12, 1]``). Bound weakly for the same
    GC reason as :func:`_animate_kinematic_body`."""
    solver = solver_ref()
    if solver is None:
        return
    uipc = solver._uipc
    del body  # only one instance per ABD mesh in the current SolverIPC layout
    geom = info.geo_slots()[0].geometry()
    transforms = np.asarray(geom.transforms().view()).astype(np.float64)
    q = np.asarray(uipc.geometry.affine_body.transform_to_q(transforms)).reshape(-1, 12)
    rdp = geom.instances().find("ref_dof_prev")
    if rdp is None:
        return
    arr = np.asarray(uipc.view(rdp))
    arr[..., 0] = q


def _animate_eac_delta_theta(solver_ref: weakref.ReferenceType, info: Any) -> None:
    """``libuipc`` ``Animator`` callback (EAC mode only): write per-step commanded
    angular increments ``delta_theta_tilde`` for every joint in the articulation
    from the solver's ``_eac_delta_theta_tilde`` scratch list.

    The animator is inserted on the single ExternalArticulationConstraint object
    that covers the entire articulation (matches the coupled-mass-matrix design in
    ``libuipc/apps/tests/sim_case/47_external_articulation_multijoints.cpp``);
    ``info.geo_slots()[0].geometry()`` returns the articulation geometry, whose
    ``joint`` sub-collection has ``n_joints`` per-joint entries and whose
    ``joint_joint`` sub-collection carries the ``n_joints x n_joints`` mass matrix.

    Values in ``_eac_delta_theta_tilde`` are indexed by ``_articulation_slots``
    slot index; ``_eac_slot_index_map`` maps EAC joint position → slot index so
    only the revolute/prismatic entries reach ``delta_theta_tilde``."""
    solver = solver_ref()
    if solver is None:
        return
    uipc = solver._uipc
    geom = info.geo_slots()[0].geometry()
    dtt = geom["joint"].find("delta_theta_tilde")
    if dtt is None:
        return
    dtt_view = np.asarray(uipc.view(dtt))
    for eac_pos, slot_idx in enumerate(solver._eac_slot_index_map):
        dtt_view[eac_pos] = solver._eac_delta_theta_tilde[slot_idx]


class SolverIPC(SolverBase):
    """Rigid-body, cloth, and tetrahedral FEM soft-body solver backed by
    `libuipc <https://github.com/spirimirror/libuipc>`_'s Incremental Potential Contact
    (IPC) engine, via the ``pyuipc`` bindings.

    Rigid bodies use ``libuipc``'s Affine Body Dynamics (ABD) formulation. Cloth
    (particles connected by :attr:`Model.tri_indices`) uses ``libuipc``'s
    ``NeoHookeanShell`` (stable Neo-Hookean membrane) with elastic, stress-plastic,
    or strain-plastic discrete shell bending FEM constitutions. Tetrahedral FEM soft
    bodies (:attr:`Model.tet_indices`) use ``libuipc``'s ``StableNeoHookean``
    constitution. Contact and friction among rigid bodies, ground (world-attached
    plane), cloth, tet-FEM soft bodies, and self-contact are handled entirely by
    ``libuipc``'s own barrier-based collision pipeline; the ``contacts`` argument to
    :meth:`step` is ignored, mirroring how :class:`~newton.solvers.SolverMuJoCo` ignores
    Newton contacts when ``use_mujoco_contacts=True``.

    Main's ``SolverIPC`` extension interprets URDF joints in ``Model``:
    :class:`~newton.JointType.REVOLUTE`, :class:`~newton.JointType.PRISMATIC`
    and :class:`~newton.JointType.FIXED` joints are lifted into native ``libuipc``
    ``AffineBodyRevoluteJoint`` / ``AffineBodyPrismaticJoint`` / ``AffineBodyFixedJoint``
    constraints plus their driving counterparts. Each ``step()`` reads
    ``Control.joint_target_q`` as the position target for every revolute/prismatic
    DoF; the resulting joint angles are written back to ``state.joint_q`` after
    each step. Bodies not covered by URDF joints (loose props, ground, hand-scripted
    kinematic targets) still follow the legacy :class:`~newton.BodyFlags.KINEMATIC` ->
    ``state_in.body_q`` path.

    Not supported: :class:`~newton.JointType.BALL`, :class:`~newton.JointType.FREE`,
    :class:`~newton.JointType.D6`, :class:`~newton.JointType.CABLE`. FREE joints are
    silently skipped (ABD implicitly permits 6-DoF free motion); the others raise
    ``NotImplementedError`` at construction time.

    Optional per-body contact filtering: pass ``gripper_link_names`` (with matching
    ``urdf_body_prefix``) to route non-gripper URDF links to a ``no_contact`` element
    that is disabled against everything else. Selected static fixtures can remain
    collidable with deformables but ignore the gripper through
    ``gripper_noncolliding_static_shape_labels``. See
    ``basic.example_ipc_g2_pico_teleop_eac_fold_paper`` for the fold-paper teleop wiring.

    Limitations (v1):
        * :class:`~newton.BodyFlags.KINEMATIC` bodies are represented as ABD rigid bodies
          pinned to a per-step target pose read from ``state_in.body_q`` (the only exception
          to the "no two-way coupling" rule below), enforced via ``libuipc``'s
          ``SoftTransformConstraint`` (a soft penalty, not a hard constraint: large contact
          forces can pull a kinematic body off its target). This path is used for bodies not
          covered by URDF joints; kinematic body targets are computed externally (e.g. via
          :func:`newton.eval_fk`).
        * At most one collision shape per :class:`~newton.BodyFlags.DYNAMIC` rigid body.
          Kinematic bodies may have any number of shapes (merged into one mesh) or none
          (skipped entirely), and do not require positive :attr:`Model.body_mass`.
        * Supported shape types: :attr:`~newton.GeoType.BOX`, :attr:`~newton.GeoType.SPHERE`,
          :attr:`~newton.GeoType.CAPSULE`, :attr:`~newton.GeoType.MESH`,
          :attr:`~newton.GeoType.CONVEX_MESH`. A world-attached (``body=-1``)
          :attr:`~newton.GeoType.PLANE` shape maps to a ``libuipc`` ground plane.
        * Joints, actuation, and multi-world (vectorized) models are not supported.
        * Tet-FEM soft bodies (:attr:`Model.tet_indices`) use ``libuipc``'s
          ``StableNeoHookean`` constitution. :attr:`Model.tet_materials` columns 0--1
          are the original Lamé parameters ``mu``/``lambda`` (``k_mu``/``k_lambda``);
          ``StableNeoHookean`` reparameterizes them internally
          (``mu' = 4/3 * mu``, ``lambda' = lambda + 5/6 * mu``; Smith et al.,
          *Stable Neo-Hookean Flesh Simulation*, ACM TOG 2018). Per-tet values are
          preserved when the component carries heterogeneous materials. Column 2
          (``k_damp``, damping) has no ``libuipc`` equivalent and is ignored. Density
          is not stored on the Newton model; per connected component, it is recovered
          as ``sum(unique particle mass) / sum(reference tet volume)`` (equivalent
          uniform density). Fixed particles are identified by
          :attr:`Model.particle_inv_mass` ``== 0``. A tet-FEM soft body's own surface
          triangles are excluded from the cloth shell; an independent cloth mesh may
          coexist with tet-FEM soft bodies in the same model, on disjoint particle sets.
          If :attr:`Model.tri_count` is zero (no triangles at all), ``libuipc`` derives
          each tet component's collision surface directly from its tet topology and
          :attr:`Model.tri_indices` is not required; once any triangle exists, every tet
          boundary face must appear in :attr:`Model.tri_indices` exactly once (a partial
          surface is not supported).
        * Nonzero :attr:`Model.tet_activations` at construction time and nonzero
          ``control.tet_activations`` at runtime both raise
          :class:`NotImplementedError`; runtime tet activation is not yet supported.
        * Independent per-material density and per-material friction are not supported;
          density is derived per connected component and friction is solver-wide.
        * Cloth shell thickness is a single solver-wide value (``cloth_thickness``),
          not derived per-particle from :attr:`Model.particle_radius`.
        * ``cloth_self_collision`` also controls tet-FEM self-collision.
        * The tet/cloth mesh topology is fixed at construction time. Cutting, tearing,
          and remeshing are not supported; :meth:`notify_model_changed` raises
          :class:`NotImplementedError`.
        * Rigid-body mass is matched exactly; the inertia distribution follows the
          collision shape's geometry (assuming uniform density) rather than
          :attr:`Model.body_inertia`.
        * Particles must form at least one triangle (:attr:`Model.tri_indices`) or
          tetrahedron (:attr:`Model.tet_indices`); bare particle clouds with no FEM
          mesh are not supported.
        * Cloth material mapping: :attr:`Model.tri_materials`' ``k_mu``/``k_lambda``
          columns map directly to ``libuipc``'s per-triangle Lamé ``mu``/``lambda``,
          and :attr:`Model.edge_bending_properties`' stiffness column maps directly to
          ``libuipc``'s per-edge bending stiffness. Damping (``tri_materials``'
          ``k_damp``, ``edge_bending_properties``' damping column) and aerodynamic
          drag/lift (``k_drag``, ``k_lift``) have no ``libuipc`` FEM-shell equivalent
          and are ignored.
        * Cloth and tet-FEM pinning follows :attr:`Model.particle_inv_mass`
          (``== 0`` pins a particle); :attr:`Model.particle_flags` is not consulted.
        * No cloth-to-rigid-body attachment/sewing: cloth interacts with rigid bodies
          and other cloth purely through ``libuipc``'s barrier contact pipeline.
        * All colliding geometry (rigid, static, cloth, and tet-FEM) must have strictly
          positive initial clearance at construction time, or ``libuipc``'s
          ``world.init()`` silently leaves the world in an invalid state.
        * :meth:`step` is not compatible with CUDA graph capture: ``libuipc``'s engine
          performs its own internal stream synchronization.
    """

    _uipc = None

    @classmethod
    def import_pyuipc(cls):
        """Import the ``pyuipc`` bindings for ``libuipc`` and cache the module."""
        if cls._uipc is None:
            try:
                import uipc

                cls._uipc = uipc
            except ImportError as e:
                raise ImportError(
                    "The Main IPC runtime could not load pyuipc. Run "
                    "`geniesim bootstrap --libuipc-only` from the Main checkout "
                    "using Linux x86_64 CPython 3.12; inspect the chained native-library error."
                ) from e
        return cls._uipc

    def __init__(
        self,
        model: Model,
        *,
        dt: float,
        contact_d_hat: float = 0.01,
        friction_enabled: bool = True,
        friction_rate: float | None = None,
        gripper_link_names: Sequence[str] | None = None,
        gripper_self_contact: bool = True,
        gripper_noncolliding_static_shape_labels: Sequence[str] | None = None,
        urdf_body_prefix: str = "",
        skip_body_labels: Sequence[str] | None = None,
        abd_kappa: float = 1.0e8,
        abd_kappa_overrides: dict[str, float] | None = None,
        cloth_thickness: float = 1.0e-3,
        cloth_self_collision: bool = True,
        cloth_bending_yield_stress: float | None = None,
        cloth_bending_yield_angle: float | None = None,
        cloth_bending_hardening_modulus: float = 0.0,
        kinematic_constraint_strength: tuple[float, float] = (100.0, 100.0),
        use_external_articulation: bool = False,
        external_articulation_mass: float = 1.0,
        external_articulation_mass_overrides: dict[str, float] | None = None,
        articulation_pin_stiffness: float = 1.0e6,
        articulation_drive_stiffness: float = 1.0e3,
        linear_system_solver: str | None = None,
        linear_system_tol_rate: float | None = None,
        linear_system_check_interval: int | None = None,
        linear_system_graph_enable: bool | None = None,
        linear_system_fused_preconditioner_enable: bool | None = None,
        abd_full_block_preconditioner: bool = False,
        collision_detection_method: str | None = None,
        newton_max_iter: int | None = None,
        newton_ccd_tol: float | None = None,
        newton_velocity_tol: float | None = None,
        newton_transrate_tol: float | None = None,
        newton_semi_implicit: bool | None = None,
        newton_semi_implicit_beta_tol: float | None = None,
        fem_mas_partition_size: int | None = None,
        workspace: str | None = None,
    ):
        """
        Args:
            model: The model to be simulated. See the class-level Limitations for
                which parts of ``model`` are supported.
            dt: The fixed simulation time step [s] this solver is configured for.
                Every call to :meth:`step` must use this same ``dt``.
            contact_d_hat: Barrier activation distance [m] for contact.
            friction_enabled: If ``True``, enable ``libuipc``'s friction model.
            friction_rate: Optional override for ``libuipc``'s default contact friction
                coefficient (Coulomb-like ``mu``). If ``None``, ``libuipc``'s built-in
                default (``0.5``) is kept. Ignored when ``friction_enabled=False``.
            gripper_link_names: Optional set of link names allowed to collide with
                deformables. When provided, every body in the ``urdf_body_prefix``
                namespace whose prefix-stripped label is not in this list is assigned
                a ``no_contact`` element disabled against all other elements. Bodies
                outside a nonempty prefix remain on the default contact element.
                ``None`` (default) keeps legacy behavior (every body on the default
                element).
            gripper_self_contact: If ``True`` (default) the gripper links (selected by
                ``gripper_link_names``) sit on the same default contact element and
                therefore collide with each other, matching the legacy behavior. If
                ``False`` they move to a dedicated element with a
                ``gripper ↔ gripper`` insert rule that disables self-collision --
                the gripper can still touch cloth and other default-element bodies
                (default enable inherits), but its own fingers no longer barrier
                against each other, so a parallel-jaw drive can close without the
                pin-adjacent finger overlap fighting the closure. Ignored when
                ``gripper_link_names`` is ``None``.
            gripper_noncolliding_static_shape_labels: Optional static shape labels
                that remain collidable with deformables but do not collide with the
                gripper links selected by ``gripper_link_names``. This can remove
                redundant gripper-deformable-fixture contact loops while preserving
                gripper-deformable and deformable-fixture contact. Requires
                ``gripper_link_names``.
            urdf_body_prefix: Prefix stripped from ``model.body_label`` before comparing
                against ``gripper_link_names``. Use ``""`` (default) when
                ``gripper_link_names`` already holds fully-qualified labels.
            abd_kappa: Default Affine Body Dynamics stiffness parameter used for rigid bodies.
            abd_kappa_overrides: Optional per-body ABD stiffness overrides. Keys are
                entries from ``model.body_label``. Bodies not present in the mapping
                use ``abd_kappa``.
            cloth_thickness: Shell thickness [m] used for every cloth triangle.
            cloth_self_collision: If ``True``, enable cloth and tet-FEM self-collision.
            cloth_bending_yield_stress: Optional yield stress [kPa] passed to libuipc's
                stress-plastic discrete shell bending model. Mutually exclusive with
                ``cloth_bending_yield_angle``.
            cloth_bending_yield_angle: Optional dihedral-angle deviation [rad] passed
                directly to libuipc's strain-plastic discrete shell bending model.
                Mutually exclusive with ``cloth_bending_yield_stress``. If neither yield
                option is provided, use elastic bending.
            cloth_bending_hardening_modulus: Isotropic hardening modulus passed to the
                selected plastic bending model: [kPa] for stress plasticity and
                dimensionless for angle plasticity. Must be zero when plastic bending
                is disabled.
            kinematic_constraint_strength: ``(eta_p, eta_a)`` strength ratio passed to
                ``libuipc``'s ``SoftTransformConstraint``, controlling how strongly
                :class:`~newton.BodyFlags.KINEMATIC` bodies track their per-step target pose.
                The constraint is soft by construction, so its Hessian block competes with
                the contact barrier: raising this past ``libuipc``'s documented ``[0, 100]``
                range makes the commanded pose infeasible while in contact and the Newton
                outer loop stops converging, which shows up as multi-second frames the
                moment a grasp forms rather than as a uniform slowdown. The extra tracking
                accuracy bought above ``100`` is on the order of tens of microns.
            use_external_articulation: When ``True``, drive URDF revolute/prismatic
                joints through ``libuipc``'s ``ExternalArticulationConstraint``
                (EAC) instead of ``AffineBodyDrivingRevoluteJoint`` /
                ``AffineBodyDrivingPrismaticJoint``. In this mode the caller writes
                a per-joint ``delta_theta_tilde`` (angular increment [rad] per step)
                into ``solver._eac_delta_theta_tilde`` before each :meth:`step`;
                :attr:`~newton.sim.Control.joint_target_q` is ignored. Every
                DYNAMIC ABD body in the supported articulation is flagged
                ``external_kinetic=1`` and given a ``ref_dof_prev`` instance
                attribute that the solver refreshes each frame from the body's
                current transform. Free rigid bodies retain their standard kinetic
                term. Intended for teleop setups that already run their own outer-loop PD.
            external_articulation_mass: Per-joint ``M^t`` scalar written into the
                EAC geometry's ``joint_joint/mass`` attribute (see
                ``libuipc/python/examples/eac_probe.py``). Only used when
                ``use_external_articulation=True``. Larger values make the joint
                harder to move; ``1.0`` is a reasonable starting point for a
                humanoid arm.
            external_articulation_mass_overrides: Optional per-joint override for
                ``external_articulation_mass``. Keys are ``model.joint_label``
                entries (URDF joint names); values are the ``M^t`` scalar for that
                joint's diagonal in ``joint_joint/mass``. Joints not present in
                the dict fall back to ``external_articulation_mass``. Use this
                to make a subset of joints stiffer -- e.g. bump the gripper's
                closing joints so the drive can overcome contact barriers against
                cloth without also stiffening the arm chain.
            articulation_pin_stiffness: Strength of the parent↔child pin penalty
                (``AffineBodyRevoluteJoint`` / ``AffineBodyPrismaticJoint`` /
                ``AffineBodyFixedJoint``). Passed directly as libuipc's
                ``strength_ratio``; internally multiplied by ``(m_parent + m_child)``.
                Too soft → joint anchors visibly separate under drive torque or
                gravity; too stiff → BDF1 instability. ``1e6`` was the original
                default, ``1e8`` typical for chains that must resist meaningful
                moment load. Applied to every URDF joint uniformly.
            articulation_drive_stiffness: Strength of the position-drive spring
                (``AffineBodyDrivingRevoluteJoint`` / ``AffineBodyDrivingPrismaticJoint``).
                Ignored when ``use_external_articulation=True`` (EAC drives the
                joint's DOF directly and no separate spring is registered).
            linear_system_solver: Optional override for ``libuipc``'s
                ``linear_system.solver`` config key (e.g. ``"fused_pcg"``, ``"linear_pcg"``).
                If ``None``, ``libuipc``'s own default is used.
            linear_system_tol_rate: Optional override for ``libuipc``'s
                ``linear_system.tol_rate`` config key -- the PCG convergence
                tolerance (larger => fewer PCG iterations, cheaper Newton
                iterations, slightly less accurate solve). libuipc's default is
                ``1e-3``; loosening to ``1e-2`` roughly halves PCG iteration
                counts and shaves ~30% off scenes where PCG dominates step time
                (see the timer breakdown for fold_paper / cloth_grid). ``None``
                keeps libuipc's default.
            linear_system_check_interval: Optional number of fused-PCG iterations
                between host convergence checks. Set to ``50`` for the B50
                configuration; ``None`` keeps libuipc's default.
            linear_system_graph_enable: Optional CUDA Graph toggle for fused PCG.
                ``None`` keeps libuipc's default.
            linear_system_fused_preconditioner_enable: Optional fused-PCG
                preconditioner toggle. ``None`` keeps libuipc's default.
            abd_full_block_preconditioner: If ``True``, cache a dense inverse of
                the first assembled ABD block and use it as the rigid-body PCG
                preconditioner. This captures cross-link articulation coupling
                while PCG continues to converge against the current matrix.
                Intended for EAC chains; the default keeps 12x12 block Jacobi.
            collision_detection_method: Optional override for ``libuipc``'s
                ``collision_detection.method`` config key. If ``None``, ``libuipc``'s
                own default is used.
            newton_max_iter: Optional override for ``libuipc``'s ``newton.max_iter``
                config key, capping the outer Newton iterations per step. Lower
                values trade convergence accuracy for a bounded per-step cost; IPC's
                incremental time-stepping absorbs an under-converged step in the
                following ones as long as CCD/line-search still runs. If ``None``,
                ``libuipc``'s own default is used.
            newton_ccd_tol: Optional override for ``libuipc``'s ``newton.ccd_tol``
                config key [dimensionless], the smallest CCD step fraction that still
                counts as converged. ``libuipc``'s default of ``1.0`` requires the
                Newton step to be entirely unrestricted by continuous collision
                detection, which a sustained contact (grasp, resting object) can never
                satisfy: its time-of-impact goes to zero as the gap closes, so the
                outer loop keeps iterating up to ``newton_max_iter`` even after every
                displacement tolerance is met. Lowering this lets such steps exit early,
                but the accepted step is then CCD-truncated, and the residual carried
                into later steps has measured worse overall than the extra iterations it
                saves. Prefer softening ``kinematic_constraint_strength``.
                If ``None``, ``libuipc``'s own default is used.
            newton_velocity_tol: Optional override for ``libuipc``'s
                ``newton.velocity_tol`` config key [m/s], scaled by ``dt`` into the
                per-step displacement tolerance checked against the largest
                axis-aligned vertex displacement. If ``None``, ``libuipc``'s own
                default is used.
            newton_transrate_tol: Optional override for ``libuipc``'s
                ``newton.transrate_tol`` config key [1/s], scaled by ``dt`` into the
                tolerance on the affine (rotation/scale/shear) components of each rigid
                body's incremental degrees of freedom. If ``None``, ``libuipc``'s own
                default is used.
            newton_semi_implicit: Optional override for ``libuipc``'s
                ``newton.semi_implicit.enable`` config key. When enabled, the outer
                Newton loop also terminates once its semi-implicit blending factor
                ``beta`` -- decayed by the line-search step fraction each iteration --
                falls below ``newton_semi_implicit_beta_tol``, independently of the
                displacement and CCD convergence checks. This is the practical answer
                to the stall described under ``newton_ccd_tol``: the same CCD
                truncation that prevents convergence there shrinks the step fraction,
                which decays ``beta`` faster, so the steps that would otherwise iterate
                to ``newton_max_iter`` are exactly the ones that exit soonest. The cost
                is accepting a less-converged step. Measured on the drawstring-bag
                teleop replay (887 frames), together with a
                ``newton_semi_implicit_beta_tol`` of 0.1: worst frame 3.4 s -> 182 ms,
                frames over 200 ms 94 -> 0, with cloth shape and grasp retention inside
                run-to-run nondeterminism. Do not combine with a loosened
                ``newton_velocity_tol``; measured worse than this alone. If ``None``,
                keep ``libuipc``'s default (disabled).
            newton_semi_implicit_beta_tol: Optional override for ``libuipc``'s
                ``newton.semi_implicit.beta_tol`` config key [dimensionless], the
                ``beta`` threshold below which ``newton_semi_implicit`` terminates the
                outer loop. Ignored unless ``newton_semi_implicit`` is enabled. Raising
                it trades accuracy for a tighter bound on the worst frame and must be
                validated against contact retention for each scene: a value large
                enough to fire before contact is fully established will drop a grasped
                object -- on the teleop replay the contact window collapses from 185
                frames to 35 between 0.15 and 0.2. If ``None``, keep ``libuipc``'s
                default (``1e-3``).
            fem_mas_partition_size: If set, partition every FEM mesh (cloth and tet)
                with METIS into parts of at most this many vertices, which switches
                ``libuipc`` from its diagonal (Jacobi) FEM preconditioner to the
                multilevel additive Schwarz one. Jacobi conditions the linear system
                poorly once contact barriers activate -- measured PCG iteration counts
                rise from ~10 on contact-free steps to several hundred on grasping
                ones, and that gap, not the outer Newton iteration count, is what
                produces latency spikes. ``16`` is ``libuipc``'s suggested size.
                Must be in ``[1, 16]``: the engine stores each part in a fixed
                16-node bank. ``None`` keeps the diagonal preconditioner.
            workspace: Optional directory used by the ``libuipc`` engine for checkpoint
                dumps. If ``None``, a temporary directory is created.
        """
        super().__init__(model)

        if cloth_bending_yield_stress is not None and cloth_bending_yield_stress <= 0.0:
            raise ValueError("cloth_bending_yield_stress must be greater than 0.0 when provided.")
        if cloth_bending_yield_angle is not None and cloth_bending_yield_angle <= 0.0:
            raise ValueError("cloth_bending_yield_angle must be greater than 0.0 when provided.")
        if cloth_bending_yield_stress is not None and cloth_bending_yield_angle is not None:
            raise ValueError("cloth_bending_yield_stress and cloth_bending_yield_angle are mutually exclusive.")
        if cloth_bending_hardening_modulus < 0.0:
            raise ValueError("cloth_bending_hardening_modulus must be greater than or equal to 0.0.")
        if (
            cloth_bending_yield_stress is None
            and cloth_bending_yield_angle is None
            and cloth_bending_hardening_modulus != 0.0
        ):
            raise ValueError("cloth_bending_hardening_modulus requires a cloth bending yield option.")
        if newton_max_iter is not None and newton_max_iter < 1:
            raise ValueError("newton_max_iter must be greater than or equal to 1.")
        if linear_system_check_interval is not None and linear_system_check_interval < 1:
            raise ValueError("linear_system_check_interval must be greater than or equal to 1.")
        # libuipc clamps the CCD step fraction to (0, 1], so a tolerance outside that
        # range is either unsatisfiable (>1) or disables the check entirely (<=0).
        if newton_ccd_tol is not None and not 0.0 < newton_ccd_tol <= 1.0:
            raise ValueError("newton_ccd_tol must be in the range (0.0, 1.0].")
        if newton_velocity_tol is not None and newton_velocity_tol <= 0.0:
            raise ValueError("newton_velocity_tol must be greater than 0.0.")
        if newton_transrate_tol is not None and newton_transrate_tol <= 0.0:
            raise ValueError("newton_transrate_tol must be greater than 0.0.")
        # beta starts at 1.0 and only decays, so a threshold above 1.0 fires on the
        # first iteration and degenerates into a single-step solve.
        if newton_semi_implicit_beta_tol is not None and not 0.0 < newton_semi_implicit_beta_tol <= 1.0:
            raise ValueError("newton_semi_implicit_beta_tol must be in the range (0.0, 1.0].")
        # libuipc's MAS engine stores each partition in a fixed 16-node bank and
        # asserts on any larger part, which would abort the process inside
        # ``world.init()``; reject it here instead.
        if fem_mas_partition_size is not None and not 1 <= fem_mas_partition_size <= 16:
            raise ValueError("fem_mas_partition_size must be in the range [1, 16].")
        if any(value <= 0.0 for value in (abd_kappa_overrides or {}).values()):
            raise ValueError("abd_kappa_overrides values must be greater than 0.0.")
        if gripper_noncolliding_static_shape_labels is not None and gripper_link_names is None:
            raise ValueError("gripper_noncolliding_static_shape_labels requires gripper_link_names.")

        uipc = self.import_pyuipc()
        # `libuipc` logs verbose per-substep collision/linear-solver diagnostics at its
        # default level; raise the threshold so SolverIPC is quiet unless something is wrong.
        uipc.Logger.set_level(uipc.Logger.Level.Warn)

        self.dt = dt
        self.contact_d_hat = contact_d_hat
        self.friction_enabled = friction_enabled
        self.friction_rate = friction_rate
        self.gripper_link_names = tuple(gripper_link_names) if gripper_link_names is not None else None
        self.gripper_self_contact = gripper_self_contact
        self.gripper_noncolliding_static_shape_labels = frozenset(gripper_noncolliding_static_shape_labels or ())
        self.urdf_body_prefix = urdf_body_prefix
        self.skip_body_labels = frozenset(skip_body_labels) if skip_body_labels is not None else frozenset()
        self.abd_kappa = abd_kappa
        self.abd_kappa_overrides = dict(abd_kappa_overrides or {})
        self.cloth_thickness = cloth_thickness
        self.cloth_self_collision = cloth_self_collision
        self.fem_mas_partition_size = fem_mas_partition_size
        self.cloth_bending_yield_stress = cloth_bending_yield_stress
        self.cloth_bending_yield_angle = cloth_bending_yield_angle
        self.cloth_bending_hardening_modulus = cloth_bending_hardening_modulus
        self.kinematic_constraint_strength = kinematic_constraint_strength
        self.newton_max_iter = newton_max_iter
        self.newton_ccd_tol = newton_ccd_tol
        self.newton_velocity_tol = newton_velocity_tol
        self.newton_transrate_tol = newton_transrate_tol
        self.newton_semi_implicit = newton_semi_implicit
        self.newton_semi_implicit_beta_tol = newton_semi_implicit_beta_tol
        self.use_external_articulation = use_external_articulation
        self.external_articulation_mass = external_articulation_mass
        self.external_articulation_mass_overrides = dict(external_articulation_mass_overrides or {})
        self.articulation_pin_stiffness = articulation_pin_stiffness
        self.articulation_drive_stiffness = articulation_drive_stiffness
        self.abd_full_block_preconditioner = abd_full_block_preconditioner

        cfg = uipc.core.Scene.default_config()
        cfg["dt"] = dt
        cfg["contact"]["d_hat"] = contact_d_hat
        cfg["contact"]["friction"]["enable"] = 1 if friction_enabled else 0
        # Real-robot URDFs put adjacent link collision meshes in contact at the rest
        # pose (arm joints, gripper base/finger overlaps). libuipc's construction-
        # time sanity check flags these as illegal, aborting world.init(). We rely
        # on the contact_tabular's no_contact element to actually disable contact
        # between these pairs at runtime.
        cfg["sanity_check"]["enable"] = 0
        if linear_system_solver is not None:
            cfg["linear_system"]["solver"] = linear_system_solver
        if linear_system_tol_rate is not None:
            cfg["linear_system"]["tol_rate"] = float(linear_system_tol_rate)
        if linear_system_check_interval is not None:
            cfg["linear_system"]["check_interval"] = int(linear_system_check_interval)
        if linear_system_graph_enable is not None:
            cfg["linear_system"]["fused_pcg"]["graph_enable"] = int(bool(linear_system_graph_enable))
        if linear_system_fused_preconditioner_enable is not None:
            cfg["linear_system"]["fused_pcg"]["fused_preconditioner_enable"] = int(
                bool(linear_system_fused_preconditioner_enable)
            )
        cfg["linear_system"]["abd_full_block_preconditioner"] = int(abd_full_block_preconditioner)
        if collision_detection_method is not None:
            cfg["collision_detection"]["method"] = collision_detection_method
        if newton_max_iter is not None:
            cfg["newton"]["max_iter"] = newton_max_iter
        if newton_ccd_tol is not None:
            cfg["newton"]["ccd_tol"] = float(newton_ccd_tol)
        if newton_velocity_tol is not None:
            cfg["newton"]["velocity_tol"] = float(newton_velocity_tol)
        if newton_transrate_tol is not None:
            cfg["newton"]["transrate_tol"] = float(newton_transrate_tol)
        if newton_semi_implicit is not None:
            cfg["newton"]["semi_implicit"]["enable"] = int(bool(newton_semi_implicit))
        if newton_semi_implicit_beta_tol is not None:
            cfg["newton"]["semi_implicit"]["beta_tol"] = float(newton_semi_implicit_beta_tol)

        if model.gravity is None or len(model.gravity) != 1:
            num_worlds = 0 if model.gravity is None else len(model.gravity)
            raise NotImplementedError(
                f"SolverIPC does not support multi-world (vectorized) models; got {num_worlds} worlds."
            )
        gravity = model.gravity.numpy()[0]
        cfg["gravity"] = [[float(gravity[0])], [float(gravity[1])], [float(gravity[2])]]

        self._scene = uipc.core.Scene(cfg)
        self._abd = uipc.constitution.AffineBodyConstitution()
        self._scene.constitution_tabular().insert(self._abd)
        self._tet_constitution = uipc.constitution.StableNeoHookean()
        self._scene.constitution_tabular().insert(self._tet_constitution)
        self._shell = uipc.constitution.NeoHookeanShell()
        if cloth_bending_yield_stress is not None:
            self._bending = uipc.constitution.StressPlasticDiscreteShellBending()
        elif cloth_bending_yield_angle is not None:
            self._bending = uipc.constitution.StrainPlasticDiscreteShellBending()
        else:
            self._bending = uipc.constitution.DiscreteShellBending()
        self._scene.constitution_tabular().insert(self._shell)
        self._scene.constitution_tabular().insert(self._bending)
        self._soft_transform_constraint = uipc.constitution.SoftTransformConstraint()
        self._scene.constitution_tabular().insert(self._soft_transform_constraint)
        # Native articulation drive constitutions. Instantiated even when no URDF
        # joint is present: cheap, and lets `_convert_to_uipc` treat the
        # articulation path as a conditional branch rather than a construction-time
        # if.  `_articulation_slots` is filled per-joint in `_convert_articulation_to_uipc`
        # (added in Task 4).
        self._abd_revolute = uipc.constitution.AffineBodyRevoluteJoint()
        self._abd_prismatic = uipc.constitution.AffineBodyPrismaticJoint()
        self._abd_fixed = uipc.constitution.AffineBodyFixedJoint()
        self._scene.constitution_tabular().insert(self._abd_revolute)
        self._scene.constitution_tabular().insert(self._abd_prismatic)
        self._scene.constitution_tabular().insert(self._abd_fixed)
        # Driving-joint constitutions are only registered on the default drive path.
        # ``AffineBodyDrivingRevoluteJoint``'s cuda init asserts its joint count
        # matches ``AffineBodyRevoluteJoint``'s: registering it but never calling
        # ``apply_to`` (as EAC mode would need) trips the assertion at world.init().
        # `ExternalArticulationConstraint` (EAC) drives revolute/prismatic joints via
        # a caller-supplied per-step ``delta_theta_tilde`` instead of libuipc's own
        # driving-joint constitution.
        if self.use_external_articulation:
            self._abd_driving_revolute = None
            self._abd_driving_prismatic = None
            self._eac = uipc.constitution.ExternalArticulationConstraint()
            self._scene.constitution_tabular().insert(self._eac)
        else:
            self._abd_driving_revolute = uipc.constitution.AffineBodyDrivingRevoluteJoint()
            self._abd_driving_prismatic = uipc.constitution.AffineBodyDrivingPrismaticJoint()
            self._scene.constitution_tabular().insert(self._abd_driving_revolute)
            self._scene.constitution_tabular().insert(self._abd_driving_prismatic)
            self._eac = None
        self._animator = self._scene.animator()

        # Override libuipc's default (friction_rate=0.5, resistance=1e9) if the caller
        # requested a different friction coefficient. Only touched when friction_enabled
        # is True; the resistance term stays at libuipc's default.
        if friction_enabled and friction_rate is not None:
            default = self._scene.contact_tabular().default_model()
            self._scene.contact_tabular().default_model(
                friction_rate=float(friction_rate),
                resistance=default.resistance(),
                enable=True,
            )

        # Kinematic bodies (e.g. a robot's links) are commonly adjacent/overlapping at
        # joints in their rest pose, which IPC's initial-intersection check rejects. They
        # are externally driven, not simulated via contact, so disable contact among them.
        self._kinematic_contact_element = self._scene.contact_tabular().create("kinematic_body")
        self._scene.contact_tabular().insert(
            self._kinematic_contact_element,
            self._kinematic_contact_element,
            0.0,
            0.0,
            False,
        )

        # ``_no_contact_element`` disables contact against everything (self / default /
        # kinematic_body). Two consumers assign bodies to it:
        # 1. ``gripper_link_names`` filter: non-gripper URDF links (fold-paper prod
        #    config only routes the 14 gripper links through the default element).
        # 2. Shape-less URDF frame bodies (e.g. ``arm_l_end_link``, ``gripper_l_camera_link``):
        #    ``_convert_to_uipc`` builds a tiny placeholder ABD so articulation joints
        #    can still attach to them, and routes them here so the placeholder never
        #    interacts with cloth or other bodies.
        # Element + insert-disable rules are created unconditionally: overhead is nil
        # and every consumer needs the same rules.
        self._no_contact_element = self._scene.contact_tabular().create("no_contact")
        # Self-off:
        self._scene.contact_tabular().insert(self._no_contact_element, self._no_contact_element, 0.0, 0.0, False)
        # Off vs default (deformables like cloth/paper live on default):
        self._scene.contact_tabular().insert(
            self._no_contact_element,
            self._scene.contact_tabular().default_element(),
            0.0,
            0.0,
            False,
        )
        # Off vs the kinematic_body element (created just above):
        self._scene.contact_tabular().insert(
            self._no_contact_element,
            self._kinematic_contact_element,
            0.0,
            0.0,
            False,
        )

        # A dedicated gripper element is needed when self-contact is disabled or
        # selected static fixtures must ignore the gripper. Default-element rules
        # continue to provide gripper ↔ deformable contact.
        self._gripper_contact_element = None
        if not gripper_self_contact or self.gripper_noncolliding_static_shape_labels:
            self._gripper_contact_element = self._scene.contact_tabular().create("gripper")
            if not gripper_self_contact:
                # Self-off: fingers vs each other stay silent.
                self._scene.contact_tabular().insert(
                    self._gripper_contact_element,
                    self._gripper_contact_element,
                    0.0,
                    0.0,
                    False,
                )
            # Off vs no_contact and kinematic (both are already
            # self-disabled and off-vs-default; add the symmetric rules so
            # the gripper element inherits the no-contact side too).
            self._scene.contact_tabular().insert(
                self._gripper_contact_element, self._no_contact_element, 0.0, 0.0, False
            )
            self._scene.contact_tabular().insert(
                self._gripper_contact_element,
                self._kinematic_contact_element,
                0.0,
                0.0,
                False,
            )
            # gripper ↔ default (where cloth / paper live) inherits the
            # scene's default_model enable=True -- exactly what we want.

        self._gripper_noncolliding_static_contact_element = None
        if self.gripper_noncolliding_static_shape_labels:
            self._gripper_noncolliding_static_contact_element = self._scene.contact_tabular().create(
                "gripper_noncolliding_static"
            )
            self._scene.contact_tabular().insert(
                self._gripper_contact_element,
                self._gripper_noncolliding_static_contact_element,
                0.0,
                0.0,
                False,
            )
            self._scene.contact_tabular().insert(
                self._no_contact_element,
                self._gripper_noncolliding_static_contact_element,
                0.0,
                0.0,
                False,
            )

        # Populated in `_convert_to_uipc`: body rows whose ABD instance was tagged
        # with `_no_contact_element`. Purely for test/introspection.
        self._no_contact_body_rows: set[int] = set()

        # Maps each row of `AffineBodyStateAccessorFeature`'s flat state array to a
        # Newton body index, or -1 for a fixed (non-Newton-owned) ABD body.
        self._uipc_row_to_body: list[int] = []
        # Maps a Newton body index to its `libuipc` ``Object`` handle. Populated by
        # ``_convert_to_uipc``'s body loop and read by ``_convert_articulation_to_uipc``
        # to attach joint constitutions to the parent/child ABD slots.
        self._body_to_uipc_obj: dict[int, Any] = {}
        # Maps a Newton body index to its ABD ``SimplicialComplexSlot`` (returned by
        # ``obj.geometries().create(mesh)``). Read by the joint constitutions'
        # ``apply_to`` which takes slot handles for the parent/child bodies.
        self._body_to_uipc_slot: dict[int, Any] = {}
        self._static_shape_to_uipc_slot: dict[int, Any] = {}
        # Per-step target transform (4x4) for each `BodyFlags.KINEMATIC` body, keyed by
        # Newton body index; read by `_animate_kinematic_body` and updated in `step()`.
        self._kinematic_targets: dict[int, np.ndarray] = {}
        # Per-URDF-joint state carried across step()s. Each entry:
        # {"kind": "revolute"|"prismatic"|"fixed",
        #  "dof_index": int,                     # index into Model.joint_q / Control.joint_target_q
        #  "geo_slot": uipc.core.GeometrySlot,   # the joint's edge mesh handle
        #  "obj": uipc.core.Object,              # the object that owns the edge geometry
        #  "edge_row": int,                      # which row of the edge attribute to drive (always 0 in v1)
        #  "eac_geo_slot": uipc.core.GeometrySlot | None,  # EAC's articulation geometry (use_external_articulation only)
        #  "eac_obj": uipc.core.Object | None}
        # `dof_index == -1` for FIXED joints (no drive target read at step time).
        # `obj == None` and `geo_slot == None` on world-anchored FIXED (no edge geometry;
        # the child body's ABD is pinned via ``is_fixed`` instead).
        # `eac_geo_slot`/`eac_obj` are non-None only for revolute/prismatic joints when
        # ``use_external_articulation`` is enabled.
        self._articulation_slots: list[dict] = []
        # Per-slot commanded angular/linear increment [rad or m] for the EAC drive
        # path. Sized after ``_convert_articulation_to_uipc`` finishes; the caller
        # writes into it each frame and the ``_animate_eac_delta_theta`` animator
        # picks it up. Empty when ``use_external_articulation`` is False.
        self._eac_delta_theta_tilde: list[float] = []

        self._convert_to_uipc(model)

        self._workspace = workspace if workspace is not None else tempfile.mkdtemp(prefix="newton_ipc_")
        self._engine = uipc.core.Engine("cuda", self._workspace)
        self._world = uipc.core.World(self._engine)
        self._world.init(self._scene)
        self._world.dump()

        self._abd_state_feature = self._world.features().find(uipc.core.AffineBodyStateAccessorFeature)
        if model.body_count > 0:
            self._abd_state_geom = self._abd_state_feature.create_geometry()
            self._abd_state_geom.instances().create(uipc.builtin.transform, np.eye(4, dtype=np.float64))
            self._abd_state_geom.instances().create(uipc.builtin.velocity, np.zeros((4, 4), dtype=np.float64))
        else:
            self._abd_state_geom = None

        self._fem_state_feature = self._world.features().find(uipc.core.FiniteElementStateAccessorFeature)
        if self._has_cloth or self._has_tets:
            self._fem_state_geom = self._fem_state_feature.create_geometry()
            self._fem_state_geom.vertices().create(uipc.builtin.position, np.zeros(3, dtype=np.float64))
            self._fem_state_geom.vertices().create(uipc.builtin.velocity, np.zeros(3, dtype=np.float64))
            fem_vertex_count = self._fem_state_feature.vertex_count()
            if fem_vertex_count != len(self._fem_vertex_to_particle):
                raise RuntimeError(
                    f"Internal error: libuipc FEM accessor reports {fem_vertex_count} vertices, but "
                    f"SolverIPC's vertex-to-particle mapping has {len(self._fem_vertex_to_particle)} entries."
                )
        else:
            self._fem_state_geom = None

        self._write_state_from_uipc(model.state())

    def _tessellate_shape(
        self, geo_type: int, scale: np.ndarray, source: object | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a closed, two-manifold (vertices, triangle indices) mesh in shape-local coordinates."""
        if geo_type == GeoType.BOX:
            mesh = Mesh.create_box(
                float(scale[0]),
                float(scale[1]),
                float(scale[2]),
                duplicate_vertices=False,
                compute_normals=False,
                compute_uvs=False,
                compute_inertia=False,
            )
            return mesh.vertices.astype(np.float64), mesh.indices.reshape(-1, 3).astype(np.int32)
        if geo_type == GeoType.SPHERE:
            mesh = Mesh.create_sphere(
                float(scale[0]),
                compute_normals=False,
                compute_uvs=False,
                compute_inertia=False,
            )
            return _weld_vertices(mesh.vertices.astype(np.float64), mesh.indices.reshape(-1, 3))
        if geo_type == GeoType.CAPSULE:
            mesh = Mesh.create_capsule(
                float(scale[0]),
                float(scale[1]),
                up_axis=Axis.Z,
                compute_normals=False,
                compute_uvs=False,
                compute_inertia=False,
            )
            return _weld_vertices(mesh.vertices.astype(np.float64), mesh.indices.reshape(-1, 3))
        # MESH / CONVEX_MESH: shape_source is a newton.Mesh, scaled per-axis by shape_scale.
        vertices = source.vertices.astype(np.float64) * np.asarray(scale, dtype=np.float64)
        faces = source.indices.reshape(-1, 3).astype(np.int32)
        return _weld_vertices(vertices, faces)

    def _tessellate_body_shapes(
        self,
        model: Model,
        shapes: list[int],
        shape_type: np.ndarray,
        shape_scale: np.ndarray,
        shape_transform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merge all of a body's collision shapes into one combined body-local mesh."""
        all_vertices = []
        all_faces = []
        offset = 0
        for s in shapes:
            geo_type = GeoType(int(shape_type[s]))
            vertices, faces = self._tessellate_shape(geo_type, shape_scale[s], model.shape_source[s])
            local_xform = shape_transform[s]
            r_local = _quat_xyzw_to_matrix(local_xform[3:])
            vertices = vertices @ r_local.T + local_xform[:3]
            all_vertices.append(vertices)
            all_faces.append(faces + offset)
            offset += vertices.shape[0]
        return np.concatenate(all_vertices, axis=0), np.concatenate(all_faces, axis=0)

    def _validate_tet_topology(self, model: Model) -> None:
        """Validate ``Model``'s tet-FEM input and cache derived per-tet data for later use.

        Malformed tet input is rejected eagerly here rather than failing opaquely inside
        ``libuipc``. Checks run in this order so that, for a model with multiple defects,
        the most fundamental one is reported first: index range, then material validity,
        then initial activation, then per-tet volume, then surface-mesh consistency. The
        cached ``self._tet_*`` attributes are consumed by :meth:`_convert_tets_to_uipc`.

        Surface-mesh consistency is only checked when :attr:`Model.tri_count` is nonzero:
        if the model has no triangles at all, ``libuipc`` derives each tet component's
        collision surface directly from its tet topology (``uipc.geometry.label_surface``
        in :meth:`_convert_tets_to_uipc`), so no matching ``Model.tri_indices`` are
        required. A *partial* surface (some but not all tet boundary faces present) is
        not supported; once any triangle exists, every tet boundary face implied by
        ``Model.tet_indices`` must appear in ``Model.tri_indices`` exactly once.
        """
        tet_indices_wp = model.tet_indices
        if tet_indices_wp is None or model.tet_count == 0:
            self._tet_indices = None
            self._tet_materials = None
            self._tet_volumes = None
            self._tet_surface_mask = None
            self._tet_components = None
            return

        tet_indices = tet_indices_wp.numpy().reshape(-1, 4).astype(np.int64)
        particle_count = model.particle_count

        out_of_range = (tet_indices < 0) | (tet_indices >= particle_count)
        if np.any(out_of_range):
            bad_tet, bad_vert = (int(x) for x in np.argwhere(out_of_range)[0])
            raise ValueError(
                f"Model.tet_indices[{bad_tet}] references particle index "
                f"{int(tet_indices[bad_tet, bad_vert])}, which is out of range for "
                f"{particle_count} particles."
            )

        tet_materials = model.tet_materials.numpy()
        k_mu = tet_materials[:, 0]
        k_lambda = tet_materials[:, 1]
        if not (np.all(np.isfinite(k_mu)) and np.all(k_mu > 0.0)):
            raise ValueError("SolverIPC requires every tet material k_mu (mu) to be finite and strictly positive.")
        if not (np.all(np.isfinite(k_lambda)) and np.all(k_lambda > 0.0)):
            raise ValueError(
                "SolverIPC requires every tet material k_lambda (lambda) to be finite and strictly positive."
            )

        tet_activations = model.tet_activations.numpy()
        if np.any(tet_activations != 0.0):
            raise NotImplementedError("SolverIPC does not support a nonzero initial tet activation.")

        particle_q = model.particle_q.numpy().astype(np.float64)
        p0 = particle_q[tet_indices[:, 0]]
        p1 = particle_q[tet_indices[:, 1]]
        p2 = particle_q[tet_indices[:, 2]]
        p3 = particle_q[tet_indices[:, 3]]
        # Initial deformation must also remain non-inverted when a separate
        # elastic reference is supplied by the USD importer.
        reference = getattr(model, "ipc_rest_particle_q", None)
        if reference is not None:
            initial_volume6 = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))
            if not np.isfinite(initial_volume6).all() or np.any(initial_volume6 <= 0):
                raise ValueError("SolverIPC initial tet positions have non-positive or nonfinite volume")
            p0, p1, p2, p3 = (reference[tet_indices[:, i]] for i in range(4))
        # `tet_poses` (the rest-pose inverse used by the VBD/XPBD solvers) is not trustworthy
        # here: it is computed once at `add_tetrahedron()` time and never revalidated against
        # `tet_indices`/`particle_q`, so a test (or a user) that mutates either after the fact
        # would go undetected. Recompute the signed volume directly from current inputs instead.
        volumes = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0)) / 6.0
        if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
            bad = int(np.flatnonzero(~np.isfinite(volumes) | (volumes <= 0.0))[0])
            raise ValueError(f"Model.tet_indices[{bad}] has non-positive volume ({volumes[bad]}).")

        if model.tri_count > 0:
            boundary_faces = _tet_boundary_faces(tet_indices)
            tri_indices = model.tri_indices.numpy().reshape(-1, 3).astype(np.int64)
            surface_mask = _classify_tet_surface_triangles(tri_indices, boundary_faces)
        else:
            # No Model.tri_indices at all (not even for an unrelated cloth mesh): rather than
            # require a surface mesh that duplicates the tet boundary, let libuipc derive it
            # directly from tet topology (`uipc.geometry.label_surface` in
            # `_convert_tets_to_uipc`). A *partial* surface — some but not all tet boundary
            # faces present — is not supported and still hits the `model.tri_count > 0` branch
            # above, which requires an exact match.
            surface_mask = np.zeros(0, dtype=bool)

        self._tet_indices = tet_indices
        self._tet_materials = tet_materials
        self._tet_volumes = volumes
        self._tet_surface_mask = surface_mask
        self._tet_components = _tet_connected_components(tet_indices, particle_count)

    def _apply_eac_body_seed(self, mesh: Any, transform: np.ndarray) -> None:
        """Prepare an ABD ``mesh`` to participate in EAC's external-kinetic term.

        Sets ``external_kinetic=1`` on the mesh's single instance and seeds a
        12-component ``ref_dof_prev`` per-instance attribute from ``transform``
        so the very first solver step sees a zero-delta prior configuration.
        Must be called after :meth:`AffineBodyConstitution.apply_to` and before
        ``obj.geometries().create(mesh)``: attributes added post-``.create`` land
        on the stale original mesh, not the scene copy. Pattern from
        ``libuipc/python/examples/eac_probe.py``'s ``make_abd_cube``.
        """
        uipc = self._uipc
        ek = mesh.instances().find(uipc.builtin.external_kinetic)
        if ek is not None:
            uipc.view(ek)[:] = 1
        rdp = mesh.instances().create("ref_dof_prev", np.zeros(12, dtype=np.float64))
        q = np.asarray(uipc.geometry.affine_body.transform_to_q(transform)).reshape(12)
        np.asarray(uipc.view(rdp))[0, :, 0] = q

    def _convert_to_uipc(self, model: Model) -> None:
        uipc = self._uipc

        if model.body_count == 0 and model.particle_count == 0:
            raise ValueError("SolverIPC requires at least one rigid body or particle in the model.")

        self._validate_tet_topology(model)

        body_flags = model.body_flags.numpy()

        body_q = model.body_q.numpy()
        body_mass = model.body_mass.numpy()

        eac_body_indices: set[int] = set()
        if self.use_external_articulation:
            joint_type = model.joint_type.numpy()
            joint_parent = model.joint_parent.numpy()
            joint_child = model.joint_child.numpy()
            supported_joint_types = {
                int(JointType.REVOLUTE),
                int(JointType.PRISMATIC),
                int(JointType.FIXED),
            }
            for j in range(model.joint_count):
                if int(joint_type[j]) not in supported_joint_types:
                    continue
                parent = int(joint_parent[j])
                child = int(joint_child[j])
                if parent >= 0:
                    eac_body_indices.add(parent)
                if child >= 0:
                    eac_body_indices.add(child)

        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_transform = model.shape_transform.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_flags = model.shape_flags.numpy()

        shapes_by_body: dict[int, list[int]] = {}
        ground_shapes: list[int] = []
        static_shapes: list[int] = []
        for s in range(model.shape_count):
            flags = int(shape_flags[s])
            if (flags & ShapeFlags.SITE) or not (flags & ShapeFlags.COLLIDE_SHAPES):
                continue
            geo_type = GeoType(int(shape_type[s]))
            b = int(shape_body[s])
            if b < 0:
                if geo_type == GeoType.PLANE:
                    ground_shapes.append(s)
                elif geo_type in _SUPPORTED_RIGID_GEO_TYPES:
                    static_shapes.append(s)
                else:
                    raise NotImplementedError(
                        f"SolverIPC does not support static shape type {geo_type.name} (shape {s})."
                    )
            else:
                if geo_type not in _SUPPORTED_RIGID_GEO_TYPES:
                    raise NotImplementedError(
                        f"SolverIPC does not support shape type {geo_type.name} on rigid body {b} (shape {s})."
                    )
                shapes_by_body.setdefault(b, []).append(s)

        for b in range(model.body_count):
            shapes = shapes_by_body.get(b, [])
            is_kinematic = bool(body_flags[b] & BodyFlags.KINEMATIC)
            # ``skip_body_labels`` completely drops a body from IPC (no ABD, no
            # joint constraints reference it). Used to trim non-essential URDF
            # links (torso, head, wheels) so only the arms + grippers participate
            # in IPC dynamics.
            if b < len(model.body_label) and model.body_label[b] in self.skip_body_labels:
                continue
            if len(shapes) == 0:
                if is_kinematic:
                    # Externally-driven body with no collision geometry (e.g. a massless
                    # URDF fixed-joint frame); nothing to add to the IPC world.
                    continue
                # DYNAMIC body without collision (e.g. URDF frame links like
                # ``arm_l_end_link``, ``gripper_l_camera_link``): pass 2 will build a
                # tiny placeholder ABD so articulation joints can still attach.
                continue
            # No cap on shape count: ``_tessellate_body_shapes`` merges every collision
            # primitive into one body-local mesh for ABD's mass_density / volume calc.

        objs = self._scene.objects()

        for b in range(model.body_count):
            shapes = shapes_by_body.get(b, [])
            is_kinematic = bool(body_flags[b] & BodyFlags.KINEMATIC)
            # ``skip_body_labels`` completely drops a body from IPC (no ABD, no
            # joint constraints reference it). Used to trim non-essential URDF
            # links (torso, head, wheels) so only the arms + grippers participate
            # in IPC dynamics.
            if b < len(model.body_label) and model.body_label[b] in self.skip_body_labels:
                continue
            if len(shapes) == 0:
                if is_kinematic:
                    # Kinematic bodies with no collision geometry were already validated (and
                    # `continue`d past) above; skip them here too, they get no ABD object at all.
                    continue
                # DYNAMIC shape-less body: build a tiny placeholder ABD box (5mm
                # half-extents, 1 mg mass) so libuipc can attach articulation joints
                # to this URDF frame. Auto-route to _no_contact_element so the
                # placeholder never influences contact.
                placeholder_scale = np.array([0.005, 0.005, 0.005], dtype=np.float32)
                vertices, faces = self._tessellate_shape(GeoType.BOX, placeholder_scale, None)
                volume = _signed_volume(vertices, faces)
                mesh = uipc.geometry.trimesh(vertices, faces)
                uipc.geometry.label_surface(mesh)
                # 1 mg / 1e-6 m^3 = 1e3 kg/m^3 (water); tiny inertia vs real links.
                self._abd.apply_to(
                    mesh,
                    kappa=self._abd_kappa_for_body(model, b),
                    mass_density=1.0e-3 / volume,
                )
                self._no_contact_element.apply_to(mesh)
                self._no_contact_body_rows.add(b)

                rot = _quat_xyzw_to_matrix(body_q[b, 3:])
                transform = np.eye(4)
                transform[:3, :3] = rot
                transform[:3, 3] = body_q[b, :3]
                uipc.view(mesh.transforms())[0] = transform

                if self.use_external_articulation and b in eac_body_indices:
                    self._apply_eac_body_seed(mesh, transform)

                obj = objs.create(f"body_{b}_placeholder")
                geo_slot, _rest_slot = obj.geometries().create(mesh)
                self._uipc_row_to_body.append(b)
                self._body_to_uipc_obj[b] = obj
                self._body_to_uipc_slot[b] = geo_slot

                if self.use_external_articulation and b in eac_body_indices:
                    # Refresh ref_dof_prev each frame from the current transform,
                    # matching the EAC POC's ``refresh_arm_ref_dof_prev`` pattern.
                    self._animator.insert(
                        obj,
                        functools.partial(_animate_ref_dof_prev, weakref.ref(self), b),
                    )
                continue
            vertices, faces = self._tessellate_body_shapes(model, shapes, shape_type, shape_scale, shape_transform)

            mass = float(body_mass[b])
            if mass <= 0.0:
                if is_kinematic:
                    # A soft-transform-constrained kinematic body's tracking accuracy does not
                    # depend on mass; substitute a placeholder so ABD's mass_density stays positive.
                    mass = 1.0
                else:
                    raise ValueError(f"Rigid body {b} has non-positive mass ({mass}); required for SolverIPC.")
            volume = _signed_volume(vertices, faces)
            if volume <= 1.0e-12:
                raise ValueError(f"Collision shape for rigid body {b} has non-positive volume ({volume}).")

            mesh = uipc.geometry.trimesh(vertices, faces)
            uipc.geometry.label_surface(mesh)
            self._abd.apply_to(
                mesh,
                kappa=self._abd_kappa_for_body(model, b),
                mass_density=mass / volume,
            )
            if is_kinematic:
                # Must run before `obj.geometries().create(mesh)` below, which copies `mesh`
                # into the scene: attributes added afterwards would land on the stale original.
                self._soft_transform_constraint.apply_to(mesh, strength_ratio=list(self.kinematic_constraint_strength))
                self._kinematic_contact_element.apply_to(mesh)

            # Contact element split for the gripper-only collision path.
            # A gripper stays on the default element only when it needs neither
            # self-contact filtering nor selected-static filtering.
            # Free props outside the URDF namespace stay on default so they still
            # collide through the gripper-default contact model.
            if self.gripper_link_names is not None and self._uses_gripper_contact_filter(model, b):
                if not self._is_gripper_link(model, b):
                    self._no_contact_element.apply_to(mesh)
                    self._no_contact_body_rows.add(b)
                elif self._gripper_contact_element is not None:
                    self._gripper_contact_element.apply_to(mesh)

            rot = _quat_xyzw_to_matrix(body_q[b, 3:])
            transform = np.eye(4)
            transform[:3, :3] = rot
            transform[:3, 3] = body_q[b, :3]
            uipc.view(mesh.transforms())[0] = transform

            # EAC replaces the standard kinetic term only for bodies in its joint
            # graph. Free props must retain ordinary inertia and gravity response.
            if self.use_external_articulation and not is_kinematic and b in eac_body_indices:
                self._apply_eac_body_seed(mesh, transform)

            obj = objs.create(f"body_{b}")
            geo_slot, _rest_slot = obj.geometries().create(mesh)
            self._uipc_row_to_body.append(b)
            self._body_to_uipc_obj[b] = obj
            self._body_to_uipc_slot[b] = geo_slot

            if is_kinematic:
                self._kinematic_targets[b] = transform.copy()
                self._animator.insert(
                    obj,
                    functools.partial(_animate_kinematic_body, weakref.ref(self), b),
                )
            elif self.use_external_articulation and b in eac_body_indices:
                # Refresh ref_dof_prev each frame from the ABD's current transform;
                # EAC's kinetic energy needs a valid prior configuration each step.
                self._animator.insert(obj, functools.partial(_animate_ref_dof_prev, weakref.ref(self), b))

        for s in static_shapes:
            geo_type = GeoType(int(shape_type[s]))
            vertices, faces = self._tessellate_shape(geo_type, shape_scale[s], model.shape_source[s])
            volume = _signed_volume(vertices, faces)
            if volume <= 1.0e-12:
                raise ValueError(f"Static collision shape {s} has non-positive volume ({volume}).")

            mesh = uipc.geometry.trimesh(vertices, faces)
            uipc.geometry.label_surface(mesh)
            self._abd.apply_to(mesh, kappa=self.abd_kappa, mass_density=1000.0)
            uipc.view(mesh.instances().find(uipc.builtin.is_fixed))[:] = 1
            if (
                self._gripper_noncolliding_static_contact_element is not None
                and s < len(model.shape_label)
                and model.shape_label[s] in self.gripper_noncolliding_static_shape_labels
            ):
                self._gripper_noncolliding_static_contact_element.apply_to(mesh)

            world_xform = shape_transform[s]
            rot = _quat_xyzw_to_matrix(world_xform[3:])
            transform = np.eye(4)
            transform[:3, :3] = rot
            transform[:3, 3] = world_xform[:3]
            uipc.view(mesh.transforms())[0] = transform

            obj = objs.create(f"static_{s}")
            geo_slot, _rest_slot = obj.geometries().create(mesh)
            self._static_shape_to_uipc_slot[s] = geo_slot
            self._uipc_row_to_body.append(-1)

        for s in ground_shapes:
            world_xform = shape_transform[s]
            rot = _quat_xyzw_to_matrix(world_xform[3:])
            normal = rot[:, 2]
            height = float(np.dot(world_xform[:3], normal))
            ground_geom = uipc.geometry.ground(height, normal)
            obj = objs.create(f"ground_{s}")
            obj.geometries().create(ground_geom)

        self._body_com = model.body_com.numpy()

        self._fem_vertex_to_particle: list[int] = []
        if model.particle_count > 0:
            self._initial_particle_q = model.particle_q.numpy().astype(np.float32, copy=True)
            self._initial_particle_qd = model.particle_qd.numpy().astype(np.float32, copy=True)
        else:
            self._initial_particle_q = None
            self._initial_particle_qd = None

        self._convert_articulation_to_uipc(model)

        self._has_tets = self._tet_indices is not None
        # A tet-FEM soft body's own boundary triangles double as its `Model.tri_indices`
        # collision surface (see `_validate_tet_topology`); only *other* triangles (e.g. an
        # independent cloth mesh) require the separate NeoHookeanShell path below. Cloth and
        # tet-FEM may coexist in the same model, on disjoint particle sets: cloth is created
        # first (see `_convert_cloth_to_uipc`), then each tet component, so
        # `_fem_vertex_to_particle` orders cloth's particles before tet's.
        if self._tet_surface_mask is not None:
            cloth_triangle_mask = ~self._tet_surface_mask
        else:
            cloth_triangle_mask = np.ones(model.tri_count, dtype=bool)
        self._has_cloth = bool(np.any(cloth_triangle_mask))

        if self._has_cloth:
            self._convert_cloth_to_uipc(model, cloth_triangle_mask)
        if self._has_tets:
            self._convert_tets_to_uipc(model)

        if model.particle_count > 0 and not self._has_cloth and not self._has_tets:
            raise NotImplementedError(
                "SolverIPC requires particles to form cloth triangles or tetrahedral FEM elements; "
                "bare particle clouds are not supported."
            )

        if len(set(self._fem_vertex_to_particle)) != len(self._fem_vertex_to_particle):
            raise RuntimeError(
                "Internal error: SolverIPC's FEM vertex-to-particle mapping contains a duplicate particle index."
            )

    def _abd_kappa_for_body(self, model: Model, body: int) -> float:
        """Return the configured ABD stiffness for one labeled rigid body."""
        label = model.body_label[body] if body < len(model.body_label) else ""
        return float(self.abd_kappa_overrides.get(label, self.abd_kappa))

    def _uses_gripper_contact_filter(self, model: Model, body: int) -> bool:
        """Return whether ``body`` belongs to the URDF contact-filter namespace."""
        label = model.body_label[body] if body < len(model.body_label) else ""
        if not label:
            return False
        return not self.urdf_body_prefix or label.startswith(self.urdf_body_prefix)

    def _is_gripper_link(self, model: Model, body: int) -> bool:
        """True iff ``body``'s label matches one of ``self.gripper_link_names``,
        interpreted as the suffix after ``self.urdf_body_prefix``. Bodies with no
        label always return True (they're static / free-floating props, not URDF
        articulation links, so they collide by default)."""
        if self.gripper_link_names is None:
            return True  # No filter: everything is "gripper" (i.e. everything collides).
        label = model.body_label[body] if body < len(model.body_label) else ""
        if not label:
            return True  # Non-URDF free bodies (table etc.) collide by default.
        if self.urdf_body_prefix:
            if not label.startswith(self.urdf_body_prefix):
                return True  # Labeled free props are outside the filtered URDF namespace.
            label = label[len(self.urdf_body_prefix) :]
        return label in self.gripper_link_names

    def _convert_articulation_to_uipc(self, model: Model) -> None:
        """For every URDF joint in ``model``, build a 1-edge line-mesh in libuipc,
        apply the appropriate joint constitution (revolute / prismatic / fixed) and
        (for non-FIXED) its driving constitution, and store a slot handle in
        ``self._articulation_slots``.

        Requires the parent/child ABD bodies to already exist in the scene (they were
        created by the body loop above). If the child body was skipped (e.g. no
        collision geometry AND no articulation), the joint is also skipped.

        Skipped entirely when every URDF body is ``KINEMATIC``: the caller is running
        the legacy soft-transform-only path (``SoftTransformConstraint`` alone pulls
        each ABD to its per-frame target ``body_q``), so adding
        ``AffineBodyRevoluteJoint`` / ``AffineBodyPrismaticJoint`` pin + drive
        constraints on top would over-constrain the system — the two constraint
        families fight each other and gripper links visibly scatter under motion.
        """
        if model.joint_count == 0:
            return

        # Legacy IPC-only path detection: no URDF body is DYNAMIC. In this mode the
        # caller writes body_q per frame via ``eval_fk`` and relies on
        # SoftTransformConstraint alone; installing revolute-joint pin+drive
        # constraints on top would double-drive every link.
        #
        # Only REVOLUTE/PRISMATIC/FIXED joints get pin+drive constitutions below,
        # so FREE joints must be excluded here too -- otherwise a free-floating
        # DYNAMIC prop (e.g. ``ModelBuilder.add_body()``'s implicit FREE joint)
        # is mistaken for a DYNAMIC articulated body, defeating this early-return
        # and double-driving an otherwise fully KINEMATIC arm.
        body_flags = model.body_flags.numpy()
        joint_type_for_guard = model.joint_type.numpy()
        joint_child = model.joint_child.numpy()
        has_dynamic_articulated_body = False
        for j in range(model.joint_count):
            if int(joint_type_for_guard[j]) not in (
                int(JointType.REVOLUTE),
                int(JointType.PRISMATIC),
                int(JointType.FIXED),
            ):
                continue
            child = int(joint_child[j])
            if child < 0:
                continue
            if not (int(body_flags[child]) & int(BodyFlags.KINEMATIC)):
                has_dynamic_articulated_body = True
                break
        if not has_dynamic_articulated_body:
            return

        uipc = self._uipc

        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_axis = model.joint_axis.numpy()
        joint_X_p = model.joint_X_p.numpy()  # transform in parent link's frame
        joint_X_c = model.joint_X_c.numpy()  # transform in child link's frame
        joint_q_start = model.joint_q_start.numpy()
        joint_label = (
            model.joint_label if model.joint_label is not None else [f"joint_{j}" for j in range(model.joint_count)]
        )
        # Initial URDF joint angles at construction time. libuipc's driving revolute
        # joint stores an `init_angle` per edge that acts as the zero-reference for
        # aim_angle; if left at its default 0.0 while the ABD instances are already
        # placed at non-zero URDF angles (e.g. G2's ready pose folds arms at 30-75°),
        # the joint drive tries to physically rotate the ABD by another `target` rad
        # every step, which explodes into thousands of Newton iterations.
        joint_q_np = model.joint_q.numpy() if model.joint_q is not None else np.zeros(0, dtype=np.float32)

        # Task 1 audit confirmed `joint_axis` is JOINT_DOF frequency (one row per DOF, not
        # per joint), so a 1-DOF joint indexes at `joint_q_start[j]`. FIXED joints have
        # `dof_dim=(0, 0)` and `joint_q_start[j] == joint_q_start[j+1]`; the axis is
        # meaningless there, and the fixed-joint constitution uses its own
        # point-only geometry (no linemesh needed).

        # Two separate stiffnesses. libuipc lets us drive the "pin" (revolute /
        # prismatic base constraint that keeps parent and child ABDs bolted at the
        # joint pivot with the right DOF) independently from the "drive" (position
        # torque toward aim_angle).
        # * pin: must be very stiff to keep the arm from falling apart under motion
        #   or gravity — this is the "fix the connection" strength.
        # * drive: moderate so the control loop doesn't oscillate. Follows Task 2
        #   POC's 100-1000 range; a single value across the whole G2 chain works
        #   because strength_ratio is multiplied by (m_parent + m_child) internally.
        pin_strengths = np.array([float(self.articulation_pin_stiffness)], dtype=np.float64)
        drive_strengths = np.array([float(self.articulation_drive_stiffness)], dtype=np.float64)
        # Fixed joints have only the pin (no drive constitution); reuse pin_strengths.
        fixed_strengths = pin_strengths

        for j in range(model.joint_count):
            j_type = int(joint_type[j])
            parent_body = int(joint_parent[j])
            child_body = int(joint_child[j])
            dof_start = int(joint_q_start[j])

            if j_type == int(JointType.FREE):
                # ABD implicitly supports free 6-DoF motion; no joint constitution needed.
                # FREE joints show up automatically for bodies added via ``ModelBuilder.add_body``.
                continue
            if j_type not in (
                int(JointType.REVOLUTE),
                int(JointType.PRISMATIC),
                int(JointType.FIXED),
            ):
                # BALL / D6 / DISTANCE / CABLE are user-configured; surface the misconfiguration
                # rather than silently drop the joint.
                raise NotImplementedError(
                    f"SolverIPC articulation: unsupported joint type {JointType(j_type).name} "
                    f"(joint {j}); only REVOLUTE / PRISMATIC / FIXED / FREE are handled."
                )

            child_slot = self._body_to_uipc_slot.get(child_body)
            if child_slot is None:
                # Child body wasn't turned into an ABD (no collision geometry). Skip;
                # its would-be joint contributes nothing to contact and Newton owns its FK.
                # ``_body_to_uipc_obj`` and ``_body_to_uipc_slot`` are populated in lockstep by
                # the body loop, so a missing slot implies a missing obj as well.
                continue

            if parent_body < 0 and j_type == int(JointType.FIXED):
                # World-anchored fixed joint: just pin the child's ABD, no joint edge needed.
                child_geo = child_slot.geometry()
                uipc.view(child_geo.instances().find(uipc.builtin.is_fixed))[:] = 1
                self._articulation_slots.append(
                    {
                        "kind": "fixed",
                        "dof_index": -1,
                        "geo_slot": None,
                        "obj": None,
                        "edge_row": 0,
                        "eac_geo_slot": None,
                        "eac_obj": None,
                        "joint_name": joint_label[j] if j < len(joint_label) else f"joint_{j}",
                    }
                )
                continue

            if parent_body < 0:
                raise NotImplementedError(
                    f"SolverIPC articulation: joint {j} has world parent but non-FIXED type "
                    f"{JointType(j_type).name}; use FIXED to anchor a floating root."
                )

            parent_slot = self._body_to_uipc_slot.get(parent_body)
            if parent_slot is None:
                # Parent body was dropped from IPC (typically via ``skip_body_labels``).
                # Treat the child as world-anchored at its current URDF FK pose so the
                # articulation chain still has a base to hang off — e.g. arm_base_link
                # when the torso is skipped.
                child_geo = child_slot.geometry()
                uipc.view(child_geo.instances().find(uipc.builtin.is_fixed))[:] = 1
                self._articulation_slots.append(
                    {
                        "kind": "fixed",
                        "dof_index": -1,
                        "geo_slot": None,
                        "obj": None,
                        "edge_row": 0,
                        "eac_geo_slot": None,
                        "eac_obj": None,
                        "joint_name": joint_label[j] if j < len(joint_label) else f"joint_{j}",
                    }
                )
                continue

            # Build joint mesh + apply constitution. REVOLUTE / PRISMATIC use a
            # 1-edge linemesh (attributes stored per-edge, one per joint DoF).
            # FIXED uses a point-only SimplicialComplex (attributes stored per
            # vertex) built by ``AffineBodyFixedJoint.create_geometry`` — passing
            # a linemesh with 2 vertices trips libuipc's FIXED joint init because
            # only vertex 0 gets its l_geo_id / r_geo_id filled, vertex 1 keeps
            # the default IndexT{-1} that then trips the "Geometry ID -1" assert.
            #
            # We use the LOCAL create_geometry API (``l_position0/1``,
            # ``r_position0/1``) rather than a manually-built linemesh + apply_to.
            # The local API stores per-body anchor + axis attributes on each
            # edge, which trigger libuipc's frame-aware ``use_local`` path in
            # ``ExternalArticulationConstraint::get_revolute_basis``. Without
            # them, EAC treats the linemesh vertices as world-frame — but our
            # ``joint_X_p[:3] + axis_unit`` vertices are in parent-body-local
            # coords, so EAC ends up computing the joint's perpendicular basis
            # in the wrong frame and drive commands rotate each body around its
            # own origin instead of the shared joint anchor, separating parent
            # and child anchors by ~|joint_X_p| per rad of drive (see fold_paper
            # diagnostic sweep at commit-time).
            if j_type in (int(JointType.REVOLUTE), int(JointType.PRISMATIC)):
                origin_p_local = np.asarray(joint_X_p[j, :3], dtype=np.float64)
                origin_c_local = np.asarray(joint_X_c[j, :3], dtype=np.float64)
                quat_p = np.asarray(joint_X_p[j, 3:7], dtype=np.float64)  # (qx,qy,qz,qw)
                quat_c = np.asarray(joint_X_c[j, 3:7], dtype=np.float64)
                axis_joint = (
                    np.asarray(joint_axis[dof_start], dtype=np.float64)
                    if dof_start < joint_axis.shape[0]
                    else np.array([0.0, 0.0, 1.0], dtype=np.float64)
                )
                axis_norm = float(np.linalg.norm(axis_joint))
                if axis_norm > 1.0e-9:
                    axis_joint = axis_joint / axis_norm
                else:
                    axis_joint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                # Axis in each body's own local frame. joint_X_p / joint_X_c
                # rotate FROM the joint frame TO the parent / child body frames,
                # so the joint's axis (stored in the joint frame) lifts to the
                # parent-local axis via joint_X_p's rotation, and similarly for
                # the child.
                axis_p_local = _quat_rotate_vec(quat_p, axis_joint)
                axis_c_local = _quat_rotate_vec(quat_c, axis_joint)

                l_pos0 = np.array([origin_p_local], dtype=np.float64)  # (1, 3)
                l_pos1 = np.array([origin_p_local + axis_p_local], dtype=np.float64)
                r_pos0 = np.array([origin_c_local], dtype=np.float64)
                r_pos1 = np.array([origin_c_local + axis_c_local], dtype=np.float64)

                if j_type == int(JointType.REVOLUTE):
                    joint_geo = self._abd_revolute.create_geometry(
                        l_pos0,
                        l_pos1,
                        r_pos0,
                        r_pos1,
                        [parent_slot],
                        np.array([0], dtype=np.int32),
                        [child_slot],
                        np.array([0], dtype=np.int32),
                        pin_strengths,
                    )
                    if not self.use_external_articulation:
                        self._abd_driving_revolute.apply_to(joint_geo, drive_strengths)
                    kind = "revolute"
                else:
                    joint_geo = self._abd_prismatic.create_geometry(
                        l_pos0,
                        l_pos1,
                        r_pos0,
                        r_pos1,
                        [parent_slot],
                        np.array([0], dtype=np.int32),
                        [child_slot],
                        np.array([0], dtype=np.int32),
                        pin_strengths,
                    )
                    if not self.use_external_articulation:
                        self._abd_driving_prismatic.apply_to(joint_geo, drive_strengths)
                    kind = "prismatic"

                # Seed init_angle from the URDF joint's construction-time value so
                # aim_angle=X interprets X as an absolute target in URDF coordinates,
                # not "rotate by X from the pose libuipc happens to see at t=0."
                # Without this, ready-pose init + a target near ready-pose still
                # produces a large aim delta and the solver diverges.
                if dof_start < joint_q_np.shape[0]:
                    init_angle_attr = joint_geo.edges().find("init_angle")
                    if init_angle_attr is not None:
                        uipc.view(init_angle_attr)[:] = float(joint_q_np[dof_start])
            elif j_type == int(JointType.FIXED):
                # ``create_geometry`` builds a 1-vertex point-only SC with the
                # attachment position auto-set to the midpoint of the two body
                # origins. Exact attachment location doesn't matter for a fixed
                # joint (it constrains both position AND orientation via the
                # 3-axis rest_t/n/b terms), so the midpoint default is fine.
                joint_geo = self._abd_fixed.create_geometry(
                    [parent_slot],
                    np.array([0], dtype=np.int32),
                    [child_slot],
                    np.array([0], dtype=np.int32),
                    fixed_strengths,
                )
                kind = "fixed"
                dof_start = -1  # No drive.
            else:
                # Unreachable: the type-filter at the top of the loop rejected anything
                # other than REVOLUTE/PRISMATIC/FIXED. Kept as a defensive assertion.
                raise AssertionError(f"joint {j} passed type filter but is {JointType(j_type).name}")

            joint_obj = self._scene.objects().create(f"joint_{j}")
            # `.geometries().create(...)` returns a (geo_slot, rest_slot) tuple; the
            # geo slot is what drive control needs (Task 6 writes ``aim_angle`` +
            # ``driving/is_constrained`` on this slot's edges each step).
            joint_geo_slot, _rest_slot = joint_obj.geometries().create(joint_geo)

            self._articulation_slots.append(
                {
                    "kind": kind,
                    "dof_index": dof_start,
                    "geo_slot": joint_geo_slot,
                    "obj": joint_obj,
                    "edge_row": 0,
                    "eac_geo_slot": None,  # Set below for the whole articulation, not per joint.
                    "eac_obj": None,
                    "joint_name": joint_label[j] if j < len(joint_label) else f"joint_{j}",
                }
            )

        # Pre-allocate the per-step target array so step() only rewrites values,
        # never re-sizes. Same length as `_articulation_slots`; fixed-joint entries
        # ignore their slot. The `_eac_delta_theta_tilde` list is sized identically
        # for the EAC drive path — callers write into it before each step.
        self._joint_aim_values: list[float] = [0.0] * len(self._articulation_slots)
        self._eac_delta_theta_tilde = [0.0] * len(self._articulation_slots)

        # EAC mode: build ONE ExternalArticulationConstraint covering every
        # revolute/prismatic joint in the articulation with a coupled
        # (n_joints x n_joints) mass matrix, matching the pattern in libuipc's
        # ``47_external_articulation_multijoints.cpp`` test. Per-joint EACs
        # (as in ``46_external_articulation_constraint.cpp``) work only when the
        # joints share no bodies; for a serial chain like G2's arm, the same body
        # appears as child of joint N and parent of joint N+1, so the joints must
        # share one articulation geometry so libuipc can propagate inertia through
        # the chain rather than treat each joint as an isolated 1-DOF system.
        #
        # A single-EAC articulation was originally per-joint here: the parent-side
        # joint anchor and the child-side anchor visibly separated by ≈|joint_X_p|
        # per radian of drive under load, because each per-joint EAC only saw the
        # local 1x1 mass and ignored downstream chain inertia.
        self._eac_geo_slot = None
        self._eac_obj = None
        # eac_pos → slot_idx; only revolute/prismatic slots participate.
        self._eac_slot_index_map: list[int] = []
        if self.use_external_articulation:
            joint_geo_slots: list[Any] = []
            edge_indices: list[int] = []
            for slot_idx, slot in enumerate(self._articulation_slots):
                if slot["kind"] not in ("revolute", "prismatic"):
                    continue
                joint_geo_slots.append(slot["geo_slot"])
                edge_indices.append(int(slot["edge_row"]))
                self._eac_slot_index_map.append(slot_idx)

            if joint_geo_slots:
                articulation_geo = self._eac.create_geometry(
                    joint_geo_slots,
                    np.asarray(edge_indices, dtype=np.int32),
                )
                # ``joint_joint`` is a flat (n x n) mass matrix in row-major layout.
                # We seed it as a diagonal with ``external_articulation_mass`` on
                # every entry — a rough proxy for the articulation's inertia matrix.
                # The correct fully-coupled matrix would follow the composite-rigid-
                # body inertia (Featherstone chapter 6) but a diagonal already
                # decouples the drive semantics from per-joint EAC + gives libuipc
                # the shared articulation view the pin constraint expects.
                mass_attr = articulation_geo["joint_joint"].find("mass")
                if mass_attr is not None:
                    n = len(joint_geo_slots)
                    mass_view = np.asarray(uipc.view(mass_attr))
                    mass_view.reshape(-1)[:] = 0.0
                    default_m = float(self.external_articulation_mass)
                    mass_flat = mass_view.reshape(-1)
                    # Each diagonal entry corresponds to one articulation slot in
                    # ``_eac_slot_index_map`` (populated above in the same order).
                    # ``joint_name`` lookup lets callers stiffen a subset of joints
                    # (e.g. the gripper) without touching arm-chain dynamics.
                    for i in range(n):
                        slot_idx = self._eac_slot_index_map[i]
                        slot = self._articulation_slots[slot_idx]
                        joint_name = slot.get("joint_name", "")
                        m = float(self.external_articulation_mass_overrides.get(joint_name, default_m))
                        mass_flat[i * n + i] = m
                self._eac_obj = self._scene.objects().create("eac_articulation")
                self._eac_geo_slot, _eac_rest = self._eac_obj.geometries().create(articulation_geo)

        # Wire an animator callback per non-fixed joint. Fixed joints don't need an
        # animator: their AffineBodyFixedJoint constitution locks the pair statically.
        # * Default drive path: write `aim_angle`/`driving/is_constrained` on the joint
        #   edge itself.
        # * EAC drive path (``use_external_articulation``): one animator on the shared
        #   articulation object writes every joint's `delta_theta_tilde` at once.
        if self.use_external_articulation:
            if self._eac_obj is not None:
                self._animator.insert(
                    self._eac_obj,
                    functools.partial(_animate_eac_delta_theta, weakref.ref(self)),
                )
        else:
            for slot_idx, slot in enumerate(self._articulation_slots):
                if slot["kind"] == "fixed":
                    continue
                joint_obj = slot["obj"]
                if joint_obj is None:
                    continue  # world-anchored FIXED (Task 4 stores obj=None); skip.
                self._animator.insert(
                    joint_obj,
                    functools.partial(_animate_articulation_joint, weakref.ref(self), slot_idx),
                )

    def _convert_cloth_to_uipc(self, model: Model, cloth_triangle_mask: np.ndarray) -> None:
        """Pack the model's non-tet-surface triangles into a single ``libuipc`` FEM shell mesh.

        Only particles actually referenced by a cloth triangle are included in the mesh:
        particles referenced solely by ``Model.tet_indices``/its surface triangles must not
        become orphaned shell vertices, which would both misrepresent them as flat membrane
        (rather than volumetric) and duplicate their simulation via
        :meth:`_convert_tets_to_uipc` (see `_convert_to_uipc`'s duplicate-mapping check).

        Args:
            model: The model being converted.
            cloth_triangle_mask: Boolean mask, shape ``[model.tri_count]``, selecting the
                rows of ``Model.tri_indices``/``tri_materials`` that are true cloth
                triangles, i.e. excluding any tet-FEM surface triangles
                (``self._tet_surface_mask``).
        """
        uipc = self._uipc

        global_triangle_rows = np.flatnonzero(cloth_triangle_mask)
        global_triangles = model.tri_indices.numpy().reshape(-1, 3).astype(np.int64)[global_triangle_rows]

        particle_q = model.particle_q.numpy().astype(np.float64)
        particle_qd = model.particle_qd.numpy().astype(np.float64)
        particle_mass = model.particle_mass.numpy()
        particle_inv_mass = model.particle_inv_mass.numpy()
        tri_materials = model.tri_materials.numpy()[global_triangle_rows]
        edge_indices = model.edge_indices.numpy().reshape(-1, 4)
        edge_bending_properties = model.edge_bending_properties.numpy()

        # Renumber to a dense, locally-contiguous vertex range (`libuipc` does not support a
        # trimesh with unreferenced or globally-indexed vertices); `global_vertices` is sorted
        # ascending, so `local -> global` is a simple array lookup and `global -> local` is a
        # binary search.
        global_vertices = np.unique(global_triangles.reshape(-1))
        global_vertices.sort()
        local_triangles = np.searchsorted(global_vertices, global_triangles).astype(np.int32)

        reference = getattr(model, "ipc_rest_particle_q", particle_q)
        mesh = uipc.geometry.trimesh(reference[global_vertices], local_triangles)
        uipc.geometry.label_surface(mesh)

        lame = uipc.constitution.ElasticModuli2D.lame(1.0, 1.0)
        self._shell.apply_to(mesh, moduli=lame, mass_density=1.0, thickness=self.cloth_thickness)

        uipc.view(mesh.triangles().find("mu"))[:] = tri_materials[:, 0]
        uipc.view(mesh.triangles().find("lambda"))[:] = tri_materials[:, 1]

        local_particle_mass = particle_mass[global_vertices]
        local_particle_inv_mass = particle_inv_mass[global_vertices]

        # `libuipc` rejects non-positive vertex volume outright, but pinned particles may carry
        # zero Newton mass; substitute a placeholder since a fixed vertex's mass never affects
        # the simulated motion.
        is_fixed = local_particle_inv_mass == 0.0
        positive_mass = local_particle_mass[~is_fixed]
        fallback_mass = float(positive_mass.mean()) if positive_mass.size > 0 else 1.0
        volume = np.where(is_fixed, fallback_mass, local_particle_mass)

        uipc.view(mesh.vertices().find("volume"))[:] = volume
        uipc.view(mesh.vertices().find(uipc.builtin.is_fixed))[:] = is_fixed.astype(np.int32)
        uipc.view(mesh.vertices().find(uipc.builtin.velocity))[:] = particle_qd[global_vertices].reshape(-1, 3, 1)

        if self.cloth_bending_yield_stress is not None:
            self._bending.apply_to(
                mesh,
                bending_stiffness=1.0,
                yield_stress=self.cloth_bending_yield_stress,
                hardening_modulus=self.cloth_bending_hardening_modulus,
            )
        elif self.cloth_bending_yield_angle is not None:
            self._bending.apply_to(
                mesh,
                bending_stiffness=1.0,
                yield_threshold=self.cloth_bending_yield_angle,
                hardening_modulus=self.cloth_bending_hardening_modulus,
            )
        else:
            self._bending.apply_to(mesh, bending_stiffness=1.0)
        self._cloth_geometry = mesh
        edge_stiffness = {
            (min(int(v1), int(v2)), max(int(v1), int(v2))): float(stiffness)
            for (v1, v2), stiffness in zip(edge_indices[:, 2:4], edge_bending_properties[:, 0], strict=True)
        }
        # `mesh.edges()` are local-vertex-indexed; a tet surface triangle can never contribute
        # one of these edges (it was excluded from `global_triangles` above), so translating
        # back through `global_vertices` and looking up `edge_stiffness` (built from every
        # `Model.edge_indices` row, cloth and tet-surface alike) cannot cross-match a tet-only
        # edge onto this cloth mesh.
        uipc_edges = mesh.edges().topo().view().reshape(-1, 2)
        bending_stiffness = np.array(
            [
                edge_stiffness.get(
                    (
                        min(int(global_vertices[v1]), int(global_vertices[v2])),
                        max(int(global_vertices[v1]), int(global_vertices[v2])),
                    ),
                    0.0,
                )
                for v1, v2 in uipc_edges
            ]
        )
        uipc.view(mesh.edges().find("bending_stiffness"))[:] = bending_stiffness

        uipc.view(mesh.meta().find("self_collision"))[0] = int(self.cloth_self_collision)

        self._partition_for_mas(mesh)

        obj = self._scene.objects().create("cloth")
        self._create_fem_geometry(obj, mesh, particle_q[global_vertices])

        self._fem_vertex_to_particle.extend(int(g) for g in global_vertices)

    def _partition_for_mas(self, mesh) -> None:
        """Tag ``mesh`` with METIS partitions so ``libuipc`` selects its MAS preconditioner.

        No-op unless ``fem_mas_partition_size`` is set. Must run before the mesh is
        handed to a scene object, since ``libuipc`` only inspects the ``mesh_part``
        vertex attribute during ``world.init()``.
        """
        if self.fem_mas_partition_size is None:
            return
        self._uipc.geometry.mesh_partition(mesh, self.fem_mas_partition_size)

    def _convert_tets_to_uipc(self, model: Model) -> None:
        """Pack each connected component of ``Model.tet_indices`` into its own ``StableNeoHookean`` tetmesh.

        Every tet-connected particle belongs to exactly one component (see
        :func:`_tet_connected_components`); each component gets its own ``libuipc`` scene
        object with vertices renumbered to a dense, locally-contiguous range (``libuipc``
        does not support a tetmesh with unreferenced or globally-indexed vertices).
        """
        uipc = self._uipc

        particle_q = model.particle_q.numpy().astype(np.float64)
        particle_qd = model.particle_qd.numpy().astype(np.float64)
        particle_mass = model.particle_mass.numpy()
        particle_inv_mass = model.particle_inv_mass.numpy()

        tet_indices = self._tet_indices
        tet_materials = self._tet_materials
        tet_volumes = self._tet_volumes
        components = self._tet_components

        # Every vertex of a given tet belongs to the same component by construction
        # (`_tet_connected_components` unions all four vertices of each tet), so the first
        # column alone is enough to label each tet with its component id.
        tet_component = components[tet_indices[:, 0]]
        component_ids = np.unique(components[components >= 0])

        self._tet_geometries = []
        for comp_id in component_ids:
            comp = int(comp_id)
            # `np.flatnonzero` returns indices in ascending order, which is what makes the
            # local vertex numbering below (and thus the resulting geometry) deterministic.
            global_indices = np.flatnonzero(components == comp)
            local_of_global = {int(g): local for local, g in enumerate(global_indices)}

            comp_mask = tet_component == comp
            comp_tet_global = tet_indices[comp_mask]
            comp_volumes = tet_volumes[comp_mask]
            comp_materials = tet_materials[comp_mask]
            local_tet_indices = np.vectorize(local_of_global.__getitem__)(comp_tet_global).astype(np.int32)

            total_mass = float(particle_mass[global_indices].sum())
            total_volume = float(comp_volumes.sum())
            if not np.isfinite(total_mass) or total_mass <= 0.0:
                raise ValueError(
                    f"Tet component {comp} (particles {global_indices.tolist()}) has a non-finite or "
                    f"non-positive total mass ({total_mass}); at least one particle must carry positive mass."
                )
            if not np.isfinite(total_volume) or total_volume <= 0.0:
                raise ValueError(
                    f"Tet component {comp} (particles {global_indices.tolist()}) has a non-finite or "
                    f"non-positive total tet volume ({total_volume})."
                )
            density = total_mass / total_volume

            reference = getattr(model, "ipc_rest_particle_q", particle_q)
            vertices = reference[global_indices]
            mesh = uipc.geometry.tetmesh(vertices, local_tet_indices)
            uipc.geometry.label_surface(mesh)

            # `moduli` seeds every tet's "mu"/"lambda" attribute with this representative value.
            # `apply_to` does not store `moduli`'s raw Lame parameters verbatim: it reparametrizes
            # them via the Stable Neo-Hookean substitution mu' = 4/3 * mu, lambda' = lambda + 5/6 * mu
            # (Smith et al., "Stable Neo-Hookean Flesh Simulation", ACM TOG 2018) before writing
            # "mu"/"lambda". Only overwrite them per-tet below when the component actually carries
            # heterogeneous materials, applying the same substitution so every tet's *effective*
            # stiffness matches what `apply_to` would have produced for that tet's own materials
            # (writing the raw, un-substituted values here would silently understate stiffness).
            lame = uipc.constitution.ElasticModuli.lame(float(comp_materials[0, 1]), float(comp_materials[0, 0]))
            self._tet_constitution.apply_to(mesh, moduli=lame, mass_density=density)

            if not np.all(comp_materials == comp_materials[0]):
                comp_mu = comp_materials[:, 0]
                comp_lambda = comp_materials[:, 1]
                uipc.view(mesh.tetrahedra().find("mu"))[:] = (4.0 / 3.0) * comp_mu
                uipc.view(mesh.tetrahedra().find("lambda"))[:] = comp_lambda + (5.0 / 6.0) * comp_mu

            is_fixed = particle_inv_mass[global_indices] == 0.0
            uipc.view(mesh.vertices().find(uipc.builtin.is_fixed))[:] = is_fixed.astype(np.int32)
            uipc.view(mesh.vertices().find(uipc.builtin.velocity))[:] = particle_qd[global_indices].reshape(-1, 3, 1)

            uipc.view(mesh.meta().find("self_collision"))[0] = int(self.cloth_self_collision)

            self._partition_for_mas(mesh)

            obj = self._scene.objects().create(f"tet_{comp}")
            self._create_fem_geometry(obj, mesh, particle_q[global_indices])
            self._tet_geometries.append(mesh)

            self._fem_vertex_to_particle.extend(int(g) for g in global_indices)

    def _create_fem_geometry(self, obj, mesh, initial_positions):
        """Keep elastic reference metrics while native frame zero starts deformed."""
        if getattr(self.model, "ipc_rest_particle_q", None) is None:
            return obj.geometries().create(mesh)
        rest_mesh = mesh.copy()
        self._uipc.view(mesh.vertices().find(self._uipc.builtin.position)).reshape(-1, 3)[:] = initial_positions
        return obj.geometries().create(mesh, rest_mesh)

    def _write_state_from_uipc(self, state: State, *, state_in: State | None = None) -> None:
        body_count = self.model.body_count
        if body_count > 0:
            self._abd_state_feature.copy_to(self._abd_state_geom)
            transforms = self._abd_state_geom.instances().find(self._uipc.builtin.transform).view()
            velocities = self._abd_state_geom.instances().find(self._uipc.builtin.velocity).view()

            # Bodies without native geometry retain live FK state during stepping;
            # construction and reset still restore the initial model state.
            if state_in is None:
                body_q = self.model.body_q.numpy().astype(np.float32, copy=True)
                body_qd = np.zeros((body_count, 6), dtype=np.float32)
            else:
                body_q = state_in.body_q.numpy().astype(np.float32, copy=True)
                body_qd = state_in.body_qd.numpy().astype(np.float32, copy=True)

            for row, body in enumerate(self._uipc_row_to_body):
                if body < 0:
                    continue
                transform = transforms[row]
                rot = transform[:3, :3]
                pos = transform[:3, 3]
                body_q[body, :3] = pos
                body_q[body, 3:] = _matrix_to_quat_xyzw(rot)

                velocity = velocities[row]
                omega_skew = velocity[:3, :3] @ rot.T
                angular = np.array(
                    [
                        (omega_skew[2, 1] - omega_skew[1, 2]) * 0.5,
                        (omega_skew[0, 2] - omega_skew[2, 0]) * 0.5,
                        (omega_skew[1, 0] - omega_skew[0, 1]) * 0.5,
                    ]
                )
                # `velocity[:3, 3]` is the linear velocity of the body-origin point; convert to
                # the center-of-mass velocity expected by `State.body_qd` (see `integrate_rigid_body`).
                linear_origin = velocity[:3, 3]
                linear_com = linear_origin + np.cross(angular, rot @ self._body_com[body])
                body_qd[body, :3] = linear_com
                body_qd[body, 3:] = angular

            state.body_q.assign(body_q)
            state.body_qd.assign(body_qd)

        if self._has_cloth or self._has_tets:
            self._fem_state_feature.copy_to(self._fem_state_geom)
            positions = self._fem_state_geom.vertices().find(self._uipc.builtin.position).view().reshape(-1, 3)
            velocities = self._fem_state_geom.vertices().find(self._uipc.builtin.velocity).view().reshape(-1, 3)
            mapping = self._fem_vertex_to_particle
            if positions.shape[0] != len(mapping) or velocities.shape[0] != len(mapping):
                raise RuntimeError(
                    f"Internal error: libuipc FEM state has {positions.shape[0]} vertices, but "
                    f"SolverIPC's vertex-to-particle mapping has {len(mapping)} entries."
                )
            # Particles outside every FEM mesh (e.g. particles not part of any tet component)
            # retain their construction-time value rather than being zeroed out.
            particle_q = self._initial_particle_q.copy()
            particle_qd = self._initial_particle_qd.copy()
            particle_q[mapping] = positions.astype(np.float32)
            particle_qd[mapping] = velocities.astype(np.float32)
            state.particle_q.assign(particle_q)
            state.particle_qd.assign(particle_qd)

        if self._articulation_slots:
            # Assemble the joint_q vector from each revolute/prismatic edge's `angle`
            # attribute (Task 2 POC-verified name). Fixed joints have no drive DoF
            # and are skipped (dof_index < 0).
            n_dof = 0
            for slot in self._articulation_slots:
                if slot["dof_index"] >= 0:
                    n_dof = max(n_dof, slot["dof_index"] + 1)
            if n_dof > 0:
                joint_q_np = np.zeros(n_dof, dtype=np.float32)
                for slot in self._articulation_slots:
                    if slot["dof_index"] < 0:
                        continue
                    geom = slot["geo_slot"].geometry()
                    angle_attr = geom.edges().find("angle")
                    if angle_attr is None:
                        continue  # Defensive: constitution didn't create attribute.
                    angle = float(np.asarray(self._uipc.view(angle_attr))[0])
                    joint_q_np[slot["dof_index"]] = angle
                if state.joint_q is None or state.joint_q.shape[0] != n_dof:
                    state.joint_q = wp.array(joint_q_np, dtype=wp.float32)
                else:
                    state.joint_q.assign(joint_q_np)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the ``libuipc`` world by one step and write the result to ``state_out``.

        ``contacts`` is always ignored. ``state_in.body_q`` is read for every
        :class:`~newton.BodyFlags.KINEMATIC` body and used as that body's target pose for
        this step (see the class docstring); every other :class:`~newton.BodyFlags.DYNAMIC`
        body and all cloth/tet-FEM particles are fully owned by the ``libuipc`` world once
        constructed (no two-way coupling in v1), and contact is handled entirely by
        ``libuipc``'s own barrier-based pipeline.

        ``control.tet_activations`` is read (not applied) solely to reject unsupported runtime
        activation: a nonzero value raises :class:`NotImplementedError`, matching the rejection
        of a nonzero initial :attr:`Model.tet_activations` at construction time.

        Reads ``control.joint_target_q`` (when present) for every URDF joint owned by the
        solver's articulation; missing or shorter arrays fall back to ``state_in.joint_q``.
        When constructed with ``use_external_articulation=True``, ``control.joint_target_q``
        is ignored — callers must write per-joint ``delta_theta_tilde`` increments into
        ``self._eac_delta_theta_tilde`` before each :meth:`step` (indexed by
        ``_articulation_slots`` slot position).

        Args:
            state_in: ``body_q`` is read for :class:`~newton.BodyFlags.KINEMATIC` bodies;
                ``joint_q`` is read as a fallback articulation target when
                ``control.joint_target_q`` is absent (see above). Bodies omitted from
                the native world retain their input ``body_q`` and ``body_qd``.
            state_out: ``body_q``/``body_qd`` are overwritten; ``particle_q``/``particle_qd``
                are also overwritten if the model has cloth or tet-FEM particles. Task 7 will
                also write ``joint_q``.
            control: ``tet_activations`` is read-only (see above); a nonzero value raises
                :class:`NotImplementedError` since runtime tet activation is not yet
                supported. ``joint_target_q`` drives the articulation when populated
                (ignored under ``use_external_articulation=True``, see above).
            contacts: Unused.
            dt: Must equal the ``dt`` this solver was constructed with.

        Raises:
            NotImplementedError: If ``control.tet_activations`` is provided and contains a
                nonzero value.
        """
        if abs(dt - self.dt) > 1.0e-9:
            raise ValueError(f"SolverIPC was constructed with dt={self.dt}, but step() was called with dt={dt}.")

        if (
            control is not None
            and control.tet_activations is not None
            and np.any(control.tet_activations.numpy() != 0.0)
        ):
            raise NotImplementedError("SolverIPC does not support a nonzero runtime tet activation.")

        if self._articulation_slots and not self.use_external_articulation:
            n_dof = 0
            for slot in self._articulation_slots:
                if slot["dof_index"] >= 0:
                    n_dof = max(n_dof, slot["dof_index"] + 1)

            target_q_arr = None
            if control is not None and control.joint_target_q is not None:
                target_q_arr = control.joint_target_q.numpy()
            elif state_in.joint_q is not None:
                # Backwards-compat: callers that already run eval_fk externally and
                # want the solver to hold that pose. Reads joint_q as target.
                target_q_arr = state_in.joint_q.numpy()

            if target_q_arr is not None and target_q_arr.shape[0] >= n_dof:
                for slot_idx, slot in enumerate(self._articulation_slots):
                    if slot["dof_index"] < 0:
                        continue
                    self._joint_aim_values[slot_idx] = float(target_q_arr[slot["dof_index"]])

        if self._kinematic_targets:
            body_q_in = state_in.body_q.numpy()
            for b in self._kinematic_targets:
                rot = _quat_xyzw_to_matrix(body_q_in[b, 3:])
                transform = np.eye(4)
                transform[:3, :3] = rot
                transform[:3, 3] = body_q_in[b, :3]
                self._kinematic_targets[b] = transform

        self._world.advance()
        self._world.retrieve()
        self._write_state_from_uipc(state_out, state_in=state_in)

    def reset(self, state: State, world_mask: wp.array | None = None, flags: Any | None = None) -> None:
        """Restore the ``libuipc`` world to its initial (construction-time) state."""
        self._world.recover(0)
        self._world.retrieve()
        self._write_state_from_uipc(state)

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Not supported: :class:`SolverIPC` builds its scene once at construction time."""
        raise NotImplementedError(
            "SolverIPC does not support notify_model_changed(); rebuild the solver after modifying the model."
        )
