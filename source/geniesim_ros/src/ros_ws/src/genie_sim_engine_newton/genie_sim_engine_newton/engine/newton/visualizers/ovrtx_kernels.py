# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Warp kernels for the inline OVRtx visualizer.

The hot-path kernel — :func:`sync_body_q_to_ovrtx_mat44d` — writes Newton's
``state.body_q`` (per-body world-space ``wp.transformf``) directly into an
OVRtx-mapped ``mat44d`` buffer.  The destination buffer is an OVRtx
:class:`AttributeMapping` returned by ``binding.map(device=Device.CUDA)``;
the kernel runs on a Warp CUDA stream that OVRtx waits on via
``unmap(stream=...)``.  No intermediate buffer is allocated and no copy is
done — the kernel's writes go straight into OVRtx's internal Fabric buffer
for the ``"omni:xform"`` attribute.

OVRtx's ``OVRTX_SEMANTIC_XFORM_MAT4x4`` expects USD row-vector
convention (translation in the LAST ROW: ``m[3][0..2]``); see the
upstream OVRtx ``writing-transforms`` skill and the matrix-layout
reference test under ``tests/docs/python/test_attribute_bindings.py``
in the OVRtx repository. Warp's :func:`wp.transform_to_matrix`
produces the column-vector form (translation in the LAST COLUMN), so
we transpose before storing.
"""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def sync_body_q_to_ovrtx_mat44d(
    ovrtx_xforms: wp.array(dtype=wp.mat44d),
    body_indices: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transformf),
):
    """Newton ``body_q`` → OVRtx row-major ``mat44d``.

    One thread per OVRtx-bound prim.  ``body_indices[i]`` selects which
    Newton body's transform feeds OVRtx slot ``i``; for a single-env
    newton-standalone scene this is identity (``i → i``) over
    ``model.body_paths``.

    Args:
        ovrtx_xforms: Mapped OVRtx buffer (``len(body_indices)`` matrices).
            Written in-place; lifetime is the enclosing
            ``with binding.map(...) as mapping`` block in
            :class:`InlineOvrtxVisualizer._render_one_frame`.
        body_indices: Static index map allocated once at startup.
        body_q: Newton's ``state.body_q`` Warp array (live; updated by
            physics every tick).  We rely on a CUDA event recorded by the
            physics thread (``physics_step_event``) and waited on by the
            OVRtx thread's stream so this read sees a committed value.
    """
    i = wp.tid()
    body_idx = body_indices[i]
    transform = body_q[body_idx]
    # transform_to_matrix → column-vector convention; transpose → row-vector
    # (USD / OVRtx OVRTX_SEMANTIC_XFORM_MAT4x4 layout).
    ovrtx_xforms[i] = wp.transpose(wp.mat44d(wp.transform_to_matrix(transform)))


@wp.kernel(enable_backward=False)
def sync_particle_q_slice_to_points(
    out_points: wp.array(dtype=wp.vec3f),
    particle_q: wp.array(dtype=wp.vec3f),
    start: wp.int32,
):
    """Copy ``particle_q[start:start+N]`` → ``out_points[0:N]``.

    Cloth bookkeeping in :mod:`engine.newton.cloth` records each cloth's
    contiguous ``[start, end)`` slice into ``state.particle_q``.  This
    kernel copies one such slice into a per-cloth output buffer that
    OVRtx's ``bind_array_attribute("points", ...)`` consumes.

    No frame transform: per the cloth bookkeeping in the isaac_scene stage
    (``genie_sim_engine_isaacsim/assemble/_scene_impl.py``),
    the cloth prim is authored at identity so writing world-space points
    into the prim's ``points`` attribute is correct (USD interprets
    ``points`` as the prim's LOCAL frame, but local == world when the
    parent xform stack is identity).

    Args:
        out_points: Per-cloth target buffer of length N (= end - start).
            Owned by :class:`InlineOvrtxVisualizer`; passed every frame to
            the OVRtx ``points`` array binding.
        particle_q: Newton's ``state.particle_q`` (Warp ``wp.vec3f`` array).
        start: First-particle index of this cloth.  ``end`` is implied by
            ``out_points.shape[0]``.
    """
    i = wp.tid()
    out_points[i] = particle_q[start + i]


@wp.kernel(enable_backward=False)
def reconstruct_tet_bound_points(
    out_points: wp.array(dtype=wp.vec3f),
    particle_q: wp.array(dtype=wp.vec3f),
    global_particle_indices: wp.array(dtype=wp.int32),
    barycentric_weights: wp.array(dtype=wp.float32),
):
    """Reconstruct authored-order world-space Mesh points from FEM particles.

    ``global_particle_indices`` and ``barycentric_weights`` are flattened
    ``[vertex_count, 4]`` arrays. The mapping is fixed at build time; only
    ``particle_q`` changes per physics step. Writing world-space points lets
    OVRtx and the future NPZ writer consume the same reconstructed array when
    the target Mesh has ``omni:resetXformStack=True`` and identity xform.
    """
    vertex = wp.tid()
    base = vertex * 4
    out_points[vertex] = (
        particle_q[global_particle_indices[base]] * barycentric_weights[base]
        + particle_q[global_particle_indices[base + 1]] * barycentric_weights[base + 1]
        + particle_q[global_particle_indices[base + 2]] * barycentric_weights[base + 2]
        + particle_q[global_particle_indices[base + 3]] * barycentric_weights[base + 3]
    )
