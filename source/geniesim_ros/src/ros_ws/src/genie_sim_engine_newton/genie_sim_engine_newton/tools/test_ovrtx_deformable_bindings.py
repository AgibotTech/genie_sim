#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure contract tests for multi-mesh OVRtx deformable bindings."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_MODULE = _HERE.parent / "engine" / "newton" / "visualizers" / "deformable_bindings.py"
_KERNELS = _HERE.parent / "engine" / "newton" / "visualizers" / "ovrtx_kernels.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ovrtx_deformable_bindings_test", _MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding(path: str, start: int, count: int, indices) -> dict:
    indices = np.asarray(indices, dtype=np.int32)
    return {
        "mesh_prim_path": path,
        "particle_start": start,
        "particle_count": count,
        "vertex_count": len(indices),
        "mapping_kind": "tet_barycentric_v1",
        "tet_particle_indices": indices,
        "barycentric_weights": np.full(indices.shape, 0.25, dtype=np.float32),
    }


def test_prepare_bindings_preserves_paths_and_globalizes_particle_indices():
    module = _load_module()
    prepared = module.prepare_deformable_bindings(
        [
            _binding("/World/objects/noodle/entity/body/visual", 10, 8, [[0, 1, 2, 3]]),
            _binding("/World/objects/carrot/entity/body/visual", 30, 8, [[4, 5, 6, 7]]),
        ],
        particle_count=40,
    )

    assert [item.prim_path for item in prepared] == [
        "/World/objects/noodle/entity/body/visual",
        "/World/objects/carrot/entity/body/visual",
    ]
    assert prepared[0].global_particle_indices.tolist() == [[10, 11, 12, 13]]
    assert prepared[1].global_particle_indices.tolist() == [[34, 35, 36, 37]]
    assert all(item.barycentric_weights.dtype == np.float32 for item in prepared)


@pytest.mark.parametrize(
    "bindings, message",
    [
        (
            [
                _binding("/World/objects/a/entity/body/visual", 0, 4, [[0, 1, 2, 3]]),
                _binding("/World/objects/a/entity/body/visual", 4, 4, [[0, 1, 2, 3]]),
            ],
            "duplicate deformable Mesh prim path",
        ),
        (
            [_binding("/World/objects/a/entity/body/visual", 0, 4, [[0, 1, 2, 4]])],
            "outside its particle slice",
        ),
        (
            [_binding("/World/objects/a/entity/body/visual", 6, 4, [[0, 1, 2, 3]])],
            "exceeds model particle count",
        ),
    ],
)
def test_prepare_bindings_rejects_ambiguous_or_out_of_range_mappings(bindings, message):
    module = _load_module()
    with pytest.raises(ValueError, match=message):
        module.prepare_deformable_bindings(bindings, particle_count=8)


def test_prepare_bindings_rejects_non_barycentric_weights():
    module = _load_module()
    binding = _binding("/World/objects/a/entity/body/visual", 0, 4, [[0, 1, 2, 3]])
    binding["barycentric_weights"][0] = [0.4, 0.4, 0.4, -0.2]

    with pytest.raises(ValueError, match="barycentric weights"):
        module.prepare_deformable_bindings([binding], particle_count=4)


def test_reconstruction_kernel_outputs_authored_order_world_points(tmp_path):
    import warp as wp

    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    spec = importlib.util.spec_from_file_location("ovrtx_deformable_kernels_test", _KERNELS)
    assert spec and spec.loader
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)

    particles = wp.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=wp.vec3f,
        device="cpu",
    )
    indices = wp.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=wp.int32, device="cpu")
    weights = wp.array(
        [0.25, 0.25, 0.25, 0.25, 0.1, 0.2, 0.3, 0.4],
        dtype=wp.float32,
        device="cpu",
    )
    output = wp.empty(2, dtype=wp.vec3f, device="cpu")

    wp.launch(
        kernels.reconstruct_tet_bound_points,
        dim=2,
        inputs=[output, particles, indices, weights],
        device="cpu",
    )

    assert np.allclose(output.numpy(), [[0.5, 0.5, 0.5], [0.6, 0.4, 0.2]])


def test_world_points_are_converted_to_mesh_local_through_parent_hierarchy():
    module = _load_module()
    parent_pose_world = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mesh_pose_parent = np.array(
        [
            [1.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, -1.0, -0.5],
            [0.0, 1.0, 0.0, 0.75],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected_local = np.array(
        [[0.0, 0.0, 0.0], [0.2, -0.3, 0.4], [-0.5, 0.1, 0.8]],
        dtype=np.float64,
    )
    mesh_pose_world = parent_pose_world @ mesh_pose_parent
    world = expected_local @ mesh_pose_world[:3, :3].T + mesh_pose_world[:3, 3]

    actual_local = module.world_points_to_mesh_local(
        world,
        parent_pose_world=parent_pose_world,
        mesh_pose_parent=mesh_pose_parent,
    )

    assert actual_local.dtype == np.float32
    assert np.allclose(actual_local, expected_local, atol=1.0e-6)
