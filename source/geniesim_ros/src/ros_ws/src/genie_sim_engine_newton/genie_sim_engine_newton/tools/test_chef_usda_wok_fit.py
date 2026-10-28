#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure geometry regression for the USDA-derived wok proxy fit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CHEF = _HERE.parents[2] / "genie_sim_bringup" / "config" / "scene_flat_g2_sp_chef.py"


def _load_chef():
    spec = importlib.util.spec_from_file_location("chef_usda_wok_fit", _CHEF)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


chef = _load_chef()


def test_usda_mode_is_opt_in_for_the_shared_legacy_plugin():
    assert chef.CONFIG.usd_objects.enabled is False


def test_fit_recovers_transformed_bowl_and_rejects_handle_extent():
    n_az = 32
    n_radial = 12
    rim_radius = 0.16
    depth = 0.09
    vertices, triangles = chef._make_wok(
        rim_r=rim_radius,
        depth=depth,
        thickness=0.003,
        n_az=n_az,
        n_radial=n_radial,
    )
    surface_triangle_count = 2 * n_az * n_radial
    bowl_triangles = triangles[: 2 * surface_triangle_count]

    # Match the authored asset convention: body-local +Y points out of the
    # bowl, while the long handle extends along body-local +X.
    center = np.array([-0.086, 0.047, 0.0])
    transformed = np.column_stack((vertices[:, 0], vertices[:, 2], -vertices[:, 1])) + center

    handle_vertices = np.array(
        [
            [0.09, 0.03, -0.01],
            [0.25, 0.03, -0.01],
            [0.25, 0.03, 0.01],
            [0.09, 0.03, 0.01],
        ]
    )
    offset = len(transformed)
    all_vertices = np.concatenate((transformed, handle_vertices), axis=0)
    all_triangles = np.concatenate(
        (bowl_triangles, np.array([[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]])),
        axis=0,
    )
    body_axes_world = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    fit = chef._fit_usda_wok_bowl(all_vertices, all_triangles, body_axes_world)

    assert np.allclose(fit["center"], center, atol=2.0e-3)
    assert np.allclose(fit["up"], [0.0, 1.0, 0.0])
    assert np.allclose(fit["handle_axis"], [1.0, 0.0, 0.0])
    assert abs(fit["rim_radius"] - rim_radius) < 2.0e-3
    assert abs(fit["depth"] - depth) < 2.0e-3
    assert 0.0095 <= fit["handle_radius"] <= 0.0115
    assert fit["component_count"] == 3


def test_fit_accepts_decimated_bowl_joined_at_rim():
    rim_radius = 0.16
    depth = 0.09
    vertices, triangles = chef._make_wok(
        rim_r=rim_radius,
        depth=depth,
        thickness=0.003,
        n_az=32,
        n_radial=12,
    )

    # Keep the rim sidewall so the inner and outer surfaces form one connected
    # shell, matching decimated production assets. The handle remains a
    # separate component and must not be mistaken for the second bowl surface.
    center = np.array([-0.086, 0.047, 0.0])
    transformed = np.column_stack((vertices[:, 0], vertices[:, 2], -vertices[:, 1])) + center
    handle_vertices = np.array(
        [
            [0.09, 0.03, -0.01],
            [0.25, 0.03, -0.01],
            [0.25, 0.03, 0.01],
            [0.09, 0.03, 0.01],
        ]
    )
    offset = len(transformed)
    all_vertices = np.concatenate((transformed, handle_vertices), axis=0)
    all_triangles = np.concatenate(
        (
            triangles,
            np.array([[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]]),
        ),
        axis=0,
    )
    body_axes_world = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    fit = chef._fit_usda_wok_bowl(all_vertices, all_triangles, body_axes_world)

    assert np.allclose(fit["center"], center, atol=2.0e-3)
    assert np.allclose(fit["up"], [0.0, 1.0, 0.0])
    assert np.allclose(fit["handle_axis"], [1.0, 0.0, 0.0])
    assert abs(fit["rim_radius"] - rim_radius) < 4.0e-3
    assert abs(fit["depth"] - depth) < 4.0e-3
    assert fit["component_count"] == 2


def test_proxy_frame_maps_canonical_bowl_and_handle_axes_to_usda():
    fit = {
        "center": np.array([0.1, 0.2, 0.3]),
        "up": np.array([0.0, 1.0, 0.0]),
        "handle_axis": np.array([1.0, 0.0, 0.0]),
    }

    center, basis = chef._usda_wok_proxy_frame(fit)
    transformed = chef._transform_wok_proxy_points(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        center,
        basis,
    )

    assert np.allclose(basis.T @ basis, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(basis) > 0.0
    assert np.allclose(transformed[0], fit["center"])
    assert np.allclose(transformed[1] - transformed[0], fit["up"])
    assert np.allclose(transformed[2] - transformed[0], fit["handle_axis"])
