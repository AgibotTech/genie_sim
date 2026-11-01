# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

import math

import numpy as np

"""
scene_name: cloth_fold_dual_g2_tshirt
description:
    Reproduces scene_0/scene.usda for the G2 dual-arm cloth-folding task on
    the double-layer T-shirt asset (cloth_001): a workspace table (table_000),
    an aligned invisible collision proxy (table_000_collision_proxy), and a
    FEM-baked T-shirt lying flat on top. The robot folds the cloth in half by
    grasping two corners.

Canonical pose (matches the original scene_0/scene.usda, including the
manual orientation fix applied to the cloth):
    table_000  : library_call leaves its bbox bottom at z=0; that automatic
                 z_offset shift produces the (0, 0, 0.367) entry seen in
                 scene_info. No further translation is applied here.
    table_000_collision_proxy: its origin pose is aligned exactly to the table
                 origin pose before the shared workspace offset is applied.
    cloth_001  : centred at (0, 0, 0.7433330051145568), with a 90-deg
                 rotation about world +Z — quat (w,x,y,z) =
                 (sqrt(2)/2, 0, 0, sqrt(2)/2) — applied BEFORE the yaw noise.
                 This base rotation is what scene_0/scene.usda hand-edited
                 onto the cloth so the T-shirt is oriented correctly for the
                 dual-arm grasp.

Generalization (shared by table, proxy, and cloth):
    - workspace X offset: dx in [0.0, +0.20] m
    - workspace Y offset: dy in [-0.05, +0.05] m
    - cloth yaw noise about world +Z (added ON TOP of the base 90-deg yaw):
                                   dyaw in [-5 deg, +5 deg]
    Workspace offsets are sampled uniformly. Yaw is drawn from
    N(0, sigma = limit/2) and clipped to its limit.
"""


# --- Canonical pose ----------------------------------------------------------
CLOTH_XYZ = (0.0, 0.0, 0.7433330051145568)

# Base cloth orientation: 90-deg rotation about world +Z.
# Both base yaw and noise yaw fold into one rotation about (0, 0, 1).
_BASE_YAW = math.pi / 2.0


# --- Generalization limits ---------------------------------------------------
SCENE_DX_MAX = 0.20
SCENE_DY_LIMIT = 0.05
DYAW_LIMIT = math.radians(5.0)


def _trunc_normal(limit, sigma_ratio=0.5):
    """Sample N(0, sigma = limit*sigma_ratio) clipped to [-limit, +limit].

    sigma_ratio=0.5 puts ~95% of the unclipped distribution inside the limit,
    so the result concentrates near 0 but occasionally reaches the boundary,
    never beyond.
    """
    sigma = float(limit) * float(sigma_ratio)
    val = float(np.random.normal(0.0, sigma))
    return float(np.clip(val, -limit, +limit))


@register()
def cloth_fold_tshirt_scene() -> Shape:
    scene_dx = float(np.random.uniform(0.0, SCENE_DX_MAX))
    scene_dy = float(np.random.uniform(-SCENE_DY_LIMIT, SCENE_DY_LIMIT))
    workspace_translation = translation_matrix([scene_dx, scene_dy, 0.0])

    # --- Workspace table -----------------------------------------------------
    # NOTE: helper.usd() already shifts the loaded shape so its bbox bottom
    # sits at z=0 (top at table_height ~0.732). The (0, 0, 0.367) entry seen in
    # scene_0/scene_info.json corresponds to that automatic z_offset — adding
    # another translation here would lift the table further and push its top
    # above the cloth.
    table = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "wood", "rectangular", "furniture"],
    )

    # Keep the proxy's asset origin exactly on the table's asset origin. Their
    # local bboxes differ, so each library_call receives a different automatic
    # z_offset and must be aligned before applying the shared XY offset.
    proxy = library_call(
        "usd",
        oid="table_000_collision_proxy",
        keywords=["table_collision_proxy", "motion_gen_obstacle", "collision_proxy"],
    )
    table_origin = np.asarray(get_object_info(table)["center"], dtype=np.float64)
    proxy_origin = np.asarray(get_object_info(proxy)["center"], dtype=np.float64)
    proxy = transform_shape(proxy, translation_matrix(table_origin - proxy_origin))

    table = transform_shape(table, workspace_translation)
    proxy = transform_shape(proxy, workspace_translation)

    # --- FEM-baked double-layer T-shirt cloth -------------------------------
    cloth = library_call(
        "usd",
        oid="cloth_001",
        keywords=[
            "target_cloth",
            "cloth",
            "tshirt",
            "fem",
            "deformable",
            "double_layer",
            "fold_target",
        ],
    )

    # 1) Base orientation + yaw noise: a single rotation about world +Z whose
    #    angle = base_yaw (90 deg) + noise. Pivot at the cloth's current
    #    centre so position is untouched by the rotation.
    c0 = compute_shape_center(cloth)
    pivot = (float(c0[0]), float(c0[1]), float(c0[2]))
    dyaw = _trunc_normal(DYAW_LIMIT)
    total_yaw = _BASE_YAW + dyaw
    cloth = transform_shape(cloth, rotation_matrix(total_yaw, (0.0, 0.0, 1.0), pivot))

    # 2) Apply the same XY translation as table + proxy, preserving the
    #    cloth's canonical position relative to the work surface.
    target_x = CLOTH_XYZ[0] + scene_dx
    target_y = CLOTH_XYZ[1] + scene_dy
    target_z = CLOTH_XYZ[2]
    cloth = transform_shape(
        cloth,
        translation_matrix(
            [target_x - pivot[0], target_y - pivot[1], target_z - pivot[2]]
        ),
    )

    return concat_shapes(table, proxy, cloth)


@register()
def root_scene() -> Shape:
    return library_call("cloth_fold_tshirt_scene")
