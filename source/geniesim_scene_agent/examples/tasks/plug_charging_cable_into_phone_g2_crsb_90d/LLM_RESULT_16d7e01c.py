# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

import math
import numpy as np


"""Scene generalization for ``dual_arm_pick_phone_plug_to_chest``.

The poses below are the object-origin world poses recorded in scene_0.  Each
generated scene samples two absolute world anchors:

* phone holder: x [-0.07, 0.06], y [0.10, 0.23], world yaw [-90, 0] deg
* power strip:  x [0.08, 0.15], y [-0.26, -0.20], world yaw [45, 90] deg

The phone follows the holder rigidly and the plug follows the power strip
rigidly, using their complete scene_0 relative transforms.
"""


# scene_0 origin poses: ((x, y, z), (w, x, y, z))
TABLE_SCENE0_POSE = (
    (0.0, 0.0, 0.365),
    (1.0, 0.0, 0.0, 0.0),
)
HOLDER_SCENE0_POSE = (
    (-0.02367731480078447, 0.16747923929253206, 0.7879999999999998),
    (0.916685, 0.0, 0.0, -0.3996106),
)
PHONE_SCENE0_POSE = (
    (-0.03173223584625647, 0.18072481903091184, 0.8173358623744371),
    (0.320745, 0.41770205, 0.16048802, -0.8348001),
)
POWERSTRIP_SCENE0_POSE = (
    (0.15286392093090337, -0.35164348774994253, 0.799999999999975),
    (0.86539435, 0.0, 0.0, 0.5010914),
)
PLUG_SCENE0_POSE = (
    (0.24815880639172458, -0.44641939485924786, 0.741471754768551),
    (0.657619, 0.68378204, -0.22901703, -0.21801503),
)

HOLDER_X_RANGE = (-0.07, 0.06)
HOLDER_Y_RANGE = (0.10, 0.23)
HOLDER_WORLD_YAW_DEG_RANGE = (-90.0, 0.0)

POWERSTRIP_X_RANGE = (0.08, 0.15)
POWERSTRIP_Y_RANGE = (-0.26, -0.20)
POWERSTRIP_WORLD_YAW_DEG_RANGE = (45.0, 90.0)


ASSET_KEYWORDS = {
    "table_000": ["left_table", "table", "workspace_table", "left", "furniture"],
    "benchmark_phone_holder_000": ["phone_holder", "holder", "stand", "left"],
    "benchmark_phone_001": ["target_phone", "phone", "left_pick_target", "left"],
    "benchmark_plug_017": ["target_plug", "plug", "right_pick_target", "right"],
    "benchmark_power_strip_storage_box_000": [
        "power_strip_storage_box",
        "storage_box",
        "right",
    ],
}


def _quat_normalize(q_wxyz):
    q = np.asarray(q_wxyz, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("A pose quaternion cannot be zero")
    return tuple(q / norm)


def _quat_multiply(lhs_wxyz, rhs_wxyz):
    """Hamilton product for two wxyz quaternions."""
    w1, x1, y1, z1 = lhs_wxyz
    w2, x2, y2, z2 = rhs_wxyz
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_conjugate(q_wxyz):
    w, x, y, z = q_wxyz
    return (w, -x, -y, -z)


def _quat_rotate(q_wxyz, vector_xyz):
    """Rotate a vector by a normalized wxyz quaternion."""
    q = _quat_normalize(q_wxyz)
    vector_quat = (0.0, *np.asarray(vector_xyz, dtype=float))
    rotated = _quat_multiply(_quat_multiply(q, vector_quat), _quat_conjugate(q))
    return np.asarray(rotated[1:], dtype=float)


def _relative_pose(parent_pose, child_pose):
    """Return ``child`` expressed in ``parent`` coordinates."""
    parent_xyz, parent_q = parent_pose
    child_xyz, child_q = child_pose
    parent_q = _quat_normalize(parent_q)
    child_q = _quat_normalize(child_q)
    parent_q_inv = _quat_conjugate(parent_q)
    relative_xyz = _quat_rotate(
        parent_q_inv,
        np.asarray(child_xyz, dtype=float) - np.asarray(parent_xyz, dtype=float),
    )
    relative_q = _quat_normalize(_quat_multiply(parent_q_inv, child_q))
    return (tuple(relative_xyz), relative_q)


def _compose_pose(parent_pose, child_in_parent_pose):
    """Compose an absolute parent pose with a child-relative pose."""
    parent_xyz, parent_q = parent_pose
    relative_xyz, relative_q = child_in_parent_pose
    parent_q = _quat_normalize(parent_q)
    world_xyz = np.asarray(parent_xyz, dtype=float) + _quat_rotate(
        parent_q, relative_xyz
    )
    world_q = _quat_normalize(_quat_multiply(parent_q, relative_q))
    return (tuple(world_xyz), world_q)


def _absolute_yaw_pose(x, y, z, yaw_degrees):
    """Construct a world pose whose rotation is an absolute world-Z yaw."""
    half_yaw = math.radians(float(yaw_degrees)) / 2.0
    return (
        (float(x), float(y), float(z)),
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
    )


# These are fixed reference transforms, computed from the current scene_0.
PHONE_IN_HOLDER_POSE = _relative_pose(HOLDER_SCENE0_POSE, PHONE_SCENE0_POSE)
PLUG_IN_POWERSTRIP_POSE = _relative_pose(
    POWERSTRIP_SCENE0_POSE, PLUG_SCENE0_POSE
)


def _sample_scene_poses(uniform=None):
    """Sample origin poses for one scene; ``uniform`` is injectable for tests."""
    if uniform is None:
        uniform = np.random.uniform

    holder_pose = _absolute_yaw_pose(
        uniform(*HOLDER_X_RANGE),
        uniform(*HOLDER_Y_RANGE),
        HOLDER_SCENE0_POSE[0][2],
        uniform(*HOLDER_WORLD_YAW_DEG_RANGE),
    )
    powerstrip_pose = _absolute_yaw_pose(
        uniform(*POWERSTRIP_X_RANGE),
        uniform(*POWERSTRIP_Y_RANGE),
        POWERSTRIP_SCENE0_POSE[0][2],
        uniform(*POWERSTRIP_WORLD_YAW_DEG_RANGE),
    )

    return {
        "table_000": TABLE_SCENE0_POSE,
        "benchmark_phone_holder_000": holder_pose,
        "benchmark_phone_001": _compose_pose(holder_pose, PHONE_IN_HOLDER_POSE),
        "benchmark_plug_017": _compose_pose(
            powerstrip_pose, PLUG_IN_POWERSTRIP_POSE
        ),
        "benchmark_power_strip_storage_box_000": powerstrip_pose,
    }


def _wxyz_to_xyzw(q_wxyz):
    w, x, y, z = _quat_normalize(q_wxyz)
    return (x, y, z, w)


def _place_asset(oid, world_pose, keywords) -> Shape:
    """Align an asset's origin marker to an absolute world pose.

    ``gen_scene_layout_info`` serializes the transform of the asset's
    ``origin`` marker, so placement must use that same marker.  In particular,
    the plug bbox is centred far down its cable and is not a valid pose anchor.
    """
    world_xyz, world_quat_wxyz = world_pose
    shape = library_call("usd", oid=oid, keywords=keywords)

    origin_before = np.asarray(get_object_info(shape)["center"], dtype=float)
    angle, axis = quaternion_to_angle_direction(_wxyz_to_xyzw(world_quat_wxyz))
    shape = transform_shape(
        shape,
        rotation_matrix(float(angle), tuple(axis), tuple(origin_before)),
    )

    origin_after = np.asarray(get_object_info(shape)["center"], dtype=float)
    translation = np.asarray(world_xyz, dtype=float) - origin_after
    return transform_shape(shape, translation_matrix(tuple(translation)))


@register()
def collected_background_layout() -> Shape:
    poses = _sample_scene_poses()
    order = (
        "table_000",
        "benchmark_phone_holder_000",
        "benchmark_phone_001",
        "benchmark_plug_017",
        "benchmark_power_strip_storage_box_000",
    )
    return concat_shapes(
        *[
            _place_asset(oid, poses[oid], ASSET_KEYWORDS[oid])
            for oid in order
        ]
    )


@register()
def root_scene() -> Shape:
    return library_call("collected_background_layout")
