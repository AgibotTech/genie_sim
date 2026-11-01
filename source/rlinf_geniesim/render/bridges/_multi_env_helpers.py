# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""USD helpers for per-environment transforms, body names and cameras.

Adapted from legacy ``rl_renderer.py``; that source is not included here.
These helpers use pxr without Kit runtime dependencies.
"""

from __future__ import annotations

from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Frame-id normalization
# ---------------------------------------------------------------------------


def normalize_frame_id(frame_id: str) -> str:
    """Convert a ROS TF child_frame_id to a prim-relative path under an env root.

    Adapted from ``rl_renderer.py::EnvTFSubscriber._normalize_frame_id``.

    Expected per-env layout::

        /World/envs/env_i/objects/...
        /World/envs/env_i/robot/...

    MuJoCo / ROS may publish either prim-relative or absolute USD-ish
    paths like ``"/World/objects/foo"`` or ``"World/objects/foo"``;
    this helper normalizes them so callers can prepend the env root.

    Also folds the lower-case ``objects/`` (MuJoCo convention) to the
    upper-case ``Objects/`` used in our scene USDs.
    """
    s = (frame_id or "").strip()
    s = s.lstrip("/")
    if s.startswith("World/"):
        s = s[len("World/") :]
    if s.startswith("objects/"):
        s = "Objects/" + s[len("objects/") :]
    return s


# ---------------------------------------------------------------------------
# Robot base pose
# ---------------------------------------------------------------------------


def set_robot_base_pose(
    stage,
    robot_prim_path: str,
    position: list[float],
    quaternion: list[float],
) -> bool:
    """Set the initial world-space pose of a robot root prim via USD xform ops.

    Adapted from ``rl_renderer.py::RLRenderer._set_robot_base_pose``.

    Args:
        stage: ``pxr.Usd.Stage``.
        robot_prim_path: e.g. ``/World/envs/env_1/robot``.
        position: ``[x, y, z]``.
        quaternion: ``[w, x, y, z]`` (note the W-first ordering).

    Returns:
        ``True`` if the prim was found and updated, ``False`` otherwise.
    """
    # Local imports keep the module importable on hosts without pxr.Usd
    # (the heavy USD libs only get pulled in when callers actually use
    # this helper).
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(robot_prim_path)
    if not prim.IsValid():
        return False

    xf = UsdGeom.Xformable(prim)
    # Reset whatever xform ops the reference may have authored.
    xf.ClearXformOpOrder()

    translate_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(*position))

    w, x, y, z = quaternion
    existing_orient_attr = prim.GetAttribute("xformOp:orient")
    existing_type = str(existing_orient_attr.GetTypeName()) if existing_orient_attr.IsValid() else ""
    use_quatd = existing_type == "quatd"

    if use_quatd:
        orient_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        orient_op.Set(Gf.Quatd(w, x, y, z))
    else:
        orient_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
        orient_op.Set(Gf.Quatf(w, x, y, z))

    return True


# ---------------------------------------------------------------------------
# body_name → prim path map
# ---------------------------------------------------------------------------


def build_body_name_map(stage, env_root_path: str) -> dict[str, str]:
    """Map leaf body-name → prim-relative path under ``env_root_path``.

    Adapted from ``rl_renderer.py::RLRenderer._build_body_name_map``.

    The legacy code first looked for a hand-crafted ``<scene>_body_map.json``
    sidecar; that's a deployment artifact that doesn't exist for our
    multi-env stages, so we always fall back to "traverse every prim
    under env_root and index by leaf name".

    Result example for ``env_root_path == "/World/envs/env_0"``::

        {
            "base_link": "robot/base_link",
            "head_link3": "robot/Geometry/head_link3",
            ...
        }

    The shallowest hit wins so top-level prims aren't masked by deeper
    children with the same leaf name.
    """
    prefix = env_root_path.rstrip("/") + "/"
    leaf_map: dict[str, str] = {}
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if not path_str.startswith(prefix):
            continue
        rel = path_str[len(prefix) :]
        leaf = rel.split("/")[-1]
        if leaf and leaf not in leaf_map:
            leaf_map[leaf] = rel
    return leaf_map


# ---------------------------------------------------------------------------
# Camera prim lookup
# ---------------------------------------------------------------------------


def find_first_camera_under(stage, root_path: str) -> Optional[str]:
    """Return the prim path of the first ``UsdGeom.Camera`` under ``root_path``.

    Adapted from ``rl_renderer.py::RLRenderer._find_first_camera_under_env``.
    Useful when the configured camera prim path is missing — we fall back
    to "any camera in this env".
    """
    from pxr import UsdGeom

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root_path.rstrip("/") + "/"):
            continue
        if prim.IsA(UsdGeom.Camera):
            return path
    return None


# ---------------------------------------------------------------------------
# Topic name parsing — env_id / cam_id round-trip
# ---------------------------------------------------------------------------


def env_topic_name(base_topic: str, env_id: int) -> str:
    """Return a topic name with an env-id segment.

    Convention used by ``scene_assembly``::

        base="/genie_sim/head_front_camera_rgb", env_id=0
            → "/genie_sim/head_front_camera_rgb"             (env 0 keeps the original)
        base="/genie_sim/head_front_camera_rgb", env_id=1
            → "/genie_sim/env_1_head_front_camera_rgb"
    """
    if env_id == 0:
        return base_topic
    # Strip leading '/', inject after the namespace prefix.
    parts = base_topic.lstrip("/").split("/", 1)
    if len(parts) == 1:
        return f"/env_{env_id}_{parts[0]}"
    ns, rest = parts
    return f"/{ns}/env_{env_id}_{rest}"


def parse_env_from_topic(topic: str) -> int:
    """Inverse of ``env_topic_name``: extract ``env_id`` from a topic name.

    Looks for ``/env_<N>_`` segment; returns 0 if not found (which is
    the convention for the env-0 topic).
    """
    parts = topic.split("/")
    for seg in parts:
        if seg.startswith("env_") and len(seg) > 4:
            tail = seg[4:]
            # tail may be "<num>" (single-segment) or "<num>_<rest>"
            # but the topic-name convention only uses "env_<num>_<rest>".
            num_str = tail.split("_", 1)[0]
            try:
                return int(num_str)
            except ValueError:
                continue
    return 0


__all__ = [
    "build_body_name_map",
    "env_topic_name",
    "find_first_camera_under",
    "normalize_frame_id",
    "parse_env_from_topic",
    "set_robot_base_pose",
]
