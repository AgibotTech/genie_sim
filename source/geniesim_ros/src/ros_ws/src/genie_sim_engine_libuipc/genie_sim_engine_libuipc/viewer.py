# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton SDK GL view for the libuipc state bridge."""

import os


def create_viewer(sim, render_hz: float, params, logger) -> object | None:
    """Instantiate ViewerGL after the engine is built.

    Returns the viewer, or None if no display is available or init fails.

    Reads optional ``viewer_camera_json`` from ``params`` -- a JSON-encoded
    ``{"pos":[x,y,z], "lookat":[x,y,z], "pitch":deg, "yaw":deg}`` originating
    in the scene YAML's ``viewer_camera`` block.  ``lookat`` (when present)
    takes precedence over pitch/yaw and also sets the orbit pivot, so RMB
    drag rotates around the chosen target.  Missing / empty / unparseable
    JSON keeps Newton's built-in default pose.

    The JSON-string indirection sidesteps ``launch_ros``'s parameter pipeline,
    which silently drops list-typed YAML values; the same trick is already
    used for ``init_joint_pos_json``.  See ``physics_backend.launch.py``.
    """
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not has_display:
        logger.warn(
            "physics_engine_visualizer=newton requested but no $DISPLAY / $WAYLAND_DISPLAY detected. "
            "Running headless. Set DISPLAY=:0 or use a virtual framebuffer (Xvfb)."
        )
        return None

    def _xyz(seq):
        try:
            xyz = [float(v) for v in seq]
        except (TypeError, ValueError):
            return None
        return xyz if len(xyz) == 3 else None

    raw_json = str((params or {}).get("viewer_camera_json", "") or "").strip()
    cam_cfg: dict = {}
    if raw_json and raw_json != "{}":
        import json as _json

        try:
            parsed = _json.loads(raw_json)
            if isinstance(parsed, dict):
                cam_cfg = parsed
        except ValueError as exc:
            logger.warn(f"viewer_camera_json parse failed: {exc!r}; using default pose.")

    try:
        from newton.viewer import ViewerGL

        viewer = ViewerGL(
            width=1280,
            height=720,
            vsync=False,
            headless=False,
        )
        viewer.set_model(sim._model)

        cam_pos = _xyz(cam_cfg.get("pos", []))
        if cam_pos is not None:
            try:
                import warp as wp

                pos_vec = wp.vec3(cam_pos[0], cam_pos[1], cam_pos[2])
                lookat = _xyz(cam_cfg.get("lookat", []))

                if lookat is not None:
                    viewer.camera.pos = viewer.camera._as_vec3(pos_vec)
                    viewer.camera.look_at((lookat[0], lookat[1], lookat[2]))
                    logger.info(f"newton GL viewer cam pos={tuple(cam_pos)} lookat={tuple(lookat)}")
                else:
                    pitch = float(cam_cfg.get("pitch", 0.0))
                    yaw = float(cam_cfg.get("yaw", -180.0))
                    viewer.set_camera(pos=pos_vec, pitch=pitch, yaw=yaw)
                    logger.info(f"newton GL viewer cam pos={tuple(cam_pos)} pitch={pitch:.1f} yaw={yaw:.1f}")
            except Exception as exc:
                logger.warn(f"newton GL viewer set_camera failed: {exc!r}; using default pose.")
        elif cam_cfg:
            logger.warn(f"viewer_camera.pos missing or invalid in scene yaml ({cam_cfg!r}); using default pose.")

        logger.info(f"newton GL viewer ready (render_hz={render_hz:.0f})")
        return viewer
    except Exception as exc:
        logger.warn(f"newton GL viewer init failed: {exc}. Running headless.")
        return None
