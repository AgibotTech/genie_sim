# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Native Isaac Sim sensor and Replicator factories.
Import sensor classes lazily so the module can load before SimulationApp.
Factories return native sensors without initializing them; callers initialize
after SimulationContext enters PLAY."""

from __future__ import annotations

from typing import Any, Tuple


# ---------------------------------------------------------------------------
# Sensor factories forward keyword arguments to native constructors.
# ---------------------------------------------------------------------------


def make_camera(prim_path: str, *, resolution: Tuple[int, int], **kwargs) -> Any:
    """Create a native Camera, forwarding resolution and additional keyword arguments."""
    from isaacsim.sensors.camera import Camera  # noqa: PLC0415

    return Camera(prim_path=prim_path, resolution=resolution, **kwargs)


def make_lidar(prim_path: str, **kwargs) -> Any:
    """Create a native LidarRtx using the existing omni.isaac.sensor import path."""
    # Preserve the existing deprecated sensor import for runtime compatibility.
    from omni.isaac.sensor import LidarRtx  # noqa: PLC0415

    return LidarRtx(prim_path=prim_path, **kwargs)


def make_imu(prim_path: str, **kwargs) -> Any:
    """Create a native IMUSensor, forwarding additional keyword arguments."""
    from omni.isaac.sensor import IMUSensor  # noqa: PLC0415

    return IMUSensor(prim_path=prim_path, **kwargs)


def make_contact_sensor(prim_path: str, **kwargs) -> Any:
    """Create a ContactSensor prim; raw sensor interfaces can be acquired separately."""
    from omni.isaac.sensor import ContactSensor  # noqa: PLC0415

    return ContactSensor(prim_path=prim_path, **kwargs)


# ---------------------------------------------------------------------------
# Replicator factories centralize rendering pipeline entry points.
# ---------------------------------------------------------------------------


def create_render_product(camera_prim_or_path, resolution: Tuple[int, int]) -> Any:
    """Create a Replicator render product for the camera and resolution."""
    import omni.replicator.core as rep  # noqa: PLC0415

    return rep.create.render_product(camera_prim_or_path, resolution)


def get_annotator(name: str) -> Any:
    """Get a Replicator annotator by name, e.g. rgb, depth, semantic, or instance."""
    import omni.replicator.core as rep  # noqa: PLC0415

    return rep.AnnotatorRegistry.get_annotator(name)


# ---------------------------------------------------------------------------
# RTX settings; startup presets are applied by runtime.init_app.
# ---------------------------------------------------------------------------


def apply_rtx_settings(preset) -> None:
    """Reserved interface; RTX presets are currently applied by runtime.init_app."""
    raise NotImplementedError(
        "The RTX preset is currently configured by sim_backend.runtime.init_app(rtx_mode=...). "
        "Extend this implementation if runtime switching is required."
    )


def set_async_rendering(enable: bool) -> None:
    """Change both Replicator and application async-rendering settings at runtime."""
    from sim_backend.runtime import get_app  # noqa: PLC0415

    app = get_app()
    app._carb_settings.set("/omni/replicator/asyncRendering", enable)
    app._carb_settings.set("/app/asyncRendering", enable)
