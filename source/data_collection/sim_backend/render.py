# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Public rendering facade backed by isaacsim_native.render.
Factories import sensor classes lazily; callers control sensor initialization.
"""

from __future__ import annotations

from typing import Any, Protocol, Tuple


class CameraProtocol(Protocol):
    """Collection camera interface; resolution is always width then height."""

    prim_path: str
    render_product_path: str

    def initialize(self) -> None: ...
    def get_resolution(self) -> Tuple[int, int]: ...
    def get_render_product_path(self) -> str: ...
    def get_frequency(self) -> float: ...
    def get_focal_length(self) -> float: ...
    def set_focal_length(self, value: float) -> None: ...
    def get_horizontal_aperture(self) -> float: ...
    def set_horizontal_aperture(self, value: float) -> None: ...
    def get_vertical_aperture(self) -> float: ...
    def set_vertical_aperture(self, value: float) -> None: ...
    def get_world_pose(self, camera_axes: str = "usd") -> tuple[Any, Any]: ...
    def set_world_pose(self, position: Any = None, orientation: Any = None, camera_axes: str = "usd") -> None: ...
    def get_local_pose(self, camera_axes: str = "usd") -> tuple[Any, Any]: ...
    def set_local_pose(self, translation: Any = None, orientation: Any = None, camera_axes: str = "usd") -> None: ...
    def add_distance_to_image_plane_to_frame(self) -> None: ...


class LidarProtocol(Protocol):
    """Lidar interface consumed by the ROS writers."""

    prim_path: str
    render_product_path: str

    def initialize(self) -> None: ...
    def get_render_product_path(self) -> str: ...


from sim_backend.backends.isaacsim_native.render import (
    CameraAdapter,
    LidarAdapter,
    apply_rtx_settings,
    create_render_product,
    get_annotator,
    make_camera,
    make_contact_sensor,
    make_imu,
    make_lidar,
    set_async_rendering,
)

__all__ = [
    "CameraAdapter",
    "CameraProtocol",
    "LidarAdapter",
    "LidarProtocol",
    "apply_rtx_settings",
    "create_render_product",
    "get_annotator",
    "make_camera",
    "make_contact_sensor",
    "make_imu",
    "make_lidar",
    "set_async_rendering",
]
