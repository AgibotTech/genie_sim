# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Public rendering facade backed by isaacsim_native.render.
Factories import sensor classes lazily; callers control sensor initialization.
"""

from __future__ import annotations

from sim_backend.backends.isaacsim_native.render import (
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
    "apply_rtx_settings",
    "create_render_product",
    "get_annotator",
    "make_camera",
    "make_contact_sensor",
    "make_imu",
    "make_lidar",
    "set_async_rendering",
]
