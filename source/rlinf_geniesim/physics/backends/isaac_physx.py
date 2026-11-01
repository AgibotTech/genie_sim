# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Isaac Sim + PhysX physics backend.

Wraps ``genie_sim_engine_isaacsim.py`` with ``physics_engine=isaac_physx``.
Kit-based; slowest to start (Isaac Sim takes ~10s) but pulls in the
full Omniverse ecosystem (sensors, replicator, materials).
"""

from __future__ import annotations

from typing import Any

from . import register_engine
from ._isaac_kit import _IsaacKitSubprocessBackend


@register_engine("isaac_physx")
class IsaacPhysXBackend(_IsaacKitSubprocessBackend):
    UPSTREAM_PACKAGE = "genie_sim_engine_isaacsim"
    UPSTREAM_EXEC = "genie_sim_engine_isaacsim_node.py"
    UPSTREAM_ENGINE = "isaac_physx"
    UPSTREAM_SOLVER = None  # isaac_physx does not accept a physics_solver
    BACKEND_NAME = "isaac_physx"

    def __init__(
        self,
        *args: Any,
        render_mode: str = "raster",
        **kwargs: Any,
    ) -> None:
        # Isaac Sim's startup is slow; bump the default ready timeout.
        kwargs.setdefault("ready_timeout_s", 120.0)
        super().__init__(*args, **kwargs)
        self._render_mode = render_mode

    def _extra_ros_params(self) -> dict[str, str]:
        return {"render_mode": self._render_mode}


__all__ = ["IsaacPhysXBackend"]
