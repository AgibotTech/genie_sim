# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Isaac Sim + Newton (wrapped MuJoCo-Warp) physics backend.

Wraps ``genie_sim_engine_isaacsim.py`` with ``physics_engine=isaac_newton``.
Use this when you want the Omniverse ecosystem AND MuJoCo-style
contact dynamics (compared to ``isaac_physx`` which uses TGS / different
contact model).
"""

from __future__ import annotations

from typing import Any

from . import register_engine
from ._isaac_kit import _IsaacKitSubprocessBackend


@register_engine("isaac_newton")
class IsaacNewtonBackend(_IsaacKitSubprocessBackend):
    UPSTREAM_PACKAGE = "genie_sim_engine_isaacsim"
    UPSTREAM_EXEC = "genie_sim_engine_isaacsim_node.py"
    UPSTREAM_ENGINE = "isaac_newton"
    # The isaac_newton wrapper only accepts mujoco-warp as the inner solver.
    UPSTREAM_SOLVER = "mujoco-warp"
    BACKEND_NAME = "isaac_newton"

    def __init__(
        self,
        *args: Any,
        physics_solver_substep: int = 0,
        physics_solver_iterations: int = 0,
        render_mode: str = "raster",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("ready_timeout_s", 120.0)
        super().__init__(*args, **kwargs)
        self._substep = int(physics_solver_substep)
        self._iters = int(physics_solver_iterations)
        self._render_mode = render_mode

    def _extra_ros_params(self) -> dict[str, str]:
        params: dict[str, str] = {"render_mode": self._render_mode}
        if self._substep:
            params["physics_solver_substep"] = str(self._substep)
        if self._iters:
            params["physics_solver_iterations"] = str(self._iters)
        return params


__all__ = ["IsaacNewtonBackend"]
