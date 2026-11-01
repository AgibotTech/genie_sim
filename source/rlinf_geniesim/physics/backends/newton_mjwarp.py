# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton-standalone + MuJoCo-Warp physics backend.

Wraps the upstream ``genie_sim_engine_newton.py`` executable with
``physics_engine=newton`` and ``physics_solver=mujoco-warp`` — the
Kit-free, GPU-batched path.
"""

from __future__ import annotations

import os
from typing import Any

from . import register_engine
from ._upstream_base import _UpstreamSubprocessBackend


@register_engine("newton_mujoco_warp")
class NewtonMuJoCoWarpBackend(_UpstreamSubprocessBackend):
    UPSTREAM_PACKAGE = "genie_sim_engine_newton"
    UPSTREAM_EXEC = "genie_sim_engine_newton_node.py"
    UPSTREAM_ENGINE = "newton"
    # The multi-env adapter, distinct from the stock mujoco-warp one. Per-env
    # njmax/nconmax scaling, the elliptic cone, and the gripper closed-chain
    # fix all live here, so stock users are unaffected. Setting this back to
    # "mujoco-warp" raises nothing: the engine still starts and every
    # multi-env capability silently disappears. Confirm from the engine log's
    # njmax/nconmax line.
    UPSTREAM_SOLVER = "mujoco-warp-multi-env"
    BACKEND_NAME = "newton_mujoco_warp"

    def __init__(
        self,
        *args: Any,
        physics_solver_substep: int = 0,
        physics_solver_iterations: int = 0,
        physics_solver_mass_matrix_interval: int = 0,
        physics_engine_visualizer: str = "none",
        render_mode: str = "raster",
        mujoco_njmax: int = 0,
        mujoco_nconmax: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._substep = int(physics_solver_substep)
        self._iters = int(physics_solver_iterations)
        self._mm_interval = int(physics_solver_mass_matrix_interval)
        self._visualizer = physics_engine_visualizer
        self._render_mode = render_mode
        # mjwarp constraint-buffer overrides; 0 = engine auto-scales with
        # scene body count (EngineNodeParams parses the matching node params)
        self._mujoco_njmax = int(mujoco_njmax)
        self._mujoco_nconmax = int(mujoco_nconmax)

    def _extra_ros_params(self) -> dict[str, str]:
        params: dict[str, str] = {
            "physics_engine_visualizer": self._visualizer,
            "render_mode": self._render_mode,
            # Our physics profile. The shared physics_params.yaml applies to
            # every engine user, so our gripper drive and contact tuning must
            # not go there. This profile is a copy of it plus our deltas, and
            # it replaces the file wholesale rather than layering on it.
            "physics_params_file": self._rlinf_physics_params_file(),
            # We do not run robot_state_publisher, so the engine's body-pose
            # topic is the only channel through which get_body_poses() can see
            # robot links. This switch makes the engine publish /tf_render
            # with the links included instead of a scene-only /tf_scene.
            #
            # It is a hard pair with the topic _ros_bridge subscribes to:
            # changing only one of them yields empty poses with no error.
            "use_tf_render": "true",
        }
        if self._substep:
            params["physics_solver_substep"] = str(self._substep)
        if self._iters:
            params["physics_solver_iterations"] = str(self._iters)
        if self._mm_interval:
            params["physics_solver_mass_matrix_interval"] = str(self._mm_interval)
        if self._mujoco_njmax > 0:
            params["mujoco_njmax"] = str(self._mujoco_njmax)
        if self._mujoco_nconmax > 0:
            params["mujoco_nconmax"] = str(self._mujoco_nconmax)
        return params

    @staticmethod
    def _rlinf_physics_params_file() -> str:
        """Absolute path of the installed ``physics_params_rlinf.yaml``.

        Returns an empty string when not found, so the engine falls back to
        its built-in defaults with a log line rather than failing to start.
        That log line is the only sign that grasping reverted to untuned
        behaviour.
        """
        try:
            from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415

            path = os.path.join(
                get_package_share_directory("genie_sim_engine_core"), "config", "physics_params_rlinf.yaml"
            )
            return path if os.path.isfile(path) else ""
        except Exception:  # noqa: BLE001
            return ""


__all__ = ["NewtonMuJoCoWarpBackend"]
