#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# Proprietary and confidential. Unauthorized copying, distribution,
# modification, reverse engineering, or use of this file is prohibited.

"""genie_sim_engine.py

Isaac Sim physics engine (100 Hz) — Python driver + C++ ROS2 bridge.

Isaac Sim's SimulationApp is Python-only, so this script is the Python
*driver*. All ROS2 I/O (publishers, subscribers, executor threads) and
the render scheduler live in a pybind11 extension
(``genie_sim_engine_py``) that this script imports. rclpy is NOT used.

Responsibilities:
  - stage open + articulation init + apply_joint_commands
  - decoupled adaptive render scheduler (GUI) / physics-only loop (headless)
  - per-tick timing / stats
  - 4WS command timeout
  - fake_slam map->odom identity
"""

from __future__ import annotations

import os
import sys
import time

# Early arg parsing (before any omni import) — shared with newton entry point.
from genie_sim_engine_core.common.early_params import _early_param  # noqa: E402

# physics_engine must be known before SimulationApp is constructed (it enables
# the Newton extension at startup time). Parse argv here — before any omni import.
from genie_sim_engine_isaacsim.kit.bootstrap import bootstrap_simulation_app  # noqa: E402

simulation_app, _HEADLESS = bootstrap_simulation_app(
    physics_engine=_early_param(sys.argv[1:], "physics_engine", "isaac_physx"),
    physics_solver=_early_param(sys.argv[1:], "physics_solver", "mujoco-warp"),
    render_mode=_early_param(sys.argv[1:], "render_mode", "raster"),
)

from genie_sim_engine_core.common.loop import _parse_args  # noqa: E402
from genie_sim_engine_newton.common.session import GenieSimEngineSession, SimpleLogger  # noqa: E402


def run(params: dict) -> int:
    logger = SimpleLogger("genie_sim_engine")

    physics_engine = params.get("physics_engine", "isaac_physx").strip().lower()
    physics_solver = params.get("physics_solver", "mujoco-warp").strip().lower()

    session = GenieSimEngineSession(
        node_name="genie_sim_engine",
        params=params,
        simulation_app=simulation_app,
        physics_engine=physics_engine,
        logger=logger,
    )

    decoupled = not _HEADLESS

    logger.info(
        f"Isaac Sim physics engine ready ({session.physics_hz:.0f} Hz physics, "
        f"{session.render_hz:.0f} Hz render target, "
        f"scene={session.scene_usda}, render={'decoupled' if decoupled else 'headless'}, "
        f"physics_engine={physics_engine}"
        + (f", physics_solver={physics_solver}" if physics_engine == "isaac_newton" else "")
        + ")"
    )

    session.startup(_HEADLESS)

    # Render-sync switch — GENIESIM_RENDER_SYNC=1 forces wp.synchronize_device
    # after each render tick (diagnostic only; stalls physics behind render).
    _render_sync_env = os.environ.get("GENIESIM_RENDER_SYNC", "").strip().lower()
    _render_sync_enabled = _render_sync_env in ("1", "true", "yes", "on")
    logger.info(
        f"render-sync: {'ENABLED (diagnostic, physics waits on render GPU work)' if _render_sync_enabled else 'disabled (parallel mode)'}"
    )

    dt = session.dt

    def _render_hook(now, next_tick, sim_time):
        if not decoupled:
            return 0.0, 0.0, False
        t_after_phys = time.monotonic()
        budget_s = (next_tick + session.dt / session.realtime_factor) - t_after_phys
        if session.core.should_render_decoupled(t_after_phys, budget_s):
            t_r0 = time.monotonic()
            simulation_app.update()
            t_phase_render = (time.monotonic() - t_r0) * 1000.0
            # Optional GPU sync — GENIESIM_RENDER_SYNC=1 for diagnostics.
            if _render_sync_enabled:
                try:
                    import warp as wp

                    wp.synchronize_device("cuda:0")
                except Exception:
                    pass
            t_phase_render_sync = (time.monotonic() - t_r0) * 1000.0
            session.core.mark_rendered_decoupled(time.monotonic())
            return t_phase_render, t_phase_render_sync, True
        return 0.0, 0.0, False

    # Spin (shutdown handled inside spin's finally block)
    return session.run(render_hook=_render_hook)


def main() -> int:
    params = _parse_args(sys.argv[1:], prefix="genie_sim_engine")
    try:
        return run(params)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"[genie_sim_engine] fatal: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
