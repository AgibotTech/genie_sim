#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run libuipc with the core ROS session and fixed-step run loop."""

from __future__ import annotations
import sys
import time
from genie_sim_engine_core.common.loop import _parse_args
from genie_sim_engine_libuipc.session import LibuipcSession, SimpleLogger


def run(params, *, gui=False):
    logger = SimpleLogger("genie_sim_engine_libuipc")
    visualizer = str(params.get("physics_engine_visualizer", "none")) if gui else "none"
    if visualizer not in ("none", "newton"):
        raise ValueError("libuipc supports none/newton GL; use a separate render node for OVRtx or Isaac cameras")
    session = None
    viewer = None
    try:
        session = LibuipcSession(
            node_name="genie_sim_engine_libuipc",
            params=params,
            simulation_app=None,
            physics_engine="libuipc",
            logger=logger,
        )
        session.startup(headless=not gui)
        if visualizer == "newton":
            from genie_sim_engine_libuipc.viewer import create_viewer

            viewer = create_viewer(session.sim, session.render_hz, params, logger)
        next_render = time.monotonic()

        def render_hook(now, next_tick, sim_time):
            nonlocal next_render
            if viewer is None or now < next_render:
                return 0.0, 0.0, False
            started = time.monotonic()
            viewer.begin_frame(sim_time)
            viewer.log_state(session.sim.raw_data())
            viewer.end_frame()
            next_render = now + 1.0 / (session.render_hz * session.realtime_factor)
            duration = (time.monotonic() - started) * 1000.0
            return duration, duration, True

        return session.run(render_hook=render_hook, exit_check=lambda: viewer is None or viewer.is_running())
    finally:
        if session is not None:
            session.sim.shutdown()
        if viewer is not None:
            viewer.close()


def main():
    try:
        return run(_parse_args(sys.argv[1:], prefix="genie_sim_engine_libuipc"), gui="--gui" in sys.argv)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
