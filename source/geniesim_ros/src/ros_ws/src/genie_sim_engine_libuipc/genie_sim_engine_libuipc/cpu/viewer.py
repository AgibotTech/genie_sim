# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read-only CPU collision display, isolated from physics and ROS execution."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CpuViewerConfig:
    """Configure the optional independent CPU model viewer."""

    enabled: bool = False
    render_hz: float = 30.0

    @classmethod
    def from_mapping(cls, raw):
        """Reject malformed display options rather than silently enabling a GUI."""
        if not isinstance(raw, dict) or set(raw) - {"enabled", "render_hz"}:
            raise ValueError("cpu_mujoco.viewer requires a mapping with enabled and render_hz only")
        cfg = cls(**raw)
        if not isinstance(cfg.enabled, bool):
            raise ValueError("cpu_mujoco.viewer.enabled must be a boolean")
        if (
            isinstance(cfg.render_hz, bool)
            or not isinstance(cfg.render_hz, (int, float))
            or not math.isfinite(cfg.render_hz)
            or cfg.render_hz <= 0
        ):
            raise ValueError("cpu_mujoco.viewer.render_hz must be finite and positive")
        return cfg


def apply_snapshot(sim, snapshot):
    """Reconstruct world poses and contacts from a completed CPU state, without stepping."""
    n = len(sim.qids)
    sim.data.qpos[sim.qids] = snapshot[6 : 6 + n]
    sim.data.qvel[sim.vids] = snapshot[6 + n : 6 + 2 * n]
    if sim.proxy_names:
        start = 6 + 2 * n + 13 * len(sim.body_names)
        velocity_start = start + 7 * len(sim.proxy_names)
        sim.set_proxy_state(snapshot[start:velocity_start].reshape((-1, 7)), snapshot[velocity_start:].reshape((-1, 6)))
    sim.data.time = snapshot[2]
    # GUI perturbations stay in this display copy and are discarded on update.
    sim.data.qfrc_applied[:] = 0
    sim.data.xfrc_applied[:] = 0
    sim.mj.mj_forward(sim.model, sim.data)


def run_viewer(model_bytes, config, names, initial, states, stop, ready=None, body_names=()):
    """Display the same MJB in a sibling process; never write to the state mailbox."""
    viewer = None
    try:
        import mujoco
        import mujoco.viewer

        from .worker import CpuStepper

        sim = CpuStepper(model_bytes, config, names, initial, body_names=body_names)
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        print("[cpu-mujoco-viewer] CPU collision hulls + contact points; read-only latest state", flush=True)
        while not stop.is_set() and viewer.is_running():
            started = time.monotonic()
            snapshot = states.read()
            with viewer.lock():
                if snapshot is not None and snapshot[0] > 0:
                    apply_snapshot(sim, snapshot)
            viewer.sync()
            if ready is not None:
                ready.set()
            stop.wait(max(0.0, 1.0 / config.viewer.render_hz - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.getLogger("genie_sim_engine_libuipc.cpu_viewer").warning(
            "CPU viewer unavailable; physics continues without it", exc_info=True
        )
    finally:
        if viewer is not None:
            viewer.close()
