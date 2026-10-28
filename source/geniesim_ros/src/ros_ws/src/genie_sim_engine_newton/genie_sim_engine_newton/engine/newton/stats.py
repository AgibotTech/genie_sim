# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton-standalone stats / 1 Hz logging mixin.

Provides the ``_StatsMixin`` class composed into
``_NewtonStandaloneBase`` via multiple inheritance — see
``engine_base.py`` for the full mixin order.  ``self.X`` references
resolve through the engine's MRO.
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warp as wp


class _StatsMixin:
    def note_render(self, render_ms: float, did_render: bool) -> None:
        """Run loop calls this each tick after the render decision so the
        engine can include render-rate / render-ms stats in its periodic log.
        """
        if did_render:
            self._render_ms_acc += render_ms
            self._render_count += 1
            if render_ms > self._render_ms_max:
                self._render_ms_max = render_ms

    def note_render_target(self, render_hz: float) -> None:
        """Tell the engine what the run loop's target render Hz is so the
        log can show actual-vs-target.
        """
        self._render_target_hz = max(1.0, float(render_hz))

    def note_phase_timing(
        self,
        step_ms: float = 0.0,
        extras_ms: float = 0.0,
        render_ms: float = 0.0,
        render_sync_ms: float = 0.0,
        did_render: bool = False,
    ) -> None:
        """Run loop calls this with per-phase timings so the engine's stats
        log can show where time is spent on render vs non-render ticks.

        Phases:
          step_ms       — sim.step() (Newton physics, including CUDA graph dispatch)
          extras_ms     — sim.tick_extras() (Fabric writeback for cloth + bodies)
          render_ms     — simulation_app.update() (CPU dispatch only)
          render_sync_ms — render_ms + GPU sync (real wall-time GPU cost)
        """
        self._phase_step_ms_acc += step_ms
        self._phase_extras_ms_acc += extras_ms
        if did_render:
            self._phase_render_ms_acc += render_ms
            self._phase_render_sync_ms_acc += render_sync_ms
            if render_sync_ms > self._phase_render_sync_max:
                self._phase_render_sync_max = render_sync_ms

    def note_publish_phase(
        self,
        clock_ms: float = 0.0,
        joints_ms: float = 0.0,
        bodies_ms: float = 0.0,
        odom_ms: float = 0.0,
    ) -> None:
        """Run loop calls this with the breakdown of the publish stage.

        Phases:
          clock_ms  — _core.publish_clock(sim_time)
          joints_ms — snapshot_joint_states + publish_joint_states (USDRT walks)
          bodies_ms — snapshot_body_transforms + publish_body_tf_scene
          odom_ms   — snapshot_odom + publish_odom
        """
        self._publish_clock_ms_acc += clock_ms
        self._publish_joints_ms_acc += joints_ms
        self._publish_bodies_ms_acc += bodies_ms
        self._publish_odom_ms_acc += odom_ms

    def _print_stats(self) -> None:
        """Demo_gui-style stats: wall time, physics+render Hz vs target,
        step/wb/render ms (avg + max), cloth state.
        """
        n_phys = self._tick_count - getattr(self, "_last_tick_count_logged", 0)
        n_phys = max(n_phys, 1)
        n_rend = max(self._render_count, 1)

        t_now = time.monotonic()
        wall = t_now - self._t_loop_start
        window = t_now - self._t_log_start
        sim_time = self._frame_id / max(self._physics_hz, 1.0)

        phys_hz = n_phys / window
        rend_hz = self._render_count / window

        graph_status = "graph" if self._cuda_graph is not None else "no-graph"
        # Newton-standalone is Kit-free: state always lands on USD via
        # ROS publishers; there is no Fabric writeback path.
        wb_path = "usd"

        # Cloth state summary
        cloth_centroid = "n/a"
        cloth_z_range = "n/a"
        n_particles = 0
        if self._cloth_solver is not None and getattr(self._state_0, "particle_q", None) is not None:
            try:
                pq = self._state_0.particle_q.numpy()
                s, e = self._cloth_particle_start, self._cloth_particle_end
                n_particles = max(0, e - s)
                if n_particles > 0:
                    pts = pq[s:e]
                    c = pts.mean(axis=0)
                    cloth_centroid = f"[{c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f}]"
                    cloth_z_range = f"[{pts[:, 2].min():.4f}..{pts[:, 2].max():.4f}]"
            except Exception:
                pass

        self._logger.info(f"[newton-standalone] t={sim_time:6.2f}s  wall={wall:5.1f}s  frame={self._frame_id}")
        # Display in sim-time: actuals are divided by rtf so they compare
        # directly against the user-configured physics_hz / render_hz targets.
        # With rtf=0.5 the wall-clock rate is half the sim-rate; "rtf=" makes
        # that explicit instead of hiding it in a scaled target number.
        _rtf = getattr(self, "_rtf", 1.0) or 1.0
        # ``viewport`` here is the physics-loop's render hook — used by the
        # Newton GL viewer.  The inline OVRtx visualizer renders on its own
        # thread and reports its rate in ``[ovrtx-viz] N frames/s`` lines;
        # ``viewport=0.0Hz`` is normal and expected for ovrtx mode.
        self._logger.info(
            f"[newton-standalone]   physics={phys_hz / _rtf:5.1f}Hz (target {self._physics_hz:.0f}, rtf={_rtf:g})  "
            f"viewport(in-loop)={rend_hz / _rtf:5.1f}Hz (target {self._render_target_hz:.0f})"
        )
        # Inline OVRtx visualizer render rate, when active.  Imported lazily
        # so this module doesn't pull in ovrtx_visualizer (and therefore
        # ovrtx + warp) on the headless / newton-GL paths.  Returns None
        # when the OVRtx thread is not running or has not produced a fresh
        # 1-second sample within the last ~3 s.
        try:
            from genie_sim_engine_newton.engine.newton.visualizers.ovrtx import get_render_stats

            ovrtx_stats = get_render_stats()
        except Exception:
            ovrtx_stats = None
        if ovrtx_stats is not None:
            ohz, oms, ofail = ovrtx_stats
            failed_part = f"  failed={ofail}" if ofail else ""
            self._logger.info(
                f"[newton-standalone]   ovrtx={ohz / _rtf:5.1f}Hz "
                f"(target {self._render_target_hz:.0f})  "
                f"avg={oms:5.2f}ms/frame{failed_part}"
            )
        self._logger.info(
            f"[newton-standalone]   step={self._step_ms_acc/n_phys:5.2f}ms (max={self._step_ms_max:5.1f})  "
            f"wb={self._writeback_ms_acc/n_phys:.2f}ms (max={self._writeback_ms_max:5.1f}) "
            f"[{wb_path},{graph_status}]"
        )
        if self._render_count > 0:
            self._logger.info(
                f"[newton-standalone]   render={self._render_ms_acc/n_rend:5.2f}ms "
                f"(max={self._render_ms_max:5.1f}, fired={self._render_count})"
            )
        self._logger.info(
            f"[newton-standalone]   phase[avg/tick]: phys={self._phase_step_ms_acc/n_phys:5.2f}ms  "
            f"extras={self._phase_extras_ms_acc/n_phys:5.2f}ms  "
            f"render(cpu)={self._phase_render_ms_acc/n_rend:5.2f}ms  "
            f"render(+sync)={self._phase_render_sync_ms_acc/n_rend:5.2f}ms "
            f"(max={self._phase_render_sync_max:5.1f})"
        )
        self._logger.info(
            f"[newton-standalone]   publish[avg/tick]: clock={self._publish_clock_ms_acc/n_phys:5.3f}ms  "
            f"joints={self._publish_joints_ms_acc/n_phys:5.3f}ms  "
            f"bodies={self._publish_bodies_ms_acc/n_phys:5.3f}ms  "
            f"odom={self._publish_odom_ms_acc/n_phys:5.3f}ms"
        )
        if self._cloth_solver is not None:
            self._logger.info(
                f"[newton-standalone]   cloth_centroid={cloth_centroid}  z={cloth_z_range}  particles={n_particles}"
            )

        self._step_ms_acc = self._writeback_ms_acc = self._render_ms_acc = 0.0
        self._step_ms_max = self._writeback_ms_max = self._render_ms_max = 0.0
        self._phase_step_ms_acc = self._phase_extras_ms_acc = 0.0
        self._phase_render_ms_acc = self._phase_render_sync_ms_acc = 0.0
        self._phase_render_sync_max = 0.0
        self._publish_clock_ms_acc = self._publish_joints_ms_acc = 0.0
        self._publish_bodies_ms_acc = self._publish_odom_ms_acc = 0.0
        self._render_count = 0
        self._last_tick_count_logged = self._tick_count
        self._t_log_start = t_now

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
