# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton-standalone state I/O mixin.

Provides the ``_StateMixin`` class composed into
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


class _StateMixin:
    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (positions, velocities, efforts) for ``self.joint_names``.

        Units match ``snapshot_joint_states``:
          * revolute  → radians, rad/s, N·m
          * prismatic → metres,  m/s,   N

        Reads pos/vel from the engine's ping-pong host mirrors of
        ``state_0.joint_q`` / ``state_0.joint_qd`` (see
        ``async_mirror.py``).  At steady state the read is a
        non-blocking, microsecond check that the copy into the
        mirror's READER slot (filled by the previous tick) has landed.

        Efforts come from ``state_0.joint_f`` when present (the
        applied joint torque array Warp integrates from PD control
        + external forces).  Zero when the field is absent or the
        readback fails — matches the v2 convention that "missing
        values are zeros, not sentinels".

        Reads from the engine's ping-pong host mirrors of
        ``state_0.joint_q`` / ``state_0.joint_qd`` (see
        ``async_mirror.py``).  At steady state the read is a
        non-blocking, microsecond check that the copy into the
        mirror's READER slot (filled by the previous tick) has landed.

        Falls through to a synchronous ``state_0.joint_q.numpy()``
        read when the mirrors haven't yet seen two refreshes (warmup,
        first ~2 publish ticks), when the reader slot's copy has not
        landed yet (the CPU ran ahead of the GPU), or when the mirrors
        have failed at runtime.

        Trade-off: the mirror lags by ONE physics tick — at 100 Hz
        that's 10 ms of /joint_states lag, well inside RViz / WBC /
        ROS state-observation tolerance, and the alternative was 6 ms
        of CPU stall every tick.

        Internal profiling (1 Hz)
        -------------------------
        ``sync`` — wall time spent in ``mirror.read()`` (the event
                   status check); microseconds at steady state.
        ``copy`` — wall time spent walking ``self._joint_names`` and
                   filling the return arrays from the host mirror.
        """
        n = len(self._joint_names)
        pos = np.zeros(n, dtype=np.float64)
        vel = np.zeros(n, dtype=np.float64)
        eff = np.zeros(n, dtype=np.float64)
        if self._state_0 is None or getattr(self._state_0, "joint_q", None) is None:
            return pos, vel, eff
        try:
            # Split the timer so the 1 Hz log can attribute cost
            # between the mirror's event check and the fallback path:
            #
            #   read_ms     time spent inside ``mirror.read()`` —
            #               a non-blocking event status check, so
            #               microseconds.
            #   fb_ms       time spent on the synchronous fallback
            #               (``state_0.joint_q.numpy()``).  Non-zero
            #               during warmup, on ticks where the reader
            #               slot's copy has not landed yet, or if the
            #               mirror has been latched off.
            #
            # ``t_sync`` wraps both so the existing log keeps its
            # meaning, while the split fields show which layer is
            # actually slow.
            t0 = time.monotonic()
            jq = self._jq_mirror.read()
            jqd = self._jqd_mirror.read()
            t_read = (time.monotonic() - t0) * 1000.0
            t_fb0 = time.monotonic()
            n_fallback = 0
            if jq is None:
                jq = self._state_0.joint_q.numpy()
                n_fallback += 1
            if jqd is None:
                jqd_arr = getattr(self._state_0, "joint_qd", None)
                jqd = jqd_arr.numpy() if jqd_arr is not None else np.zeros_like(jq)
                n_fallback += 1
            t_fb = (time.monotonic() - t_fb0) * 1000.0
            t_sync = t_read + t_fb
            # NaN/inf early-warning — one shot. If the integrator blew up we
            # want to know it here, not from a flood of RViz TF_NAN errors.
            if not getattr(self, "_joint_state_nan_warned", False):
                if not np.all(np.isfinite(jq)) or not np.all(np.isfinite(jqd)):
                    bad = [i for i in range(len(jq)) if not np.isfinite(jq[i])]
                    self._logger.warn(
                        f"[newton-standalone] state_0.joint_q has non-finite values "
                        f"at DOF idx {bad[:8]}{'…' if len(bad)>8 else ''} — "
                        f"the integrator likely blew up; check init pose, "
                        f"PD gains, and URDF mass/inertia."
                    )
                    self._joint_state_nan_warned = True
            # --- numpy index walk + return-array fill ---
            t1 = time.monotonic()
            static_q = getattr(self, "_static_joint_q", None) or {}
            q_idx_map = getattr(self, "_joint_name_to_q_idx", None) or {}
            # ``state.joint_f`` is qd-indexed (applied joint torques,
            # one entry per DOF).  Optional — Warp exposes it only
            # when the solver populates it; fall back to zeros.
            jf_arr = None
            try:
                jf = getattr(self._state_0, "joint_f", None)
                if jf is not None:
                    jf_arr = jf.numpy() if hasattr(jf, "numpy") else np.asarray(jf)
            except Exception:
                jf_arr = None
            for i, name in enumerate(self._joint_names):
                idx = self._joint_name_to_dof.get(name)
                if idx is None:
                    # No DOF in joint_q — this is a "synthetic" entry for
                    # a joint that fix_base/fix_head/fix_body collapsed to
                    # FixedJoint.  Publish the cached static value so
                    # robot_state_publisher's URDF FK stays consistent
                    # with what Newton's body_q reports.  Velocity stays
                    # at 0 (the joint can't move).
                    if name in static_q:
                        pos[i] = float(static_q[name])
                    continue
                # ``state.joint_q`` is q-indexed (FREE joints carry an
                # extra coord vs the qd vector), so use the q-index map
                # to read positions.  ``state.joint_qd`` is qd-indexed
                # like every other DOF-sized array, so use the dof map.
                q_idx = q_idx_map.get(name, idx)
                if 0 <= q_idx < len(jq):
                    pos[i] = float(jq[q_idx])
                if 0 <= idx < len(jqd):
                    vel[i] = float(jqd[idx])
                if jf_arr is not None and 0 <= idx < len(jf_arr):
                    eff[i] = float(jf_arr[idx])
            t_copy = (time.monotonic() - t1) * 1000.0
            # Roll a 1 Hz aggregate so we can see avg sync vs copy.
            if not hasattr(self, "_get_js_log_t0"):
                self._get_js_log_t0 = time.monotonic()
                self._get_js_sync_acc = 0.0
                self._get_js_copy_acc = 0.0
                self._get_js_sync_max = 0.0
                self._get_js_read_acc = 0.0
                self._get_js_read_max = 0.0
                self._get_js_fb_acc = 0.0
                self._get_js_fb_max = 0.0
                self._get_js_fb_n = 0
                self._get_js_n = 0
            self._get_js_sync_acc += t_sync
            self._get_js_copy_acc += t_copy
            self._get_js_sync_max = max(self._get_js_sync_max, t_sync)
            self._get_js_read_acc += t_read
            self._get_js_read_max = max(self._get_js_read_max, t_read)
            self._get_js_fb_acc += t_fb
            self._get_js_fb_max = max(self._get_js_fb_max, t_fb)
            self._get_js_fb_n += n_fallback
            self._get_js_n += 1
            if (time.monotonic() - self._get_js_log_t0) >= 1.0 and self._get_js_n > 0:
                avg_sync = self._get_js_sync_acc / self._get_js_n
                avg_copy = self._get_js_copy_acc / self._get_js_n
                avg_read = self._get_js_read_acc / self._get_js_n
                avg_fb = self._get_js_fb_acc / self._get_js_n
                # ``sync`` aggregates ``read`` (mirror event check) +
                # ``fb`` (synchronous .numpy() fallback when the
                # mirror returned None).  A steadily non-zero ``fb``
                # means the CPU keeps running ahead of the GPU, so the
                # previous tick's copy has not landed by publish time.
                desc = (
                    "ping-pong host-mirror"
                    if self._jq_mirror.ok
                    else "synchronous fallback (mirror warming up or failed)"
                )
                self._logger.info(
                    f"[newton-standalone] get_joint_states (1Hz window, {self._get_js_n} calls): "
                    f"sync={avg_sync:.3f}ms (max={self._get_js_sync_max:.3f})  "
                    f"copy={avg_copy:.3f}ms  ({desc})\n"
                    f"  └─ split: read={avg_read:.3f}ms (max={self._get_js_read_max:.3f}, mirror.read())  "
                    f"fb={avg_fb:.3f}ms (max={self._get_js_fb_max:.3f}, "
                    f"fallback .numpy() {self._get_js_fb_n}/{self._get_js_n * 2})"
                )
                self._get_js_log_t0 = time.monotonic()
                self._get_js_sync_acc = 0.0
                self._get_js_copy_acc = 0.0
                self._get_js_sync_max = 0.0
                self._get_js_read_acc = 0.0
                self._get_js_read_max = 0.0
                self._get_js_fb_acc = 0.0
                self._get_js_fb_max = 0.0
                self._get_js_fb_n = 0
                self._get_js_n = 0
        except Exception as exc:
            if not getattr(self, "_joint_state_warned", False):
                self._logger.warn(f"[newton-standalone] get_joint_states: {exc}")
                self._joint_state_warned = True
        return pos, vel, eff

    def get_scene_joint_states(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return ``(positions, velocities, efforts)`` for
        ``self.scene_joint_names`` — the articulated-scene DoFs
        (door hinge, drawer slide) published on ``/joint_states_scene``.

        Units match :meth:`get_joint_states`: revolute → rad, rad/s, N·m;
        prismatic → m, m/s, N.

        Hinges and slides only — a free-floating scene body (a loose cup)
        is NOT here. Its 7-wide q-slice against a 6-wide qd-slice has no
        scalar ``JointState.position``, and its pose already rides
        ``/tf_scene`` as a body transform, which is the right channel for
        6 DoF. ``_build_joint_map`` drops FREE joints before the ownership
        test for exactly that reason.

        Returns ``None`` when the scene has no articulated DoFs, which the
        publish loop treats as "nothing to publish" and leaves the topic
        quiet — distinguishable from a stalled publisher.

        Reads the SAME ping-pong host mirrors :meth:`get_joint_states`
        already pulled this tick (``_jq_mirror`` / ``_jqd_mirror``), so
        adding this channel costs no extra GPU sync — just an index walk
        over the scene DoF arrays. Same one-physics-tick lag as
        ``/joint_states``, and since both channels are filled from one
        mirror read they stay mutually consistent.

        Scene props are unactuated, so ``state.joint_f`` carries only the
        constraint / contact torque the solver resolved — exactly what
        benchmark checkers threshold on for a pushed door. Zeros when the
        solver does not populate the field.
        """
        names = getattr(self, "_scene_joint_names", None)
        if not names:
            return None
        n = len(names)
        pos = np.zeros(n, dtype=np.float64)
        vel = np.zeros(n, dtype=np.float64)
        eff = np.zeros(n, dtype=np.float64)
        if self._state_0 is None or getattr(self._state_0, "joint_q", None) is None:
            return pos, vel, eff
        try:
            # Mirrors were refreshed by this tick's get_joint_states call;
            # read() returns None when the reader slot's copy has not landed
            # yet or during warmup; fall back to a synchronous read, same as
            # the robot path.
            jq = self._jq_mirror.read()
            if jq is None:
                jq = self._state_0.joint_q.numpy()
            jqd = self._jqd_mirror.read()
            if jqd is None:
                jqd_arr = getattr(self._state_0, "joint_qd", None)
                jqd = jqd_arr.numpy() if jqd_arr is not None else np.zeros_like(jq)
            jf_arr = None
            try:
                jf = getattr(self._state_0, "joint_f", None)
                if jf is not None:
                    jf_arr = jf.numpy() if hasattr(jf, "numpy") else np.asarray(jf)
            except Exception:  # noqa: BLE001
                jf_arr = None
            q_idx = self._scene_joint_q_idx
            dof_idx = self._scene_joint_dof_idx
            for i in range(n):
                qi = q_idx[i]
                if 0 <= qi < len(jq):
                    pos[i] = float(jq[qi])
                di = dof_idx[i]
                if 0 <= di < len(jqd):
                    vel[i] = float(jqd[di])
                if jf_arr is not None and 0 <= di < len(jf_arr):
                    eff[i] = float(jf_arr[di])
        except Exception as exc:  # noqa: BLE001
            if not getattr(self, "_scene_joint_state_warned", False):
                self._logger.warn(f"[newton-standalone] get_scene_joint_states: {exc}")
                self._scene_joint_state_warned = True
        return pos, vel, eff

    def get_body_transforms(self) -> Tuple[np.ndarray, List[str]]:
        """Return ``(Nx7 array of (x,y,z,qw,qx,qy,qz), absolute_prim_paths)``.

        Newton's ``state_0.body_q`` is per-body world-space ``wp.transformf`` =
        ``(x, y, z, qx, qy, qz, qw)``. We reorder to the publisher's expected
        ``(x, y, z, qw, qx, qy, qz)`` and return paths from ``self._body_paths``.

        Reads from ``self._bq_mirror`` (see ``async_mirror.py``).  Same
        one-tick staleness as :meth:`get_joint_states`; falls through
        to a synchronous ``state_0.body_q.numpy()`` during warmup.

        NOTE: this returns WORLD-space poses, while the USD path returns
        LOCAL-relative poses (each body relative to its USD parent). For
        the renderer's hierarchy composition this matters; if your
        publisher needs local poses, compose against the parent transform
        on the consumer side.
        """
        if self._state_0 is None or getattr(self._state_0, "body_q", None) is None or not self._body_paths:
            return np.zeros((0, 7), dtype=np.float64), []
        try:
            body_q = self._bq_mirror.read()
            if body_q is None:
                body_q = self._state_0.body_q.numpy()
        except Exception as exc:
            if not getattr(self, "_body_xform_warned", False):
                self._logger.warn(f"[newton-standalone] get_body_transforms: {exc}")
                self._body_xform_warned = True
            return np.zeros((0, 7), dtype=np.float64), []

        # One-shot floating-base runaway tripwire.  joint_q can stay finite
        # while the un-actuated free-joint base drifts arbitrarily far (no
        # gravity / no floor contact inside the captured robot substep).
        # When that happens robot_state_publisher reports nominal joint
        # angles but RViz shows the chain shot off into space — what the
        # user calls "the robot exploded" with no NaN warning.
        # Catches that case ON THE FIRST frame the base translates more than
        # 0.5 m from its initial position or any body's pose stops being
        # finite, and dumps body name + pose + velocity + last commanded
        # joint targets so we can see which DOF pushed the base off.
        if not getattr(self, "_base_runaway_warned", False):
            try:
                nonfinite_idx: List[int] = []
                for i in range(min(len(body_q), len(self._body_paths))):
                    q = body_q[i]
                    if not (
                        np.isfinite(q[0])
                        and np.isfinite(q[1])
                        and np.isfinite(q[2])
                        and np.isfinite(q[3])
                        and np.isfinite(q[4])
                        and np.isfinite(q[5])
                        and np.isfinite(q[6])
                    ):
                        nonfinite_idx.append(i)

                if nonfinite_idx:
                    self._base_runaway_warned = True
                    bad = nonfinite_idx[:6]
                    self._logger.warn(
                        f"[newton-standalone] body_q has NON-FINITE values at body idx "
                        f"{bad}{'…' if len(nonfinite_idx) > 6 else ''} "
                        f"({[self._body_paths[i] for i in bad]}) — integrator blew up; "
                        f"check whether the un-actuated free base is being pushed by an "
                        f"un-damped joint reaction."
                    )
                else:
                    # Cache the initial base position the first time the
                    # tripwire runs (which is also the first publish tick).
                    # After that, watch for ‖pos − pos0‖ > 0.5 m.
                    base_idx = getattr(self, "_odom_base_idx", None)
                    if base_idx is None:
                        for i, path in enumerate(self._body_paths):
                            if path.endswith("/base_link"):
                                base_idx = i
                                break
                        self._odom_base_idx = base_idx
                    if base_idx is not None and 0 <= base_idx < len(body_q):
                        bq = body_q[base_idx]
                        cur = np.array([float(bq[0]), float(bq[1]), float(bq[2])])
                        if not hasattr(self, "_base_pos0"):
                            self._base_pos0 = cur.copy()
                            self._base_pos0_t = time.monotonic()
                        delta = float(np.linalg.norm(cur - self._base_pos0))
                        if delta > 0.5:
                            self._base_runaway_warned = True
                            # Dump every body that has moved noticeably plus
                            # last commanded targets.
                            tgt_buf = None
                            try:
                                if hasattr(self, "_adapter") and self._adapter is not None:
                                    ta = self._adapter.target_buffer()
                                    if ta is not None:
                                        tgt_buf = ta.numpy()
                            except Exception:
                                tgt_buf = None
                            moved = []
                            for i, path in enumerate(self._body_paths[: len(body_q)]):
                                q = body_q[i]
                                p = np.array([float(q[0]), float(q[1]), float(q[2])])
                                moved.append((float(np.linalg.norm(p)), path, p))
                            moved.sort(key=lambda r: -r[0])
                            top = moved[:6]
                            self._logger.warn(
                                f"[newton-standalone] BASE RUNAWAY: base_link drifted "
                                f"‖Δ‖={delta:.3f} m from t0 in "
                                f"{time.monotonic() - self._base_pos0_t:.2f}s.  "
                                f"base@start={self._base_pos0.tolist()} → now={cur.tolist()}.  "
                                f"Largest ‖pos‖ bodies (likely runaway): "
                                + ", ".join(f"{p}@{n:.2f}m" for n, p, _ in top)
                            )
                            if tgt_buf is not None:
                                # Show the 8 joints with the largest |target|
                                # — the most likely PD-reaction culprits.
                                pairs: List[Tuple[float, str, int]] = []
                                for name, dof in self._joint_name_to_dof.items():
                                    if 0 <= dof < len(tgt_buf):
                                        pairs.append((float(tgt_buf[dof]), name, int(dof)))
                                pairs.sort(key=lambda r: -abs(r[0]))
                                top_tgt = pairs[:8]
                                self._logger.warn(
                                    "[newton-standalone] BASE RUNAWAY: last joint_target_q top-|target|: "
                                    + ", ".join(f"{n}[dof={d}]={v:+.3f}" for v, n, d in top_tgt)
                                )
            except Exception as exc:
                # Never let the tripwire kill the publish phase.
                if not getattr(self, "_base_runaway_tripwire_warned", False):
                    self._logger.warn(f"[newton-standalone] base runaway tripwire failed: {exc}")
                    self._base_runaway_tripwire_warned = True
        n = min(len(self._body_paths), len(body_q))
        out = np.zeros((n, 7), dtype=np.float64)
        for i in range(n):
            q = body_q[i]
            out[i, 0] = float(q[0])
            out[i, 1] = float(q[1])
            out[i, 2] = float(q[2])
            out[i, 3] = float(q[6])  # qw
            out[i, 4] = float(q[3])  # qx
            out[i, 5] = float(q[4])  # qy
            out[i, 6] = float(q[5])  # qz
        return out, list(self._body_paths[:n])

    def get_odom(self, sim_time: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return ``(pose7=(x,y,z,qw,qx,qy,qz), twist6=(vx,vy,vz,wx,wy,wz))``.

        Returns base_link's full world-frame pose verbatim.  The C++
        publish_odom path then splits this into:

          * ``odom -> base_footprint`` -- ground-projected pose
            (z=0, roll=pitch=0, yaw kept).  Published on /tf and on
            nav_msgs/Odometry as the standard mobile-base frame the
            SRDF planar virtual_joint anchors against.
          * ``base_footprint -> base_link`` -- the residual: a dynamic
            transform carrying the height + tilt of the chassis above
            its ground projection.  Published on /tf each tick.

        The split lives on the C++ side (see ``publish_odom`` in
        ``realtime_ros_node.cpp``) so this Python entry point stays a
        pure ``body_q[base_link]`` read with no quat math.

        Twist is finite-differenced against the previous call's pose,
        transformed into the base_link frame.
        """
        if self._state_0 is None or getattr(self._state_0, "body_q", None) is None or not self._body_paths:
            return None
        # Cached base index
        base_idx = getattr(self, "_odom_base_idx", None)
        if base_idx is None:
            for i, path in enumerate(self._body_paths):
                if path.endswith("/base_link"):
                    base_idx = i
                    break
            self._odom_base_idx = base_idx
        if base_idx is None or base_idx < 0:
            return None
        try:
            body_q = self._bq_mirror.read()
            if body_q is None:
                body_q = self._state_0.body_q.numpy()
        except Exception:
            return None
        if base_idx >= len(body_q):
            return None
        q = body_q[base_idx]
        pose7 = np.array(
            [float(q[0]), float(q[1]), float(q[2]), float(q[6]), float(q[3]), float(q[4]), float(q[5])],
            dtype=np.float64,
        )
        # Twist via finite difference (pose7 prev cached on self)
        prev = getattr(self, "_odom_prev", None)
        twist6 = np.zeros(6, dtype=np.float64)
        if prev is not None:
            prev_pose, prev_t = prev
            dt = max(sim_time - prev_t, 1e-6)
            dv_world = (pose7[:3] - prev_pose[:3]) / dt
            # Rotate world-frame v into base_link frame: v_base = R^T · v_world.
            # Quat (qw, qx, qy, qz) → rotation matrix
            qw, qx, qy, qz = pose7[3], pose7[4], pose7[5], pose7[6]
            R = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ],
                dtype=np.float64,
            )
            twist6[:3] = R.T @ dv_world
            # Angular velocity: omitted here — finite-differencing quaternions
            # is noisy and rarely needed for cloth/folding tasks.
        self._odom_prev = (pose7.copy(), sim_time)
        return pose7, twist6
