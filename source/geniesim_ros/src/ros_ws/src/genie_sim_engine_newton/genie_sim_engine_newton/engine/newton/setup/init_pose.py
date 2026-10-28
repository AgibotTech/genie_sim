# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build-phase 8: name maps + init_joint_pos + MJCF keyframe + state sync.

Last phase before warmup / graph capture.  Builds joint/body/mimic
lookup maps, applies launcher init_joint_pos, captures the
init pose into the MJCF dump keyframe, syncs state.joint_q from
model, and selects the per-substep body.
"""

from __future__ import annotations

import json
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warp as wp


class _InitPoseMixin:
    def _phase_finalize_init_state(self) -> None:
        """Phase 8: maps + ``init_joint_pos`` + MJCF keyframe + state sync.

        Builds joint/body/mimic name lookup maps, applies launcher's
        ``init_joint_pos`` (re-runs ``eval_fk`` afterwards), captures
        the init pose as ``<keyframe name="home">`` in the MJCF dump,
        lets the adapter finalise PD drives now that DOF mapping
        exists, then syncs state.joint_q from model and selects the
        substep body.
        """
        import newton  # noqa: PLC0415

        # ``_warmup_renders`` is a no-op on newton-standalone (no Kit
        # viewport, no Fabric path), so name maps can be built directly
        # after the model finalises.
        self._build_joint_map()
        self._build_body_map()
        self._build_mimic_map()

        # Apply launcher's init_joint_pos to model.joint_q + control.joint_target_q.
        # Must run AFTER _build_joint_map (we need the name→DOF index).
        # Then re-run eval_fk so body_q reflects the new joint positions before
        # warmup/capture see the wrong initial pose.
        self._apply_init_joint_pos()

        # [diag-init] Verify joint_q is still holding the init pose at
        # THIS point in the lifecycle, immediately before the keyframe
        # capture reads it.  If `_apply_init_joint_pos`'s post-assign
        # log showed the right values but this readback shows zeros,
        # something between _apply_init_joint_pos and here clobbered
        # joint_q — likely a downstream state init in _build_*_map or
        # adapter prep that called eval_fk / state_0 reset without
        # propagating the seeded joint_q.
        try:
            ip = getattr(self, "_init_joint_pos", None) or {}
            if ip and self._model is not None and self._model.joint_q is not None:
                jq_check = self._model.joint_q.numpy()
                check_lines = []
                for name in ip.keys():
                    idx = self._joint_name_to_dof.get(name)
                    if idx is None or idx >= len(jq_check):
                        continue
                    check_lines.append(f"  {name}[dof={idx}]: joint_q={float(jq_check[idx]):.6f}")
                if check_lines:
                    self._logger.info(
                        f"[diag-init] joint_q at keyframe-dump entry ({len(check_lines)} init joint(s)):\n"
                        + "\n".join(check_lines)
                    )
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[diag-init] keyframe-entry joint_q readback failed: {exc}")

        # Capture the post-init-pose joint_q into a ``<keyframe name="home">``
        # in the dumped MJCF so pure MuJoCo can reproduce the runtime's
        # starting state via ``mj_resetDataKeyframe(m, d, m.key('home').id)``.
        # Newton's MJCF emit doesn't author qpos0, so the dump would
        # otherwise load with every joint at 0 — visually wrong and
        # behaviourally different from frame 1 of the live engine.
        # Skipped silently when no MJCF dump path is configured (e.g.
        # ``runtime_usd_dump_path`` not set).
        try:
            mjcf_path = getattr(self._adapter, "_save_to_mjcf", "")
            if mjcf_path:
                from genie_sim_engine_newton.common.mjcf_postprocess import add_init_pose_keyframe  # noqa: PLC0415

                add_init_pose_keyframe(
                    mjcf_path=mjcf_path,
                    qpos=self._model.joint_q.numpy(),
                    name="home",
                    logger=self._logger,
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[lifecycle] init-pose keyframe step failed (continuing): {exc!r}")
        # Solver-specific tweaks that need the joint name -> DOF map.
        #
        #   mjwarp:       no-op (PD was set in prepare_model before
        #                 SolverMuJoCo init); logs a one-line confirmation.
        #   featherstone: applies selective PD on PASSIVE joints (body/head
        #                 etc) so heavy root chains resist gravity without
        #                 PD fighting velocity injection on the arms.
        self._adapter.post_joint_map(self._model, self._jindex, self._control, self._logger)
        # mjwarp adapter: emit per-joint PD diagnostic logging.  Featherstone
        # already logs from inside post_joint_map.
        if self._adapter.name.startswith("mujoco-warp"):
            self._configure_pd_drives()
        # Sync state_0.joint_q ← model.joint_q (and state_0.joint_qd ← 0).
        #
        # ``newton.eval_fk(model, model.joint_q, model.joint_qd, state)`` reads
        # joint_q from the model and writes ONLY ``state.body_q`` — it does
        # NOT touch ``state.joint_q`` / ``state.joint_qd``, which stay at the
        # zeros from ``model.state()``.
        #
        # If we leave them at zero, the very first Featherstone step starts
        # from "all joints at 0" (the state's view) but with PD targets at
        # the init pose (e.g. arm_joint2=-90°). The error is the full init
        # pose, the proportional torque is huge, and the integrator blows
        # straight to NaN — matching the RViz "TF_NAN_INPUT" floods we saw.
        # Also: ``get_joint_states`` reads from ``state_0.joint_q`` so it
        # would report zeros, not the init pose.
        for st in (self._state_0, self._state_1):
            if st is None:
                continue
            jq = getattr(st, "joint_q", None)
            jqd = getattr(st, "joint_qd", None)
            if jq is not None and self._model.joint_q is not None:
                jq.assign(self._model.joint_q)
            if jqd is not None:
                jqd.zero_()
        # Target-position buffer.  Each adapter owns its own buffer:
        #
        #   mjwarp adapter: seeds control.joint_target_q with the init
        #     pose so SolverMuJoCo's JOINT_TARGET actuators start from the
        #     correct position before any /joint_command arrives.
        #
        #   featherstone adapter: allocates a dedicated wp.array and seeds
        #     with the init pose so the velocity-injection kernel computes
        #     qd = (init − init) = 0 on the first substep — no spurious
        #     motion before /joint_command.
        #
        # In both cases the captured CUDA graph reads the buffer by pointer;
        # apply_commands mutates contents via .assign() to preserve the
        # captured pointer.
        self._adapter.init_target_buffer(self._model, self._control, self._logger)
        newton.eval_fk(self._model, self._model.joint_q, self._model.joint_qd, self._state_0)

        # Push the init pose INTO the live mjw_data.  The mjwarp adapter
        # runs SolverMuJoCo with ``update_data_interval=0`` (the per-substep
        # Newton→mjw re-sync was pumping energy into resting contacts — see
        # ``build_solver``), so the state_0.joint_q writes above no longer
        # reach the solver on their own.  Without this one-shot sync the
        # solver would boot from the pre-init qpos and only the PD targets
        # would drag the joints toward the init pose.
        sync = getattr(self._adapter, "sync_state_into_solver", None)
        if sync is not None:
            sync(self._state_0, self._logger)
            self._logger.info(
                "[newton-standalone] init pose synced into solver qpos " "(one-shot; update_data_interval=0)"
            )

        # Lock in the per-substep body now that the cloth solver and
        # adapter are both constructed.  The captured CUDA graph baked
        # below in ``_capture_graph`` records whichever branch this
        # selects; swapping branches mid-run would require a re-capture.
        self._pick_substep_body()

        # Snapshot the build-time init state (full per-world joint_q +
        # init PD targets) so engine.reset(env_ids) can restore selected
        # worlds without rebuilding the model.  Must run AFTER
        # _apply_init_joint_pos + init_target_buffer (above) so the
        # snapshot captures the init pose, not URDF defaults.
        self._snapshot_reset_state()

        self._logger.info(
            f"[newton-standalone] model — "
            f"{self._model.body_count} bodies, "
            f"{self._model.joint_dof_count} DOFs, "
            f"{self._model.particle_count} cloth particles"
        )

        # [diag-base-pose] One-shot dump of base_link's world pose after the
        # final eval_fk.  This is what ``get_odom`` reads and what publish_odom
        # ships as ``odom -> base_link`` in /tf.  If init_base_pose says
        # ``z=0.04`` but this log shows ``z=0.0``, the USD Xform translate on
        # ``/<robot_prefix>`` didn't reach Newton's body_q (parser quirk:
        # composed pose not honoured, or our session-layer edit didn't land
        # on the right edit target) — fall back to writing the offset
        # directly into ``model.joint_q`` for the FREE-joint slot (or
        # ``state_0.body_q`` for the welded case).  Logged once at startup;
        # zero per-tick cost.
        try:
            base_pose_cfg = ((self._scene_cfg or {}).get("robot") or {}).get("init_base_pose") or {}
            if isinstance(base_pose_cfg, dict) and self._state_0 is not None:
                body_paths = getattr(self, "_body_paths", None) or []
                base_idx = next(
                    (i for i, p in enumerate(body_paths) if p.endswith("/base_link")),
                    None,
                )
                if base_idx is not None and getattr(self._state_0, "body_q", None) is not None:
                    bq = self._state_0.body_q.numpy()
                    if base_idx < len(bq):
                        q = bq[base_idx]
                        # body_q stores (x, y, z, qx, qy, qz, qw); reorder to wxyz for log.
                        want = (
                            float(base_pose_cfg.get("x", 0.0)),
                            float(base_pose_cfg.get("y", 0.0)),
                            float(base_pose_cfg.get("z", 0.0)),
                            float(base_pose_cfg.get("theta", 0.0)),
                        )
                        self._logger.info(
                            f"[diag-base-pose] base_link[{base_idx}]={body_paths[base_idx]} "
                            f"body_q xyz=({float(q[0]):.4f}, {float(q[1]):.4f}, {float(q[2]):.4f}) "
                            f"quat_xyzw=({float(q[3]):.4f}, {float(q[4]):.4f}, {float(q[5]):.4f}, {float(q[6]):.4f}) "
                            f"| yaml init_base_pose xyz=({want[0]:.4f}, {want[1]:.4f}, {want[2]:.4f}) "
                            f"theta={want[3]:.4f} rad"
                        )
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[diag-base-pose] startup log failed: {exc!r}")

        # Snapshot the settled init state for episode reset. This is the
        # known-good baseline: all bodies at rest, PD targets seeded, CUDA
        # graph captured (if featherstone), FK evaluated. When episode_reset
        # is called (via RosResetCoordinator), the mixin copies this snapshot
        # back into the live state arrays, then applies randomization deltas
        # on top.
        self._snapshot_reset_state()
