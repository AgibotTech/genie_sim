# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton-standalone control / target-writing mixin.

Provides the ``_ControlMixin`` class composed into
``_NewtonStandaloneBase`` via multiple inheritance — see
``engine_base.py`` for the full mixin order.  ``self.X`` references
resolve through the engine's MRO.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from genie_sim_engine_core.engine._mimic import expand_targets


class _ControlMixin:
    def apply_commands(self, cmd_positions, cmd_4ws_steer_pos, cmd_4ws_drive_vel, cmd_4ws_stamp) -> None:
        """Forward joint targets from ROS into the adapter's target buffer(s).

        Each adapter owns its own position-target buffer:
          * mjwarp adapter:       ``control.joint_target_q`` — what
                                  SolverMuJoCo's JOINT_TARGET actuators
                                  read each step.
          * featherstone adapter: a dedicated ``wp.array`` that the
                                  captured CUDA graph reads each substep
                                  via ``_kernel_velocity_inject_masked``
                                  (joint_qd = target − joint_q for
                                  controlled DOFs only).

        The buffer is mutated IN PLACE via the numpy bridge so the CUDA
        graph's captured pointer sees the new values.  Never reassign
        the array object — that creates a new pointer the graph can't
        reach.

        Three input channels reach this method:

          * ``cmd_positions``       — from ``/joint_command``.  ALL joint
                                      names land here regardless of joint
                                      kind (arm, head, body, gripper,
                                      and chassis_steer/chassis_drive
                                      via /joint_command).  Treated as
                                      position targets.
          * ``cmd_4ws_steer_pos``   — from ``/cmd_4ws``, names ending in
                                      ``joint1`` (the 4WS steer axis).
                                      Position target → same buffer as
                                      cmd_positions.
          * ``cmd_4ws_drive_vel``   — from ``/cmd_4ws``, names ending in
                                      ``joint2`` (the 4WS drive axis,
                                      mode=VELOCITY).  Velocity target
                                      → ``control.joint_target_qd``.

        ``cmd_4ws_stamp`` is the monotonic-now timestamp of the most
        recent ``/cmd_4ws`` message.  When the elapsed since that stamp
        exceeds ``cmd_4ws_timeout_s``, the 4WS channel is treated as
        stale and every chassis_drive DOF is force-braked to vel=0 to
        keep the robot from running away if /cmd_4ws stops publishing.
        ``cmd_4ws_steer_pos`` is left untouched on timeout (the steer
        joints' position-mode actuators just hold whatever target was
        last written — wheels at rest, not a runaway risk).  Mirrors
        the PhysX path in ``kit/stage.py::_apply_joint_commands``.

        Input units match the PhysX path / ROS conventions:
          * revolute  → radians   (positions and velocities)
          * prismatic → metres / metres-per-second
        """
        # Drain pending episode resets (enqueued by RosResetCoordinator's
        # /reset_envs listener). Runs here because apply_commands is called
        # every tick on the loop thread BEFORE sim.step() -- the safe
        # between-frames window to rewrite state (no CUDA graph in flight).
        # No-op (and cheap) when coordinator is absent.
        if hasattr(self, "_lifecycle"):
            self._lifecycle.drain_resets()

        # Resolve stamp + timeout from the engine's params; treat the
        # /cmd_4ws channel as stale until a message has actually been
        # received (stamp=0 ⇒ never received).
        try:
            timeout_s = float(self._params.cmd_4ws_timeout_s)
        except Exception:  # noqa: BLE001
            timeout_s = 0.1
        if cmd_4ws_stamp:
            elapsed = time.monotonic() - float(cmd_4ws_stamp)
        else:
            elapsed = float("inf")
        cmd_4ws_stale = elapsed > timeout_s

        # On the very first call build a small set of joint names known
        # to be chassis_drive (joint2 of a free-spin wheel).  Used by
        # the timeout brake below to zero the right DOFs.  Cached on
        # the engine instance so the classification only runs once.
        # Classifier is regex-only — no Pxr / Newton dependency — so
        # this works regardless of how the model was built.
        if not hasattr(self, "_chassis_drive_dofs"):
            self._chassis_drive_dofs = self._build_chassis_drive_dof_set()

        # Early-return only when EVERY channel is empty AND we don't
        # need to fire the timeout brake.  The brake has to fire on
        # the first tick after /cmd_4ws goes stale even if cmd_4ws_drive_vel
        # is now empty — that's literally the case we're guarding.
        has_any_cmd = bool(cmd_positions) or bool(cmd_4ws_steer_pos) or bool(cmd_4ws_drive_vel)
        needs_brake = (
            cmd_4ws_stale and bool(self._chassis_drive_dofs) and not getattr(self, "_chassis_brake_settled", False)
        )
        if not has_any_cmd and not needs_brake:
            return

        tgt_attr = self._adapter.target_buffer()
        if tgt_attr is None:
            return

        # Velocity buffer lives on ``control.joint_target_qd`` (same
        # shape as joint_target_q).  Not abstracted via the adapter
        # today — mjwarp is the only adapter that uses velocity-mode
        # actuators, and it shares ``control`` with the engine.
        vel_attr = None
        try:
            vel_attr = getattr(self._control, "joint_target_qd", None) if self._control is not None else None
        except Exception:  # noqa: BLE001
            vel_attr = None

        try:
            # Host mirrors of the target buffers, allocated once on the
            # first call and mutated in place every tick thereafter.
            #
            # Why caching matters: ``tgt_attr.numpy()`` on a wp.array whose
            # pointer is captured by a CUDA graph triggers a synchronizing
            # device→host memcpy — which blocks until the previous tick's
            # graph has finished executing.  Doing that unconditionally on
            # every tick burns 5–10 ms per call (measured: ~8.4 ms on a
            # G2-class scene at 100 Hz), turning the command path into the
            # dominant loop cost despite physics itself taking 0.07 ms.
            #
            # The host mirror works because the host is the ONLY writer
            # of the position-target buffer: physics never modifies it,
            # so the host copy is always authoritative once initialised.
            # We seed it once from the device buffer (which the adapter
            # has already filled with the init pose), then keep it warm
            # across calls — only ``.assign()`` (host→device) runs each
            # tick, and only when there were actual writes.
            if getattr(self, "_apply_cmd_tgt_host", None) is None or self._apply_cmd_tgt_host.shape != (
                int(tgt_attr.size),
            ):
                self._apply_cmd_tgt_host = tgt_attr.numpy().copy()
            tgt = self._apply_cmd_tgt_host
            vel = None
            if vel_attr is not None:
                if getattr(self, "_apply_cmd_vel_host", None) is None or self._apply_cmd_vel_host.shape != (
                    int(vel_attr.size),
                ):
                    self._apply_cmd_vel_host = vel_attr.numpy().copy()
                vel = self._apply_cmd_vel_host
            else:
                self._apply_cmd_vel_host = None

            # Broadcast each commanded master's value to its mimic
            # followers so multi-finger grippers actually move. Without
            # this, followers keep target=0 and lock the master via the
            # inertial chain (Featherstone has no native mimic constraint).
            # The shared helper returns ONLY the new follower entries —
            # we merge so an explicit follower command (rare) is still
            # overridden by the mimic-derived value, which is the
            # correct semantics for a master-anchored chain.  Mimic
            # only applies to the joint-command channel, not /cmd_4ws.
            extra = expand_targets(cmd_positions, self._mimic_followers)
            cmd_positions_eff = {**cmd_positions, **extra} if extra else cmd_positions

            # Merge cmd_positions + cmd_4ws_steer_pos into one
            # position-target sweep.  The PhysX path does the same — both
            # channels write to the same articulation DOFs, just via
            # different ROS topics.  4WS-channel writes are gated by the
            # timeout: a stale /cmd_4ws stops moving the steer joints so
            # an external commander disconnecting doesn't leave the
            # rack frozen at the last commanded angle.
            pos_writes: List[Tuple[str, int, float]] = []
            unknown: List[str] = []
            for name, val in cmd_positions_eff.items():
                idx = self._joint_name_to_dof.get(name)
                if idx is None or idx >= len(tgt):
                    unknown.append(name)
                    continue
                fval = float(val)
                tgt[idx] = fval
                pos_writes.append((name, idx, fval))
            if not cmd_4ws_stale:
                for name, val in cmd_4ws_steer_pos.items():
                    idx = self._joint_name_to_dof.get(name)
                    if idx is None or idx >= len(tgt):
                        unknown.append(name)
                        continue
                    fval = float(val)
                    tgt[idx] = fval
                    pos_writes.append((name, idx, fval))

            # Velocity sweep — /cmd_4ws drive channel.  On stale stamp,
            # actively zero every chassis_drive DOF (not just the ones
            # in the current dict) so a publisher dropping mid-roll
            # brings the wheels to a stop instead of letting them coast
            # at the last commanded velocity.
            vel_writes: List[Tuple[str, int, float]] = []
            if vel is not None:
                if cmd_4ws_stale:
                    for name, idx in self._chassis_drive_dofs.items():
                        if idx < len(vel):
                            vel[idx] = 0.0
                            vel_writes.append((name, idx, 0.0))
                    # One-shot: don't keep rewriting zeros every tick
                    # once they've landed (saves the host round-trip).
                    self._chassis_brake_settled = True
                else:
                    for name, val in cmd_4ws_drive_vel.items():
                        idx = self._joint_name_to_dof.get(name)
                        if idx is None or idx >= len(vel):
                            unknown.append(name)
                            continue
                        fval = float(val)
                        vel[idx] = fval
                        vel_writes.append((name, idx, fval))
                    # Fresh stamp received — re-arm the brake so the
                    # next stale window fires once.
                    self._chassis_brake_settled = False

            if pos_writes:
                # In-place GPU memcpy into the existing buffer (preserves the
                # pointer the captured CUDA graph holds).
                tgt_attr.assign(tgt)
            if vel_writes and vel_attr is not None:
                vel_attr.assign(vel)

            # One-time-per-joint diagnostic: log the FIRST command we see
            # for each unique joint name (rather than only the first call,
            # which is often a partial gripper-only command). Also tracks
            # commands actually written so we can confirm the arm is
            # receiving targets through /joint_command.
            if not hasattr(self, "_apply_cmd_seen"):
                self._apply_cmd_seen = set()
            new_joints = [
                (name, idx, val) for name, idx, val in pos_writes + vel_writes if name not in self._apply_cmd_seen
            ]
            for name, _i, _v in new_joints:
                self._apply_cmd_seen.add(name)
            if new_joints:
                self._logger.info(
                    f"[newton-standalone] apply_commands first-target for "
                    f"{len(new_joints)} new joint(s): " + ", ".join(f"{n}[dof={i}]={v:.4f}" for n, i, v in new_joints)
                )
            if unknown and not getattr(self, "_apply_cmd_unknown_logged", False):
                self._apply_cmd_unknown_logged = True
                self._logger.warn(
                    f"[newton-standalone] apply_commands: unknown joint(s) in /joint_command or "
                    f"/cmd_4ws: {unknown}; known: {list(self._joint_name_to_dof.keys())}"
                )
        except Exception as exc:
            self._logger.warn(f"[newton-standalone] apply_commands: {exc}")

    def _build_chassis_drive_dof_set(self) -> Dict[str, int]:
        """Map ``{chassis_drive joint name: DOF index}`` for the brake path.

        Used by :meth:`apply_commands` to zero drive-wheel velocities
        when ``/cmd_4ws`` goes stale.  Built lazily on first command so
        it picks up the final post-``_build_joint_map`` state of
        ``self._joint_name_to_dof``.

        Implementation mirrors the adapter's per-DOF classifier
        (``MuJoCoWarpAdapter._params_class_for_dof``) — same
        ``classify_joint_by_name`` + ``is_chassis_wheel_free`` regex
        path, just expressed in radians via the Newton-side limit
        threshold so we don't have to round-trip through degrees.
        """
        from genie_sim_engine_newton.common.joint_classification import (  # noqa: PLC0415
            JK_CHASSIS_DRIVE,
            JK_CHASSIS_WHEEL,
            classify_joint_by_name,
            is_chassis_wheel_free,
        )

        out: Dict[str, int] = {}
        if not self._joint_name_to_dof or self._model is None:
            return out
        try:
            jq_lower = self._model.joint_limit_lower.numpy() if self._model.joint_limit_lower is not None else None
            jq_upper = self._model.joint_limit_upper.numpy() if self._model.joint_limit_upper is not None else None
        except Exception:  # noqa: BLE001
            jq_lower = jq_upper = None

        for name, dof in self._joint_name_to_dof.items():
            kind = classify_joint_by_name(name)
            if kind == JK_CHASSIS_DRIVE:
                out[name] = dof
                continue
            if kind != JK_CHASSIS_WHEEL:
                continue
            # Pre-split chassis wheel: refine via limit width on the
            # Newton-side threshold (radians; ≈ 12.217 ≈ 700°).
            lo = float(jq_lower[dof]) if jq_lower is not None and dof < len(jq_lower) else None
            hi = float(jq_upper[dof]) if jq_upper is not None and dof < len(jq_upper) else None
            if is_chassis_wheel_free(lo, hi, threshold=12.217):
                out[name] = dof
        if out:
            self._logger.info(
                f"[newton-standalone] chassis_drive DOF set ({len(out)} joint(s)) "
                f"cached for /cmd_4ws timeout brake: {list(out.keys())}"
            )
        return out

    # ------------------------------------------------------------------
    # Direct accessors for the run loop's snapshot/publish stage.
    # USD-backed snapshot helpers (snapshot_joint_states, snapshot_body_transforms,
    # snapshot_odom) read PhysX-written USD attributes — Newton-direct doesn't
    # write USD, only Fabric. The run loop prefers these accessors when present.
    # ------------------------------------------------------------------
