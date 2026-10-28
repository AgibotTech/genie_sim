# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""ROS contract for episode lifecycle — reset coordination and activity signaling.

Provides ``RosResetCoordinator``, a backend-agnostic ROS serving layer for
RL episode resets. Handles the ``/reset_envs`` subscriber, thread-safe queue,
reset_epoch credential, and ``/joint_command`` activity stamping. Backends
implement ``episode_reset(env_ids, seeds, instances)`` to actually rewrite
state; this class coordinates the ROS → backend call.

Reusable by any backend that serves ROS-based RL clients. IsaacSim, mujoco_next,
or any future backend can compose this without modification.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Sequence


class RosResetCoordinator:
    """ROS contract for per-world resets + activity signaling.

    Backend-agnostic: handles the rclpy subscriber, thread-safe pending queue,
    ``reset_epoch`` credential (written into ``/joint_states`` frame_id), and
    ``/joint_command`` activity stamping (for idle gate / lockstep). The backend
    implements ``episode_reset(env_ids, seeds, instances)`` to actually rewrite
    state; this coordinator orchestrates the ROS → backend call.

    Thread safety
    -------------
    ``push_reset`` runs on the rclpy executor thread (callback). ``drain_resets``
    runs on the loop thread (inside ``apply_commands``, between frames). The
    pending list is guarded by ``_lock``.

    Activity signal
    ---------------
    ``last_cmd_wall`` and ``idle_activity_ok`` are read by ``RealtimePacer``
    (idle gate) and ``LockstepPacer`` (startup grace + burst engage). The rclpy
    ``/joint_command`` subscriber stamps ``_last_cmd_wall`` on every message
    arrival (not on value change — RL legitimately re-sends identical actions).
    ``idle_activity_ok`` is True once the subscriber is up; False before then,
    so a backend without the activity signal degrades to free-run rather than
    tripping the idle gate during bring-up.
    """

    def __init__(self, backend: Any, logger: Any, topic: str = "/rlinf/reset_envs") -> None:
        """
        Parameters
        ----------
        backend
            The physics backend. Must implement
            ``episode_reset(env_ids, seeds, instances)``.
        logger
            Logger instance (``logging.Logger`` or compatible).
        topic
            Base topic for reset messages. Subscribers:
              * ``{topic}`` — ``std_msgs/Int32MultiArray`` of world indices
              * ``{topic}_seeded`` — ``std_msgs/Int64MultiArray`` of flat
                triples ``[env, seed, inst, ...]``
        """
        self._backend = backend
        self._logger = logger
        self._topic = topic
        self._lock = threading.Lock()
        self._pending_resets: List = []
        self._last_cmd_wall: float = 0.0
        self._idle_activity_ok: bool = False
        self._node = None
        self._executor = None
        self._spin_thread = None

    def start_listener(self) -> None:
        """Bring up a small rclpy node subscribing reset + activity topics.

        Separate from the engine's C++ ROS bridge (different participant in
        the same process — same pattern the debug pubs use). The callbacks
        only enqueue; actual state writes happen on the loop thread via
        ``drain_resets``.

        Subscriptions
        -------------
        * ``/rlinf/reset_envs`` (``Int32MultiArray``, RELIABLE) — world indices
        * ``/rlinf/reset_envs_seeded`` (``Int64MultiArray``, RELIABLE) — flat
          triples ``[env, seed, inst, ...]`` for per-env deterministic
          randomization and/or in-group instance switching
        * ``/joint_command`` (``JointState``, BEST_EFFORT) — activity stamp
          for idle gate and lockstep engage

        Errors
        ------
        rclpy import failure, subscription setup failure, or already-running
        all log a warning and return without raising. A failure here means
        the engine runs without RL reset capability, but other paths (teleop,
        data collection) are unaffected.
        """
        try:
            import rclpy  # noqa: PLC0415
            from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor  # noqa: PLC0415
            from rclpy.node import Node  # noqa: PLC0415
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: PLC0415
            from std_msgs.msg import Int32MultiArray  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[lifecycle-ros] rclpy import failed: {exc}; /reset_envs disabled")
            return

        try:
            if not rclpy.ok():
                rclpy.init()

            node = Node("genie_sim_engine_reset")
            # RELIABLE: a reset is a one-shot event — dropping it would leave
            # a world un-reset (corrupt episode). depth 4 absorbs bursts.
            qos = QoSProfile(depth=4, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)

            def _cb(msg) -> None:
                self.push_reset(list(msg.data))

            node.create_subscription(Int32MultiArray, self._topic, _cb, qos)

            # Seeded variant: Int64MultiArray of FLAT TRIPLES
            # [env, seed, instance, ...] — per-env deterministic randomization
            # and/or in-group instance switching at reset (-1 = no-op slot).
            # Kept as a separate topic so the plain /reset_envs stays
            # byte-compatible.
            from std_msgs.msg import Int64MultiArray  # noqa: PLC0415

            def _cb_seeded(msg) -> None:
                d = list(msg.data)
                self.push_reset(d[0::3], seeds=d[1::3], instances=d[2::3])

            node.create_subscription(Int64MultiArray, self._topic + "_seeded", _cb_seeded, qos)

            # Command-ARRIVAL stamp for the idle gate (common/pacing.py). The C++
            # bridge's pop_commands LATCHES (snapshot, never clears), so command
            # dicts look non-empty forever after the first message — arrival
            # time is the only correct activity signal. Value ignored on
            # purpose: RL legitimately re-sends identical actions (hold pose),
            # so value-change detection would false-pause a live client.
            try:
                from sensor_msgs.msg import JointState  # noqa: PLC0415

                def _cmd_cb(_msg) -> None:
                    self._last_cmd_wall = time.monotonic()

                node.create_subscription(
                    JointState,
                    "/joint_command",
                    _cmd_cb,
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST),
                )
                self._last_cmd_wall = 0.0
                self._idle_activity_ok = True
            except Exception as exc:  # noqa: BLE001
                self._logger.warn(
                    f"[lifecycle-ros] /joint_command activity sub failed: {exc}; " f"idle gate stays disabled"
                )

            ex = SingleThreadedExecutor()
            ex.add_node(node)
            def _spin() -> None:
                try:
                    ex.spin()
                except ExternalShutdownException:
                    # SIGINT closes the ROS context before this daemon exits.
                    pass

            th = threading.Thread(target=_spin, name="reset-listener-spin", daemon=True)
            th.start()
            self._node = node
            self._executor = ex
            self._spin_thread = th
            self._logger.info(
                f"[lifecycle-ros] reset listener up on {self._topic} " f"(std_msgs/Int32MultiArray, RELIABLE)"
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[lifecycle-ros] start_listener failed: {exc}")

    def push_reset(
        self,
        env_ids: Sequence[int],
        seeds: Optional[Sequence[int]] = None,
        instances: Optional[Sequence[int]] = None,
    ) -> None:
        """Enqueue a reset request (called from rclpy callback, thread-safe).

        Parameters
        ----------
        env_ids
            World indices to reset. Empty list = all worlds.
        seeds
            Optional per-env randomization seeds. ``-1`` = no randomization.
        instances
            Optional per-env instance IDs for in-group switching. ``-1`` = no-op.
        """
        with self._lock:
            self._pending_resets.append((list(env_ids), seeds, instances))

    @staticmethod
    def _per_env_map(env_ids: Sequence[int], values: Optional[Sequence[int]]) -> Optional[Dict[int, int]]:
        """Zip a wire-side parallel array onto its env ids, dropping no-ops.

        The seeded topic carries flat triples ``[env, seed, inst, ...]``, so
        seeds and instances arrive as arrays aligned with ``env_ids`` — but
        ``episode_reset`` takes ``{env_id: value}`` mappings, because a batch
        may name the same world twice and a positional array cannot say which
        entry wins. ``-1`` is the wire's "no-op for this slot" and must not
        reach the backend as a real value: a seed of -1 would look like a
        randomization request, and an instance of -1 would index nothing.

        Returns ``None`` rather than an empty dict when nothing survives, so
        the backend's ``if seeds and w in seeds`` guards short-circuit on the
        common unseeded reset.
        """
        if values is None:
            return None
        out = {int(e): int(v) for e, v in zip(env_ids, values) if int(v) >= 0}
        return out or None

    def drain_resets(self) -> None:
        """Apply all pending resets (called from loop thread, between frames).

        Pops the pending queue and calls ``backend.episode_reset(env_ids,
        seeds, instances)`` once per pushed batch. No-op (and cheap) when
        nothing is pending.

        One call per batch, not one coalesced call for the queue: the reset
        epoch is the credential a wrapper waits on, and it advances once per
        completed ``episode_reset``, so merging two requests into one write
        would leave a caller that asked twice waiting for a bump that never
        comes.

        An empty ``env_ids`` is the wire's "every world" (the ``/reset_envs``
        topic has no other way to say it) and is forwarded as ``None``, which
        is the ``episode_reset`` spelling of the same thing.
        """
        with self._lock:
            batch = self._pending_resets[:]
            self._pending_resets.clear()

        for env_ids, seeds, instances in batch:
            ids = [int(w) for w in env_ids]
            try:
                self._backend.episode_reset(
                    ids or None,
                    seeds=self._per_env_map(ids, seeds),
                    instances=self._per_env_map(ids, instances),
                )
            except Exception as exc:  # noqa: BLE001
                # Deliberately does NOT advance the epoch: a failed reset left
                # state as it was, and issuing the credential anyway would tell
                # a waiting wrapper that stale state is post-reset.
                self._logger.warn(f"[lifecycle-ros] episode_reset failed: {exc}")

    @property
    def reset_epoch(self) -> int:
        """Monotonic reset credential, starting at 0.

        Read straight off the backend, which bumps it inside
        ``episode_reset`` only after every state write has landed. This
        coordinator deliberately keeps no counter of its own: it is not the
        only way to reach ``episode_reset`` (an RL consumer holding the
        engine calls it directly), and a second counter would answer the
        credential question differently depending on which door the reset
        came through.

        Written into ``/joint_states`` header.frame_id as ``reset_epoch:{N}``
        so RL clients can detect when state has been reset (causal ordering).
        """
        return int(getattr(self._backend, "reset_epoch", 0) or 0)

    @property
    def last_cmd_wall(self) -> float:
        """Wall time of the last ``/joint_command`` arrival (monotonic seconds).

        Stamped by the rclpy subscriber on every message, independent of value.
        Read by ``RealtimePacer`` (idle gate) and ``LockstepPacer`` (engage).
        Returns 0.0 before the first command.
        """
        return self._last_cmd_wall

    @property
    def reset_pending(self) -> bool:
        """Is a reset queued and not yet carried out?

        Read by the pacer as an anti-idle signal, which is the whole reason
        it exists: a queued reset needs a tick to be applied ON, and a client
        that just asked for one is by definition not idle. Without this, a
        reset arriving into a quiet run would sit in the queue until the next
        command woke the loop up.

        Unlocked on purpose. The answer is a hint, sampled every tick, and it
        is already stale by the time the caller branches on it -- taking
        ``_lock`` here would serialise the hot loop against the rclpy
        callback thread to sharpen a value whose only two outcomes are "step
        now" and "step on the next arrival".
        """
        return bool(self._pending_resets)

    @property
    def idle_activity_ok(self) -> bool:
        """Is the activity signal ready?

        True once the ``/joint_command`` subscriber is up; False before then.
        Pacers check this before reading ``last_cmd_wall`` — a False here
        means degrade to free-run rather than tripping the idle gate during
        bring-up.
        """
        return self._idle_activity_ok
