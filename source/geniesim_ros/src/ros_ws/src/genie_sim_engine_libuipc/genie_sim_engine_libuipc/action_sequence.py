# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Immutable action requests counted only by committed native IPC solves.

Locally adapted from the action scheduler in history 0ab2b58. Request payloads
and replies remain owned for the entire epoch: evicting an old request ID would
allow a delayed retry to execute the same action twice.
"""

from __future__ import annotations

import copy
import json
import math
import threading

import numpy as np


def topic_from_scene(scene_cfg, key):
    config = (scene_cfg.get("libuipc") or {}).get(key)
    if config is None:
        return None
    if (
        not isinstance(config, dict)
        or set(config) != {"topic"}
        or not isinstance(config["topic"], str)
        or not config["topic"].startswith("/")
        or config["topic"] == "/"
        or config["topic"].endswith("/")
    ):
        raise ValueError(f"libuipc.{key} requires an absolute topic prefix without a trailing slash")
    return config["topic"]


def identity_packet(engine, status):
    """Plain discovery data, created only at a physics-owner lifecycle boundary."""
    if status not in ("ready", "failed", "closed"):
        raise ValueError("invalid physical source status")
    if status == "ready":
        if getattr(engine, "_closed", False):
            status = "closed"
        elif getattr(engine, "_execution_failure", None) is not None:
            status = "failed"
    actions = getattr(engine, "_actions", None)
    return dict(
        version=1,
        source_id=engine._source_id,
        epoch=engine._epoch,
        tick=engine._tick,
        sim_time=engine._sim_time,
        physics_hz=engine._physics_hz,
        joint_names=list(actions.names if actions is not None else engine._scalar),
        reported_joint_names=list(engine._joint_names),
        status=status,
        ready=status == "ready",
    )


class StateIdentitySource:
    """Latched discovery on each enabled protocol prefix, using one existing node."""

    def __init__(self, engine, node, prefixes):
        from std_msgs.msg import String
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        if node is None:
            raise RuntimeError("state identity requires the existing lifecycle ROS node")
        self.engine, self.node, self._String = engine, node, String
        self._publishers = []
        self._closed = False
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        try:
            for prefix in dict.fromkeys(prefixes):
                self._publishers.append(node.create_publisher(String, prefix + "/identity", qos))
        except BaseException:
            for publisher in self._publishers:
                node.destroy_publisher(publisher)
            raise

    def publish(self, status):
        if self._closed:
            return
        try:
            message = self._String(data=json.dumps(identity_packet(self.engine, status), allow_nan=False))
            for publisher in self._publishers:
                publisher.publish(message)
        except Exception as exc:
            report = getattr(self.engine, "report_transport_failure", None)
            if report is not None:
                report(exc)
            raise

    def close(self):
        if self._closed:
            return
        try:
            self.publish("closed")
        finally:
            self._closed = True
            failure = None
            for publisher in self._publishers:
                try:
                    self.node.destroy_publisher(publisher)
                except BaseException as exc:
                    failure = failure or exc
            if failure is not None:
                raise failure


class ActionSequence:
    def __init__(self, names, initial, *, physics_hz, source_id, epoch=0):
        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("controlled joint names must be nonempty and unique")
        self.physics_hz = float(physics_hz)
        if not math.isfinite(self.physics_hz) or self.physics_hz <= 0:
            raise ValueError("action physics_hz must be positive and finite")
        if not isinstance(source_id, str) or not source_id or type(epoch) is not int or epoch < 0:
            raise ValueError("invalid action source identity")
        self.source_id, self.epoch = source_id, epoch
        self._held = self._targets(initial)
        self._pending = self._active = self._in_step = None
        self._requests = {}
        self._closed = False
        self._lock = threading.RLock()

    def _targets(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.names),) or not np.isfinite(values).all():
            raise ValueError("invalid action positions")
        return values.copy()

    def _validate(self, packet):
        if not isinstance(packet, dict):
            raise ValueError("action request must be an object")
        if set(packet) != {"request_id", "source_id", "epoch", "names", "positions", "duration_s"}:
            raise ValueError("invalid action request fields")
        token = packet.get("request_id")
        if not isinstance(token, str) or not token or len(token) > 128:
            raise ValueError("invalid action request_id")
        if packet.get("source_id") != self.source_id:
            raise ValueError("stale action source_id")
        if type(packet.get("epoch")) is not int or packet["epoch"] != self.epoch:
            raise ValueError("stale action epoch")
        names = packet.get("names")
        if (
            not isinstance(names, list)
            or not all(isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
            or set(names) != set(self.names)
        ):
            raise ValueError("action must contain each controlled joint exactly once")
        raw = self._targets(packet.get("positions"))
        by_name = dict(zip(names, raw))
        targets = np.asarray([by_name[name] for name in self.names])
        duration = packet.get("duration_s")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("action duration must be positive and finite")
        ticks = duration * self.physics_hz
        if ticks < 1 or not math.isclose(ticks, round(ticks), rel_tol=0, abs_tol=1e-7):
            raise ValueError("action duration must contain an integer number of IPC steps")
        # The canonical payload permits joint reordering, while retaining the
        # requested duration to detect changed payloads sharing an old ID.
        payload = (tuple(targets), float(duration))
        return token, targets, int(round(ticks)), payload

    def submit(self, packet):
        with self._lock:
            token, targets, ticks, payload = self._validate(packet)
            previous = self._requests.get(token)
            if previous is not None:
                if previous["payload"] != payload:
                    raise ValueError("action request_id payload conflict")
                return copy.deepcopy(previous["reply"])
            if self._closed:
                raise RuntimeError("action source is closed")
            if self._active is not None or self._pending is not None:
                raise RuntimeError("previous action is still executing")
            reply = self._reply(token, "accepted", ipc_steps=ticks)
            action = dict(
                request_id=token, targets=targets, remaining=ticks, ipc_steps=ticks, completed=0, cpu_start=None
            )
            self._requests[token] = dict(payload=payload, reply=reply)
            self._pending = action
            return copy.deepcopy(reply)

    def _reply(self, token, status, **fields):
        return dict(request_id=token, source_id=self.source_id, epoch=self.epoch, status=status, **fields)

    def begin_step(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("action source is closed")
            if self._active is None and self._pending is not None:
                self._active, self._pending = self._pending, None
                self._held = self._active["targets"].copy()
            self._in_step = self._active
            return self._held.copy()

    def complete_step(self, *, tick, sim_time, cpu_steps=None, cpu_batch_steps=None):
        with self._lock:
            action, self._in_step = self._in_step, None
            if action is None:
                return None
            if cpu_steps is not None and action["cpu_start"] is None:
                # CPU counts belong to a worker owner, not the engine clock.
                batch = cpu_batch_steps if cpu_batch_steps is not None else cpu_steps
                action["cpu_start"] = cpu_steps - batch
            action["remaining"] -= 1
            action["completed"] += 1
            if action["remaining"]:
                return None
            result = self._reply(
                action["request_id"],
                "completed",
                ipc_steps=action["ipc_steps"],
                executed_sim_time_s=action["ipc_steps"] / self.physics_hz,
                tick=int(tick),
                sim_time=float(sim_time),
            )
            if cpu_steps is not None:
                result["cpu_steps"] = int(cpu_steps - action["cpu_start"])
                result["cpu_total_steps"] = int(cpu_steps)
            self._requests[action["request_id"]]["reply"] = result
            self._active = None
            return copy.deepcopy(result)

    def cancel(self, reason, *, hold):
        with self._lock:
            self._held = self._targets(hold)
            action = self._active or self._pending
            self._active = self._pending = self._in_step = None
            if action is None:
                return None
            result = self._reply(action["request_id"], "cancelled", error=str(reason), ipc_steps=action["completed"])
            self._requests[action["request_id"]]["reply"] = result
            return copy.deepcopy(result)

    def reset(self, *, epoch, hold):
        with self._lock:
            if type(epoch) is not int or epoch <= self.epoch:
                raise ValueError("reset action epoch must increase")
            result = self.cancel("episode reset", hold=hold)
            self.epoch = epoch
            self._requests.clear()
            self._closed = False
            return result

    def close(self, reason="engine closed"):
        with self._lock:
            result = self.cancel(reason, hold=self._held)
            self._closed = True
            return result


class ActionSequenceSource:
    """JSON transport on the engine's existing lifecycle node/executor."""

    def __init__(self, actions, node, topic, *, on_failure=None):
        from std_msgs.msg import String

        if node is None:
            raise RuntimeError("action_sequence requires the existing lifecycle ROS node")
        self.actions, self.node, self._String = actions, node, String
        self._on_failure = on_failure
        self._transport_lock = threading.RLock()
        self._closed = False
        self.publisher = node.create_publisher(String, topic + "/done", 10)
        try:
            self.subscription = node.create_subscription(String, topic + "/request", self.request, 10)
        except BaseException:
            node.destroy_publisher(self.publisher)
            raise

    def request(self, message):
        with self._transport_lock:
            if self._closed:
                return None
            return self._request(message)

    def _request(self, message):
        packet = None
        try:
            packet = json.loads(message.data)
            reply = self.actions.submit(packet)
        except Exception as exc:
            token = packet.get("request_id") if isinstance(packet, dict) else None
            reply = dict(
                request_id=token if isinstance(token, str) else None,
                source_id=self.actions.source_id,
                epoch=self.actions.epoch,
                status="rejected",
                error=str(exc),
            )
        try:
            self.publish(reply)
        except Exception:
            if self._on_failure is None:
                raise
            # The fault sink only records plain data; owner-thread cleanup is
            # deferred so the shared ROS executor and archive remain safe.
        return reply

    def publish(self, reply):
        with self._transport_lock:
            if reply is not None and not self._closed:
                try:
                    self.publisher.publish(self._String(data=json.dumps(reply, allow_nan=False)))
                except Exception as exc:
                    if self._on_failure is not None:
                        self._on_failure(exc)
                    raise

    def close(self):
        with self._transport_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.node.destroy_subscription(self.subscription)
            finally:
                self.node.destroy_publisher(self.publisher)
