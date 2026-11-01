# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Receive authoritative per-physics-tick Wuji task scores, independently of RGB."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
import threading
import time
from typing import Any


@dataclass(frozen=True)
class TaskProgress:
    session_id: str
    reset_epoch: int
    tick: int
    sim_time: float
    published_ns: int
    score: int
    max_score: int
    milestones: Mapping[str, bool]
    failed: bool
    failure_reason: str | None
    terminal: bool
    metrics: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    error: str | None = None

    @classmethod
    def from_message(cls, value: Mapping[str, Any]) -> TaskProgress:
        if (
            not isinstance(value, Mapping)
            or type(value.get("version")) is not int
            or value.get("version") != 1
            or value.get("task_name") != "wok_flip"
        ):
            raise ValueError("unsupported Wuji task progress protocol")
        session = value.get("session_id")
        if not isinstance(session, str) or not session:
            raise ValueError("task progress requires a nonempty session_id")
        for key in ("reset_epoch", "tick", "published_ns", "score", "max_score"):
            if type(value.get(key)) is not int or value[key] < 0:
                raise ValueError(f"task progress {key} must be a nonnegative integer")
        sim_time = value.get("sim_time")
        if (
            isinstance(sim_time, bool)
            or not isinstance(sim_time, (int, float))
            or not math.isfinite(sim_time)
            or sim_time < 0
        ):
            raise ValueError("task progress sim_time must be finite and nonnegative")
        milestones = value.get("milestones")
        if (
            not isinstance(milestones, Mapping)
            or set(milestones) != {"stable_lift", "food_toss"}
            or any(type(item) is not bool for item in milestones.values())
            or (milestones["food_toss"] and not milestones["stable_lift"])
        ):
            raise ValueError("invalid Wuji scoring milestones")
        if value["max_score"] != 2 or value["score"] != sum(milestones.values()):
            raise ValueError("task progress score does not match its two milestones")
        failed, terminal, reason = value.get("failed"), value.get("terminal"), value.get("failure_reason")
        if type(failed) is not bool or type(terminal) is not bool or terminal != failed:
            raise ValueError("task progress terminal must equal its failure state")
        if (failed and (not isinstance(reason, str) or not reason)) or (not failed and reason is not None):
            raise ValueError("invalid task progress failure reason")
        for key in ("metrics", "thresholds"):
            if not isinstance(value.get(key), Mapping):
                raise ValueError(f"task progress requires {key}")
            json.dumps(dict(value[key]), allow_nan=False)
        error = value.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError("task progress error must be text")
        return cls(
            session,
            value["reset_epoch"],
            value["tick"],
            float(sim_time),
            value["published_ns"],
            value["score"],
            value["max_score"],
            deepcopy(dict(milestones)),
            failed,
            reason,
            terminal,
            deepcopy(dict(value["metrics"])),
            deepcopy(dict(value["thresholds"])),
            error,
        )

    @property
    def task_success(self) -> bool:
        return self.score == self.max_score and not self.failed

    def as_dict(self) -> dict[str, Any]:
        return deepcopy({"version": 1, "task_name": "wok_flip", **vars(self)})

    def wire_progress(self) -> list[dict[str, Any]]:
        return [
            {
                "class_name": name,
                "progress": {
                    "SCORE": int(self.milestones[key]),
                    "STATUS": "SUCCESS" if self.milestones[key] else "PENDING",
                },
            }
            for key, name in (("stable_lift", "WokStableLift"), ("food_toss", "WokFoodToss"))
        ]


class RosTaskProgress:
    """Wait for the score of a completed tick before issuing another action.

    Callbacks only update a cache. Pause/control operations stay on the runner
    thread, outside the client's exclusive step operation.
    """

    def __init__(self, topic: str):
        if not isinstance(topic, str) or not topic.startswith("/"):
            raise ValueError("task progress topic must be an absolute ROS topic")
        self._topic = topic
        self._condition = threading.Condition()
        self._latest: TaskProgress | None = None
        self._error: Exception | None = None
        self._node = self._executor = self._thread = None
        self._closed = False

    def _receive(self, message) -> None:
        try:
            item = TaskProgress.from_message(json.loads(message.data))
            with self._condition:
                previous = self._latest
                if previous is not None:
                    if previous.session_id != item.session_id:
                        raise RuntimeError("task progress changed engine session")
                    if item.reset_epoch < previous.reset_epoch or (
                        item.reset_epoch == previous.reset_epoch and item.tick < previous.tick
                    ):
                        return  # A delayed retained message cannot roll the state back.
                    if item.reset_epoch == previous.reset_epoch:
                        if item.score < previous.score or (previous.failed and not item.failed):
                            raise RuntimeError("task progress lost a latched milestone or failure")
                self._latest = item
                self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def wait(
        self, *, reset_epoch: int, after_tick: int | None = None, after_ns: int = 0, timeout_s: float = 30
    ) -> TaskProgress:
        if type(reset_epoch) is not int or reset_epoch < 0:
            raise ValueError("task scoring requires a confirmed reset_epoch")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("task progress timeout must be finite and positive")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError("Wuji task progress transport failed") from self._error
                if self._closed:
                    raise RuntimeError("Wuji task progress is closed")
                item = self._latest
                if item is not None:
                    if item.reset_epoch > reset_epoch:
                        raise RuntimeError("unexpected task reset while evaluating an episode")
                    if (
                        item.reset_epoch == reset_epoch
                        and item.published_ns >= after_ns
                        and (after_tick is None or item.tick > after_tick)
                    ):
                        return TaskProgress.from_message(item.as_dict())
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no matching task score after the completed physics tick/pause")
                self._condition.wait(min(remaining, 0.1))

    def start(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = Node(f"wuji_eval_progress_{os.getpid()}")
            qos = QoSProfile(
                depth=32, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
            self._node.create_subscription(String, self._topic, self._receive, qos)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._thread = threading.Thread(target=self._spin, name="wuji-eval-progress", daemon=True)
            self._thread.start()
            return self
        except BaseException:
            self.close()
            raise

    def _spin(self):
        try:
            self._executor.spin()
        except Exception as error:
            with self._condition:
                if not self._closed:
                    self._error = error
                    self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        try:
            if self._executor is not None:
                self._executor.shutdown()
        finally:
            try:
                if self._thread is not None:
                    self._thread.join(timeout=2)
            finally:
                try:
                    if self._node is not None:
                        self._node.destroy_node()
                finally:
                    self._node = self._executor = self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
