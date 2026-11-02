# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lazy ROS JSON transport with a private context and no process ownership."""

from __future__ import annotations

from collections import deque
import json
import threading
import time
import uuid


class IPCROSBridge:
    def __init__(self, config, *, discovery_timeout_s=120.0):
        self.config = config
        self.discovery_timeout_s = discovery_timeout_s
        self._condition = threading.Condition()
        self._queue = deque()
        self._error = None
        self._node = self._context = self._executor = self._thread = None
        self._publishers = {}
        self._closed = False

    def start(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self._String = String
        try:
            # A private context preserves both initialized and uninitialized host
            # default contexts. Never call the global init/shutdown pair here.
            self._context = Context()
            rclpy.init(args=[], context=self._context)
            self._node = Node("ipc_client_" + uuid.uuid4().hex[:12], context=self._context)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            qos = QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
            topics = {
                "identity": (self.config.action_topic + "/identity", qos),
                "done": (self.config.action_topic + "/done", 10),
                "snapshot": (self.config.observation_topic + "/snapshot", 10),
                "observation": (self.config.observation_topic + "/observation", 10),
            }
            for kind, (topic, policy) in topics.items():
                self._node.create_subscription(String, topic, lambda msg, key=kind: self._receive(key, msg), policy)
            for kind, topic in (("action", self.config.action_topic), ("snapshot", self.config.observation_topic)):
                self._publishers[kind] = self._node.create_publisher(String, topic + "/request", 10)
            self._thread = threading.Thread(target=self._spin, name="ipc-client-ros", daemon=True)
            self._thread.start()
            return self
        except BaseException:
            self.close()
            raise

    def _receive(self, kind, message):
        with self._condition:
            try:

                def invalid(value):
                    raise ValueError(f"non-finite JSON value: {value}")

                packet = json.loads(message.data, parse_constant=invalid)
                if not isinstance(packet, dict):
                    raise ValueError("IPC transport requires JSON objects")
                self._queue.append((kind, packet, message.data))
            except BaseException as exc:
                self._error = exc
            self._condition.notify_all()

    def _spin(self):
        try:
            self._executor.spin()
            if not self._closed:
                raise RuntimeError("IPC ROS executor stopped unexpectedly")
        except BaseException as exc:
            with self._condition:
                if not self._closed:
                    self._error = exc
                self._condition.notify_all()

    def publish(self, kind, packet):
        if self._closed or self._node is None:
            raise RuntimeError("IPC ROS bridge is closed")
        publisher = self._publishers[kind]
        deadline = time.monotonic() + self.discovery_timeout_s
        while publisher.get_subscription_count() < 1:
            if time.monotonic() >= deadline:
                raise TimeoutError("IPC request subscriber discovery timed out")
            with self._condition:
                if self._error is not None or self._closed:
                    raise RuntimeError("IPC bridge failed during subscriber discovery")
                self._condition.wait(0.01)
        publisher.publish(self._String(data=json.dumps(packet, allow_nan=False)))

    def wait_observation_ready(self, timeout_s, *, on_progress=None):
        """Discover the renderer before requesting state; images still need a correlated reply."""
        deadline = time.monotonic() + timeout_s
        while True:
            # Identity processing may read this transport queue. Keep callbacks
            # outside its condition and consume failures even if no renderer appears.
            if on_progress is not None:
                on_progress()
            with self._condition:
                if self._closed or self._node is None:
                    raise RuntimeError("IPC ROS bridge is closed")
                if self._error is not None:
                    raise RuntimeError(f"IPC ROS transport failed: {self._error}") from self._error
                if self._node.count_publishers(self.config.observation_topic + "/observation") > 0:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("IPC observation publisher discovery timed out")
                self._condition.wait(min(remaining, 0.01))

    def next(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"IPC ROS transport failed: {self._error}") from self._error
            while not self._queue:
                if self._error is not None:
                    raise RuntimeError(f"IPC ROS transport failed: {self._error}") from self._error
                if self._closed:
                    raise RuntimeError("IPC ROS bridge is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("IPC ROS response timed out")
                self._condition.wait(remaining)
            kind, packet, self.last_raw = self._queue.popleft()
            return kind, packet

    def close(self):
        self._closed = True
        failure = None
        for close in (
            lambda: self._executor.shutdown(timeout_sec=5) if self._executor else None,
            lambda: self._thread.join(timeout=5) if self._thread else None,
            lambda: self._node.destroy_node() if self._node else None,
            lambda: self._context.try_shutdown() if self._context else None,
        ):
            try:
                close()
            except BaseException as exc:
                failure = failure or exc
        if self._thread is not None and self._thread.is_alive():
            failure = failure or RuntimeError("IPC ROS executor thread did not stop")
        with self._condition:
            self._condition.notify_all()
        if failure is not None:
            raise RuntimeError(f"IPC transport cleanup failed: {failure}") from failure
