# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Receive actual Newton GL camera batches without integrating physics."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping

import numpy as np


def decode_image(message) -> np.ndarray:
    """Decode ROS color images, respecting row padding and channel order."""
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}
    if message.encoding not in channels:
        raise ValueError(f"unsupported camera encoding: {message.encoding}")
    count = channels[message.encoding]
    height, width, stride = int(message.height), int(message.width), int(message.step)
    if min(height, width) <= 0 or stride < width * count or len(message.data) != height * stride:
        raise ValueError("invalid ROS image dimensions, stride or data length")
    pixels = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(height, stride)
    pixels = pixels[:, : width * count].reshape(height, width, count)[:, :, :3]
    if message.encoding in ("bgr8", "bgra8"):
        pixels = pixels[:, :, ::-1]
    return pixels.copy()


class RosImages:
    """Keep the latest stamped batch; closing leaves the shared ROS context alive."""

    def __init__(self, topics: Mapping[str, str]):
        if set(topics) != {"head", "hand_left", "hand_right"} or len(set(topics.values())) != 3:
            raise ValueError("configure three distinct head/hand_left/hand_right image topics")
        self._topics = dict(topics)
        self._condition = threading.Condition()
        self._frames = {}
        self._error = None
        self._node = None
        self._executor = None
        self._thread = None

    def start(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = Node(f"wuji_eval_images_{os.getpid()}")
            qos = QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
            for name, topic in self._topics.items():
                self._node.create_subscription(Image, topic, lambda msg, key=name: self._receive(key, msg), qos)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._thread = threading.Thread(target=self._spin, name="wuji-eval-images", daemon=True)
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
                self._error = error
                self._condition.notify_all()

    def _receive(self, name, message):
        try:
            frame = decode_image(message)
            stamp = message.header.stamp
            stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            with self._condition:
                self._frames[name] = (stamp_ns, frame)
                self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def wait(self, *, after_ns: int, timeout_s: float) -> tuple[int, dict[str, np.ndarray]]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError("Newton camera transport failed") from self._error
                if set(self._frames) == set(self._topics):
                    stamps = {stamp for stamp, _ in self._frames.values()}
                    if len(stamps) == 1 and next(iter(stamps)) >= after_ns:
                        return next(iter(stamps)), {name: value[1].copy() for name, value in self._frames.items()}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no fresh, matching three-camera batch after confirmed physics pause")
                self._condition.wait(min(remaining, 0.1))

    def close(self):
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
                    self._executor = self._thread = self._node = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
