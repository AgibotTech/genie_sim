# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Synchronous snapshot rendering in an owned, separate Isaac Sim process.

This module deliberately imports no Kit, Newton or ROS packages. The child
receives completed scene snapshots; it never advances the physics process.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from multiprocessing import Pipe
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

import numpy as np


class IsaacSimRenderer:
    """Own one renderer worker; any transport/protocol failure is terminal.

    ``render`` is a blocking capture barrier. Returned RGB arrays are owned
    copies, and only the response for the exact outstanding sequence is usable.
    The Pipe carries trusted local child data, not a network-facing protocol.
    """

    def __init__(
        self,
        worker_path: Path,
        log_path: Path,
        *,
        timeout_s: float = 30.0,
        startup_timeout_s: float = 120.0,
    ) -> None:
        self._timeout = self._positive_timeout(timeout_s, "timeout_s")
        startup_timeout = self._positive_timeout(startup_timeout_s, "startup_timeout_s")
        worker = Path(worker_path).expanduser().resolve(strict=True)
        if not worker.is_file():
            raise ValueError("Isaac Sim worker_path must be a file")
        self.log_path = Path(log_path).expanduser().resolve()
        self._connection = None
        self._process = None
        self._log = None
        self._sequence = 0
        self._closed = False
        self._poisoned = False
        self._lock = threading.RLock()
        self._io_thread = None
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("xb")
            self._connection, child = Pipe(duplex=True)
            try:
                self._process = subprocess.Popen(
                    [sys.executable, str(worker), "--ipc-fd", str(child.fileno())],
                    stdin=subprocess.DEVNULL,
                    stdout=self._log,
                    stderr=self._log,
                    pass_fds=(child.fileno(),),
                    start_new_session=True,
                )
            finally:
                child.close()
            ready = self._exchange(None, startup_timeout)
            self._check_error(ready)
            if not isinstance(ready, dict) or ready.get("ready") is not True or ready.get("renderer") != "isaacsim":
                raise RuntimeError("invalid Isaac Sim worker readiness response")
        except BaseException:
            self._abort()
            raise

    @staticmethod
    def _positive_timeout(value: float, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite and positive")
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be finite and positive") from error
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    def _bounded_io(self, operation: Callable[[], Any], timeout: float) -> Any:
        # A poll only bounds the first readable byte. A partial message recv,
        # or a large snapshot send to a stalled child, can still block. Keep
        # those calls off the caller thread so timeout can terminate the peer.
        done = threading.Event()
        outcome = {}

        def run() -> None:
            try:
                outcome["value"] = operation()
            except BaseException as error:
                outcome["error"] = error
            finally:
                done.set()

        thread = threading.Thread(target=run, name="isaacsim-render-ipc", daemon=True)
        self._io_thread = thread
        thread.start()
        if not done.wait(timeout):
            raise TimeoutError(f"Isaac Sim renderer timed out; see {self.log_path}")
        thread.join()
        self._io_thread = None
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    def _exchange(self, request: dict | None, timeout: float) -> Any:
        connection = self._connection

        def exchange():
            if request is not None:
                connection.send(request)
            if not connection.poll(timeout):
                raise TimeoutError(f"Isaac Sim renderer timed out; see {self.log_path}")
            return connection.recv()

        try:
            return self._bounded_io(exchange, timeout)
        except TimeoutError:
            raise
        except (EOFError, OSError) as error:
            raise RuntimeError(f"Isaac Sim renderer disconnected; see {self.log_path}") from error

    @staticmethod
    def _check_error(response: Any) -> None:
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Isaac Sim worker failed: {response['error']}")

    @staticmethod
    def _cameras(cameras: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = []
        names = set()
        for camera in cameras:
            if not isinstance(camera, Mapping):
                raise ValueError("each camera must be a mapping")
            name = camera.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("camera names must be nonempty and unique")
            names.add(name)
            row = {"name": name}
            for key in ("eye", "target", "up"):
                value = np.asarray(camera.get(key), dtype=np.float64)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(f"camera {name} requires finite {key} xyz")
                row[key] = value.tolist()
            fov = camera.get("fov_degrees")
            if isinstance(fov, bool):
                raise ValueError("camera fov_degrees must be between 0 and 180")
            try:
                fov = float(fov)
            except (TypeError, ValueError) as error:
                raise ValueError("camera fov_degrees must be between 0 and 180") from error
            if not math.isfinite(fov) or not 0 < fov < 180:
                raise ValueError("camera fov_degrees must be between 0 and 180")
            row["fov_degrees"] = fov
            result.append(row)
        if not result:
            raise ValueError("at least one camera is required")
        return result

    def render(
        self,
        scene: Mapping[str, Any],
        cameras: Sequence[Mapping[str, Any]],
        width: int,
        height: int,
    ) -> dict[str, np.ndarray]:
        with self._lock:
            if self._poisoned:
                raise RuntimeError("Isaac Sim renderer is poisoned after a previous failure")
            if self._closed:
                raise RuntimeError("Isaac Sim renderer is closed")
            if type(width) is not int or type(height) is not int or min(width, height) < 1:
                raise ValueError("render width and height must be positive integers")
            if not isinstance(scene, Mapping):
                raise ValueError("render scene must be a mapping")
            rows = self._cameras(cameras)
            self._sequence += 1
            try:
                response = self._exchange(
                    dict(
                        op="render",
                        sequence=self._sequence,
                        scene=dict(scene),
                        cameras=rows,
                        width=width,
                        height=height,
                    ),
                    self._timeout,
                )
                self._check_error(response)
                if (
                    not isinstance(response, dict)
                    or response.get("renderer") != "isaacsim"
                    or type(response.get("sequence")) is not int
                    or response["sequence"] != self._sequence
                ):
                    raise RuntimeError("Isaac Sim renderer response identity or sequence mismatch")
                images = response.get("images")
                if not isinstance(images, dict) or set(images) != {camera["name"] for camera in rows}:
                    raise RuntimeError("Isaac Sim renderer returned an incomplete camera batch")
                result = {}
                for name, pixels in images.items():
                    if (
                        not isinstance(pixels, np.ndarray)
                        or pixels.dtype != np.uint8
                        or pixels.shape != (height, width, 3)
                    ):
                        raise RuntimeError(f"Isaac Sim renderer returned invalid RGB image: {name}")
                    result[name] = np.array(pixels, copy=True, order="C")
                return result
            except BaseException:
                self._abort()
                raise

    def _abort(self) -> None:
        self._poisoned = True
        try:
            self.close()
        except Exception:
            # Preserve the original startup/capture failure. close() still
            # closes the Pipe and log even if process reaping itself fails.
            pass

    def close(self) -> None:
        with self._lock:
            if self._closed and self._process is None:
                return
            self._closed = True
            process = self._process
            try:
                if process is not None:
                    if process.poll() is None and not self._poisoned:
                        try:
                            self._bounded_io(lambda: self._connection.send({"op": "close"}), 0.1)
                            process.wait(timeout=1.0)
                        except (OSError, EOFError, TimeoutError, subprocess.TimeoutExpired):
                            pass
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=1.0)
                    else:
                        process.wait(timeout=1.0)
                    self._process = None
            finally:
                if self._connection is not None:
                    self._connection.close()
                    self._connection = None
                if self._io_thread is not None:
                    self._io_thread.join(timeout=1.0)
                    self._io_thread = None
                if self._log is not None:
                    self._log.close()
                    self._log = None
