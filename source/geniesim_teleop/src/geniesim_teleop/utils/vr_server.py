# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import socket
import threading
import time

from geniesim_teleop.utils.logger import logger


class VRServer:
    def __init__(self, host=None, port=8080, input_timeout_s=0.25):
        self.host, self.port = host, port
        self.input_timeout_s = input_timeout_s
        self._sample = (None, 0.0)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host or "", port))
        self.sock.settimeout(0.1)
        logger.info(f"[VRServer] listening on {host or '0.0.0.0'}:{self.sock.getsockname()[1]}")
        self.listener_thread = threading.Thread(target=self.udp_listener, daemon=True)
        self.listener_thread.start()

    def udp_listener(self):
        while not self._closed.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
                content = json.loads(data.decode("utf-8").replace("False", "false").replace("True", "true"))
                with self._lock:
                    self._sample = (content, time.monotonic())
            except socket.timeout:
                continue
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            except OSError:
                if not self._closed.is_set():
                    logger.warning("Pico UDP socket closed unexpectedly")
                return

    def on_update(self):
        with self._lock:
            content, received_at = self._sample
        return content if time.monotonic() - received_at < self.input_timeout_s else None

    def close(self):
        self._closed.set()
        self.sock.close()
        self.listener_thread.join(timeout=0.5)
