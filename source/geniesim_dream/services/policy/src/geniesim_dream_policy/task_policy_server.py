"""Keep websocket heartbeats alive while a task's checkpoint loads off-loop."""

from __future__ import annotations

import asyncio
import logging
import time
import traceback

import websockets
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi_client import msgpack_numpy


class TaskPolicyServer(WebsocketPolicyServer):
    async def _handler(self, websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                observation = msgpack_numpy.unpackb(await websocket.recv())
                started = time.monotonic()
                # TaskPolicy serializes all model work. Loading 7GB of weights
                # here synchronously would block the websocket's ping/pong loop.
                result = await asyncio.to_thread(self._policy.infer, observation)
                result["server_timing"] = {"infer_ms": (time.monotonic() - started) * 1000}
                await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                return
            except Exception:
                logging.exception("pi05 task inference failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="pi05 task inference failed")
                return
