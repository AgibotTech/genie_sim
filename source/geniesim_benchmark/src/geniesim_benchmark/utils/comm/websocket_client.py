# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import inspect
import logging
import time
from typing import Any, Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client
from geniesim_benchmark.utils.comm.retry import (
    TRANSIENT_EXC_TYPES,
    backoff_delay,
)
from geniesim_benchmark.utils.msgpack_numpy import *


def ws_connect_compat(uri: str, **kwargs: Any) -> "websockets.sync.client.ClientConnection":
    """Call websockets.sync.client.connect, dropping kwargs the installed
    version does not support (e.g. ping_interval/ping_timeout on older
    websockets releases)."""
    try:
        params = inspect.signature(websockets.sync.client.connect).parameters
    except (TypeError, ValueError):
        params = {}
    if params and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return websockets.sync.client.connect(uri, **kwargs)


# Timeouts tuned for an inference server that may briefly stall under
# concurrent load (multi-env scenarios). ping_timeout in particular needs to
# tolerate a single inference taking longer than the default 20s.
_OPEN_TIMEOUT_SEC = 30
_PING_INTERVAL_SEC = 20
_PING_TIMEOUT_SEC = 60


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws = None
        self._server_metadata = None

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def reset_connection(self) -> None:
        """Drop the current connection so the next infer() reconnects."""
        self.close()

    def __del__(self):
        self.close()

    def _connect(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """Make a single connect attempt. Transient failures are raised so the
        retry budget owned by the caller (run_with_inference_retry) can count
        them — we don't loop here, otherwise an outage would block forever
        inside one infer() call and bypass the budget."""
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        conn = ws_connect_compat(
            self._uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            open_timeout=_OPEN_TIMEOUT_SEC,
            ping_interval=_PING_INTERVAL_SEC,
            ping_timeout=_PING_TIMEOUT_SEC,
        )
        metadata = unpackb(conn.recv())
        return conn, metadata

    def send_recv_bytes(self, data: bytes) -> bytes:
        """Lazy-connect, send one frame, return the response bytes.

        Pulled out of infer() so callers that share this client across threads
        can pack/unpack outside their critical section
        and only serialise the IO. On any IO error the dead socket is dropped
        so the next call reconnects.
        """
        if self._ws is None:
            self._ws, self._server_metadata = self._connect()
        try:
            self._ws.send(data)
            response = self._ws.recv()
        except Exception:
            # Connection is likely dead (keepalive ping timeout, server reset,
            # etc.). Drop it so the next call reconnects instead of reusing a
            # half-closed socket.
            self.close()
            raise
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return response

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        response = self.send_recv_bytes(data)
        return unpackb(response)

    @override
    def reset(self) -> None:
        pass
