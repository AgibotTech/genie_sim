# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded OpenPI WebSocket transport using the official ndarray codec."""


class OpenPiTransport:
    """OpenPI metadata handshake and ndarray codec, with bounded network waits.

    Uses the official codec since the ndarray wire format is not plain msgpack. OpenPI has no remote reset RPC."""

    def __init__(self, host, port, timeout_s=30):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        uri = host if host.startswith(("ws://", "wss://")) else "ws://" + host
        if port is not None:
            uri += f":{port}"
        self.codec, self.timeout_s = msgpack_numpy, timeout_s
        self.ws = connect(
            uri, compression=None, max_size=64 * 1024 * 1024, open_timeout=timeout_s, close_timeout=5
        )
        try:
            self.metadata = self.codec.unpackb(self.ws.recv(timeout=timeout_s))
        except Exception:
            self.ws.close()
            raise

    def infer(self, observation):
        self.ws.send(self.codec.Packer().pack(observation))
        response = self.ws.recv(timeout=self.timeout_s)
        if isinstance(response, str):
            raise RuntimeError("OpenPI server returned an inference error")
        return self.codec.unpackb(response)

    def reset(self):
        pass  # OpenPI inference is stateless; stateful servers need a custom backend.

    def close(self):
        self.ws.close()
