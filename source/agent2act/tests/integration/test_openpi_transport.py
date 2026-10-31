# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Real local OpenPI framing using the pinned upstream ndarray codec."""

import threading

import numpy as np
import pytest
from openpi_client import msgpack_numpy
from websockets.sync.server import serve

from agent2act.adapters.transports.openpi import OpenPiTransport

pytestmark = pytest.mark.integration


def test_openpi_metadata_and_ndarray_round_trip():
    received = []

    def handle(connection):
        connection.send(msgpack_numpy.Packer().pack({"checkpoint": "local-test"}))
        received.append(msgpack_numpy.unpackb(connection.recv(timeout=2)))
        connection.send(msgpack_numpy.Packer().pack({"actions": np.array([[0.01, 0, 0.5, 0.5]])}))

    with serve(handle, "127.0.0.1", 0) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            transport = OpenPiTransport("127.0.0.1", server.socket.getsockname()[1], timeout_s=2)
            try:
                assert transport.metadata == {"checkpoint": "local-test"}
                result = transport.infer({"state": np.zeros(4, dtype=np.float32), "prompt": "local goal"})
                assert isinstance(result["actions"], np.ndarray)
                assert received[0]["state"].dtype == np.float32
                assert received[0]["prompt"] == "local goal"
            finally:
                transport.close()
        finally:
            server.shutdown()
            worker.join(timeout=3)


def test_openpi_metadata_wait_is_bounded():
    release = threading.Event()

    def handle(connection):
        release.wait(timeout=2)

    with serve(handle, "127.0.0.1", 0) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with pytest.raises(TimeoutError):
                OpenPiTransport("127.0.0.1", server.socket.getsockname()[1], timeout_s=0.1)
        finally:
            release.set()
            server.shutdown()
            worker.join(timeout=3)
