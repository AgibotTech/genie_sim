# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Private local transport for render-only state and RGB buffers."""

import pickle
import struct

MAX_MESSAGE = 64 * 1024 * 1024


def send_message(connection, value):
    payload = pickle.dumps(value, protocol=4)
    if len(payload) > MAX_MESSAGE:
        raise ValueError("Renderer message too large")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def receive_exact(connection, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = connection.recv(size - len(chunks))
        if not part:
            raise EOFError("Renderer connection closed")
        chunks.extend(part)
    return bytes(chunks)


def receive_message(connection):
    size = struct.unpack("!I", receive_exact(connection, 4))[0]
    if size > MAX_MESSAGE:
        raise ValueError("Renderer message too large")
    return pickle.loads(receive_exact(connection, size))
