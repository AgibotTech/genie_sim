# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""ZMQ comm backend (cross-machine): same verbs + contract, tcp carriage,
optional JPEG wire encoding for images. Requires pyzmq (lazy: importing this
package registers the classes; zmq itself imports on first use)."""
from . import client  # noqa: F401
