# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import time
from typing import Dict, Optional, Tuple
import requests
from geniesim.plugins.logger import logger
import json_numpy

json_numpy.patch()


class HttpClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8999,
        endpoint: str = "/act",
        headers: Optional[Dict] = None,
    ) -> None:
        self.server_url = f"http://{host}:{port}{endpoint}"
        if headers is None:
            self.headers = {"Content-Type": "application/json"}
        self.proxies = {"http": None, "httpshttps": None}

    def infer(self, payload: Dict) -> Dict:  # noqa: UP006
        response = requests.post(self.server_url, json=payload, headers=self.headers)
        return response

    def reset(self) -> None:
        pass
