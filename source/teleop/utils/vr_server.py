# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import socket
import json
import threading

from utils.logger import logger
import socket


class VRServer:
    def __init__(self, host=None, port=8080):
        self.data = None
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        listener_thread = threading.Thread(target=self.udp_listener)
        listener_thread.daemon = True
        listener_thread.start()
        self.counter = 0

    def udp_listener(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
                message = data.decode("utf-8")
                _new_message = message.replace("False", "false")
                json_data = json.loads(_new_message)
                self.data = json_data
            except json.JSONDecodeError:
                logger.info(f"Receive {addr} NON-JSON data: {_new_message}")

    def on_update(self):
        self.counter += 1
        if self.data is not None:
            return self.data
        # else:
        #     logger.info("No data received")
        return None


if __name__ == "__main__":
    vr_server = VRServer(host="", port=8080)
    while True:
        vr_server.on_update()
