# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .base import BasePolicy
import threading
import rclpy


class DemoPolicy(BasePolicy):
    def __init__(self, task_name) -> None:
        super().__init__(task_name)
        self.init_ros_node()

    def init_ros_node(self):
        pass

    def act(self, observations, **kwargs):
        return

    def reset(self):
        target_position = []
        return target_position
