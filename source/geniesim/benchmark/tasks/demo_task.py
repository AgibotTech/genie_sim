# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import copy


from .base_task import BaseTask


class DemoTask(BaseTask):
    def __init__(self, env):
        super().__init__(env)
        self.reward_functions = []

        self.initialize(env)
        self.state_history = {}

    def initialize(self, env):
        pass

    def step(self, env):
        pass
