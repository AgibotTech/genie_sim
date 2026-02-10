#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

class TeleopDevice(object):
    """
    Teleop device
    Input: device input
    Output: xyz, rpy in body center
    """
    def __init__(self):
        self.output = {"left": [0,0,0,0,0,0], "right": [0,0,0,0,0,0]}

    def initialize(self):
        pass

    def update(self):
        pass


    def reset(self):
        pass

    def get_output(self):
        return self.output
