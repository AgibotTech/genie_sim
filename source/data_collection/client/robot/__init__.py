# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from abc import abstractmethod


class Robot:
    @abstractmethod
    def get_prim_world_pose(self, prim_path):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def open_gripper(self, id="left", width=0.1):
        pass

    @abstractmethod
    def close_gripper(self, id="left", force=50):
        pass

    @abstractmethod
    def move(self, target, type="pose", speed=None):
        pass

    @abstractmethod
    def get_ee_pose(self, id="right"):
        pass
