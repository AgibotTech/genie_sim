# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os

from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)
from isaacsim.core.utils.extensions import get_extension_path_from_name

import yaml
import os
from pathlib import Path
import sys


class Kinematics_Solver:
    def __init__(self, robot_description_path, urdf_path, end_effector_name, articulation):
        main_path = Path(sys.modules["__main__"].__file__).resolve()
        main_dir = str(main_path.parent)
        cfg_directory = main_dir + "/robot_cfg"
        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=cfg_directory + robot_description_path,
            urdf_path=cfg_directory + urdf_path,
        )
        self.robot_description_path = cfg_directory + robot_description_path
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            articulation, self._kinematics_solver, end_effector_name
        )

    def update(self, locked_joints):
        with open(self.robot_description_path, "r") as file:
            robot_description = yaml.safe_load(file)
