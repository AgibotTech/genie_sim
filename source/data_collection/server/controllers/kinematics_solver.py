# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os

import yaml
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)


class KinematicsSolver:
    def __init__(self, robot_description_path, urdf_path, end_effector_name, articulation):
        current_directory = (
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/config/robot_cfg"
        )
        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=current_directory + robot_description_path,
            urdf_path=current_directory + urdf_path,
        )
        self.robot_description_path = current_directory + robot_description_path
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            articulation, self._kinematics_solver, end_effector_name
        )

    def update(self, locked_joints):
        with open(self.robot_description_path, "r") as file:
            yaml.safe_load(file)
