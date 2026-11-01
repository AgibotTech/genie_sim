# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from pathlib import Path

from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)

import yaml


class Kinematics_Solver:
    def __init__(self, robot_description_path, urdf_path, end_effector_name, articulation):
        cfg_directory = Path(__file__).resolve().parents[1] / "robot_cfg"
        description_path = cfg_directory / robot_description_path.lstrip("/")
        model_path = cfg_directory / urdf_path.lstrip("/")
        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=str(description_path),
            urdf_path=str(model_path),
        )
        self.robot_description_path = description_path
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            articulation, self._kinematics_solver, end_effector_name
        )

    def update(self, locked_joints):
        with self.robot_description_path.open("r") as file:
            robot_description = yaml.safe_load(file)
