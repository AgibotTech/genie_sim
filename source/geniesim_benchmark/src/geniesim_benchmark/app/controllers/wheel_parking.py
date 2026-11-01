# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0

"""Run an asset's wheel ScriptNode filter for direct benchmark actions.

The asset owns the algorithm, parameters and physics-step subscription. This
adapter supplies the ScriptNode input/output contract without requiring ROS or
OmniScripting execution. It never edits the asset's USD or copies its tuning.
"""

from copy import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class AssetWheelParking:
    def __init__(self, articulation, script_path):
        self.articulation = articulation
        spec = importlib.util.spec_from_file_location("_asset_wheel_parking", script_path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.db = SimpleNamespace(
            per_instance_state=SimpleNamespace(),
            inputs=SimpleNamespace(robotPath=articulation.prim_path),
            outputs=SimpleNamespace(),
            log_error=self._error,
        )
        self.module.setup(self.db)
        self.db.per_instance_state.robot = articulation

    @staticmethod
    def _error(message):
        # A bad asset configuration must not silently bypass wheel limits.
        raise RuntimeError(message)

    @classmethod
    def from_asset(cls, articulation, stage):
        prim = stage.GetPrimAtPath(articulation.prim_path)
        if not prim or prim.GetAttribute("wheelParking:enabled").Get() is not True:
            return None
        script = prim.GetAttribute("wheelParking:filterScript").Get()
        path = script.resolvedPath if script else ""
        if not path or not Path(path).is_file():
            raise RuntimeError(f"WheelParking: unresolved filterScript on {articulation.prim_path}")
        graph = prim.GetAttribute("wheelParking:graphPath").Get()
        if graph and stage.GetPrimAtPath(graph + "/WheelParkingFilter"):
            raise RuntimeError("WheelParking: ROS graph already owns the wheel filter; use one command source")
        return cls(articulation, path)

    def filter_action(self, action):
        indices = action.joint_indices
        self.db.inputs.jointNames = (
            list(self.articulation.dof_names)
            if indices is None
            else [self.articulation.dof_names[int(i)] for i in indices]
        )
        self.db.inputs.positionCommand = [] if action.joint_positions is None else action.joint_positions
        self.db.inputs.velocityCommand = [] if action.joint_velocities is None else action.joint_velocities
        if not self.module.compute(self.db):
            raise RuntimeError("WheelParking: asset filter failed")
        result = copy(action)
        positions = np.asarray(self.db.outputs.positionCommand)
        velocities = np.asarray(self.db.outputs.velocityCommand)
        result.joint_positions = positions if positions.size else None
        result.joint_velocities = velocities if velocities.size else None
        # Efforts, indices and any other action fields are preserved.
        return result

    def close(self):
        self.module.cleanup(self.db)
