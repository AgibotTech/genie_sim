# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Public physics facade backed by isaacsim_native.physics.
Isaac Sim runtime classes are imported inside methods that need them;
call these methods after SimulationApp starts."""

from __future__ import annotations

from sim_backend.backends.isaacsim_native.physics import (
    apply_action,
    get_dof_limits,
    get_dof_names,
    get_joint_positions,
    get_joint_velocities,
    get_joints_state,
    get_measured_joint_efforts,
    get_prim_world_pose,
    initialize_view,
    set_gains,
    set_joint_positions,
    set_joint_velocities,
    set_max_efforts,
    set_rigid_velocity,
)

__all__ = [
    "apply_action",
    "get_dof_limits",
    "get_dof_names",
    "get_joint_positions",
    "get_joint_velocities",
    "get_joints_state",
    "get_measured_joint_efforts",
    "get_prim_world_pose",
    "initialize_view",
    "set_gains",
    "set_joint_positions",
    "set_joint_velocities",
    "set_max_efforts",
    "set_rigid_velocity",
]
