# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Retime policy waypoints into speed-bounded position-actuator references."""

import mujoco
import numpy as np


class NativePositionController:
    def __init__(self, model, data, groups, hand_profile, velocity_limits, action_time_scale):
        self.model, self.data = model, data
        for name in ("armq", "handq", "armj", "handj", "armv", "actq"):
            setattr(self, name, getattr(groups, name))
        self.hand_profile = hand_profile
        self.asset_velocity_limits = np.asarray(velocity_limits, dtype=float)
        # Reuse only the original expert's slow reference rates, not its bias
        # compensation or pose filtering. Hardware limits remain separate.
        arms = np.isin(self.actq, np.r_[self.armq["l"], self.armq["r"]])
        self.velocity_limits = np.minimum(self.asset_velocity_limits, np.where(arms, groups.arm_rate, 2.5))
        self.action_time_scale = action_time_scale
        if self.velocity_limits.shape != self.actq.shape or not np.all(
            np.isfinite(self.velocity_limits) & (self.velocity_limits > 0)
        ):
            raise ValueError("Each native motor needs a positive finite velocity limit")
        self.command = data.qpos[self.actq].copy()
        self.start = self.command.copy()
        self.goal = self.command.copy()
        self.segment_steps = 1
        self.segment_step = 0
        self.retimed_actions = 0
        self.extra_ticks = 0
        self.maximum_command_speeds = np.zeros_like(self.command)

    def set_target(self, target, minimum_steps):
        self.start = self.command.copy()
        self.goal = np.asarray(target, dtype=float).copy()
        # Quintic smoothstep has zero endpoint velocity AND acceleration,
        # with peak ds/dt = 1.875/T. Extend all nominal action intervals too.
        # One common duration preserves arm/hand phase and every waypoint.
        required_seconds = 1.875 * float(np.max(np.abs(self.goal - self.start) / self.velocity_limits))
        self.segment_steps = max(
            int(np.ceil(minimum_steps * self.action_time_scale)),
            int(np.ceil(required_seconds / self.model.opt.timestep)),
            1,
        )
        self.segment_step = 0
        if self.segment_steps > minimum_steps:
            self.retimed_actions += 1
            self.extra_ticks += self.segment_steps - minimum_steps
        return self.segment_steps

    def step(self, target):
        previous = self.command.copy()
        self.segment_step = min(self.segment_step + 1, self.segment_steps)
        phase = self.segment_step / self.segment_steps
        blend = phase**3 * (10 + phase * (-15 + 6 * phase))
        self.command = self.start + blend * (self.goal - self.start)
        self.maximum_command_speeds = np.maximum(
            self.maximum_command_speeds, np.abs(self.command - previous) / self.model.opt.timestep
        )
        # Only actuator references change. Original PD gains, force limits and
        # solver integration remain intact; no qpos/qvel edits or extra forces.
        self.data.ctrl[:] = self.command
        mujoco.mj_step(self.model, self.data)

    def diagnostics(self):
        return {
            "interpolation": "synchronized quintic smoothstep at every physics substep",
            "action_time_scale": self.action_time_scale,
            "retimed_actions": self.retimed_actions,
            "extra_time_is_planned": True,
            "extra_physics_ticks": self.extra_ticks,
            "extra_simulation_seconds": self.extra_ticks * self.model.opt.timestep,
            "motor_joint_names": [self.model.joint(j).name for j in self.model.actuator_trnid[:, 0]],
            "motor_velocity_limits_radians_per_second": self.asset_velocity_limits.tolist(),
            "command_speed_limits_radians_per_second": self.velocity_limits.tolist(),
            "maximum_command_speeds_radians_per_second": self.maximum_command_speeds.tolist(),
        }
