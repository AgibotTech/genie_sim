# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pico analog axes to a body-frame velocity command (ROS: x forward, y left)."""

from dataclasses import dataclass, fields
import math

import yaml


@dataclass(frozen=True)
class ChassisControl:
    max_linear_speed: float = 0.4  # m/s
    max_angular_speed: float = 0.5  # rad/s
    axis_deadzone: float = 0.05
    input_timeout_s: float = 0.25

    def __post_init__(self):
        if not all(
            isinstance(getattr(self, field.name), (int, float)) and math.isfinite(getattr(self, field.name))
            for field in fields(self)
        ):
            raise ValueError("teleop.chassis values must be finite")
        if self.max_linear_speed <= 0 or self.max_angular_speed <= 0 or self.input_timeout_s <= 0:
            raise ValueError("teleop.chassis speeds and input_timeout_s must be positive")
        if not 0 <= self.axis_deadzone < 1:
            raise ValueError("teleop.chassis.axis_deadzone must be in [0, 1)")

    @classmethod
    def from_task(cls, path):
        settings = yaml.safe_load(path.read_text()).get("teleop", {}).get("chassis", {})
        unknown = set(settings) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown teleop.chassis settings: {sorted(unknown)}")
        return cls(**settings)

    def velocity(self, inputs):
        def axis(name):
            value = float(inputs.get(name, 0.0))
            if not math.isfinite(value):
                raise ValueError("Pico axes must be finite")
            value = max(-1.0, min(1.0, value))
            return 0.0 if abs(value) <= self.axis_deadzone else value

        try:
            x, y, yaw = axis("l_axisY"), -axis("l_axisX"), -axis("r_axisX")
        except (TypeError, ValueError):
            return (0.0, 0.0, 0.0)
        norm = max(1.0, math.hypot(x, y))
        return (self.max_linear_speed * x / norm, self.max_linear_speed * y / norm, self.max_angular_speed * yaw)
