# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Robot-independent operating pace within a robot's declared safety limits."""

from dataclasses import dataclass
from typing import Literal

from .models import ContractError, number


@dataclass(frozen=True)
class MotionTiming:
    speed_fraction: float = 1.0
    min_duration_s: float = 0.0
    min_gripper_duration_s: float = 0.0
    target_hold_s: float = 0.0
    profile: Literal["quintic", "smooth_cruise"] = "quintic"
    default_steps: int = 10

    def __post_init__(self) -> None:
        if type(self.default_steps) is not int or self.default_steps < 1:
            raise ContractError("motion.default_steps must be a positive integer")
        if self.profile not in ("quintic", "smooth_cruise"):
            raise ContractError("motion.profile must be quintic or smooth_cruise")
        if not 0 < number(self.speed_fraction, "motion.speed_fraction") <= 1:
            raise ContractError("motion.speed_fraction must be in (0,1]")
        for name in ("min_duration_s", "min_gripper_duration_s", "target_hold_s"):
            if not 0 <= number(getattr(self, name), f"motion.{name}") <= 60:
                raise ContractError(f"motion.{name} must be in [0,60]")
