# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Domain-randomization configuration, sampling and plans.

RandomizationSpec resolves parameter domains; Sampler draws independent seeded
streams; RandomizationPlan stores results. Engines sample reset poses, render
bridges apply scene changes, and VecEnv applies sensor image augmentation."""

from __future__ import annotations

from .plan import RandomizationPlan
from .sampler import Sampler, substreams
from .spec import (
    BaseXYChannelConflictError,
    RandomizationSpec,
    check_base_xy_channel_conflict,
    resolve_benchmark_camera_generalization,
    resolve_benchmark_generalization,
)

__all__ = [
    "RandomizationPlan",
    "RandomizationSpec",
    "Sampler",
    "substreams",
    "BaseXYChannelConflictError",
    "check_base_xy_channel_conflict",
    "resolve_benchmark_generalization",
    "resolve_benchmark_camera_generalization",
]
