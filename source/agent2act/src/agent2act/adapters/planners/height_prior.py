# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Explicit static height calibration; never infer object poses or grasp heights."""

import math
from typing import Any

from agent2act.domain.errors import ContractError


class HeightPrior:
    def __init__(self, calibration: dict[str, Any] | None) -> None:
        keys = {"frame", "table_surface_world_z_m", "base_link_world_z_m", "source"}
        if not isinstance(calibration, dict) or set(calibration) != keys:
            raise ContractError(
                "height_calibration requires frame, table_surface_world_z_m, base_link_world_z_m, source"
            )
        if calibration["frame"] != "base_link":
            raise ContractError("height_calibration currently requires an upright base_link frame")
        if not isinstance(calibration["source"], str) or not calibration["source"].strip():
            raise ContractError("height_calibration requires a nonempty calibration source")
        for key in ("table_surface_world_z_m", "base_link_world_z_m"):
            value = calibration[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ContractError(f"height_calibration {key} must be a finite number")
        self.calibration = dict(calibration)

    def context(self, frame: str | None, axes: list[str] | None) -> dict[str, Any]:
        if frame != "base_link" or axes != ["forward", "left", "up"]:
            raise ContractError("height prior requires base_link with forward/left/up axes")
        return {
            "frame": self.calibration["frame"],
            "table_surface_z_m": self.calibration["table_surface_world_z_m"]
            - self.calibration["base_link_world_z_m"],
            "meaning": (
                "Static calibrated table surface for this scene and fixed upright base; "
                "height_above_table_m = pose.z - table_surface_z_m. It is the table "
                "surface, not the object top or a contact measurement."
            ),
        }
