# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Explicit static gripper geometry calibration; never a collision or contact guarantee."""

import math
from typing import Any

from agent2act.domain.errors import ContractError


class GraspGeometry:
    """Nominal finger-pad geometry relative to the TCP reference point.

    Static mesh-derived bounds along the TCP approach axis. Not a model of passive joints, aperture
    or collision bodies, and never a guarantee of collision-free motion, contact or grasp success."""

    def __init__(self, calibration: dict[str, Any] | None) -> None:
        top_keys = {
            "frame",
            "fingertip_beyond_tcp_m",
            "pad_contact_zone_m",
            "pad_width_m",
            "pad_thickness_m",
            "geometry_basis",
            "source",
        }
        optional_keys = {"pad_widths_per_finger_m", "gripper_notes", "tcp_axes", "aperture_by_command"}
        if not isinstance(calibration, dict) or not top_keys <= set(calibration) <= top_keys | optional_keys:
            raise ContractError(
                "grasp_geometry_calibration requires frame, fingertip_beyond_tcp_m, "
                "pad_contact_zone_m, pad_width_m, pad_thickness_m, geometry_basis, source"
            )
        if calibration["frame"] != "tcp_approach_axis":
            raise ContractError("grasp_geometry_calibration requires the tcp_approach_axis frame")
        if not isinstance(calibration["geometry_basis"], str) or not calibration["geometry_basis"].strip():
            raise ContractError("grasp_geometry_calibration requires a nonempty geometry_basis")
        if not isinstance(calibration["source"], str) or not calibration["source"].strip():
            raise ContractError("grasp_geometry_calibration requires a nonempty source")
        # A TCP may sit exactly at the fingertips, so reach beyond it can be zero; sizes cannot.
        for key, minimum in (("fingertip_beyond_tcp_m", 0), ("pad_width_m", None), ("pad_thickness_m", None)):
            value = calibration[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value < 0 if minimum == 0 else value <= 0)
            ):
                kind = "non-negative" if minimum == 0 else "positive"
                raise ContractError(f"grasp_geometry_calibration {key} must be a {kind} finite number")
        zone = calibration["pad_contact_zone_m"]
        if not isinstance(zone, dict) or set(zone) != {"behind_tcp_m", "beyond_tcp_m"}:
            raise ContractError("pad_contact_zone_m requires behind_tcp_m and beyond_tcp_m")
        for key, value in zone.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value < 0 if key == "beyond_tcp_m" else value <= 0)
            ):
                kind = "non-negative" if key == "beyond_tcp_m" else "positive"
                raise ContractError(f"pad_contact_zone_m {key} must be a {kind} finite number")
        if zone["beyond_tcp_m"] > calibration["fingertip_beyond_tcp_m"] + 1e-9:
            raise ContractError("pad_contact_zone beyond_tcp_m cannot exceed fingertip_beyond_tcp_m")
        per_finger = calibration.get("pad_widths_per_finger_m")
        if per_finger is not None:
            if not isinstance(per_finger, dict) or not per_finger:
                raise ContractError("pad_widths_per_finger_m must be a nonempty name-to-width map")
            for finger, width in per_finger.items():
                if not isinstance(finger, str) or not finger.strip():
                    raise ContractError("pad_widths_per_finger_m keys must be nonempty finger names")
                if (
                    isinstance(width, bool)
                    or not isinstance(width, (int, float))
                    or not math.isfinite(width)
                    or width <= 0
                ):
                    raise ContractError("pad_widths_per_finger_m values must be positive finite numbers")
            calibration = dict(calibration)
            calibration["pad_widths_per_finger_m"] = dict(per_finger)
        notes = calibration.get("gripper_notes")
        if notes is not None and (not isinstance(notes, str) or not notes.strip()):
            raise ContractError("gripper_notes must be a nonempty string")
        axes = calibration.get("tcp_axes")
        if axes is not None:
            if not isinstance(axes, dict) or set(axes) != {"approach", "open_close", "pad_width"}:
                raise ContractError("tcp_axes requires approach, open_close and pad_width")
            if sorted(v.lstrip("+-") for v in axes.values() if isinstance(v, str)) != ["x", "y", "z"]:
                raise ContractError("tcp_axes must name each TCP axis x, y, z exactly once")
        aperture = calibration.get("aperture_by_command")
        if aperture is not None:
            if not isinstance(aperture, list) or len(aperture) < 2:
                raise ContractError("aperture_by_command must list at least two command rows")
            for row in aperture:
                if not isinstance(row, dict) or set(row) != {
                    "command",
                    "gap_m",
                    "pad_face_from_tcp_m",
                    "fingertip_beyond_tcp_m",
                }:
                    raise ContractError(
                        "aperture_by_command rows require command, gap_m, pad_face_from_tcp_m, "
                        "fingertip_beyond_tcp_m"
                    )
                values = [
                    row["command"],
                    row["gap_m"],
                    row["fingertip_beyond_tcp_m"],
                    *row["pad_face_from_tcp_m"],
                ]
                if len(row["pad_face_from_tcp_m"]) != 2 or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in values
                ):
                    raise ContractError("aperture_by_command values must be finite numbers")
        self.calibration = dict(calibration)
        self.calibration["pad_contact_zone_m"] = dict(zone)

    def context(self, table_surface_z_m: float | None = None) -> dict[str, Any]:
        if table_surface_z_m is not None and (
            isinstance(table_surface_z_m, bool)
            or not isinstance(table_surface_z_m, (int, float))
            or not math.isfinite(table_surface_z_m)
        ):
            raise ContractError("grasp geometry requires a finite calibrated table height")
        calibration = self.calibration
        block = {
            "fingertip_beyond_tcp_m": calibration["fingertip_beyond_tcp_m"],
            "pad_contact_zone_m": calibration["pad_contact_zone_m"],
            "pad_width_m": calibration["pad_width_m"],
            "pad_thickness_m": calibration["pad_thickness_m"],
            "meaning": (
                "Nominal mesh-derived finger geometry relative to the reported TCP along "
                "its approach axis; not measured contact, collision shapes or runtime aperture."
            ),
        }
        if table_surface_z_m is not None:
            block["table_surface_z_m"] = table_surface_z_m
        per_finger = calibration.get("pad_widths_per_finger_m")
        if per_finger is not None:
            block["pad_widths_per_finger_m"] = per_finger
        for key in ("tcp_axes", "aperture_by_command", "gripper_notes"):
            if key in calibration:
                block[key] = calibration[key]
        return block
