# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Versioned gripper descriptions, independent of tasks, routing and robot kinematics."""

import hashlib
import json
import re
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

from agent2act.adapters.planners.grasp_geometry import GraspGeometry
from agent2act.adapters.planners.tool_center import CENTER_INSTRUCTIONS, ToolCenter
from agent2act.domain.errors import ConfigurationError, ContractError
from agent2act.domain.models import vector

HARDWARE_FIELDS = frozenset(
    {
        "gripper_center_offset_m",
        "gripper_open_closed",
        "pose_reference",
        "grasp_geometry_calibration",
    }
)
DESCRIPTION_VERSION = "gripper-description-v1"


def definition_hash(definition: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(definition, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_definition(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigurationError("embodiment must be an object")
    required = {
        "schema_version",
        "name",
        "revision",
        "source",
        "robot_type",
        "tcp",
        "gripper_open_closed",
        "grasp_geometry_calibration",
    }
    if not required <= set(data) <= required | {"observations"}:
        raise ConfigurationError(
            "embodiment requires name, source and robot_type and its versioned hardware fields only"
        )
    if type(data["schema_version"]) is not int or data["schema_version"] != 2:
        raise ConfigurationError("unsupported embodiment schema_version; expected 2")
    for key in ("name", "source", "robot_type", "revision"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ConfigurationError(f"embodiment {key} must be a nonempty string")
    tcp = data["tcp"]
    if not isinstance(tcp, dict) or set(tcp) != {"axes", "offset_m"} or tcp["axes"] != "wrist_parallel":
        raise ConfigurationError(
            "embodiment tcp requires wrist_parallel axes and offset_m; rotated TCP frames are unsupported"
        )
    geometry = data["grasp_geometry_calibration"]
    if not isinstance(geometry, dict) or "gripper_notes" in geometry:
        raise ConfigurationError(
            "version 2 gripper_notes are generated; use sourced observations for additional facts"
        )
    if "tcp_axes" not in geometry:
        raise ConfigurationError("version 2 geometry requires tcp_axes")
    observations = data.get("observations", [])
    if not isinstance(observations, list) or any(
        not isinstance(row, dict)
        or set(row) != {"text", "source"}
        or any(not isinstance(v, str) or not v.strip() for v in row.values())
        for row in observations
    ):
        raise ConfigurationError("embodiment observations require nonempty text and source")
    try:
        ToolCenter(tcp["offset_m"])
        GraspGeometry(geometry)
        commands = data["gripper_open_closed"]
        if not isinstance(commands, dict) or set(commands) != {"left", "right"}:
            raise ContractError("gripper_open_closed requires left and right")
        for values in commands.values():
            opened, closed = vector(values, 2, "gripper open/closed commands")
            if opened == closed:
                raise ContractError("gripper open and closed commands must differ")
    except (ContractError, TypeError, ValueError, KeyError, AttributeError) as exc:
        raise ConfigurationError(f"invalid embodiment hardware: {exc}") from exc
    return deepcopy(data)


def load_definition(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot load embodiment {path}: {exc}") from exc
    return validate_definition(data)


def bind_reference(reference: Any, base_dir: Path | None = None) -> dict[str, Any]:
    """Resolve once; frozen definitions can subsequently load without the original file."""
    if not isinstance(reference, dict):
        raise ConfigurationError("embodiment reference must be an object")
    if set(reference) == {"preset"}:
        name = reference["preset"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name):
            raise ConfigurationError("invalid embodiment preset name")
        try:
            definition = validate_definition(
                json.loads(
                    files(__package__).joinpath("data").joinpath(name + ".json").read_text(encoding="utf-8")
                )
            )
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"cannot load embodiment preset {name}: {exc}") from exc
        origin = "builtin:" + name
    elif set(reference) == {"path"}:
        if not isinstance(reference["path"], str) or not reference["path"]:
            raise ConfigurationError("embodiment path must be a nonempty string")
        path = Path(reference["path"])
        if not path.is_absolute():
            path = (base_dir or Path.cwd()) / path
        definition, origin = load_definition(path), str(path.resolve())
    elif set(reference) == {"definition", "sha256", "origin", "description_version"}:
        definition = validate_definition(reference["definition"])
        if reference["sha256"] != definition_hash(definition):
            raise ConfigurationError("frozen embodiment sha256 differs from its definition")
        if reference["description_version"] != DESCRIPTION_VERSION:
            raise ConfigurationError(
                "frozen embodiment description version differs; use the matching code version"
            )
        origin = reference["origin"]
        if not isinstance(origin, str) or not origin:
            raise ConfigurationError("frozen embodiment origin must be nonempty")
    else:
        raise ConfigurationError(
            "embodiment requires exactly one preset, path, or a complete frozen definition"
        )
    return freeze_definition(definition, origin)


def freeze_definition(definition: dict[str, Any], origin: str) -> dict[str, Any]:
    definition = validate_definition(definition)
    return {
        "definition": definition,
        "sha256": definition_hash(definition),
        "origin": origin,
        "description_version": DESCRIPTION_VERSION,
    }


def planner_hardware(definition: dict[str, Any], tcp_enabled: bool) -> dict[str, Any]:
    """Generate facts only. Geometry text is sent only through the geometry feature gate."""
    result = {
        "gripper_center_offset_m": deepcopy(definition["tcp"]["offset_m"]),
        "gripper_open_closed": deepcopy(definition["gripper_open_closed"]),
        "grasp_geometry_calibration": deepcopy(definition["grasp_geometry_calibration"]),
    }
    if tcp_enabled:
        offsets = "; ".join(
            f"{arm}: {json.dumps(value)} m in wrist-local coordinates"
            for arm, value in result["gripper_center_offset_m"].items()
        )
        result["pose_reference"] = (
            CENTER_INSTRUCTIONS + " Calibrated wrist-to-TCP translations: " + offsets + "."
        )
    geometry = result["grasp_geometry_calibration"]
    axes = geometry["tcp_axes"]
    facts = [
        f"TCP {axes['approach']} is the approach axis, {axes['open_close']} is the jaw open/close axis, and {axes['pad_width']} is the pad-width axis."
    ]
    widths = geometry.get("pad_widths_per_finger_m")
    if widths:
        facts.append(
            "Nominal pad widths: "
            + "; ".join(f"{name} {width * 1000:g} mm" for name, width in widths.items())
            + "."
        )
    else:
        facts.append(f"Nominal pad width: {geometry['pad_width_m'] * 1000:g} mm.")
    facts.append(
        f"Farthest nominal fingertip extends {geometry['fingertip_beyond_tcp_m'] * 1000:g} mm beyond the TCP along the approach axis."
    )
    for arm, (opened, closed) in result["gripper_open_closed"].items():
        facts.append(
            f"{arm} gripper setpoints: fully open {opened:g}, fully closed {closed:g}; these are commands, not measured contact or retention."
        )
    facts.extend(
        f"Additional observation: {row['text']} Source: {row['source']}"
        for row in definition.get("observations", [])
    )
    facts.append(
        "Use geometry_basis and source to interpret these nominal dimensions; they do not certify contact or collision clearance."
    )
    geometry["gripper_notes"] = " ".join(facts)
    return result
