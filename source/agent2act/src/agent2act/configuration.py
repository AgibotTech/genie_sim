# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Validated deployment settings and JSON/TOML loading.

Plugin option dictionaries are extension boundaries validated by the registered factory;
application settings reject extras."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .adapters.embodiments import (
    HARDWARE_FIELDS,
    bind_reference,
    planner_hardware,
)
from .application.settings import ExecutionConfig
from .domain.errors import ConfigurationError
from .domain.models import RobotSpec
from .domain.timing import MotionTiming

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class PluginSettings(BaseModel):
    """Preserves the existing flat {type, ...options} configuration format."""

    model_config = ConfigDict(extra="allow", frozen=True)
    type: Annotated[str, Field(min_length=1, strict=True)]

    @property
    def options(self) -> dict[str, Any]:
        return self.model_dump(exclude={"type"})


class PolicySettings(SettingsModel):
    default: str
    backends: dict[str, PluginSettings]

    @model_validator(mode="after")
    def check_default(self) -> PolicySettings:
        if self.default not in self.backends:
            raise ValueError("default policy must name a configured backend")
        return self


class ServerSettings(SettingsModel):
    host: Annotated[str, Field(min_length=1, strict=True)] = "127.0.0.1"
    port: Annotated[int, Field(ge=0, le=65535, strict=True)] = 8999
    max_message_bytes: Annotated[int, Field(gt=0, le=128 * 1024 * 1024, strict=True)] = 33554432
    max_connections: Annotated[int, Field(ge=1, le=1024, strict=True)] = 16


class LoggingSettings(SettingsModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"
    trace_dir: str | None = None


class MotionSettings(SettingsModel):
    """Deployment pace settings; interpolation steps come from the execution budget."""

    speed_fraction: Annotated[float, Field(strict=True)] = 1.0
    min_duration_s: Annotated[float, Field(strict=True)] = 0.0
    min_gripper_duration_s: Annotated[float, Field(strict=True)] = 0.0
    target_hold_s: Annotated[float, Field(strict=True)] = 0.0
    profile: Literal["quintic", "smooth_cruise"] = "quintic"

    def to_timing(self, execute_steps: int) -> MotionTiming:
        return MotionTiming(**self.model_dump(), default_steps=execute_steps)

    @model_validator(mode="after")
    def validate_timing(self) -> MotionSettings:
        self.to_timing(1)
        return self


class AppSettings(SettingsModel):
    version: Literal[1]
    robot: RobotSpec
    execution: ExecutionConfig
    planner: PluginSettings
    embodiment: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_hardware(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, dict) or value.get("embodiment") is None:
            return value
        data = deepcopy(value)
        reference = data["embodiment"]
        binding = bind_reference(reference, (info.context or {}).get("config_dir"))
        definition = binding["definition"]
        planner = data.get("planner")
        if not isinstance(planner, dict):
            raise ValueError("embodiment requires a planner configuration")
        robot = data.get("robot", {})
        robot_type = (
            robot.get("robot_type") if isinstance(robot, dict) else getattr(robot, "robot_type", None)
        )
        if definition["robot_type"] != robot_type:
            raise ValueError(
                f"embodiment {definition['name']} describes robot_type {definition['robot_type']}, "
                f"but configuration serves {robot_type}; use that robot's configuration"
            )
        enabled = planner.get("use_gripper_center")
        if type(enabled) is not bool:
            raise ValueError("embodiment requires explicit use_gripper_center=true/false")
        hardware = planner_hardware(definition, enabled)
        if planner.get("use_grasp_geometry", False) and (
            enabled is False
            or "gripper_center_offset_m" not in hardware
            or "grasp_geometry_calibration" not in hardware
        ):
            raise ValueError(
                "use_grasp_geometry requires TCP and calibrated geometry in the selected embodiment"
            )
        hardware["embodiment"] = definition["name"]
        owned = HARDWARE_FIELDS | {"embodiment"}
        repeated = owned & planner.keys()
        frozen = isinstance(reference, dict) and "definition" in reference
        if repeated and not frozen:
            raise ValueError(
                f"embodiment reference conflicts with inline planner hardware: {sorted(repeated)}"
            )
        if frozen and any(key not in hardware or planner[key] != hardware[key] for key in repeated):
            raise ValueError(
                "expanded planner hardware differs from frozen embodiment; change the preset, not individual fields"
            )
        planner.update(hardware)
        bounds = (
            robot.get("gripper_bounds", [])
            if isinstance(robot, dict)
            else getattr(robot, "gripper_bounds", [])
        )
        commands = hardware.get("gripper_open_closed", {})
        if len(bounds) != 2:
            raise ValueError("embodiment requires left/right robot gripper_bounds")
        for arm, (low, high) in zip(("left", "right"), bounds):
            if any(not low <= command <= high for command in commands.get(arm, [])):
                raise ValueError(
                    f"embodiment {definition['name']} gripper commands exceed robot gripper_bounds"
                )
        data["embodiment"] = binding
        return data

    motion: MotionSettings = Field(default_factory=MotionSettings)
    motion_backend: PluginSettings = Field(default_factory=lambda: PluginSettings(type="local_interpolation"))
    ranked_motion_candidates: bool = Field(default=False, strict=True)
    ranked_motion_candidate_count: Annotated[int, Field(strict=True)] = 5
    policy: PolicySettings | None = None
    kinematics: PluginSettings | None = None
    simulator_codec: PluginSettings = Field(default_factory=lambda: PluginSettings(type="corobot"))
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @field_validator("ranked_motion_candidate_count")
    @classmethod
    def validate_ranked_motion_candidate_count(cls, value: int) -> int:
        if value not in (5, 10):
            raise ValueError("ranked_motion_candidate_count must be 5 or 10")
        return value

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: Any) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("configuration version must be integer 1")
        return value

    @field_validator("robot", mode="before")
    @classmethod
    def validate_robot(cls, value: Any) -> RobotSpec:
        if isinstance(value, RobotSpec):
            return value
        if not isinstance(value, dict):
            raise ValueError("robot must be an object")
        try:
            return RobotSpec(**value)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("execution", mode="before")
    @classmethod
    def validate_execution(cls, value: Any) -> ExecutionConfig:
        if isinstance(value, ExecutionConfig):
            return value
        if not isinstance(value, dict):
            raise ValueError("execution must be an object")
        try:
            return ExecutionConfig(**value)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def validate_composition(self) -> AppSettings:
        modes = self.execution.allowed_modes if self.execution.mode == "auto" else [self.execution.mode]
        if self.ranked_motion_candidates and (modes != ["motion"] or self.execution.proposal_review):
            raise ValueError(
                "ranked_motion_candidates requires motion-only execution without proposal review"
            )
        if self.execution.proposal_review and "policy" in modes:
            if self.kinematics is None:
                raise ValueError("proposal review requires kinematics")
            if self.policy is None or self.execution.proposal_backend not in self.policy.backends:
                raise ValueError("proposal backend must be installed")
        if "policy" in modes and self.policy is None:
            raise ValueError("policy mode requires an installed policy backend configuration")
        if "motion" in modes and not self.robot.supports_eef:
            raise ValueError("motion mode requires supports_eef=true")
        if self.execution.execute_steps > self.robot.max_chunk_steps:
            raise ValueError("execute_steps exceeds robot max_chunk_steps")
        if (self.execution.policy_execute_steps or 0) > self.robot.max_chunk_steps:
            raise ValueError("policy_execute_steps exceeds robot max_chunk_steps")
        if (self.execution.body_execute_steps or 0) > self.robot.max_chunk_steps:
            raise ValueError("body_execute_steps exceeds robot max_chunk_steps")
        return self

    def hardware_snapshot(self) -> dict[str, Any] | None:
        """Only hardware and feature choices; never model credentials or endpoint headers."""
        if self.embodiment is None:
            return None
        options = self.planner.options
        return {
            "embodiment": deepcopy(self.embodiment),
            "planner_hardware": {key: deepcopy(options[key]) for key in HARDWARE_FIELDS if key in options},
            "features": {key: options.get(key) for key in ("use_gripper_center", "use_grasp_geometry")},
            "robot_type": self.robot.robot_type,
            "gripper_bounds": deepcopy(self.robot.gripper_bounds),
            "running_asset_identity_verified": False,
        }

    def override(
        self,
        *,
        mode: str | None = None,
        host: str | None = None,
        port: int | None = None,
        depth: bool | None = None,
        gripper_center: bool | None = None,
        grasp_geometry: bool | None = None,
        height_prior: bool | None = None,
        height_calibration: dict[str, Any] | None = None,
    ) -> AppSettings:
        """CLI overrides are revalidated, not applied with unchecked model_copy()."""
        data = self.model_dump()
        if self.embodiment is not None:
            # Verify the expanded copy before replacing fields for new feature flags.
            parse_settings(data)
            for key in HARDWARE_FIELDS | {"embodiment"}:
                data["planner"].pop(key, None)
        if mode is not None:
            data["execution"]["mode"] = mode
        if host is not None:
            data["server"]["host"] = host
        if port is not None:
            data["server"]["port"] = port
        if depth is not None:
            data["planner"]["use_depth"] = depth
            data["execution"]["need_depth"] = depth
        if gripper_center is not None:
            data["planner"]["use_gripper_center"] = gripper_center
        if grasp_geometry is not None:
            data["planner"]["use_grasp_geometry"] = grasp_geometry
        if height_prior is not None:
            data["planner"]["use_height_prior"] = height_prior
        if height_calibration is not None:
            data["planner"]["height_calibration"] = height_calibration
        return parse_settings(data)


def parse_settings(data: Any, *, base_dir: Path | None = None) -> AppSettings:
    try:
        return AppSettings.model_validate(data, context={"config_dir": base_dir})
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'config'}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise ConfigurationError(details) from exc


def load_config(path: str | Path, *, gripper_center: bool | None = None) -> AppSettings:
    file = Path(path)
    try:
        if file.suffix == ".toml":
            with file.open("rb") as stream:
                data = tomllib.load(stream)
        elif file.suffix == ".json":
            data = json.loads(file.read_text(encoding="utf-8"))
        else:
            raise ConfigurationError("configuration must be a .json or .toml file")
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot load configuration {file}: {exc}") from exc
    if isinstance(data, dict) and isinstance(data.get("planner"), dict):
        if gripper_center is not None:
            if isinstance(data.get("embodiment"), dict) and "definition" in data["embodiment"]:
                # Validate a portable snapshot before regenerating text for an explicit feature override.
                parse_settings(data)
                for key in HARDWARE_FIELDS | {"embodiment"}:
                    data["planner"].pop(key, None)
            data["planner"]["use_gripper_center"] = gripper_center
    return parse_settings(data, base_dir=file.resolve().parent)
