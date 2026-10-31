# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Explicit typed component registration. Duplicate names cannot silently replace plugins."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar

from .adapters.simulators.codec import SimulatorCodec
from .domain.errors import ConfigurationError
from .domain.models import RobotSpec
from .domain.ports import ActionExecutor, Kinematics, Planner, PolicyBackend
from .domain.timing import MotionTiming
from .service.observer import Observer

T = TypeVar("T")
Options = dict[str, Any]
PlannerFactory = Callable[[Options], Planner]
PolicyFactory = Callable[[RobotSpec, Options], PolicyBackend]
KinematicsFactory = Callable[[RobotSpec, Options], Kinematics]
MotionFactory = Callable[[RobotSpec, MotionTiming, Options], ActionExecutor]
CodecFactory = Callable[[RobotSpec, Options], SimulatorCodec]
ObserverFactory = Callable[[Path, int], Observer]


class Registry(Generic[T]):
    def __init__(self, category: str) -> None:
        self.category = category
        self._factories: dict[str, T] = {}

    def register(self, name: str, factory: T) -> None:
        if not name or name in self._factories:
            raise ConfigurationError(f"duplicate or empty {self.category} registration: {name!r}")
        self._factories[name] = factory

    def resolve(self, name: str) -> T:
        try:
            return self._factories[name]
        except KeyError as exc:
            raise ConfigurationError(f"unknown {self.category}: {name}") from exc

    def items(self) -> list[tuple[str, T]]:
        return list(self._factories.items())


class ComponentRegistry:
    def __init__(self) -> None:
        self.planners: Registry[PlannerFactory] = Registry("planner")
        self.policies: Registry[PolicyFactory] = Registry("policy")
        self.kinematics: Registry[KinematicsFactory] = Registry("kinematics")
        self.motions: Registry[MotionFactory] = Registry("motion")
        self.codecs: Registry[CodecFactory] = Registry("simulator codec")
        self.observers: Registry[ObserverFactory] = Registry("observer")
