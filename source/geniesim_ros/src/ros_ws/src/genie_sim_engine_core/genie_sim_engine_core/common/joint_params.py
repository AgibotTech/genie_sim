# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Runtime Channel B: sparse, name-keyed values override class defaults.

Callers resolve asset/package paths. This module knows neither robot packages
nor backend runtimes; unit conversion belongs at each solver boundary.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from genie_sim_engine_core.schema.joint_params import OverlayStack, load_joint_overlays


class JointParamResolver:
    def __init__(self, stack: OverlayStack | None = None, logger: Any = None) -> None:
        self.stack = stack or OverlayStack()
        self.logger = logger
        self._values = {
            name: {field: r.value for field, r in self.stack.resolve_joint(name).items() if r.value is not None}
            for name in self.stack.joint_names()
        }

    @classmethod
    def load(cls, paths: Sequence[Any], *, engine: str, robot: Mapping[str, Any], logger: Any = None):
        stack = load_joint_overlays(paths, logger).for_engine(engine)
        for overlay in stack.overlays:
            for key, expected in overlay.applies_to.items():
                actual = robot.get(key)
                if actual != expected:
                    raise ValueError(
                        f"[joint_params] {overlay.id}: applies_to.{key}={expected!r}, robot has {actual!r}"
                    )
        for name in stack.joint_names():
            for field, resolution in stack.resolve_joint(name).items():
                if resolution.needs_engine:
                    raise ValueError(f"[joint_params] {name}.{field}: no {engine} branch")
        resolver = cls(stack, logger)
        if logger is not None:
            for name in stack.joint_names():
                for field, resolution in stack.resolve_joint(name).items():
                    if resolution.unconsumed:
                        logger.warn(f"[joint_params] {name}.{field}={resolution.value}: not implemented at runtime")
                    else:
                        logger.info(
                            f"[joint_params] {name}.{field}={resolution.value} source={resolution.source} engine={engine}"
                        )
        return resolver

    def __bool__(self) -> bool:
        return bool(self.stack)

    def joint(self, name: str) -> Mapping[str, float]:
        return self._values.get(name, {})

    def validate_names(self, names: Iterable[str]) -> None:
        missing = set(self._values) - set(names)
        if missing:
            raise ValueError(f"[joint_params] overlay joints absent from robot: {sorted(missing)}")

    def drive(self, name: str, kp: float, kd: float, effort: float) -> tuple[float, float, float]:
        values = self.joint(name)
        return (
            values.get("stiffness", kp),
            values.get("damping_drive", kd),
            values.get("max_effort", values.get("effort", effort)),
        )
