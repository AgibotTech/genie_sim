# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Physics backend registry + factory.

Every concrete backend file in this package registers itself with
``@register_engine("name")``. The factory ``make_engine`` looks the
name up here and instantiates.

Adding a new backend = drop a new file in this directory, decorate
its class with ``@register_engine("foo")``, and add one line below to
trigger import. Nothing else changes.
"""

from __future__ import annotations

from typing import Any

from .base import (
    BodyKind,
    ClothMeshSpec,
    DynamicsParams,
    HealthStatus,
    SceneSchema,
    VecPhysicsBackend,
)

# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[VecPhysicsBackend]] = {}


def register_engine(name: str):
    """Class decorator: register a concrete backend under ``name``."""

    def deco(cls: type[VecPhysicsBackend]) -> type[VecPhysicsBackend]:
        if name in _REGISTRY:
            raise ValueError(
                f"engine '{name}' already registered " f"(existing={_REGISTRY[name].__name__}, new={cls.__name__})"
            )
        if not issubclass(cls, VecPhysicsBackend):
            raise TypeError(f"{cls.__name__} must subclass VecPhysicsBackend")
        _REGISTRY[name] = cls
        return cls

    return deco


def make_engine(name: str, **cfg: Any) -> VecPhysicsBackend:
    """Instantiate the backend registered under ``name``."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown physics backend '{name}'; " f"valid: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**cfg)


def list_engines() -> list[str]:
    """Return all registered backend names (sorted)."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Trigger registration of every concrete backend module.
# Each ``from . import foo`` runs that module's ``@register_engine`` decorator
# and adds an entry to ``_REGISTRY``.
# ---------------------------------------------------------------------------

# Upstream-subprocess wrappers (require the geniesim_ros colcon workspace
# to be built and sourced — see source/geniesim_ros/AGENTS.md).
from . import newton_mjwarp  # noqa: E402, F401  → "newton_mujoco_warp"
from . import newton_featherstone  # noqa: E402, F401  → "newton_featherstone"
from . import newton_avbd  # noqa: E402, F401  → "newton_avbd"
from . import isaac_physx  # noqa: E402, F401  → "isaac_physx"
from . import isaac_newton  # noqa: E402, F401  → "isaac_newton"
from . import isaac_fused  # noqa: E402, F401  → "isaac_fused" (physics+render, ONE Isaac Sim process)


__all__ = [
    "BodyKind",
    "ClothMeshSpec",
    "DynamicsParams",
    "HealthStatus",
    "SceneSchema",
    "VecPhysicsBackend",
    "register_engine",
    "make_engine",
    "list_engines",
]
