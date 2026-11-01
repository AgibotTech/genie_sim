# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Render backend registry + factory.

Mirror of ``physics/backends/__init__.py`` for renderers.
"""

from __future__ import annotations

import inspect
from typing import Any

from .base import (
    CameraSpec,
    RenderHealthStatus,
    RenderResult,
    VecRenderBackend,
)

# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[VecRenderBackend]] = {}


def register_bridge(name: str):
    def deco(cls: type[VecRenderBackend]) -> type[VecRenderBackend]:
        if name in _REGISTRY:
            raise ValueError(
                f"render bridge '{name}' already registered "
                f"(existing={_REGISTRY[name].__name__}, new={cls.__name__})"
            )
        if not issubclass(cls, VecRenderBackend):
            raise TypeError(f"{cls.__name__} must subclass VecRenderBackend")
        _REGISTRY[name] = cls
        return cls

    return deco


def make_bridge(name: str, **cfg: Any) -> VecRenderBackend:
    if name not in _REGISTRY:
        raise ValueError(f"unknown render backend '{name}'; " f"valid: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**cfg)


def list_bridges() -> list[str]:
    return sorted(_REGISTRY)


def bridge_supports_kwarg(name: str, kwarg: str) -> bool:
    """Return whether the registered constructor explicitly declares a keyword.

    A generic **kwargs parameter does not establish support. This lets callers pass
    backend-specific options without silently accepting options a bridge ignores."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown render backend '{name}'; valid: {sorted(_REGISTRY)}")
    return kwarg in inspect.signature(cls.__init__).parameters


def bridges_supporting_kwarg(kwarg: str) -> list[str]:
    """Which registered bridges' constructors explicitly declare ``kwarg`` —
    used to answer "which bridges support it" in error messages; the test is
    exactly the same one ``bridge_supports_kwarg`` uses."""
    return sorted(name for name, cls in _REGISTRY.items() if kwarg in inspect.signature(cls.__init__).parameters)


# ---------------------------------------------------------------------------
# Trigger registration of every concrete render backend.
# ---------------------------------------------------------------------------

from . import ovrtx_node  # noqa: E402, F401  → "ovrtx_node"
from . import isaac_tiled  # noqa: E402, F401  → "isaac_tiled" (RTX tiled)
from . import null  # noqa: E402, F401  → "none" (no-op, physics-only)


__all__ = [
    "CameraSpec",
    "RenderHealthStatus",
    "RenderResult",
    "VecRenderBackend",
    "register_bridge",
    "make_bridge",
    "list_bridges",
    "bridge_supports_kwarg",
    "bridges_supporting_kwarg",
]
