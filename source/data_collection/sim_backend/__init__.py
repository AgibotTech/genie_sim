# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Unified Isaac Sim physics and rendering backend.
Initialize with init_app(backend_config=BackendConfig(...)) before creating
World or invoking physics/render methods. Import Isaac Sim runtime submodules
after application startup."""

from __future__ import annotations

from .backend import (
    BackendConfig,
    Device,
    TensorsAs,
    cuda_torch_device,
    get_backend,
    init_backend,
    is_backend_initialized,
    physics_device_str,
    pipeline_aware_input,
    render_device_str,
    resolve_as_torch,
    tensors_as,
    to_numpy,
    to_torch,
    torch_device,
)
from .runtime import (
    AppHandle,
    RtxMode,
    get_app,
    init_app,
    is_app_started,
    shutdown_app,
)

# Expose physics and render as public namespaces.
from . import physics, render

__all__ = [
    # backend
    "BackendConfig",
    "Device",
    "TensorsAs",
    "cuda_torch_device",
    "get_backend",
    "init_backend",
    "is_backend_initialized",
    "physics_device_str",
    "pipeline_aware_input",
    "render_device_str",
    "resolve_as_torch",
    "tensors_as",
    "to_numpy",
    "to_torch",
    "torch_device",
    # runtime
    "AppHandle",
    "RtxMode",
    "get_app",
    "init_app",
    "is_app_started",
    "shutdown_app",
    # subpackages
    "physics",
    "render",
]
