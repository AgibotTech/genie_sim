# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Backend configuration, process singleton, and tensor conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

import numpy as np


class Device(str, Enum):
    """Physics/render device enum; string values can be passed directly to APIs."""

    CPU = "cpu"
    CUDA = "cuda"


TensorsAs = Literal["numpy", "torch"]


@dataclass(frozen=True)
class BackendConfig:
    """Physics and rendering backend configuration.
    physics_device selects PhysX execution and the SimulationManager backend.
    render_device selects RTX execution; tensors_as controls output tensor types,
    defaulting to the physics device. cuda_index selects the GPU."""

    physics_device: Device
    render_device: Device
    tensors_as: Optional[TensorsAs] = None
    cuda_index: int = 0

    def resolve_tensors_as(self) -> TensorsAs:
        if self.tensors_as is not None:
            return self.tensors_as
        # The CUDA pipeline upgrades SimulationManager to torch; match its tensor type.
        return "torch" if self.physics_device == Device.CUDA else "numpy"

    def physics_device_str(self) -> str:
        if self.physics_device == Device.CUDA:
            return f"cuda:{self.cuda_index}"
        return "cpu"

    def render_device_str(self) -> str:
        if self.render_device == Device.CUDA:
            return f"cuda:{self.cuda_index}"
        return "cpu"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(cfg: BackendConfig) -> None:
    if cfg.render_device == Device.CPU:
        raise ValueError(
            "Invalid BackendConfig: render_device=CPU is unsupported. "
            "RTX rendering requires CUDA."
        )
    if cfg.physics_device == Device.CUDA and cfg.tensors_as == "numpy":
        raise ValueError(
            "Invalid BackendConfig: physics_device=CUDA is incompatible with tensors_as='numpy'. "
            "With the CUDA pipeline, SimulationManager._backend is automatically promoted to 'torch', "
            "and the SingleArticulation public API accepts only torch.Tensor. "
            "Set tensors_as to 'torch' or leave it unset for automatic selection."
        )
    if cfg.cuda_index < 0:
        raise ValueError(f"cuda_index must be >= 0; got {cfg.cuda_index}")


# ---------------------------------------------------------------------------
# Process singleton
# ---------------------------------------------------------------------------

_backend: Optional[BackendConfig] = None


def init_backend(cfg: BackendConfig) -> None:
    """Validate and initialize the process singleton; reject repeated initialization."""
    global _backend
    if _backend is not None:
        raise RuntimeError(
            "init_backend has already been called; current backend="
            f"{_backend!r}. Initialization is allowed only once per process."
        )
    _validate(cfg)
    _backend = cfg


def get_backend() -> BackendConfig:
    if _backend is None:
        raise RuntimeError(
            "Backend is not initialized. Call sim_backend.runtime.init_app(...) or "
            " sim_backend.backend.init_backend(...)"
        )
    return _backend


def is_backend_initialized() -> bool:
    return _backend is not None


def physics_device_str() -> str:
    return get_backend().physics_device_str()


def render_device_str() -> str:
    return get_backend().render_device_str()


def tensors_as() -> TensorsAs:
    return get_backend().resolve_tensors_as()


def torch_device():
    """Return the configured physics torch.device for tensor placement."""
    import torch  # Import lazily so importing sim_backend does not load torch.

    cfg = get_backend()
    if cfg.physics_device == Device.CUDA:
        return torch.device("cuda", cfg.cuda_index)
    return torch.device("cpu")


def cuda_torch_device():
    """Return cuda:cuda_index regardless of the configured physics device.
    Use this for cuRobo and other operators that require CUDA tensors even when
    physics runs on CPU. torch_device instead follows physics_device."""
    import torch  # noqa: PLC0415

    cfg = get_backend()
    return torch.device("cuda", cfg.cuda_index)


# ---------------------------------------------------------------------------
# Tensor bridge
# ---------------------------------------------------------------------------


def to_torch(x, *, dtype=None, device=None):
    """Convert to torch.Tensor on the configured physics device unless overridden."""
    import torch

    if device is None:
        device = torch_device()
    if isinstance(x, torch.Tensor):
        t = x.to(device=device)
    else:
        t = torch.as_tensor(np.asarray(x), device=device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


def to_numpy(x) -> np.ndarray:
    """Convert to ndarray, detaching torch tensors and moving them to CPU."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def resolve_as_torch(explicit: Optional[bool]) -> bool:
    """Use the explicit output-type preference, or fall back to the backend."""
    if explicit is not None:
        return explicit
    return tensors_as() == "torch"


def pipeline_aware_input(x: Any) -> Any:
    """Convert array-like input to the type expected by SimulationManager.
    SingleArticulation setters and apply_action require ndarray for numpy backends
    and torch.Tensor for torch backends; they do not convert automatically.
    Physics setter/apply wrappers use this bridge to avoid type mismatches.
    None is passed through; other inputs are converted, including empty arrays."""
    if x is None:
        return None
    target = tensors_as()
    if target == "torch":
        import torch

        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(np.asarray(x), device=torch_device())
    # target == "numpy"
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)
