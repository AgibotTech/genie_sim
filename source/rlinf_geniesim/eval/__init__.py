# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Engine-agnostic eval layer: lets the benchmark success checkers run on any
physics backend by reading only the VecPhysicsBackend generic interface."""

from .api_core_shim import ApiCoreShim

__all__ = ["ApiCoreShim"]
