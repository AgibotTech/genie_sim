# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Public API for the assemble subsystem.

Backend packages should import from this module, not from the submodules
directly. Anything not re-exported here is considered private and may
move between submodules without notice.
"""

from __future__ import annotations

from .context import (
    AssembleArtifacts,
    AssembleContext,
    AssembleInputs,
    RobotSource,
    StageResult,
)
from .manifest import MANIFEST_VERSION, Manifest, RAW_MANIFEST_SENTINEL
from .middleware import (
    DebugDumpMiddleware,
    InvariantMiddleware,
    ManifestValidatorMiddleware,
    Middleware,
    TimingMiddleware,
)
from .pipeline import Pipeline, PipelineRunner
from .stage import AssembleStage

__all__ = [
    "AssembleArtifacts",
    "AssembleContext",
    "AssembleInputs",
    "AssembleStage",
    "DebugDumpMiddleware",
    "InvariantMiddleware",
    "MANIFEST_VERSION",
    "Manifest",
    "ManifestValidatorMiddleware",
    "Middleware",
    "Pipeline",
    "PipelineRunner",
    "RAW_MANIFEST_SENTINEL",
    "RobotSource",
    "StageResult",
    "TimingMiddleware",
]
