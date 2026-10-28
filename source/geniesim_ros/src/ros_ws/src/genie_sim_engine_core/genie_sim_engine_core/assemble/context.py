# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Assemble-pipeline context types.

Frozen ``AssembleInputs`` are what stages read; ``AssembleArtifacts`` is
the mutable accumulator the runner threads through the pipeline.
Stages don't touch ``AssembleArtifacts`` directly — they return a
``StageResult`` and the runner merges it. This keeps stages
side-effect-free with respect to sibling stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schema.scene import SceneConfig as _SchemaSceneConfig


@dataclass(frozen=True)
class RobotSource:
    """Resolved ``robot.robot_source`` block from the scene YAML.

    ``kind`` is the discriminant tag every stage keys off of. Concrete
    values used today: ``"urdf"``, ``"prebaked"``, ``"usd"``.
    ``config`` is the raw sub-block; each stage picks fields it needs.

    Retained for BC alongside the richer
    :class:`genie_sim_engine_core.schema.scene.RobotSource` accessible
    via ``AssembleInputs.scene.robot.robot_source``; new stage code
    should prefer the typed version. This lightweight shape stays until
    every stage has migrated.
    """

    kind: str
    config: Dict[str, Any]


@dataclass(frozen=True)
class AssembleInputs:
    """Everything a stage may read.

    Frozen — a stage that wants to communicate downstream returns
    ``StageResult`` and lets the runner merge into ``artifacts``.

    Two views on the scene YAML travel side-by-side during the schema
    migration: ``scene_cfg`` (legacy raw dict) and ``scene`` (typed
    :class:`SceneConfig`). Stages should prefer ``scene`` for new
    reads; ``scene_cfg`` is scheduled for removal once every existing
    reader has migrated.
    """

    scene_path: Path
    output_dir: Path
    base_path: Path
    physics_engine: str
    scene_cfg: Dict[str, Any]
    robot_source: RobotSource
    #: Typed view of ``scene_cfg`` — see :class:`SceneConfig`.
    scene: Optional[_SchemaSceneConfig] = None


@dataclass
class AssembleArtifacts:
    """Accumulator threaded through the pipeline.

    ``files`` is keyed by logical name (``"robot.usda"``,
    ``"manifest.json"``) so downstream stages can look up prior outputs
    without hard-coding filesystem paths.
    """

    files: Dict[str, Path] = field(default_factory=dict)
    manifest_contributions: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class StageResult:
    """Return value from ``AssembleStage.run``.

    Every path in ``files_added`` must be present under
    ``ctx.inputs.output_dir`` after ``run`` returns —
    ``InvariantMiddleware`` enforces this.
    """

    files_added: Dict[str, Path] = field(default_factory=dict)
    manifest_slice: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AssembleContext:
    """What a stage sees during ``run``.

    ``inputs`` is frozen; ``artifacts`` is the runner's accumulator that
    stages MAY inspect (to look up upstream files by logical name) but
    MUST NOT mutate. All new outputs go through ``StageResult``.
    """

    inputs: AssembleInputs
    artifacts: AssembleArtifacts


__all__ = [
    "AssembleArtifacts",
    "AssembleContext",
    "AssembleInputs",
    "RobotSource",
    "StageResult",
]
