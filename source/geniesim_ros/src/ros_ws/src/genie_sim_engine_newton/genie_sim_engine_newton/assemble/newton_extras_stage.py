# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""NewtonExtrasStage — writes ``newton_solvers.json`` for the runtime.

Wraps :mod:`genie_sim_engine.assemble._newton_extras_impl` in the
``AssembleStage`` contract so the pipeline runner can invoke it. The
impl module carries every YAML validation rule and formatting choice
from the previous ``scripts/assemble_newton.py``; this stage adds the
core-level metadata (name, produces, skip_reason, manifest slice) and
translates between :class:`AssembleContext` and the impl's positional
API.
"""

from __future__ import annotations

from genie_sim_engine_core.assemble import AssembleContext, AssembleStage, StageResult

from ._newton_extras_impl import (
    SIDECAR_NAME,
    build_and_write,
    check_scene_has_newton_extras,
)


class NewtonExtrasStage(AssembleStage):
    """Emit ``newton_solvers.json`` for cloth / soft / particle entries.

    Skips when the scene has no ``newton:`` block or empty
    ``newton.entries``. On success, contributes:

    * ``newton_solvers.json`` under ``StageResult.files_added``
    * a small metadata slice under ``physics.<engine>.newton_extras`` so
      launchers can inspect entry counts / solver without reading the
      sidecar
    """

    name = "newton_extras"
    produces = frozenset({SIDECAR_NAME})

    def skip_reason(self, ctx: AssembleContext):
        return check_scene_has_newton_extras(ctx.inputs.scene_cfg)

    def run(self, ctx: AssembleContext) -> StageResult:
        sidecar_path = build_and_write(
            scene_path=ctx.inputs.scene_path,
            output_dir=ctx.inputs.output_dir,
            base_path=str(ctx.inputs.base_path) if ctx.inputs.base_path else "",
            scene_cfg=ctx.inputs.scene_cfg,
        )
        newton_cfg = ctx.inputs.scene_cfg.get("newton") or {}
        solver_prefer = ""
        if isinstance(newton_cfg.get("solver"), dict):
            solver_prefer = str(newton_cfg["solver"].get("prefer") or "").strip().lower()
        return StageResult(
            files_added={SIDECAR_NAME: sidecar_path},
            manifest_slice={
                "physics": {
                    ctx.inputs.physics_engine: {
                        "newton_extras": {
                            "sidecar": SIDECAR_NAME,
                            "entry_count": len(newton_cfg.get("entries") or []),
                            "solver_prefer": solver_prefer,
                        }
                    }
                }
            },
        )


__all__ = ["NewtonExtrasStage"]
