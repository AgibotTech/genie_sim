# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""IsaacSceneStage — Isaac Sim scene assembly.

Runs :mod:`genie_sim_engine_libuipc.assemble._scene_impl` in a **fresh Python
subprocess**. Same rationale as :class:`UrdfToUsdStage`: Isaac Sim's
``SimulationApp`` shutdown can terminate the interpreter, and a second
Isaac Sim init in the same process is unsupported. Subprocess
isolation keeps that shutdown inside a process the pipeline runner
can outlive.

Manifest flow: the subprocess writes a schema-shaped ``manifest.json``
via the shared :class:`~genie_sim_engine_core.schema.Manifest` writer;
this stage reads it back and forwards each top-level field as a normal
``manifest_slice`` entry so the pipeline runner's :class:`Manifest`
composes the manifest without backend-specific extras.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from genie_sim_engine_core.assemble import (
    AssembleContext,
    AssembleStage,
    StageResult,
)
from genie_sim_engine_core.schema import Manifest

from .urdf_to_usd_stage import _extract_robot_body_paths, _extract_robot_joint_names


class IsaacSceneStage(AssembleStage):
    """Stage: compose the Isaac Sim scene assets (runs as a subprocess).

    Requires nothing statically — the scene may reference either
    ``robot.usda`` produced by :class:`UrdfToUsdStage` or a pre-baked
    ``robot.usda`` inside the asset tree. The impl handles both paths
    based on ``robot.robot_source``.

    ``produces`` intentionally excludes ``manifest.json``: core writes
    it from the merged manifest state after this stage returns, so
    listing it here would trip :class:`InvariantMiddleware` (the file
    is not yet at ``output_dir/manifest.json`` when the middleware
    checks).
    """

    name = "isaac_scene"
    produces = frozenset({"render_layer.usda"})

    def skip_reason(self, ctx: AssembleContext) -> Optional[str]:
        # The impl handles every ``robot.robot_source`` shape (urdf,
        # prebaked, usd); there is no skip case at this layer.
        return None

    def run(self, ctx: AssembleContext) -> StageResult:
        cmd = [
            sys.executable,
            "-m",
            "genie_sim_engine_libuipc.assemble._scene_impl",
            "--scene",
            str(ctx.inputs.scene_path),
            "--output-dir",
            str(ctx.inputs.output_dir),
            "--base-path",
            str(ctx.inputs.base_path) if ctx.inputs.base_path else "",
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"isaac_scene subprocess exited with {result.returncode}; see stdout above")

        out = Path(ctx.inputs.output_dir)
        manifest_path = out / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"isaac_scene subprocess returned 0 but did not write {manifest_path}. "
                "Check the subprocess stdout for the actual failure."
            )
        subprocess_manifest = Manifest.read(manifest_path)

        files = {"render_layer.usda": out / "render_layer.usda"}
        robot_visual = out / "robot_visual.usda"
        if robot_visual.exists():
            files["robot_visual.usda"] = robot_visual

        # Forward each top-level field the Isaac subprocess populated
        # so other pipeline stages can merge their
        # ``physics.*`` slice on top instead of getting stomped.
        slice_: Dict[str, Any] = {
            "scene_yaml": subprocess_manifest.scene_yaml,
            "scene_usda": subprocess_manifest.scene_usda,
            "robot_usda": subprocess_manifest.robot_usda,
            "robot_visual_usda": subprocess_manifest.robot_visual_usda,
            "render_layer_usda": subprocess_manifest.render_layer_usda,
            "robot_prefix": subprocess_manifest.robot_prefix,
            "robot_from_urdf": subprocess_manifest.robot_from_urdf,
            "free_cam_prim_path": subprocess_manifest.free_cam_prim_path,
            "cameras": list(subprocess_manifest.cameras),
            "lidars": list(subprocess_manifest.lidars),
        }

        # v2 data-exchange protocol: emit ``robot.body_paths`` whether
        # or not :class:`UrdfToUsdStage` ran this launch. When the URDF
        # hash matches the previous bake, the launcher passes
        # ``--skip-stage urdf_to_usd`` and UrdfToUsdStage's slice never
        # contributes; the cached ``robot.usda`` + ``robot.urdf`` still
        # sit on disk though, and we can re-walk them here so the final
        # manifest carries the body_paths table the renderer needs.
        # Robot poses lookup on ``/tf`` fails silently without it.
        robot_urdf = out / "robot.urdf"
        robot_usda = out / "robot.usda"
        if robot_urdf.is_file() and robot_usda.is_file():
            body_paths = _extract_robot_body_paths(
                urdf_path=robot_urdf,
                robot_usda=robot_usda,
                robot_prefix=subprocess_manifest.robot_prefix or "genie",
            )
            joint_names = _extract_robot_joint_names(urdf_path=robot_urdf)
            robot_slice: Dict[str, Any] = {}
            if body_paths:
                robot_slice["body_paths"] = body_paths
            if joint_names:
                robot_slice["joint_names"] = joint_names
            if robot_slice:
                slice_["robot"] = robot_slice
        return StageResult(
            files_added=files,
            manifest_slice=slice_,
        )


__all__ = ["IsaacSceneStage"]
