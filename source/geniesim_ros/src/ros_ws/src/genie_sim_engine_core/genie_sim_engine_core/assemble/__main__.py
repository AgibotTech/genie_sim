# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``python -m genie_sim_engine_core.assemble`` — the pipeline CLI.

Loads scene YAML, resolves ``robot.robot_source``, imports every module
listed via ``--pipeline-module`` so their ``@Pipeline.register``
decorators run, and dispatches to the pipeline named by
``--physics-engine``.

Exit codes (contract for launchers):

* ``0`` — success
* ``1`` — pipeline validation / configuration failure
* ``2`` — stage runtime failure
* ``3`` — manifest write failure
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .context import AssembleArtifacts, AssembleContext, AssembleInputs, RobotSource
from .middleware import (
    DebugDumpMiddleware,
    InvariantMiddleware,
    ManifestValidatorMiddleware,
    Middleware,
    TimingMiddleware,
)
from .pipeline import Pipeline, PipelineRunner

_LOG = logging.getLogger("genie_sim_engine_core.assemble")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m genie_sim_engine_core.assemble",
        description="Compose a per-scene stage dir for a physics-engine backend.",
    )
    parser.add_argument("--scene", type=Path, help="Scene YAML path")
    parser.add_argument("--output-dir", type=Path, help="Stage output directory")
    parser.add_argument(
        "--base-path",
        type=Path,
        help="Asset-root anchor for resolving relative paths in the scene YAML",
    )
    parser.add_argument(
        "--physics-engine",
        help="Name of the registered pipeline to run (matches @Pipeline.register)",
    )
    parser.add_argument(
        "--pipeline-module",
        action="append",
        default=[],
        help=(
            "Python import path of a backend module that registers pipelines "
            "via @Pipeline.register. Repeatable; imported before dispatch so "
            "the decorator runs. Same idiom as the PhysicsEngine registry."
        ),
    )
    parser.add_argument(
        "--skip-stage",
        action="append",
        default=[],
        help=(
            "Skip the named stage even when its skip_reason() returns None. "
            "Repeatable. Used by launchers to encode caller-side gating "
            "(e.g. a cache-hit check that skips the URDF→USD stage)."
        ),
    )
    parser.add_argument(
        "--launcher-config",
        default="",
        help=(
            "Launcher YAML name / path passed by the bringup launcher. "
            "Stamped into ``manifest.launcher_config`` so a subsequent "
            "launch under a different launcher can detect the swap and "
            "invalidate cache-gated stages. Empty string omits the stamp."
        ),
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Enable DebugDumpMiddleware, writing per-stage snapshots here",
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="List registered pipelines (after processing --pipeline-module) and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG-level logging",
    )
    return parser.parse_args(argv)


def _import_pipeline_modules(module_paths: List[str]) -> None:
    """Import each module so its ``@Pipeline.register`` decorators fire.

    A broken backend package logs and does NOT stop discovery of others.
    Matches the tolerant import loop the ``PhysicsEngine`` smoke check
    documents.
    """
    for path in module_paths:
        try:
            importlib.import_module(path)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("pipeline module %r failed to import: %s", path, exc)


def _load_scene_yaml(scene_path: Path) -> Dict[str, Any]:
    with scene_path.open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"scene YAML at {scene_path} did not parse to a mapping")
    return cfg


def _resolve_robot_source(scene_cfg: Dict[str, Any]) -> RobotSource:
    """Resolve ``robot.robot_source`` into a discriminated ``RobotSource``.

    Ordering below defines the discriminant precedence — first match
    wins. Extra sibling keys are preserved in ``config`` for the stage
    to inspect.
    """
    robot = (scene_cfg.get("robot") or {}).get("robot_source") or {}
    if not isinstance(robot, dict):
        raise ValueError("robot.robot_source must be a mapping")
    for kind in ("urdf", "prebaked", "usd"):
        if kind in robot:
            return RobotSource(kind=kind, config=dict(robot))
    # Empty / unknown — still return a RobotSource so stages can skip.
    return RobotSource(kind="", config=dict(robot))


def _build_context(args: argparse.Namespace) -> AssembleContext:
    scene_cfg = _load_scene_yaml(args.scene)
    # Typed view of the scene YAML — new stages should read from this;
    # ``scene_cfg`` (raw dict) stays for the duration of the schema
    # migration.
    from ..schema import SceneConfig as _SchemaSceneConfig

    inputs = AssembleInputs(
        scene_path=args.scene.resolve(),
        output_dir=args.output_dir.resolve(),
        base_path=args.base_path.resolve(),
        physics_engine=args.physics_engine,
        scene_cfg=scene_cfg,
        robot_source=_resolve_robot_source(scene_cfg),
        scene=_SchemaSceneConfig.from_dict(scene_cfg, source_path=args.scene.resolve()),
    )
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    return AssembleContext(inputs=inputs, artifacts=AssembleArtifacts())


def _build_middleware(args: argparse.Namespace) -> List[Middleware]:
    middleware: List[Middleware] = [
        TimingMiddleware(),
        InvariantMiddleware(),
        ManifestValidatorMiddleware(),
    ]
    if args.debug_dir is not None:
        middleware.append(DebugDumpMiddleware(args.debug_dir))
    return middleware


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    _import_pipeline_modules(args.pipeline_module)

    if args.list_pipelines:
        names = Pipeline.registered_names()
        if names:
            print("\n".join(names))
        else:
            print("(no pipelines registered)")
        return 0

    for required in ("scene", "output_dir", "base_path", "physics_engine"):
        if getattr(args, required, None) is None:
            _LOG.error("--%s is required", required.replace("_", "-"))
            return 1

    try:
        pipeline = Pipeline.get(args.physics_engine)
    except KeyError as err:
        _LOG.error("%s", err)
        return 1

    try:
        ctx = _build_context(args)
    except (FileNotFoundError, ValueError) as err:
        _LOG.error("failed to build context: %s", err)
        return 1

    runner = PipelineRunner(pipeline, _build_middleware(args))
    if args.skip_stage:
        # Validate that every explicit skip names a real stage in the
        # selected pipeline; otherwise a caller typo (or a rename)
        # would silently no-op.
        stage_names = {s.name for s in pipeline.stages}
        unknown = [n for n in args.skip_stage if n not in stage_names]
        if unknown:
            _LOG.error(
                "--skip-stage: unknown stage name(s) %r; pipeline %r has %s",
                unknown,
                pipeline.name,
                sorted(stage_names),
            )
            return 1
        runner.explicit_skips = set(args.skip_stage)
    try:
        manifest = runner.run(ctx)
    except Exception:  # noqa: BLE001
        _LOG.exception("pipeline %r failed during stage execution", args.physics_engine)
        return 2

    # Stamp the launcher-config source onto the manifest so a
    # subsequent launch under a different launcher YAML can detect the
    # swap and invalidate cache-gated stages (see
    # ``genie_sim_bringup/launch/utils.py`` mismatch check).  Empty
    # string means "caller did not stamp" — treated the same as a
    # mismatch on next launch.
    if args.launcher_config:
        manifest.launcher_config = args.launcher_config

    manifest_path = ctx.inputs.output_dir / "manifest.json"
    try:
        manifest.write(manifest_path)
    except OSError as err:
        _LOG.error("failed to write manifest at %s: %s", manifest_path, err)
        return 3
    _LOG.info("[assemble] wrote %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
