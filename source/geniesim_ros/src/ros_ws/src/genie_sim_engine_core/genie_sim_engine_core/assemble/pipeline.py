# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``Pipeline`` — an ordered stage list.

Two responsibilities:

1. Register named pipelines via ``@Pipeline.register(name)``.
2. Validate the ``produces``/``requires`` DAG before any stage runs.

The register-decorator pattern mirrors ``PhysicsEngine.register`` so
consumers only need to remember one registration idiom across the
codebase. Registration is triggered by import; the CLI's
``--pipeline-module`` flag lists modules to import before dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Type

from .context import AssembleArtifacts, AssembleContext, StageResult
from .manifest import Manifest
from .stage import AssembleStage

_LOG = logging.getLogger(__name__)

#: Pipeline factories keyed by pipeline name. Populated by
#: ``@Pipeline.register`` and consulted by the CLI.
_REGISTRY: Dict[str, Callable[[], "Pipeline"]] = {}


@dataclass
class Pipeline:
    """A named, ordered list of stage classes.

    ``stages`` are classes, not instances — the runner instantiates
    each per run so stages remain stateless between runs.
    """

    name: str
    stages: List[Type[AssembleStage]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    @staticmethod
    def register(name: str) -> Callable[[Callable[[], "Pipeline"]], Callable[[], "Pipeline"]]:
        """Decorator that registers a pipeline factory under ``name``.

        The factory returns a fresh ``Pipeline`` — do not share state
        across calls, and do not import backend runtimes in the factory
        body. Backend imports belong inside each stage's ``run``.

        Example::

            @Pipeline.register("my_backend")
            def my_backend_pipeline() -> Pipeline:
                return Pipeline(name="my_backend", stages=[Stage1, Stage2])
        """

        def _decorator(factory: Callable[[], "Pipeline"]) -> Callable[[], "Pipeline"]:
            if name in _REGISTRY:
                _LOG.warning("pipeline %r is being re-registered; overriding previous factory", name)
            _REGISTRY[name] = factory
            return factory

        return _decorator

    @staticmethod
    def registered_names() -> List[str]:
        """Names of currently-registered pipelines, sorted."""
        return sorted(_REGISTRY)

    @staticmethod
    def get(name: str) -> "Pipeline":
        """Return the registered pipeline instance, or raise ``KeyError``."""
        try:
            factory = _REGISTRY[name]
        except KeyError as err:
            raise KeyError(f"no pipeline registered under {name!r}; " f"known: {Pipeline.registered_names()}") from err
        return factory()

    # ------------------------------------------------------------------
    # DAG validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Fail fast on dead-end ``requires`` — before any stage runs.

        Every logical name in a stage's ``requires`` set must appear in
        the ``produces`` set of some earlier stage in this pipeline.
        Inputs to the pipeline (scene YAML, output dir) are implicit
        and NOT modelled as ``produces`` — they are accessed via
        ``ctx.inputs`` directly.
        """
        if not self.stages:
            raise ValueError(f"pipeline {self.name!r} has zero stages")

        produced: set = set()
        seen_names: set = set()
        for stage_cls in self.stages:
            name = getattr(stage_cls, "name", "")
            if not name:
                raise ValueError(
                    f"pipeline {self.name!r}: stage class {stage_cls.__name__} " "has an empty 'name' class attr"
                )
            if name in seen_names:
                raise ValueError(f"pipeline {self.name!r}: duplicate stage name {name!r}")
            seen_names.add(name)

            requires = getattr(stage_cls, "requires", frozenset())
            missing = requires - produced
            if missing:
                raise ValueError(
                    f"pipeline {self.name!r}: stage {name!r} requires "
                    f"{sorted(missing)!r} but no earlier stage produces them"
                )
            produced.update(getattr(stage_cls, "produces", frozenset()))


class PipelineRunner:
    """Executes a pipeline over an ``AssembleContext``.

    Middleware hooks fire in the order they were passed, wrapping each
    stage's ``run``. On exception, ``after`` fires with ``err`` set;
    the exception then propagates to the caller.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        middleware: Optional[Iterable["Middleware"]] = None,  # noqa: F821
    ) -> None:
        from .middleware import Middleware  # local to avoid cycle

        self._pipeline = pipeline
        self._middleware: List[Middleware] = list(middleware or [])
        self._log = logging.getLogger(f"{__name__}.PipelineRunner")
        #: Stage names the CLI's ``--skip-stage`` (or another caller)
        #: has requested to bypass. Skipped stages still fire middleware
        #: ``after`` hooks so timers close cleanly.
        self.explicit_skips: set = set()

    def run(self, ctx: AssembleContext) -> Manifest:
        self._pipeline.validate()
        manifest = Manifest(
            physics_engine=ctx.inputs.physics_engine,
            base_path=ctx.inputs.base_path,
            stage_dir=ctx.inputs.output_dir,
            scene_yaml=ctx.inputs.scene_path,
        )
        for stage_cls in self._pipeline.stages:
            stage = stage_cls()
            for mw in self._middleware:
                mw.before(stage, ctx)
            if stage.name in self.explicit_skips:
                self._log.info("skip stage %s: explicit --skip-stage", stage.name)
                self._fire_after(stage, ctx, result=None, err=None)
                continue
            reason = stage.skip_reason(ctx)
            if reason is not None:
                self._log.info("skip stage %s: %s", stage.name, reason)
                self._fire_after(stage, ctx, result=None, err=None)
                continue
            err: Optional[BaseException] = None
            result: Optional[StageResult] = None
            try:
                result = stage.run(ctx)
                if not isinstance(result, StageResult):
                    raise TypeError(
                        f"stage {stage.name!r} returned {type(result).__name__}; " f"must return StageResult"
                    )
                # Fold into artifacts before firing 'after' so middleware
                # inspecting artifacts sees the post-stage state.
                _fold(ctx.artifacts, result)
                manifest.merge_slice(result.manifest_slice, stage.name)
            except BaseException as caught:  # noqa: BLE001
                err = caught
                raise
            finally:
                self._fire_after(stage, ctx, result=result, err=err)
        return manifest

    def _fire_after(
        self,
        stage: AssembleStage,
        ctx: AssembleContext,
        *,
        result: Optional[StageResult],
        err: Optional[BaseException],
    ) -> None:
        # Reverse order so 'after' unwinds symmetrically vs 'before'.
        #
        # Middleware exceptions:
        # - on the SUCCESS path (err is None) the middleware IS the check
        #   (InvariantMiddleware, ManifestValidatorMiddleware); its
        #   exception must propagate to fail the pipeline. Swallowing here
        #   would defeat the whole point of those middleware.
        # - on the ERROR path (stage already raised) middleware exceptions
        #   are logged instead of replacing the original — the stage's
        #   exception is the one the caller needs to see.
        for mw in reversed(self._middleware):
            try:
                mw.after(stage, ctx, result, err)
            except Exception:  # noqa: BLE001
                if err is None:
                    raise
                self._log.exception(
                    "middleware %s.after raised while unwinding stage %r error — "
                    "logging and continuing so the original error is preserved",
                    type(mw).__name__,
                    stage.name,
                )


def _fold(artifacts: AssembleArtifacts, result: StageResult) -> None:
    artifacts.files.update(result.files_added)
    if result.manifest_slice:
        artifacts.manifest_contributions.append(result.manifest_slice)
    if result.warnings:
        artifacts.warnings.extend(result.warnings)


__all__ = ["Pipeline", "PipelineRunner"]
