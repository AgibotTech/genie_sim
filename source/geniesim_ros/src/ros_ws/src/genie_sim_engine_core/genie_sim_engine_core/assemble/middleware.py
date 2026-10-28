# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Middleware hooks — cross-cutting concerns around stage execution.

Middleware wrap each stage's ``run`` symmetrically. On skip,
``before`` is skipped and ``after`` fires with ``result=None,
err=None`` so timers close cleanly.

Ships four middleware. Add more here, not in stages.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional, Protocol

from .context import AssembleContext, StageResult
from .stage import AssembleStage

_LOG = logging.getLogger(__name__)


class Middleware(Protocol):
    """Callback protocol for pipeline middleware."""

    def before(self, stage: AssembleStage, ctx: AssembleContext) -> None: ...

    def after(
        self,
        stage: AssembleStage,
        ctx: AssembleContext,
        result: Optional[StageResult],
        err: Optional[BaseException],
    ) -> None: ...


class TimingMiddleware:
    """Log wall time per stage."""

    def __init__(self) -> None:
        self._t0: dict[str, float] = {}

    def before(self, stage: AssembleStage, ctx: AssembleContext) -> None:
        self._t0[stage.name] = time.monotonic()

    def after(
        self,
        stage: AssembleStage,
        ctx: AssembleContext,
        result: Optional[StageResult],
        err: Optional[BaseException],
    ) -> None:
        t0 = self._t0.pop(stage.name, None)
        if t0 is None:
            return
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        status = "ERR" if err is not None else ("SKIP" if result is None else "OK")
        _LOG.info("[assemble] stage %s: %s (%.1f ms)", stage.name, status, elapsed_ms)


class InvariantMiddleware:
    """Verify every ``produces`` path exists under the output dir.

    Catches "stage claimed to write X but didn't" bugs uniformly. Fires
    only on successful stages (skip / err → nothing to check).
    """

    def before(self, stage: AssembleStage, ctx: AssembleContext) -> None:
        pass

    def after(
        self,
        stage: AssembleStage,
        ctx: AssembleContext,
        result: Optional[StageResult],
        err: Optional[BaseException],
    ) -> None:
        if err is not None or result is None:
            return
        expected = set(stage.produces)
        got = set(result.files_added)
        missing = expected - got
        if missing:
            raise RuntimeError(
                f"stage {stage.name!r} declares produces={sorted(expected)!r} "
                f"but did not return {sorted(missing)!r} in files_added"
            )
        for logical, path in result.files_added.items():
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = ctx.inputs.output_dir / resolved
            if not resolved.exists():
                raise RuntimeError(
                    f"stage {stage.name!r} claims to have produced "
                    f"{logical!r} at {resolved}, but the file does not exist"
                )


class ManifestValidatorMiddleware:
    """Verify manifest_slice writes stay within the stage's own bucket.

    ``Manifest.merge_slice`` already enforces core vs backend key
    ownership. This middleware additionally guards that a stage doesn't
    dump into another backend's ``physics.<engine_key>`` bucket by
    accident — that would silently pollute another engine's slice.
    """

    def before(self, stage: AssembleStage, ctx: AssembleContext) -> None:
        pass

    def after(
        self,
        stage: AssembleStage,
        ctx: AssembleContext,
        result: Optional[StageResult],
        err: Optional[BaseException],
    ) -> None:
        if err is not None or result is None:
            return
        physics = result.manifest_slice.get("physics")
        if physics is None:
            return
        expected_key = ctx.inputs.physics_engine
        stray = [key for key in physics if key != expected_key]
        if stray:
            raise RuntimeError(
                f"stage {stage.name!r} wrote to physics.{stray!r} but the "
                f"active engine is {expected_key!r}; each stage may only "
                f"write under its own engine key"
            )


class DebugDumpMiddleware:
    """Snapshot the stage dir after each stage into ``debug_dir/<stage>/``.

    Off by default — the CLI wires this only when ``--debug-dir`` is
    passed. Copies files by their logical names so bisection is
    straightforward.
    """

    def __init__(self, debug_dir: Path) -> None:
        self._debug_dir = Path(debug_dir)
        self._debug_dir.mkdir(parents=True, exist_ok=True)

    def before(self, stage: AssembleStage, ctx: AssembleContext) -> None:
        pass

    def after(
        self,
        stage: AssembleStage,
        ctx: AssembleContext,
        result: Optional[StageResult],
        err: Optional[BaseException],
    ) -> None:
        if result is None:
            return
        dest_root = self._debug_dir / stage.name
        dest_root.mkdir(parents=True, exist_ok=True)
        for logical, path in result.files_added.items():
            src = Path(path)
            if not src.is_absolute():
                src = ctx.inputs.output_dir / src
            if not src.exists():
                # InvariantMiddleware will raise; nothing useful to copy.
                continue
            dst = dest_root / logical
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)


__all__ = [
    "DebugDumpMiddleware",
    "InvariantMiddleware",
    "ManifestValidatorMiddleware",
    "Middleware",
    "TimingMiddleware",
]
