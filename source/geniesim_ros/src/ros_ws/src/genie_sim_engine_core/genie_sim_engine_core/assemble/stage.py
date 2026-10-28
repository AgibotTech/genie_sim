# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``AssembleStage`` ABC.

Every stage declares its logical inputs / outputs statically via
``requires`` / ``produces``. ``Pipeline.validate()`` runs a DAG check
across the ordered stage list before ``run`` is called anywhere, so
missing prerequisites fail at construction time.

Skip mechanics:

* ``optional = True`` — a static "this stage may be skipped" declaration.
  Skipping still fires ``Middleware.after(result=None)``.
* ``skip_reason(ctx)`` — a dynamic skip. Return a non-``None`` string
  and the runner logs the reason and moves on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, FrozenSet

from .context import AssembleContext, StageResult


class AssembleStage(ABC):
    """Base class for every pipeline stage.

    Subclasses set the class attrs then implement ``run``. Optionally
    override ``skip_reason`` for scene-yaml-conditional stages.
    """

    #: Stage identifier used in logs and middleware. Snake case,
    #: backend-prefixed if backend-specific (e.g. ``"newton_extras"``).
    name: ClassVar[str] = ""

    #: Logical names of files this stage writes into
    #: ``ctx.inputs.output_dir``. Enforced by ``InvariantMiddleware``.
    produces: ClassVar[FrozenSet[str]] = frozenset()

    #: Logical names of files this stage reads from prior stages.
    #: ``Pipeline.validate()`` checks these against upstream
    #: ``produces``. An empty set means the stage only reads inputs.
    requires: ClassVar[FrozenSet[str]] = frozenset()

    #: Static declaration that this stage may be omitted from a run.
    #: Distinct from ``skip_reason``, which is dynamic per context.
    optional: ClassVar[bool] = False

    def skip_reason(self, ctx: AssembleContext) -> str | None:
        """Return a non-None reason to skip this stage.

        The runner logs the returned string and moves on. Default: never
        skip.
        """
        return None

    @abstractmethod
    def run(self, ctx: AssembleContext) -> StageResult:
        """Execute the stage. See subclass docstrings for behaviour.

        Backend-heavy imports MUST live inside this method, not at
        module scope.
        """


__all__ = ["AssembleStage"]
