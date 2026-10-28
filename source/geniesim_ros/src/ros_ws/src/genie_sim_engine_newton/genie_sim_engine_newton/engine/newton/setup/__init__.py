# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build-phase mixins for ``NewtonHeadlessEngine``.

Each module owns one slice of the engine startup pipeline; the
``_build`` table-of-contents lives in :mod:`runtime`.  Composed
into ``_NewtonStandaloneBase`` (in ``engine_base.py``) via plain
multiple inheritance — no metaclasses, no auto-discovery.

File-system order mirrors execution order::

    stage      -> open stage + session-layer overrides
    model      -> ModelBuilder + add_usd + finalize
    normalize  -> mass clamp + contact mats + group/articulation unification
    solver     -> states + robot solver + cloth solver
    debug_pubs -> optional rclpy marker publishers
    init_pose  -> name maps + init_joint_pos + MJCF keyframe + state sync
    static_facts -> one-shot latched publishes of build-decided facts
    runtime    -> _build (TOC) + _warmup + _capture_graph + _dump_runtime_usd
"""

from genie_sim_engine_newton.engine.newton.setup.stage import _StageMixin
from genie_sim_engine_newton.engine.newton.setup.model import _ModelMixin
from genie_sim_engine_newton.engine.newton.setup.normalize import _NormalizeMixin
from genie_sim_engine_newton.engine.newton.setup.solver import _SolverMixin
from genie_sim_engine_newton.engine.newton.setup.debug_pubs import _DebugPubsMixin
from genie_sim_engine_newton.engine.newton.setup.init_pose import _InitPoseMixin
from genie_sim_engine_newton.engine.newton.setup.static_facts import _StaticFactsMixin
from genie_sim_engine_newton.engine.newton.setup.runtime import _RuntimeMixin

__all__ = [
    "_StageMixin",
    "_ModelMixin",
    "_NormalizeMixin",
    "_SolverMixin",
    "_DebugPubsMixin",
    "_InitPoseMixin",
    "_StaticFactsMixin",
    "_RuntimeMixin",
]
