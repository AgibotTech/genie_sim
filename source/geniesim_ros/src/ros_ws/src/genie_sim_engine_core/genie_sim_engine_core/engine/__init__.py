# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""genie_sim_engine_core.engine — PhysicsEngine ABC + shared engine helpers.

Modules:
  - base:          PhysicsEngine ABC + factory
  - lifecycle:     lifecycle records + the ActivitySignal Protocol.  The
                   control VERBS live on ``base.PhysicsEngine``, which is
                   this tree's lifecycle contract; there is no Protocol
                   mirror of it here on purpose (see that module).
  - publish_plan:  PublishPlan — the per-run v2 wire-routing decision
  - _mimic:        mimic-joint helper (parse_mimic, expand_targets)

Nothing is re-exported here: the submodule names are short and explicit, and
a backend importing ``engine.lifecycle`` should not pay for ``engine.base``'s
numpy import to get a dataclass.
"""
