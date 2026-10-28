# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""genie_sim_engine_core.common — shared run-loop / session / params utilities.

Modules:
  - early_params: parameter inspection that must run pre-executor
  - loop:         EngineRunLoop step scaffold
  - session:      EngineSession setup orchestration
  - params:       EngineNodeParams typed config coercion
  - scene_config: scene-yaml flag parsers
  - usd_walk:     shared body-name → USD prim-path resolver
  - reset_randomization: seeded reset draw shared by the Newton engine and
                 the fused PhysX node
"""
