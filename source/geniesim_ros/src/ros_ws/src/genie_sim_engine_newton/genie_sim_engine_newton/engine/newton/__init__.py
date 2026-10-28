# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``engine.newton`` — Newton-direct physics engine implementation.

Newton-standalone is Kit-free.  ``NewtonHeadlessEngine`` is the single
concrete engine class.  For the Kit viewport with Newton physics, use
``physics_engine:=isaac_newton`` instead — that wrapper handles Fabric
internally.
"""

from genie_sim_engine_newton.engine.newton.engine import NewtonHeadlessEngine

__all__ = ["NewtonHeadlessEngine"]
