# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""genie_sim_engine_isaacsim — Isaac Sim / Kit backend for the GenieSim engine platform.

Ships the Kit lifecycle helpers (``kit/bootstrap.py``,
``kit/stage.py``), the ``PhysicsEngine`` implementations that run under
Isaac Sim (``kit/isaac_newton.py``, ``kit/isaac_physx.py``), the URDF→USD
+ scene-assembly stages, and the ``isaac_newton`` pipeline factory.

Importing this package's submodules does NOT itself boot Isaac Sim —
Kit / omni / pxr imports live inside functions so consumers on
Kit-less machines can inspect the pipeline registration path.

The paired Newton-standalone (Kit-free) package is
:mod:`genie_sim_engine` (to be renamed to ``genie_sim_engine_newton`` in
a follow-up commit).
"""
