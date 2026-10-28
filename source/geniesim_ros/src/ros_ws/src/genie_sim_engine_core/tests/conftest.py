# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Package import path for tests that sit at the ``tests/`` root.

Every test subdirectory carries its own copy of this ``sys.path`` insert
because ``genie_sim_engine_core`` is not importable from a bare shell —
not even with ``/opt/ros/humble/setup.bash`` and ``install/setup.bash``
sourced.  Importability comes from these conftests alone.

A test at the ``tests/`` root had been relying on ``python -m pytest``
being invoked from the package directory, which puts cwd on ``sys.path``
and makes the import work by accident.  Run the same file from
``ros_ws`` and collection fails outright, taking the whole suite with it.
This makes the root behave like the subdirectories.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
