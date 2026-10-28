# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared pytest fixtures for EngineSession tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Make `genie_sim_engine_core` importable without going through
# ament_python_install (colcon test runs before install by default).
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
