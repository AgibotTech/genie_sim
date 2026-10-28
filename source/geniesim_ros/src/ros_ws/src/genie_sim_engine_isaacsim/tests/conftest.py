# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from pathlib import Path
import sys

_workspace = Path(__file__).resolve().parents[2]
for _package in ("genie_sim_engine_isaacsim", "genie_sim_engine_core", "genie_sim_engine_newton"):
    sys.path.insert(0, str(_workspace / _package))
