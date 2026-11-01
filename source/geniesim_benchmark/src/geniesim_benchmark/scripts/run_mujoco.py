# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Native MuJoCo entry point; never imports Isaac Sim or ROS."""

from pathlib import Path
import sys
import os

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geniesim_benchmark.backends.mujoco.rubik.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
