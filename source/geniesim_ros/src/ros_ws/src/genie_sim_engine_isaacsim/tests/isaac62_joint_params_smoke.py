#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run the runtime sysid regressions with native Isaac 6.2 schemas loaded."""

from pathlib import Path
import sys


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "disable_viewport_updates": True})
    code = 1
    try:
        import pytest

        code = pytest.main(["-q", "--tb=short", str(Path(__file__).with_name("test_runtime_joint_params.py"))])
    finally:
        app.close(exit_code=int(code))
    return code


if __name__ == "__main__":
    sys.exit(main())
