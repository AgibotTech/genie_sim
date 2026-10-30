# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Prepare MC using the canonical Wuji URDF."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from geniesim_teleop.start_mc import prepare_mc
from geniesim_teleop.wuji.contracts import HandCommands, load_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-urdf", type=Path, required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    HandCommands(args.model_urdf)
    task = replace(load_task(args.config), urdf_path=args.model_urdf.resolve())
    profile = prepare_mc(task, cache_root=args.cache_dir)
    os.execv(
        "/bin/bash",
        [
            "bash",
            profile["mc_launcher"],
            f"--robot={profile['mc_robot']}",
            "-C",
            profile["mc_configuration"],
            "--no-tool",
        ],
    )


if __name__ == "__main__":
    main()
