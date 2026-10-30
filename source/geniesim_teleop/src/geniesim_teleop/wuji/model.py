# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Generate the canonical Wuji URDF in the session directory."""

import os
from pathlib import Path
import subprocess
import tempfile


def generate_urdf(task, destination):
    model = task.benchmark_root.parents[3] / "source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model"
    destination = Path(destination).resolve()
    with tempfile.TemporaryDirectory(prefix="wuji-model-") as directory:
        prefix = Path(directory)
        marker = prefix / "share/ament_index/resource_index/packages/genie_sim_robot_model"
        marker.parent.mkdir(parents=True)
        marker.touch()
        (prefix / "share/genie_sim_robot_model").symlink_to(model, target_is_directory=True)
        env = os.environ.copy()
        env["AMENT_PREFIX_PATH"] = str(prefix) + os.pathsep + env.get("AMENT_PREFIX_PATH", "")
        subprocess.run(
            [
                "xacro",
                str(model / "robots/genie/g2/g2_crsB_wuji.urdf.xacro"),
                "-o",
                str(destination),
            ],
            env=env,
            check=True,
        )
    return destination
