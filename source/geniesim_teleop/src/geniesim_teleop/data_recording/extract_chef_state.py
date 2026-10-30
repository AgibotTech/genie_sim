# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Compatibility entrypoint for the Wuji chef deformable recording profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from geniesim_teleop.data_recording.deformable_state import CHEF_TOPIC_PREFIX
from geniesim_teleop.data_recording.extract_deformable_state import (
    extract_deformable_state,
)

DEFAULT_SCENE_NAME = "scene_flat_g2_sp_chef_wuji"
DEFAULT_SCENE_USD = "background/home/fried_noodles/background.usda"
DEFAULT_ROBOT_NAME = "G2_crsB_wuji"
DEFAULT_DATA_VERSION = "chef_deformable_v3_world"


def extract_chef_state(
    bag_path: str | Path,
    output_dir: str | Path,
    *,
    scene_name: str = DEFAULT_SCENE_NAME,
    scene_usd: str = DEFAULT_SCENE_USD,
    robot_name: str = DEFAULT_ROBOT_NAME,
    robot_root_frame: str = "base_link",
    output_fps: float = 30.0,
) -> list[Path]:
    """Convert a Wuji chef bag through the generic deformable converter."""
    return extract_deformable_state(
        bag_path,
        output_dir,
        scene_name=scene_name,
        scene_usd=scene_usd,
        robot_name=robot_name,
        topic_prefix=CHEF_TOPIC_PREFIX,
        robot_root_frame=robot_root_frame,
        output_fps=output_fps,
        sim_raw_data_version=DEFAULT_DATA_VERSION,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scene-name", default=DEFAULT_SCENE_NAME)
    parser.add_argument("--scene-usd", default=DEFAULT_SCENE_USD)
    parser.add_argument("--robot-name", default=DEFAULT_ROBOT_NAME)
    parser.add_argument("--robot-root-frame", default="base_link")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    outputs = extract_chef_state(
        args.bag,
        args.output,
        scene_name=args.scene_name,
        scene_usd=args.scene_usd,
        robot_name=args.robot_name,
        robot_root_frame=args.robot_root_frame,
        output_fps=args.fps,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
