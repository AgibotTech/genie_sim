# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, json, time, shutil, subprocess, argparse
from pathlib import Path
from extract_ros_bag import Ros_Extrater
from sim_data_converter import SimDataConverter
import asyncio


def root_before_main(start: Path) -> Path:
    for p in (start, *start.parents):
        if p.name == "main":
            return p.parent.resolve()
    raise FileNotFoundError("no 'main' folder")


CURRENT_DIR = Path(__file__).resolve().parent
SIM_MAIN_ROOT = str(root_before_main(CURRENT_DIR)) + "/"

# Global stats
stats = dict(
    total_bags=0,  # total bags received
    saved=0,  # saved
    discarded=0,  # discarded
    extract_and_converter_ok=0,  # sim_converter
    check_ok=0,  # check passed
    upload_ok=0,  # register_data_to_simubotix success
)
now_task_id = 0
processed_dirs = set()  # track processed absolute paths


def handle_folder(folder, user=[]):
    """
    Process a single numeric folder.
    Returns True/False to indicate whether it was actually processed (for counting).
    """
    abs_path = str(folder.absolute())
    if abs_path in processed_dirs:
        return

    # 1. Skip if already processed (has aligned_joints.h5)
    if (folder / "aligned_joints.h5").exists():
        processed_dirs.add(abs_path)
        return

    # 2. Required file check
    required_files = ("recording_info.json", "metadata.yaml")
    mcap_list = list(folder.glob("*.mcap"))
    if not mcap_list or any(not (folder / f).exists() for f in required_files):
        return

    with open(folder / "recording_info.json", encoding="utf-8") as task_result_json:
        task_result_data = json.load(task_result_json)
        if "teleop_result" not in task_result_data:
            return
        stats["total_bags"] += 1
        current_dir = task_result_data["output_dir"]
        if task_result_data["teleop_result"] == False:
            if current_dir and os.path.isdir(current_dir):
                subprocess.run(["rm", "-rf", current_dir])
                stats["discarded"] += 1
        else:
            display_dir = (current_dir or "").removeprefix("/geniesim/") or (current_dir or "")
            print(f"##############Start converter {display_dir} ....\n", flush=True)
            stats["saved"] += 1
            try:
                ros_extrater = Ros_Extrater(
                    bag_file=current_dir,
                    output_dir=current_dir,
                    task_info=task_result_data,
                )
                asyncio.run(ros_extrater.extract())
            except Exception as e:
                # print(f"##############[EXTRACT-ERROR] {display_dir}: {e}\n")
                processed_dirs.add(current_dir)
                return
            try:
                converter = SimDataConverter(
                    record_path=current_dir,
                    output_path=current_dir,
                    job_id=0,
                    task_id=0,
                    episode_id=0,
                    gripper_names=task_result_data["gripper_names"],
                    robot_type="G2",
                )
                converter.convert()
                stats["extract_and_converter_ok"] += 1
                print(f"##############[CONVERT-OK] {display_dir}\n", flush=True)
            except Exception as e:
                # print(f"##############[CONVERT-ERROR] {display_dir}: {e}\n")
                processed_dirs.add(current_dir)
                return


def scan_loop(root: Path, user: str):
    try:
        while True:
            try:
                for folder in sorted(
                    root.iterdir(),
                    key=lambda p: int(p.name) if p.name.isdigit() else 9999,
                ):
                    if folder.is_dir() and folder.name.isdigit():
                        handle_folder(folder, user)
            except Exception as e:
                print("[SCAN-ERROR]", e)

            time.sleep(10)
    except KeyboardInterrupt:
        print("\n==========  Final stats  ==========")
        print(f"Total bags     : {stats['total_bags']}")
        print(f"Saved          : {stats['saved']}")
        print(f"Discarded      : {stats['discarded']}")
        print(f"Convert ok     : {stats['extract_and_converter_ok']}")
        print(f"User           : {user}")
        print("================================\n")


def wait_for_dir(target_dir: Path, interval: float = 10.0):
    """Block until directory exists, print message every interval seconds."""
    while not target_dir.exists():
        print(f"[WAIT] Directory does not exist yet, retry in {interval}s: {target_dir}")
        time.sleep(interval)
    print(f"[OK] Directory ready: {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="main/output/recording_data/<data_dir>",
    )

    parser.add_argument("--user", type=str, nargs="+", required=True, help="user name")
    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    main_root = script_path.parents[3]
    data_dir = main_root / "output" / "recording_data" / args.data_dir
    wait_for_dir(data_dir)
    scan_loop(data_dir, args.user)
