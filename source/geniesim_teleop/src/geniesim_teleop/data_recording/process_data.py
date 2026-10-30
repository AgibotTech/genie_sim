# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import asyncio
import json
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path

from geniesim_teleop.data_recording.sim_data_converter import SimDataConverter, export_complete

stats = dict(
    total_bags=0,
    saved=0,
    discarded=0,
    extract_and_converter_ok=0,
)
processed_dirs = set()  # track processed absolute paths


def handle_folder(folder):
    """
    Process a single numeric folder.
    Returns True/False to indicate whether it was actually processed (for counting).
    """
    abs_path = str(folder.absolute())
    if abs_path in processed_dirs:
        return

    required_files = ("recording_info.json", "metadata.yaml")
    if any(not (folder / f).exists() for f in required_files):
        return
    task_result_data = json.loads((folder / "recording_info.json").read_text(encoding="utf-8"))
    if "teleop_result" not in task_result_data:
        return  # The supervisor writes this only after the recorder finishes.
    is_wuji = "wuji" in task_result_data.get("robot_name", "").lower() or any(
        "wuji" in name.lower() for name in task_result_data.get("gripper_names", [])
    )
    if export_complete(folder):
        processed_dirs.add(abs_path)
        return
    current_dir = str(folder.resolve())
    if task_result_data["teleop_result"] is False:
        shutil.rmtree(folder)
        stats["total_bags"] += 1
        stats["discarded"] += 1
        processed_dirs.add(abs_path)
        return

    bag_files = sorted([*folder.glob("*.mcap"), *folder.glob("*.db3")])
    extracted = all((folder / name).is_file() for name in ("state.json", "aligned_joints_all.h5"))
    extraction_marker = folder / "post_proc/teleop_extraction.json"
    bag_signature = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in bag_files}
    if bag_files and (not is_wuji or extraction_marker.exists()):
        # Re-extract changed bags or legacy ordinary joint layouts. Legacy Wuji
        # intermediates already retain named fingers and can finish without ROS.
        try:
            extracted = extracted and json.loads(extraction_marker.read_text()) == bag_signature
        except (OSError, ValueError):
            extracted = False
    if not bag_files and not extracted:
        return
    print(f"##############Start converter {current_dir} ....\n", flush=True)
    if not extracted:
        try:
            from geniesim_teleop.data_recording.extract_ros_bag import Ros_Extrater

            # Isolate chunks and images from prior failed extraction attempts.
            with tempfile.TemporaryDirectory(prefix=".teleop-extract-", dir=folder) as work:
                asyncio.run(Ros_Extrater(current_dir, work, task_result_data).extract())
                if bag_signature != {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in bag_files}:
                    raise RuntimeError("Source bag changed during extraction")
                for artifact in Path(work).iterdir():
                    destination = folder / artifact.name
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    os.replace(artifact, destination)
            extraction_marker.parent.mkdir(parents=True, exist_ok=True)
            extraction_marker.write_text(json.dumps(bag_signature))
        except (Exception, SystemExit) as e:
            print(f"##############[EXTRACT-ERROR] {current_dir}: {e}\n", flush=True)
            return  # Retain inputs and retry on the next scan.
    try:
        converter = SimDataConverter(
            record_path=current_dir,
            output_path=current_dir,
            job_id=task_result_data.get("job_id", 0),
            task_id=task_result_data.get("task_id", 0),
            episode_id=task_result_data.get("episode_id", int(folder.name)),
            gripper_names=task_result_data["gripper_names"],
            robot_type=task_result_data.get("robot_name", "G2"),
        )
        converter.convert()
    except Exception as e:
        print(f"##############[CONVERT-ERROR] {current_dir}: {e}\n", flush=True)
        return
    stats["total_bags"] += 1
    stats["saved"] += 1
    stats["extract_and_converter_ok"] += 1
    processed_dirs.add(abs_path)
    print(f"##############[CONVERT-OK] {current_dir}\n", flush=True)
    return True


def scan_loop(root: Path, user: str):
    try:
        while True:
            try:
                for folder in sorted(
                    root.iterdir(),
                    key=lambda p: int(p.name) if p.name.isdigit() else 9999,
                ):
                    if folder.is_dir() and folder.name.isdigit():
                        handle_folder(folder)
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
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, signal.default_int_handler)
    # Recordings land under $SIM_REPO_ROOT/output/recording_data (same root the
    # simulator's recording_output_path() writes to). Fall back to walking up
    # from this file to the repo root if the env var isn't set.
    repo_root = os.environ.get("SIM_REPO_ROOT")
    if repo_root:
        main_root = Path(repo_root)
    else:
        # .../<repo>/source/geniesim_teleop/src/geniesim_teleop/data_recording/process_data.py
        main_root = Path(__file__).resolve().parents[5]
    data_dir = main_root / "output" / "recording_data" / args.data_dir
    try:
        wait_for_dir(data_dir)
        scan_loop(data_dir, args.user)
    except KeyboardInterrupt:
        pass
