# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import asyncio
import json
import sys

from extract_ros_bag import RosExtrater
from recording.sim_data_converter import SimDataConverter

from common.base_utils.logger import logger
from common.data_filter.check_collected_data import filter_folder_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and convert data from ROS bag file")
    parser.add_argument("--path_to_save", type=str, help="Path to the ROS bag file")
    parser.add_argument("--task_info_path", type=str, help="Path to the task_info")
    parser.add_argument("--metric_config_path", type=str, help="Path to the metric config file")
    args = parser.parse_args()

    # Extract data from ROS bag file
    path_to_save = args.path_to_save
    task_info_path = args.task_info_path

    metric_config_path = args.metric_config_path
    with open(task_info_path, "r") as f:
        task_info = json.load(f)
    gripper_names = task_info.get("gripper_names")
    robot_name = task_info.get("robot_name")
    extract_ros = RosExtrater(bag_file=path_to_save, output_dir=path_to_save, task_info=task_info)
    try:
        asyncio.run(extract_ros.extract())
    except Exception as e:
        logger.error(f"Error in extracting data{e}")
        sys.exit()

    converter = SimDataConverter(
        path_to_save,
        path_to_save,
        0,
        0,
        0,
        gripper_names,
        robot_name,
    )
    converter.convert()

    with open(metric_config_path, "r") as f:
        metric_config_info = json.load(f)
    data_valid, result_code, status = filter_folder_data(path_to_save, metric_config_info)
    isSuccess = data_valid
    logger.info(f"Status: {status} (Result Code: {result_code})")

    result = {
        "task_name": task_info.get("task_name"),
        "fail_stage_step": task_info.get("failStep"),
        "fps": task_info.get("fps"),
        "task_status": isSuccess,
        "camera_info": task_info.get("camera_info_list"),
        "return_code": result_code,
        "metric_status": status,
        "playback_times": len(task_info.get("playback_timerange")),
    }
    with open(
        path_to_save + "/task_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, ensure_ascii=False)
