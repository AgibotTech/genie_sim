# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Support direct execution and module import
try:
    # Try relative import (when imported as module)
    from .filter_rules.data_filter import DataFilter
except ImportError:
    # If relative import fails, use absolute import (when run directly)
    from common.data_filter.filter_rules.data_filter import DataFilter

from common.base_utils.logger import logger


def filter_folder_data(target_folder, config):
    """
    Filter data in specified folder
    :param target_folder: Path to folder to filter
    :param config: Filter configuration dictionary
    :return: (data_valid, result_code) tuple, indicating whether data is valid and result code
    """
    # Check if folder exists
    if not os.path.exists(target_folder):
        logger.info(f"Folder does not exist: {target_folder}")
        error_msg = f"{target_folder} not exit"
        return False, 4, error_msg  # 4 indicates file not found error

    # Check if required files exist
    state_file = os.path.join(target_folder, "state.json")
    data_info_file = os.path.join(target_folder, "data_info.json")

    if not os.path.exists(state_file) or not os.path.exists(data_info_file):
        logger.info(f"Required files do not exist: {state_file} or {data_info_file}")
        error_msg = f"{state_file} or {data_info_file} not exit"
        return False, 4, error_msg

    # Create data filter instance
    data_filter = DataFilter(config)

    try:
        # Execute filtering
        data_valid, result_code, rule_name = data_filter.filter_data(target_folder)
        return data_valid, result_code, rule_name
    except FileNotFoundError as e:
        logger.info(f"File not found error {target_folder}: {str(e)}")
        error_msg = f"{target_folder} not exit"
        return False, 4, error_msg


if __name__ == "__main__":
    # Get absolute path of current script
    current_script_path = os.path.abspath(__file__)

    # Get directory where current script is located
    current_dir = os.path.dirname(current_script_path)

    # Build target folder path
    target_folder = ""

    logger.info(f"Current script path: {current_script_path}")
    logger.info(f"Target folder path: {target_folder}")

    # Check if folder exists
    if os.path.exists(target_folder):
        contents = os.listdir(target_folder)
        logger.info(f"Folder contents {contents}")

        # Example configuration
        config = {
            "filter_rules": [
                {
                    "params": {
                        "camera": "head",
                        "downsample_ratio": 0.2,
                        "gripper": "right",
                        "out_view_allow_time": 0.2,
                    },
                    "result_code": 4,
                    "rule_name": "is_gripper_in_view",
                },
                {
                    "params": {
                        "camera": "head",
                        "downsample_ratio": 0.2,
                        "gripper": "left",
                        "out_view_allow_time": 0.1,
                    },
                    "result_code": 4,
                    "rule_name": "is_gripper_in_view",
                },
                {
                    "params": {
                        "objects": ["brain_benchmark_building_blocks_003"],
                        "target": "brain_benchmark_building_blocks_000",
                        "relative_position_range": [[-0.07, 0.07], [-0.07, 0.07], [-0.07, 0.07]],
                    },
                    "result_code": 1,
                    "rule_name": "is_object_relative_position_in_target",
                },
            ]
        }

        # Execute filtering
        data_valid, result_code, _ = filter_folder_data(target_folder, config)

        logger.info("\nFilter results:")
        logger.info(f"Data valid: {data_valid}")
        logger.info(f"Result code: {result_code}")
    else:
        logger.info("Folder does not exist")
