# Copyright (c) 2023-2026, AgiBot Inc.
# SPDX-License-Identifier: MPL-2.0
"""Task metric evaluation for collection recordings."""

from common.base_utils.logger import logger
from common.data_filter.check_collected_data import filter_folder_data


def evaluate_collection_metric(data_dir, metric_config):
    """Return (valid, result_code, status) without format-checker coupling."""
    if not metric_config.get("filter_rules"):
        return True, 0, "no filter_rules configured"
    try:
        return filter_folder_data(str(data_dir), metric_config)
    except Exception as exc:
        logger.exception(f"Task metric evaluation failed for {data_dir}")
        return False, 1, f"metric evaluation failed: {exc}"
