# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from enum import Enum

from geniesim_assets import ASSETS_PATH
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()  # Create singleton instance

PKG_ROOT_REL = "source/geniesim_benchmark/src/geniesim_benchmark"


def _pkg_root():
    return os.path.join(os.environ["SIM_REPO_ROOT"], PKG_ROOT_REL)


def check_and_fix_env():
    # check SIM_REPO_ROOT
    env_root_path = os.getenv("SIM_REPO_ROOT")
    if not env_root_path:
        # system_utils.py lives at <repo>/source/geniesim_benchmark/src/geniesim_benchmark/utils/
        # so the repo root is 6 parents up.
        env_root_path = Path(__file__).resolve().parents[5]
        os.environ["SIM_REPO_ROOT"] = env_root_path.as_posix()
        logger.warning(f"Warning: env [SIM_REPO_ROOT] empty, will use default: {env_root_path}")
    else:
        logger.info(f"using env SIM_REPO_ROOT={env_root_path}")

    if not os.path.exists(env_root_path):
        os.makedirs(env_root_path, exist_ok=True)

    # check SIM_ASSETS. The CLI and container entrypoint export the canonical
    # GENIESIM_ASSETS_PATH; SIM_ASSETS is the legacy name this package reads.
    assets_path = os.getenv("SIM_ASSETS") or os.getenv("GENIESIM_ASSETS_PATH")
    if not assets_path:
        assets_path = os.path.join(os.path.expanduser("~"), "assets")
        logger.warning(f"Warning: env [SIM_ASSETS] empty, will use default: {assets_path}")
    else:
        logger.info(f"using env SIM_ASSETS={assets_path}")
    os.environ["SIM_ASSETS"] = assets_path


def config_path():
    return os.path.join(_pkg_root(), "config")


def benchmark_conf_path():
    return os.path.join(_pkg_root(), "benchmark/config")


def tgs_conf_path():
    return os.path.join(_pkg_root(), "plugins/tgs/config")


def app_root_path():
    return os.path.join(_pkg_root(), "app")


def assets_path():
    # SIM_ASSETS overrides the packaged assets (e.g. a mounted checkout in a
    # control container); otherwise assets come from the `geniesim_assets` pip package.
    configured_path = os.getenv("SIM_ASSETS")
    if configured_path:
        return configured_path
    return ASSETS_PATH


def benchmark_root_path():
    return os.path.join(_pkg_root(), "benchmark")


def generator_path():
    # Note: no in-tree directory currently matches; kept for API stability.
    return os.path.join(_pkg_root(), "generator")


def teleop_root_path():
    return os.path.join(_pkg_root(), "teleop")


def benchmark_output_path():
    return os.path.join(os.environ["SIM_REPO_ROOT"], "output/benchmark")


def recording_output_path():
    return os.path.join(os.environ["SIM_REPO_ROOT"], "output/recording_data")


def plugins_ader_path():
    return os.path.join(_pkg_root(), "plugins/ader")


def benchmark_layout_path():
    # No in-tree directory currently matches this path.
    return os.path.join(_pkg_root(), "layout")


def benchmark_task_definitions_path():
    return os.path.join(_pkg_root(), "benchmark/config/task_definitions")


def load_json(json_file):
    if not os.path.exists(json_file):
        raise ValueError("Json file not found: {}".format(json_file))
    with open(json_file) as f:
        return json.load(f)
    return None


def generate_new_file_path(dir_path, prefix_name, suffix="json"):
    count = 0
    while os.path.exists(os.path.join(dir_path, f"{prefix_name}_{count:02d}.{suffix}")):
        count += 1
    new_filename = f"{prefix_name}_{count:02d}.{suffix}"
    return os.path.join(dir_path, new_filename)


def TIMENOW():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_local_time():
    tz_utc8 = timezone(timedelta(hours=8))
    utc_now = datetime.now(tz_utc8)
    return str(utc_now)


def ConvertEnum2Int(code: Enum):
    return int(code.value)
