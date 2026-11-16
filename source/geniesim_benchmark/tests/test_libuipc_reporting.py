# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Preserve measured IPC episode wall time in the common evaluation report."""

import copy
import json

import pytest

from geniesim_benchmark.libuipc.reporting import common_result
from geniesim_benchmark.plugins.output_system.dexhand2 import EVALUATION_FILENAME, write_summary


TASKS = ("cloth_hanger", "fold_cloth", "drawstring_bag")


def execution(seconds, *, status="completed"):
    return {
        "execution_status": status,
        "scoring_status": "scored" if status == "completed" else "pending",
        "wall_seconds": seconds,
        "executed_sim_time_s": 2.0,
        "steps": 60,
        "inference_requests": 2,
    }


@pytest.mark.parametrize("task", TASKS)
def test_episode_duration_preserves_wall_time_including_inference_wait(task):
    measured = execution(335.2059958311729)
    original = copy.deepcopy(measured)

    report = common_result(task, measured, {})

    assert report["duration"] == measured["wall_seconds"]
    assert report["execution"]["executed_sim_time_s"] == 2.0
    assert report["execution"] == original
    assert measured == original


@pytest.mark.parametrize("task", TASKS)
def test_summary_sums_wall_time_of_valid_episodes_only(task, tmp_path):
    reports = [
        common_result(task, execution(12.5), {}),
        common_result(task, execution(7.25), {}),
        common_result(task, execution(100.0, status="failed"), {}),
    ]

    write_summary(tmp_path, reports)
    saved = json.loads((tmp_path / EVALUATION_FILENAME).read_text())

    assert saved["statistics"]["timecost"] == 19.75
    assert [row["duration"] for row in saved["details"]] == [12.5, 7.25, 100.0]
    assert [row["valid"] for row in saved["details"]] == [True, True, False]


@pytest.mark.parametrize("task", TASKS)
def test_failure_before_episode_retains_zero_duration_and_invalid_status(task, tmp_path):
    failure = {"execution_status": "failed", "scoring_status": "pending", "error": "warmup failed"}

    report = common_result(task, failure, {})
    write_summary(tmp_path, [report])
    saved = json.loads((tmp_path / EVALUATION_FILENAME).read_text())

    assert report["duration"] == 0.0
    assert report["valid"] is False
    assert saved["statistics"]["timecost"] == 0.0
    assert "wall_seconds" not in failure
