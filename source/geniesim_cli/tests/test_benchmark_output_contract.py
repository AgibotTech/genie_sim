# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""The public command owns one artifact contract for every task runtime."""

from pathlib import Path

import pytest
import yaml

from geniesim_cli.commands import benchmark


@pytest.mark.parametrize("runtime", ["isaac", "libuipc", "newton", "mujoco"])
@pytest.mark.parametrize("explicit", [False, True])
def test_platform_output_root_and_common_flags(tmp_path, monkeypatch, runtime, explicit):
    config = tmp_path / "task.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "runtime": runtime,
                "task_name": "wok_flip",
                "benchmark": {"task_name": "task", "sub_task_name": "instruction", "output_dir": "legacy-output"},
            }
        )
    )
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "worker output"))
    monkeypatch.delenv("GENIESIM_BENCHMARK_OUTPUT_ROOT", raising=False)
    monkeypatch.setattr(benchmark, "_build_app_cmd", lambda path, args, host: ["python", *args])
    monkeypatch.setattr(benchmark, "_runtime_environment", lambda *args: None)
    monkeypatch.setattr(benchmark, "_task_runtime", lambda *args: runtime)
    flags = ["--app.headless=true", "--benchmark.record=true", "--benchmark.resume=true"]
    if explicit:
        flags.append("--benchmark.output_dir=" + str(tmp_path / "explicit"))
    command, environment = benchmark._prepared_app_cmd(config, flags, "model:8999")
    root = tmp_path / ("explicit" if explicit else "worker output")
    assert environment["GENIESIM_BENCHMARK_OUTPUT_ROOT"] == str(root)
    assert all(flag in command for flag in flags[:3])
    if runtime != "isaac":
        value = benchmark._extract_flag(command, "benchmark.output_dir" if runtime == "mujoco" else "output-dir")[0]
        assert Path(value) == root / "task" / "instruction"


def test_native_output_names_do_not_require_isaac_task_metadata(tmp_path):
    from geniesim_benchmark.execution import task_output_directory

    result = task_output_directory({"runtime": "newton", "task_name": "wok_flip"}, tmp_path)
    assert result == tmp_path / "wok_flip" / "wok_flip"


@pytest.mark.parametrize("selection", ["default", "worker", "explicit"])
def test_isaac_score_directory_consumes_the_selected_root(tmp_path, monkeypatch, selection):
    from geniesim_benchmark.utils import system_utils

    monkeypatch.setenv("SIM_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    monkeypatch.delenv("GENIESIM_BENCHMARK_OUTPUT_ROOT", raising=False)
    expected = tmp_path / "output/benchmark"
    if selection in ("worker", "explicit"):
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "worker"))
        expected = tmp_path / "worker"
    if selection == "explicit":
        monkeypatch.setenv("GENIESIM_BENCHMARK_OUTPUT_ROOT", str(tmp_path / "explicit"))
        expected = tmp_path / "explicit"
    assert Path(system_utils.benchmark_output_path()) == expected
