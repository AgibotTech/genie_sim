# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Managed IPC tasks must load the built client before entering their runner."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from geniesim_cli.commands import benchmark


def test_unsourced_libuipc_launch_can_import_the_real_client(tmp_path, monkeypatch):
    from geniesim_cli.commands import ros

    # A built overlay is present, but the invoking terminal has not sourced it.
    repo = tmp_path / "checkout with spaces"
    (repo / "source/geniesim_ros/src/ros_ws/src/genie_sim_bringup").mkdir(parents=True)
    config = repo / "task.yaml"
    config.write_text("runtime: libuipc\nbenchmark: {task_name: cloth_hanger}\n")
    overlay = repo / "devel"
    marker = overlay / "share/ament_index/resource_index/packages/genie_sim_client"
    marker.parent.mkdir(parents=True)
    marker.touch()
    client_source = Path(__file__).parents[2] / "geniesim_ros/src/ros_ws/src/genie_sim_client"
    (overlay / "setup.bash").write_text(
        f"export AMENT_PREFIX_PATH={shlex.quote(str(overlay))}\n"
        f"export PYTHONPATH={shlex.quote(str(client_source))}\n"
    )
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("GENIESIM_PY_CMD", raising=False)
    monkeypatch.setenv("PYTHONPATH", "")
    monkeypatch.setattr(ros, "_discover_opt_ros_distro", lambda: None)
    monkeypatch.setattr(benchmark, "_rt_python_cmd", lambda: sys.executable)
    cmd, environment = benchmark._prepared_app_cmd(config, [], "localhost:8999")
    result = subprocess.run(
        [cmd[0], "-c", "from genie_sim_client import IPCClient, IPCClientConfig; print(IPCClient.__name__)"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "IPCClient"
    assert "AMENT_PREFIX_PATH" not in os.environ  # Preparation changes only the child.
    assert environment["GENIESIM_PY_CMD"] == cmd[0]


@pytest.mark.parametrize("runtime", ["newton", "libuipc"])
def test_managed_tasks_use_the_prepared_overlay_python(tmp_path, monkeypatch, runtime):
    from geniesim_benchmark.rt_engine import environment

    config = tmp_path / "task.yaml"
    config.write_text(f"runtime: {runtime}\nbenchmark: {{task_name: task}}\n")
    python = tmp_path / "rt_python/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.delenv("GENIESIM_PY_CMD", raising=False)
    monkeypatch.setattr(benchmark, "_rt_python_cmd", lambda: sys.executable)
    monkeypatch.setattr(environment, "runtime_environment", lambda path: {"VIRTUAL_ENV": str(python.parent.parent)})
    command, child = benchmark._prepared_app_cmd(config, [], None)
    assert command[0] == str(python)
    assert child["GENIESIM_PY_CMD"] == str(python)


def test_missing_libuipc_overlay_reports_the_build_command(tmp_path, monkeypatch):
    from geniesim_cli.commands import ros

    config = tmp_path / "task.yaml"
    config.write_text("runtime: libuipc\nbenchmark: {task_name: cloth_hanger}\n")
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
    monkeypatch.setattr(ros, "_discover_opt_ros_distro", lambda: None)
    with pytest.raises(RuntimeError, match="geniesim ros build dev"):
        benchmark._runtime_environment(config, [])


def test_dry_run_needs_no_prepared_overlay(tmp_path, monkeypatch):
    from geniesim_benchmark.rt_engine import environment

    config = tmp_path / "task.yaml"
    config.write_text("runtime: libuipc\nbenchmark: {task_name: cloth_hanger}\n")
    monkeypatch.setattr(environment, "runtime_environment", lambda path: pytest.fail("dry run must not prepare ROS"))
    assert benchmark._runtime_environment(config, ["--dry-run"]) == dict(os.environ)


def test_generated_hidden_config_uses_explicit_checkout_for_overlay(tmp_path, monkeypatch):
    from geniesim_benchmark.rt_engine import environment

    config = tmp_path / "generated.yaml"
    config.write_text("runtime: libuipc\n")
    repo = tmp_path / "checkout"
    calls = []
    monkeypatch.setattr(environment, "runtime_environment", lambda path: calls.append(path) or {"prepared": "yes"})
    child = benchmark._runtime_environment(config, ["--repo-root", str(repo)])
    assert child["prepared"] == "yes"
    assert repo in calls[0].parents
