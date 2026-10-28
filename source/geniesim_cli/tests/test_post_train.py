# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise the post-training CLI through the real process and script boundaries."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "source/rlinf_geniesim/scripts"


_RUNTIME_SCRIPTS = (
    "start_sim_servers.sh",
    "start_sim_server.sh",
    "_server_slot.sh",
    "_ws_build_stamp.sh",
    "host_run_teleop.sh",
)
requires_runtime = pytest.mark.skipif(
    not all((SCRIPTS / name).is_file() for name in _RUNTIME_SCRIPTS),
    reason="requires post-training runtime launchers",
)


@pytest.fixture
def runtime(tmp_path):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GENIESIM_", "RLINF_", "TELEOP_", "SIM_"))
    }
    env.update(
        PYTHONPATH=str(ROOT / "source/geniesim_cli/src"),
        GENIESIM_REPO_ROOT=str(ROOT),
        RLINF_RUN_ROOT=str(tmp_path / "runtime with spaces"),
        NO_COLOR="1",
    )
    return env


def invoke(runtime, *args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, "-S", "-m", "geniesim_cli", "post-train", *args],
        env=runtime,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    "args", [[], ["--help"], ["server"], ["server", "start", "--help"], ["server", "logs", "--help"]]
)
def test_help_needs_no_checkout_or_simulator(runtime, tmp_path, args):
    runtime["GENIESIM_REPO_ROOT"] = str(tmp_path / "missing")
    result = invoke(runtime, *args, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "geniesim post-train" in result.stdout
    assert "bootstrap" not in result.stdout


@requires_runtime
def test_start_uses_batch_generation_for_multiple_gpus(runtime):
    result = invoke(runtime, "server", "start", "--count", "2", "--base-port", "62011", "--devices", "0,1", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert any("--port 62011" in line and "--device 0" in line for line in lines)
    assert any("--port 62012" in line and "--device 1" in line for line in lines)
    assert not Path(runtime["RLINF_RUN_ROOT"]).exists()


@requires_runtime
def test_start_preserves_plan_filename_with_spaces(runtime, tmp_path):
    plan = tmp_path / "server plan.conf"
    plan.write_text("server port=62011 gpu=2\nserver port=62012 gpu=3\n")
    result = invoke(runtime, "server", "start", str(plan), "--only", "62012", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--port 62012" in result.stdout and "--device 3" in result.stdout
    assert "--port 62011" not in result.stdout


@pytest.mark.parametrize("action", ["status", "stop"])
@requires_runtime
def test_lifecycle_only_operates_selected_slots(runtime, action):
    records = []
    for port in range(62011, 62014):
        record = Path(runtime["RLINF_RUN_ROOT"]) / f"rlinf_srv_p{port}/sim_server.pid"
        record.parent.mkdir(parents=True)
        record.write_text("invalid\n")
        records.append(record)
    result = invoke(runtime, "server", action, "--count", "3", "--base-port", "62011", "--only", "62011,62013")
    assert result.returncode == 0, result.stdout + result.stderr
    assert records[1].exists()
    assert records[0].exists() == (action == "status")
    assert records[2].exists() == (action == "status")
    if action == "status":
        assert "p62011 stopped" in result.stdout and "p62013 stopped" in result.stdout
        assert "p62012" not in result.stdout


@pytest.mark.parametrize(
    "action, flag", [("start", "--stop"), ("start", "--status"), ("status", "--stop"), ("stop", "--status")]
)
def test_action_flags_cannot_change_the_selected_operation(runtime, action, flag):
    result = invoke(runtime, "server", action, "--count", "1", flag, "--dry-run")
    assert result.returncode == 2
    assert "action" in result.stderr.lower()
    assert not Path(runtime["RLINF_RUN_ROOT"]).exists()


@requires_runtime
def test_batch_validation_exit_code_is_preserved(runtime):
    result = invoke(runtime, "server", "start", "--count", "0", "--dry-run")
    assert result.returncode == 2
    assert not Path(runtime["RLINF_RUN_ROOT"]).exists()


@pytest.mark.parametrize("action", ["status", "stop"])
@pytest.mark.parametrize(
    "option", ["--count", "--base-port", "--devices", "--code", "--assets", "--only", "--barrier-cap-s"]
)
@requires_runtime
def test_lifecycle_action_cannot_be_consumed_as_a_missing_option_value(runtime, action, option):
    result = invoke(runtime, "server", action, "--count", "1", "--dry-run", option)
    assert result.returncode == 2
    assert f"Missing value for {option}" in result.stderr
    assert not Path(runtime["RLINF_RUN_ROOT"]).exists()


def test_missing_explicit_checkout_is_not_replaced_by_cwd(runtime, tmp_path):
    runtime["GENIESIM_REPO_ROOT"] = str(tmp_path / "not a checkout")
    result = invoke(runtime, "server", "status", "--count", "1")
    assert result.returncode != 0
    assert "not a checkout" in result.stderr


@requires_runtime
def test_checkout_can_be_discovered_from_a_nested_directory(runtime, tmp_path):
    runtime.pop("GENIESIM_REPO_ROOT")
    checkout = tmp_path / "checkout with spaces"
    target = checkout / "source/rlinf_geniesim/scripts"
    target.mkdir(parents=True)
    for name in ("start_sim_servers.sh", "start_sim_server.sh", "_server_slot.sh", "_ws_build_stamp.sh"):
        shutil.copy2(SCRIPTS / name, target / name)
    result = invoke(runtime, "server", "status", "--count", "1", cwd=target)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "p5601 stopped" in result.stdout


@requires_runtime
def test_logs_use_the_slot_path_and_last_100_lines(runtime):
    log = Path(runtime["RLINF_RUN_ROOT"]) / "rlinf_srv_p62011/sim.log"
    log.parent.mkdir(parents=True)
    log.write_text("".join(f"line {number}\n" for number in range(150)))
    result = invoke(runtime, "server", "logs", "--port", "62011")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [f"line {number}" for number in range(50, 150)]
    assert list(log.parent.iterdir()) == [log]


@requires_runtime
def test_missing_logs_fail_without_creating_a_slot(runtime):
    result = invoke(runtime, "server", "logs", "--port", "62011")
    assert result.returncode != 0
    assert "log" in result.stderr.lower()
    assert not Path(runtime["RLINF_RUN_ROOT"]).exists()


@pytest.mark.parametrize("port", ["0", "65536", "../other"])
def test_logs_reject_invalid_ports(runtime, port):
    result = invoke(runtime, "server", "logs", "--port", port)
    assert result.returncode == 2
    assert not Path(runtime["RLINF_RUN_ROOT"]).exists()


@requires_runtime
def test_follow_streams_new_lines_and_preserves_interrupts(runtime, tmp_path):
    log = Path(runtime["RLINF_RUN_ROOT"]) / "rlinf_srv_p62011/sim.log"
    log.parent.mkdir(parents=True)
    log.write_text("first\n")
    captured = tmp_path / "follow.log"
    with captured.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-S", "-m", "geniesim_cli", "post-train", "server", "logs", "--port", "62011", "--follow"],
            env=runtime,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for expected in ("first", "second"):
                deadline = time.monotonic() + 5
                while expected not in captured.read_text() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert expected in captured.read_text(), captured.read_text()
                if expected == "first":
                    with log.open("a") as stream:
                        stream.write("second\n")
            os.killpg(process.pid, signal.SIGINT)
            assert process.wait(timeout=3) in (-signal.SIGINT, 130)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)


@requires_runtime
def test_teleop_preserves_arguments_source_path_and_exit_code(runtime, tmp_path):
    captured = tmp_path / "teleop.json"
    interpreter = tmp_path / "python with spaces"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps([sys.argv[1:],os.environ['PYTHONPATH'],os.environ['SIM_REPO_ROOT']]))\n"
        "sys.exit(17)\n"
    )
    interpreter.chmod(0o755)
    marker = tmp_path / "must-not-exist"
    args = ["--task", f"literal $(touch {marker})", "--addr", "tcp://server:5601"]
    runtime.update(TELEOP_PYTHON=str(interpreter), CAPTURE=str(captured), SIM_REPO_ROOT="/stale/checkout")
    result = invoke(runtime, "teleop", *args)
    assert result.returncode == 17, result.stdout + result.stderr
    argv, pythonpath, repo = json.loads(captured.read_text())
    assert argv == [str(SCRIPTS / "teleop_pico_debug.py"), *args]
    assert str(ROOT / "source/geniesim_teleop/src") in pythonpath.split(":")
    assert repo == str(ROOT)
    assert not marker.exists()


@pytest.mark.parametrize("args", [["unknown"], ["server", "unknown"]])
def test_unknown_commands_are_errors(runtime, args):
    result = invoke(runtime, *args)
    assert result.returncode != 0
    assert "unknown" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize(
    "command, script, forwarded",
    [
        (["server", "start", "--count", "2", "--dry-run"], "start_sim_servers.sh", ["--count", "2", "--dry-run"]),
        (["server", "status", "--count", "2"], "start_sim_servers.sh", ["--status", "--count", "2"]),
        (["server", "stop", "--only", "5601"], "start_sim_servers.sh", ["--stop", "--only", "5601"]),
        (
            ["server", "logs", "--port", "5602", "--follow"],
            "start_sim_server.sh",
            ["--port", "5602", "--logs", "--follow"],
        ),
        (["teleop", "--task", "literal $(touch marker)"], "host_run_teleop.sh", ["--task", "literal $(touch marker)"]),
    ],
)
def test_dispatch_preserves_arguments_and_exit_status_without_runtime(runtime, tmp_path, command, script, forwarded):
    checkout = tmp_path / "checkout with spaces"
    scripts = checkout / "source/rlinf_geniesim/scripts"
    scripts.mkdir(parents=True)
    (scripts / script).write_text(
        "#!/usr/bin/env bash\n"
        """"$TEST_PYTHON" -c 'import json,os,sys; print(json.dumps([sys.argv[1:],os.environ["SIM_REPO_ROOT"]])); sys.exit(17)' "$@"
"""
    )
    runtime.update(GENIESIM_REPO_ROOT=str(checkout), TEST_PYTHON=sys.executable)
    result = invoke(runtime, *command)
    assert result.returncode == 17, result.stdout + result.stderr
    assert json.loads(result.stdout) == [forwarded, str(checkout)]
    assert not (checkout / "marker").exists()
