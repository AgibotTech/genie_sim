# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Teleop selects the prepared Isaac 6.2 runtime without starting devices."""

import os
from pathlib import Path
import subprocess

import pytest

from geniesim_cli.commands import teleop


def test_defaults_to_python312_even_when_legacy_wrapper_exists(monkeypatch):
    monkeypatch.delenv("GENIESIM_PY_CMD", raising=False)
    monkeypatch.setattr(
        teleop.shutil, "which", lambda name: "/usr/bin/python3.12" if name == "python3.12" else "omni_python"
    )
    assert teleop._python_cmd() == "/usr/bin/python3.12"


def test_explicit_prepared_interpreter_is_retained(monkeypatch):
    monkeypatch.setenv("GENIESIM_PY_CMD", "/prepared/bin/python")
    assert teleop._python_cmd() == "/prepared/bin/python"


def test_missing_system_runtime_does_not_fall_back_to_legacy(monkeypatch):
    monkeypatch.delenv("GENIESIM_PY_CMD", raising=False)
    monkeypatch.setattr(teleop.shutil, "which", lambda name: "omni_python" if name == "omni_python" else None)
    with pytest.raises(RuntimeError, match="Python 3.12"):
        teleop._python_cmd()


@pytest.mark.parametrize("flags", [[], ["--isaac-version", "6.2"], ["--isaac-version=6.2"]])
def test_host_scripts_select_same_container_and_forward_control_flags(flags):
    script = Path(__file__).resolve().parents[2] / "geniesim_teleop/scripts/teleop_runtime.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; shift; teleop_select_runtime "$@" || exit $?; printf "%s\\n" "$TELEOP_DEFAULT_CONTAINER" "$TELEOP_DEFAULT_SIM_PY" "${TELEOP_RUNTIME_ARGS[@]}"',
            "teleop-test",
            str(script),
            *flags,
            "--external-simulator",
        ],
        env={key: value for key, value in os.environ.items() if key != "GENIESIM_TELEOP_ISAAC_VERSION"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == ["geniesim4", "python3.12", "--external-simulator"]


def test_host_scripts_reject_retired_runtime():
    script = Path(__file__).resolve().parents[2] / "geniesim_teleop/scripts/teleop_runtime.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; teleop_select_runtime --isaac-version 5.1', "teleop-test", str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "teleop requires 6.2" in result.stderr
