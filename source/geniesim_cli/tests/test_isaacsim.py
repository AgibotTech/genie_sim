# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Image respins must resolve to an available wheel for the target Python."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from geniesim_cli import _isaacsim, cli
from geniesim_cli.commands import docker


def _indices(monkeypatch, wheels):
    monkeypatch.setattr(_isaacsim, "ngc_tags", lambda timeout: ["6.2.0", "6.2.0.1"])
    monkeypatch.setattr(_isaacsim, "pypi_wheels", lambda timeout: wheels)


def test_requested_respin_uses_published_wheel(monkeypatch):
    _indices(monkeypatch, {"6.2.0.0": {"cp312"}})
    assert _isaacsim.resolve("6.2.0.1", python_tag="cp312") == ("6.2.0.1", "6.2.0.0")
    assert _isaacsim.resolve("6.X", python_tag="cp312") == ("6.2.0.1", "6.2.0.0")


def test_explicit_original_tag_does_not_select_the_respin(monkeypatch):
    _indices(monkeypatch, {"6.2.0.0": {"cp312"}})
    assert _isaacsim.resolve("6.2.0", python_tag="cp312") == ("6.2.0", "6.2.0.0")


def test_respin_does_not_use_an_unavailable_or_other_wheel(monkeypatch):
    _indices(monkeypatch, {"5.1.0.0": {"cp312"}, "6.2.0.1": {"cp312"}})
    assert _isaacsim.resolve("6.2.0.1", python_tag="cp312") is None


def test_respin_rejects_the_wrong_python_wheel(monkeypatch):
    _indices(monkeypatch, {"6.2.0.0": {"cp311"}})
    assert _isaacsim.resolve("6.2.0.1", python_tag="cp312") is None


def test_other_respins_do_not_inherit_the_pairing():
    assert not _isaacsim._wheel_matches_tag("6.2.0.0", "6.2.0.2")
    assert _isaacsim._wheel_matches_tag("6.2.0.2", "6.2.0.2")
    assert _isaacsim._wheel_matches_tag("5.1.0.0", "5.1.0")


def test_docker62_build_dispatches_requested_pair_without_bootstrap(monkeypatch, capsys):
    _indices(monkeypatch, {"6.2.0.0": {"cp312"}})
    monkeypatch.delenv("GENIESIM_ISAACSIM_TAG", raising=False)
    monkeypatch.delenv("GENIESIM_ISAACSIM_PIP_VERSION", raising=False)
    monkeypatch.setattr(docker, "_active_variant", "5.1")
    monkeypatch.setattr(docker, "_invoked_as", "docker")
    monkeypatch.setattr(docker, "_ensure_docker_cli", lambda: None)
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(docker, "_repo_root", lambda: root)
    monkeypatch.setattr(docker, "_image", lambda: "geniesim-test:6.2")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(docker.subprocess, "run", run)

    def find_spec(name):
        pytest.fail(f"Docker commands must not probe siblings or trigger bootstrap: {name}")

    monkeypatch.setattr("importlib.util.find_spec", find_spec)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["docker6.2", "build"])
    assert exit_info.value.code == 0
    assert docker._active_variant == ""
    assert "geniesim docker6.2 build" in capsys.readouterr().out
    assert docker.get_variant("").isaacsim_series == "6.2.0.1"
    assert "ISAACSIM_TAG=6.2.0.1" in commands[0]
    assert "ISAACSIM_PIP_VERSION=6.2.0.0" in commands[0]


def test_offline_pin_keeps_requested_base(monkeypatch):
    monkeypatch.delenv("GENIESIM_ISAACSIM_TAG", raising=False)
    monkeypatch.delenv("GENIESIM_ISAACSIM_PIP_VERSION", raising=False)
    monkeypatch.setattr(docker, "_active_variant", "")
    monkeypatch.setattr(_isaacsim, "ngc_tags", lambda timeout: [])
    monkeypatch.setattr(_isaacsim, "pypi_wheels", lambda timeout: {})
    assert docker._isaacsim_build_args("6.2.0.1") == []
    root = Path(__file__).resolve().parents[3]
    dockerfile = (root / "docker/Dockerfile").read_text().splitlines()
    assert "ARG ISAACSIM_TAG=6.2.0.1" in dockerfile
    assert "ARG ISAACSIM_PIP_VERSION=6.2.0.0" in dockerfile


@pytest.mark.parametrize("command", ["docker", "docker5.1"])
def test_default_and_explicit_docker51_keep_the_combo(monkeypatch, command):
    calls = []
    monkeypatch.setattr(docker, "run", lambda args, **kwargs: calls.append((args, kwargs)))

    def find_spec(name):
        pytest.fail(f"Docker commands must not probe siblings or trigger bootstrap: {name}")

    monkeypatch.setattr("importlib.util.find_spec", find_spec)
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "build"])
    assert exit_info.value.code == 0
    assert calls == [(["build"], {"variant": "5.1", "invoked_as": command})]
    variant = docker.get_variant("5.1")
    assert variant.isaacsim_series is None
    assert variant.env["GENIESIM_PY_CMD"] == "omni_python"
    root = Path(__file__).resolve().parents[3]
    dockerfile = (root / "docker" / variant.dockerfile).read_text().splitlines()
    assert "ARG ISAACSIM_TAG=5.1.0" in dockerfile
    assert "ARG ISAACSIM_PIP_VERSION=6.2.0.0" in dockerfile
