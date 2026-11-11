# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Public variant values and CLI build options retain their contracts."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from geniesim_cli import _isaacsim
from geniesim_cli.commands import docker


def test_variant_env_is_copied_frozen_and_hashable():
    env = {"ROS_DISTRO": "jazzy"}
    variant = docker.Variant("Dockerfile", "image", "container", env)
    env["ROS_DISTRO"] = "humble"
    assert variant.env["ROS_DISTRO"] == "jazzy"
    assert {variant: "value"}[replace(variant)] == "value"
    assert variant != replace(variant, env={"ROS_DISTRO": "humble"})
    with pytest.raises(TypeError):
        variant.env["ROS_DISTRO"] = "humble"
    with pytest.raises(FrozenInstanceError):
        variant.image = "other"


@pytest.mark.parametrize("name", ["", "4.5", "5.1"])
def test_registry_access_and_compatibility_mapping(name):
    variant = docker.get_variant(name)
    assert docker.variant_env(name) is variant.env
    assert docker._VARIANTS[name]["image"] == variant.image
    with pytest.raises(TypeError):
        docker.VARIANTS[name] = variant
    with pytest.raises(TypeError):
        docker._VARIANTS[name]["image"] = "other"


def test_unknown_variant_has_a_helpful_error():
    with pytest.raises(KeyError, match="unknown docker variant 'unknown'"):
        docker.get_variant("unknown")


@pytest.mark.parametrize("name,deprecated", [("", False), ("4.5", True), ("5.1", True)])
def test_deprecated_variant_help_warns_without_confirmation(monkeypatch, capsys, name, deprecated):
    monkeypatch.setattr(docker, "_active_variant", "")
    monkeypatch.setattr(docker, "_active_profile", "")
    monkeypatch.setattr(docker, "_invoked_as", "docker")
    monkeypatch.setattr(docker, "_ensure_docker_cli", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("deprecation must not prompt"))
    assert docker.get_variant(name).deprecated is deprecated
    docker.run(["--help"], variant=name, invoked_as=f"docker{name}")
    output = capsys.readouterr().out
    assert ("E.O.L. and deprecated" in output) is deprecated
    if deprecated:
        assert "Use geniesim docker or geniesim docker6.2" in output


@pytest.fixture
def build_calls(monkeypatch):
    monkeypatch.setattr(docker, "_active_variant", "")
    monkeypatch.setattr(docker, "_libuipc_build_args", lambda _: ["--build-arg", "LIBUIPC_SM=89"])
    monkeypatch.setattr(docker, "_repo_root", lambda: Path(__file__).resolve().parents[3])
    monkeypatch.delenv("GENIESIM_ISAACSIM_TAG", raising=False)
    monkeypatch.delenv("GENIESIM_ISAACSIM_PIP_VERSION", raising=False)
    calls = []
    monkeypatch.setattr(
        docker.subprocess, "run", lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0)
    )
    return calls


def test_no_resolve_preserves_docker_options_and_china_mirrors(monkeypatch, build_calls):
    def resolve(*args, **kwargs):
        pytest.fail("--no-resolve must not access either index")

    monkeypatch.setattr(_isaacsim, "resolve", resolve)
    docker._do_build(["--no-resolve", "--china", "--no-cache", "--progress=plain"])
    command = build_calls[0]
    assert "ISAACSIM_REGISTRY=nvcr.m.daocloud.io/nvidia" in command
    assert "--no-cache" in command and "--progress=plain" in command
    assert "--no-resolve" not in command and "--china" not in command
    assert not any(arg.startswith("ISAACSIM_TAG=") for arg in command)


@pytest.mark.parametrize("option", [["--isaacsim", "6.X"], ["--isaacsim=6.X"]])
def test_explicit_series_is_resolved_for_the_image_python(monkeypatch, build_calls, option):
    resolutions = []
    monkeypatch.setattr(
        _isaacsim, "resolve", lambda spec, **kwargs: resolutions.append((spec, kwargs)) or ("6.2.0.1", "6.2.0.0")
    )
    docker._do_build(option)
    assert resolutions == [("6.X", {"python_tag": "cp312"})]
    assert "ISAACSIM_TAG=6.2.0.1" in build_calls[0]
    assert "ISAACSIM_PIP_VERSION=6.2.0.0" in build_calls[0]


def test_missing_series_fails_before_docker_build(build_calls):
    with pytest.raises(SystemExit) as exc:
        docker._do_build(["--isaacsim"])
    assert exc.value.code == 2
    assert build_calls == []


def test_env_pins_override_resolution_even_with_no_resolve(monkeypatch, build_calls):
    monkeypatch.setenv("GENIESIM_ISAACSIM_TAG", "6.2.0.1")
    monkeypatch.setenv("GENIESIM_ISAACSIM_PIP_VERSION", "6.2.0.0")
    monkeypatch.setattr(_isaacsim, "resolve", lambda *args, **kwargs: pytest.fail("env pins must skip index access"))
    docker._do_build(["--no-resolve"])
    assert "ISAACSIM_TAG=6.2.0.1" in build_calls[0]
    assert "ISAACSIM_PIP_VERSION=6.2.0.0" in build_calls[0]
