# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Public variant values and CLI build options retain their contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from geniesim_cli import _isaacsim
from geniesim_cli.commands import docker


@pytest.fixture
def build_calls(monkeypatch):
    monkeypatch.setattr(docker, "_active_variant", "")
    monkeypatch.setattr(docker, "_active_profile", "")
    monkeypatch.setattr(docker, "_repo_root", lambda: Path(__file__).resolve().parents[3])
    monkeypatch.delenv("GENIESIM_ISAACSIM_TAG", raising=False)
    monkeypatch.delenv("GENIESIM_ISAACSIM_PIP_VERSION", raising=False)
    calls = []
    monkeypatch.setattr(
        docker.subprocess, "run", lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0)
    )
    return calls


@pytest.mark.parametrize("variant", ["", "4.5", "5.1"])
def test_base_builds_default_to_upstream(monkeypatch, build_calls, variant):
    monkeypatch.setattr(docker, "_active_variant", variant)
    docker._do_build(["--no-resolve"])
    command = build_calls[0]
    assert "ISAACSIM_REGISTRY=nvcr.io/nvidia" in command
    assert "APT_MIRROR=upstream" in command
    assert "PIP_INDEX_URL=https://pypi.org/simple" in command


@pytest.mark.parametrize("profile", ["post-train", "post-train-teleop"])
def test_derived_builds_inherit_unless_a_preset_is_selected(monkeypatch, build_calls, profile):
    monkeypatch.setattr(docker, "_active_profile", profile)
    docker._do_build([])
    assert "--build-arg" not in build_calls[0]
    docker._do_build(["--mirrors", "upstream"])
    assert "APT_MIRROR=upstream" in build_calls[1]
    assert "PIP_INDEX_URL=https://pypi.org/simple" in build_calls[1]
    assert not any(token.startswith("ISAACSIM_REGISTRY=") for token in build_calls[1])


@pytest.mark.parametrize("flags", [["--china"], ["--mirrors", "china"], ["--mirrors=china"]])
def test_china_alias_and_preset_are_equivalent(build_calls, flags):
    docker._do_build(["--no-resolve", *flags])
    assert "APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu" in build_calls[0]
    assert "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple" in build_calls[0]
    assert not any(token.startswith("--mirrors") for token in build_calls[0])


def test_explicit_arguments_override_presets_once_and_are_reported(build_calls, capsys):
    docker._do_build(
        [
            "--no-resolve",
            "--china",
            "--build-arg",
            "PIP_INDEX_URL=https://pypi.nvidia.com",
            "--build-arg=PIP_INDEX_URL=https://pypi.org/simple",
            "--no-cache",
        ]
    )
    command = build_calls[0]
    assert [token for token in command if token.startswith("PIP_INDEX_URL=")] == [
        "PIP_INDEX_URL=https://pypi.org/simple"
    ]
    assert "--no-cache" in command
    assert "PIP_INDEX_URL: https://pypi.org/simple" in capsys.readouterr().out


def test_bare_build_argument_uses_host_environment(monkeypatch, build_calls):
    monkeypatch.setenv("PIP_INDEX_URL", "https://pypi.org/simple")
    docker._do_build(["--no-resolve", "--china", "--build-arg", "PIP_INDEX_URL"])
    assert "PIP_INDEX_URL" in build_calls[0]
    assert not any(token.startswith("PIP_INDEX_URL=") for token in build_calls[0])


@pytest.mark.parametrize(
    "flags",
    [
        ["--mirrors"],
        ["--mirrors="],
        ["--mirrors", "unknown"],
        ["--build-arg"],
        ["--china", "--mirrors", "upstream"],
    ],
)
def test_invalid_mirror_selection_fails_before_resolution(monkeypatch, build_calls, flags):
    monkeypatch.setattr(_isaacsim, "resolve", lambda *args, **kwargs: pytest.fail("invalid flags must not resolve"))
    with pytest.raises(SystemExit) as exc:
        docker._do_build(flags)
    assert exc.value.code == 2
    assert not build_calls


def test_custom_preset_uses_configuration_without_python_changes(monkeypatch, build_calls, tmp_path):
    from geniesim_cli import _mirrors

    path = tmp_path / "mirrors.toml"
    path.write_text('[custom]\nPIP_INDEX_URL = "https://pypi.org/simple"\n')
    preset = _mirrors.load_preset(path, "custom")
    monkeypatch.setattr(_mirrors, "load_preset", lambda *_: preset)
    docker._do_build(["--no-resolve", "--mirrors", "custom"])
    assert "PIP_INDEX_URL=https://pypi.org/simple" in build_calls[0]


@pytest.mark.parametrize(
    "body",
    [
        '[custom]\nUNKNOWN_KEY = "value"\n',
        "[custom]\nPIP_INDEX_URL = 42\n",
        '[custom]\nPIP_INDEX_URL = "file:///tmp/index"\n',
        '[custom]\nPIP_INDEX_URL = "upstream"\n',
        '[custom]\nISAACSIM_REGISTRY = "https://nvcr.io/nvidia"\n',
        "[custom\n",
    ],
)
def test_malformed_preset_is_rejected(tmp_path, body):
    from geniesim_cli import _mirrors

    path = tmp_path / "mirrors.toml"
    path.write_text(body)
    with pytest.raises(ValueError):
        _mirrors.load_preset(path, "custom")


def test_explicit_pin_does_not_duplicate_resolved_pin(monkeypatch, build_calls):
    monkeypatch.setattr(_isaacsim, "resolve", lambda *_, **__: ("6.2.0.1", "6.2.0.0"))
    docker._do_build(["--build-arg", "ISAACSIM_TAG=6.2.0.1"])
    assert build_calls[0].count("ISAACSIM_TAG=6.2.0.1") == 1
