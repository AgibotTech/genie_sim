# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Kit startup compatibility must preserve legacy runtimes and caller options."""

import pytest

from genie_sim_engine_isaacsim.kit import launch_config


@pytest.fixture(autouse=True)
def isolate_runtime_version(monkeypatch):
    monkeypatch.delenv("ISAAC_PATH", raising=False)


def test_62_manual_capture_config_preserves_caller_options(monkeypatch):
    monkeypatch.setattr(launch_config.metadata, "version", lambda name: "6.2.0.0")
    config = {"headless": True, "extra_args": ["--/renderer/asyncInit=false"]}
    result = launch_config.prepare_launch_config(config)
    assert result == {
        "headless": True,
        "extra_args": ["--/renderer/asyncInit=false", "--/rtx/hydra/supportMultiTickRate=false"],
    }
    assert config == {"headless": True, "extra_args": ["--/renderer/asyncInit=false"]}


@pytest.mark.parametrize("version", ["5.1.0", "4.5.0", "7.0.0.0"])
def test_other_sdk_versions_preserve_their_startup_contract(monkeypatch, version):
    monkeypatch.setattr(launch_config.metadata, "version", lambda name: version)
    config = {"headless": False, "extra_args": ["--/custom/setting=true"]}
    assert launch_config.prepare_launch_config(config) == config


def test_unknown_runtime_without_wheel_metadata_is_unchanged(monkeypatch):
    def missing(name):
        raise launch_config.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(launch_config.metadata, "version", missing)
    assert launch_config.prepare_launch_config({"headless": True}) == {"headless": True}


@pytest.mark.parametrize(
    "bundled_version, wheel_version, expected_manual_capture",
    [
        ("6.2.0-rc.15+release.53422.03030206.gl", None, True),
        ("6.2.0", "5.1.0", True),
        ("5.1.0", "6.2.0.0", False),
        ("6.0.0-rc.22", None, False),
    ],
)
def test_bundled_interpreter_uses_its_own_version(
    monkeypatch, tmp_path, bundled_version, wheel_version, expected_manual_capture
):
    import sys

    def version(name):
        if wheel_version is None:
            raise launch_config.metadata.PackageNotFoundError(name)
        return wheel_version

    (tmp_path / "VERSION").write_text(bundled_version)
    monkeypatch.setenv("ISAAC_PATH", str(tmp_path))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "kit/python/bin/python3"))
    monkeypatch.setattr(launch_config.metadata, "version", version)
    result = launch_config.prepare_launch_config({"headless": True})
    assert ("--/rtx/hydra/supportMultiTickRate=false" in result.get("extra_args", [])) is expected_manual_capture


def test_system_interpreter_uses_wheel_version_even_with_bundled_path(monkeypatch, tmp_path):
    import sys

    (tmp_path / "VERSION").write_text("5.1.0")
    monkeypatch.setenv("ISAAC_PATH", str(tmp_path))
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(launch_config.metadata, "version", lambda name: "6.2.0.0")
    result = launch_config.prepare_launch_config({"headless": True})
    assert "--/rtx/hydra/supportMultiTickRate=false" in result["extra_args"]
