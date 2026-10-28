# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""AS3 cache ownership failures must be diagnosed before conversion mutates it."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from genie_sim_engine_isaacsim.assemble import _robot_impl as impl


def test_writable_cache_and_new_output_pass_without_creating_files(tmp_path):
    payloads = tmp_path / "payloads" / "Physics"
    payloads.mkdir(parents=True)
    (payloads / "physics.usda").write_text("unchanged")
    before = sorted(tmp_path.rglob("*"))
    impl._check_assembly_output_permissions(str(tmp_path))
    impl._check_assembly_output_permissions(str(tmp_path / "new_scene"))
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.skipif(os.geteuid() == 0, reason="Root bypasses ordinary file permissions")
@pytest.mark.parametrize("blocked", ["robot.usda", "robot_raw.urdf", "payloads", "payloads/Physics"])
def test_unwritable_outputs_report_absolute_path_and_owner_without_mutating_cache(tmp_path, blocked):
    (tmp_path / "payloads" / "Physics").mkdir(parents=True)
    (tmp_path / "payloads" / "robot.usda").write_text("old payload")
    (tmp_path / "robot.usda").write_text("old root")
    (tmp_path / "robot_raw.urdf").write_text("old URDF")
    target = tmp_path / blocked
    original_mode = target.stat().st_mode
    target.chmod(0o555 if target.is_dir() else 0o444)
    try:
        with pytest.raises(PermissionError) as caught:
            impl._check_assembly_output_permissions(str(tmp_path))
        assert caught.value.filename == str(target)
        assert f"owner={os.geteuid()}:" in str(caught.value)
        assert "geniesim docker6.2 into" in str(caught.value)
        assert (tmp_path / "payloads" / "robot.usda").read_text() == "old payload"
        assert (tmp_path / "robot.usda").read_text() == "old root"
    finally:
        target.chmod(original_mode)


@pytest.mark.skipif(os.geteuid() == 0, reason="Requires a directory owned by an ordinary user")
@pytest.mark.parametrize("new_scene", [False, True])
def test_root_cannot_pollute_user_owned_cache(monkeypatch, tmp_path, new_scene):
    monkeypatch.setattr(impl.os, "geteuid", lambda: 0)
    output = tmp_path / "new_scene" if new_scene else tmp_path
    with pytest.raises(PermissionError, match="Refusing to create root-owned") as caught:
        impl._check_assembly_output_permissions(str(output))
    assert caught.value.filename == str(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.geteuid() == 0, reason="Requires a directory owned by an ordinary user")
def test_root_owned_scene_directory_still_checks_user_owned_parent(monkeypatch, tmp_path):
    output = tmp_path / "scene"
    output.mkdir()
    original_stat = Path.stat

    def stat_with_root_owned_scene(path, **kwargs):
        info = original_stat(path, **kwargs)
        return SimpleNamespace(st_uid=0) if path == output else info

    monkeypatch.setattr(Path, "stat", stat_with_root_owned_scene)
    monkeypatch.setattr(impl.os, "geteuid", lambda: 0)
    with pytest.raises(PermissionError, match="Refusing to create root-owned") as caught:
        impl._check_assembly_output_permissions(str(output))
    assert caught.value.filename == str(tmp_path)
    assert not list(output.iterdir())


@pytest.mark.skipif(os.geteuid() == 0, reason="Root bypasses ordinary file permissions")
def test_isaac62_permission_check_precedes_urdf_fetch_and_staging(monkeypatch, tmp_path):
    scene = tmp_path / "scene.yaml"
    scene.write_text("robot:\n  robot_source:\n    urdf: {}\n")
    output = tmp_path / "output"
    output.mkdir()
    payloads = output / "payloads"
    payloads.mkdir()
    payloads.chmod(0o555)
    monkeypatch.setattr(impl, "_isaacsim_version", lambda: (6, 2))

    def unexpected_fetch(**kwargs):
        pytest.fail("URDF fetch must not run for an unwritable cache")

    monkeypatch.setattr(impl, "_fetch_urdf_from_topic", unexpected_fetch)
    try:
        with pytest.raises(PermissionError) as caught:
            impl.assemble_robot(str(scene), str(output))
        assert caught.value.filename == str(payloads)
        assert not (output / "robot_raw.urdf").exists()
    finally:
        payloads.chmod(0o755)


@pytest.mark.parametrize("version", [(4, 5), (5, 1)])
def test_legacy_assembly_does_not_use_as3_permission_check(monkeypatch, tmp_path, version):
    scene = tmp_path / "scene.yaml"
    scene.write_text("robot:\n  robot_source:\n    urdf: {}\n")
    monkeypatch.setattr(impl, "_isaacsim_version", lambda: version)
    monkeypatch.setattr(impl, "_check_assembly_output_permissions", lambda _: pytest.fail("AS2 must stay unchanged"))

    def stop_after_check(**kwargs):
        raise RuntimeError("reached URDF fetch")

    monkeypatch.setattr(impl, "_fetch_urdf_from_topic", stop_after_check)
    with pytest.raises(RuntimeError, match="reached URDF fetch"):
        impl.assemble_robot(str(scene), str(tmp_path / "output"))
