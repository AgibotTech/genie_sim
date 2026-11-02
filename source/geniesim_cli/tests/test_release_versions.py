# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Release synchronization covers source packages and excludes independent modules."""

import importlib.util
import inspect
from pathlib import Path

import pytest

from geniesim_cli.commands import tool, version

REPO = Path(__file__).resolve().parents[3]


def _package(repo, name="geniesim_scene_agent", value="3.2.0"):
    package = repo / "source" / name
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
    (package / "VERSION").write_text(value + "\n")
    return package / "VERSION"


def _ros_package(repo, name, value="3.2.0"):
    path = repo / "source/geniesim_ros/src/ros_ws/src" / name / "package.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<package><name>{name}</name><version>{value}</version></package>\n")
    return path


def test_sync_repairs_package_links_and_new_ros_packages_only(tmp_path):
    (tmp_path / "VERSION").write_text("3.5.0\n")
    package_version = _package(tmp_path)
    manifests = [_ros_package(tmp_path, name) for name in ("genie_sim_client", "genie_sim_engine_libuipc")]
    independent = _package(tmp_path, "rlinf_geniesim", "0.1.0")
    independent_ros = tmp_path / "source/rlinf_geniesim/ros_interfaces/package.xml"
    independent_ros.parent.mkdir(parents=True)
    independent_ros.write_text("<package><version>0.1.0</version></package>\n")
    originals = {path: path.read_bytes() for path in (independent, independent_ros)}

    assert version._do_sync(tmp_path, "3.5.0", dry_run=True) == 3
    assert package_version.read_text() == "3.2.0\n"
    assert version._do_sync(tmp_path, "3.5.0", dry_run=False) == 3
    assert package_version.is_symlink()
    assert package_version.resolve() == tmp_path / "VERSION"
    assert all("<version>3.5.0</version>" in path.read_text() for path in manifests)
    assert all(path.read_bytes() == original for path, original in originals.items())
    assert version._do_sync(tmp_path, "3.5.0", dry_run=False) == 0


def test_same_release_bump_still_repairs_metadata(tmp_path):
    (tmp_path / "VERSION").write_text("3.5.0\n")
    path = _ros_package(tmp_path, "genie_sim_engine_libuipc")
    version._do_bump(tmp_path, "3.5.0")
    assert "<version>3.5.0</version>" in path.read_text()


def test_bundled_message_dependencies_are_not_reversioned(tmp_path):
    (tmp_path / "VERSION").write_text("3.5.0\n")
    path = tmp_path / "source/geniesim_teleop/src/geniesim_teleop/app/share/geniesim_msg/package.xml"
    path.parent.mkdir(parents=True)
    original = "<package><name>geniesim_msg</name><version>1.0.0</version></package>\n"
    path.write_text(original)
    assert version._do_sync(tmp_path, "3.5.0", dry_run=False) == 0
    assert path.read_text() == original


def test_check_rejects_a_detached_package_version_even_if_value_matches(tmp_path):
    (tmp_path / "VERSION").write_text("3.5.0\n")
    _package(tmp_path, value="3.5.0")
    with pytest.raises(SystemExit) as error:
        version._do_check(tmp_path, "3.5.0")
    assert error.value.code == 1
    version._do_sync(tmp_path, "3.5.0", dry_run=False)
    with pytest.raises(SystemExit) as result:
        version._do_check(tmp_path, "3.5.0")
    assert result.value.code == 0


def test_registry_covers_every_first_class_package_and_owned_ros_manifest():
    packages = {path.name for path in (REPO / "source").glob("geniesim*") if (path / "pyproject.toml").is_file()}
    assert set(version._FIRST_CLASS_PACKAGES) == packages
    manifests = {str((path / "package.xml").relative_to(REPO)) for path in tool._enumerate_ros_packages(REPO)}
    assert manifests <= set(version._ROS_PACKAGE_XMLS)
    assert not any(path.startswith("source/rlinf_geniesim/") for path in version._ROS_PACKAGE_XMLS)


def test_checkout_release_versions_are_in_sync():
    expected = (REPO / "VERSION").read_text().strip()
    assert not version._scan_package_versions(REPO, expected)
    assert not version._scan_package_xmls(REPO, expected)[1]


def test_version_output_lists_all_first_class_packages(monkeypatch, capsys):
    monkeypatch.setattr("importlib.metadata.version", lambda name: "3.5.0")
    version._show_installed()
    output = capsys.readouterr().out
    assert all(name in output for name in version._FIRST_CLASS_PACKAGES)


@pytest.mark.parametrize("name", version._FIRST_CLASS_PACKAGES)
def test_source_and_installed_version_resolution(name, monkeypatch):
    path = REPO / "source" / name / "src" / name / "_version.py"
    spec = importlib.util.spec_from_file_location("release_version_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    resolve = module._resolve_version
    args = (name,) if inspect.signature(resolve).parameters else ()
    assert resolve(*args) == (REPO / "VERSION").read_text().strip()

    # A wheel has no source VERSION; installed metadata must be sufficient.
    monkeypatch.setattr(module, "_read_repo_version", lambda: None)
    monkeypatch.setattr(module, "_pkg_version", lambda dist: "9.8.7")
    assert resolve(*args) == "9.8.7"


@pytest.mark.parametrize("name", version._FIRST_CLASS_PACKAGES)
def test_installed_package_ignores_ancestor_sdk_version(name, tmp_path, monkeypatch):
    sdk = tmp_path / "sdk"
    path = sdk / "kit/python/site-packages" / name / "_version.py"
    path.parent.mkdir(parents=True)
    (sdk / "VERSION").write_text("6.2.0\n")
    path.write_bytes((REPO / "source" / name / "src" / name / "_version.py").read_bytes())
    spec = importlib.util.spec_from_file_location("sdk_version_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._read_repo_version() is None
    monkeypatch.setattr(module, "_pkg_version", lambda dist: "3.5.0")
    args = (name,) if inspect.signature(module._resolve_version).parameters else ()
    assert module._resolve_version(*args) == "3.5.0"
