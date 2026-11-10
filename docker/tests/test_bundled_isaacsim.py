# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Qualify the adapter's ownership, lazy activation and dependency guards."""

import importlib.util
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest


@pytest.fixture
def adapter():
    path = Path(__file__).resolve().parents[1] / "bundled_isaacsim.py"
    spec = importlib.util.spec_from_file_location("bundled_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_other_imports_do_not_activate_sdk(adapter, monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "sdk_environment", lambda root: pytest.fail("Unexpected SDK activation"))
    finder = adapter.BundledIsaacFinder(tmp_path)
    for name in ("numpy", "rclpy", "newton", "warp", "torch"):
        assert finder.find_spec(name) is None


def test_wheel_owns_only_adapter_and_metadata(adapter, monkeypatch, tmp_path):
    sdk = tmp_path / "sdk"
    for name in ("VERSION", "python_packages/isaacsim/__init__.py", "kit/libcarb.so", "setup_python_env.sh"):
        path = sdk / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("6.2.0-rc.15+release.test" if name == "VERSION" else "NVIDIA payload")
    monkeypatch.setattr(adapter, "sys", SimpleNamespace(version_info=(3, 12)))
    monkeypatch.setattr(adapter.subprocess, "check_output", lambda *a, **kw: "3.12\n")
    monkeypatch.setattr(adapter, "dependency_inventory", lambda root: (["newton[sim]==1.6.1"], [str(sdk)]))
    wheel = adapter.build_adapter(sdk, "6.2.0.0", tmp_path / "wheels")
    with ZipFile(wheel) as archive:
        assert all(not name.startswith("isaacsim/") for name in archive.namelist())
        metadata = archive.read("isaacsim-6.2.0.0.dist-info/METADATA").decode()
        assert "Requires-Dist: newton[sim]==1.6.1" in metadata
        assert "isaacsim-kernel" not in metadata
        manifest = json.loads(archive.read("isaacsim-6.2.0.0.dist-info/geniesim-bundled.json"))
        assert manifest["root"] == str(sdk)
        assert manifest["sdk_version"] == "6.2.0-rc.15+release.test"
    assert (sdk / "kit/libcarb.so").read_text() == "NVIDIA payload"
    with pytest.raises(RuntimeError, match="Unsupported SDK/version pair"):
        adapter.build_adapter(sdk, "5.1.0.0", tmp_path / "wheels")


def test_incomplete_sdk_is_not_registered(adapter, tmp_path):
    (tmp_path / "VERSION").write_text("6.2.0")
    with pytest.raises(RuntimeError, match="Incomplete bundled SDK"):
        adapter.build_adapter(tmp_path, "6.2.0.0", tmp_path / "wheels")


def test_inventory_excludes_unrelated_conflicting_archives(adapter, monkeypatch, tmp_path):
    def package(leaf, name, version):
        path = tmp_path / leaf
        info = path / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        return path

    startup = package("extscache/kit/pip_prebundle", "numpy", "2.3.1")
    unrelated = package("exts/isaacsim.robot_motion.pink/pip_prebundle", "Pillow", "12.3.0")
    package("exts/omni.pip.compute/pip_prebundle", "scipy", "1.17.0")
    for name, version in (("newton", "1.6.1"), ("mujoco", "3.12.0"), ("mujoco-warp", "3.12.0")):
        package("exts/isaacsim.pip.newton/pip_prebundle", name, version)
    package("exts/isaacsim.asset.importer.urdf/urdf_prebundle", "urdf-usd-converter", "0.3.2")
    package("extsDeprecated/omni.isaac.ml_archive/pip_prebundle", "torch", "2.11.0+cu128")
    monkeypatch.setattr(adapter, "sdk_environment", lambda root: {"PYTHONPATH": f"{startup}:{unrelated}"})
    requirements, paths = adapter.dependency_inventory(tmp_path)
    assert "newton[sim]==1.6.1" in requirements
    assert "torch==2.11.0+cu128" in requirements
    assert "numpy==2.3.1" in requirements
    assert not any("pillow" in requirement for requirement in requirements)
    assert str(unrelated) not in paths


def test_recipe_installs_adapter_instead_of_sdk_wheels():
    recipe = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
    assert "bundled_isaacsim.py" in recipe
    assert "isaacsim[all,extscache]" not in recipe
    assert "touch /isaac-sim/kit/EULA_ACCEPTED" in recipe


def test_torch_finder_routes_only_selected_modules_without_kit(adapter, monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "sdk_environment", lambda root: pytest.fail("Unexpected Kit activation"))
    ml = tmp_path / "extsDeprecated/omni.isaac.ml_archive/pip_prebundle"
    for name in ("torch", "torchvision", "torchaudio", "functorch", "torchgen"):
        package = ml / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        importlib.invalidate_caches()
        spec = adapter.BundledIsaacFinder(tmp_path).find_spec(name)
        assert Path(spec.origin) == package / "__init__.py"
    assert adapter.BundledIsaacFinder(tmp_path).find_spec("sympy") is None


def test_torch_providers_preserve_dependencies_and_never_own_sdk_files(adapter, tmp_path):
    sdk = tmp_path / "sdk"
    ml = sdk / "extsDeprecated/omni.isaac.ml_archive/pip_prebundle"
    cuda = sdk / "exts/isaacsim.pip.nv/pip_prebundle"
    for directory, name, version in (
        (ml, "torch", "2.11.0+cu128"),
        (ml, "torchvision", "0.26.0+cu128"),
        (ml, "torchaudio", "2.11.0+cu128"),
        (cuda, "cuda_bindings", "12.9.4"),
        (cuda, "nvidia_cuda_runtime_cu12", "12.8.90"),
    ):
        info = directory / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n" "Requires-Dist: triton==3.6.0\n"
        )
        (info / "RECORD").write_text("../../SDK-owned-native.so,,\n")
        (info / "LICENSE").write_text("upstream license")
    wheels = adapter.build_torch_providers(sdk, tmp_path / "wheels")
    assert len(wheels) == 5
    for wheel in wheels:
        with ZipFile(wheel) as archive:
            assert all(".dist-info/" in name for name in archive.namelist())
            record = archive.read(next(name for name in archive.namelist() if name.endswith("/RECORD"))).decode()
            assert "SDK-owned" not in record
            assert ".." not in record
            meta = archive.read(next(name for name in archive.namelist() if name.endswith("/METADATA"))).decode()
            assert "Requires-Dist: triton==3.6.0" in meta
            assert (
                archive.read(next(name for name in archive.namelist() if name.endswith("/LICENSE")))
                == b"upstream license"
            )
    assert any("torch-2.11.0+cu128-" in wheel.name for wheel in wheels)


def test_bytecode_cleanup_preserves_sourceless_and_external_payload(adapter, tmp_path):
    site = tmp_path / "site"
    cache = site / "pkg/__pycache__"
    cache.mkdir(parents=True)
    (cache.parent / "module.py").write_text("pass\n")
    generated = cache / "module.cpython-312.pyc"
    generated.write_bytes(b"regenerable")
    sourceless = cache / "vendor.cpython-312.pyc"
    sourceless.write_bytes(b"only copy")
    external = tmp_path / "sdk"
    external.mkdir()
    (external / "module.py").write_text("pass\n")
    (external / "module.pyc").write_bytes(b"SDK owned")
    (site / "sdk-link").symlink_to(external, target_is_directory=True)
    assert adapter.clean_bytecode(site) == len(b"regenerable")
    assert not generated.exists()
    assert sourceless.read_bytes() == b"only copy"
    assert (external / "module.pyc").read_bytes() == b"SDK owned"
