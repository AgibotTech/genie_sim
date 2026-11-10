# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Respect SDK markers, enabled extras, and conflicting installed versions."""

import importlib.util
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def collector(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "collect_runtime_constraints.py"
    spec = importlib.util.spec_from_file_location("runtime_constraints", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    packages = {
        "isaacsim": ("6.2.0.0", ['isaacsim-core==6.2.0.0; extra == "all"']),
        "isaacsim-core": ("6.2.0.0", ["numpy==2.3.1", "newton[sim]==1.6.1", "etils[epath]==1.13.0"]),
        "colcon-core": ("0.21.3", ["setuptools<80,>=30.3.0", 'unneeded; extra == "test"']),
        "newton": ("1.6.1", ['mujoco~=3.12.0; extra == "sim"', 'unneeded; extra == "docs"']),
        "etils": ("1.14.0", ['fsspec; extra == "epath"', 'unneeded; python_version < "3.8"']),
        "numpy": ("2.5.4", []),
        "setuptools": ("81.0.0", []),
        "mujoco": ("3.12.0", []),
        "fsspec": ("2026.2.0", []),
    }

    def distribution(name):
        if name not in packages:
            raise metadata.PackageNotFoundError(name)
        version, requirements = packages[name]
        return SimpleNamespace(
            metadata={"Name": name}, version=version, requires=requirements, read_text=lambda name: None
        )

    monkeypatch.setattr(module.metadata, "distribution", distribution)
    monkeypatch.setattr(module.metadata, "distributions", lambda: [distribution(n) for n in packages])
    monkeypatch.setattr(module.metadata, "version", lambda name: distribution(name).version)
    return module, packages


def test_sdk_extras_are_protected_without_enabling_unrequested_extras(collector):
    module, packages = collector
    constraints = module.collect_constraints()
    assert "isaacsim-core==6.2.0.0" in constraints
    assert "newton==1.6.1" in constraints
    assert "mujoco~=3.12.0" in constraints
    assert "etils==1.13.0" in constraints
    assert "fsspec" in constraints
    assert not any("unneeded" in raw for raw in constraints)
    assert all("[" not in raw and ";" not in raw for raw in constraints)
    assert constraints == sorted(set(constraints))


def test_only_incompatible_versions_require_repair(collector):
    module, packages = collector
    assert module.repair_requirements(module.collect_constraints()) == [
        "etils==1.13.0",
        "numpy==2.3.1",
        "setuptools<80,>=30.3.0",
    ]


def test_missing_dependency_requires_installation(collector):
    module, packages = collector
    del packages["numpy"]
    assert "numpy==2.3.1" in module.repair_requirements(module.collect_constraints())


def test_repair_preserves_the_intersection_of_consumer_requirements(collector):
    module, packages = collector
    assert module.repair_requirements(["setuptools>=68", "setuptools<80,>=30.3.0"]) == ["setuptools<80,>=30.3.0,>=68"]


def test_extras_cycles_terminate(collector):
    module, packages = collector
    packages["newton"] = ("1.6.1", ['newton[sim]; extra == "sim"', 'mujoco~=3.12.0; extra == "sim"'])
    assert "mujoco~=3.12.0" in module.collect_constraints()


def test_unavailable_runtime_metadata_is_an_error(collector):
    module, packages = collector
    del packages["isaacsim"]
    with pytest.raises(RuntimeError, match="Isaac Sim must be installed"):
        module.collect_constraints()
    packages["isaacsim"] = ("6.2.0.0", [])
    del packages["colcon-core"]
    with pytest.raises(metadata.PackageNotFoundError):
        module.collect_constraints()


def test_direct_urls_cannot_silently_escape_constraints(collector):
    module, packages = collector
    packages["isaacsim-core"] = ("6.2.0.0", ["numpy @ file:///tmp/numpy.whl"])
    with pytest.raises(RuntimeError, match="direct URL requirement"):
        module.collect_constraints()


def test_bundled_metadata_supplies_extras_before_system_dependencies_are_installed(collector, monkeypatch):
    module, packages = collector
    original = module.metadata.distribution
    bundled_newton = original("newton")
    packages["isaacsim"] = ("6.2.0.0", ["newton[sim]==1.6.1"])
    del packages["isaacsim-core"]
    del packages["newton"]

    def distribution(name):
        result = original(name)
        if name == "isaacsim":
            result.read_text = lambda name: '{"dependency_paths": ["/sdk/newton"]}'
        return result

    monkeypatch.setattr(module.metadata, "distribution", distribution)
    monkeypatch.setattr(
        module.metadata,
        "distributions",
        lambda **kwargs: [bundled_newton] if kwargs.get("path") else [distribution(n) for n in packages],
    )
    constraints = module.collect_constraints()
    assert "mujoco~=3.12.0" in constraints
    assert "newton==1.6.1" in module.repair_requirements(constraints)
