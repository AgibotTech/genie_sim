# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Mirror selection for independent collection and scene-authoring images."""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from geniesim_cli.commands import data_collection

ROOT = Path(__file__).resolve().parents[3]
COLLECTION = ROOT / "source/data_collection"
SCENE = ROOT / "source/geniesim_scene_agent"


def build_arguments(command):
    return [command[i + 1] for i, token in enumerate(command) if token == "--build-arg"]


@pytest.fixture
def collection_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(data_collection, "_ensure_docker", lambda: None)
    monkeypatch.setattr(data_collection, "_dc_root", lambda: COLLECTION)
    monkeypatch.setattr(
        data_collection.subprocess,
        "run",
        lambda cmd, **kw: calls.append((cmd, {**kw, "env": {"DOCKER_BUILDKIT": kw["env"]["DOCKER_BUILDKIT"]}}))
        or SimpleNamespace(returncode=0),
    )
    return calls


def test_collection_china_and_explicit_urls(collection_calls):
    with pytest.raises(SystemExit) as result:
        data_collection._do_build(
            [
                "--china",
                "--build-arg",
                "PIP_INDEX_URL=https://pypi.nvidia.com",
                "--build-arg=APT_MIRROR=http://archive.ubuntu.com/ubuntu",
                "--no-cache",
            ]
        )
    assert result.value.code == 0
    command, kwargs = collection_calls[0]
    arguments = build_arguments(command)
    assert not any(value.startswith("ISAACSIM_REGISTRY=") for value in arguments)
    assert arguments.count("PIP_INDEX_URL=https://pypi.nvidia.com") == 1
    assert arguments.count("APT_MIRROR=http://archive.ubuntu.com/ubuntu") == 1
    assert "--china" not in command
    assert "--no-cache" in command
    assert command[-1] == "."
    assert f"geniesim_mirrors={ROOT / 'docker'}" in command
    assert kwargs["cwd"] == str(COLLECTION)
    assert kwargs["env"]["DOCKER_BUILDKIT"] == "1"


def test_collection_inherits_custom_parent_without_preset(collection_calls):
    with pytest.raises(SystemExit):
        data_collection._do_build(["--build-arg", "GENIESIM_BASE_IMAGE=parent:local"])
    assert build_arguments(collection_calls[0][0]) == ["GENIESIM_BASE_IMAGE=parent:local"]


@pytest.mark.parametrize("options", [["--china", "--mirrors=upstream"], ["--mirrors"], ["--mirrors=missing"]])
def test_collection_rejects_invalid_presets_before_docker(collection_calls, options):
    with pytest.raises(SystemExit) as error:
        data_collection._do_build(options)
    assert error.value.code == 2
    assert collection_calls == []


def test_collection_dry_run_consumes_mirror_options(collection_calls, capsys):
    data_collection._do_build(["--mirrors=upstream", "--dry-run"])
    output = capsys.readouterr().out
    assert "APT_MIRROR=upstream" in output
    assert "PIP_INDEX_URL=https://pypi.org/simple" in output
    assert "--mirrors=" not in output
    assert collection_calls == []


def test_collection_environment_backed_override_is_not_duplicated(collection_calls):
    with pytest.raises(SystemExit):
        data_collection._do_build(["--china", "--build-arg", "PIP_INDEX_URL"])
    arguments = build_arguments(collection_calls[0][0])
    assert arguments.count("PIP_INDEX_URL") == 1
    assert not any(value.startswith("PIP_INDEX_URL=") for value in arguments)


def load_scene_builder():
    script = SCENE / "docker/build_image.py"
    assert script.is_file(), "Scene image builder must expose the shared mirror selection"
    spec = importlib.util.spec_from_file_location("scene_image_builder", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_vl_builder_reuses_presets_and_keeps_full_parent_override(monkeypatch):
    builder = load_scene_builder()
    calls = []
    monkeypatch.setattr(
        builder.subprocess,
        "run",
        lambda cmd, **kw: calls.append((cmd, {**kw, "env": {"DOCKER_BUILDKIT": kw["env"]["DOCKER_BUILDKIT"]}}))
        or SimpleNamespace(returncode=0),
    )
    assert builder.main(["--target", "vl", "--china", "--build-arg", "ISAACSIM_IMAGE=parent:local"]) == 0
    command, kwargs = calls[0]
    assert command[command.index("--target") + 1] == "vl"
    assert command[command.index("-t") + 1].endswith(":vl")
    assert "ISAACSIM_IMAGE=parent:local" in build_arguments(command)
    assert "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple" in build_arguments(command)
    assert "--china" not in command
    assert f"geniesim_mirrors={ROOT / 'docker'}" in command
    assert kwargs["cwd"] == str(SCENE)


def test_scene_dry_run_is_available_without_installing_cli(tmp_path):
    script = SCENE / "docker/build_image.py"
    assert script.is_file(), "Scene image builder is missing"
    result = subprocess.run(
        ["python3", str(script), "--mirrors=upstream", "--dry-run"], cwd=str(tmp_path), text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert "PIP_INDEX_URL=https://pypi.org/simple" in result.stdout
    assert "--target runtime" in result.stdout


@pytest.mark.parametrize("recipe", [COLLECTION / "docker/Dockerfile", SCENE / "docker/Dockerfile.6.2"])
def test_module_recipes_persist_mirrors_before_dependencies(recipe):
    text = recipe.read_text()
    assert "ARG APT_MIRROR\n" in text
    assert "ARG PIP_INDEX_URL\n" in text
    assert "PIP_CONFIG_FILE=/etc/pip.conf" in text
    assert "COPY --from=geniesim_mirrors" in text
    assert 'geniesim-configure-mirrors --apt "${APT_MIRROR}" --pip "${PIP_INDEX_URL}"' in text
    first_install = "RUN missing=()" if recipe.parent.parent == COLLECTION else "RUN /isaac-sim/python.sh -m venv"
    assert text.index("geniesim-configure-mirrors --apt") < text.index(first_install)
    assert '--index-url "${PIP_INDEX_URL}"' not in text
