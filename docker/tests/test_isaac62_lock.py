# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Reject stale runtime locks and keep SDK providers outside registry installs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest


@pytest.fixture
def lock(monkeypatch):
    directory = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("isaac62_lock", directory / "lock_isaac62.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matching_lock_needs_no_resolver(lock, tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("package==1.0\n")
    actual = {"inputs": {"source/peer/pyproject.toml": "abc"}, "constraints_sha256": "sdk", "providers": {}}
    expected = {**actual, "base_image": "sdk@sha256:provenance", "requirements_sha256": lock.digest(requirements)}
    lock.validate(expected, actual, requirements)


@pytest.mark.parametrize("changed", ["inputs", "constraints_sha256", "providers", "uv_version", "python"])
def test_changed_source_or_runtime_requires_explicit_update(lock, tmp_path, changed):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("package==1.0\n")
    expected = {changed: "before", "requirements_sha256": lock.digest(requirements)}
    with pytest.raises(RuntimeError, match="Stale Isaac 6.2 lock"):
        lock.validate(expected, {changed: "after"}, requirements)


def test_edited_lock_is_rejected(lock, tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("package==1.0\n")
    expected = {"requirements_sha256": lock.digest(requirements)}
    requirements.write_text("package==2.0\n")
    with pytest.raises(RuntimeError, match="requirements.txt"):
        lock.validate(expected, {}, requirements)


def test_provider_guard_checks_metadata_and_actual_ik_bytes(lock, tmp_path):
    def wheel(name, extra):
        with ZipFile(tmp_path / name, "w") as archive:
            archive.writestr("provider.dist-info/METADATA", "Name: provider\nVersion: 1.0\n")
            archive.writestr("payload", extra)

    wheel(lock.IK_WHEEL, "original IK binary")
    wheel("torch-1.0-py3-none-any.whl", "metadata provider")
    before = lock.provider_inventory(tmp_path)
    wheel(lock.IK_WHEEL, "changed IK binary")
    wheel("torch-1.0-py3-none-any.whl", "regenerated provider ZIP")
    after = lock.provider_inventory(tmp_path)
    assert before[lock.IK_WHEEL] != after[lock.IK_WHEEL]
    assert before["torch-1.0-py3-none-any.whl"] == after["torch-1.0-py3-none-any.whl"]


def test_update_resolves_with_sdk_constraints_but_excludes_local_providers(lock, monkeypatch, tmp_path):
    providers = tmp_path / "providers"
    providers.mkdir()
    (providers / "torch-2.11.0+cu128-py3-none-any.whl").touch()
    lock_dir = tmp_path / "lock"
    lock_dir.mkdir()
    (lock_dir / "requirements.in").write_text("numba>=0.62\n")
    (lock_dir / "nvidia.in").write_text("ovrtx\n")
    args = SimpleNamespace(
        lock_dir=lock_dir,
        source=tmp_path / "source",
        providers=providers,
        wuji_sdk_version="2026.8.17",
        index_url="https://primary.example/simple",
        extra_index_url="https://sdk.example/simple",
        seed_installed=False,
        base_image="sdk@sha256:provenance",
    )
    monkeypatch.setattr(lock.subprocess, "check_output", lambda *a, **k: "consumer>=1\n")

    def compile(command, **kwargs):
        assert command[:4] == ["uv", "--no-config", "pip", "compile"]
        assert "--no-python-downloads" in command
        output = Path(command[command.index("--output-file") + 1])
        if "--format" in command:
            assert "--no-deps" in command
            output.write_text(
                "\n".join(
                    f'[[packages]]\nname = "{name}"\nversion = "1.0"\n'
                    f'wheels = [{{url = "https://example.test/{name}-1.0-py3-none-any.whl", '
                    f'hashes = {{sha256 = "{"a" * 64}"}}}}]\n'
                    for name in ("consumer", "ovrtx")
                )
            )
            return
        assert "--generate-hashes" in command
        assert command[command.index("--no-emit-package") + 1] == "torch"
        assert command[command.index("--default-index") + 1] == args.index_url
        assert command[command.index("--index") + 1] == args.extra_index_url
        assert command[command.index("--index-strategy") + 1] == "unsafe-best-match"
        assert Path(command[command.index("--constraint") + 1]).read_text() == "numpy==2.3.1\n"
        inputs = Path(command[4]).read_text()
        assert "numpy==2.3.1" in inputs and "consumer>=1" in inputs
        output.write_text("".join(f'{name}==1.0 \\\n    --hash=sha256:{"a" * 64}\n' for name in ("consumer", "ovrtx")))

    monkeypatch.setattr(lock.subprocess, "run", compile)
    lock.update(args, {"schema": 1}, "numpy==2.3.1\n")
    manifest = json.loads((lock_dir / "manifest.json").read_text())
    assert manifest["requirements_sha256"] == lock.digest(lock_dir / "requirements.txt")
    assert manifest["base_image"] == args.base_image
    assert (lock_dir / "nvidia-requirements.txt").read_text() == f'ovrtx==1.0 \\\n    --hash=sha256:{"a" * 64}\n'


def test_compact_hashes_keeps_only_compatible_previously_locked_artifacts(lock, monkeypatch):
    from packaging.tags import Tag

    monkeypatch.setattr(
        lock, "sys_tags", lambda: [Tag("cp312", "cp312", "manylinux_2_28_x86_64"), Tag("py3", "none", "any")]
    )
    wheels = [
        ("cp312-cp312-manylinux_2_28_x86_64", "a"),
        ("py3-none-any", "b"),
        ("cp311-cp311-manylinux_2_28_x86_64", "c"),
        ("cp312-cp312-win_amd64", "d"),
        ("cp312-cp312-manylinux_2_28_aarch64", "e"),
        ("py3-none-any", "f"),  # A new hash must never expand the original trust set.
    ]
    package = {
        "name": "example",
        "version": "1.0",
        "wheels": [
            {"url": f"https://example.test/example-1.0-{tag}.whl", "hashes": {"sha256": value * 64}}
            for tag, value in wheels
        ],
        "sdist": {"hashes": {"sha256": "0" * 64}},
    }
    text = "example==1.0 \\\n    " + " \\\n    ".join(f"--hash=sha256:{value * 64}" for value in "abcde0") + "\n"
    text += "\n# The following packages were excluded from the output:\n# torch\n"
    compact = lock.compact_hashes(text, [package])
    assert compact.count("--hash=") == 3
    for value in "ab0":
        assert value * 64 in compact
    assert lock.compact_hashes(compact, [package]) == compact


def test_compact_hashes_rejects_missing_version_or_untrusted_vendor_artifact(lock):
    text = f'ovrtx==1.0 \\\n    --hash=sha256:{"a" * 64}\n'
    # The bootstrap sdist must not replace the full NVIDIA wheel.
    package = {"name": "ovrtx", "version": "1.0", "sdist": {"hashes": {"sha256": "b" * 64}}}
    for packages in ([], [package], [{**package, "version": "2.0"}]):
        with pytest.raises(RuntimeError, match="No already-locked compatible artifact"):
            lock.compact_hashes(text, packages)


def test_vendor_routing_preserves_the_selected_artifact_hash(lock):
    text = "common==1.0 \\\n    --hash=sha256:common\novrtx==0.4.0 \\\n    --hash=sha256:nvidia\n"
    common, vendor = lock.split_vendor_lock(text, {"ovrtx"})
    assert common == "common==1.0 \\\n    --hash=sha256:common\n"
    assert vendor == "ovrtx==0.4.0 \\\n    --hash=sha256:nvidia\n"
    with pytest.raises(RuntimeError, match="absent"):
        lock.split_vendor_lock(text, {"missing"})


def test_edited_vendor_lock_is_rejected(lock, tmp_path):
    common, vendor = tmp_path / "requirements.txt", tmp_path / "nvidia-requirements.txt"
    common.write_text("common==1.0\n")
    vendor.write_text("ovrtx==0.4.0\n")
    manifest = {"requirements_sha256": lock.digest(common), "nvidia_requirements_sha256": lock.digest(vendor)}
    vendor.write_text("ovrtx==0.5.0\n")
    with pytest.raises(RuntimeError, match="nvidia-requirements.txt"):
        lock.validate(manifest, {}, common, vendor)


def test_checked_in_lock_matches_inputs_and_keeps_sdk_payloads_local(lock):
    repo = Path(__file__).resolve().parents[2]
    directory = repo / "docker/isaac62"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert "docker/libuipc-artifacts.json" in manifest["inputs"]
    assert manifest["inputs"] == lock.input_hashes(repo / "source", directory)
    assert manifest["requirements_sha256"] == lock.digest(directory / "requirements.txt")
    assert manifest["nvidia_requirements_sha256"] == lock.digest(directory / "nvidia-requirements.txt")
    records = (
        (directory / "requirements.txt").read_text() + (directory / "nvidia-requirements.txt").read_text()
    ).splitlines()
    for i, line in enumerate(records):
        if not line or line[0].isspace() or line.startswith("#"):
            continue
        assert "==" in line and " @ " not in line, line
        assert records[i + 1].strip().startswith("--hash=sha256:"), line
        name = line.split("==")[0].replace("_", "-")
        assert name not in {"isaacsim", "ik-solver", "torch", "torchvision", "torchaudio"}
        assert not name.startswith("nvidia-"), line
