# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise native artifact integrity and mode boundaries without native execution."""

from contextlib import nullcontext
from email.parser import Parser
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("install_libuipc", ROOT / "docker/install_libuipc.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


@pytest.fixture
def fixture_wheels(monkeypatch, tmp_path):
    """Use tiny archive fixtures for install control flow; real bytes are release QA."""
    records = {}
    for target, original in installer.ARTIFACTS["artifacts"].items():
        row = {**original}
        wheel = tmp_path / row["path"]
        wheel.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(wheel, "w") as archive:
            archive.writestr("pyuipc-0.9.0.dist-info/METADATA", "Name: pyuipc\nVersion: " + row["version"] + "\n")
        row.update(size=wheel.stat().st_size, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
        records[target] = row
    monkeypatch.setattr(installer, "ARTIFACTS", {**installer.ARTIFACTS, "artifacts": records})
    monkeypatch.setenv("GENIESIM_ASSETS_SRC", str(tmp_path))
    monkeypatch.setenv("GENIESIM_LIBUIPC_CACHE", str(tmp_path / "cache"))


def test_asset_artifact_hash_license_and_native_abi():
    if not os.environ.get("GENIESIM_ASSETS_SRC"):
        pytest.skip("Set GENIESIM_ASSETS_SRC for real artifact release QA")
    wheel = installer.select_wheel()
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_path = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        wheel_metadata = Parser().parsestr(archive.read(metadata_path.replace("METADATA", "WHEEL")).decode())
        license_path = next(name for name in names if name.endswith(".dist-info/licenses/LICENSE"))
        license_root = str(Path(license_path).parent)
        upstream_license = f"{license_root}/LICENSES/upstream-libuipc-Apache-2.0.txt"
        assert archive.read(upstream_license) == (ROOT / "3rdparty/pyuipc-LICENSE").read_bytes()
        assert f"{license_root}/NOTICE" in names
        for license_name in ("upstream-libuipc-Apache-2.0.txt", "upstream-python-GPL-3.0.txt", "MPL-2.0.txt"):
            assert f"{license_root}/LICENSES/{license_name}" in names
        assert wheel_metadata["Tag"] == "cp312-cp312-linux_x86_64"
        assert any(name.endswith("pyuipc.cpython-312-x86_64-linux-gnu.so") for name in names)
        assert ".sm89sm120.cu128." in metadata["Version"]
        assert all("newton" not in requirement.lower() for requirement in metadata.get_all("Requires-Dist", []))


def test_changed_artifact_rejected(monkeypatch, fixture_wheels):
    monkeypatch.setitem(installer.ARTIFACTS["artifacts"]["universal"], "sha256", "0" * 64)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        installer.select_wheel()


def test_dry_run_does_not_install_or_load_native(monkeypatch, fixture_wheels):
    monkeypatch.setattr(installer, "validate_platform", lambda: None)
    monkeypatch.setattr(installer, "install", lambda *args, **kwargs: pytest.fail("installation attempted"))
    monkeypatch.setattr(installer, "verify_image", lambda *args: pytest.fail("native verification attempted"))
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: pytest.fail("GPU queried"))
    installer.main(["--dry-run"])


@pytest.mark.parametrize(
    "arguments", [["--build-image", "--dry-run"], ["--download-only", "--if-needed"], ["--dependencies-only", "--if-needed"]]
)
def test_invalid_modes_rejected_before_platform_or_gpu(monkeypatch, arguments):
    monkeypatch.setattr(installer, "validate_platform", lambda: pytest.fail("platform accessed"))
    with pytest.raises(SystemExit) as error:
        installer.main(arguments)
    assert error.value.code == 2


def test_image_build_records_artifact_without_gpu_query(monkeypatch, tmp_path, fixture_wheels):
    wheel = installer.select_wheel()
    calls = []
    monkeypatch.setattr(installer, "_installation_lock", nullcontext)
    monkeypatch.setattr(installer.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setattr(installer, "_install", lambda selected: calls.append(selected))
    installer.install_image(wheel)
    assert calls == [wheel]
    assert json.loads((tmp_path / ".libuipc-image.json").read_text()) == {
        "sha256": installer.ARTIFACTS["artifacts"]["universal"]["sha256"]
    }


def test_unmarked_image_skips_verification(monkeypatch, tmp_path):
    monkeypatch.setattr(installer.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setattr(installer, "check_native", lambda: pytest.fail("native loading attempted"))
    assert installer.verify_image(Path("unused.whl")) is False


def test_mismatched_image_fails_without_replacing_package(monkeypatch, tmp_path):
    (tmp_path / ".libuipc-image.json").write_text(json.dumps({"sm": 89, "sha256": "0" * 64}))
    monkeypatch.setattr(installer.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setattr(installer, "_installation_lock", nullcontext)
    monkeypatch.setattr(installer, "_install", lambda *args: pytest.fail("package replacement attempted"))
    monkeypatch.setattr(installer, "check_native", lambda: pytest.fail("native loading attempted"))
    with pytest.raises(RuntimeError, match="Image artifact does not match the pinned universal wheel"):
        installer.verify_image(Path("unused.whl"))


def test_verified_image_returns_before_installation(monkeypatch, fixture_wheels):
    wheel = installer.select_wheel()
    monkeypatch.setattr(installer, "validate_platform", lambda: None)
    monkeypatch.setattr(installer, "verify_image", lambda selected: selected == wheel)
    monkeypatch.setattr(installer, "install", lambda *args, **kwargs: pytest.fail("installation attempted"))
    installer.main(["--if-needed"])


def test_native_build_installs_universal_without_architecture(tmp_path):
    text = (ROOT / "docker/Dockerfile").read_text()
    block = text.split("COPY docker/install_libuipc.py", 1)[1].split("\n\n", 1)[0]
    command = block.split("RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked", 1)[1]
    command = command.replace("--mount=from=libuipc_assets,target=/libuipc-assets/wheels,ro", "")
    log = tmp_path / "calls"
    python = tmp_path / "python3"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CALLS"\n')
    python.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", "-eu", "-c", command],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "CALLS": str(log),
            "LIBUIPC_SOURCE": "modelscope",
            "LIBUIPC_REVISION": "",
        },
        capture_output=True,
        text=True,
    )
    assert log.read_text().splitlines() == ["/opt/geniesim/docker/install_libuipc.py", "--build-image"]
    assert result.returncode == 0


@pytest.mark.parametrize("code", [0, 1])
@pytest.mark.parametrize("prepared", [False, True])
def test_entrypoint_only_bootstraps_native_prepared_images(tmp_path, code, prepared):
    text = (ROOT / "docker/entrypoint.sh").read_text()
    block = text.split("# Phase 3b —", 1)[1].split("# Phase 4 —", 1)[0]
    block = block[block.index("if [ -f ") :]
    installer = tmp_path / "install_libuipc.py"
    if prepared:
        installer.touch()
    block = block.replace("/opt/geniesim/docker/install_libuipc.py", str(installer))
    log = tmp_path / "calls"
    python = tmp_path / "python3"
    python.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$CALLS"\n'
        'printf "%s\\n" "$GENIESIM_REPO_ROOT" "$PYTHONPATH" > "$CALLS.env"\n'
        'exit "$VERIFY_CODE"\n'
    )
    python.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", "-eu", "-c", block],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "CALLS": str(log),
            "VERIFY_CODE": str(code),
            "YELLOW": "",
            "RST": "",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    if not prepared:
        assert not log.exists()
        assert not result.stdout
        return
    assert log.read_text().splitlines() == ["-m", "geniesim_cli", "bootstrap", "--libuipc-only"]
    root, path = Path(str(log) + ".env").read_text().splitlines()
    assert root == "/workspace" and path.split(":")[0] == "/workspace/source/geniesim_cli/src"
    assert ("bootstrap unavailable" in result.stdout) == bool(code)


def test_cuda128_artifact_uses_complete_cuda12_provider(monkeypatch, tmp_path):
    providers = [tmp_path / "cuda13", tmp_path / "cuda12"]
    cuda12 = {
        "libcudart.so.12": "cuda_runtime",
        "libcublas.so.12": "cublas",
        "libcublasLt.so.12": "cublas",
        "libcusparse.so.12": "cusparse",
        "libcusolver.so.11": "cusolver",
        "libnvJitLink.so.12": "nvjitlink",
    }
    cuda13 = {
        "libcudart.so.13": "cuda_runtime",
        "libcublas.so.13": "cublas",
        "libcublasLt.so.13": "cublas",
        "libcusparse.so.12": "cusparse",
        "libcusolver.so.12": "cusolver",
        "libnvJitLink.so.13": "nvjitlink",
    }
    for root, components in zip(providers, [cuda13, cuda12]):
        for name, component in components.items():
            library = root / component / "lib" / name
            library.parent.mkdir(parents=True, exist_ok=True)
            library.write_bytes(b"runtime")
    requested = []

    def roots(major):
        requested.append(major)
        return providers

    monkeypatch.setattr(installer, "_cuda_runtime_roots", roots)
    monkeypatch.setattr(installer, "check_native", lambda *args: pytest.fail("provider not selected"))
    payload = tmp_path / "payload"
    native = payload / "uipc/_native"
    native.mkdir(parents=True)
    info = payload / "pyuipc-test.dist-info"
    info.mkdir()
    (info / "RECORD").write_text("")
    installer._link_cuda_runtime(payload)
    assert requested == [12]
    assert {path.name for path in native.iterdir()} == set(cuda12)
    for name, component in cuda12.items():
        assert (native / name).resolve() == providers[1] / component / "lib" / name
