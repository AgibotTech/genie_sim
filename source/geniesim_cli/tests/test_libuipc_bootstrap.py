# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Native bootstrap and Docker GPU selection remain independent of simulator imports."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from geniesim_cli import _env
from geniesim_cli.commands import bootstrap, docker


@pytest.mark.parametrize("capability,sm", [("8.9\n", "89"), ("12.0\n", "120")])
def test_build_selects_host_gpu_zero(monkeypatch, capability, sm):
    calls = []

    def query(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=capability)

    monkeypatch.setattr(docker.subprocess, "run", query)
    assert docker._libuipc_build_args([]) == ["--build-arg", f"LIBUIPC_SM={sm}"]
    assert calls[0][0] == ["nvidia-smi", "--id=0", "--query-gpu=compute_cap", "--format=csv,noheader"]
    assert calls[0][1]["timeout"] == 15


@pytest.mark.parametrize(
    "flags", [["--build-arg", "LIBUIPC_SM=120"], ["--build-arg=LIBUIPC_SM=120"], ["--build-arg", "LIBUIPC_SM"]]
)
def test_explicit_architecture_needs_no_host_gpu(monkeypatch, flags):
    monkeypatch.setenv("LIBUIPC_SM", "120")
    monkeypatch.setattr(docker.subprocess, "run", lambda *a, **k: pytest.fail("GPU queried"))
    assert docker._libuipc_build_args(flags) == []


@pytest.mark.parametrize("value", ["", "90", "sm_89"])
def test_invalid_explicit_architecture_fails_before_gpu(monkeypatch, value):
    monkeypatch.setattr(docker.subprocess, "run", lambda *a, **k: pytest.fail("GPU queried"))
    with pytest.raises(RuntimeError, match="LIBUIPC_SM must be 89 or 120"):
        docker._libuipc_build_args(["--build-arg", f"LIBUIPC_SM={value}"])


@pytest.mark.parametrize("capability", ["9.0", "", "8.9\n12.0", "N/A"])
def test_unsupported_or_ambiguous_gpu_never_selects_fallback(monkeypatch, capability):
    monkeypatch.setattr(docker.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=capability))
    with pytest.raises(RuntimeError, match="Unsupported host GPU capability"):
        docker._libuipc_build_args([])


def test_missing_gpu_reports_manual_build_override(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(docker.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="pass --build-arg LIBUIPC_SM=89"):
        docker._libuipc_build_args([])


@pytest.mark.parametrize(
    "variant,profile,expected", [("", "", True), ("5.1", "", False), ("4.5", "", False), ("", "post-train", False)]
)
def test_real_build_command_selects_native_runtime_only_for_prepared_recipes(monkeypatch, variant, profile, expected):
    monkeypatch.setattr(docker, "_active_variant", variant)
    monkeypatch.setattr(docker, "_active_profile", profile)
    monkeypatch.setattr(docker, "_repo_root", lambda: Path(__file__).resolve().parents[3])
    monkeypatch.setattr(docker, "_isaacsim_build_args", lambda _: [])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="8.9\n")

    monkeypatch.setattr(docker.subprocess, "run", run)
    docker._do_build(["--no-resolve"])
    assert (calls[0][0] == "nvidia-smi") is expected
    assert ("LIBUIPC_SM=89" in calls[-1]) is expected


def test_combo_build_omits_native_runtime_even_with_asset_environment(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(docker, "_active_variant", "5.1")
    monkeypatch.setattr(docker, "_active_profile", "")
    monkeypatch.setattr(docker, "_repo_root", lambda: Path(__file__).resolve().parents[3])
    monkeypatch.setattr(docker, "_isaacsim_build_args", lambda _: [])
    monkeypatch.setattr(docker, "_libuipc_build_args", lambda _: pytest.fail("native GPU selection attempted"))
    (tmp_path / "wheels").mkdir()
    monkeypatch.setenv("GENIESIM_ASSETS_SRC", str(tmp_path))
    monkeypatch.setenv("GENIESIM_LIBUIPC_SOURCE", "huggingface")
    monkeypatch.setenv("GENIESIM_LIBUIPC_REVISION", "a" * 40)
    calls = []
    monkeypatch.setattr(docker.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0))
    docker._do_build(["--no-resolve"])
    assert len(calls) == 1 and calls[0][:2] == ["docker", "build"]
    assert not any("LIBUIPC" in arg or "libuipc_assets=" in arg for arg in calls[0])
    docker._print_usage()
    assert "LIBUIPC_SM" not in capsys.readouterr().out


def test_failed_selection_does_not_start_docker(monkeypatch):
    monkeypatch.setattr(docker, "_active_variant", "")
    monkeypatch.setattr(docker, "_active_profile", "")
    monkeypatch.setattr(docker, "_repo_root", lambda: Path(__file__).resolve().parents[3])
    monkeypatch.setattr(docker, "_isaacsim_build_args", lambda _: [])
    calls = []
    monkeypatch.setattr(docker.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(stdout="9.0"))
    with pytest.raises(SystemExit) as error:
        docker._do_build([])
    assert error.value.code == 2
    assert [cmd[0] for cmd in calls] == ["nvidia-smi"]


@pytest.fixture
def native_checkout(monkeypatch, tmp_path):
    script = tmp_path / "docker/install_libuipc.py"
    script.parent.mkdir()
    script.write_text("raise SystemExit(0)\n")
    monkeypatch.setattr(_env, "repo_root", lambda: str(tmp_path))
    monkeypatch.setattr(_env, "repo_path", lambda: None)
    return script


def test_native_only_uses_shared_installer_without_peer_probes_or_prompts(monkeypatch, native_checkout, capsys):
    monkeypatch.setattr(bootstrap, "inspect", lambda _: pytest.fail("peer inspected"))
    monkeypatch.setattr(bootstrap, "_prompt", lambda *a, **k: pytest.fail("prompted"))
    calls = []
    monkeypatch.setattr(bootstrap, "_run_libuipc", lambda cmd: calls.append(cmd) or 0, raising=False)
    bootstrap.run(["--libuipc-only"])
    assert calls == [[sys.executable, str(native_checkout), "--if-needed"]]
    assert "libuipc installed and native library loaded" in capsys.readouterr().out


def test_optional_native_failure_warns_without_reporting_full_success(monkeypatch, native_checkout, capsys):
    monkeypatch.setattr(bootstrap, "_run_libuipc", lambda cmd: 7, raising=False)
    bootstrap.run(["--libuipc-only"])
    output = capsys.readouterr().out
    assert "native installer exited with code 7" in output
    assert "libuipc unavailable (optional)" in output
    assert "🎉" not in output


def test_missing_checkout_warns_without_installation(monkeypatch, native_checkout, capsys):
    native_checkout.unlink()
    monkeypatch.setattr(bootstrap, "_run_libuipc", lambda cmd: pytest.fail("installer started"), raising=False)
    bootstrap.run(["--libuipc-only"])
    assert "Cannot find docker/install_libuipc.py" in capsys.readouterr().out


def test_healthy_default_bootstrap_checks_native_without_reinstalling_peers(monkeypatch, native_checkout):
    monkeypatch.setattr(
        bootstrap,
        "inspect",
        lambda _: dict(installed=True, import_ok=True, deps=[], submodules=[], editable=True, version="3.2.0"),
    )
    monkeypatch.setattr(bootstrap, "_prompt", lambda *a, **k: False)
    calls = []
    monkeypatch.setattr(bootstrap, "_run_libuipc", lambda cmd: calls.append(cmd) or 0, raising=False)
    bootstrap.run([])
    assert calls == [[sys.executable, str(native_checkout), "--if-needed"]]


def test_required_peer_failures_still_fail_after_optional_check(monkeypatch, native_checkout):
    monkeypatch.setattr(bootstrap, "_TARGETS", ("geniesim",))
    monkeypatch.setattr(bootstrap, "inspect", lambda _: dict(installed=False))
    monkeypatch.setattr(bootstrap, "_prompt", lambda *a, **k: True)
    monkeypatch.setattr(bootstrap, "_install_command", lambda _: [sys.executable, "-c", "raise SystemExit(9)"])
    calls = []
    monkeypatch.setattr(bootstrap, "_run_libuipc", lambda cmd: calls.append(cmd) or 0, raising=False)
    with pytest.raises(SystemExit) as error:
        bootstrap.run([])
    assert error.value.code == 1
    assert calls == [[sys.executable, str(native_checkout), "--if-needed"]]


def test_installer_timeout_terminates_process_group(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "_LIBUIPC_TIMEOUT", 0.3, raising=False)
    monkeypatch.setattr(bootstrap, "_LIBUIPC_STOP_TIMEOUT", 0.1, raising=False)
    marker = tmp_path / "pid"
    code = "import os,time; from pathlib import Path; Path(%r).write_text(str(os.getpid())); time.sleep(10)" % str(
        marker
    )
    with pytest.raises(RuntimeError, match="timed out"):
        bootstrap._run_libuipc([sys.executable, "-c", code])
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


@pytest.mark.parametrize("entrypoint", ["direct", "cli"])
def test_sigterm_cleans_up_detached_installer_and_preserves_cancellation(tmp_path, entrypoint):
    marker = tmp_path / "pid"
    child = "import os,time; from pathlib import Path; Path(%r).write_text(str(os.getpid())); time.sleep(30)" % str(
        marker
    )
    wrapper = (
        "import sys; from geniesim_cli.commands.bootstrap import _run_libuipc; "
        f"_run_libuipc([sys.executable, '-c', {child!r}])"
    )
    script = tmp_path / "docker/install_libuipc.py"
    script.parent.mkdir()
    script.write_text(child)
    command = [sys.executable, "-c", wrapper]
    if entrypoint == "cli":
        command = [sys.executable, "-m", "geniesim_cli", "bootstrap", "--libuipc-only"]
    process = subprocess.Popen(
        command,
        env={**os.environ, "GENIESIM_REPO_ROOT": str(tmp_path)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid = None
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), process.poll()
        pid = int(marker.read_text())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=5)
        assert process.returncode == 143
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_installer_restores_sigterm_handler():
    handler = signal.getsignal(signal.SIGTERM)
    assert bootstrap._run_libuipc([sys.executable, "-c", "pass"]) == 0
    assert signal.getsignal(signal.SIGTERM) == handler


def test_base_build_uses_local_wheel_context_and_download_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(docker, "_active_variant", "")
    monkeypatch.setattr(docker, "_active_profile", "")
    monkeypatch.setattr(docker, "_repo_root", lambda: Path(__file__).resolve().parents[3])
    monkeypatch.setattr(docker, "_isaacsim_build_args", lambda _: [])
    wheels = tmp_path / "assets" / "wheels"
    wheels.mkdir(parents=True)
    monkeypatch.setenv("GENIESIM_ASSETS_SRC", str(wheels.parent))
    monkeypatch.setenv("GENIESIM_LIBUIPC_SOURCE", "huggingface")
    monkeypatch.setenv("GENIESIM_LIBUIPC_REVISION", "a" * 40)
    calls = []
    monkeypatch.setattr(docker.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0))
    docker._do_build(["--no-resolve", "--build-arg", "LIBUIPC_SM=89", "--build-arg", "LIBUIPC_SOURCE=modelscope"])
    command = calls[-1]
    assert f"libuipc_assets={wheels}" in command
    assert "LIBUIPC_SOURCE=modelscope" in command
    assert "LIBUIPC_SOURCE=huggingface" not in command
    assert "LIBUIPC_REVISION=" + "a" * 40 in command


def test_explicit_native_asset_context_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setattr(docker, "_active_variant", "")
    monkeypatch.setattr(docker, "_active_profile", "")
    monkeypatch.setattr(docker, "_repo_root", lambda: Path(__file__).resolve().parents[3])
    monkeypatch.setattr(docker, "_isaacsim_build_args", lambda _: [])
    (tmp_path / "wheels").mkdir()
    monkeypatch.setenv("GENIESIM_ASSETS_SRC", str(tmp_path))
    calls = []
    monkeypatch.setattr(docker.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0))
    docker._do_build(
        ["--no-resolve", "--build-arg", "LIBUIPC_SM=89", "--build-context", "libuipc_assets=/explicit/wheels"]
    )
    assert calls[-1].count("--build-context") == 1
    assert "libuipc_assets=/explicit/wheels" in calls[-1]
