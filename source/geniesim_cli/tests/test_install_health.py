# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Diagnostics and recovery must work before the rest of the stack is installed."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from geniesim_cli.commands import doctor, status


@pytest.fixture
def standalone_cli(tmp_path):
    site = tmp_path / "site-packages"
    package = Path(status.__file__).resolve().parents[1]
    shutil.copytree(package, site / "geniesim_cli", ignore=shutil.ignore_patterns("__pycache__"))
    metadata = site / "geniesim_cli-3.2.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: geniesim_cli\nVersion: 3.2.0\n")
    if sys.version_info < (3, 11):
        import tomli

        shutil.copytree(Path(tomli.__file__).parent, site / "tomli")
    empty = tmp_path / "empty"
    empty.mkdir()

    def run(*args, cwd=empty):
        return subprocess.run(
            [sys.executable, "-S", *args],
            cwd=cwd,
            env={**os.environ, "PYTHONPATH": str(site), "NO_COLOR": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )

    return run


@pytest.mark.parametrize("args", [("status",), ("bootstrap", "--help")])
def test_standalone_diagnostics_without_umbrella(standalone_cli, args):
    result = standalone_cli("-m", "geniesim_cli", *args)
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    if args[0] == "status":
        assert "geniesim_assets" in result.stdout
        assert "Some components need attention" in result.stdout
    else:
        assert "Usage: geniesim bootstrap" in result.stdout
        assert "Install plan" not in result.stdout


def test_wheel_uses_current_checkout_for_tiers_and_recovery(standalone_cli, tmp_path):
    repo = tmp_path / "checkout"
    umbrella = repo / "source/geniesim"
    umbrella.mkdir(parents=True)
    (umbrella / "pyproject.toml").write_text(
        '[project]\nname = "geniesim"\ndependencies = ["geniesim_cli", "geniesim_test_peer"]\n'
    )
    peer = repo / "source/geniesim_test_peer"
    peer.mkdir()
    (peer / "pyproject.toml").write_text('[project]\nname = "geniesim_test_peer"\n')
    nested = repo / "nested"
    nested.mkdir()
    result = standalone_cli(
        "-c",
        "import json; from geniesim_cli._tiers import tier1; "
        "from geniesim_cli.commands.bootstrap import _install_command; "
        "print(json.dumps([tier1(), _install_command('geniesim_test_peer')]))",
        cwd=nested,
    )
    assert result.returncode == 0, result.stderr
    tiers, command = json.loads(result.stdout)
    assert tiers == ["geniesim_cli", "geniesim_test_peer"]
    assert command[-2:] == ["-e", str(peer)]


def test_bootstrap_without_sources_reports_recovery_steps(standalone_cli):
    result = standalone_cli(
        "-c",
        "from geniesim_cli.commands import bootstrap; "
        "bootstrap._install_command = lambda _: None; bootstrap.run([])",
    )
    assert result.returncode == 1
    assert "must be installed from a local checkout" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "tier,installed,broken,healthy",
    [
        ("optional", False, None, True),
        ("required", False, None, False),
        ("optional", True, "import", False),
        ("optional", True, "dependency", False),
        ("optional", True, "submodule", False),
    ],
)
def test_doctor_and_status_agree(monkeypatch, capsys, tier, installed, broken, healthy):
    base = dict(status.DISTRIBUTIONS[0])
    peer = {**base, "dist": "geniesim_test_peer", "tier": tier}
    monkeypatch.setattr(status, "DISTRIBUTIONS", (base, peer))

    def inspect(spec):
        is_peer = spec is peer
        present = installed if is_peer else True
        return {
            "installed": present,
            "version": "3.2.0" if present else None,
            "editable": False,
            "location": None,
            "import_ok": not (is_peer and broken == "import") if present else None,
            "import_err": None,
            "deps": [("dependency", "dependency", False, "missing")] if is_peer and broken == "dependency" else [],
            "submodules": [("submodule", False, "missing")] if is_peer and broken == "submodule" else [],
            "extras": [],
        }

    monkeypatch.setattr(status, "inspect", inspect)
    with pytest.raises(SystemExit) as exit_info:
        status.run([])
    assert exit_info.value.code == 0
    assert ("All systems go!" in capsys.readouterr().out) is healthy
    verdict, lines, fix, _ = doctor._check_stack()
    assert verdict == ("ok" if healthy else "fail")
    if healthy:
        assert fix is None
        assert "optional, not installed" in "\n".join(lines)
