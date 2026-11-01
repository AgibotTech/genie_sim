# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""A prepared native benchmark must not trigger unrelated engine installation."""

import pytest

from geniesim_cli import cli
from geniesim_cli.commands import benchmark


def test_native_run_bypasses_umbrella_bootstrap(monkeypatch, tmp_path):
    config = tmp_path / "native.yaml"
    config.write_text("benchmark:\n  backend: mujoco\n")

    def unexpected_bootstrap(command):
        pytest.fail("native benchmark tried to bootstrap Isaac/ROS")

    calls = []
    monkeypatch.setattr(cli, "_ensure_bootstrap", unexpected_bootstrap)
    monkeypatch.setattr(benchmark, "run", calls.append)
    with pytest.raises(SystemExit) as result:
        cli.main(["benchmark", "run", str(config)])
    assert result.value.code == 0
    assert calls == [["run", str(config)]]
