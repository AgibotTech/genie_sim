# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Keep Isaac consumers usable while supporting the optional runtime selector."""

import sys
from types import ModuleType

import pytest

from geniesim_cli.commands import benchmark


@pytest.mark.parametrize("runtime", [None, "isaac", "newton", "libuipc"])
def test_runtime_selection_preserves_config_and_forwarded_arguments(monkeypatch, tmp_path, runtime):
    package = tmp_path / "benchmark with spaces"
    config = package / "config/task.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n")
    monkeypatch.setattr(benchmark, "_benchmark_root", lambda: package)
    monkeypatch.setattr(benchmark, "_python_cmd", lambda: "isaac-python")
    monkeypatch.setenv("GENIESIM_PY_CMD", "newton-python")
    selector = None
    if runtime is not None:
        selector = ModuleType("geniesim_benchmark.runtime")
        selector.runtime_for_config = lambda path, overrides=(): runtime
    monkeypatch.setitem(sys.modules, "geniesim_benchmark.runtime", selector)

    result = benchmark._build_app_cmd(config, ["--literal", "value with spaces"], "localhost:8999")

    expected = (
        ["newton-python", "-m", f"geniesim_benchmark.{runtime}.runner"]
        if runtime in ("newton", "libuipc")
        else ["isaac-python", str(package / "app/app.py")]
    )
    assert result == [
        *expected,
        "--config",
        str(config),
        "--benchmark.infer_host=localhost:8999",
        "--literal",
        "value with spaces",
    ]


def test_layout_command_reports_missing_consumer_before_exec(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(benchmark, "_benchmark_root", lambda: tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        benchmark._do_layouts("layouts", ["--help"])
    assert exit_info.value.code == 1
    assert "does not provide Newton layout commands" in capsys.readouterr().err
