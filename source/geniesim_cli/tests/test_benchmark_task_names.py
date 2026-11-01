# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""New task names resolve and filter without changing their runtime payloads."""

import hashlib
import json
from pathlib import Path

import pytest

from geniesim_cli.commands import benchmark

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "source/geniesim_benchmark/src/geniesim_benchmark"
MANIFEST_PATH = PACKAGE / "benchmark/config/task_entry_names.json"
if not MANIFEST_PATH.is_file():
    pytest.skip("requires the benchmark task-entry catalog", allow_module_level=True)
MANIFEST = json.loads(MANIFEST_PATH.read_text())
ENTRIES = MANIFEST["entries"]
CATEGORIES = {entry["category"] for entry in ENTRIES}


@pytest.mark.parametrize("entry", ENTRIES, ids=[entry["name"] for entry in ENTRIES])
def test_each_new_entry_resolves_filters_and_preserves_configuration(monkeypatch, entry):
    monkeypatch.setattr(benchmark, "_benchmark_root", lambda: PACKAGE)
    name = entry["name"]
    path = PACKAGE / "config" / f"{name}.yaml"
    assert benchmark._resolve_config(name) == path.resolve()
    assert benchmark._split_name(name) == (entry["robot"], entry["category"])
    assert path in benchmark._filter_configs(entry["robot"], entry["category"], name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("category", sorted(CATEGORIES))
def test_category_filters_include_every_declared_task(monkeypatch, category):
    monkeypatch.setattr(benchmark, "_benchmark_root", lambda: PACKAGE)
    found = {path.stem for path in benchmark._filter_configs(None, category, None)}
    expected = {entry["name"] for entry in ENTRIES if entry["category"] == category}
    assert expected <= found


def test_inventory_is_unique_and_complete():
    assert len(ENTRIES) == len({entry["name"] for entry in ENTRIES})
    assert {"finemanip", "dexmanip", "wbc", "crsemb"} <= CATEGORIES
    for category, count in (("finemanip", 8), ("wbc", 8), ("crsemb", 24)):
        assert sum(entry["category"] == category for entry in ENTRIES) == count
    assert not any("ipc" in entry["name"] for entry in ENTRIES)


@pytest.mark.parametrize(
    "name,module",
    [
        ("g2crsbwuji_dexmanip_wok_flip", "geniesim_benchmark.newton.runner"),
        ("g2crsb90d_finemanip_phone_packaging", None),
    ],
)
def test_renamed_entries_keep_newton_and_isaac_dispatch(monkeypatch, name, module):
    monkeypatch.setattr(benchmark, "_benchmark_root", lambda: PACKAGE)
    monkeypatch.setattr(benchmark, "_python_cmd", lambda: "isaac-python")
    monkeypatch.setattr(benchmark, "_rt_python_cmd", lambda: "rt-python")
    path = benchmark._resolve_config(name)
    cmd = benchmark._build_app_cmd(path, [], "127.0.0.1:8999")
    assert str(path) in cmd
    assert "--benchmark.infer_host=127.0.0.1:8999" in cmd
    if module:
        assert cmd[1:3] == ["-m", module]
    else:
        assert cmd[1] == str(PACKAGE / "app/app.py")
