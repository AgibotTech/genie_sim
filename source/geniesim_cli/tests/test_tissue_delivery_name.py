# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Task-entry renaming must not duplicate evaluation or change its inputs."""

from pathlib import Path
import json

from geniesim_cli.commands import benchmark


PACKAGE = Path(__file__).resolve().parents[3] / "source/geniesim_benchmark/src/geniesim_benchmark"
CANONICAL = "g2crsb90d_wbc_deliver_tissues"
LEGACY = "g2crsb90d_wbc_sending_food"


def test_canonical_and_legacy_names_resolve_one_task(monkeypatch):
    monkeypatch.setattr(benchmark, "_benchmark_root", lambda: PACKAGE)
    path = PACKAGE / "config" / (CANONICAL + ".yaml")
    for name in (CANONICAL, LEGACY, CANONICAL + ".yaml", LEGACY + ".yaml"):
        assert benchmark._resolve_config(name) == path.resolve()
    listed = benchmark._filter_configs(None, "wbc", None)
    assert path in listed
    assert not any(p.stem == LEGACY for p in listed)
    assert benchmark._filter_configs(None, "wbc", LEGACY) == [path]
    assert "sub_task_name: sending_food" in path.read_text()
    entries = json.loads((PACKAGE / "benchmark/config/task_entry_names.json").read_text())["entries"]
    assert sum(row["name"] == CANONICAL for row in entries) == 1
    assert not any(row["name"] == LEGACY for row in entries)


def test_hidden_recipe_uses_canonical_entry_with_legacy_scene_key():
    templates = PACKAGE / "benchmark/config/hidden_templates/config"
    if templates.is_dir():
        assert "sub_task_name: sending_food" in (templates / (CANONICAL + ".yaml")).read_text()
        assert not (templates / (LEGACY + ".yaml")).exists()
