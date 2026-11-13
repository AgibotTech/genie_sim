# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Public defaults retain the original 950 episodes and exact layout resources."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

PACKAGE = Path(__file__).resolve().parents[1] / "src/geniesim_benchmark"
INVENTORY = json.loads((Path(__file__).parent / "fixtures/public_layouts.json").read_text())


@pytest.mark.parametrize("name,entry", INVENTORY["tasks"].items())
def test_public_task_keeps_original_config_and_layouts(name, entry):
    config = PACKAGE / "config" / (name + ".yaml")
    data = yaml.safe_load(config.read_text())
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == entry["config_value_sha256"]
    assert data["benchmark"]["num_instances"] == entry["episodes"]
    assert data["benchmark"].get("num_episode", data.get("episodes", 1)) == 1
    if "layout_dir" in entry:
        folder = PACKAGE / entry["layout_dir"]
        assert {p.name for p in folder.iterdir() if p.is_dir() and p.name.isdigit()} == {str(i) for i in range(20)}
        digest = hashlib.sha256()
        files = sorted(p for p in folder.rglob("*") if p.is_file())
        assert len(files) == entry["layout_files"]
        for path in files:
            digest.update(
                (
                    path.relative_to(folder).as_posix() + "\0" + hashlib.sha256(path.read_bytes()).hexdigest() + "\n"
                ).encode()
            )
        assert digest.hexdigest() == entry["layout_sha256"]


def test_public_suite_has_950_episodes_and_original_rubik_catalogs():
    assert len(INVENTORY["tasks"]) == 50
    assert sum(row["episodes"] for row in INVENTORY["tasks"].values()) == INVENTORY["total_episodes"] == 950
    for name, expected in INVENTORY["catalogs"].items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == expected


def test_rubik_public_entry_selects_ten_distinct_original_scenes(tmp_path):
    from geniesim_benchmark.backends.mujoco.rubik.config import load_config
    from geniesim_benchmark.backends.mujoco.rubik.scenes import SceneCatalog, episode_scenes

    cfg = load_config(
        PACKAGE / "config/g2crsbwuji_dexmanip_rubik_cube_solve.yaml",
        [f"--rubik.package_root={tmp_path}", "--benchmark.policy=hold"],
    )
    assert Path(cfg.rubik.scene_archive).name == "rubik_2turn_10_ranked.zip"
    scenes = episode_scenes(
        SceneCatalog(cfg.rubik.scene_archive),
        cfg.benchmark.task_name,
        cfg.benchmark.seed,
        cfg.benchmark.num_instances,
        cfg.benchmark.num_episode,
    )
    assert len(scenes) == len({scene.identifier for scene in scenes}) == 10
