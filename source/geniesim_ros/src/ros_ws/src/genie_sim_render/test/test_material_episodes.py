# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).parents[1] / "scripts/_material_generalization.py"
spec = importlib.util.spec_from_file_location("material_episodes", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_material_selection_tracks_epoch_and_never_reuses_previous_receipt():
    entries = [{"episode_plan": {"id": i}, "receipt_path": f"episode{i}/receipt.json"} for i in range(3)]
    episodes = module.MaterialEpisodes(dict(entries[0], episode_sequence=entries))
    assert episodes.select({"episode_index": 0, "epoch": 0}) == ({"id": 0}, "episode0/receipt.json", False)
    assert episodes.select({"episode_index": 1, "epoch": 1}) == ({"id": 1}, "episode1/receipt.json", True)
    assert episodes.select({"episode_index": 1, "epoch": 1})[-1] is False
    for index, epoch in [(0, 0), (2, 1), (3, 3)]:
        with pytest.raises(ValueError):
            episodes.select({"episode_index": index, "epoch": epoch})
