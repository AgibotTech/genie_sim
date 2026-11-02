# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Material choice policy shared by legacy Isaac and episode-based runners."""

import numpy as np


def select_materials(material_info, material_config, *, rng=None):
    """Filter, sort and group the existing material bank before sampling once.

    ``None`` preserves the historical global NumPy random stream. A caller can
    supply a local RandomState/Generator without changing that stream.
    """
    rng = np.random if rng is None else rng
    targets = material_config.get("targets", ["door", "table"])
    allowed_names = set(material_config.get("allowed_materials", []))
    share_by_parent = material_config.get("share_by_parent", False)
    chosen_by_parent = {}
    selections = []
    for mesh_path, material_list in material_info.items():
        if not any(keyword in mesh_path for keyword in targets):
            continue
        choices = sorted(
            path for path in material_list
            if not allowed_names or path.rsplit("/", 1)[-1] in allowed_names
        )
        if not choices:
            raise ValueError(f"No allowed materials for {mesh_path}: {sorted(allowed_names)}")
        group = mesh_path.rsplit("/", 1)[0] if share_by_parent else mesh_path
        if group not in chosen_by_parent:
            chosen_by_parent[group] = str(rng.choice(choices))
        selections.append(dict(mesh_path=mesh_path, candidates=choices, material_path=chosen_by_parent[group]))
    return selections
