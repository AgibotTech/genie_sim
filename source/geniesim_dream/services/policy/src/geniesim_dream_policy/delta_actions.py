"""Runtime action-space selection for pi0.5 checkpoint assets."""

from __future__ import annotations

import json
from pathlib import Path

from openpi import transforms

ARM_DIMS = 14
GRIPPER_DIMS = 2
A2D_DELTA_MASK: tuple[bool, ...] = transforms.make_bool_mask(ARM_DIMS, -GRIPPER_DIMS)
MARKER_NAME = "action_space.json"
ABSOLUTE = "absolute"
DELTA = "delta"


def space_name(delta: bool) -> str:
    return DELTA if delta else ABSOLUTE


def install(group: transforms.Group) -> transforms.Group:
    """Apply the same state-relative mask in both serving directions."""
    for transform in (*group.inputs, *group.outputs):
        if isinstance(transform, transforms.DeltaActions | transforms.AbsoluteActions):
            raise ValueError("Action-space transforms are already installed")
    return transforms.Group(
        inputs=(*group.inputs, transforms.DeltaActions(A2D_DELTA_MASK)),
        outputs=(transforms.AbsoluteActions(A2D_DELTA_MASK), *group.outputs),
    )


def read_marker(assets_dir: str | Path, asset_id: str) -> bool | None:
    path = Path(assets_dir) / asset_id / MARKER_NAME
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get("action_space")
    if value not in (ABSOLUTE, DELTA):
        raise ValueError(f"{path} has invalid action_space: {value!r}")
    return value == DELTA
