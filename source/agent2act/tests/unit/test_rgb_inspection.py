# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import base64
import io
from copy import deepcopy

import pytest
from PIL import Image

from agent2act.adapters.planners.rgb_inspection import inspect_rgb_region
from agent2act.domain.errors import ContractError


def rgb(width=100, height=80):
    image = Image.new("RGB", (width, height), (20, 60, 120))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def test_crop_preserves_source_pixels_and_ignores_depth(observation):
    raw = rgb()
    observation.images = {
        "head": {"encoding": "JPEG", "image_data": raw},
        "head_depth": {"encoding": "RAW", "image_data": b"forbidden"},
    }
    before = deepcopy(observation.images)
    result = inspect_rgb_region(observation, dict(camera="head", left=100, top=250, right=600, bottom=750))
    assert result["source_box_pixels"] == [10, 20, 60, 60]
    assert result["crop_size"] == [50, 40]
    output = Image.open(io.BytesIO(base64.b64decode(result["image_url"].split(",")[1])))
    source = Image.open(io.BytesIO(raw)).convert("RGB").crop((10, 20, 60, 60))
    assert output.tobytes() == source.tobytes()
    assert observation.images == before


@pytest.mark.parametrize(
    "changes",
    [
        {"camera": "head_depth"},
        {"camera": "other"},
        {"left": True},
        {"top": float("nan")},
        {"right": -1},
        {"left": 600, "right": 500},
        {"right": 1},
        {"extra": 1},
    ],
)
def test_invalid_region_rejected(observation, changes):
    observation.images = {"head": {"encoding": "JPEG", "image_data": rgb()}}
    with pytest.raises(ContractError):
        inspect_rgb_region(
            observation, {"camera": "head", "left": 0, "top": 0, "right": 1000, "bottom": 1000, **changes}
        )


def test_large_crop_is_bounded_without_upscaling(observation):
    observation.images = {"head": {"encoding": "JPEG", "image_data": rgb(1600, 1000)}}
    result = inspect_rgb_region(observation, dict(camera="head", left=0, top=0, right=1000, bottom=1000))
    assert result["crop_size"] == [768, 480]


def test_missing_or_corrupt_current_rgb_rejected(observation):
    args = dict(camera="head", left=0, top=0, right=1000, bottom=1000)
    observation.images = {}
    with pytest.raises(ContractError, match="unavailable"):
        inspect_rgb_region(observation, args)
    observation.images = {"head": {"encoding": "JPEG", "image_data": b"bad"}}
    with pytest.raises(ContractError, match="decoded"):
        inspect_rgb_region(observation, args)
