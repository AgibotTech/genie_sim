# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import io

import numpy as np
import pytest
from PIL import Image

from agent2act.adapters.planners.rgb_matching import match_rgb_patch
from agent2act.domain.errors import ContractError


def jpeg(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, "JPEG", quality=95)
    return stream.getvalue()


def texture():
    return np.random.default_rng(91).integers(0, 256, (160, 200), dtype=np.uint8)


@pytest.mark.parametrize("dx,dy", [(9, -6), (-11, 8), (0, 0)])
def test_distinct_texture_translates_without_claiming_metric_motion(dx, dy):
    before = texture()
    after = np.roll(before, (dy, dx), axis=(0, 1))
    result = match_rgb_patch(jpeg(before), jpeg(after), 500, 500)
    assert result["status"] == "matched"
    assert result["delta_xy"] == pytest.approx([1000 * dx / 199, 1000 * dy / 159])
    assert result["motion_executed"] is False
    assert "3D" in result["meaning"]
    assert "depth" not in result


def test_flat_repeated_occluded_and_out_of_search_motion_abstain():
    flat = np.full((160, 200), 128, dtype=np.uint8)
    repeated = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (80, 100))
    before = texture()
    occluded = before.copy()
    occluded[60:110, 75:130] = 128
    far = np.roll(before, 50, axis=1)
    for a, b in [(flat, flat), (repeated, repeated), (before, occluded), (before, far)]:
        result = match_rgb_patch(jpeg(a), jpeg(b), 500, 500)
        assert result["status"] == "unavailable"
        assert "current_xy" not in result
        assert "delta_xy" not in result


def test_frame_dimensions_boundaries_and_nonfinite_points():
    encoded = jpeg(texture())
    assert match_rgb_patch(encoded, encoded, 0, 0)["status"] == "unavailable"
    with pytest.raises(ContractError, match="dimensions"):
        match_rgb_patch(encoded, jpeg(texture()[:100]), 500, 500)
    for x in [float("nan"), float("inf"), -1, 1001, True]:
        with pytest.raises(ContractError):
            match_rgb_patch(encoded, encoded, x, 500)
