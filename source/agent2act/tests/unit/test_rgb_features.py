# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from agent2act.adapters.planners.rgb_features import match_rgb_feature_patch
from agent2act.domain.models import ContractError


def jpeg(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, "JPEG", quality=95)
    return stream.getvalue()


def texture():
    noise = np.random.default_rng(15).integers(0, 256, (320, 400), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 0.8)


@pytest.mark.parametrize("angle,scale,dx,dy", [(0, 1, 70, 15), (12, 0.82, 60, -20), (-15, 1.2, -50, 20)])
def test_scaled_rotated_and_large_translation_match_known_material_point(angle, scale, dx, dy):
    before = texture()
    transform = cv2.getRotationMatrix2D((199.5, 159.5), angle, scale)
    transform[:, 2] += [dx, dy]
    after = cv2.warpAffine(before, transform, (400, 320))
    result = match_rgb_feature_patch(jpeg(before), jpeg(after), 500, 500)
    assert result["status"] == "matched", result
    expected = transform @ [199.5, 159.5, 1]
    actual = np.array(result["current_xy"]) / 1000 * [399, 319]
    assert np.linalg.norm(actual - expected) < 0.8
    assert result["motion_executed"] is False
    assert result["support_points"] >= 6
    assert "3D" in result["meaning"]


def test_occluded_point_abstains_even_when_surrounding_texture_supports_model():
    before = texture()
    after = np.roll(before, 65, axis=1)
    after[149:171, 254:276] = 128
    result = match_rgb_feature_patch(jpeg(before), jpeg(after), 500, 500)
    assert result["status"] == "unavailable", result
    assert "current_xy" not in result


def test_flat_and_duplicate_surface_identity_are_not_inferred():
    before = texture()
    duplicated = np.zeros_like(before)
    duplicated[100:220, 30:150] = before[100:220, 140:260]
    duplicated[100:220, 250:370] = before[100:220, 140:260]
    flat = np.full_like(before, 128)
    for after in (flat, duplicated):
        result = match_rgb_feature_patch(jpeg(before), jpeg(after), 500, 500)
        assert result["status"] == "unavailable", result
        assert "current_xy" not in result


def test_point_outside_feature_support_does_not_extrapolate():
    before = np.full((320, 400), 128, np.uint8)
    before[80:160, 120:280] = texture()[80:160, 120:280]
    # A separate tiny textured point below a rich textured surface must not inherit
    # that surface's displacement simply because nearby affine matches exist.
    before[172:180, 196:204] = texture()[172:180, 196:204]
    after = np.roll(before, 60, axis=1)
    result = match_rgb_feature_patch(jpeg(before), jpeg(after), 500, 550)
    assert result["status"] == "unavailable", result


def test_affine_input_validation_and_boundaries():
    encoded = jpeg(texture())
    assert match_rgb_feature_patch(encoded, encoded, 0, 0)["status"] == "unavailable"
    for x in (True, float("nan"), -1, 1001):
        with pytest.raises(ContractError):
            match_rgb_feature_patch(encoded, encoded, x, 500)
    with pytest.raises(ContractError, match="dimensions"):
        match_rgb_feature_patch(encoded, jpeg(texture()[:100]), 500, 500)
