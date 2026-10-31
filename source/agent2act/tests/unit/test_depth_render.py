# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import base64
import io
import json
import struct

import pytest
from PIL import Image

from agent2act.adapters.planners.vision import (
    append_camera_inputs,
    depth_render,
    depth_summary,
    encode_raw_depth,
)
from agent2act.service.trace import RequestTrace


def raw_depth(width, height, values):
    return {
        "encoding": "RAW_UINT16",
        "width": width,
        "height": height,
        "image_data": struct.pack(f"<{width * height}H", *values),
    }


def test_depth_render_preserves_near_far_and_invalid_pixels():
    metadata, png = depth_render(raw_depth(2, 2, [0, 500, 65535, 1500]), None)
    with Image.open(io.BytesIO(png)) as im:
        assert im.size == (2, 2)
        assert list(im.tobytes()) == [0, 255, 0, 64]
    assert metadata["render"]["near_mm"] == 500
    assert metadata["render"]["far_mm"] == 1500
    assert metadata["render"]["valid_fraction"] == 0.5


@pytest.mark.parametrize("values", [[0] * 4, [65535] * 4])
def test_invalid_depth_has_no_misleading_image(values):
    metadata, png = depth_render(raw_depth(2, 2, values), 640)
    assert not metadata["available"]
    assert png is None
    assert depth_summary(raw_depth(2, 2, values))["available"] is False


def test_constant_depth_and_bounded_nearest_resize():
    metadata, png = depth_render(raw_depth(8, 4, [1200] * 32), 4)
    with Image.open(io.BytesIO(png)) as im:
        assert im.size == (4, 2)
        assert set(im.tobytes()) == {255}
    assert metadata["render"]["center_mm"] == 1200
    assert metadata["render"]["far_mm"] > metadata["render"]["near_mm"]


def test_malformed_depth_is_reported_without_crashing():
    metadata, png = depth_render({"encoding": "RAW_UINT16", "width": 2, "height": 2}, None)
    assert not metadata["available"]
    assert png is None


def test_depth_toggle_sends_lossless_raw_depth_and_trace(tmp_path, monkeypatch):
    rgb = io.BytesIO()
    Image.new("RGB", (2, 2)).save(rgb, "JPEG")
    images = {
        "head": {"encoding": "JPEG", "image_data": rgb.getvalue()},
        "head_depth": raw_depth(2, 2, [500, 600, 700, 800]),
    }
    disabled = []
    append_camera_inputs(disabled, images, "current", 640, 85, False)
    assert len(disabled) == 2

    def unexpected_render(*args, **kwargs):
        pytest.fail("planner must not generate a depth visualization")

    monkeypatch.setattr("agent2act.adapters.planners.vision.depth_render", unexpected_render)
    enabled = []
    append_camera_inputs(enabled, images, "previous", 640, 85, True)
    calibration = json.loads(enabled[2]["text"])["calibration"]
    assert calibration["available"] is False
    label = json.loads(enabled[-1]["text"])
    assert label["image_role"] == "previous"
    assert label["depth"]["available"] is True
    assert label["depth"]["units"] == "mm"
    assert "render" not in label["depth"]
    assert len([part for part in enabled if part["type"] == "input_image"]) == 2
    url = enabled[1]["image_url"]
    assert url.startswith("data:image/jpeg;base64,")
    RequestTrace(tmp_path)("llm-request", enabled)
    saved = json.loads((tmp_path / "001-llm-request.json").read_text())
    ref = saved[1]["image_url"]
    assert ref["representation"] == "data:image/jpeg;base64"
    assert (tmp_path / ref["$artifact"]).read_bytes() == base64.b64decode(url.split(",")[1])
    ref = saved[4]["image_url"]
    assert ref["representation"] == "data:image/png;base64"
    with Image.open(tmp_path / ref["$artifact"]) as im:
        assert list(im.get_flattened_data()) == [500, 600, 700, 800]


def test_raw_depth_preserves_all_16_bit_values_and_full_resolution():
    values = list(range(65536))
    metadata, png = encode_raw_depth(raw_depth(2048, 32, values))
    assert "grid" not in metadata
    assert metadata["invalid_values"] == [0, 65535]
    with Image.open(io.BytesIO(png)) as im:
        assert im.size == (2048, 32)
        assert list(im.get_flattened_data()) == values


def test_depth_tool_mode_keeps_raw_samples_local_and_sends_metadata_only():
    rgb = io.BytesIO()
    Image.new("RGB", (2, 2)).save(rgb, "JPEG")
    images = {
        "head": {"encoding": "JPEG", "image_data": rgb.getvalue()},
        "head_depth": raw_depth(2, 2, [500, 600, 700, 800]),
    }
    content = []
    append_camera_inputs(content, images, "current", 640, 85, True, send_raw_depth_image=False)

    assert len([part for part in content if part["type"] == "input_image"]) == 1
    assert all("data:image/png" not in str(part) for part in content)
    depth = next(json.loads(part["text"])["depth"] for part in content if '"depth"' in part.get("text", ""))
    assert depth["available"] is True
    assert depth["encoding"] == "RAW_UINT16_LOCAL"
    assert depth["source_size"] == [2, 2]
    metadata, png = encode_raw_depth(images["head_depth"], include_png=False)
    assert metadata == depth
    assert png is None


@pytest.mark.parametrize("item", [None, {}, raw_depth(2, 2, [0] * 4) | {"width": 3}])
def test_raw_depth_rejects_missing_or_malformed_payload(item):
    metadata, png = encode_raw_depth(item)
    assert not metadata["available"]
    assert png is None
