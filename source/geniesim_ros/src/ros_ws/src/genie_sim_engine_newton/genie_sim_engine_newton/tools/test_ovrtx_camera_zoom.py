#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure regression tests for inline-OVRtx FreeCam optical zoom."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_MODULE = _HERE.parent / "engine" / "newton" / "visualizers" / "camera_zoom.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ovrtx_camera_zoom_test", _MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.01, 0.25),
        (0.25, 0.25),
        (1.0, 1.0),
        (8.0, 8.0),
        (99.0, 8.0),
    ],
)
def test_clamp_zoom_factor(value, expected):
    module = _load_module()
    assert module.clamp_zoom_factor(value) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_clamp_zoom_factor_rejects_non_finite_values(value):
    module = _load_module()
    with pytest.raises(ValueError, match="finite"):
        module.clamp_zoom_factor(value)


def test_scaled_intrinsics_match_optical_focal_length_zoom():
    module = _load_module()
    K, P = module.scaled_intrinsics(
        base_fx=640.0,
        base_fy=640.0,
        cx=640.0,
        cy=360.0,
        zoom=2.0,
    )

    assert K.dtype == np.float64
    assert P.dtype == np.float64
    assert K.tolist() == [1280.0, 0.0, 640.0, 0.0, 1280.0, 360.0, 0.0, 0.0, 1.0]
    assert P.tolist() == [
        1280.0,
        0.0,
        640.0,
        0.0,
        0.0,
        1280.0,
        360.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]


def test_visualizer_applies_zoom_to_focal_length_and_camera_info(monkeypatch):
    pytest.importorskip("warp")
    pytest.importorskip("genie_sim_engine_py")
    sys.path.insert(0, str(_HERE.parents[1]))

    from genie_sim_engine_newton.engine.newton.visualizers import ovrtx

    class _Core:
        @staticmethod
        def take_free_cam_zoom():
            return 2.0

    class _Renderer:
        def __init__(self):
            self.writes = []

        def write_attribute(self, paths, name, values):
            self.writes.append((paths, name, values.copy()))

    cam = ovrtx.CameraCfg(
        path="FreeCam",
        topic="/free_cam",
        width=1280,
        height=720,
        fx=640.0,
        fy=640.0,
        cx=640.0,
        cy=360.0,
        is_free_cam=True,
    )
    viz = ovrtx.InlineOvrtxVisualizer.__new__(ovrtx.InlineOvrtxVisualizer)
    viz._camera_zoom_topic = "/test/camera_zoom"
    viz._free_cam_cfg = cam
    viz._free_cam_prim_path = "/RenderOVRTX/Cameras/FreeCam"
    viz._free_cam_base_fx = 640.0
    viz._free_cam_base_fy = 640.0
    viz._free_cam_base_focal_length = 18.0
    viz._free_cam_zoom = 1.0
    viz._renderer = _Renderer()
    monkeypatch.setattr(ovrtx, "_core", _Core())

    viz._apply_free_cam_zoom()

    assert len(viz._renderer.writes) == 1
    paths, name, values = viz._renderer.writes[0]
    assert paths == ["/RenderOVRTX/Cameras/FreeCam"]
    assert name == "focalLength"
    assert values.dtype == np.float32
    assert values.tolist() == [36.0]
    assert cam.fx == 1280.0
    assert cam.fy == 1280.0
    assert cam.K_np[[0, 4]].tolist() == [1280.0, 1280.0]
    assert cam.P_np[[0, 5]].tolist() == [1280.0, 1280.0]
