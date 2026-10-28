# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Tests for :mod:`genie_sim_engine_core.schema.manifest`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genie_sim_engine_core.assemble.manifest import (
    MANIFEST_VERSION,
    RAW_MANIFEST_SENTINEL,
    Manifest as AssembleManifest,
)
from genie_sim_engine_core.schema import CameraEntry, Manifest, SchemaError, SchemaVersionError
from genie_sim_engine_core.schema.manifest import CameraManifest, LidarManifest


def _minimal_manifest_dict(tmp_path: Path) -> dict:
    return {
        "version": MANIFEST_VERSION,
        "physics_engine": "example_engine",
        "base_path": str(tmp_path),
        "stage_dir": "scenes/foo",
        "scene_yaml": str(tmp_path / "scene.yaml"),
        "robot_prefix": "G2",
        "robot_from_urdf": False,
        "scene_usda": "scenes/foo/scene.usda",
        "robot_usda": "robot/G2_test/robot.usda",
        "robot_visual_usda": "",
        "render_layer_usda": "scenes/foo/render_layer.usda",
        "free_cam_prim_path": "/RenderOVRTX/Cameras/FreeCam",
        "cameras": [
            {
                "path": "head_link3/Head_Front_Camera",
                "topic": "/genie_sim/head_front_camera_rgb",
                "depth_topic": "/genie_sim/head_front_camera_depth",
                "render_product_path": "/RenderOVRTX/Cam_0",
                "frame_id": "head_link3",
                "width": 1280,
                "height": 800,
                "fx": 610.65,
                "fy": 610.70,
                "cx": 644.5,
                "cy": 401.2,
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
                "k4": 0.0,
                "model": "",
                "min_range": 0.01,
                "max_range": 10000.0,
                "is_free_cam": False,
                "dds_topic": "/camera/head_color",
                "dds_depth_topic": "/camera/head_depth",
            }
        ],
        "lidars": [],
        "files": {},
        "robot": {},
        "physics": {},
    }


def test_from_dict_reads_flat_fields(tmp_path):
    """Manifest.from_dict resolves relative paths against base_path."""
    m = Manifest.from_dict(_minimal_manifest_dict(tmp_path))
    assert m.version == MANIFEST_VERSION
    assert m.physics_engine == "example_engine"
    assert m.base_path == tmp_path
    assert m.stage_dir == tmp_path / "scenes/foo"
    assert m.scene_usda == tmp_path / "scenes/foo/scene.usda"
    assert m.render_layer_usda == tmp_path / "scenes/foo/render_layer.usda"
    assert m.free_cam_prim_path == "/RenderOVRTX/Cameras/FreeCam"
    assert len(m.cameras) == 1
    assert m.cameras[0].path == "head_link3/Head_Front_Camera"
    assert m.cameras[0].dds_topic == "/camera/head_color"


def test_from_dict_empty_optional_paths_become_none(tmp_path):
    """Empty string path fields resolve to None, not Path('')."""
    d = _minimal_manifest_dict(tmp_path)
    d["robot_visual_usda"] = ""
    m = Manifest.from_dict(d)
    assert m.robot_visual_usda is None


def test_from_dict_background_usda_resolved_when_set(tmp_path):
    """background_usda follows the same base_path-relative convention as scene_usda."""
    d = _minimal_manifest_dict(tmp_path)
    d["background_usda"] = "scenes/foo/background.usda"
    m = Manifest.from_dict(d)
    assert m.background_usda == tmp_path / "scenes/foo/background.usda"


def test_from_dict_missing_background_usda_is_none(tmp_path):
    """Old manifests (pre-E1) lack the key entirely — must parse fine, field is None.

    This is the same-version backward-compatibility contract: no
    MANIFEST_VERSION bump was needed for this additive optional field.
    """
    d = _minimal_manifest_dict(tmp_path)
    assert "background_usda" not in d
    m = Manifest.from_dict(d)
    assert m.background_usda is None


def test_background_usda_roundtrip(tmp_path):
    """write() relativizes background_usda under base_path; read() resolves it back."""
    d = _minimal_manifest_dict(tmp_path)
    d["background_usda"] = "scenes/foo/background.usda"
    m1 = Manifest.from_dict(d)
    out = tmp_path / "manifest.json"
    m1.write(out)
    on_disk = json.loads(out.read_text())
    assert on_disk["background_usda"] == "scenes/foo/background.usda"
    m2 = Manifest.read(out)
    assert m2.background_usda == m1.background_usda


def test_merge_slice_accepts_background_usda(tmp_path):
    """Stages may claim background_usda directly, like scene_usda/robot_usda."""
    m = Manifest(base_path=tmp_path)
    m.merge_slice({"background_usda": "scenes/foo/background.usda"}, "isaac_scene")
    assert m.background_usda == Path("scenes/foo/background.usda")


def test_read_write_roundtrip(tmp_path: Path):
    """Manifest.write → read yields equal in-memory manifest."""
    d = _minimal_manifest_dict(tmp_path)
    m1 = Manifest.from_dict(d)
    out = tmp_path / "manifest.json"
    m1.write(out)
    assert out.exists()
    m2 = Manifest.read(out)
    assert m2.to_dict() == m1.to_dict()


def test_read_version_mismatch(tmp_path: Path):
    """A stale version is a SchemaVersionError, not a KeyError deep in a stage."""
    d = _minimal_manifest_dict(tmp_path)
    d["version"] = MANIFEST_VERSION - 1
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps(d))
    with pytest.raises(SchemaVersionError):
        Manifest.read(out)


def test_read_missing_file(tmp_path: Path):
    with pytest.raises(SchemaError):
        Manifest.read(tmp_path / "no_such.json")


def test_camera_manifest_from_scene_entry():
    """The flatten used by both mujoco and isaacsim producers is centralized."""
    entry = CameraEntry.from_dict(
        {
            "prim_path": "head_link3/Head_Front_Camera",
            "frame_id": "head_link3",
            "topic": {"rgb": "/rgb", "depth": "/depth"},
            "sensor": {"width": 1920, "height": 1080, "min_range": 0.05, "max_range": 20.0},
            "intrinsic": {"fx": 700.0, "fy": 700.0, "cx": 960.0, "cy": 540.0},
            "dds_topic": "/dds/rgb",
            "dds_depth_topic": "/dds/depth",
            "model": "opencvFisheye",
        }
    )
    cam = CameraManifest.from_scene_entry(entry, index=2)
    assert cam.path == "head_link3/Head_Front_Camera"
    assert cam.topic == "/rgb"
    assert cam.depth_topic == "/depth"
    assert cam.render_product_path == "/RenderOVRTX/Cam_2"
    assert cam.width == 1920
    assert cam.height == 1080
    assert cam.fx == 700.0
    assert cam.dds_topic == "/dds/rgb"
    assert cam.model == "opencvFisheye"


def test_camera_manifest_required_path():
    with pytest.raises(SchemaError):
        CameraManifest.from_dict({"topic": "/x"})  # no path


def test_camera_manifest_freecam_allows_missing_topic():
    """Free-cam entries with is_free_cam=True don't need a topic — matches legacy."""
    cam = CameraManifest.from_dict(
        {
            "path": "FreeCam",
            "topic": "",
            "is_free_cam": True,
        }
    )
    assert cam.is_free_cam is True
    assert cam.topic == ""


def test_merge_slice_valid_keys(tmp_path):
    m = Manifest(base_path=tmp_path)
    m.merge_slice({"physics": {"newton": {"solver": "mjwarp"}}}, "newton_extras")
    m.merge_slice({"robot": {"urdf_sha256": "abc"}}, "urdf_to_usd")
    m.merge_slice(
        {
            "cameras": [
                CameraManifest(path="a/b", topic="/rgb"),
            ]
        },
        "isaac_scene",
    )
    assert m.physics["newton"]["solver"] == "mjwarp"
    assert m.robot["urdf_sha256"] == "abc"
    assert m.cameras[0].path == "a/b"


def test_merge_slice_rejects_core_owned_key():
    m = Manifest()
    with pytest.raises(ValueError):
        m.merge_slice({"version": 999}, "bad_stage")


def test_raw_manifest_sentinel_retired():
    """RAW_MANIFEST_SENTINEL is retired — any use raises loudly.

    Ensures no producer accidentally regresses to the pre-schema
    dict-shipping pattern. If a legitimate need for a byte-preserving
    escape hatch resurfaces, that's a schema-versioning discussion,
    not a silent revival.
    """
    # Using the sentinel as a dict key hashes it → raises.
    with pytest.raises(RuntimeError):
        {RAW_MANIFEST_SENTINEL: {}}  # noqa: B018 — intentional trigger


def test_raw_manifest_sentinel_repr_mentions_retirement():
    """The tombstone repr guides operators back to the typed API."""
    assert "retired" in repr(RAW_MANIFEST_SENTINEL)
    assert "Manifest.merge_slice" in repr(RAW_MANIFEST_SENTINEL)


def test_files_slice(tmp_path):
    m = Manifest(base_path=tmp_path)
    m.merge_slice({"files": {"robot.usda": "robot/G2/robot.usda"}}, "urdf_to_usd")
    assert m.files == {"robot.usda": "robot/G2/robot.usda"}


def test_scene_usda_slice(tmp_path):
    """Stages may claim scene_usda / robot_usda / render_layer_usda directly."""
    m = Manifest(base_path=tmp_path)
    m.merge_slice({"render_layer_usda": "scenes/foo/render_layer.usda"}, "isaac_scene")
    assert m.render_layer_usda == Path("scenes/foo/render_layer.usda")


def test_merge_slice_accepts_typed_camera_object():
    """isaac_scene_stage forwards CameraManifest instances; merge accepts both."""
    m = Manifest()
    cam = CameraManifest(path="a/b", topic="/rgb", render_product_path="/RenderOVRTX/Cam_0")
    m.merge_slice({"cameras": [cam]}, "isaac_scene")
    assert m.cameras[0] is cam


def test_merge_slice_accepts_path_strings_for_usda_fields():
    """isaac_scene forwards subprocess-loaded absolute Paths; merge stores them typed."""
    from pathlib import Path as _P

    m = Manifest()
    m.merge_slice(
        {
            "render_layer_usda": _P("/abs/path/render_layer.usda"),
            "robot_prefix": "G2",
        },
        "isaac_scene",
    )
    assert m.render_layer_usda == _P("/abs/path/render_layer.usda")
    assert m.robot_prefix == "G2"


def test_lidar_slice_accepts_typed_objects():
    m = Manifest()
    lid = LidarManifest(render_product_path="/RenderOVRTX/Lidar_0", topic="/lidar/front")
    m.merge_slice({"lidars": [lid]}, "isaac_scene")
    assert m.lidars[0] is lid


def test_lidar_slice_accepts_raw_dict():
    m = Manifest()
    m.merge_slice(
        {"lidars": [{"render_product_path": "/x/l", "topic": "/lidar/back"}]},
        "isaac_scene",
    )
    assert m.lidars[0].render_product_path == "/x/l"
    assert m.lidars[0].topic == "/lidar/back"


def test_lidar_manifest_roundtrip():
    lid = LidarManifest.from_dict(
        {
            "render_product_path": "/RenderOVRTX/Lidar_0",
            "topic": "/lidar/livox_front",
            "frame_id": "base_link",
            "parent_body": "base_link",
            "base_link_T_sensor": {
                "xyz": [0.1, 0.0, 0.2],
                "wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        }
    )
    assert lid.parent_body == "base_link"
    assert lid.base_link_T_sensor.xyz == (0.1, 0.0, 0.2)
    round = LidarManifest.from_dict(lid.to_dict())
    assert round.base_link_T_sensor.xyz == lid.base_link_T_sensor.xyz
