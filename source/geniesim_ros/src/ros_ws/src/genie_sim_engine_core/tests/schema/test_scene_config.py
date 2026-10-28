# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Tests for :mod:`genie_sim_engine_core.schema.scene`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from genie_sim_engine_core.schema import SceneConfig, SchemaError
from genie_sim_engine_core.schema.scene import (
    CameraEntry,
    CameraIntrinsic,
    CameraSensor,
    CameraTopic,
    ContactPublishSection,
    Pose,
    RobotSource,
)


def _minimal_scene() -> str:
    return textwrap.dedent("""
        robot:
          robot_name: G2_test
          robot_prefix: G2
          robot_source:
            robot_model: G2
            body: t2
            arm: crsB
            gripper: scc
            package: genie_robot_description
            prebaked:
              usda: robot/G2_test/robot.usda
              mjcf_robot: mujoco/mjcf/G2_test.xml
          init_base_pose2d:
            x: 1.0
            y: -0.5
            theta: 3.14
          init_joint_pos:
            head_joint: 0.1
        scene: scenes/foo/foo.usda
        cameras:
          - prim_path: head_link3/Head_Front_Camera
            frame_id: head_link3
            topic:
              rgb: /genie_sim/head_front_camera_rgb
              depth: /genie_sim/head_front_camera_depth
            sensor:
              width: 1280
              height: 800
              min_range: 0.01
              max_range: 10000.0
            intrinsic:
              fx: 610.65472
              fy: 610.70917
              cx: 644.53473
              cy: 401.22052
            dds_topic: /camera/head_color
            dds_depth_topic: /camera/head_depth
        lidars: []
        """).lstrip()


def test_from_dict_roundtrip():
    """Read → to_dict → from_dict → to_dict is idempotent."""
    import yaml

    raw = yaml.safe_load(_minimal_scene())
    scene = SceneConfig.from_dict(raw)
    round_a = scene.to_dict()
    round_b = SceneConfig.from_dict(round_a).to_dict()
    assert round_a == round_b


def test_read_from_disk(tmp_path: Path):
    p = tmp_path / "scene.yaml"
    p.write_text(_minimal_scene())
    scene = SceneConfig.read(p)
    assert scene.source_path == p
    assert scene.robot.robot_name == "G2_test"
    assert scene.robot.robot_prefix == "G2"
    assert scene.robot.robot_source.kind == "prebaked"
    assert scene.robot.robot_source.prebaked is not None
    assert scene.robot.robot_source.prebaked.usda == "robot/G2_test/robot.usda"
    # ``init_base_pose2d`` is no longer a typed field — it had zero consumers
    # (every backend reads ``init_base_pose``), so it was dropped rather than
    # kept as dead API.  The removal is lossless: the key lands in ``extra``
    # and round-trips verbatim, so an old yaml neither raises nor loses data,
    # and nothing pretends the value is honoured.
    assert not hasattr(scene.robot, "init_base_pose2d")
    assert scene.robot.extra["init_base_pose2d"] == {"x": 1.0, "y": -0.5, "theta": 3.14}
    assert scene.robot.to_dict()["init_base_pose2d"] == {"x": 1.0, "y": -0.5, "theta": 3.14}
    assert scene.robot.init_joint_pos == {"head_joint": 0.1}
    assert scene.scene_usda == "scenes/foo/foo.usda"
    assert len(scene.cameras) == 1
    cam = scene.cameras[0]
    assert cam.prim_path == "head_link3/Head_Front_Camera"
    assert cam.topic.rgb == "/genie_sim/head_front_camera_rgb"
    assert cam.topic.depth == "/genie_sim/head_front_camera_depth"
    assert cam.sensor.width == 1280
    assert cam.intrinsic.fx == pytest.approx(610.65472)
    assert cam.dds_topic == "/camera/head_color"
    assert cam.dds_depth_topic == "/camera/head_depth"


def test_camera_missing_prim_path_raises():
    """CameraEntry with neither prim_path nor path is a hard error."""
    with pytest.raises(SchemaError):
        CameraEntry.from_dict({"topic": {"rgb": "/foo"}})


def test_camera_topic_scalar_or_dict():
    """topic: '/foo' and topic: {rgb: /foo} both work — legacy scenes."""
    e1 = CameraEntry.from_dict({"prim_path": "foo/bar", "topic": "/foo"})
    e2 = CameraEntry.from_dict({"prim_path": "foo/bar", "topic": {"rgb": "/foo"}})
    assert e1.topic.rgb == e2.topic.rgb == "/foo"


def test_robot_source_kind_dispatch():
    """RobotSource.kind is picked by presence of urdf > prebaked > usd."""
    urdf = RobotSource.from_dict({"urdf": {"xacro_entry": "foo.xacro"}})
    assert urdf.kind == "urdf"
    prebaked = RobotSource.from_dict({"prebaked": {"usda": "x.usda"}})
    assert prebaked.kind == "prebaked"
    usd = RobotSource.from_dict({"usd": "x.usda"})
    assert usd.kind == "usd"
    both = RobotSource.from_dict({"urdf": {}, "prebaked": {"usda": "x"}})
    assert both.kind == "urdf"  # urdf wins
    empty = RobotSource.from_dict({})
    assert empty.kind == ""


def test_convert_joints_to_fixed_accepts_str_or_list():
    """convert_joints_to_fixed: base and [base, head] both work — legacy scenes."""
    a = RobotSource.from_dict({"convert_joints_to_fixed": "base"})
    b = RobotSource.from_dict({"convert_joints_to_fixed": ["base", "head"]})
    assert a.convert_joints_to_fixed == ["base"]
    assert b.convert_joints_to_fixed == ["base", "head"]


def test_pin_base_to_world_string_bool_coercion():
    """YAML strings 'false' / 'no' should coerce to False."""
    for false_str in ("false", "False", "no", "0", "off", ""):
        rs = RobotSource.from_dict({"pin_base_to_world": false_str})
        assert rs.pin_base_to_world is False
    for true_str in ("true", "yes", "on", "1", "sure"):  # anything else -> True
        rs = RobotSource.from_dict({"pin_base_to_world": true_str})
        assert rs.pin_base_to_world is True


def test_resolve_robot_params_scene_wins():
    """SceneConfig.resolve_robot_params: scene YAML wins where non-empty."""
    scene = SceneConfig.from_dict(
        {
            "robot": {"robot_source": {"robot_model": "G2", "arm": "crsB"}},
        }
    )
    resolved = scene.resolve_robot_params(robot_model="genie", body="t2", arm="fallback", gripper="scc")
    # Scene provided robot_model and arm; body and gripper fall through.
    # lgripper/rgripper mirror the resolved gripper (no split in scene).
    assert resolved == {
        "robot_model": "G2",
        "body": "t2",
        "arm": "crsB",
        "gripper": "scc",
        "lgripper": "scc",
        "rgripper": "scc",
    }


def test_resolve_robot_params_reads_nested_urdf():
    """Nested urdf.gripper reachable via the flat resolve, matching legacy _pick_robot_source_str."""
    scene = SceneConfig.from_dict(
        {
            "robot": {
                "robot_source": {
                    "urdf": {"gripper": "scc"},
                },
            },
        }
    )
    resolved = scene.resolve_robot_params(gripper="cli_default")
    # Legacy behaviour: nested urdf.gripper wins over CLI default.
    assert resolved["gripper"] == "scc"


def test_resolve_robot_params_flat_wins_over_nested():
    """When both flat and nested exist, flat wins."""
    scene = SceneConfig.from_dict(
        {
            "robot": {
                "robot_source": {
                    "gripper": "flat",
                    "urdf": {"gripper": "nested"},
                },
            },
        }
    )
    resolved = scene.resolve_robot_params(gripper="cli")
    assert resolved["gripper"] == "flat"


def test_resolve_robot_params_simple_gripper_mirrors_to_split():
    """Simple config: robot_source.gripper: scc → lgripper=rgripper=scc.

    Preserves the ``lgripper = gripper; rgripper = gripper`` fallback
    the launcher has always applied. Two-gripper scenes override via
    ``urdf.lgripper`` / ``urdf.rgripper``.
    """
    scene = SceneConfig.from_dict(
        {
            "robot": {"robot_source": {"gripper": "scc"}},
        }
    )
    r = scene.resolve_robot_params(gripper="cli_default")
    assert r["gripper"] == "scc"
    assert r["lgripper"] == "scc"
    assert r["rgripper"] == "scc"


def test_resolve_robot_params_split_gripper_wins_over_simple():
    """Scenes that author both hands separately keep the split."""
    scene = SceneConfig.from_dict(
        {
            "robot": {
                "robot_source": {
                    "gripper": "fallback_scc",
                    "urdf": {"lgripper": "scc_l", "rgripper": "scc_r"},
                },
            },
        }
    )
    r = scene.resolve_robot_params(gripper="cli_default")
    # Simple gripper still wins for the ``gripper`` slot (BC), but the
    # split urdf.lgripper / urdf.rgripper override on their own axes.
    assert r["gripper"] == "fallback_scc"
    assert r["lgripper"] == "scc_l"
    assert r["rgripper"] == "scc_r"


def test_resolve_robot_params_split_gripper_only():
    """No simple ``gripper`` set — the split still wins over CLI."""
    scene = SceneConfig.from_dict(
        {
            "robot": {
                "robot_source": {
                    "urdf": {"lgripper": "scc_l", "rgripper": "scc_r"},
                },
            },
        }
    )
    r = scene.resolve_robot_params(gripper="cli_default", lgripper="cli_L", rgripper="cli_R")
    # Simple ``gripper`` falls back to CLI default because no scene
    # value is present at any position; each half of the split, though,
    # is fully driven by the scene urdf block.
    assert r["gripper"] == "cli_default"
    assert r["lgripper"] == "scc_l"
    assert r["rgripper"] == "scc_r"


def test_resolve_robot_params_cli_lgripper_fallback():
    """Without scene split OR simple gripper, CLI lgripper/rgripper fill in."""
    scene = SceneConfig.from_dict({"robot": {"robot_source": {}}})
    r = scene.resolve_robot_params(lgripper="scc_l", rgripper="scc_r")
    assert r["gripper"] == ""
    assert r["lgripper"] == "scc_l"
    assert r["rgripper"] == "scc_r"


def test_extra_top_level_keys_roundtripped():
    """Unknown top-level scene keys survive round-trip."""
    scene = SceneConfig.from_dict({"custom_launcher_key": {"foo": 1}})
    assert scene.extra == {"custom_launcher_key": {"foo": 1}}
    assert scene.to_dict().get("custom_launcher_key") == {"foo": 1}


def test_engine_specific_blocks_opaque():
    """newton / imu / characters are stored verbatim for the engine to parse."""
    scene = SceneConfig.from_dict(
        {
            "newton": {"solver": {"prefer": "mjwarp"}, "entries": []},
            "imu": {"topic": "/imu"},
            "characters": [{"name": "alice"}],
        }
    )
    assert scene.newton["solver"]["prefer"] == "mjwarp"
    assert scene.imu["topic"] == "/imu"
    assert scene.characters == [{"name": "alice"}]


def test_contact_publish_defaults_are_manipulation_tuned():
    """Missing contact_publish block yields the manipulation defaults."""
    scene = SceneConfig.from_dict({})
    assert scene.contact_publish.scope == "robot_scene"
    assert scene.contact_publish.include_self_contacts is False
    # Default block does NOT appear in to_dict — keeps emitted YAML clean.
    assert "contact_publish" not in scene.to_dict()


def test_contact_publish_parses_overrides():
    scene = SceneConfig.from_dict({"contact_publish": {"scope": "all", "include_self_contacts": True}})
    assert scene.contact_publish.scope == "all"
    assert scene.contact_publish.include_self_contacts is True
    # Non-default block round-trips through to_dict.
    assert scene.to_dict()["contact_publish"] == {
        "scope": "all",
        "include_self_contacts": True,
    }


def test_contact_publish_rejects_unknown_scope():
    with pytest.raises(SchemaError):
        ContactPublishSection.from_dict({"scope": "everything"})


def test_pose_identity_default():
    """Missing extrinsic falls back to identity."""
    pose = Pose.from_dict(None)
    assert pose.is_identity()
    assert pose.xyz == (0.0, 0.0, 0.0)
    assert pose.wxyz == (1.0, 0.0, 0.0, 0.0)


def test_pose_malformed_falls_back():
    """A bad extrinsic type shouldn't crash the whole parse — matches Isaac path."""
    pose = Pose.from_dict({"xyz": "not_a_list", "wxyz": None})
    assert pose.is_identity()


def test_camera_intrinsic_defaults_when_missing_block():
    """No intrinsic block yields the render-node defaults, not None."""
    cam = CameraEntry.from_dict({"prim_path": "foo/bar", "topic": "/foo"})
    assert cam.intrinsic.fx == 610.0
    assert cam.intrinsic.k4 == 0.0
    assert isinstance(cam.intrinsic, CameraIntrinsic)


def test_write_roundtrip(tmp_path: Path):
    """SceneConfig.write → read is byte-stable enough for downstream diffs."""
    import yaml

    scene = SceneConfig.from_dict(yaml.safe_load(_minimal_scene()))
    p = tmp_path / "out.yaml"
    scene.write(p)
    back = SceneConfig.read(p)
    assert back.to_dict() == scene.to_dict()
