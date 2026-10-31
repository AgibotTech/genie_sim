# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Wire-frame adaptation leaves planner/runtime coordinates unchanged."""

from copy import deepcopy

import pytest

from agent2act.adapters.simulators.corobot_ik_frame import CoRobotIKFrameCodec
from agent2act.bootstrap import build_session, default_registry
from agent2act.domain.models import ActionChunk, ContractError


@pytest.fixture
def frame_options(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="test">
      <joint name="left_joint" type="prismatic">
        <parent link="arm_base_link"/><child link="left_tip"/>
        <origin xyz="0 1 0"/><axis xyz="1 0 0"/>
      </joint>
      <joint name="right_joint" type="prismatic">
        <parent link="arm_base_link"/><child link="right_tip"/>
        <origin xyz="0 -1 0"/><axis xyz="1 0 0"/>
      </joint></robot>""")
    return dict(path=str(path), left_link="left_tip", right_link="right_tip")


def frame_request(request_payload, shift=0):
    request = deepcopy(request_payload)
    # base_from_arm: +90deg about Z and translation [2+shift, 3, 4].
    q = [0, 0, 2**-0.5, 2**-0.5]
    request["params"]["states"]["end_pose"]["base_link"] = {
        "left_arm": dict(position=[1 + shift, 3, 4], orientation=q),
        "right_arm": dict(position=[3 + shift, 3, 4], orientation=q),
    }
    return request


def test_pose_rotation_translation_and_grippers_at_wire_only(spec, frame_options, request_payload):
    codec = CoRobotIKFrameCodec(spec, frame_options)
    try:
        for shift in (0, 0.2):
            request = frame_request(request_payload, shift)
            original_request = deepcopy(request)
            observation = codec.decode(request)
            left = observation.eef("left", "base_link")
            right = observation.eef("right", "base_link")
            # A base +Y movement is arm-base +X, with orientation returned to identity.
            moved = [right[0], right[1] + 0.1, right[2], *right[3:]]
            chunk = ActionChunk(
                "EEF_ABS", "base_link", spec.dt_s, [left, left], [right, moved], [[0.5, 0.5], [0.5, 0.1]]
            )
            original_chunk = deepcopy(chunk)
            wire = codec.encode(chunk, observation, history={}, need_depth=False)["result"]
            assert wire["left_arm"]["base_link"] == "arm_base_link"
            for pose in wire["left_arm"]["values"]:
                assert pose == pytest.approx([0, 1, 0, 0, 0, 0, 1])
            assert wire["right_arm"]["values"][0] == pytest.approx([0, -1, 0, 0, 0, 0, 1])
            assert wire["right_arm"]["values"][1] == pytest.approx([0.1, -1, 0, 0, 0, 0, 1])
            assert wire["right_effector"] == [[0.5], [0.1]]
            assert chunk == original_chunk
            assert request == original_request
            codec.reset()
    finally:
        codec.close()
    codec.close()
    with pytest.raises(RuntimeError, match="closed"):
        codec.decode(frame_request(request_payload))


@pytest.mark.parametrize("bad", ["missing_pose", "translation", "rotation", "joint_order"])
def test_inconsistent_observation_rejected_before_planning(spec, frame_options, request_payload, bad):
    codec = CoRobotIKFrameCodec(spec, frame_options)
    request = frame_request(request_payload)
    states = request["params"]["states"]
    if bad == "missing_pose":
        states["end_pose"] = {}
    elif bad == "translation":
        states["end_pose"]["base_link"]["right_arm"]["position"][0] += 0.02
    elif bad == "rotation":
        states["end_pose"]["base_link"]["right_arm"]["orientation"] = [0, 0, 0, 1]
    else:
        states["arm_joint_states"] = [0.1, -0.1]
    try:
        with pytest.raises(ContractError):
            codec.decode(request)
    finally:
        codec.close()


def test_session_holds_in_runtime_frame_and_encodes_in_ik_frame(config, frame_options, request_payload):
    config["simulator_codec"] = dict(type="corobot_ik_frame", **frame_options)
    config["planner"]["decisions"][0]["motion"]["waypoints"] = [{"duration_s": 0.1}]
    with build_session(config) as session:
        wire = session.handle(frame_request(request_payload))["result"]
        assert wire["right_arm"]["values"][-1] == pytest.approx([0, -1, 0, 0, 0, 0, 1])
        assert session.runtime._last_issued.frame == "base_link"
        assert session.runtime._last_issued.right_arm[-1][:3] == [3, 3, 4]
        assert "wire_frame_conversion" not in session.runtime._last_issued.metadata
        request = frame_request(request_payload)
        request["params"]["states"]["end_pose"]["base_link"]["right_arm"]["position"][0] += 0.1
        with pytest.raises(ContractError, match="frame mismatch"):
            session.handle(request)
        assert session.runtime.sequence == 1


def test_codec_cleanup_on_later_construction_failure(config, spec, frame_options):
    codec = CoRobotIKFrameCodec(spec, frame_options)
    registry = default_registry()
    registry.codecs.register("test_codec", lambda spec, options: codec)
    config["simulator_codec"] = {"type": "test_codec"}
    config["motion_backend"] = {"type": "missing"}
    with pytest.raises(ValueError):
        build_session(config, registry=registry)
    assert codec.fk.closed


def test_invalid_options_and_unsupported_commands(spec, frame_options, request_payload):
    with pytest.raises(ContractError):
        CoRobotIKFrameCodec(spec, {**frame_options, "offset": [0, 0, 0]})
    codec = CoRobotIKFrameCodec(spec, frame_options)
    try:
        obs = codec.decode(frame_request(request_payload))
        chunk = ActionChunk(
            "EEF_ABS",
            "arm_base_link",
            spec.dt_s,
            [[0, 1, 0, 0, 0, 0, 1]],
            [[0, -1, 0, 0, 0, 0, 1]],
            [[0.5, 0.5]],
        )
        with pytest.raises(ContractError, match="internal EEF"):
            codec.encode(chunk, obs, history={}, need_depth=False)
        chunk.frame = "base_link"
        chunk.body = [[0.1]]
        with pytest.raises(ContractError, match="moving body"):
            codec.encode(chunk, obs, history={}, need_depth=False)
    finally:
        codec.close()


@pytest.mark.parametrize("example", ["g2_llm_recipe.json"])
def test_g2_api_examples_preserve_hold_through_selected_codec(
    example, config, frame_options, request_payload
):
    import json
    from pathlib import Path

    deployment = json.loads((Path(__file__).resolve().parents[2] / "examples" / example).read_text())
    # Keep the deployment's codec selection, replace the machine-specific robot with a portable fixture.
    config["simulator_codec"] = {**deployment["simulator_codec"], **frame_options}
    config["planner"]["decisions"][0]["motion"]["waypoints"] = [{"duration_s": 0.1}]
    with build_session(config) as session:
        wire = session.handle(frame_request(request_payload))["result"]
        assert wire["left_arm"]["base_link"] == "arm_base_link"
        assert wire["left_arm"]["values"][-1] == pytest.approx([0, 1, 0, 0, 0, 0, 1])
        assert wire["right_arm"]["values"][-1] == pytest.approx([0, -1, 0, 0, 0, 0, 1])
        assert session.runtime._last_issued.frame == "base_link"
