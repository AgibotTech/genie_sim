# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Analytic geometry checks; no simulator/model dependency."""

import math
from dataclasses import replace

import pytest

from agent2act.adapters.kinematics.urdf import UrdfKinematics
from agent2act.domain.models import ActionChunk, ContractError


def planar_model(path, spec):
    path.write_text(f'''<robot name="test">
    <joint name="{spec.left_joints[0]}" type="revolute"><parent link="base_link"/><child link="left"/>
    <axis xyz="0 0 1"/></joint>
    <joint name="tip" type="fixed"><parent link="left"/><child link="tip"/><origin xyz="1 0 0"/></joint>
    <joint name="{spec.right_joints[0]}" type="prismatic"><parent link="base_link"/><child link="right"/>
    <axis xyz="0 1 0"/><origin xyz="0 0 1"/></joint></robot>''')
    return UrdfKinematics(spec, dict(path=str(path), frame="base_link", left_link="tip", right_link="right"))


def test_analytic_fk_and_live_consistency(tmp_path, spec, observation):
    fk = planar_model(tmp_path / "robot.urdf", spec)
    poses = fk.poses([math.pi / 2, 0.2], [])
    assert poses["left"][:3] == pytest.approx([0, 1, 0])
    assert poses["right"][:3] == pytest.approx([0, 0.2, 1])
    obs = replace(
        observation,
        joints=[math.pi / 2, 0.2],
        end_pose={
            "base_link": {side + "_arm": dict(position=p[:3], orientation=p[3:]) for side, p in poses.items()}
        },
    )
    chunk = ActionChunk("JOINT_ABS", "base_link", 0.05, [[math.pi / 2]], [[0.2]], [[0, 0]])
    preview = fk.preview(obs, chunk, 1)
    assert preview["current_fk_errors"]["left"]["translation_m"] < 1e-9
    obs.joints[0] = 0
    with pytest.raises(ContractError, match="FK observation mismatch"):
        fk.preview(obs, chunk, 1)
    fk.close()
    with pytest.raises(RuntimeError, match="closed"):
        fk.poses([0, 0], [])


def test_unknown_chain_joint_rejected(tmp_path, spec):
    with pytest.raises(ContractError, match="lacks measured state"):
        UrdfKinematics(
            spec,
            dict(path=str(write_unknown(tmp_path)), frame="base_link", left_link="tip", right_link="tip"),
        )


def write_unknown(tmp_path):
    p = tmp_path / "unknown.urdf"
    p.write_text(
        '<robot><joint name="unknown" type="revolute"><parent link="base_link"/><child link="tip"/></joint></robot>'
    )
    return p


def test_subtree_preserves_parent_geometry(tmp_path, spec):
    base = tmp_path / "base.urdf"
    base.write_text("""<robot><joint name="mount" type="fixed">
    <parent link="base_link"/><child link="mount"/><origin xyz="0 0 2"/>
    </joint></robot>""")
    arms = tmp_path / "arms.urdf"
    planar_model(arms, spec)
    arms.write_text(arms.read_text().replace('link="base_link"', 'link="mount"'))
    fk = UrdfKinematics(
        spec,
        dict(
            path=str(base),
            subtree_path=str(arms),
            subtree_root="mount",
            frame="base_link",
            left_link="tip",
            right_link="right",
        ),
    )
    poses = fk.poses([math.pi / 2, 0.2], [])
    assert poses["left"][:3] == pytest.approx([0, 1, 2])
    assert poses["right"][:3] == pytest.approx([0, 0.2, 3])


def test_fk_preview_rejects_wrong_frame(tmp_path, spec, observation):
    fk = planar_model(tmp_path / "robot.urdf", spec)
    chunk = ActionChunk("JOINT_ABS", "arm_base_link", 0.05, [[0]], [[0]], [[0, 0]])
    with pytest.raises(ContractError, match="configured frame"):
        fk.preview(observation, chunk, 1)


def test_full_fk_envelope_catches_unsampled_out_and_back(tmp_path, spec, observation, monkeypatch):
    fk = planar_model(tmp_path / "robot.urdf", spec)
    poses = fk.poses([0, 0], [])
    obs = replace(
        observation,
        end_pose={
            "base_link": {side + "_arm": dict(position=p[:3], orientation=p[3:]) for side, p in poses.items()}
        },
    )
    left = [[0.0] for _ in range(30)]
    left[17] = [0.1]
    chunk = ActionChunk("JOINT_ABS", "base_link", 0.05, left, [[0]] * 30, [[0.5, 0.5]] * 30)
    artifacts = []
    monkeypatch.setattr(
        "agent2act.adapters.kinematics.urdf.record_artifact", lambda name, value: artifacts.append(name)
    )
    preview = fk.preview(obs, chunk, 10)
    chunk.metadata["command_bounds"] = {"limits": {}}
    fk.preview(obs, chunk, chunk.steps)
    assert artifacts == ["policy-proposal", "bounded-command-preview"]
    assert 18 not in [row["step"] for row in preview["trajectory"]]
    assert preview["command_envelope"]["left"]["translation_m"] == pytest.approx(2 * math.sin(0.05))
    assert preview["command_envelope"]["left"]["rotation_rad"] == pytest.approx(0.1)
    assert preview["executable_command_envelope"]["steps"] == 10
    assert preview["executable_command_envelope"]["left"]["translation_m"] == 0
    fk.close()
