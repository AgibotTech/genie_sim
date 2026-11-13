# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Finite arm commands saturate at limits instead of invalidating the policy."""

import numpy as np
import pytest

from geniesim_benchmark.backends.mujoco.rubik.policy import PolicyError, parse_actions


def reply_for(targets, encoding):
    if encoding == "packed":
        return {"result": {"kind": "JOINT_ABS", "joint_targets": targets.tolist()}}
    return {
        "result": {
            "left_arm": {"kind": "JOINT_ABS", "values": targets[:, :7].tolist()},
            "right_arm": {"kind": "JOINT_ABS", "values": targets[:, 7:14].tolist()},
            "left_effector": targets[:, 14:34].tolist(),
            "right_effector": targets[:, 34:].tolist(),
        }
    }


@pytest.mark.parametrize("encoding", ["packed", "corobot"])
def test_arm_targets_clip_both_sides_of_every_limit(encoding):
    bounds = np.tile([-1.0, 2.0], (54, 1))
    targets = np.zeros((3, 54))
    targets[0, :14] = -100.0
    targets[1, :14] = 100.0
    targets[2, :14] = np.linspace(-0.5, 1.5, 14)
    targets[:, 14:] = 0.25
    diagnostics = {}
    actions, done = parse_actions(reply_for(targets, encoding), {"bounds": bounds}, 256, projection_log=diagnostics)
    np.testing.assert_array_equal(actions[:, :14], np.clip(targets[:, :14], -1.0, 2.0))
    np.testing.assert_array_equal(actions[:, 14:], targets[:, 14:])
    assert not done
    assert diagnostics["arm_changed_values"] == 28
    assert diagnostics["arm_maximum_projection_radians"] == 99.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_arm_targets_still_reject(value):
    targets = np.zeros((1, 54))
    targets[0, 0] = value
    with pytest.raises(PolicyError, match="finite"):
        parse_actions(reply_for(targets, "corobot"), {"bounds": np.tile([-1.0, 1.0], (54, 1))}, 256)


def test_arm_saturation_does_not_change_hand_validation():
    targets = np.zeros((1, 54))
    targets[0, 0] = 100.0
    targets[0, 14] = 2.0
    with pytest.raises(PolicyError, match="Hand targets"):
        parse_actions(reply_for(targets, "corobot"), {"bounds": np.tile([-1.0, 1.0], (54, 1))}, 256)
