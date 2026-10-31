# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Quantitative command envelopes do not infer contact or task outcome."""

from copy import deepcopy

import pytest

from agent2act.domain.command_bounds import validate_command_bounds, wrist_envelope
from agent2act.domain.models import ActionChunk, ContractError
from agent2act.domain.review import CommandBounds


def bounds_dict():
    return {
        "frame": "base_link",
        "left": {"translation_m": 0.005, "rotation_rad": 0.01, "gripper_change": 0.1},
        "right": {"translation_m": 0.005, "rotation_rad": 0.01, "gripper_change": 0},
    }


@pytest.mark.parametrize("bad", [-1, True, float("nan"), float("inf"), "0.1", None])
@pytest.mark.parametrize("field", ["translation_m", "rotation_rad", "gripper_change"])
def test_bounds_reject_invalid_limits(bad, field):
    data = bounds_dict()
    data["left"][field] = bad
    with pytest.raises(ContractError, match="finite nonnegative"):
        CommandBounds.from_dict(data)


@pytest.mark.parametrize("change", ["missing_arm", "unknown_field", "unknown_arm_field", "frame"])
def test_bounds_require_complete_explicit_contract(change):
    data = bounds_dict()
    if change == "missing_arm":
        del data["right"]
    elif change == "unknown_field":
        data["contact"] = True
    elif change == "unknown_arm_field":
        data["left"]["collision_free"] = True
    else:
        data["frame"] = "world"
    with pytest.raises(ContractError):
        CommandBounds.from_dict(data)


def test_every_command_checked_even_when_endpoint_returns_to_start(spec, observation):
    left = [observation.eef("left", "base_link") for _ in range(30)]
    right = [observation.eef("right", "base_link") for _ in range(30)]
    left[17][0] += 0.02
    chunk = ActionChunk("EEF_ABS", "base_link", 0.05, left, right, [[0.5, 0.5]] * 30)
    envelope = wrist_envelope(observation, chunk.frame, left, right)
    with pytest.raises(ContractError, match="left.translation_m"):
        validate_command_bounds(CommandBounds.from_dict(bounds_dict()), observation, chunk, spec, envelope)


@pytest.mark.parametrize("change", ["missing", "count", "nan", "negative", "bool", "missing_arm"])
def test_bounds_reject_incomplete_or_malformed_geometry(spec, observation, change):
    left, right = [observation.eef("left", "base_link")], [observation.eef("right", "base_link")]
    chunk = ActionChunk("EEF_ABS", "base_link", 0.05, left, right, [[0.5, 0.5]])
    envelope = deepcopy(wrist_envelope(observation, chunk.frame, left, right))
    if change == "missing":
        envelope = {}
    elif change == "count":
        envelope["steps"] = 2
    elif change == "missing_arm":
        del envelope["right"]
    else:
        envelope["left"]["translation_m"] = {"nan": float("nan"), "negative": -1, "bool": False}[change]
    with pytest.raises(ContractError):
        validate_command_bounds(CommandBounds.from_dict(bounds_dict()), observation, chunk, spec, envelope)
