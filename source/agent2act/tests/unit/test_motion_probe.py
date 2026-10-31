# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy

import pytest

from agent2act.adapters.planners.motion_probe import MotionProbePlanner
from agent2act.bootstrap import build_session
from agent2act.domain.errors import ConfigurationError, ContractError


def probe_options():
    return {
        "frame": "base_link",
        "trials": [{"name": "x", "arm": "left", "translation_mm": [5, 0, 0], "duration_s": 0.1}],
        "settle_s": 0.2,
        "return_duration_s": 0.3,
    }


def test_fixed_probe_samples_return_and_completion(config, observation, monkeypatch):
    samples = []
    monkeypatch.setattr(MotionProbePlanner, "_emit", staticmethod(samples.append))
    config["planner"] = {"type": "motion_probe", **probe_options()}
    config["execution"].update(motion_frame="base_link", execute_steps=120, max_replans=0)
    original = copy.deepcopy(observation)
    with build_session(config) as session:
        for _ in range(7):
            before = copy.deepcopy(observation)
            chunk = session.runtime.step(observation)
            assert observation == before
            for arm in ("left", "right"):
                target = getattr(chunk, f"{arm}_arm")[-1]
                observation.end_pose["base_link"][f"{arm}_arm"] = {
                    "position": target[:3],
                    "orientation": target[3:],
                }
            observation.grippers = chunk.grippers[-1]
    rows = [s for s in samples if s["event"] == "motion_probe_sample"]
    assert [s["phase"] for s in rows] == [
        "setup",
        "baseline_hold",
        "pulse",
        "target_hold",
        "return",
        "baseline_hold",
    ]
    assert rows[2]["target"]["left_pose"][0] == pytest.approx(original.eef("left", "base_link")[0] + 0.005)
    assert rows[2]["target"] == rows[3]["target"]
    assert rows[-1]["observed"]["left_pose"] == original.eef("left", "base_link")
    assert samples[-1]["event"] == "motion_probe_finished"
    assert all(s["feedback"]["discarded_steps"] == 0 for s in rows)


def test_probe_retries_do_not_advance_and_reset_clears_reference(config, observation):
    config["planner"] = {"type": "motion_probe", **probe_options()}
    config["execution"].update(motion_frame="base_link", execute_steps=120)
    with build_session(config) as session:
        runtime = session.runtime
        context = {
            "sequence": 1,
            "execution": runtime.router.describe(),
            "robot": runtime.spec.describe(),
            "gripper_commands": observation.command_grippers(runtime.spec),
            "observed_grippers_in_command_units": observation.command_grippers(runtime.spec),
        }
        p = runtime.planner
        first = p.decide(observation, context)
        assert p.decide(observation, context) == first
        assert p._index == 1
        p.reset()
        assert p._reference is None
        p.close()
        with pytest.raises(RuntimeError, match="closed"):
            p.decide(observation, context)


@pytest.mark.parametrize(
    "changes",
    [
        {"translation_mm": [1, 1, 0]},
        {"translation_mm": [51, 0, 0]},
        {"duration_s": 0},
        {"duration_s": True},
        {"gripper_delta": 0.1},
        {"unknown": 1},
        {"arm": "third"},
    ],
)
def test_invalid_probe_trial(changes):
    options = probe_options()
    options["trials"][0].update(changes)
    with pytest.raises(ContractError):
        MotionProbePlanner(**options)


def test_probe_rejects_truncated_feedback(config, observation):
    config["planner"] = {"type": "motion_probe", **probe_options()}
    config["execution"].update(motion_frame="base_link", execute_steps=1, max_replans=0)
    with build_session(config) as session:
        session.runtime.step(observation)
        with pytest.raises(ContractError, match="untruncated"):
            session.runtime.step(observation)


def test_probe_records_partial_measured_closure_without_relaxing_target(config, observation, monkeypatch):
    samples = []
    monkeypatch.setattr(MotionProbePlanner, "_emit", staticmethod(samples.append))
    config["planner"] = {
        "type": "motion_probe",
        **probe_options(),
        "trials": [{"name": "close", "arm": "left", "gripper_delta": 0.1, "duration_s": 0.1}],
    }
    config["robot"].update(gripper_observation_scale=[-1, 2], gripper_observation_offset=[0.8, 0.1])
    config["execution"].update(motion_frame="base_link", execute_steps=120, max_replans=0)
    observation.grippers = [0.2, 0.2]
    with build_session(config) as session:
        for sequence in range(1, 6):
            chunk = session.runtime.step(observation)
            if sequence == 3:
                assert chunk.grippers[-1] == pytest.approx([0.7, 0.5])
                observation.grippers = [0.16, 0.2]  # Only partial measured closure.
            if sequence == 4:
                assert chunk.grippers[-1] == pytest.approx([0.7, 0.5])
    rows = [s for s in samples if s["event"] == "motion_probe_sample"]
    pulse, held = rows[2:4]
    assert pulse["before"]["grippers"] == pytest.approx([0.6, 0.5])
    assert pulse["observed"]["grippers"] == pytest.approx([0.64, 0.5])
    assert held["before"]["grippers"] == pytest.approx([0.64, 0.5])
    assert held["observed"]["grippers"] == pytest.approx([0.64, 0.5])
    assert pulse["target"]["grippers"] == held["target"]["grippers"] == pytest.approx([0.7, 0.5])


def test_probe_cleanup_on_compiler_construction_failure(config, monkeypatch):
    import agent2act.bootstrap as bootstrap

    closed = []
    monkeypatch.setattr(MotionProbePlanner, "close", lambda self: closed.append(True))

    def fail(*args):
        raise ContractError("construction failure")

    monkeypatch.setattr(bootstrap, "MotionCompiler", fail)
    config["planner"] = {"type": "motion_probe", **probe_options()}
    with pytest.raises(ConfigurationError):
        build_session(config)
    assert closed == [True]


def test_probe_report_separates_episodes_and_preserves_missing_hold(tmp_path):
    import json
    import runpy
    from pathlib import Path

    summarize = runpy.run_path(str(Path(__file__).parents[2] / "scripts/analyze_motion_probe.py"))[
        "summarize"
    ]
    trial = {"name": "x", "arm": "left", "translation_mm": [5, 0, 0]}
    pulse = {
        "message": "motion_probe_sample",
        "session_id": "one",
        "trial": "x",
        "phase": "pulse",
        "before": {"left_pose": [0, 0, 0]},
        "target": {"left_pose": [0.005, 0, 0]},
        "observed": {"left_pose": [0.001, 0, 0]},
        "duration_s": 0.1,
        "dt_s": 0.05,
        "feedback": {"issued_steps": 2},
    }
    start = {"message": "motion_probe_started", "session_id": "one", "trials": [trial]}
    held = {**pulse, "phase": "target_hold", "observed": {"left_pose": [0.004, 0, 0]}}
    log = tmp_path / "probe.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in [start, pulse, held, start, pulse]))
    result = summarize(log)
    assert len(result) == 2
    assert result[0]["held_delta"] == 4
    assert result[0]["held_residual"] == 1
    assert result[1]["held_delta"] is None
    assert result[1]["diagnostic_episode"] == 2
