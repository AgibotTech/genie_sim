# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Ranked target selection uses analytic IK, with no simulator or live model."""

import copy
import json
from dataclasses import asdict, replace

import pytest

from agent2act.adapters.kinematics.urdf import UrdfKinematics
from agent2act.adapters.planners.chat_completions import ChatCompletionsPlanner
from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.ranked_motion import RankedMotionExecutor, RankedMotionPlanner
from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.adapters.planners.scripted import ScriptedPlanner
from agent2act.adapters.planners.tool_center import ToolCenter
from agent2act.bootstrap import ApplicationFactory, _joint_motion, build_session, default_registry
from agent2act.configuration import parse_settings
from agent2act.domain.errors import ConfigurationError, PlanningExhaustedError
from agent2act.domain.models import ContractError, Decision
from agent2act.domain.timing import MotionTiming


@pytest.fixture
def robot(tmp_path, spec, observation):
    path = tmp_path / "robot.urdf"
    path.write_text(f'''<robot>
    <joint name="{spec.left_joints[0]}" type="prismatic"><parent link="base_link"/>
    <child link="left"/><axis xyz="1 0 0"/><limit lower="-0.5" upper="0.5" velocity="1"/></joint>
    <joint name="{spec.right_joints[0]}" type="prismatic"><parent link="base_link"/>
    <child link="right"/><axis xyz="0 1 0"/><limit lower="-0.5" upper="0.5" velocity="1"/></joint>
    </robot>''')
    options = dict(path=str(path), frame="base_link", left_link="left", right_link="right")
    fk = UrdfKinematics(spec, options)
    obs = replace(
        observation,
        end_pose={
            "base_link": {
                side + "_arm": dict(position=p[:3], orientation=p[3:])
                for side, p in fk.poses(observation.joints, []).items()
            }
        },
    )
    fk.close()
    return options, obs


def batch(obs, targets):
    return Decision(
        "motion",
        obs.instruction,
        "ranked targets",
        motion={"frame": "base_link", "candidates": [{"left_pose": target} for target in targets]},
    )


def targets():
    return [[0, 0.1, 0, 0, 0, 0, 1]] + [[x, 0, 0, 0, 0, 0, 1] for x in (0.02, 0.03, 0.04, 0.05)]


def test_ranked_duration_only_hold_continues_without_new_batch(config, spec, robot):
    options, obs = robot
    config["ranked_motion_candidates"] = True
    config["motion_backend"] = {"type": "urdf_joint", **options}
    config["execution"].update(execute_steps=8, max_motion_continuations=1)
    decision = batch(obs, targets())
    decision.motion["candidates"] = [{"duration_s": 16 * spec.dt_s}] * 5
    config["planner"]["decisions"] = [asdict(decision)]
    events = []
    with build_session(config, event_sink=events.append) as session:
        first = session.runtime.step(obs)
        second = session.runtime.step(copy.deepcopy(obs))
        assert first.steps == second.steps == 8
        assert first.left_arm == second.left_arm == [[obs.joints[0]]] * 8
        assert first.right_arm == second.right_arm == [[obs.joints[1]]] * 8
        assert first.metadata["ranked_motion_selection"]["selected_rank"] == 1
        assert len(first.metadata["ranked_motion_selection"]["checks"]) == 5
        assert session.runtime._pending_motion is None
    assert len([e for e in events if e.get("status") == "motion_continued"]) == 1


def test_checks_all_five_in_order_and_reuses_first_passing_chunk(spec, robot, monkeypatch):
    options, obs = robot
    base = _joint_motion(spec, MotionTiming(), options)
    executor = RankedMotionExecutor(base, spec)
    planner = RankedMotionPlanner(ScriptedPlanner([asdict(batch(obs, targets()))]), executor)
    calls = []
    generate = base.generate

    def counted(observation, decision):
        calls.append(copy.deepcopy(decision.motion))
        return generate(observation, decision)

    monkeypatch.setattr(base, "generate", counted)
    before = copy.deepcopy(obs)
    selected = planner.decide(obs, {})
    assert len(calls) == 5
    assert [m["waypoints"][0]["left_pose"] for m in calls] == targets()
    assert selected.motion == {"frame": "base_link", "waypoints": [{"left_pose": targets()[1]}]}
    chunk = executor.generate(obs, Decision.from_dict(asdict(selected)))
    assert len(calls) == 5  # No second IK solve during issuance.
    assert chunk.left_arm[-1] == pytest.approx([0.02], abs=1e-5)
    report = chunk.metadata["ranked_motion_selection"]
    assert report["selected_rank"] == 2
    assert [c["status"] for c in report["checks"]] == ["rejected"] + ["passed"] * 4
    assert "did not converge" in report["checks"][0]["reason"]
    assert obs == before
    # New measured state must invalidate the cached commands.
    planner.planner.reset()
    selected = planner.decide(obs, {})
    newer = copy.deepcopy(obs)
    newer.issued_gripper_targets = [0.6, 0.7]
    executor.generate(newer, selected)
    assert len(calls) == 11
    planner.close()
    executor.close()
    executor.close()
    assert base.closed and base.fk.closed


@pytest.mark.parametrize("count", [0, 1, 4, 6])
def test_requires_exactly_five_alternatives(spec, robot, count):
    options, obs = robot
    executor = RankedMotionExecutor(_joint_motion(spec, MotionTiming(), options), spec)
    with pytest.raises(ContractError, match="exactly five"):
        executor.select(obs, batch(obs, [targets()[1]] * count))
    executor.close()


def test_ten_candidates_checks_every_target_and_selects_sixth(spec, robot):
    options, obs = robot
    executor = RankedMotionExecutor(_joint_motion(spec, MotionTiming(), options), spec, 10)
    proposed = [targets()[0]] * 5 + [targets()[1]] * 5
    planner = RankedMotionPlanner(ScriptedPlanner([asdict(batch(obs, proposed))]), executor)
    selected = planner.decide(obs, {})
    chunk = executor.generate(obs, selected)
    report = chunk.metadata["ranked_motion_selection"]
    assert executor.describe()["ranked_motion_candidates"] == 10
    assert report["candidate_count"] == 10
    assert report["selected_rank"] == 6
    assert [r["status"] for r in report["checks"]] == ["rejected"] * 5 + ["passed"] * 5
    assert chunk.left_arm[-1] == pytest.approx([0.02], abs=1e-5)
    with pytest.raises(ContractError, match="exactly ten"):
        executor.select(obs, batch(obs, proposed[:5]))
    with pytest.raises(ContractError, match="all ten"):
        executor.select(obs, batch(obs, [targets()[0]] * 10))
    planner.close()
    executor.close()


@pytest.mark.parametrize("count", [True, "10", 10.0, 6, 0, 20])
def test_candidate_count_rejects_unsupported_or_noninteger_values(config, count):
    config["ranked_motion_candidate_count"] = count
    with pytest.raises(ConfigurationError, match="ranked_motion_candidate_count"):
        parse_settings(config)


def test_ten_candidate_bootstrap_selects_passing_rank(config, spec, robot):
    options, obs = robot
    config.update(
        ranked_motion_candidates=True,
        ranked_motion_candidate_count=10,
        motion_backend={"type": "urdf_joint", **options},
    )
    config["planner"]["decisions"] = [asdict(batch(obs, [targets()[0]] * 5 + [targets()[1]] * 5))]
    with build_session(config) as session:
        chunk = session.runtime.step(obs)
        assert chunk.metadata["ranked_motion_selection"]["selected_rank"] == 6


def test_all_fail_replans_with_each_reason_and_preserves_budget(config, spec, robot):
    options, obs = robot
    config["ranked_motion_candidates"] = True
    config["motion_backend"] = {"type": "urdf_joint", **options}
    config["execution"]["max_replans"] = 1
    bad = asdict(batch(obs, [targets()[0]] * 5))
    good = asdict(batch(obs, targets()))
    config["planner"] = {"type": "scripted", "decisions": [bad, good]}
    contexts = []
    registry = default_registry()

    class RecordingPlanner(ScriptedPlanner):
        def decide(self, observation, context):
            contexts.append(copy.deepcopy(context))
            return super().decide(observation, context)

    registry.planners.register("recording", lambda o: RecordingPlanner(**o))
    config["planner"]["type"] = "recording"
    with ApplicationFactory(parse_settings(config), registry).create_session() as session:
        chunk = session.runtime.step(obs)
        assert chunk.metadata["ranked_motion_selection"]["selected_rank"] == 2
        assert len(contexts) == 2
        text = json.dumps(contexts[1])
        for rank in range(1, 6):
            assert f"candidate {rank}:" in text
        assert contexts[1]["planning_attempt"] == 1
    config["planner"]["decisions"] = [bad, bad]
    with ApplicationFactory(parse_settings(config), registry).create_session() as session:
        with pytest.raises(PlanningExhaustedError, match="all five"):
            session.runtime.step(obs)


def test_unsupported_executor_fails_and_closes_resources(config, monkeypatch):
    config["ranked_motion_candidates"] = True
    registry = default_registry()
    created = []
    original = registry.motions.resolve("local_interpolation")

    def build(s, t, o):
        motion = original(s, t, o)
        created.append(motion)
        monkeypatch.setattr(motion, "close", lambda: created.append("closed"))
        return motion

    registry.motions.register("test", build)
    config["motion_backend"] = {"type": "test"}
    with pytest.raises(ConfigurationError, match="endpoint IK"):
        ApplicationFactory(parse_settings(config), registry).create_session()
    assert "closed" in created


def test_later_construction_failure_closes_ranked_components(config, robot):
    options, _ = robot
    config.update(
        ranked_motion_candidates=True,
        motion_backend={"type": "urdf_joint", **options},
        kinematics={"type": "missing"},
    )
    registry = default_registry()
    created = []

    def build(s, t, o):
        motion = _joint_motion(s, t, o)
        created.append(motion)
        return motion

    registry.motions.register("test", build)
    config["motion_backend"]["type"] = "test"
    with pytest.raises(ConfigurationError, match="unknown kinematics"):
        ApplicationFactory(parse_settings(config), registry).create_session()
    assert created[0].closed and created[0].fk.closed


def test_center_converts_every_candidate_without_mutating_model_output(observation):
    original = batch(observation, targets())
    converted = ToolCenter({"left": [0, 0, 0.1], "right": [0, 0, 0.1]}).decision(original)
    assert [c["left_pose"][2] for c in converted.motion["candidates"]] == [-0.1] * 5
    assert [c["left_pose"][2] for c in original.motion["candidates"]] == [0] * 5


def test_shared_model_prompt_requests_ranked_batch_and_reset_clears_it(observation, monkeypatch):
    planner = ResponsesPlanner(model="fixture", api_key_env=None)
    context = {
        "execution": {
            "mode": "motion",
            "allowed_modes": ["motion"],
            "motion_frame": "base_link",
            "executors": {"motion": {"ranked_motion_candidates": 5}},
        }
    }
    payload = asdict(batch(observation, targets()))
    monkeypatch.setattr(
        planner,
        "post",
        lambda body: {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(payload)}]}]
        },
    )
    planner.decide(observation, context)
    instructions = planner._context_instructions()
    assert "exactly FIVE" in instructions and '"candidates"' in instructions
    planner.reset()
    assert "exactly FIVE" not in planner._context_instructions()
    planner.close()


def test_terminal_decision_and_reset_do_not_require_or_reuse_candidates(spec, robot):
    options, obs = robot
    executor = RankedMotionExecutor(_joint_motion(spec, MotionTiming(), options), spec)
    done = Decision("motion", obs.instruction, "inspect completion", status="complete")
    planner = RankedMotionPlanner(ScriptedPlanner([asdict(batch(obs, targets())), asdict(done)]), executor)
    planner.decide(obs, {})
    assert executor.pending is not None
    assert planner.decide(obs, {}) == done
    assert executor.pending is None
    planner.reset()
    selected = planner.decide(obs, {})
    assert executor.pending is not None
    executor.reset()
    assert executor.pending is None
    assert executor.generate(obs, selected).left_arm[-1] == pytest.approx([0.02], abs=1e-5)
    planner.close()
    executor.close()


@pytest.mark.parametrize("value", [1, "true", None])
def test_candidate_switch_requires_boolean(config, value):
    config["ranked_motion_candidates"] = value
    with pytest.raises(ConfigurationError, match="ranked_motion_candidates"):
        parse_settings(config)


def test_candidate_switch_rejects_proposal_review(config):
    config["ranked_motion_candidates"] = True
    config["execution"].update(
        proposal_review=True, proposal_backend="fixture", proposal_capability="fixture"
    )
    with pytest.raises(ConfigurationError, match="without proposal review"):
        parse_settings(config)


def test_candidate_switch_rejects_policy_routing(config):
    config["ranked_motion_candidates"] = True
    config["execution"].update(mode="auto", allowed_modes=["motion", "policy"])
    with pytest.raises(ConfigurationError, match="motion-only"):
        parse_settings(config)


@pytest.mark.parametrize("planner_type", [ResponsesPlanner, ChatCompletionsPlanner, EmbodiedPlanner])
@pytest.mark.parametrize("candidate_count", [5, 10])
def test_model_batch_normalization_center_conversion_and_execution(
    planner_type, spec, robot, candidate_count
):
    options, obs = robot
    executor = RankedMotionExecutor(_joint_motion(spec, MotionTiming(), options), spec, candidate_count)
    model_targets = [[p[0], p[1], 0.1] for p in (targets() * 2)[:candidate_count]]
    payload = asdict(batch(obs, model_targets))
    for candidate in payload["motion"]["candidates"]:
        candidate["right_pose"] = None
        candidate["grippers"] = None
    bodies = []

    def post(body):
        bodies.append(body)
        if planner_type is ChatCompletionsPlanner:
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}]}
        return {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(payload)}]}]
        }

    model = planner_type(
        model="fixture",
        api_key_env=None,
        post=post,
        gripper_center_offset_m={"left": [0, 0, 0.1], "right": [0, 0, 0.1]},
    )
    planner = RankedMotionPlanner(model, executor)
    context = {
        "execution": {
            "mode": "motion",
            "allowed_modes": ["motion"],
            "motion_frame": "base_link",
            "executors": {"motion": executor.describe()},
        }
    }
    selected = planner.decide(obs, context)
    assert len(bodies) == 1  # All five alternatives arrive in one model response.
    assert ("exactly FIVE" if candidate_count == 5 else "exactly TEN") in json.dumps(bodies[0])
    assert selected.motion["waypoints"][0]["left_pose"] == targets()[1]
    chunk = executor.generate(obs, selected)
    assert chunk.left_arm[-1] == pytest.approx([0.02], abs=1e-5)
    assert chunk.metadata["ranked_motion_selection"]["selected_rank"] == 2
    planner.close()
    executor.close()


def test_all_ik_failures_preserve_configured_replan_exhaustion_behavior(config, robot):
    options, obs = robot
    config.update(ranked_motion_candidates=True, motion_backend={"type": "urdf_joint", **options})
    config["execution"].update(max_replans=1, retry_ik_exhaustion=True)
    bad = asdict(batch(obs, [targets()[0]] * 5))
    config["planner"] = {"type": "scripted", "decisions": [bad, bad]}
    with ApplicationFactory(parse_settings(config)).create_session() as session:
        chunk = session.runtime.step(obs)
        assert chunk.metadata["status"] == "ik_replan_required"
        assert chunk.metadata.get("episode_terminal") is not True
        assert "candidate 5:" in chunk.metadata["error"]
