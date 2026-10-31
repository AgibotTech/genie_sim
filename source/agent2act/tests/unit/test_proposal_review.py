# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Candidate identity, evidence gates, routing, and resource ownership."""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from agent2act.application.proposals import PolicyProposal
from agent2act.bootstrap import ApplicationFactory, default_registry
from agent2act.configuration import parse_settings
from agent2act.domain.errors import ConfigurationError, ContractError, PlanningExhaustedError
from agent2act.domain.models import ActionChunk, Decision
from agent2act.domain.review import ObservationRequest, ReviewAssessment


def assessment(execution="not_started", intent="aligned"):
    return ReviewAssessment(
        execution,
        "Observed robot and scene",
        intent,
        "Candidate heads toward the goal",
        "current operation",
        "approach target",
        "approach target",
        [],
        ["verify result"],
    )


def test_proposal_identity_and_single_consumption(observation):
    source = Decision("policy", observation.instruction, "candidate", backend="p", policy_capability="c")
    chunk = ActionChunk("JOINT_ABS", "base_link", 0.05, [[0.01]], [[0]], [[0, 0]])
    p = PolicyProposal(observation, 1, source, chunk, {"trajectory": []}, 1, 10)
    chunk.left_arm[0][0] = 99
    accepted = replace(source, proposal_id=p.id)
    with pytest.raises(ContractError, match="exact reviewed"):
        p.consume(observation, 1, replace(accepted, backend="wrong"))
    changed = deepcopy(observation)
    changed.joints[0] += 0.01
    with pytest.raises(ContractError, match="observation changed"):
        p.consume(changed, 1, accepted)
    assert p.consume(observation, 1, accepted).left_arm == [[0.01]]
    with pytest.raises(ContractError, match="already consumed"):
        p.consume(observation, 1, accepted)


def test_proposal_expires_without_action(observation, monkeypatch):
    source = Decision("policy", observation.instruction, "candidate")
    chunk = ActionChunk("JOINT_ABS", "base_link", 0.05, [[0]], [[0]], [[0, 0]])
    p = PolicyProposal(observation, 1, source, chunk, {}, 1, 5)
    monkeypatch.setattr("agent2act.application.proposals.time.monotonic", lambda: p.created + 6)
    with pytest.raises(ContractError, match="expired"):
        p.consume(observation, 1, replace(source, proposal_id=p.id))
    assert not p.consumed


class Policy:
    def __init__(self):
        self.calls = 0
        self.closed = False
        self.fail = False

    def describe(self):
        return {
            "capabilities": [
                dict(name="whole", description="test", instruction_scope="original_task", max_policy_chunks=2)
            ]
        }

    def generate(self, obs, instruction):
        self.calls += 1
        if self.fail:
            raise RuntimeError("backend failure")
        assert instruction == obs.instruction
        return ActionChunk(
            "JOINT_ABS", "base_link", 0.05, [[0.01 * self.calls]] * 3, [[0]] * 3, [list(obs.grippers)] * 3
        )

    def reset(self):
        pass

    def close(self):
        self.closed = True


class Geometry:
    def __init__(self):
        self.closed = False

    def preview(self, obs, chunk, steps):
        return {"trajectory": [{"target": chunk.left_arm[-1]}]}

    def reset(self):
        pass

    def close(self):
        self.closed = True


class Reviewer:
    def __init__(self, mode="policy", intent="aligned", budget=1):
        self.mode, self.intent, self.budget = mode, intent, budget
        self.contexts = []
        self.closed = False

    def decide(self, obs, context):
        self.contexts.append(deepcopy(context))
        p = context["proposal"]
        data = dict(
            mode=self.mode,
            instruction=obs.instruction,
            reason="offline review",
            assessment=asdict(
                assessment("not_started" if not context["tracking_feedback"] else "progressing", self.intent)
            ),
        )
        if self.mode == "policy":
            data.update(
                backend="p",
                policy_capability="whole",
                proposal_id=p["proposal_id"],
                policy_chunks=self.budget,
            )
        else:
            data.update(motion={"frame": "base_link", "waypoints": [{"duration_s": 0.05}]})
        return Decision.from_dict(data)

    def reset(self):
        pass

    def close(self):
        self.closed = True


def factory(config, reviewer, policy, geometry):
    config["execution"].update(
        mode="auto",
        allowed_modes=["motion", "policy"],
        proposal_review=True,
        proposal_backend="p",
        proposal_capability="whole",
        max_replans=0,
        execute_steps=1,
        policy_execute_steps=2,
        max_policy_chunks=4,
    )
    config["policy"] = {"default": "p", "backends": {"p": {"type": "fake"}}}
    config["kinematics"] = {"type": "fake"}
    config["planner"] = {"type": "fake"}
    registry = default_registry()
    registry.policies.register("fake", lambda *_: policy)
    registry.planners.register("fake", lambda _: reviewer)
    registry.kinematics.register("fake", lambda *_: geometry)
    return ApplicationFactory(parse_settings(config), registry)


def test_review_consumes_same_candidate_then_infers_fresh(config, observation):
    policy, reviewer, geometry = Policy(), Reviewer(), Geometry()
    with factory(config, reviewer, policy, geometry).create_session() as session:
        first = session.runtime.step(observation)
        second = session.runtime.step(observation)
        assert first.left_arm == [[0.01], [0.01]]
        assert second.left_arm == [[0.02], [0.02]]
        assert policy.calls == 2
        assert first.metadata["proposal_id"] != second.metadata["proposal_id"]
        assert reviewer.contexts[1]["previous_assessment"]["remaining"] == ["verify result"]
    assert policy.closed and reviewer.closed and geometry.closed


def test_expired_review_does_not_replan_same_observation(config, observation, monkeypatch):
    policy, reviewer, geometry = Policy(), Reviewer(), Geometry()
    now = [0.0]
    monkeypatch.setattr("agent2act.application.proposals.time.monotonic", lambda: now[0])
    decide = reviewer.decide

    def slow_review(obs, context):
        decision = decide(obs, context)
        now[0] += 301
        return decision

    monkeypatch.setattr(reviewer, "decide", slow_review)
    f = factory(config, reviewer, policy, geometry)
    config["execution"]["max_replans"] = 2
    f = ApplicationFactory(parse_settings(config), f.registry)
    with f.create_session() as session:
        with pytest.raises(PlanningExhaustedError, match="fresh observation required"):
            session.runtime.step(observation)
        assert len(reviewer.contexts) == 1
        assert policy.calls == 1
        assert session.runtime._last_issued is None
        assert session.runtime._policy_run is None


def test_bounded_grant_skips_review_but_not_policy_inference(config, observation):
    policy, reviewer, geometry = Policy(), Reviewer(budget=2), Geometry()
    with factory(config, reviewer, policy, geometry).create_session() as session:
        session.runtime.step(observation)
        continued = session.runtime.step(observation)
        assert "proposal_id" not in continued.metadata
        assert len(reviewer.contexts) == 1 and policy.calls == 2
        session.runtime.step(observation)
        assert len(reviewer.contexts) == 2 and policy.calls == 3
        assert reviewer.contexts[-1]["policy_review"]["reason"] == "budget_exhausted"


@pytest.mark.parametrize(
    "mode,intent,match", [("auto", "uncertain", "correction requires"), ("policy", "misaligned", "forbidden")]
)
def test_uncertainty_and_fixed_policy_do_not_authorize_correction(config, observation, mode, intent, match):
    policy, reviewer, geometry = Policy(), Reviewer("motion", intent), Geometry()
    f = factory(config, reviewer, policy, geometry)
    f = ApplicationFactory(f._settings.override(mode=mode), f.registry)
    with f.create_session() as session:
        with pytest.raises(PlanningExhaustedError, match=match):
            session.runtime.step(observation)
        assert session.runtime._last_issued is None


def test_correction_discards_candidate_and_next_step_reinfers(config, observation):
    policy, reviewer, geometry = Policy(), Reviewer("motion", "misaligned"), Geometry()
    with factory(config, reviewer, policy, geometry).create_session() as session:
        assert session.runtime.step(observation).kind == "EEF_ABS"
        assert session.runtime._policy_run is None
        reviewer.mode = "policy"
        assert session.runtime.step(observation).left_arm == [[0.02], [0.02]]
        assert policy.calls == 2


def test_fixed_motion_never_generates_proposal(config, observation):
    policy, reviewer, geometry = Policy(), Reviewer("motion"), Geometry()
    f = factory(config, reviewer, policy, geometry)
    with ApplicationFactory(f._settings.override(mode="motion"), f.registry).create_session() as session:
        session.runtime.step(observation)
        assert policy.calls == 0 and reviewer.contexts[0]["proposal"] is None


def test_capability_budget_checked_before_issuing(config, observation):
    policy, reviewer, geometry = Policy(), Reviewer(budget=3), Geometry()
    with factory(config, reviewer, policy, geometry).create_session() as session:
        with pytest.raises(PlanningExhaustedError, match="capability budget"):
            session.runtime.step(observation)
        assert session.runtime._last_issued is None


def test_backend_failure_revokes_grant(config, observation):
    policy, reviewer, geometry = Policy(), Reviewer(budget=2), Geometry()
    with factory(config, reviewer, policy, geometry).create_session() as session:
        session.runtime.step(observation)
        policy.fail = True
        with pytest.raises(RuntimeError, match="backend failure"):
            session.runtime.step(observation)
        assert session.runtime._policy_run is None
        policy.fail = False
        session.runtime.step(observation)
        assert len(reviewer.contexts) == 2


def test_construction_failure_closes_geometry_and_models(config):
    policy, reviewer, geometry = Policy(), Reviewer(), Geometry()
    f = factory(config, reviewer, policy, geometry)
    policy.describe = lambda: {"capabilities": []}
    with pytest.raises(ConfigurationError, match="original_task"):
        f.create_session()
    assert policy.closed and reviewer.closed and geometry.closed


def test_episode_boundary_clears_review_progress_and_grant(config, observation):
    policy, reviewer, geometry = Policy(), Reviewer(budget=2), Geometry()
    with factory(config, reviewer, policy, geometry).create_session() as session:
        session.runtime.step(observation)
        done = session.runtime.step(replace(observation, done=True))
        assert done.metadata["status"] == "episode_done"
        assert policy.calls == 1 and session.runtime._assessment is None
        session.runtime.step(replace(observation, episode_key=("other-task", 2)))
        assert reviewer.contexts[-1]["previous_assessment"] is None
        assert reviewer.contexts[-1]["policy_review"] is None
        assert len(reviewer.contexts) == 2


def test_corobot_session_consumes_candidate_and_records_trace(config, request_payload, tmp_path):
    policy, reviewer, geometry = Policy(), Reviewer(), Geometry()
    config["logging"] = {"trace_dir": str(tmp_path)}
    with factory(config, reviewer, policy, geometry).create_session() as session:
        response = session.handle(request_payload)["result"]
        assert response["left_arm"]["values"] == [[0.01], [0.01]]
        assert (
            response["agent2act"]["reviewed_proposal_id"] == reviewer.contexts[-1]["proposal"]["proposal_id"]
        )
        assert response["agent2act"]["assessment"]["execution_status"] == "not_started"
        assert policy.calls == 1
    assert len(list(tmp_path.rglob("*-simulator-response.json"))) == 1


def observation_request():
    return {
        "missing_evidence": "The supported object's receiving surface is occluded.",
        "candidate_commitment": "The candidate removes the current grasp before the surface is visible.",
        "expected_observation": "Moving the empty camera arm reveals the surface from another angle.",
        "preserved_conditions": ["Retain the grasp", "Keep the held arm stationary", "Clear free space"],
    }


class InformationReviewer(Reviewer):
    def __init__(self):
        super().__init__("motion", "uncertain")
        self.gripper_target = None

    def decide(self, obs, context):
        decision = super().decide(obs, context)
        if self.mode == "motion":
            decision.assessment.observation_request = ObservationRequest.from_dict(observation_request())
            target = obs.eef("right", "base_link")
            target[0] += 0.02
            waypoint = {"right_pose": target}
            waypoints = [waypoint]
            if self.gripper_target is not None:
                waypoints.append({"grippers": self.gripper_target})
            decision.motion = {"frame": "base_link", "waypoints": waypoints}
        return decision


@pytest.mark.parametrize(
    "field,value",
    [
        ("missing_evidence", ""),
        ("candidate_commitment", " "),
        ("expected_observation", None),
        ("preserved_conditions", []),
        ("preserved_conditions", [""]),
        ("preserved_conditions", "maintain support"),
    ],
)
def test_information_request_requires_complete_evidence(field, value):
    data = asdict(assessment(intent="uncertain"))
    data["observation_request"] = {**observation_request(), field: value}
    with pytest.raises(ContractError, match="observation_request requires"):
        ReviewAssessment.from_dict(data)


def test_information_request_cannot_label_known_alignment_or_accept_policy():
    data = asdict(assessment())
    data["observation_request"] = observation_request()
    with pytest.raises(ContractError, match="requires uncertain intent"):
        ReviewAssessment.from_dict(data)
    data["intent_status"] = "uncertain"
    for mode, status in (("policy", "continue"), ("motion", "complete")):
        with pytest.raises(ContractError, match="requires continuing motion"):
            Decision.from_dict(
                dict(mode=mode, status=status, instruction="observe", reason="test", assessment=data)
            )


def test_information_action_preserves_commands_and_requires_fresh_review(config, observation):
    policy, reviewer, geometry = Policy(), InformationReviewer(), Geometry()
    f = factory(config, reviewer, policy, geometry)
    with f.create_session() as session:
        first = session.runtime.step(observation)
        assert first.kind == "EEF_ABS"
        assert first.right_arm[-1][0] > observation.eef("right", "base_link")[0]
        assert first.left_arm[-1] == observation.eef("left", "base_link")
        assert first.grippers == [observation.grippers]
        assert session.runtime._policy_run is None
        assert session.runtime._pending_motion is None
        # A contact/lag residual is not a new setpoint. The information motion
        # must preserve the command actually issued, not the measured aperture.
        changed_measurement = replace(observation, grippers=[0.48, 0.48])
        second = session.runtime.step(changed_measurement)
        assert second.grippers == [observation.grippers]
        reviewer.mode = "policy"
        third = session.runtime.step(observation)
        assert third.left_arm == [[0.03], [0.03]]
        assert len(reviewer.contexts) == policy.calls == 3
        assert reviewer.contexts[-1]["previous_assessment"]["observation_request"] == {
            **observation_request(),
            "command_bounds": None,
        }


@pytest.mark.parametrize("hand", [0, 1])
def test_information_action_rejects_gripper_change_even_beyond_issued_prefix(config, observation, hand):
    policy, reviewer, geometry = Policy(), InformationReviewer(), Geometry()
    reviewer.gripper_target = list(observation.grippers)
    reviewer.gripper_target[hand] += 0.02
    with factory(config, reviewer, policy, geometry).create_session() as session:
        with pytest.raises(PlanningExhaustedError, match="preserve both gripper commands"):
            session.runtime.step(observation)
        assert session.runtime._last_issued is None


def test_information_action_does_not_override_fixed_policy(config, observation):
    policy, reviewer, geometry = Policy(), InformationReviewer(), Geometry()
    f = factory(config, reviewer, policy, geometry)
    with ApplicationFactory(f._settings.override(mode="policy"), f.registry).create_session() as session:
        with pytest.raises(PlanningExhaustedError, match="forbidden"):
            session.runtime.step(observation)
        assert session.runtime._last_issued is None


def test_information_request_requires_a_current_candidate(config, observation):
    policy, reviewer, geometry = Policy(), InformationReviewer(), Geometry()
    f = factory(config, reviewer, policy, geometry)
    with ApplicationFactory(f._settings.override(mode="motion"), f.registry).create_session() as session:
        with pytest.raises(PlanningExhaustedError, match="requires a current reviewed candidate"):
            session.runtime.step(observation)
        assert policy.calls == 0
        assert session.runtime._last_issued is None


def motion_alternative():
    return dict(
        expected_effect="Advance the visible free hand toward its observed target",
        geometry_evidence="Current measured wrist and visible target establish the offset",
        path_evidence="The short translation remains in observed free space",
        omitted_effects="No grasp or support change is required for this free-space translation",
        preserved_conditions=["Both gripper commands", "The supported object remains stationary"],
    )


class AlternativeReviewer(Reviewer):
    def __init__(self):
        super().__init__("motion", "aligned")
        self.duration = 0.05
        self.gripper_target = None

    def decide(self, obs, context):
        from agent2act.domain.review import MotionAlternative

        decision = super().decide(obs, context)
        if self.mode == "motion":
            decision.assessment.motion_alternative = MotionAlternative.from_dict(motion_alternative())
            target = obs.eef("right", "base_link")
            target[0] += 0.002
            waypoints = [{"right_pose": target, "duration_s": self.duration}]
            if self.gripper_target is not None:
                waypoints.append({"grippers": self.gripper_target})
            decision.motion = {"frame": "base_link", "waypoints": waypoints}
        return decision


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_effect", ""),
        ("geometry_evidence", None),
        ("path_evidence", " "),
        ("omitted_effects", []),
        ("preserved_conditions", []),
        ("preserved_conditions", [""]),
    ],
)
def test_motion_alternative_requires_evidence(field, value):
    data = asdict(assessment())
    data["motion_alternative"] = {**motion_alternative(), field: value}
    with pytest.raises(ContractError, match="motion_alternative requires"):
        ReviewAssessment.from_dict(data)


@pytest.mark.parametrize("intent", ["uncertain", "misaligned"])
def test_motion_alternative_does_not_relabel_uncertainty_as_efficiency(intent):
    data = asdict(assessment(intent=intent))
    data["motion_alternative"] = motion_alternative()
    with pytest.raises(ContractError, match="requires aligned intent"):
        ReviewAssessment.from_dict(data)


@pytest.mark.parametrize(
    "mode,status,budget", [("policy", "continue", 1), ("motion", "complete", 1), ("motion", "continue", 2)]
)
def test_motion_alternative_requires_single_continuing_motion(mode, status, budget):
    data = asdict(assessment())
    data["motion_alternative"] = motion_alternative()
    with pytest.raises(ContractError, match="requires continuing motion"):
        Decision.from_dict(
            dict(
                mode=mode,
                status=status,
                policy_chunks=budget,
                instruction="move",
                reason="evidence",
                assessment=data,
            )
        )


def test_shorter_aligned_motion_preserves_commands_and_returns_to_review(config, observation):
    policy, reviewer, geometry = Policy(), AlternativeReviewer(), Geometry()
    with factory(config, reviewer, policy, geometry).create_session() as session:
        first = session.runtime.step(observation)
        assert first.steps == 1
        assert first.right_arm[-1][0] > observation.eef("right", "base_link")[0]
        assert first.grippers == [observation.grippers]
        cost = first.metadata["motion_alternative"]
        assert cost["motion_duration_s"] == 0.05
        assert cost["policy_prefix_duration_s"] == 0.1
        assert session.runtime._policy_run is None and session.runtime._pending_motion is None
        changed = replace(observation, grippers=[0.48, 0.48])
        second = session.runtime.step(changed)
        assert second.grippers == [observation.grippers]
        reviewer.mode = "policy"
        session.runtime.step(observation)
        assert policy.calls == len(reviewer.contexts) == 3
        assert reviewer.contexts[-1]["previous_assessment"]["motion_alternative"] == motion_alternative()


@pytest.mark.parametrize("mode", ["motion", "policy"])
def test_motion_alternative_requires_current_candidate_and_respects_fixed_mode(config, observation, mode):
    policy, reviewer, geometry = Policy(), AlternativeReviewer(), Geometry()
    f = factory(config, reviewer, policy, geometry)
    with ApplicationFactory(f._settings.override(mode=mode), f.registry).create_session() as session:
        with pytest.raises(PlanningExhaustedError, match="current reviewed candidate|forbidden"):
            session.runtime.step(observation)
        assert session.runtime._last_issued is None


@pytest.mark.parametrize("hand", [0, 1])
def test_motion_alternative_checks_grippers_in_unissued_tail(config, observation, hand):
    policy, reviewer, geometry = Policy(), AlternativeReviewer(), Geometry()
    reviewer.gripper_target = list(observation.grippers)
    reviewer.gripper_target[hand] += 0.02
    with factory(config, reviewer, policy, geometry).create_session() as session:
        with pytest.raises(PlanningExhaustedError, match="preserve both gripper commands"):
            session.runtime.step(observation)
        assert session.runtime._last_issued is None


@pytest.mark.parametrize(
    "motion_limit,message", [(1, "entire executable prefix"), (2, "shorter than the policy prefix")]
)
def test_motion_alternative_cannot_claim_savings_from_truncation_or_policy_suffix(
    config, observation, motion_limit, message
):
    policy, reviewer, geometry = Policy(), AlternativeReviewer(), Geometry()
    reviewer.duration = 0.1
    f = factory(config, reviewer, policy, geometry)
    config["execution"]["execute_steps"] = motion_limit
    settings = parse_settings(config)
    with ApplicationFactory(settings, f.registry).create_session() as session:
        with pytest.raises(PlanningExhaustedError, match=message):
            session.runtime.step(observation)
        assert session.runtime._last_issued is None


class BoundedProbeReviewer(Reviewer):
    def __init__(self):
        super().__init__("motion", "uncertain")
        self.bounds = {
            "frame": "base_link",
            "left": {"translation_m": 0.005, "rotation_rad": 0.01, "gripper_change": 0.1},
            "right": {"translation_m": 0.005, "rotation_rad": 0.01, "gripper_change": 0},
        }
        self.extra = {}
        self.retry_accept = False

    def decide(self, obs, context):
        if self.retry_accept and self.contexts:
            self.mode, self.intent = "policy", "aligned"
        decision = super().decide(obs, context)
        if self.mode == "motion":
            decision.assessment.observation_request = ObservationRequest.from_dict(
                {**observation_request(), "command_bounds": self.bounds}
            )
            decision.motion = {
                "frame": "base_link",
                "waypoints": [{"grippers": [0.55, 0.5], **self.extra}],
            }
        return decision


def probe_factory(config, reviewer, geometry=None):
    policy = Policy()
    f = factory(config, reviewer, policy, geometry or Geometry())
    config["execution"].update(execute_steps=100, max_replans=1 if reviewer.retry_accept else 0)
    return ApplicationFactory(parse_settings(config), f.registry), policy


def test_bounded_closure_probe_requires_fresh_review_and_uses_command_reference(config, observation):
    reviewer = BoundedProbeReviewer()
    f, policy = probe_factory(config, reviewer)
    with f.create_session() as session:
        first = session.runtime.step(observation)
        assert first.grippers[-1] == [0.55, 0.5]
        assert all(row == observation.eef("left", "base_link") for row in first.left_arm)
        assert first.metadata["discarded_steps"] == 0
        assert "command_bounds" in first.metadata
        reviewer.bounds["left"]["gripper_change"] = 0
        # Measurement lags, but preserving the last issued command remains legal.
        second = session.runtime.step(replace(observation, grippers=[0.48, 0.48]))
        assert second.grippers[-1] == [0.55, 0.5]
        assert policy.calls == len(reviewer.contexts) == 2
        assert session.runtime._policy_run is None
        assert session.runtime._pending_motion is None


@pytest.mark.parametrize("violation", ["translation", "rotation", "other_gripper", "frame", "truncation"])
def test_bounded_probe_rejects_violation(config, observation, violation):
    reviewer = BoundedProbeReviewer()
    expected = "command_bounds"
    if violation == "translation":
        target = observation.eef("left", "base_link")
        target[0] += 0.02
        reviewer.extra = {"left_pose": target}
    elif violation == "rotation":
        import math

        target = observation.eef("right", "base_link")
        target[3:] = [0, 0, math.sin(0.05), math.cos(0.05)]
        reviewer.extra = {"right_pose": target}
    elif violation == "other_gripper":
        reviewer.extra = {"grippers": [0.55, 0.52]}
    elif violation == "frame":
        reviewer.bounds["frame"] = "arm_base_link"
    else:
        reviewer.extra = {"duration_s": 6}
        expected = "entire executable prefix"
    f, _ = probe_factory(config, reviewer)
    with f.create_session() as session:
        with pytest.raises(PlanningExhaustedError, match=expected):
            session.runtime.step(observation)
        assert session.runtime._last_issued is None


def test_rejected_probe_can_accept_same_candidate_without_policy_call(config, observation):
    reviewer = BoundedProbeReviewer()
    reviewer.extra = {"grippers": [0.55, 0.52]}
    reviewer.retry_accept = True
    f, policy = probe_factory(config, reviewer)
    with f.create_session() as session:
        result = session.runtime.step(observation)
        assert result.kind == "JOINT_ABS"
        assert policy.calls == 1
        assert len(reviewer.contexts) == 2
        assert (
            reviewer.contexts[0]["proposal"]["proposal_id"] == reviewer.contexts[1]["proposal"]["proposal_id"]
        )


def test_bounded_probe_still_respects_fixed_routing(config, observation):
    f, _ = probe_factory(config, BoundedProbeReviewer())
    with ApplicationFactory(f._settings.override(mode="policy"), f.registry).create_session() as session:
        with pytest.raises(PlanningExhaustedError, match="forbidden"):
            session.runtime.step(observation)


@pytest.mark.parametrize("provide_envelope", [False, True])
def test_joint_probe_requires_all_command_geometry(config, observation, provide_envelope):
    from agent2act.domain.command_bounds import wrist_envelope

    class JointProbe(Policy):
        mode = "motion"

        def generate(self, obs, decision):
            return ActionChunk("JOINT_ABS", "base_link", 0.05, [[0]], [[0]], [[0.55, 0.5]])

    class ProbeGeometry(Geometry):
        def __init__(self):
            super().__init__()
            self.lengths = []

        def preview(self, obs, chunk, steps):
            self.lengths.append((chunk.steps, steps))
            if not provide_envelope:
                return super().preview(obs, chunk, steps)
            # Deterministic port fixture; real URDF all-step FK has separate analytic tests.
            return {
                "command_envelope": wrist_envelope(
                    obs,
                    chunk.frame,
                    [obs.eef("left", chunk.frame)] * chunk.steps,
                    [obs.eef("right", chunk.frame)] * chunk.steps,
                )
            }

    geometry = ProbeGeometry()
    f, policy = probe_factory(config, BoundedProbeReviewer(), geometry)
    f.registry.motions.register("joint_probe", lambda *_: JointProbe())
    config["motion_backend"] = {"type": "joint_probe"}
    with ApplicationFactory(parse_settings(config), f.registry).create_session() as session:
        if provide_envelope:
            result = session.runtime.step(observation)
            assert result.grippers == [[0.55, 0.5]]
        else:
            with pytest.raises(PlanningExhaustedError, match="command_bounds"):
                session.runtime.step(observation)
            assert session.runtime._last_issued is None
        assert geometry.lengths == [(3, 2), (1, 1)]
        assert policy.calls == 1
