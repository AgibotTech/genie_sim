# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Per-session, transport-independent closed-loop application."""

import math
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

from agent2act.domain.actions import body_start, gripper_start, hold_chunk
from agent2act.domain.command_bounds import validate_command_bounds, wrist_envelope
from agent2act.domain.errors import ExpiredProposalError, LocalIKConvergenceError, PlanningExhaustedError
from agent2act.domain.geometry import quaternion_angle
from agent2act.domain.models import ActionChunk, ContractError, Decision, Observation, RobotSpec, vector
from agent2act.domain.ports import EventSink, Kinematics, Planner
from agent2act.domain.validation import ActionValidator

from .policy_run import PolicyRun
from .proposals import PolicyProposal
from .routing import ExecutionRouter

_EFFECT_TRACKING_KEYS = (
    "previous_pose",
    "observed_pose",
    "issued_pose",
    "commanded_delta_m",
    "observed_delta_m",
    "remaining_delta_m",
    "rotation_error_rad",
)


def _translation_feedback(
    before: list[float], actual: list[float], target: list[float]
) -> dict[str, list[float] | float | None]:
    """Endpoint-relative evidence, not path following or object/contact inference."""
    commanded = [b - a for a, b in zip(before[:3], target[:3])]
    observed = [b - a for a, b in zip(before[:3], actual[:3])]
    remaining = [b - a for a, b in zip(actual[:3], target[:3])]
    distance = math.hypot(*commanded)
    result: dict[str, list[float] | float | None] = {
        "commanded_delta_m": commanded,
        "observed_delta_m": observed,
        "remaining_delta_m": remaining,
        "translation_error_m": math.hypot(*remaining),
        "observed_displacement_m": math.hypot(*observed),
        "commanded_displacement_m": distance,
        "along_track_displacement_m": None,
        "along_track_fraction": None,
        "cross_track_error_m": None,
    }
    # A hold/tiny command has no useful direction. Preserve raw vectors and drift
    # without amplifying noise into an apparent completion percentage.
    if distance > 1e-4:
        direction = [value / distance for value in commanded]
        along = sum(value * axis for value, axis in zip(observed, direction))
        result.update(
            along_track_displacement_m=along,
            along_track_fraction=along / distance,
            cross_track_error_m=math.hypot(
                *(value - along * axis for value, axis in zip(observed, direction))
            ),
        )
    return result


class AgentRuntime:
    def __init__(
        self,
        spec: RobotSpec,
        planner: Planner,
        router: ExecutionRouter,
        event_sink: EventSink,
        kinematics: Kinematics | None = None,
    ) -> None:
        self.spec, self.planner, self.router = spec, planner, router
        self.validator = ActionValidator(spec)
        self.config = router.config
        self.kinematics = kinematics
        self._assessment: dict[str, Any] | None = None
        self._proposal_limit = 1
        if self.config.proposal_review and "policy" in router.describe()["allowed_modes"]:
            if kinematics is None:
                raise ContractError("proposal review requires kinematics")
            backends = router.describe()["executors"]["policy"]["backends"]
            backend = backends.get(self.config.proposal_backend, {})
            capabilities = backend.get("capabilities", [])
            if not any(
                c.get("name") == self.config.proposal_capability
                and c.get("instruction_scope") == "original_task"
                for c in capabilities
            ):
                raise ContractError("proposal source must declare an original_task capability")
            self._proposal_limit = next(
                c["max_policy_chunks"] for c in capabilities if c["name"] == self.config.proposal_capability
            )
        if self.config.execute_steps > spec.max_chunk_steps:
            raise ContractError("execute_steps exceeds robot max_chunk_steps")
        if self.router.step_limit("policy") > spec.max_chunk_steps:
            raise ContractError("policy_execute_steps exceeds robot max_chunk_steps")
        if (self.config.body_execute_steps or 0) > spec.max_chunk_steps:
            raise ContractError("body_execute_steps exceeds robot max_chunk_steps")
        # A final hold only repeats the current command, so it may exceed max_chunk_steps.
        self._hold_validator = ActionValidator(
            replace(spec, max_chunk_steps=max(spec.max_chunk_steps, self.config.complete_hold_steps or 0))
        )
        self.episode_key: tuple[str, int, str] | None = None
        self.memory: list[dict[str, Any]] = []
        self._pending_motion: Decision | None = None
        self._policy_run: PolicyRun | None = None
        self._policy_review: dict[str, Any] | None = None
        self._continuations = 0
        self._last_issued: ActionChunk | None = None
        self._last_observed: Observation | None = None
        self.sequence = 0
        self.event_sink = event_sink
        self._closed = False

    def reset(self) -> None:
        self.memory.clear()
        self._assessment = None
        if self.kinematics is not None:
            self.kinematics.reset()
        self._pending_motion = None
        self._policy_run = None
        self._policy_review = None
        self._continuations = 0
        self._last_issued = None
        self._last_observed = None
        self.sequence = 0
        self.planner.reset()
        self.router.reset()
        self.episode_key = None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            with ExitStack() as cleanup:
                if self.kinematics is not None:
                    cleanup.callback(self.kinematics.close)
                cleanup.callback(self.planner.close)
                cleanup.callback(self.router.close)

    def _record(self, event: dict[str, Any]) -> None:
        event = {"sequence": self.sequence, **event}
        self.memory.append(event)
        del self.memory[: -self.config.memory_size]
        self.event_sink({"episode": self.episode_key, "sequence": self.sequence, **event})

    @contextmanager
    def _phase(self, phase: str, attempt: int) -> Iterator[None]:
        start = time.monotonic()
        fields = {"episode": self.episode_key, "sequence": self.sequence, "phase": phase, "attempt": attempt}
        self.event_sink({**fields, "status": "phase_started"})
        outcome, exception_type = "ok", None
        try:
            yield
        except BaseException as exc:
            outcome, exception_type = "failed", type(exc).__name__
            raise
        finally:
            self.event_sink(
                {
                    **fields,
                    "status": "phase_finished",
                    "outcome": outcome,
                    "exception_type": exception_type,
                    "duration_s": round(time.monotonic() - start, 6),
                }
            )

    def _take_continuation(self, feedback: dict[str, Any], obs: Observation) -> Decision | None:
        pending, self._pending_motion = self._pending_motion, None
        if pending is None or self._continuations >= self.config.max_motion_continuations:
            return None
        for arm in ("left", "right"):
            observed = feedback.get(arm, {})
            translation_limit = self.config.continuation_translation_tolerance_m
            commanded = observed.get("commanded_displacement_m", 0)
            if commanded > 1e-4:
                translation_limit = min(translation_limit, commanded / 2)
            if (
                observed.get("translation_error_m", math.inf) > translation_limit
                or observed.get("rotation_error_rad", math.inf)
                > self.config.continuation_rotation_tolerance_rad
            ):
                return None
        if (
            max((abs(v) for v in feedback.get("gripper_command_error", [math.inf])), default=math.inf)
            > self.config.continuation_gripper_tolerance
        ):
            return None
        self._continuations += 1
        return pending

    def _remember_continuation(self, decision: Decision, remaining_steps: int) -> None:
        self._pending_motion = None
        if (
            self.config.max_motion_continuations == 0
            or remaining_steps <= 0
            or decision.mode != "motion"
            or decision.status != "continue"
            or decision.motion is None
            or len(decision.motion["waypoints"]) != 1
        ):
            return
        # Preserve the semantic target, not stale dense commands. The compiler starts from
        # the next actual observation and rechecks all speed and action constraints.
        self._pending_motion = deepcopy(decision)
        assert self._pending_motion.motion is not None
        self._pending_motion.motion["waypoints"][0]["duration_s"] = remaining_steps * self.spec.dt_s

    def _review_policy(self, run: PolicyRun, reason: str) -> None:
        self._policy_review = run.summary(reason)
        self._record({"status": "policy_review_required", **self._policy_review})

    def _tracking_feedback(self, obs: Observation) -> dict[str, Any]:
        chunk, previous = self._last_issued, self._last_observed
        if chunk is None or previous is None:
            return {}
        result: dict[str, Any] = {
            "previous_sequence": self.sequence - 1,
            "frame": chunk.frame,
            "kind": chunk.kind,
            "gripper_command_error": [
                a - b for a, b in zip(obs.command_grippers(self.spec), chunk.grippers[-1])
            ],
            "issued_steps": chunk.steps,
            "generated_steps": chunk.metadata.get("generated_steps", chunk.steps),
            "discarded_steps": chunk.metadata.get("discarded_steps", 0),
            "planner_status": chunk.metadata.get("planner_status"),
            "playback_status": "unverified; issued counts are commands, not simulator acknowledgements",
            "grippers": {
                "observed": obs.command_grippers(self.spec),
                "issued_target": list(chunk.grippers[-1]),
                "units": "robot gripper command units; polarity and contact are not inferred",
            },
            "task_success": "unverified",
            "meaning": (
                "Robot tracking only: translation_error_m is distance from observed EEF to the "
                "previous issued EEF endpoint, NOT distance to an object or contact surface. "
                "Observed displacement measures robot motion, NOT object motion or task success. "
                "Delta vectors are XYZ metres in frame: commanded=issued endpoint minus previous "
                "observation, observed=current minus previous observation, remaining=issued endpoint "
                "minus current observation. Along-track is signed projection onto the commanded "
                "endpoint direction; its fraction is not clipped (negative=reverse, >1=overshoot). "
                "Cross-track is distance to that infinite line, not to the executed trajectory. "
                "Projection fields are null for commanded displacement <=0.0001 m."
            ),
        }
        targets: dict[str, list[float]] | None = None
        if chunk.kind == "EEF_ABS":
            if "generated_endpoint" in chunk.metadata:
                result["generated_endpoint"] = deepcopy(chunk.metadata["generated_endpoint"])
            targets = {"left": chunk.left_arm[-1], "right": chunk.right_arm[-1]}
        else:
            result["joint_error_rad"] = [
                a - b for a, b in zip(obs.joints, chunk.left_arm[-1] + chunk.right_arm[-1])
            ]
            path = chunk.metadata.get("cartesian_path")
            if path is not None:
                # The compiler's FK of the issued joint rows, not an estimate from joint residuals.
                result["cartesian_target_source"] = "issued_joint_fk"
                targets = {arm: path[arm][chunk.steps - 1] for arm in ("left", "right")}
                result["generated_endpoint"] = {arm + "_arm": path[arm][-1] for arm in ("left", "right")}
                if "generated_endpoint" in chunk.metadata:
                    result["generated_endpoint"]["grippers"] = chunk.metadata["generated_endpoint"][
                        "grippers"
                    ]
        if targets is not None:
            for arm, target in targets.items():
                try:
                    before, actual = previous.eef(arm, chunk.frame), obs.eef(arm, chunk.frame)
                    result[arm] = {
                        "previous_pose": before,
                        "observed_pose": actual,
                        "issued_pose": target,
                        **_translation_feedback(before, actual, target),
                        "rotation_error_rad": quaternion_angle(actual[3:], target[3:]),
                    }
                except ContractError as exc:
                    result[arm] = {"unavailable": str(exc)}
        # Measured Cartesian motion is useful even when the issued command was joint-space.
        # Without compiler FK, do not fabricate an EEF target or call this object progress.
        result["measured_eef_motion"] = {}
        for arm in ("left", "right"):
            try:
                before, actual = previous.eef(arm, chunk.frame), obs.eef(arm, chunk.frame)
                result["measured_eef_motion"][arm] = {
                    "previous_pose": before,
                    "observed_pose": actual,
                    "translation_delta_m": [a - b for a, b in zip(actual[:3], before[:3])],
                    "rotation_delta_rad": quaternion_angle(before[3:], actual[3:]),
                }
            except ContractError as exc:
                result["measured_eef_motion"][arm] = {"unavailable": str(exc)}
        if chunk.body:
            result["body_error_rad"] = [a - b for a, b in zip(obs.body, chunk.body[-1])]
        summary = [
            f"Issued {chunk.steps}/{result['generated_steps']} generated steps; actual playback is unverified."
        ]
        if chunk.kind != "EEF_ABS":
            summary.append(f"Joint residuals to issued target (rad): {result['joint_error_rad']}.")
        if targets is not None:
            for arm in ("left", "right"):
                if "translation_error_m" in result[arm]:
                    summary.append(
                        f"{arm} EEF residual to issued endpoint: "
                        f"{result[arm]['translation_error_m']:.6g} m, "
                        f"{result[arm]['rotation_error_rad']:.6g} rad."
                    )
                    if result[arm]["along_track_fraction"] is not None:
                        summary.append(
                            f"{arm} signed endpoint-direction progress: "
                            f"{result[arm]['along_track_fraction']:.6g} of commanded displacement; "
                            f"cross-track offset: {result[arm]['cross_track_error_m']:.6g} m."
                        )
        summary.append(
            f"Grippers observed {result['grippers']['observed']}, "
            f"issued target {result['grippers']['issued_target']}. "
            "Residuals do not identify contact, object alignment, or a successful grasp. "
            "Choose whether to continue the target, inspect, or revise it from fresh observations."
        )
        result["summary"] = " ".join(summary)
        return result

    def step(self, obs: Observation) -> ActionChunk:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if obs.robot_type != self.spec.robot_type:
            raise ContractError("observation robot_type does not match runtime")
        vector(obs.joints, len(self.spec.left_joints) + len(self.spec.right_joints), "observed joints")
        vector(obs.grippers, 2, "observed grippers")
        vector(obs.body, len(self.spec.body_joints), "observed body")
        # One runtime per socket; reset on task/episode or user instruction changes.
        key = (*obs.episode_key, obs.instruction)
        if key != self.episode_key:
            self.reset()
            self.episode_key = key
        # Only the actually issued prefix supplies command continuity. Never use
        # a discarded endpoint, rejected decision, or caller-provided reference.
        obs = replace(
            obs,
            issued_gripper_targets=(
                list(self._last_issued.grippers[-1]) if self._last_issued is not None else None
            ),
            issued_body_targets=(
                list(self._last_issued.body[-1])
                if self._last_issued is not None and self._last_issued.body
                else None
            ),
        )
        self.sequence += 1
        feedback = self._tracking_feedback(obs)
        if feedback:
            for event in reversed(self.memory):
                if event.get("status") == "issued" and event.get("sequence") == self.sequence - 1:
                    event["observed_effect"] = {
                        "observation_sequence": self.sequence,
                        "frame": feedback["frame"],
                        "measured_eef_motion": deepcopy(feedback["measured_eef_motion"]),
                        "tracking": {
                            arm: {
                                k: deepcopy(v) for k, v in feedback[arm].items() if k in _EFFECT_TRACKING_KEYS
                            }
                            for arm in ("left", "right")
                            if "issued_pose" in feedback.get(arm, {})
                        },
                        "grippers": deepcopy(feedback["grippers"]),
                        "meaning": "Measured robot change after this issued action; object motion, playback and task success unverified.",
                    }
                    break
            self.event_sink(
                {
                    "episode": self.episode_key,
                    "sequence": self.sequence,
                    "status": "observed_tracking",
                    **feedback,
                }
            )
        if obs.done:
            chunk = hold_chunk(obs, self.spec)
            self.validator.validate(chunk, obs)
            chunk.metadata = {"status": "episode_done", "sequence": self.sequence}
            self._record({"status": "episode_done"})
            self.reset()
            return chunk
        continuation = self._take_continuation(feedback, obs)
        # Take ownership before any inference. An exception cannot leave a stale grant
        # queued for a later request. Only successful validation/issuance re-arms it.
        policy_run, self._policy_run = self._policy_run, None
        if policy_run is not None and policy_run.exhausted:
            self._review_policy(policy_run, "budget_exhausted")
            policy_run = None
        start = time.monotonic()
        proposal = None
        proposal_error = None
        reviewing = (
            self.config.proposal_review
            and "policy" in self.router.describe()["allowed_modes"]
            and policy_run is None
        )
        if reviewing:
            try:
                source = Decision(
                    "policy",
                    obs.instruction,
                    "Configured observation-bound candidate source",
                    backend=self.config.proposal_backend,
                    policy_capability=self.config.proposal_capability,
                )
                with self._phase("proposal_generation", 0):
                    candidate = self.router.generate(obs, source)
                    self.validator.validate(candidate, obs)
                    assert self.kinematics is not None
                    preview = self.kinematics.preview(obs, candidate, self.router.step_limit("policy"))
                proposal = PolicyProposal(
                    obs,
                    self.sequence,
                    source,
                    candidate,
                    preview,
                    self.router.step_limit("policy"),
                    self.config.proposal_max_age_s,
                )
                self.event_sink(
                    {"sequence": self.sequence, "status": "proposal_prepared", **proposal.describe()}
                )
            except ContractError as exc:
                proposal_error = str(exc)
                self._record({"status": "proposal_rejected", "error": proposal_error})
        for attempt in range(self.config.max_replans + 1):
            decision = None
            try:
                context = {
                    "robot": self.spec.describe(),
                    "execution": self.router.describe(),
                    "recent_events": list(self.memory),
                    "sequence": self.sequence,
                    "planning_attempt": attempt,
                    "tracking_feedback": feedback,
                    "gripper_commands": gripper_start(obs, self.spec)[0],
                    "body_commands": body_start(obs, self.spec),
                    "gripper_commands_source": (
                        "last_issued_target; playback unverified"
                        if obs.issued_gripper_targets is not None
                        else "bounded measured position at episode start"
                    ),
                    "observed_grippers_in_command_units": obs.command_grippers(self.spec),
                    "policy_review": deepcopy(self._policy_review),
                    "proposal": proposal.describe() if proposal is not None else None,
                    "proposal_error": proposal_error,
                    "previous_assessment": deepcopy(self._assessment),
                }
                with self._phase("planner", attempt):
                    if attempt == 0 and policy_run is not None:
                        decision = deepcopy(policy_run.decision)
                        self._record(
                            {"status": "policy_continued", **policy_run.summary("fresh_observation")}
                        )
                    elif attempt == 0 and continuation is not None:
                        decision = continuation
                        self._record(
                            {
                                "status": "motion_continued",
                                "continuation": self._continuations,
                                "reason": "observed tracking within configured tolerances",
                            }
                        )
                    else:
                        self._continuations = 0
                        decision = self.planner.decide(obs, context)
                    # Validate custom planner decisions against the public contract.
                    decision = Decision.from_dict(asdict(decision))
                    self.router.check(decision)
                    if (
                        decision.assessment is not None
                        and (
                            decision.assessment.observation_request is not None
                            or decision.assessment.motion_alternative is not None
                        )
                        and not (reviewing and proposal is not None)
                    ):
                        raise ContractError("explicit motion request requires a current reviewed candidate")
                    if reviewing and proposal is not None:
                        proposal.check_current(obs, self.sequence)
                        if decision.policy_chunks > self._proposal_limit:
                            raise ContractError("policy_chunks exceeds proposal capability budget")
                        if decision.assessment is None:
                            raise ContractError("candidate review requires an assessment")
                        if (self._last_issued is None) != (
                            decision.assessment.execution_status == "not_started"
                        ):
                            raise ContractError("not_started is only valid before the first issued action")
                        if (
                            decision.mode == "motion"
                            and decision.status == "continue"
                            and not decision.assessment.permits_correction
                        ):
                            raise ContractError(
                                "correction requires observed failure, misaligned intent, or an explicit "
                                "observation_request or motion_alternative; uncertainty alone is insufficient"
                            )
                    elif decision.proposal_id is not None and policy_run is None:
                        raise ContractError("no current proposal to accept")
                    if (
                        reviewing
                        and proposal is None
                        and decision.mode == "policy"
                        and decision.status == "continue"
                    ):
                        raise ContractError(f"candidate unavailable: {proposal_error}")
                    if decision.mode == "policy" and decision.status == "continue" and policy_run is None:
                        policy_run = PolicyRun(deepcopy(decision), self.sequence)
                with self._phase("action_generation", attempt):
                    if decision.status in ("complete", "abandon"):
                        chunk = hold_chunk(obs, self.spec, self.config.complete_hold_steps or 1)
                        chunk.metadata["status"] = (
                            "planner_complete_unverified"
                            if decision.status == "complete"
                            else "planner_abandoned"
                        )
                    else:
                        if reviewing and proposal is not None and decision.mode == "policy":
                            chunk = proposal.consume(obs, self.sequence, decision)
                        else:
                            chunk = self.router.generate(obs, decision)
                    if self.spec.body_joints and not chunk.body:
                        chunk.body = [body_start(obs, self.spec) for _ in range(chunk.steps)]
                with self._phase("action_validation", attempt):
                    # Full chunk before truncation.
                    (self._hold_validator if decision.status != "continue" else self.validator).validate(
                        chunk, obs
                    )
                    if decision.assessment is not None and (
                        decision.assessment.observation_request is not None
                        or decision.assessment.motion_alternative is not None
                    ):
                        request = decision.assessment.observation_request
                        bounds = request.command_bounds if request is not None else None
                        if bounds is not None:
                            if chunk.steps > self.router.step_limit("motion"):
                                raise ContractError(
                                    "bounded observation_request must fit the entire executable prefix"
                                )
                            if chunk.kind == "EEF_ABS":
                                envelope = wrist_envelope(obs, chunk.frame, chunk.left_arm, chunk.right_arm)
                            elif chunk.kind == "JOINT_ABS" and self.kinematics is not None:
                                # Identify this diagnostic preview separately from the reviewed policy.
                                chunk.metadata["command_bounds"] = {"limits": asdict(bounds)}
                                envelope = self.kinematics.preview(obs, chunk, chunk.steps).get(
                                    "command_envelope", {}
                                )
                            else:
                                raise ContractError("command_bounds require all-command geometry")
                            validate_command_bounds(bounds, obs, chunk, self.spec, envelope)
                            chunk.metadata["command_bounds"] = {
                                "limits": asdict(bounds),
                                "command_envelope": envelope,
                                "meaning": "Command bounds only; actual motion, contact and success need fresh observation.",
                            }
                        commands = gripper_start(obs, self.spec)[0]
                        if bounds is None and any(
                            abs(value - command) > 1e-9
                            for row in chunk.grippers
                            for value, command in zip(row, commands)
                        ):
                            raise ContractError("explicit motion request must preserve both gripper commands")
                        if decision.assessment.motion_alternative is not None:
                            if proposal is None:
                                raise ContractError(
                                    "motion_alternative requires a current reviewed candidate"
                                )
                            proposal.check_current(obs, self.sequence)
                            if chunk.steps > self.router.step_limit("motion"):
                                raise ContractError(
                                    "motion_alternative must fit the entire executable prefix"
                                )
                            if chunk.steps * chunk.dt_s >= proposal.executable_duration_s - 1e-9:
                                raise ContractError(
                                    "motion_alternative must be shorter than the policy prefix"
                                )
                            chunk.metadata["motion_alternative"] = {
                                "proposal_id": proposal.id,
                                "policy_prefix_duration_s": proposal.executable_duration_s,
                                "motion_duration_s": chunk.steps * chunk.dt_s,
                                "meaning": "Command-duration comparison only; physical effect and preserved conditions remain model assessments.",
                            }
                generated_steps = chunk.steps
                planned_endpoint = {
                    "left_arm": chunk.left_arm[-1],
                    "right_arm": chunk.right_arm[-1],
                    "grippers": chunk.grippers[-1],
                    "body": chunk.body[-1] if chunk.body else [],
                }
                self.event_sink(
                    {
                        "episode": self.episode_key,
                        "sequence": self.sequence,
                        "status": "planned",
                        "decision": asdict(decision),
                        "generated_steps": generated_steps,
                        "planned_endpoint": planned_endpoint,
                    }
                )
                step_limit = self.router.step_limit(decision.mode)
                if decision.status != "continue" and self.config.complete_hold_steps is not None:
                    # The whole final hold is issued so the simulator can finish its checks.
                    step_limit = max(step_limit, self.config.complete_hold_steps)
                if self.config.body_execute_steps is not None and any(
                    row != body_start(obs, self.spec) for row in chunk.body
                ):
                    # A commanded body motion (e.g. a waist turn) is issued as one longer prefix.
                    step_limit = max(step_limit, self.config.body_execute_steps)
                chunk = chunk.prefix(step_limit)
                chunk.metadata.update(
                    {
                        "sequence": self.sequence,
                        "mode": decision.mode,
                        "generated_steps": generated_steps,
                        "issued_steps": chunk.steps,
                        "generated_endpoint": deepcopy(planned_endpoint),
                        "discarded_steps": generated_steps - chunk.steps,
                        "planner_status": decision.status,
                    }
                )
                # Target-hold tail is not an unfinished semantic motion to recompile.
                motion_steps = chunk.metadata.get("motion_steps", generated_steps)
                self._remember_continuation(decision, max(0, motion_steps - chunk.steps))
                self._last_issued = chunk
                self._last_observed = obs
                if decision.assessment is not None:
                    self._assessment = asdict(decision.assessment)
                    chunk.metadata["assessment"] = deepcopy(self._assessment)
                if reviewing and proposal is not None:
                    chunk.metadata["reviewed_proposal_id"] = proposal.id
                if policy_run is not None:
                    # Resolve an omitted default once, preserving backend identity throughout the grant.
                    policy_run.decision.backend = chunk.metadata.get("backend", decision.backend)
                    # The grant authorizes fresh inference, never reuse of the reviewed proposal id.
                    policy_run.decision.proposal_id = None
                    policy_run.decision.assessment = None
                    policy_run.issued_chunks += 1
                    policy_run.issued_steps += chunk.steps
                    policy_run.issued_duration_s += chunk.steps * chunk.dt_s
                    self._policy_run = policy_run
                    self._policy_review = None
                    chunk.metadata["policy_run"] = policy_run.summary("issued")
                self._record(
                    {
                        "status": "issued",
                        "discarded_steps": generated_steps - chunk.steps,
                        "frame": chunk.frame,
                        "kind": chunk.kind,
                        "issued_endpoint": {
                            "left_arm": chunk.left_arm[-1],
                            "right_arm": chunk.right_arm[-1],
                            "grippers": chunk.grippers[-1],
                            "body": chunk.body[-1] if chunk.body else [],
                        },
                        "mode": decision.mode,
                        "backend": chunk.metadata.get("backend"),
                        "instruction": decision.instruction,
                        "reason": decision.reason,
                        "success_condition": decision.success_condition,
                        "planner_status": decision.status,
                        "generated_steps": generated_steps,
                        "issued_steps": chunk.steps,
                        "issued_duration_s": round(chunk.steps * self.spec.dt_s, 6),
                        "attempt": attempt,
                        "latency_s": round(time.monotonic() - start, 3),
                        "policy_run": chunk.metadata.get("policy_run"),
                        "reviewed_proposal_id": chunk.metadata.get("reviewed_proposal_id"),
                        "assessment": chunk.metadata.get("assessment"),
                    }
                )
                return chunk
            except ContractError as exc:
                if policy_run is not None:
                    self._review_policy(policy_run, f"rejected: {exc}")
                    policy_run = None
                    self._policy_run = None
                if (
                    self.config.proposal_review
                    and not reviewing
                    and "policy" in self.router.describe()["allowed_modes"]
                ):
                    # A rejected continuation revokes the grant. No unreviewed replacement
                    # policy action can be issued in the subsequent planner retry.
                    reviewing = True
                    proposal_error = str(exc)
                self._record(
                    {
                        "status": "rejected",
                        "mode": decision.mode if decision else None,
                        "error": str(exc),
                        "attempt": attempt,
                        "latency_s": round(time.monotonic() - start, 3),
                    }
                )
                if isinstance(exc, ExpiredProposalError) or attempt == self.config.max_replans:
                    if isinstance(exc, LocalIKConvergenceError) and self.config.retry_ik_exhaustion:
                        chunk = hold_chunk(obs, self.spec, steps=self.config.execute_steps)
                        self.validator.validate(chunk, obs)
                        chunk.metadata.update(
                            {
                                "status": "ik_replan_required",
                                "task_success": "unverified",
                                "error": str(exc),
                                "sequence": self.sequence,
                            }
                        )
                        self._record({"status": "ik_replan_required", "error": str(exc)})
                        return chunk
                    if self.config.terminal_on_planning_exhaustion:
                        chunk = hold_chunk(obs, self.spec)
                        self.validator.validate(chunk, obs)
                        chunk.metadata.update(
                            {
                                "status": "planning_exhausted",
                                "episode_terminal": True,
                                "task_success": "unverified",
                                "error": str(exc),
                                "sequence": self.sequence,
                            }
                        )
                        self._record({"status": "planning_exhausted", "error": str(exc)})
                        self.reset()
                        return chunk
                    raise PlanningExhaustedError(f"replanning budget exhausted: {exc}") from exc
            except Exception as exc:
                if policy_run is not None:
                    self._review_policy(policy_run, f"backend_error: {type(exc).__name__}")
                    self._policy_run = None
                raise
        raise AssertionError("unreachable")
