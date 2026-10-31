# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Select ranked single-target alternatives without changing the runtime contract."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

from agent2act.adapters.telemetry import record_artifact
from agent2act.domain.errors import LocalIKConvergenceError
from agent2act.domain.models import ActionChunk, ContractError, Decision, Observation, RobotSpec
from agent2act.domain.ports import ActionExecutor, Planner
from agent2act.domain.validation import ActionValidator

RANKED_MOTION_INSTRUCTIONS = """
Ranked-target output contract overrides the ordinary motion.waypoints format:
For status=continue, return motion={"frame":<configured frame>,"candidates":[<five waypoint objects>]}.
Generate exactly FIVE independent alternative targets for the SAME next action in ONE response,
ordered from best to worst. Each candidate is one waypoint, not a sequential path. Preserve the
intended task effect across alternatives; vary approach/orientation/position where appropriate.
Use the ordinary waypoint fields and coordinate reference (including gripper-center when enabled).
All five candidates are checked sequentially with local endpoint IK and full command validation
from the SAME current observation. Only the first passing candidate is issued. If all fail, use
reported reasons to generate a new ranked batch within the existing replanning budget.
Checking never moves the robot. IK/command validity does not prove collision clearance, contact,
playback or task success. For complete/abandon use motion=null as usual.
"""


def ranked_motion_instructions(count: int) -> str:
    if type(count) is not int or count not in (5, 10):
        raise ContractError("ranked motion candidate count must be 5 or 10")
    if count == 5:
        return RANKED_MOTION_INSTRUCTIONS
    return (
        RANKED_MOTION_INSTRUCTIONS.replace("FIVE", "TEN").replace("five", "ten")
        + """
Use the additional alternatives to explore distinct approach positions and orientations.
After joint-limit rejections, include less wrist tilt or a different supported waist posture
instead of ten small position perturbations with the same rejected orientation. Preserve the
same intended task effect. Identical alternatives are appropriate only for a confirmation hold.
"""
    )


class RankedMotionExecutor:
    """Own the IK executor and reuse an observation-bound checked chunk exactly once."""

    mode = "motion"

    def __init__(self, executor: ActionExecutor, spec: RobotSpec, candidate_count: int = 5) -> None:
        with ExitStack() as rollback:
            rollback.callback(executor.close)
            ranked_motion_instructions(candidate_count)
            if not executor.describe().get("compiler", {}).get("endpoint_ik"):
                raise ContractError("ranked_motion_candidates requires an executor with endpoint IK")
            self.executor = executor
            self.candidate_count = candidate_count
            self.validator = ActionValidator(spec)
            self.pending: tuple[Observation, Decision, ActionChunk] | None = None
            self.closed = False
            rollback.pop_all()

    def describe(self) -> dict[str, Any]:
        return {**self.executor.describe(), "ranked_motion_candidates": self.candidate_count}

    def select(self, observation: Observation, decision: Decision) -> Decision:
        self.pending = None
        if self.closed:
            raise RuntimeError("ranked motion executor is closed")
        motion = decision.motion
        if not isinstance(motion, dict) or set(motion) != {"frame", "candidates"}:
            raise ContractError("ranked motion requires exactly frame and candidates")
        candidates = motion["candidates"]
        count_word = "five" if self.candidate_count == 5 else "ten"
        if not isinstance(candidates, list) or len(candidates) != self.candidate_count:
            raise ContractError(
                f"ranked motion requires exactly {count_word} candidates ordered best to worst"
            )
        checks: list[dict[str, Any]] = []
        selected: tuple[Decision, ActionChunk] | None = None
        selected_rank: int | None = None
        all_local_ik_failures = True
        for rank, waypoint in enumerate(candidates, 1):
            candidate = replace(
                decision, motion={"frame": motion["frame"], "waypoints": [deepcopy(waypoint)]}
            )
            try:
                chunk = self.executor.generate(deepcopy(observation), candidate)
                self.validator.validate(chunk, observation)
            except ContractError as exc:
                all_local_ik_failures = all_local_ik_failures and isinstance(exc, LocalIKConvergenceError)
                checks.append(
                    {"rank": rank, "status": "rejected", "reason": str(exc), "error_type": type(exc).__name__}
                )
            else:
                checks.append({"rank": rank, "status": "passed"})
                if selected is None:
                    selected, selected_rank = (candidate, chunk), rank
        report = {
            "candidate_count": self.candidate_count,
            "selected_rank": selected_rank,
            "checks": checks,
            "candidates": deepcopy(candidates),
        }
        record_artifact("ranked-motion-checks", report)
        if selected is None:
            reasons = "; ".join(f"candidate {row['rank']}: {row['reason']}" for row in checks)
            error = LocalIKConvergenceError if all_local_ik_failures else ContractError
            raise error(f"all {count_word} ranked motion candidates rejected: {reasons}")
        candidate, chunk = selected
        chunk.metadata["ranked_motion_selection"] = report
        self.pending = (deepcopy(observation), deepcopy(candidate), deepcopy(chunk))
        return candidate

    def generate(self, observation: Observation, decision: Decision) -> ActionChunk:
        if self.closed:
            raise RuntimeError("ranked motion executor is closed")
        pending, self.pending = self.pending, None
        if pending is not None and pending[0] == observation and pending[1] == decision:
            return deepcopy(pending[2])
        # Continuations start from fresh measurements and run IK again.
        return self.executor.generate(observation, decision)

    def reset(self) -> None:
        self.pending = None
        self.executor.reset()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.pending = None
            self.executor.close()


class RankedMotionPlanner:
    def __init__(self, planner: Planner, motion: RankedMotionExecutor) -> None:
        self.planner, self.motion = planner, motion
        self.closed = False

    def decide(self, observation: Observation, context: dict[str, Any]) -> Decision:
        if self.closed:
            raise RuntimeError("ranked motion planner is closed")
        self.motion.pending = None
        decision = Decision.from_dict(asdict(self.planner.decide(observation, context)))
        if decision.status != "continue":
            return decision
        if decision.mode != "motion":
            raise ContractError("ranked motion candidates requires motion decisions")
        return self.motion.select(observation, decision)

    def reset(self) -> None:
        self.motion.pending = None
        self.planner.reset()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.motion.pending = None
            self.planner.close()
