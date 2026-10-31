# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Opt-in deterministic motion-response experiment, never a task policy."""

import logging
import math
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from agent2act.adapters.telemetry import record_artifact
from agent2act.domain.models import FRAMES, ContractError, Decision, Observation, number, vector

logger = logging.getLogger(__name__)


class MotionProbePlanner:
    """Probe fixed targets around one captured reference; sample immediate and held response."""

    def __init__(
        self,
        frame: str,
        trials: list[dict[str, Any]],
        setup_left_mm: list[float] | None = None,
        setup_right_mm: list[float] | None = None,
        return_duration_s: float = 1.0,
        settle_s: float = 0.5,
    ) -> None:
        if frame not in FRAMES:
            raise ContractError("invalid probe frame")
        if not isinstance(trials, list) or not 1 <= len(trials) <= 100:
            raise ContractError("probe needs 1..100 trials")
        self.frame = frame
        self.return_duration_s = self._duration(return_duration_s)
        self.settle_s = self._duration(settle_s)
        self.setup = [
            self._delta(setup_left_mm if setup_left_mm is not None else [0, 0, 0]),
            self._delta(setup_right_mm if setup_right_mm is not None else [0, 0, 0]),
        ]
        self.trials: list[dict[str, Any]] = []
        for trial in trials:
            if not isinstance(trial, dict) or set(trial) - {
                "name",
                "arm",
                "translation_mm",
                "gripper_delta",
                "duration_s",
            }:
                raise ContractError("unknown probe trial fields")
            if not isinstance(trial.get("name"), str) or not trial["name"] or len(trial["name"]) > 100:
                raise ContractError("probe trial requires a bounded name")
            if trial.get("arm") not in ("left", "right"):
                raise ContractError("probe arm must be left or right")
            if ("translation_mm" in trial) == ("gripper_delta" in trial):
                raise ContractError("probe trial requires translation_mm OR gripper_delta")
            item = deepcopy(trial)
            item["duration_s"] = self._duration(trial.get("duration_s"))
            if "translation_mm" in trial:
                item["translation_mm"] = self._delta(trial["translation_mm"])
                if sum(v != 0 for v in item["translation_mm"]) != 1:
                    raise ContractError("probe translation must change exactly one frame axis")
            else:
                item["gripper_delta"] = number(trial["gripper_delta"], "gripper_delta")
                if not 0 < abs(item["gripper_delta"]) <= 0.2:
                    raise ContractError(
                        "probe gripper delta must be nonzero with magnitude <=0.2 command units"
                    )
            if any(existing["name"] == item["name"] for existing in self.trials):
                raise ContractError("probe trial names must be unique")
            self.trials.append(item)
        self._closed = False
        self.reset()

    @staticmethod
    def _duration(value: Any) -> float:
        result = number(value, "probe duration")
        if not 0 < result <= 5:
            raise ContractError("probe duration must be in (0,5] seconds")
        return result

    @staticmethod
    def _delta(value: Any) -> list[float]:
        result = vector(value, 3, "probe translation_mm")
        if math.dist(result, [0, 0, 0]) > 50:
            raise ContractError("probe translation exceeds 50 mm")
        return result

    def reset(self) -> None:
        self._reference: dict[str, Any] | None = None
        self._program: list[dict[str, Any]] = []
        self._index = 0
        self._pending: dict[str, Any] | None = None
        self._cached: tuple[int, Decision] | None = None
        self._finished = False

    @staticmethod
    def _emit(event: dict[str, Any]) -> None:
        record_artifact("motion-probe", event)
        logger.info(event["event"], extra={"event": event})

    def _initialize(self, observation: Observation, context: dict[str, Any]) -> None:
        reference = {
            "left_pose": observation.eef("left", self.frame),
            "right_pose": observation.eef("right", self.frame),
            "grippers": vector(context["gripper_commands"], 2, "probe grippers"),
        }
        for arm, delta in zip(("left", "right"), self.setup):
            reference[f"{arm}_pose"][:3] = [reference[f"{arm}_pose"][i] + delta[i] / 1000 for i in range(3)]
        bounds = context["robot"]["gripper_bounds"]
        program = []

        def add(name: str, phase: str, target: dict[str, Any], duration: float) -> None:
            program.append(
                {"trial": name, "phase": phase, "target": deepcopy(target), "duration_s": duration}
            )

        add("setup", "setup", reference, self.return_duration_s)
        add("setup", "baseline_hold", reference, self.settle_s)
        for trial in self.trials:
            target = deepcopy(reference)
            arm = trial["arm"]
            if "translation_mm" in trial:
                target[f"{arm}_pose"][:3] = [
                    reference[f"{arm}_pose"][i] + trial["translation_mm"][i] / 1000 for i in range(3)
                ]
            else:
                i = 0 if arm == "left" else 1
                target["grippers"][i] += trial["gripper_delta"]
                if not bounds[i][0] <= target["grippers"][i] <= bounds[i][1]:
                    raise ContractError(f"probe {trial['name']} gripper target outside configured bounds")
            add(trial["name"], "pulse", target, trial["duration_s"])
            add(trial["name"], "target_hold", target, self.settle_s)
            add(trial["name"], "return", reference, self.return_duration_s)
            add(trial["name"], "baseline_hold", reference, self.settle_s)
        self._reference, self._program = reference, program
        self._emit(
            {
                "event": "motion_probe_started",
                "frame": self.frame,
                "reference": reference,
                "trials": self.trials,
                "meaning": "fixed command-response experiment; clearance, contact and task success unverified",
            }
        )

    def decide(self, observation: Observation, context: dict[str, Any]) -> Decision:
        if self._closed:
            raise RuntimeError("probe planner is closed")
        execution = context["execution"]
        if (
            execution.get("motion_frame") != self.frame
            or execution.get("mode") != "motion"
            or execution.get("max_motion_continuations", 0) != 0
        ):
            raise ContractError(
                "probe requires fixed motion mode, matching motion_frame and disabled continuation"
            )
        sequence = context["sequence"]
        if self._cached is not None and self._cached[0] == sequence:
            return Decision.from_dict(asdict(self._cached[1]))
        if self._reference is None:
            self._initialize(observation, context)
        if self._pending is not None:
            feedback = context.get("tracking_feedback", {})
            if (
                feedback.get("previous_sequence") != self._pending["sequence"]
                or feedback.get("discarded_steps") != 0
            ):
                raise ContractError(
                    "probe requires untruncated matching feedback; inspect execute_steps budget"
                )
            result = {
                **self._pending,
                "event": "motion_probe_sample",
                "feedback": deepcopy(feedback),
                "observed": {
                    "left_pose": observation.eef("left", self.frame),
                    "right_pose": observation.eef("right", self.frame),
                    "grippers": list(context["observed_grippers_in_command_units"]),
                },
            }
            self._emit(result)
            self._pending = None
        step: dict[str, Any]
        if self._index >= len(self._program):
            if not self._finished:
                self._emit(
                    {
                        "event": "motion_probe_finished",
                        "sequence": sequence,
                        "meaning": "last baseline return observed; experiment ended, task success unverified",
                    }
                )
                self._finished = True
            assert self._reference is not None
            step = {
                "trial": "finished",
                "phase": "idle",
                "target": self._reference,
                "duration_s": self.settle_s,
            }
        else:
            step = self._program[self._index]
            self._index += 1
        decision = Decision(
            mode="motion",
            instruction=f"probe {step['trial']} / {step['phase']}",
            reason="Fixed diagnostic target; no LLM, no object localization or task-success inference.",
            motion={
                "frame": self.frame,
                "waypoints": [{**deepcopy(step["target"]), "duration_s": step["duration_s"]}],
            },
            success_condition="Compare the next observation with the issued target; task success is not assessed.",
        )
        self._cached = (sequence, decision)
        if not self._finished:
            self._pending = {
                **deepcopy(step),
                "sequence": sequence,
                "frame": self.frame,
                "dt_s": context["robot"]["dt_s"],
                "before": {
                    "left_pose": observation.eef("left", self.frame),
                    "right_pose": observation.eef("right", self.frame),
                    "grippers": list(context["observed_grippers_in_command_units"]),
                },
            }
        return Decision.from_dict(asdict(decision))

    def close(self) -> None:
        self._closed = True
