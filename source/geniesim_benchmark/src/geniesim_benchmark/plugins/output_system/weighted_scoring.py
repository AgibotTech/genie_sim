# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Task weight profiles and native checkpoint validation; no simulator imports."""

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


def canonical_score_layout(name: str) -> str:
    """Read former layout identifiers without duplicating their profiles."""
    aliases = {
        "wuji_ipc_cloth_hanger": "cloth_hanger",
        "wuji_ipc_fold_cloth": "fold_cloth",
        "wuji_ipc_drawstring_bag": "drawstring_bag",
    }
    return aliases.get(name, name)


@dataclass(frozen=True)
class WeightProfile:
    version: str
    task: str
    instruction_mode: str
    steps: tuple[str, ...]
    weights: tuple[float, ...]
    rationale: str
    process_points: float = 100.0
    completion_bonus_points: float = 25.0
    completion_rule: str = "all_steps_and_native_e2e"

    def __post_init__(self) -> None:
        if not self.steps or len(self.steps) != len(self.weights):
            raise ValueError(f"Weight/checker count mismatch: {self.task}")
        if any(
            isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0
            for w in self.weights
        ):
            raise ValueError(f"Weights must be positive finite numbers: {self.task}")
        if not math.isclose(sum(self.weights), 100.0, abs_tol=1e-8):
            raise ValueError(f"Weights must sum to 100: {self.task}")
        if (
            type(self.process_points) not in (int, float)
            or type(self.completion_bonus_points) not in (int, float)
            or (self.process_points, self.completion_bonus_points) != (100, 25)
        ):
            raise ValueError("Weighted scoring requires 100 process points and 25 completion points")
        if self.completion_rule != "all_steps_and_native_e2e":
            raise ValueError(f"Unsupported completion rule: {self.completion_rule}")

    @property
    def total_points(self) -> float:
        return self.process_points + self.completion_bonus_points

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task": self.task,
            "instruction_mode": self.instruction_mode,
            "steps": list(self.steps),
            "weights": list(self.weights),
            "rationale": self.rationale,
            "process_points": self.process_points,
            "completion_bonus_points": self.completion_bonus_points,
            "completion_rule": self.completion_rule,
        }


@lru_cache(maxsize=8)
def load_weight_profiles(weights_path: Path | None = None) -> dict[str, dict[str, WeightProfile]]:
    path = weights_path if weights_path is not None else Path(__file__).with_name("task_score_weights.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        mode: {
            task: WeightProfile(
                data["version"],
                task,
                mode,
                tuple(spec["steps"]),
                tuple(spec["weights"]),
                spec["rationale"],
                data["process_points"],
                data["completion_bonus_points"],
                data["completion_rule"],
            )
            for task, spec in data[mode].items()
        }
        for mode in ("full", "subtask")
    }


def get_weight_profile(
    sub_task_name: str,
    steps: Sequence[str] | None,
    instruction_mode: str = "full",
    *,
    weights_path: Path | None = None,
) -> WeightProfile | None:
    """Validate checker layouts against the active or explicitly supplied weights file.

    Offline callers can pass steps=None only with weights_path, using that
    file's authored layout rather than relabelling native checkers.
    """
    if instruction_mode not in ("full", "subtask"):
        raise ValueError(f"Unsupported instruction mode: {instruction_mode}")
    if steps is None and weights_path is None:
        raise ValueError("An explicit weights file is required to select a historical layout")
    profiles = load_weight_profiles(weights_path)
    key = sub_task_name
    if key not in profiles["full"] and key not in profiles["subtask"]:
        key = re.sub(r"_\d+$", "", key)
        key = canonical_score_layout(key)
    mode = "subtask" if instruction_mode == "subtask" and key in profiles["subtask"] else "full"
    profile = profiles[mode].get(key)
    if profile is not None and steps is not None and tuple(steps) != profile.steps:
        raise ValueError(f"Checker layout differs from {profile.version} weights: {key} ({mode})")
    return profile


def native_stage_scores(
    rows: Sequence[Mapping[str, Any]],
    profile: WeightProfile,
    native_e2e: int | float | None,
) -> tuple[list[float], bool]:
    """Validate raw checkpoints without calculating an unbudgeted total.

    Repeated checker names are separate positions. Independent later successes
    retain process credit. Missing steps and native E2E alone cannot earn the
    completion bonus; every configured checkpoint must have score exactly one.
    """
    values = [0.0] * len(profile.steps)
    seen: set[int] = set()
    for row in rows:
        index = row["step"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(values):
            raise ValueError(f"Invalid step index for {profile.task}: {index}")
        if index in seen or row["name"] != profile.steps[index]:
            raise ValueError(f"Duplicate or mismatched checker at step {index}: {profile.task}")
        score = row["score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError(f"Invalid checker score at step {index}: {score}")
        seen.add(index)
        values[index] = float(score)
    if native_e2e is not None and native_e2e not in (0, 1):
        raise ValueError(f"Native E2E must be 0, 1 or absent: {native_e2e}")
    complete = native_e2e == 1 and len(seen) == len(values) and all(s == 1.0 for s in values)
    return values, complete
