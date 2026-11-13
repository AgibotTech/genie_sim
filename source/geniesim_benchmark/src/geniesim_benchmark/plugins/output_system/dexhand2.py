"""Common DexHand 2 reporting after native episode completion."""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from .policy_budget import episode_policy_budget_breakdown, write_policy_budget_scores
from .weighted_scoring import get_weight_profile

WEIGHTS_PATH = Path(__file__).with_name("dexhand2_score_weights.json")
RUBIK_TASK = "rubik_cube_solve_2turn"
RUBIK_STAGES = ("PickUp", "Turn", "Solve")
WOK_TASK = "wok_flip"
WOK_STAGES = ("stable_lift", "food_toss")
EVALUATION_FILENAME = "evaluate_ret_00.json"
LEGACY_SUMMARY_FILENAME = "evaluation_summary.json"


def attach_episode_budget(
    result: dict[str, Any],
    task: str,
    *,
    count: int | None,
    step_out: int,
    source: str,
    strict: bool = True,
) -> None:
    """Append a budget result without changing native checkpoints or E2E."""
    scores = result["scores"]
    rows = scores["STEPS"]
    profile = breakdown = None
    try:
        profile = get_weight_profile(task, None, weights_path=WEIGHTS_PATH)
        if profile is None:
            raise ValueError(f"missing DexHand 2 weight profile: {task}")
        if count is not None:
            if tuple(row["name"] for row in rows) != profile.steps:
                raise ValueError("DexHand 2 stages do not match the weight profile")
            breakdown = episode_policy_budget_breakdown(
                rows, profile, scores["E2E"], step_out, count
            )
            result["budget_evidence"] = {
                "source": source,
                "accepted_batches": count,
                "episode_budget": breakdown["budget"],
            }
    except (OSError, ValueError) as exc:
        if strict:
            raise
        result["budget_scoring_error"] = str(exc)
    write_policy_budget_scores(result, profile, breakdown)


def make_evaluation(
    task: str,
    instruction: str,
    native: dict[str, Any],
    *,
    stages: tuple[str, ...],
    stage_scores: dict[str, int | float] | None,
    e2e: int | None,
    count: int | None,
    step_out: int,
    steps: int,
    duration: float,
    valid: bool,
) -> dict[str, Any]:
    """Map a MuJoCo or Newton episode into the shared evaluation format."""
    rows = (
        []
        if stage_scores is None
        else [
            {"step": i, "name": name, "score": stage_scores[name]}
            for i, name in enumerate(stages)
        ]
    )
    result = {
        "code": 0 if valid else 1,
        "step": steps,
        "scores": {"STEPS": rows, "E2E": e2e},
    }
    attach_episode_budget(
        result,
        task,
        count=count if valid else None,
        step_out=step_out,
        source="native_runner",
        strict=False,
    )
    report = {
        "task_name": task,
        "score_layout_name": task,
        "task_instruction": instruction,
        "duration": duration,
        "valid": valid,
        "result": result,
        "native_result": native,
    }
    if count is not None:
        report["policy_dispatch_count"] = count
    return report


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Keep a failed derived export separate from the native episode outcome."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).warning(
            "Cannot write DexHand 2 report %s: %s", path, exc
        )
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def read_summary_details(directory: Path) -> list[dict[str, Any]]:
    path = directory / EVALUATION_FILENAME
    if not path.is_file():
        path = directory / LEGACY_SUMMARY_FILENAME
        if not path.is_file():
            return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        details = value.get("details") if isinstance(value, dict) else None
        if not isinstance(details, list) or not all(isinstance(row, dict) for row in details):
            raise ValueError("expected a list of evaluation details")
        return details
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).warning(
            "Cannot read DexHand 2 report %s: %s", path, exc
        )
        return []


def write_summary(directory: Path, evaluations: list[dict[str, Any]]) -> None:
    """Write the common details/statistics report alongside native summaries."""
    from .eval_utils import get_statistics

    if not evaluations:
        return
    try:
        task = evaluations[0]["task_name"]
        profile = get_weight_profile(task, None, weights_path=WEIGHTS_PATH)
        if profile is None or any(
            evaluation["task_name"] != task for evaluation in evaluations
        ):
            raise ValueError("DexHand 2 summary requires one supported task")
        eligible = [evaluation for evaluation in evaluations if evaluation["valid"]]
        statistics = get_statistics(eligible, profile.steps, profile)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logging.getLogger(__name__).warning(
            "Cannot aggregate DexHand 2 reports in %s: %s", directory, exc
        )
        return
    write_report(
        directory / EVALUATION_FILENAME,
        {"details": evaluations, "statistics": statistics},
    )
