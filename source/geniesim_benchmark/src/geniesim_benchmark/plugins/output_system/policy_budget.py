# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Stage efficiency scores; native scores and termination are untouched."""

import math
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .weighted_scoring import WeightProfile, native_stage_scores

POLICY_BUDGET_RULE = "policy-dispatch-budget"


class PolicyBudgetError(ValueError):
    """An explicitly selected budget rule cannot fall back to a legacy task."""


def select_scoring_problem(problem: Mapping[str, Any], robot_cfg: str | None) -> Mapping[str, Any]:
    """Select a declared robot scoring layout from the same native task file."""
    variants = problem.get("ScoringByRobot")
    if variants is None:
        return problem
    if not isinstance(variants, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(spec, dict)
        or set(spec) != {"Acts", "ScoreLayout", "ScoreCheckers"}
        or not isinstance(spec["ScoreLayout"], str)
        or not spec["ScoreLayout"]
        for name, spec in variants.items()
    ):
        raise PolicyBudgetError("ScoringByRobot requires Acts, ScoreLayout and ScoreCheckers per robot")
    selected = variants.get(robot_cfg)
    return {**problem, **deepcopy(selected)} if selected is not None else problem


def configured_policy_points(values: Sequence[Any]) -> list[float]:
    points = list(values)
    if not points or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in points):
        raise ValueError("PolicyDispatchBudget needs finite nonnegative points for every stage")
    if not math.isclose(math.fsum(points), 50, abs_tol=1e-9, rel_tol=0):
        raise ValueError("PolicyDispatchBudget stage points must sum to 50")
    return points


def configured_scoring_action(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Compile a standalone score checker without editing the native Acts tree."""
    if "ScoreCheckers" not in problem:
        return problem["Acts"][0]
    try:
        checkers = problem["ScoreCheckers"]
        if (
            not isinstance(checkers, list)
            or len(checkers) != 1
            or not isinstance(checkers[0], dict)
            or set(checkers[0]) != {"PolicyDispatchBudget"}
        ):
            raise ValueError("ScoreCheckers requires one PolicyDispatchBudget configuration")
        config = checkers[0]["PolicyDispatchBudget"]
        if not isinstance(config, dict) or set(config) != {"steps"}:
            raise ValueError("PolicyDispatchBudget configuration requires only steps")
        steps = config["steps"]
        if not isinstance(steps, list) or any(
            not isinstance(s, dict)
            or set(s) != {"checker", "points"}
            or not isinstance(s["checker"], str)
            or not s["checker"]
            for s in steps
        ):
            raise ValueError("Each policy budget step requires checker and points")
        configured_policy_points([s["points"] for s in steps])
        index = 0

        def wrap(action, name, inherited=None):
            nonlocal index
            if index >= len(steps) or steps[index]["checker"] != name:
                raise ValueError("PolicyDispatchBudget checkers must cover native stages in order")
            spec = {
                "step": index,
                "points": steps[index]["points"],
                "action": action,
            }
            if inherited is not None:
                spec["step_out"] = inherited
            index += 1
            return {"PolicyDispatchBudget": spec}

        def visit(node, budget=None):
            if not isinstance(node, dict) or len(node) != 1:
                raise ValueError("PolicyDispatchBudget requires single-key native actions")
            key, value = next(iter(node.items()))
            if key in ("ActionSetWaitAny", "ActionSetSuccessFirst"):
                exits = [item["StepOut"] for item in value if "StepOut" in item]
                if len(exits) > 1 or (not exits and budget is None):
                    raise ValueError("PolicyDispatchBudget needs exactly one native stage StepOut")
                stage_budget = exits[0] if exits else budget
                policy_budget_baseline(stage_budget)
                guards = {"StepOut", "Onfloor", "OnSurface", "TimeOut"}
                leaves = [k for item in value for k in item if k not in guards]
                # Either-hand alternatives are one native scoring checkpoint:
                # score_layout_steps takes the best of consecutive same-class rows.
                # Wrap the group once so a dispatch is never charged to both arms.
                if (
                    leaves
                    and len(set(leaves)) == 1
                    and leaves[0] not in ("ActionList", "ActionSetWaitAny", "ActionSetSuccessFirst", "ActionSetWaitAll")
                    and not leaves[0].startswith("ActionSetWaitSome")
                ):
                    return wrap(node, leaves[0], None if exits else stage_budget)
                return {key: [visit(item, stage_budget) for item in value]}
            # A layout may score an all-of group of same-class predicates as
            # one checkpoint (e.g. two objects near the robot). Keep that group
            # intact when the authored score list declares just one entry.
            if key == "ActionSetWaitAll" and len(value) > 1:
                names = [next(iter(item)) for item in value]
                if len(set(names)) == 1 and index < len(steps) and steps[index]["checker"] == names[0]:
                    remaining = [item["checker"] for item in steps[index:]]
                    if len(remaining) == 1 or remaining[1] != names[0]:
                        if budget is None:
                            raise ValueError("PolicyDispatchBudget cannot find the native stage StepOut")
                        return wrap(node, names[0], budget)
            # Concurrent all-of and N-of-M checkpoints retain their native
            # completion/cancellation rules and inherit the enclosing timeout.
            if key in ("ActionList", "ActionSetWaitAll") or key.startswith("ActionSetWaitSome"):
                return {key: [visit(item, budget) for item in value]}
            # Failure exits retain their native lifecycle and never earn points.
            if key in ("StepOut", "Onfloor", "OnSurface", "TimeOut"):
                return node
            if budget is None:
                raise ValueError("PolicyDispatchBudget cannot find the native stage StepOut")
            return wrap(node, key, budget)

        if len(problem["Acts"]) != 1:
            raise ValueError("PolicyDispatchBudget requires one native Acts tree")
        action = visit(deepcopy(problem["Acts"][0]))
        if index != len(steps):
            raise ValueError("PolicyDispatchBudget checkers must cover every native stage")
        return action
    except Exception as exc:
        raise PolicyBudgetError(f"Invalid ScoreCheckers: {exc}") from exc


def policy_budget_baseline(step_out: int) -> int:
    """Allocate 80% of one dispatch per 30 steps without reducing StepOut."""
    if type(step_out) is not int or step_out <= 0:
        raise ValueError("Policy budget requires a positive integer StepOut")
    baseline = step_out // 30
    if baseline == 0:
        raise ValueError("Policy budget baseline rounds to zero")
    return max(1, baseline * 4 // 5)


def assess_policy_budget(step_out: int, count: int) -> dict[str, Any]:
    baseline = policy_budget_baseline(step_out)
    if type(count) is not int or count < 0:
        raise ValueError("Policy dispatch count must be a nonnegative integer")
    remaining = max(0, min(baseline, 3 * baseline - 2 * count))
    factor = remaining / baseline
    return {
        "version": POLICY_BUDGET_RULE,
        "step_out": step_out,
        "baseline": baseline,
        "count": count,
        "over_budget": count > baseline,
        "penalty_fraction": 1.0 - factor,
        "factor": factor,
    }


def policy_budget_rule() -> dict[str, Any]:
    """Record the numerical rule so comparisons do not depend on its label."""
    return {
        "version": POLICY_BUDGET_RULE,
        "baseline_rounding": "floor_then_scale_floor_min_one",
        "reference_steps_per_dispatch": 30,
        "baseline_multiplier": "4/5",
        "penalty_rule": "twice_proportional_excess_capped_at_one",
    }


def episode_policy_budget_breakdown(
    rows: Sequence[Mapping[str, Any]], profile: WeightProfile, native_e2e: int,
    step_out: int, count: int,
) -> dict[str, Any]:
    """Apply the DexHand 2 episode budget once to all native checkpoints."""
    budget = assess_policy_budget(step_out, count)
    raw_scores, complete = native_stage_scores(rows, profile, native_e2e)
    factor = budget["factor"]
    steps = []
    for index, (name, raw, weight) in enumerate(zip(profile.steps, raw_scores, profile.weights, strict=True)):
        points = weight * profile.process_points / 100
        score = raw * factor
        steps.append({
            "step": index, "name": name, "raw_score": raw, "score": score, "factor": factor,
            "points": points, "earned_points": score * points,
        })
    process = math.fsum(row["earned_points"] for row in steps) / profile.total_points
    eligible = complete and not budget["over_budget"]
    bonus = profile.completion_bonus_points / profile.total_points if eligible else 0.0
    return {
        "steps": steps,
        "process": process,
        "completion_bonus": bonus,
        "total": process + bonus,
        "bonus_eligible": eligible,
        "scoring": {
            "scope": "episode",
            **policy_budget_rule(),
            "step_out": step_out,
            "baseline": budget["baseline"],
        },
        "budget": budget,
    }


def policy_budget_breakdown(
    result: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    profile: WeightProfile | None,
) -> dict[str, Any] | None:
    """Use recorded episode evidence or explicitly registered stage monitors."""
    scoring = result.get("policy_budget_scoring")
    if isinstance(scoring, Mapping) and scoring.get("scope") == "episode":
        if profile is None or result.get("weighted_scoring") != profile.to_dict():
            raise ValueError("Episode policy budget requires its recorded weight profile")
        evidence = result["budget_evidence"]
        breakdown = episode_policy_budget_breakdown(
            rows, profile, result["scores"]["E2E"], scoring["step_out"], evidence["accepted_batches"]
        )
        if scoring != breakdown["scoring"] or evidence["episode_budget"] != breakdown["budget"]:
            raise ValueError("Episode policy budget evidence disagrees with the recorded configuration")
        scores = result["scores"]
        total = breakdown["total"]
        if scores.get("POLICY_BUDGET_WEIGHTED") != total or scores.get("WEIGHTED", total) != total:
            raise ValueError("Episode policy budget total disagrees with its evidence")
        return breakdown
    monitors = [r for r in result.get("progress", []) if r.get("class_name") == "PolicyDispatchBudget"]
    if not monitors:
        return None
    if profile is None:
        raise ValueError("PolicyDispatchBudget requires an explicit task weight profile")
    native_scores, native_complete = native_stage_scores(rows, profile, result.get("scores", {}).get("E2E"))
    specs = {}
    for monitor in monitors:
        progress = monitor.get("progress")
        spec = progress.get("POLICY_BUDGET") if isinstance(progress, dict) else None
        if not isinstance(spec, dict) or spec.get("version") != POLICY_BUDGET_RULE:
            raise ValueError("Missing or unsupported policy budget evidence")
        index = spec.get("step")
        if type(index) is not int or not 0 <= index < len(profile.steps) or index in specs:
            raise ValueError("Duplicate or invalid policy budget step index")
        if index != len(specs):
            raise ValueError("Policy budget step indices must follow native stage order")
        policy_budget_baseline(spec.get("step_out"))
        specs[index] = spec
    if set(specs) != set(range(len(profile.steps))):
        raise ValueError("PolicyDispatchBudget must cover every configured stage")
    configured_points = [specs[i].get("points") for i in range(len(profile.steps))]
    if any(p is not None for p in configured_points):
        configured_points = configured_policy_points(configured_points)
    else:
        configured_points = [w / 2 for w in profile.weights]
    # Authored task points remain a 50-unit proportion. Scale once here so
    # saved monitor evidence and task configs need no numeric migration.
    authored_points = configured_points
    configured_points = [p * profile.process_points / 50 for p in authored_points]
    adjusted = []
    within_budget = True
    for index, name in enumerate(profile.steps):
        spec = specs[index]
        raw = native_scores[index]
        count = spec.get("count")
        if count is None:
            if raw != 0:
                raise ValueError("Successful stage is missing policy budget evidence")
            # An unexecuted failed stage has no count, never an invented zero.
            assessment = {
                "baseline": policy_budget_baseline(spec["step_out"]),
                "count": None,
                "over_budget": None,
                "factor": None,
            }
            assessment["penalty_fraction"] = None
            score = 0.0
            within_budget = False
        else:
            assessment = assess_policy_budget(spec["step_out"], count)
            score = raw * assessment["factor"]
            within_budget = within_budget and not assessment["over_budget"]
        adjusted.append(
            {
                "step": index,
                "name": name,
                "raw_score": raw,
                "score": score,
                "step_out": spec["step_out"],
                "points": configured_points[index],
                "earned_points": configured_points[index] * score,
                **assessment,
            }
        )
    process = math.fsum(row["earned_points"] for row in adjusted) / profile.total_points
    eligible = native_complete and within_budget
    bonus = profile.completion_bonus_points / profile.total_points if eligible else 0.0
    return {
        "steps": adjusted,
        "process": process,
        "completion_bonus": bonus,
        "total": process + bonus,
        "bonus_eligible": eligible,
        "scoring": {
            **policy_budget_rule(),
            "completion_rule": "all_steps_and_native_e2e_and_every_stage_count_at_most_baseline",
            "stage_step_outs": [specs[i]["step_out"] for i in range(len(profile.steps))],
            "stage_points": configured_points,
            "configured_stage_points": authored_points,
            "weight_profile": profile.to_dict(),
        },
    }


def add_policy_budget_scores(
    result: dict[str, Any], rows: Sequence[Mapping[str, Any]], profile: WeightProfile | None
) -> None:
    """Publish canonical budget scores, leaving missing evidence unavailable."""
    write_policy_budget_scores(result, profile, policy_budget_breakdown(result, rows, profile))


def write_policy_budget_scores(
    result: dict[str, Any], profile: WeightProfile | None, breakdown: dict[str, Any] | None,
) -> None:
    """Populate the shared score fields from a validated budget breakdown."""
    scores = result.setdefault("scores", {})
    scores["POLICY_BUDGET_STEPS"] = (
        [{key: value for key, value in row.items() if key != "raw_score"} for row in breakdown["steps"]]
        if breakdown is not None else None
    )
    for weighted_key, policy_key, source in (
        ("WEIGHTED", "POLICY_BUDGET_WEIGHTED", "total"),
        ("WEIGHTED_PROCESS", "POLICY_BUDGET_PROCESS", "process"),
        ("WEIGHTED_COMPLETION_BONUS", "POLICY_BUDGET_COMPLETION_BONUS", "completion_bonus"),
        ("WEIGHTED_COMPLETE", "POLICY_BUDGET_BONUS_ELIGIBLE", "bonus_eligible"),
    ):
        value = breakdown[source] if breakdown is not None else None
        if source == "bonus_eligible" and value is not None:
            value = int(value)
        scores.pop(weighted_key, None)
        scores[policy_key] = value
    result["weighted_scoring_metric"] = "POLICY_BUDGET_WEIGHTED"
    result["weighted_scoring"] = profile.to_dict() if profile is not None else None
    result["policy_budget_scoring"] = breakdown["scoring"] if breakdown is not None else None
