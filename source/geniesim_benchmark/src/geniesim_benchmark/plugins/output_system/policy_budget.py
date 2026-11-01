# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Versioned stage efficiency scores; native scores and termination are untouched."""

import math
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .weighted_scoring import WeightProfile, weighted_score_breakdown

POLICY_BUDGET_LEGACY_VERSION = "policy-dispatch-budget-v1"
POLICY_BUDGET_VERSION = "policy-dispatch-budget-v2"
POLICY_BUDGET_VERSIONS = (POLICY_BUDGET_LEGACY_VERSION, POLICY_BUDGET_VERSION)


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
        if not isinstance(config, dict) or "steps" not in config or set(config) - {"version", "steps"}:
            raise ValueError("PolicyDispatchBudget configuration requires steps; version is optional")
        version = config.get("version", POLICY_BUDGET_VERSION)
        if version not in POLICY_BUDGET_VERSIONS:
            raise ValueError("Unsupported PolicyDispatchBudget version")
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
                "version": version,
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
                policy_budget_baseline(stage_budget, version)
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


def policy_budget_baseline(step_out: int, version: str = POLICY_BUDGET_VERSION) -> int:
    """v2: floor(StepOut / 30); explicit v1 preserves its old 1.1 multiplier."""
    if version not in POLICY_BUDGET_VERSIONS:
        raise ValueError("Unsupported PolicyDispatchBudget version")
    if type(step_out) is not int or step_out <= 0:
        raise ValueError("Policy budget requires a positive integer StepOut")
    baseline = step_out // 30 if version == POLICY_BUDGET_VERSION else step_out * 11 // 300
    if baseline == 0:
        raise ValueError("Policy budget baseline rounds to zero")
    return baseline


def assess_policy_budget(step_out: int, count: int, version: str = POLICY_BUDGET_VERSION) -> dict[str, Any]:
    baseline = policy_budget_baseline(step_out, version)
    if type(count) is not int or count < 0:
        raise ValueError("Policy dispatch count must be a nonnegative integer")
    # Each FULL 10% excess is one deduction, always relative to the raw score.
    tenths = min(10, max(0, (count - baseline) * 10 // baseline))
    return {
        "version": version,
        "step_out": step_out,
        "baseline": baseline,
        "count": count,
        "over_budget": count > baseline,
        "penalty_tenths": tenths,
        "factor": (10 - tenths) / 10,
    }


def policy_budget_breakdown(
    result: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    profile: WeightProfile | None,
) -> dict[str, Any] | None:
    """Use explicitly registered stage monitors; never infer old missing counts."""
    monitors = [r for r in result.get("progress", []) if r.get("class_name") == "PolicyDispatchBudget"]
    if not monitors:
        return None
    if profile is None:
        raise ValueError("PolicyDispatchBudget requires an explicit task weight profile")
    native = weighted_score_breakdown(rows, profile, result.get("scores", {}).get("E2E"))
    specs = {}
    for monitor in monitors:
        progress = monitor.get("progress")
        spec = progress.get("POLICY_BUDGET") if isinstance(progress, dict) else None
        if not isinstance(spec, dict) or spec.get("version") not in POLICY_BUDGET_VERSIONS:
            raise ValueError("Missing or unsupported policy budget evidence")
        index = spec.get("step")
        if type(index) is not int or not 0 <= index < len(profile.steps) or index in specs:
            raise ValueError("Duplicate or invalid policy budget step index")
        if index != len(specs):
            raise ValueError("Policy budget step indices must follow native stage order")
        policy_budget_baseline(spec.get("step_out"), spec["version"])
        specs[index] = spec
    if set(specs) != set(range(len(profile.steps))):
        raise ValueError("PolicyDispatchBudget must cover every configured stage")
    versions = {spec["version"] for spec in specs.values()}
    if len(versions) != 1:
        raise ValueError("PolicyDispatchBudget stages must use one rule version")
    version = versions.pop()
    configured_points = [specs[i].get("points") for i in range(len(profile.steps))]
    if any(p is not None for p in configured_points):
        configured_points = configured_policy_points(configured_points)
    else:
        configured_points = [w / 2 for w in profile.weights]
    # Authored task points remain a 50-unit proportion. Scale once here so
    # saved monitor evidence and task configs need no numeric migration.
    authored_points = configured_points
    configured_points = [p * profile.process_points / 50 for p in authored_points]
    raw_scores = {r["step"]: float(r["score"]) for r in rows}
    adjusted = []
    within_budget = True
    for index, name in enumerate(profile.steps):
        spec = specs[index]
        raw = raw_scores.get(index, 0.0)
        count = spec.get("count")
        if count is None:
            if raw != 0:
                raise ValueError("Successful stage is missing policy budget evidence")
            # An unexecuted failed stage has no count, never an invented zero.
            assessment = {
                "baseline": policy_budget_baseline(spec["step_out"], version),
                "count": None,
                "over_budget": None,
                "penalty_tenths": None,
                "factor": None,
            }
            score = 0.0
            within_budget = False
        else:
            assessment = assess_policy_budget(spec["step_out"], count, version)
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
    eligible = native.complete and within_budget
    bonus = profile.completion_bonus_points / profile.total_points if eligible else 0.0
    return {
        "steps": adjusted,
        "process": process,
        "completion_bonus": bonus,
        "total": process + bonus,
        "bonus_eligible": eligible,
        "scoring": {
            "version": version,
            "baseline_rounding": "floor",
            "reference_steps_per_dispatch": 30,
            "baseline_multiplier": "1" if version == POLICY_BUDGET_VERSION else "11/10",
            "penalty_rule": "each_full_10_percent_excess_deducts_10_percent_of_raw_stage_score",
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
    breakdown = policy_budget_breakdown(result, rows, profile)
    if breakdown is None:
        return
    result["scores"].update(
        {
            "POLICY_BUDGET_STEPS": breakdown["steps"],
            "POLICY_BUDGET_PROCESS": breakdown["process"],
            "POLICY_BUDGET_COMPLETION_BONUS": breakdown["completion_bonus"],
            "POLICY_BUDGET_WEIGHTED": breakdown["total"],
            "POLICY_BUDGET_BONUS_ELIGIBLE": int(breakdown["bonus_eligible"]),
        }
    )
    result["policy_budget_scoring"] = breakdown["scoring"]
