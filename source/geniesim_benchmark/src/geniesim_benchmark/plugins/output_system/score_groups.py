# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared scoring identity, fixed denominators and completion checks (CPU only)."""
from collections import Counter
from itertools import groupby
import math

SCORE_SCHEMA_VERSION = 2
FAILURE_CHECKERS = {"StepOut", "TimeOut", "Onfloor"}


def score_group_ids(task_progress):
    """Keep independent leaves separate; group direct same-class ANY candidates.

    The parser's live action objects retain the hierarchy lost by the flat
    progress list. Capture it before dropping those objects for JSON output.
    IDs are episode-local, like the existing checker IDs, not cross-run keys.
    """
    if not task_progress or not all(item.get("acion_obj") is not None for item in task_progress):
        return {}
    by_action = {id(item["acion_obj"]): item for item in task_progress}
    groups = {item["id"]: f"checker:{item['id']}" for item in task_progress}
    for parent in task_progress:
        if parent["class_name"] not in {"ActionSetWaitAny", "ActionSetSuccessFirst"}:
            continue
        for child in parent["acion_obj"]._actions:
            item = by_action.get(id(child))
            # A nested ALL/list is not an alternative leaf. Exit conditions
            # terminate evaluation; they are not alternative success goals.
            if item is None or hasattr(child, "_actions"):
                continue
            if item["class_name"] in {"StepOut", "TimeOut", "Onfloor"}:
                continue
            groups[item["id"]] = f"any:{parent['id']}:{item['class_name']}"
    return groups


def goal_tree_metadata(rows):
    """Serialize logical requirements before discarding live action objects."""
    if not rows or not all(row.get("acion_obj") is not None for row in rows):
        return {}
    identities = {id(row["acion_obj"]): row["id"] for row in rows}
    metadata = {}
    for row in rows:
        action = row["acion_obj"]
        name = row["class_name"]
        children = list(getattr(action, "_actions", []))
        if name == "PolicyDispatchBudget":
            children = [action.stage]
        operator = "leaf"
        if children:
            operator = {"ActionSetWaitAny": "any", "ActionSetSuccessFirst": "any", "ActionSetWaitSome": "some"}.get(
                name, "all"
            )
        metadata[row["id"]] = {
            "goal_operator": operator,
            "goal_children": [identities[id(child)] for child in children],
        }
        if operator == "some":
            metadata[row["id"]]["goal_min_complete"] = action.min_complete
        if hasattr(action, "is_finished"):
            metadata[row["id"]]["goal_finished"] = bool(action.is_finished())
    return metadata


def goal_tree_complete(rows):
    """Evaluate ANY/ALL/SOME and sequential goals, excluding termination guards.

    Return None for historical flat exports, whose lost topology cannot be
    reconstructed. Their conservative fallback requires every configured slot.
    """
    if not rows or not any("goal_operator" in row for row in rows):
        return None
    if not all("goal_operator" in row for row in rows):
        raise ValueError("Incomplete goal tree metadata")
    by_id = {row["id"]: row for row in rows}
    children = {child for row in rows for child in row["goal_children"]}
    ignored = FAILURE_CHECKERS | {"EvalExitAction"}

    def complete(row):
        progress = row.get("progress") or {}
        if row["goal_operator"] == "leaf":
            if "GOAL_COMPLETE" in progress:
                return progress["GOAL_COMPLETE"] is True
            if "SCORE" in progress:
                return progress["SCORE"] == 1
            return row.get("goal_finished", False)
        goals = [by_id[child] for child in row["goal_children"] if by_id[child]["class_name"] not in ignored]
        passed = sum(complete(goal) for goal in goals)
        if row["goal_operator"] == "any":
            return passed >= 1
        if row["goal_operator"] == "some":
            return passed >= row["goal_min_complete"]
        return bool(goals) and passed == len(goals)

    roots = [row for row in rows if row["id"] not in children and row["class_name"] not in ignored]
    return bool(roots) and all(complete(row) for row in roots)


def score_layout_steps(rows, steps):
    """Use explicit goal groups, or supported legacy flat-list layouts.

    Legacy records can express one row per slot or one consecutive run per
    slot. Unequal, mixed group sizes cannot be recovered from class names;
    reject that ambiguity instead of inventing equal-size partitions.
    """
    rows = list(rows)
    steps = list(steps)
    if not steps:
        return []

    def score(row):
        progress = row.get("progress")
        return float(progress.get("SCORE", 0)) if isinstance(progress, dict) else 0.0

    groups = []
    if any("score_group" in row for row in rows):
        if not all(isinstance(row.get("score_group"), str) and row["score_group"] for row in rows):
            raise ValueError("Incomplete or invalid score_group metadata")
        by_group = {}
        for row in rows:
            key = row["score_group"]
            if key not in by_group:
                group = {"name": row["class_name"], "score": score(row)}
                by_group[key] = group
                groups.append(group)
            else:
                group = by_group[key]
                if group["name"] != row["class_name"]:
                    raise ValueError("A score_group cannot contain different checker classes")
                group["score"] = max(group["score"], score(row))
    else:
        runs = [(name, list(items)) for name, items in groupby(rows, key=lambda row: row["class_name"])]
        row_counts = Counter(row["class_name"] for row in rows)
        run_counts = Counter(name for name, _ in runs)
        slot_counts = Counter(steps)
        for name, run in runs:
            if name not in slot_counts:
                continue
            if len(run) == 1 or row_counts[name] == slot_counts[name]:
                groups.extend({"name": name, "score": score(row)} for row in run)
            elif slot_counts[name] == 1 or run_counts[name] == slot_counts[name]:
                groups.append({"name": name, "score": max(score(row) for row in run)})
            else:
                raise ValueError(
                    f"Legacy {name} scoring is ambiguous; score_group metadata or the original task tree is required"
                )

    result = []
    cursor = 0
    for index, name in enumerate(steps):
        match = next((i for i in range(cursor, len(groups)) if groups[i]["name"] == name), None)
        value = 0.0
        if match is not None:
            value = groups[match]["score"]
            cursor = match + 1
        result.append({"step": index, "name": name, "score": value})
    return result


def normalize_steps(recorded, layout):
    """Retain configured slots, filling unrecorded goals with zero credit."""
    result = [{"step": i, "name": name, "score": 0.0} for i, name in enumerate(layout)]
    seen = set()
    for item in recorded:
        index = item["step"]
        if type(index) is not int or index < 0 or index >= len(layout) or index in seen:
            raise ValueError("Invalid or duplicate scoring step index")
        if item["name"] != layout[index]:
            raise ValueError("Recorded mismatched checker in the scoring layout")
        value = float(item["score"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Checker scores must be finite values in [0, 1]")
        result[index]["score"] = value
        seen.add(index)
    return result


def episode_complete(result, scores, layout):
    """Failure exits and incomplete goals never count as end-to-end success."""
    if result.get("code", 0) != 0 or not layout or len(scores) != len(layout):
        return False
    for row in result.get("progress", []):
        progress = row.get("progress") or {}
        if row["class_name"] in FAILURE_CHECKERS and "SCORE" in progress:
            return False
    complete = goal_tree_complete(result.get("progress", []))
    if complete is not None:
        # Subtask checkers are instantiated lazily. A complete partial tree
        # cannot stand in for later configured milestones that do not exist yet.
        declared = score_layout_steps([{**row, "progress": {"SCORE": 1}} for row in result["progress"]], layout)
        return complete and all(step["score"] == 1 for step in declared)
    if any(s["step"] != i or s["name"] != name or s["score"] != 1.0 for i, (s, name) in enumerate(zip(scores, layout))):
        return False
    if any((row.get("progress") or {}).get("GOAL_COMPLETE", True) is not True for row in result.get("progress", [])):
        return False
    return True


def episode_counts(results):
    counts = dict(attempted=len(results), completed=0, failed=0, invalid=0)
    for item in results:
        result = item["result"]
        if result.get("code", 0) not in (0, 2):
            counts["invalid"] += 1
        elif result.get("scores", {}).get("E2E") == 1:
            counts["completed"] += 1
        else:
            counts["failed"] += 1
    return counts
