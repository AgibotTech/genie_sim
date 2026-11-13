# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Write budget-adjusted weighted scores to a new file, preserving the source."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from geniesim_benchmark.plugins.output_system.eval_utils import extract_scores, get_statistics, get_task_steps
from geniesim_benchmark.plugins.output_system.weighted_scoring import (
    canonical_score_layout, get_weight_profile,
)
from geniesim_benchmark.plugins.output_system.policy_budget import add_policy_budget_scores


def reweight_results(
    data: dict[str, Any],
    sub_task_name: str,
    instruction_mode: str = "full",
    *,
    weights_path: Path | None = None,
) -> dict[str, Any]:
    """Keep native fields and replace weighted outputs with policy-budget values.

    The caller chooses the source records, including any prior abnormal-attempt
    selection. This function neither retries nor filters nor selects episodes.
    """
    scoring_tasks = set()
    for detail in data["details"]:
        layout = detail.get("score_layout_name")
        if layout is not None and (not isinstance(layout, str) or not layout):
            raise ValueError("ScoreLayout must name a registered scoring layout")
        scoring_tasks.add(canonical_score_layout(layout or sub_task_name))
    if len(scoring_tasks) > 1:
        raise ValueError("Input contains different scoring layouts")
    scoring_task = next(iter(scoring_tasks), sub_task_name)
    steps = get_task_steps(scoring_task, instruction_mode) if weights_path is None else None
    profile = get_weight_profile(scoring_task, steps, instruction_mode, weights_path=weights_path)
    if profile is None:
        raise ValueError(f"No explicit weight profile for {scoring_task}")
    steps = list(profile.steps)
    updated = deepcopy(data)
    for detail in updated["details"]:
        if detail.get("sub_task_name", sub_task_name) != sub_task_name:
            raise ValueError("Input contains a different sub_task_name")
        if detail.get("instruction_mode", instruction_mode) != instruction_mode:
            raise ValueError("Input contains a different instruction_mode")
        detail.setdefault("sub_task_name", sub_task_name)
        detail.setdefault("instruction_mode", instruction_mode)
        result = detail["result"]
        rows = extract_scores(result, steps)
        add_policy_budget_scores(result, rows, profile)
    stats = get_statistics(updated["details"], steps, profile)
    statistics = updated.setdefault("statistics", {})
    for alias in (
        "weighted", "weighted_by_instruction", "weighted_process",
        "weighted_completion_bonus", "weighted_completion_rate", "weighted_episodes",
    ):
        statistics.pop(alias, None)
    statistics.update(
        {
            key: value
            for key, value in stats.items()
            if key.startswith("policy_budget_") or key in ("weighted_scoring", "weighted_scoring_metric")
        }
    )
    return updated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="New file; existing files are never overwritten"
    )
    parser.add_argument(
        "--sub-task-name", required=True, help="Exact benchmark sub_task_name, required for older files"
    )
    parser.add_argument("--instruction-mode", choices=("full", "subtask"), default="full")
    parser.add_argument(
        "--weights-file",
        type=Path,
        help="Explicit versioned weights/layout JSON for historical results; defaults to the current profile",
    )
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        updated = reweight_results(
            data, args.sub_task_name, args.instruction_mode, weights_path=args.weights_file
        )
        # Serialize before creating the output so invalid data cannot leave a partial file.
        content = json.dumps(updated, indent=4, ensure_ascii=False, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as output:
            output.write(content)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Weighted results -> {args.output}")


if __name__ == "__main__":
    main()
