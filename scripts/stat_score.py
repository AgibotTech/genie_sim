#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Count scores from evaluate_ret_*.json files in the specified directory
- Count average E2E scores
- Count average scores for each step
"""

import json
import os
import glob
import argparse
from collections import defaultdict
from pathlib import Path


def load_json_files(directory):
    """Load all evaluate_ret_*.json files in the directory"""
    pattern = os.path.join(directory, "evaluate_ret_*.json")
    json_files = glob.glob(pattern)
    json_files.sort()
    return json_files


def process_json_file(filepath):
    """Process a single JSON file and extract E2E and step scores"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    e2e_scores = []
    step_scores = defaultdict(list)  # step_name -> [score1, score2, ...]

    # Iterate through details array
    if "details" in data:
        for detail in data["details"]:
            if "result" not in detail:
                continue

            result = detail["result"]

            # Method 1: Extract from result.scores (priority)
            if "scores" in result:
                scores = result["scores"]

                # Extract E2E score
                if "E2E" in scores:
                    e2e_score = scores["E2E"]
                    if isinstance(e2e_score, (int, float)):
                        e2e_scores.append(e2e_score)

                # Extract step scores (from STEPS array)
                if "STEPS" in scores and isinstance(scores["STEPS"], list):
                    for step in scores["STEPS"]:
                        if isinstance(step, dict) and "name" in step and "score" in step:
                            step_name = step["name"]
                            step_score = step["score"]
                            if isinstance(step_score, (int, float)):
                                step_scores[step_name].append(step_score)

            # Method 2: Extract from result.progress array (if scores field is not present)
            elif "progress" in result and isinstance(result["progress"], list):
                # Extract step scores from progress array
                # Ignored class_name: ActionList, ActionSetWaitAny, StepOut
                ignored_classes = {"ActionList", "ActionSetWaitAny", "StepOut"}

                for item in result["progress"]:
                    if isinstance(item, dict):
                        class_name = item.get("class_name", "")
                        progress = item.get("progress", {})

                        # If class_name is not in the ignore list and has SCORE field
                        if class_name and class_name not in ignored_classes:
                            if isinstance(progress, dict) and "SCORE" in progress:
                                score = progress["SCORE"]
                                if isinstance(score, (int, float)):
                                    step_scores[class_name].append(score)

    return e2e_scores, step_scores


def calculate_statistics(directory):
    """Calculate statistics"""
    json_files = load_json_files(directory)

    if not json_files:
        print(f"Warning: No evaluate_ret_*.json files found in directory {directory}")
        return None

    all_e2e_scores = []
    all_step_scores = defaultdict(list)

    # Process all JSON files
    for filepath in json_files:
        e2e_scores, step_scores = process_json_file(filepath)
        all_e2e_scores.extend(e2e_scores)
        for step_name, scores in step_scores.items():
            all_step_scores[step_name].extend(scores)

    # Calculate average values
    result = {"directory": os.path.abspath(directory), "file_count": len(json_files), "e2e": {}, "steps": {}}

    # E2E statistics
    if all_e2e_scores:
        result["e2e"] = {
            "average": sum(all_e2e_scores) / len(all_e2e_scores),
            "count": len(all_e2e_scores),
            "total": sum(all_e2e_scores),
        }
    else:
        result["e2e"] = {"average": None, "count": 0, "total": 0}

    # Step statistics
    for step_name, scores in all_step_scores.items():
        if scores:
            result["steps"][step_name] = {
                "average": sum(scores) / len(scores),
                "count": len(scores),
                "total": sum(scores),
            }

    return result


def main():
    parser = argparse.ArgumentParser(description="Count scores from evaluate_ret_*.json files")
    parser.add_argument("--dir", nargs="?", default=".", help="Directory path to count (default: current directory)")
    parser.add_argument("-o", "--output", help="Output file path (default: output to stdout)")

    args = parser.parse_args()

    # Ensure directory exists
    if not os.path.isdir(args.dir):
        print(f"Error: Directory {args.dir} does not exist")
        return 1

    # Calculate statistics
    result = calculate_statistics(args.dir)

    if result is None:
        return 1

    # Output results
    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Statistics saved to: {args.output}")
    else:
        with open(os.path.join(args.dir, "result.json"), "w", encoding="utf-8") as f:
            f.write(output_json)

    return 0


if __name__ == "__main__":
    exit(main())
