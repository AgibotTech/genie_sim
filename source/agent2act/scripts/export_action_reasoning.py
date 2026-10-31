# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Export public decision reasons and summaries from existing logs, paired with actions and later feedback."""

import argparse
import json
from pathlib import Path
from typing import Any

from agent2act.service.action_trace import response_metadata


def export(log: Path, output: Path) -> list[dict[str, Any]]:
    if output.resolve() == log.resolve():
        raise ValueError("output must differ from input log")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    issued: dict[tuple[str, str, int], dict[str, Any]] = {}
    for number, line in enumerate(log.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        inference = event.get("inference_id")
        if not inference:
            continue
        session = event.get("session_id", "")
        row = rows.setdefault(
            (session, inference),
            {
                "session_id": session,
                "inference_id": inference,
                "timestamp": event.get("timestamp"),
                "source_log": str(log.resolve()),
                "source_line": number,
                "episode": None,
                "sequence": None,
                "responses": [],
                "planned": [],
                "issued": [],
                "rejected": [],
                "feedback_after_action": [],
                "artifacts": [],
                "export_warnings": [],
                "meaning": "Public action reasons and API summaries only. Issued is not observed success.",
            },
        )
        for key in ("episode", "sequence"):
            if key in event:
                row[key] = event[key]
        status = event.get("status")
        if status in ("planned", "issued", "rejected"):
            row[status].append(event)
        if status == "issued":
            key = (session, json.dumps(row["episode"]), event["sequence"])
            issued[key] = row
        elif status == "observed_tracking":
            key = (session, json.dumps(row["episode"]), event["previous_sequence"])
            previous = issued.get(key)
            if previous is not None:
                previous["feedback_after_action"].append(event)
            else:
                row["export_warnings"].append("Previous issued action missing for tracking feedback")
        elif status == "episode_done":
            row["episode_done"] = True
            # Sequence numbers can restart even for identical instructions.
            for key in list(issued):
                if key[:2] == (session, json.dumps(row["episode"])):
                    del issued[key]
        if event.get("message") == "artifact_saved":
            kind = event["artifact_kind"]
            path = Path(event["artifact_path"])
            row["artifacts"].append({"kind": kind, "path": str(path)})
            if kind not in (
                "llm-request",
                "llm-response",
                "model-identity",
                "episode-interrupted",
                "diagnostic-input",
                "decision-diagnostics",
                "planner-output-decision",
                "reasoning-summary",
                "task-context",
            ):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                row["export_warnings"].append(f"{path}: {type(exc).__name__}")
                continue
            if kind in (
                "diagnostic-input",
                "decision-diagnostics",
                "planner-output-decision",
                "reasoning-summary",
                "task-context",
            ):
                row.setdefault(kind.replace("-", "_"), []).append({"artifact_path": str(path), **data})
            elif kind == "llm-request":
                row["requested_model"] = data.get("model")
            elif kind == "model-identity":
                if row["responses"]:
                    row["responses"][-1]["model_identity_attempt"] = data.get("model_identity_attempt")
            elif kind == "episode-interrupted":
                row["episode_interrupted"] = data
            else:
                latest = row["rejected"][-1] if row["rejected"] else None
                attempt = latest.get("attempt", 0) + 1 if latest else 0
                row["responses"].append(
                    {
                        "planning_attempt": attempt,
                        "artifact_path": str(path),
                        **response_metadata(data),
                    }
                )
    result = list(rows.values())
    for row in result:
        requested = row.get("requested_model")
        for response in row["responses"]:
            reported = response["reported_model"]
            response["model_identity_status"] = (
                "unreported"
                if not requested or not reported
                else "match"
                if requested == reported
                else "mismatch"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in result),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.log.with_suffix(".action-reasons.jsonl")
    rows = export(args.log, output)
    print(f"Exported {len(rows)} requests to {output.resolve()}")


if __name__ == "__main__":
    main()
