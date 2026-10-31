# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Append public planner evidence and action events for later recording alignment."""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_write_lock = threading.Lock()


def action_log_path(trace_dir: Path) -> Path:
    return trace_dir.with_name(trace_dir.name.removesuffix(".trace") + ".action-events.jsonl")


def response_metadata(data: dict[str, Any]) -> dict[str, Any]:
    output = data.get("output")
    summaries = [
        part["text"]
        for item in (output if isinstance(output, list) else [])
        if isinstance(item, dict) and item.get("type") == "reasoning"
        for part in (item["summary"] if isinstance(item.get("summary"), list) else [])
        if isinstance(part, dict)
        and part.get("type") == "summary_text"
        and isinstance(part.get("text"), str)
        and part["text"]
    ]
    return {
        "reported_model": data.get("model"),
        "reasoning_summary": summaries,
        "reasoning_summary_status": "available" if summaries else "not_provided",
    }


class ActionTrace:
    """Request-scoped identity; one append-only file shared by a run's requests."""

    def __init__(self, path: Path, session_id: str, inference_id: str) -> None:
        self.path = path
        self.context: dict[str, Any] = {"session_id": session_id, "inference_id": inference_id}

    def record(self, kind: str, value: Any, artifact_path: Path) -> None:
        if not isinstance(value, dict):
            return
        payload: dict[str, Any]
        if kind == "simulator-request":
            params = value.get("params", {})
            self.context["episode"] = [
                params.get("task_name"),
                params.get("episode_idx"),
                params.get("prompt"),
            ]
            payload = {
                "observation_timestamps": params.get("timestamps"),
                "episode_done": params.get("episode_done"),
                "video_alignment": "unmapped; observation timestamps are not video PTS or frame indices",
                "video_pts_s": None,
                "video_frame_index": None,
            }
        elif kind == "runtime-event":
            for key in ("episode", "sequence", "attempt"):
                if key in value:
                    self.context[key] = value[key]
            if value.get("status") not in (
                "planned",
                "issued",
                "rejected",
                "observed_tracking",
                "episode_done",
            ):
                return
            payload = value
        elif kind == "llm-request":
            self.context["requested_model"] = value.get("model")
            payload = {"requested_model": value.get("model")}
        elif kind == "llm-response":
            payload = response_metadata(value)
            requested, reported = self.context.get("requested_model"), payload["reported_model"]
            payload["model_identity_status"] = (
                "unreported"
                if not requested or not reported
                else "match"
                if requested == reported
                else "mismatch"
            )
        elif kind in (
            "observation-options",
            "gripper-center-execution-decision",
            "model-identity",
            "model_identity_retry",
            "model_identity_exhausted",
            "episode-interrupted",
            "diagnostic-input",
            "decision-diagnostics",
            "planner-output-decision",
            "reasoning-summary",
            "task-context",
        ):
            payload = value
        elif kind in ("wire-frame-conversion", "simulator-response"):
            # Full dense actions remain in their trace file; avoid duplicating image/action blobs.
            payload = {"meaning": "Encoded response is not playback acknowledgement"}
        else:
            # Index all committed artifacts, including tool rounds/errors and passive UI snapshots.
            # Large image/action payloads stay in the referenced file.
            payload = {key: value[key] for key in ("sequence", "planning_attempt") if key in value}
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            **self.context,
            "kind": kind,
            "artifact_path": str(artifact_path.resolve()),
            "data": payload,
        }
        text = json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
        with _write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(text)
