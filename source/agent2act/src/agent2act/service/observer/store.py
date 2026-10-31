# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Incremental trace index and on-demand, read-only request inspection."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..action_trace import action_log_path, response_metadata
from .views import build_views


def read_json(path: Path) -> Any:
    # A writer may still be filling this file. Retry on the next refresh.
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("artifact too large to preview")
    return json.loads(path.read_text(encoding="utf-8"))


class TraceStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.log = action_log_path(self.root)
        self.offset = 0
        self.identity: tuple[int, int] | None = None
        self.rows: dict[str, dict[str, Any]] = {}
        self.episodes: dict[str, dict[str, Any]] = {}
        self.sequences: dict[tuple[str, int], str] = {}
        self.generations: dict[str, int] = {}
        self.revision = 0
        self.malformed = 0

    def resolve(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("path outside selected run")
        return path

    def refresh(self) -> None:
        try:
            stat = self.log.stat()
        except FileNotFoundError:
            return
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.offset = 0
            self.rows.clear()
            self.episodes.clear()
            self.sequences.clear()
            self.generations.clear()
            self.revision += 1
        self.identity = identity
        with self.log.open("rb") as stream:
            stream.seek(self.offset)
            for _ in range(2000):
                line = stream.readline(8 * 1024 * 1024)
                if not line or not line.endswith(b"\n"):
                    break
                self.offset = stream.tell()
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("event must be an object")
                    self._event(event)
                except (ValueError, TypeError, KeyError):
                    self.malformed += 1
                self.revision += 1

    def _event(self, event: dict[str, Any]) -> None:
        session, inference = event.get("session_id"), event.get("inference_id")
        if not all(isinstance(v, str) and re.fullmatch(r"[\w-]+", v) for v in (session, inference)):
            return
        assert isinstance(session, str) and isinstance(inference, str)
        key = f"{session}/{inference}"
        kind, data = event.get("kind"), event.get("data", {})
        if not isinstance(data, dict):
            return
        if key not in self.rows:
            episode = event.get("episode")
            episode = episode if isinstance(episode, list) and len(episode) == 3 else [None, None, None]
            identity = [session, self.generations.get(session, 0), episode]
            group = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:24]
            self.episodes.setdefault(
                group,
                {
                    "id": group,
                    "session": session,
                    "task": episode[0],
                    "episode": episode[1],
                    "instruction": episode[2],
                    "started": event.get("timestamp"),
                    "count": 0,
                },
            )["count"] += 1
            self.rows[key] = {
                "id": key,
                "episode_id": group,
                "sequence": None,
                "status": "received",
                "started": event.get("timestamp"),
                "updated": event.get("timestamp"),
                "instruction": "",
                "revision": 0,
                "feedback": [],
            }
        row = self.rows[key]
        row["updated"] = event.get("timestamp")
        row["revision"] += 1
        sequence = data.get("sequence", event.get("sequence"))
        if type(sequence) is int:
            row["sequence"] = sequence
            self.sequences[(row["episode_id"], sequence)] = key
        if kind == "runtime-event":
            status = data.get("status")
            if status in ("planned", "issued", "rejected", "episode_done"):
                row["status"] = status
            decision = data.get("decision", data)
            row["instruction"] = decision.get("instruction", row["instruction"])
            if status == "observed_tracking" and type(data.get("previous_sequence")) is int:
                previous = self.sequences.get((row["episode_id"], data["previous_sequence"]))
                if previous and previous != key:
                    self.rows[previous]["feedback"].append(key)
                    self.rows[previous]["revision"] += 1
            if status == "episode_done":
                self.generations[session] = self.generations.get(session, 0) + 1
        elif kind in ("request-error", "episode-interrupted"):
            row["status"] = "failed"
        elif kind == "llm-request":
            row["status"] = "waiting_model"
        elif kind == "llm-response":
            row["status"] = "model_returned"
        elif kind == "simulator-response" and row["status"] not in ("issued", "episode_done"):
            row["status"] = "response_recorded"

    def catalog(self, episode: str | None, offset: int = 0) -> dict[str, Any]:
        self.refresh()
        groups = list(self.episodes.values())
        selected = episode or (groups[-1]["id"] if groups else None)
        rows = [r for r in self.rows.values() if r["episode_id"] == selected]
        return {
            "run": self.root.name,
            "revision": self.revision,
            "episodes": groups,
            "selected_episode": selected,
            "total": len(rows),
            "requests": list(reversed(rows))[offset : offset + 100],
            "malformed_events": self.malformed,
            "catching_up": self.log.exists() and self.log.stat().st_size > self.offset,
        }

    def artifacts(self, key: str) -> tuple[list[dict[str, Any]], list[str]]:
        folder = self.resolve(key)
        if key not in self.rows:
            raise KeyError(key)
        records, warnings = [], []
        for path in sorted(folder.glob("[0-9]*-*.json"), key=lambda p: int(p.name.split("-", 1)[0])):
            try:
                safe = self.resolve(str(path.relative_to(self.root)))
                value = read_json(safe)
            except (OSError, ValueError):
                warnings.append(f"{path.name}: incomplete or unreadable; retrying on refresh")
                continue
            records.append({"name": path.name, "kind": path.name.split("-", 1)[1][:-5], "value": value})
        return records, warnings

    def detail(self, key: str) -> dict[str, Any]:
        self.refresh()
        records, warnings = self.artifacts(key)
        row = self.rows[key]
        summaries = [r["value"] for r in records if r["kind"] == "reasoning-summary"]
        if not summaries:
            for record in records:
                if record["kind"] == "llm-response" and isinstance(record["value"], dict):
                    metadata = response_metadata(record["value"])
                    if metadata["reasoning_summary"]:
                        summaries.append({"source": record["name"], **metadata})
        feedback: list[dict[str, Any]] = []
        for next_key in dict.fromkeys(row["feedback"]):
            next_records, _ = self.artifacts(next_key)
            feedback.extend(
                {"source_request": next_key, "data": r["value"]}
                for r in next_records
                if r["kind"] == "runtime-event"
                and isinstance(r["value"], dict)
                and r["value"].get("status") == "observed_tracking"
                and r["value"].get("previous_sequence") == row["sequence"]
            )
        return {
            "request": row,
            "episode": self.episodes[row["episode_id"]],
            "warnings": warnings,
            "artifacts": [{"name": r["name"], "kind": r["kind"]} for r in records],
            "views": build_views(records, key, self.resolve),
            "events": [r["value"] for r in records if r["kind"] in ("runtime-event", "request-error")],
            "summaries": summaries,
            "diagnostics": [r["value"] for r in records if r["kind"] == "decision-diagnostics"],
            "feedback": feedback,
        }
