# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Inspectable request artifacts with binary attachments, without HTTP credentials."""

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .action_trace import ActionTrace

logger = logging.getLogger(__name__)


class RequestTrace:
    def __init__(self, directory: Path, action_log: Path | None = None) -> None:
        self.directory = directory
        self.index = 0
        self.action_trace = (
            ActionTrace(action_log, directory.parent.name, directory.name) if action_log else None
        )

    def _attachment(self, data: bytes, suffix: str) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        relative = f"assets/{digest}{suffix}"
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return {"$artifact": relative, "sha256": digest, "bytes": len(data)}

    def _encode(self, value: Any) -> Any:
        if isinstance(value, bytes):
            return {
                **self._attachment(value, ".jpg" if value.startswith(b"\xff\xd8") else ".bin"),
                "representation": "bytes",
            }
        if isinstance(value, str) and value.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")):
            return {
                **self._attachment(
                    base64.b64decode(value.split(",", 1)[1], validate=True),
                    ".png" if value.startswith("data:image/png;") else ".jpg",
                ),
                "representation": value.split(",", 1)[0],
            }
        if isinstance(value, dict):
            return {str(k): self._encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._encode(v) for v in value]
        return value

    def __call__(self, kind: str, value: Any) -> None:
        # Instrumentation must not turn a successfully issued action into a failed request.
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.index += 1
            path = self.directory / f"{self.index:03d}-{kind}.json"
            path.write_text(
                json.dumps(self._encode(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            logger.info(
                "artifact_saved",
                extra={"event": {"artifact_kind": kind, "artifact_path": str(path.resolve())}},
            )
            if kind == "decision-diagnostics" and isinstance(value, dict):
                report = value.get("report")
                if value.get("status") == "available" and isinstance(report, dict):
                    location = report.get("object_location")
                    if isinstance(location, dict):
                        logger.info("model_object_location", extra={"event": {"location": location}})
            elif kind == "planner-output-decision" and isinstance(value, dict):
                decision = value.get("decision")
                motion = decision.get("motion") if isinstance(decision, dict) else None
                waypoints = motion.get("waypoints") if isinstance(motion, dict) else None
                if (
                    isinstance(motion, dict)
                    and isinstance(waypoints, list)
                    and waypoints
                    and isinstance(waypoints[-1], dict)
                ):
                    logger.info(
                        "model_motion_target",
                        extra={
                            "event": {
                                "frame": motion.get("frame"),
                                "left_pose": waypoints[-1].get("left_pose"),
                                "right_pose": waypoints[-1].get("right_pose"),
                            }
                        },
                    )
            if self.action_trace is not None:
                try:
                    self.action_trace.record(kind, value, path)
                except (OSError, ValueError, TypeError) as exc:
                    logger.warning(
                        "action_trace_failed",
                        extra={"event": {"artifact_kind": kind, "exception_type": type(exc).__name__}},
                    )
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                "artifact_failed",
                extra={"event": {"artifact_kind": kind, "exception_type": type(exc).__name__}},
            )
