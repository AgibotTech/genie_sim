# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Structured logs without model credentials, observation images or raw responses."""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent2act.adapters.telemetry import record_artifact

_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def timed_event(logger: logging.Logger, phase: str) -> Iterator[None]:
    start = time.monotonic()
    outcome = "ok"
    exception_type = None
    logger.info("phase_started", extra={"event": {"phase": phase}})
    try:
        yield
    except BaseException as exc:
        outcome, exception_type = "failed", type(exc).__name__
        raise
    finally:
        logger.info(
            "phase_finished",
            extra={
                "event": {
                    "phase": phase,
                    "duration_s": round(time.monotonic() - start, 6),
                    "outcome": outcome,
                    "exception_type": exception_type,
                }
            },
        )


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        result: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        result.update(_context.get())
        result.update(getattr(record, "event", {}))
        return json.dumps(result, ensure_ascii=False, allow_nan=False, default=str)


class LoggingEventSink:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.logger = logging.getLogger("agent2act.events")

    def __call__(self, event: dict[str, Any]) -> None:
        record_artifact("runtime-event", event)
        self.logger.info("runtime_event", extra={"event": {**event, "session_id": self.session_id}})


def configure_logging(level: str = "INFO", format: str = "json", log_file: Path | None = None) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if format == "json" else HumanFormatter())
    if format == "text":
        handler.addFilter(HumanFilter())
    handlers: list[logging.Handler] = [handler]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        # Persist before publishing readiness to the terminal.
        handlers.insert(0, file_handler)
    logging.basicConfig(level=level, handlers=handlers, force=True)


class HumanFormatter(logging.Formatter):
    """One readable line per meaningful transition; the file keeps all JSON events."""

    @staticmethod
    def _xyz(pose: Any) -> str:
        if not isinstance(pose, (list, tuple)) or len(pose) < 3:
            return "unknown"
        if any(type(value) not in (int, float) for value in pose[:3]):
            return "unknown"
        return "[" + ",".join(f"{value:.3f}" for value in pose[:3]) + "]m"

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "event", {})
        context = {**_context.get(), **event}
        request = str(context.get("inference_id", ""))[:8]
        stamp = datetime.fromtimestamp(record.created).astimezone().strftime("%H:%M:%S")
        message = record.getMessage()
        if message == "runtime_event":
            message = str(event.get("status", message))
        fields = []
        for key in (
            "sequence",
            "planning_attempt",
            "attempt",
            "mode",
            "backend",
            "phase",
            "outcome",
            "duration_s",
            "http_status",
            "request_bytes",
            "issued_steps",
            "discarded_steps",
            "issued_duration_s",
            "error",
            "instruction",
            "reason",
            "delay_s",
            "timeout_s",
            "exception_type",
            "host",
            "port",
        ):
            if key in event:
                raw = event[key]
                if raw is None or (key == "instruction" and message == "policy_review_required"):
                    continue
                value = f"{raw:.2f}" if isinstance(raw, float) else str(raw)
                value = value.replace("\n", " ").replace("\r", " ")
                if key == "instruction" and len(value) > 160:
                    value = value[:157] + "..."
                fields.append(f"{key}={value}")
        if message == "llm_http_started":
            fields.append("waiting for planner reply; no new action issued this round")
        elif message == "rejected":
            fields.append("action rejected as a whole and not issued; replanning within budget")
        elif message == "issued":
            fields.append("handed to response encoding; execution and task success await later observations")
            endpoint = event.get("issued_endpoint", {})
            if event.get("kind") == "EEF_ABS" and isinstance(endpoint, dict):
                for arm in ("left", "right"):
                    if endpoint.get(f"{arm}_arm") is not None:
                        fields.append(f"{arm}_issued={self._xyz(endpoint[f'{arm}_arm'])}")
        elif message == "pi05_gripper_clipped":
            corrections = event.get("corrections", [])
            maximum = max((abs(item["delta"]) for item in corrections), default=0)
            fields.append(
                f"gripper bound correction on {len(corrections)} values, max {maximum:.6f} rad; pending action validation"
            )
        elif message == "policy_review_required":
            fields.append(f"issued {event.get('issued_chunks', 0)} chunks; reason: {event.get('reason', '')}")
        elif message == "model_object_location":
            location = event.get("location", {})
            if isinstance(location, dict):
                xyz = self._xyz([location.get(axis) for axis in ("x_m", "y_m", "z_m")])
                fields.append(
                    f"object={location.get('label', '?')} XYZ={xyz} "
                    f"frame={location.get('frame') or '?'} basis={location.get('basis') or '?'}"
                )
                if location.get("camera") and all(
                    type(location.get(axis)) in (int, float) for axis in ("u_norm", "v_norm")
                ):
                    fields.append(
                        f"image={location['camera']}({location['u_norm']:.0f},{location['v_norm']:.0f})/1000"
                    )
        elif message == "model_motion_target":
            fields.append(f"frame={event.get('frame') or '?'}")
            for arm in ("left", "right"):
                if event.get(f"{arm}_pose") is not None:
                    fields.append(f"{arm}_target={self._xyz(event[f'{arm}_pose'])}")
        if message == "observed_tracking":
            for arm in ("left", "right"):
                metrics = event.get(arm, {})
                if "translation_error_m" in metrics:
                    fields.append(f"{arm}_error_mm={metrics['translation_error_m'] * 1000:.1f}")
        return f"{stamp} {record.levelname:7} {request:8} {message} {' '.join(fields)}".rstrip()


class HumanFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        event = getattr(record, "event", {})
        if record.levelno >= logging.WARNING:
            return True
        if record.getMessage() in ("artifact_saved", "camera_input"):
            return False
        if record.getMessage() == "phase_started":
            return event.get("phase") == "inference_request"
        if record.getMessage() == "phase_finished":
            return event.get("phase") == "inference_request"
        if record.getMessage() == "runtime_event":
            return event.get("status") not in ("phase_started", "phase_finished", "planned")
        return True
