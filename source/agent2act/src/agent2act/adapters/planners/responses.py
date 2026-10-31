# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Multimodal Responses API planner adapter."""

import errno
import http.client
import json
import logging
import math
import os
import random
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from email.utils import parsedate_to_datetime
from typing import Any
from uuid import uuid4

from agent2act.adapters.telemetry import record_artifact
from agent2act.domain.errors import ModelIdentityError, TransientBackendError
from agent2act.domain.models import ContractError, Decision, Observation, vector

from .diagnostics import (
    DIAGNOSTIC_INSTRUCTIONS,
    evidence_warnings,
    extract_diagnostics,
    input_manifest,
    model_input_manifest,
    response_summary,
)
from .grasp_geometry import GraspGeometry
from .height_prior import HeightPrior
from .motion_context import (
    STRATEGY_VERSION,
    compact_motion_state,
    confirmation_wait_instructions,
    decision_history,
    fixed_motion_instructions,
    motion_only_state,
    tracking_summary,
    without_speed_context,
)
from .prompts import (
    FINAL_HOLD_INSTRUCTIONS,
    STATIC_RGB_CALIBRATION_INSTRUCTIONS,
    WAIST_INSTRUCTIONS,
    PromptProfile,
    planner_instructions,
    prompt_profile,
)
from .ranked_motion import ranked_motion_instructions
from .static_rgb_calibration import StaticRgbCalibration
from .tool_center import CENTER_INSTRUCTIONS, ToolCenter
from .vision import append_camera_inputs

logger = logging.getLogger(__name__)
# 52x/530 are gateway-to-origin failures (e.g. Cloudflare), transient like 502/504.
_RETRYABLE_HTTP = {400, 408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 530}


class InvalidEncryptedContentError(RuntimeError):
    """A rejected reasoning continuation; no final decision was returned."""


def _has_encrypted_reasoning(body: dict[str, Any]) -> bool:
    items = body.get("input")
    return isinstance(items, list) and any(
        isinstance(item, dict)
        and item.get("type") == "reasoning"
        and isinstance(item.get("encrypted_content"), str)
        and bool(item["encrypted_content"])
        for item in items
    )


def _normalize_motion_decision(data: Any, observation: Observation) -> Any:
    """Repair harmless omissions in model output at the adapter boundary.

    JSON ``null`` optional targets mean "hold"; a three-element position means "move there holding
    the observed orientation". Nothing else is invented."""
    if not isinstance(data, dict) or data.get("mode") != "motion":
        return data
    motion = data.get("motion")
    if not isinstance(motion, dict):
        return data
    field = "candidates" if "candidates" in motion else "waypoints"
    if not isinstance(motion.get(field), list):
        return data
    frame = motion.get("frame")
    if frame not in ("base_link", "arm_base_link"):
        return data
    result = dict(data)
    result_motion = dict(motion)
    waypoints = []
    for waypoint in motion[field]:
        if not isinstance(waypoint, dict):
            waypoints.append(waypoint)
            continue
        normalized = dict(waypoint)
        for arm in ("left", "right"):
            key = f"{arm}_pose"
            value = normalized.get(key, ...)
            if value is None:
                normalized.pop(key, None)
                continue
            if isinstance(value, list) and len(value) == 3:
                observed = observation.eef(arm, frame)
                normalized[key] = value + observed[3:]
        for key in ("grippers", "waist_yaw"):
            if normalized.get(key) is None:
                normalized.pop(key, None)
        waypoints.append(normalized)
    result_motion[field] = waypoints
    result["motion"] = result_motion
    return result


# Error codes an event stream can report after HTTP 200 that a retry may clear,
# like the 5xx/429 statuses of a non-streaming request.
_RETRYABLE_STREAM_ERRORS = {
    "server_error",
    "internal_error",
    "rate_limit_exceeded",
    "overloaded",
    "service_unavailable",
}


class StreamedServerError(http.client.IncompleteRead):
    """A transient server failure reported inside an event stream; retried like a 5xx response."""

    def __init__(self, code: str) -> None:
        super().__init__(b"", None)
        self.code = code

    def __str__(self) -> str:
        return f"LLM stream reported {self.code}"


def _transient_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, BaseException) and _transient_connection_error(exc.reason)
    if isinstance(exc, ssl.SSLCertVerificationError):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError, http.client.IncompleteRead, ssl.SSLEOFError)):
        return True
    if isinstance(exc, socket.gaierror):
        return exc.errno == socket.EAI_AGAIN
    return isinstance(exc, OSError) and exc.errno in {errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ETIMEDOUT}


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


class ResponsesPlanner:
    """Multimodal Responses HTTP API, with injectable transport for offline tests."""

    def __init__(
        self,
        model: str,
        endpoint: str = "https://api.openai.com/v1/responses",
        api_key_env: str | None = "OPENAI_API_KEY",
        timeout_s: float = 60.0,
        max_output_tokens: int = 4096,
        history_frames: int = 2,
        post: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        retry_max_delay_s: float = 10.0,
        retry_budget_s: float | None = None,
        invalid_encrypted_content_retries: int = 0,
        model_identity_policy: str = "warn",
        model_identity_max_retries: int = 2,
        user_agent: str = "Agent2Act/0.1",
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        stream: bool = False,
        diagnostic_mode: bool = False,
        image_max_side: int | None = None,
        image_quality: int = 85,
        use_depth: bool = False,
        send_raw_depth_image: bool = True,
        json_mode: bool = False,
        compact_motion_context: bool = False,
        current_observation_only: bool = False,
        concise_tracking_feedback: bool = False,
        decision_history_turns: int = 0,
        embodiment: str | None = None,
        pose_reference: str | None = None,
        gripper_open_closed: dict[str, list[float]] | None = None,
        gripper_center_offset_m: dict[str, list[float]] | None = None,
        use_gripper_center: bool | None = None,
        use_height_prior: bool = False,
        height_calibration: dict[str, Any] | None = None,
        use_grasp_geometry: bool = False,
        grasp_geometry_calibration: dict[str, Any] | None = None,
        rgb_camera_calibration: dict[str, Any] | None = None,
    ) -> None:
        if (
            type(invalid_encrypted_content_retries) is not int
            or not 0 <= invalid_encrypted_content_retries <= 10
        ):
            raise ContractError("invalid_encrypted_content_retries must be an integer between 0 and 10")
        self.invalid_encrypted_content_retries = invalid_encrypted_content_retries
        self._recovery_deadline: float | None = None
        if reasoning_summary is not None and reasoning_summary not in ("auto", "concise", "detailed"):
            raise ContractError("reasoning_summary must be auto, concise, detailed or null")
        if (
            not isinstance(user_agent, str)
            or not user_agent.strip()
            or "\r" in user_agent
            or "\n" in user_agent
        ):
            raise ContractError("user_agent must be a nonempty single-line string")
        self.user_agent = user_agent.strip()
        if type(stream) is not bool:
            raise ContractError("stream must be boolean")
        # Server-sent events keep a long reasoning request alive through gateways
        # that drop idle connections; the completed response is reassembled.
        self.stream = stream
        if type(diagnostic_mode) is not bool:
            raise ContractError("diagnostic_mode must be boolean")
        self.reasoning_summary = reasoning_summary
        self.diagnostic_mode = diagnostic_mode
        if use_gripper_center is not None and type(use_gripper_center) is not bool:
            raise ContractError("use_gripper_center must be boolean or null")
        if use_gripper_center is True and gripper_center_offset_m is None:
            raise ContractError("use_gripper_center requires calibrated gripper_center_offset_m")
        self._tool_center = (
            ToolCenter(gripper_center_offset_m)
            if gripper_center_offset_m is not None and use_gripper_center is not False
            else None
        )
        if type(use_height_prior) is not bool:
            raise ContractError("use_height_prior must be boolean")
        self._height_prior = HeightPrior(height_calibration) if use_height_prior else None
        if type(use_grasp_geometry) is not bool:
            raise ContractError("use_grasp_geometry must be boolean")
        if use_grasp_geometry and self._tool_center is None:
            raise ContractError("use_grasp_geometry requires the gripper-center TCP interface")
        self._grasp_geometry = GraspGeometry(grasp_geometry_calibration) if use_grasp_geometry else None
        self._static_rgb_calibration = (
            StaticRgbCalibration(rgb_camera_calibration) if rgb_camera_calibration is not None else None
        )
        if embodiment is not None and (not isinstance(embodiment, str) or not embodiment.strip()):
            raise ContractError("embodiment must be a nonempty name or null")
        self.embodiment = embodiment
        if pose_reference is not None:
            if not isinstance(pose_reference, str) or not pose_reference.strip():
                raise ContractError("pose_reference must be a nonempty string or null")
            if self._tool_center is None:
                raise ContractError("pose_reference describes the gripper-center TCP interface")
        self._pose_reference = pose_reference or CENTER_INSTRUCTIONS
        if type(compact_motion_context) is not bool:
            raise ContractError("compact_motion_context must be boolean")
        self.compact_motion_context = compact_motion_context
        if type(current_observation_only) is not bool:
            raise ContractError("current_observation_only must be boolean")
        self.current_observation_only = current_observation_only
        if type(concise_tracking_feedback) is not bool:
            raise ContractError("concise_tracking_feedback must be boolean")
        self.concise_tracking_feedback = concise_tracking_feedback
        if type(decision_history_turns) is not int or not 0 <= decision_history_turns <= 32:
            raise ContractError("decision_history_turns must be an integer between 0 and 32")
        self.decision_history_turns = decision_history_turns
        self.gripper_open_closed = {}
        if gripper_open_closed is not None:
            if not isinstance(gripper_open_closed, dict) or set(gripper_open_closed) != {"left", "right"}:
                raise ContractError("gripper references require left and right")
            self.gripper_open_closed = {
                arm: vector(value, 2, "open/closed commands") for arm, value in gripper_open_closed.items()
            }
            if any(opened == closed for opened, closed in self.gripper_open_closed.values()):
                raise ContractError("open and closed commands must differ")
        if type(json_mode) is not bool:
            raise ContractError("json_mode must be boolean")
        if model_identity_policy not in ("warn", "require_match"):
            raise ContractError("model_identity_policy must be warn or require_match")
        if type(model_identity_max_retries) is not int or not 0 <= model_identity_max_retries <= 5:
            raise ContractError("model_identity_max_retries must be an integer in [0,5]")
        self.model_identity_policy = model_identity_policy
        self.model_identity_max_retries = model_identity_max_retries
        self._identity_deadline: float | None = None
        if not isinstance(model, str) or not model or model.startswith("YOUR_"):
            raise ContractError("configure an available multimodal LLM model")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
            or type(max_output_tokens) is not int
            or max_output_tokens < 1
            or type(history_frames) is not int
            or not 0 <= history_frames <= 8
        ):
            raise ContractError("invalid planner timeout, token budget or history_frames")
        if (
            type(max_retries) is not int
            or not 0 <= max_retries <= 99
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in (retry_backoff_s, retry_max_delay_s)
            )
            or retry_backoff_s > retry_max_delay_s
            or retry_max_delay_s > 60
        ):
            raise ContractError("invalid planner retry settings")
        if retry_budget_s is not None and (
            isinstance(retry_budget_s, bool)
            or not isinstance(retry_budget_s, (int, float))
            or not math.isfinite(retry_budget_s)
            or retry_budget_s <= 0
        ):
            raise ContractError("retry_budget_s must be a positive finite number or null")
        if reasoning_effort is not None and reasoning_effort not in ("low", "medium", "high", "xhigh"):
            raise ContractError("reasoning_effort must be low, medium, high, xhigh or null")
        if image_max_side is not None and (
            type(image_max_side) is not int or not 128 <= image_max_side <= 2048
        ):
            raise ContractError("image_max_side must be null or an integer in [128,2048]")
        if (
            type(image_quality) is not int
            or not 40 <= image_quality <= 95
            or type(use_depth) is not bool
            or type(send_raw_depth_image) is not bool
        ):
            raise ContractError("invalid image_quality or use_depth")
        depth_queries_enabled = getattr(self, "enable_raw_depth_query", False) or getattr(
            self, "enable_depth_measurement", False
        )
        if use_depth and not send_raw_depth_image and not depth_queries_enabled:
            raise ContractError("omitting raw depth images requires an enabled depth query tool")
        self.image_max_side, self.image_quality, self.use_depth = image_max_side, image_quality, use_depth
        self.send_raw_depth_image = send_raw_depth_image
        self.reasoning_effort = reasoning_effort
        self.json_mode = json_mode
        self.model, self.endpoint, self.api_key_env = model, endpoint, api_key_env
        self.timeout_s, self.max_output_tokens = timeout_s, max_output_tokens
        self.history_frames = history_frames
        self.post = post or self._post
        self.max_retries = max_retries
        self.retry_backoff_s, self.retry_max_delay_s = retry_backoff_s, retry_max_delay_s
        self.retry_budget_s = retry_budget_s
        self._closed = threading.Event()
        self._call_context: dict[str, Any] = {}
        self._ranked_motion_candidates = False
        self._ranked_motion_candidate_count = 5
        self._confirmation_wait_instructions = ""
        self._fixed_motion_steps: int | None = None
        self._unrestricted_fixed_motion = False
        self._prompt_profile: PromptProfile = "review"
        self._waist_control_active = False
        self._final_hold_active = False
        self._episode_start_body: list[float] | None = None
        self._visual_sequence: int | None = None
        self._last_visual: list[dict[str, Any]] = []
        self._visual_reference: list[dict[str, Any]] = []
        self._visual_reference_frames = 0
        self._visual_history: list[list[dict[str, Any]]] = []

    def reset(self) -> None:
        self._prompt_profile = "review"
        self._waist_control_active = False
        self._final_hold_active = False
        self._episode_start_body = None
        self._ranked_motion_candidates = False
        self._ranked_motion_candidate_count = 5
        self._confirmation_wait_instructions = ""
        self._fixed_motion_steps = None
        self._unrestricted_fixed_motion = False
        self._visual_sequence = None
        self._last_visual = []
        self._visual_reference = []
        self._visual_reference_frames = 0
        self._visual_history = []

    def _send_http(self, request: urllib.request.Request, event: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        outcome, exception_type = "ok", None
        details: dict[str, Any] = {}
        logger.info("llm_http_started", extra={"event": event})
        try:
            with urllib.request.urlopen(request, timeout=event["timeout_s"]) as response:
                headers_received = time.monotonic()
                details["headers_wait_s"] = round(headers_received - start, 6)
                details["http_status"] = getattr(response, "status", None)
                headers = getattr(response, "headers", None) if self.stream else None
                if headers is not None and "text/event-stream" in (headers.get("Content-Type") or ""):
                    result, size = self._read_event_stream(response)
                    details["streamed"] = True
                else:
                    raw = response.read(16 * 1024 * 1024 + 1)
                    size = len(raw)
                    if size > 16 * 1024 * 1024:
                        raise ContractError("LLM response exceeds size limit")
                    result = json.loads(raw)
                details["body_read_s"] = round(time.monotonic() - headers_received, 6)
                details["response_bytes"] = size
                if self._closed.is_set():
                    raise RuntimeError("LLM planner is closed")
                # Only whitelisted numeric usage; never record raw model output.
                if isinstance(result, dict):
                    reasoning = result.get("reasoning")
                    if isinstance(reasoning, dict) and reasoning.get("effort") in (
                        "low",
                        "medium",
                        "high",
                        "xhigh",
                    ):
                        details["reported_reasoning_effort"] = reasoning["effort"]
                    usage = result.get("usage")
                    if isinstance(usage, dict):
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "total_tokens",
                            "prompt_tokens",
                            "completion_tokens",
                        ):
                            if type(usage.get(key)) is int:
                                details[key] = usage[key]
                        output_details = usage.get("output_tokens_details") or usage.get(
                            "completion_tokens_details"
                        )
                        if (
                            isinstance(output_details, dict)
                            and type(output_details.get("reasoning_tokens")) is int
                        ):
                            details["reasoning_tokens"] = output_details["reasoning_tokens"]
                return result
        except BaseException as exc:
            outcome, exception_type = "failed", type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                details["http_status"] = exc.code
                code = self._record_http_error(request, exc, event)
                if exc.code == 400 and code == "invalid_encrypted_content":
                    exc.close()
                    raise InvalidEncryptedContentError(
                        "LLM HTTP status 400: invalid_encrypted_content"
                    ) from exc
            raise
        finally:
            logger.info(
                "llm_http_finished",
                extra={
                    "event": {
                        **event,
                        **details,
                        "outcome": outcome,
                        "exception_type": exception_type,
                        "duration_s": round(time.monotonic() - start, 6),
                    }
                },
            )

    def _read_event_stream(self, response: Any) -> tuple[dict[str, Any], int]:
        """Return the final response object of a Responses API event stream."""
        size = 0
        data: list[str] = []
        while True:
            line = response.readline(16 * 1024 * 1024 + 1)
            size += len(line)
            if size > 16 * 1024 * 1024:
                raise ContractError("LLM response exceeds size limit")
            if self._closed.is_set():
                raise RuntimeError("LLM planner is closed")
            if not line:
                raise http.client.IncompleteRead(b"", None)
            text = line.decode().rstrip("\r\n")
            if text.startswith("data:"):
                data.append(text[5:].strip())
                continue
            if text or not data:
                continue
            payload = "\n".join(data)
            data = []
            if payload == "[DONE]":
                raise http.client.IncompleteRead(b"", None)
            event = json.loads(payload)
            kind = event.get("type") if isinstance(event, dict) else None
            if kind in ("response.completed", "response.incomplete", "response.failed"):
                final = event.get("response")
                if not isinstance(final, dict):
                    raise ContractError("LLM event stream ended without a response object")
                if kind == "response.failed":
                    self._raise_if_retryable_stream_error(final.get("error"))
                return final, size
            if kind == "error":
                error = event.get("error") if isinstance(event.get("error"), dict) else event
                self._raise_if_retryable_stream_error(error)
                # Same shape as a failed non-streaming response; _response_text rejects it.
                return {"status": "failed", "error": error}, size

    @staticmethod
    def _raise_if_retryable_stream_error(error: Any) -> None:
        if isinstance(error, dict):
            for key in ("code", "type"):
                value = error.get(key)
                if isinstance(value, str) and value in _RETRYABLE_STREAM_ERRORS:
                    raise StreamedServerError(value)

    # Credential headers per provider. Gemini authenticates with x-goog-api-key and never
    # sends Authorization, so a redaction that reads only Authorization misses it.
    _CREDENTIAL_HEADERS = ("authorization", "x-api-key", "x-goog-api-key")

    @classmethod
    def _credential_secrets(cls, request: urllib.request.Request) -> list[str]:
        """Collect every credential value a provider might echo back, longest first.

        Uses header_items() because Request normalizes header names. Only the token is redacted, so an
        echoed "Bearer <key>" keeps its scheme word."""
        secrets: list[str] = []
        for name, value in request.header_items():
            if name.lower() not in cls._CREDENTIAL_HEADERS or not isinstance(value, str):
                continue
            token = value.removeprefix("Bearer ").strip()
            if token and token not in secrets:
                secrets.append(token)
        return sorted(secrets, key=len, reverse=True)

    def _record_http_error(
        self, request: urllib.request.Request, error: urllib.error.HTTPError, event: dict[str, Any]
    ) -> str | None:
        """Keep bounded diagnostics and return the structured provider error code."""
        artifact: dict[str, Any] = {**event, "http_status": error.code}
        try:
            raw = error.read(16 * 1024 + 1)
            artifact["body_truncated"] = len(raw) > 16 * 1024
            if not artifact["body_truncated"]:
                body = json.loads(raw)
                detail = body.get("error") if isinstance(body, dict) else None
                if isinstance(detail, dict):
                    secrets = self._credential_secrets(request)
                    fields = {}
                    for key in ("type", "code", "param", "message"):
                        value = detail.get(key)
                        if isinstance(value, str):
                            # Some providers echo credentials in error messages.
                            for secret in secrets:
                                value = value.replace(secret, "[REDACTED]")
                            fields[key] = value[:4096]
                    artifact["error"] = fields
                    setattr(error, "agent2act_error_code", fields.get("code"))
        except (ValueError, OSError, http.client.HTTPException) as exc:
            artifact["diagnostic_error_type"] = type(exc).__name__
        record_artifact("llm-http-error", artifact)
        code = artifact.get("error", {}).get("code")
        return code if isinstance(code, str) else None

    def _http_headers(self, key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": self.user_agent}
        if key:
            headers["Authorization"] = "Bearer " + key
        return headers

    def _http_request(self, endpoint: str, payload: bytes, headers: dict[str, str]) -> urllib.request.Request:
        """Build the request with credential headers bound to this endpoint.

        Credentials are marked unredirected so urllib's redirect handler never copies them to another
        scheme, host or port."""
        request = urllib.request.Request(
            endpoint,
            payload,
            {key: value for key, value in headers.items() if key.lower() not in self._CREDENTIAL_HEADERS},
        )
        for key, value in headers.items():
            if key.lower() in self._CREDENTIAL_HEADERS and value:
                request.add_unredirected_header(key, value)
        return request

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if self.api_key_env and not key:
            raise ContractError(f"missing API key environment variable {self.api_key_env}")
        headers = self._http_headers(key)
        payload = json.dumps(body, allow_nan=False).encode()
        request = self._http_request(self.endpoint, payload, headers)
        request_id = uuid4().hex
        deadline = None if self.retry_budget_s is None else time.monotonic() + self.retry_budget_s
        if self._identity_deadline is not None:
            deadline = self._identity_deadline if deadline is None else min(deadline, self._identity_deadline)
        if self._recovery_deadline is not None:
            deadline = self._recovery_deadline if deadline is None else min(deadline, self._recovery_deadline)
        encrypted_failures = transient_failures = 0
        has_encrypted_reasoning = _has_encrypted_reasoning(body)
        continuation_fallback_enabled = not (
            self.invalid_encrypted_content_retries or getattr(self, "invalid_encrypted_content_replans", 0)
        )
        continuation_fallback_used = False
        max_attempts = self.max_retries + self.invalid_encrypted_content_retries + 1
        for attempt in range(max_attempts):
            if self._closed.is_set():
                raise RuntimeError("LLM planner is closed")
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                logger.error(
                    "llm_retry_budget_exhausted",
                    extra={"event": {**self._call_context, "request_id": request_id, "attempts": attempt}},
                )
                raise TransientBackendError("LLM retry budget exhausted")
            event = {
                **self._call_context,
                "request_id": request_id,
                "attempt": attempt + 1,
                "max_attempts": max_attempts,
                "request_bytes": len(payload),
                "timeout_s": self.timeout_s if remaining is None else min(self.timeout_s, remaining),
                "requested_reasoning_effort": self.reasoning_effort,
                "requested_model": self.model,
            }
            try:
                try:
                    return self._send_http(request, event)
                except InvalidEncryptedContentError:
                    if (
                        not continuation_fallback_enabled
                        or continuation_fallback_used
                        or not has_encrypted_reasoning
                    ):
                        raise
                    # One continuation fallback unless identical retries/fresh replans are selected.
                    continuation_fallback_used = True
                    items = body["input"]
                    body = {
                        **body,
                        "input": [
                            item
                            for item in items
                            if not (
                                isinstance(item, dict)
                                and item.get("type") == "reasoning"
                                and item.get("encrypted_content")
                            )
                        ],
                    }
                    has_encrypted_reasoning = False
                    payload = json.dumps(body, allow_nan=False).encode()
                    request = self._http_request(self.endpoint, payload, headers)
                    if self._closed.is_set():
                        raise RuntimeError("LLM planner is closed")
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TransientBackendError("LLM retry budget exhausted")
                    event = {
                        **event,
                        "request_bytes": len(payload),
                        "timeout_s": self.timeout_s if remaining is None else min(self.timeout_s, remaining),
                    }
                    record_artifact(
                        "llm-encrypted-content-recovery",
                        {
                            **event,
                            "removed_reasoning_items": len(items) - len(body["input"]),
                            "action_issued": False,
                        },
                    )
                    logger.warning("llm_encrypted_content_recovery", extra={"event": event})
                    return self._send_http(request, event)
            except (
                InvalidEncryptedContentError,
                urllib.error.URLError,
                OSError,
                http.client.HTTPException,
            ) as exc:
                encrypted_error = isinstance(exc, InvalidEncryptedContentError)
                status = (
                    400 if encrypted_error else exc.code if isinstance(exc, urllib.error.HTTPError) else None
                )
                retry_after = None
                if encrypted_error:
                    retryable = has_encrypted_reasoning
                    retries_exhausted = encrypted_failures >= self.invalid_encrypted_content_retries
                    encrypted_failures += 1
                elif isinstance(exc, urllib.error.HTTPError):
                    retryable = status in _RETRYABLE_HTTP
                    retry_after = _retry_after(exc.headers.get("Retry-After"))
                    exc.close()
                    retries_exhausted = transient_failures >= self.max_retries
                    transient_failures += 1
                else:
                    retryable = _transient_connection_error(exc)
                    retries_exhausted = transient_failures >= self.max_retries
                    transient_failures += 1
                delay = min(self.retry_max_delay_s, self.retry_backoff_s * 2**attempt)
                delay *= random.uniform(0.75, 1.0)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                event = {
                    **event,
                    "request_id": request_id,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "http_status": status,
                    "exception_type": type(exc).__name__,
                }
                if encrypted_error:
                    event.update(
                        error_code="invalid_encrypted_content", encrypted_content_attempt=encrypted_failures
                    )
                budget_exhausted = deadline is not None and time.monotonic() + delay >= deadline
                if budget_exhausted:
                    event["retry_budget_exhausted"] = True
                if not retryable or retries_exhausted or delay > self.retry_max_delay_s or budget_exhausted:
                    logger.error("llm_request_failed", extra={"event": event})
                    message = f"LLM HTTP status {status}" if status is not None else "LLM connection failed"
                    if encrypted_error and has_encrypted_reasoning:
                        if budget_exhausted:
                            raise TransientBackendError("LLM retry budget exhausted") from exc
                        raise
                    error_type = TransientBackendError if retryable else RuntimeError
                    raise error_type(message) from exc
                logger.warning("llm_request_retry", extra={"event": {**event, "delay_s": round(delay, 3)}})
                wait_start = time.monotonic()
                cancelled = self._closed.wait(delay)
                logger.info(
                    "llm_backoff_finished",
                    extra={
                        "event": {
                            **event,
                            "duration_s": round(time.monotonic() - wait_start, 6),
                            "outcome": "cancelled" if cancelled else "ok",
                        }
                    },
                )
                if cancelled:
                    raise RuntimeError("LLM planner is closed") from exc
        raise AssertionError("unreachable")

    def _append_images(
        self,
        content: list[dict[str, Any]],
        images: dict[str, Any],
        label: str,
        static_rgb_calibration: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        append_camera_inputs(
            content,
            images,
            label,
            self.image_max_side,
            self.image_quality,
            self.use_depth,
            static_rgb_calibration,
            self.send_raw_depth_image,
        )

    def _select_visual_history(
        self, observation: Observation, context: dict[str, Any]
    ) -> list[list[dict[str, Any]]]:
        return self._visual_history

    def decide(self, observation: Observation, context: dict[str, Any]) -> Decision:
        if self._closed.is_set():
            raise RuntimeError("LLM planner is closed")
        self._prompt_profile = prompt_profile(context)
        self._confirmation_wait_instructions = (
            confirmation_wait_instructions(context) if self._prompt_profile == "motion" else ""
        )
        self._ranked_motion_candidates = bool(
            context.get("execution", {})
            .get("executors", {})
            .get("motion", {})
            .get("ranked_motion_candidates")
        )
        self._ranked_motion_candidate_count = (
            context.get("execution", {})
            .get("executors", {})
            .get("motion", {})
            .get("ranked_motion_candidates", 5)
            if self._ranked_motion_candidates
            else 5
        )
        self._waist_control_active = self._prompt_profile == "motion" and bool(
            context.get("execution", {}).get("executors", {}).get("motion", {}).get("waist_control")
        )
        self._final_hold_active = bool(context.get("execution", {}).get("complete_hold_steps"))
        self._fixed_motion_steps = (
            context.get("execution", {})
            .get("executors", {})
            .get("motion", {})
            .get("compiler", {})
            .get("fixed_steps")
            if self._prompt_profile == "motion"
            else None
        )
        compiler = context.get("execution", {}).get("executors", {}).get("motion", {}).get("compiler", {})
        self._unrestricted_fixed_motion = (
            self._fixed_motion_steps is not None
            and compiler.get("retime_to_limits") is False
            and compiler.get("speed_limits_enforced") is False
        )
        if self._tool_center:
            if self._prompt_profile != "motion":
                raise ContractError(
                    "gripper center interface requires motion-only planning without proposal review"
                )
            context = self._tool_center.context(context)
        end_pose = observation.end_pose
        frame = context.get("execution", {}).get("motion_frame")
        if frame is not None:
            # Present one coordinate system, rather than asking the model to mentally
            # transform poses between differently oriented robot links.
            for arm in ("left", "right"):
                observation.eef(arm, frame)
            end_pose = {frame: observation.end_pose[frame]}
        if self._tool_center:
            end_pose = self._tool_center.end_pose(end_pose)
        state: dict[str, Any] = {
            "instruction": observation.instruction,
            "robot_type": observation.robot_type,
            "states": {**observation.state_groups, "end_pose": end_pose},
            "context": context,
        }
        current_only = self.current_observation_only and self._prompt_profile == "motion"
        if (self.compact_motion_context or current_only) and self._prompt_profile == "motion":
            state = compact_motion_state(state)
        if current_only:
            state["context"].pop("recent_events", None)
            state["context"].pop("tracking_feedback", None)
            # Rejections of this observation's attempt are correction feedback, not episode history.
            errors = [
                event
                for event in context.get("recent_events", [])
                if event.get("status") == "rejected" and event.get("sequence") == context.get("sequence")
            ]
            if errors:
                state["context"]["current_rejections"] = deepcopy(errors)
        if self.concise_tracking_feedback and self._prompt_profile == "motion":
            summary = tracking_summary(context.get("tracking_feedback", {}))
            state["context"] = dict(state["context"])
            state["context"].pop("tracking_feedback", None)
            if summary:
                state["context"]["tracking_feedback"] = summary
        if self._prompt_profile == "motion":
            state = motion_only_state(state)
            if self.decision_history_turns:
                # Text-only memory of issued decisions survives current_observation_only.
                history = decision_history(context.get("recent_events", []), self.decision_history_turns)
                if history:
                    state["context"]["decision_history"] = history
        waist = state["context"].get("execution", {}).get("waist_control")
        if waist is not None:
            # The runtime resets the planner on every episode, so the first body is the start posture.
            if self._episode_start_body is None:
                self._episode_start_body = list(observation.body)
            waist["episode_start_rad"] = self._episode_start_body[waist["state_index"]]
            commands = context.get("body_commands")
            if isinstance(commands, list) and len(commands) > waist["state_index"]:
                waist["commanded_rad"] = commands[waist["state_index"]]
        if self._unrestricted_fixed_motion:
            state = without_speed_context(state)
        if self._tool_center:
            state["pose_reference"] = self._pose_reference
        if self._height_prior:
            state["height_prior"] = self._height_prior.context(
                frame, context.get("execution", {}).get("motion_frame_axes")
            )
        if self._grasp_geometry:
            table_z = state["height_prior"]["table_surface_z_m"] if self._height_prior else None
            state["grasp_geometry"] = self._grasp_geometry.context(table_z)
        record_artifact(
            "observation-options",
            {
                "depth_enabled": self.use_depth,
                "raw_depth_image_attached": self.use_depth and self.send_raw_depth_image,
                "gripper_center_enabled": self._tool_center is not None,
                "height_prior_enabled": self._height_prior is not None,
                "height_calibration": self._height_prior.calibration if self._height_prior else None,
                "grasp_geometry_enabled": self._grasp_geometry is not None,
                "grasp_geometry_calibration": (
                    self._grasp_geometry.calibration if self._grasp_geometry else None
                ),
                "strategy_version": STRATEGY_VERSION,
                "embodiment": self.embodiment,
                "pose_reference": self._pose_reference if self._tool_center else None,
            },
        )
        if self.gripper_open_closed:
            state["gripper_command_reference"] = {
                arm: {"fully_open_command": pair[0], "fully_closed_command": pair[1]}
                for arm, pair in self.gripper_open_closed.items()
            }
        content: list[dict[str, Any]] = [{"type": "input_text", "text": json.dumps(state, allow_nan=False)}]
        if self.json_mode:
            # Compatible endpoints may validate input separately from instructions.
            content.append({"type": "input_text", "text": "Return one valid JSON decision object."})
        current: list[dict[str, Any]] = []
        self._append_images(
            current,
            observation.images,
            "current",
            self._static_rgb_calibration.for_observation(observation)
            if self._static_rgb_calibration
            else None,
        )
        # Passive snapshot of already prepared inputs; never appended to the model prompt.
        record_artifact(
            "observer-input",
            {
                "sequence": context.get("sequence"),
                "planning_attempt": context.get("planning_attempt"),
                "content": current,
                "end_pose": end_pose,
                "gripper_center_offset_m": self._tool_center.offsets if self._tool_center else None,
                "reference": "gripper_center" if self._tool_center else "wrist",
            },
        )
        remaining = 0 if current_only else self.history_frames
        sequence = context.get("sequence")
        if remaining and type(sequence) is int:
            if sequence != self._visual_sequence:
                if self._last_visual:
                    self._visual_history.append(self._last_visual)
                    del self._visual_history[: -self.history_frames]
                selected = self._select_visual_history(observation, context)
                self._visual_reference = [part for frame in selected for part in frame]
                self._visual_reference_frames = len(selected)
                self._visual_sequence = sequence
                self._last_visual = []
                label = f"previous planner observation sequence={sequence} (before subsequent actions)"
                if current:
                    # Serialize now so later observations or retries cannot rewrite
                    # the measured robot state paired with these reference images.
                    reference_state = {
                        "sequence": sequence,
                        "states": state["states"],
                        "meaning": (
                            "Measured robot state at these previous planner images, before subsequent "
                            "actions. Not an issued target, object pose, or grasp-center calibration."
                        ),
                    }
                    if "observed_grippers_in_command_units" in context:
                        reference_state["observed_grippers_in_command_units"] = context[
                            "observed_grippers_in_command_units"
                        ]
                    self._last_visual.append(
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {"previous_robot_observation": reference_state}, allow_nan=False
                            ),
                        }
                    )
                for part in current:
                    cached = dict(part)
                    if cached["type"] == "input_text":
                        text = cached["text"]
                        if text.startswith("current camera="):
                            cached["text"] = text.replace("current", label, 1)
                        else:
                            depth = json.loads(text)
                            depth["image_role"] = label
                            cached["text"] = json.dumps(depth)
                    self._last_visual.append(cached)
            if self._visual_reference:
                content.extend(deepcopy(self._visual_reference))
                remaining -= self._visual_reference_frames
        frames = observation.history
        if remaining and isinstance(frames, list):
            selected_frames = frames[-remaining:]
            for index, frame in enumerate(selected_frames):
                if isinstance(frame, dict):
                    self._append_images(
                        content,
                        frame.get("images", frame),
                        f"previous chunk history {index + 1}/{len(selected_frames)} "
                        "(in-chunk; may be near the current frame, not a before-action reference)",
                    )
        content.extend(current)
        self._call_context = {
            "sequence": context.get("sequence"),
            "planning_attempt": context.get("planning_attempt"),
        }
        if self.diagnostic_mode:
            manifest = input_manifest(content)
            record_artifact(
                "diagnostic-input",
                {
                    **self._call_context,
                    "input_manifest": manifest,
                    "depth_enabled": self.use_depth,
                    "raw_depth_image_attached": self.use_depth and self.send_raw_depth_image,
                    "gripper_center_enabled": self._tool_center is not None,
                    "height_prior_enabled": self._height_prior is not None,
                    "grasp_geometry_enabled": self._grasp_geometry is not None,
                    "meaning": "Initial shared model input after visual preprocessing. Exact transport requests "
                    "and subsequent tool rounds are in their own trace artifacts. IDs refer to this input list.",
                },
            )
            content.append(
                {"type": "input_text", "text": json.dumps({"input_manifest": model_input_manifest(manifest)})}
            )
        body = self._request_body(content)
        response = self._checked_decision_response(body, observation)
        text = self._response_text(response)
        try:
            payload = json.loads(text)
            if self.diagnostic_mode:
                payload, report = extract_diagnostics(payload)
                if report["status"] == "available":
                    report["input_consistency_warnings"] = evidence_warnings(report["report"], manifest)
                record_artifact("decision-diagnostics", {**self._call_context, **report})
            if self._fixed_motion_steps is None:
                payload = _normalize_motion_decision(payload, observation)
            decision = Decision.from_dict(payload)
            record_artifact("planner-output-decision", {**self._call_context, "decision": asdict(decision)})
            if self._tool_center:
                if decision.mode != "motion" or decision.assessment is not None:
                    raise ContractError(
                        "gripper center interface accepts motion decisions without assessment"
                    )
                decision = self._tool_center.decision(decision)
                record_artifact("gripper-center-execution-decision", asdict(decision))
            return decision
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"LLM must return one valid JSON decision object: {exc.msg} "
                f"at line {exc.lineno} column {exc.colno}"
            ) from exc

    def _checked_decision_response(self, body: dict[str, Any], observation: Observation) -> dict[str, Any]:
        strict = self.model_identity_policy == "require_match"
        max_attempts = self.model_identity_max_retries + 1 if strict else 1
        started = time.monotonic()
        self._identity_deadline = (
            started + self.retry_budget_s if strict and self.retry_budget_s is not None else None
        )
        attempts = 0
        try:
            for attempt in range(max_attempts):
                if self._closed.is_set():
                    raise RuntimeError("LLM planner is closed")
                self._call_context["model_identity_attempt"] = attempt + 1
                if self._identity_deadline is not None and time.monotonic() >= self._identity_deadline:
                    break
                # Reuse the identical observation/prompt; rejected output never becomes model input.
                record_artifact("llm-request", body)
                attempts += 1
                response = self._decision_response(deepcopy(body), observation)
                record_artifact("llm-response", response)
                if self.reasoning_summary is not None and not response.get("_summary_recorded"):
                    record_artifact(
                        "reasoning-summary",
                        {
                            **self._call_context,
                            "source": "responses",
                            **response_summary(response),
                        },
                    )
                identity = self._record_model_identity(response)
                if not strict or identity["model_identity_status"] == "match":
                    return response
                exhausted = attempt + 1 == max_attempts
                event = {
                    **identity,
                    "max_attempts": max_attempts,
                    "duration_s": round(time.monotonic() - started, 3),
                    "action_issued": False,
                }
                if not exhausted:
                    record_artifact("model_identity_retry", event)
                    logger.warning("model_identity_retry", extra={"event": event})
            event = {
                **self._call_context,
                "attempts": attempts,
                "max_attempts": max_attempts,
                "duration_s": round(time.monotonic() - started, 3),
                "action_issued": False,
            }
            record_artifact("model_identity_exhausted", event)
            logger.warning("model_identity_exhausted", extra={"event": event})
            raise ModelIdentityError(
                "model identity verification exhausted; no action issued; restart episode"
            )
        finally:
            self._identity_deadline = None
            self._call_context.pop("model_identity_attempt", None)

    def _record_model_identity(self, response: dict[str, Any]) -> dict[str, Any]:
        """Record reported identity without inferring backend weights."""
        reported = response.get("model") if isinstance(response, dict) else None
        if not isinstance(reported, str) or not reported.strip():
            reported = None
        status = "unreported" if reported is None else "match" if reported == self.model else "mismatch"
        identity = {
            **self._call_context,
            "requested_model": self.model,
            "reported_model": reported,
            "model_identity_status": status,
            "fixed_model_comparison_eligible": status == "match",
            "meaning": "Exact response-label comparison only; not verification of backend weights.",
        }
        record_artifact("model-identity", identity)
        logger.log(
            logging.INFO if status == "match" else logging.WARNING,
            "llm_model_identity",
            extra={"event": identity},
        )
        return identity

    def _decision_response(self, body: dict[str, Any], observation: Observation) -> dict[str, Any]:
        return self.post(body)

    def _with_gripper_example(self, instructions: str) -> str:
        """Show the configured open commands in the output example, never another robot's."""
        if not self.gripper_open_closed:
            return instructions
        opened = [self.gripper_open_closed[arm][0] for arm in ("left", "right")]
        return instructions.replace(
            '"grippers":[-0.785,-0.785]', f'"grippers":{json.dumps(opened, separators=(",", ":"))}'
        )

    def _request_body(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        """Protocol boundary; subclasses reuse observation, transport and lifecycle handling."""
        instructions = (
            fixed_motion_instructions(
                self._unrestricted_fixed_motion,
                grasp_geometry=self._grasp_geometry is not None,
                table_height_available=self._height_prior is not None,
            )
            if self._fixed_motion_steps is not None
            else planner_instructions(
                self._prompt_profile,
                grasp_geometry=self._grasp_geometry is not None,
                table_height_available=self._height_prior is not None,
            )
        )
        instructions = (
            self._with_gripper_example(instructions)
            + self._diagnostic_instructions()
            + self._context_instructions()
        )
        if self._static_rgb_calibration is not None:
            instructions += STATIC_RGB_CALIBRATION_INSTRUCTIONS
        reasoning = {}
        if self.reasoning_effort is not None:
            reasoning["effort"] = self.reasoning_effort
        if self.reasoning_summary is not None:
            reasoning["summary"] = self.reasoning_summary
        return {
            "model": self.model,
            "instructions": "Follow the robot control contract in the developer message in input.",
            # Some compatible gateways replace top-level instructions with their
            # own agent persona. Send the full contract once, in input.
            "input": [
                {"type": "message", "role": "user", "content": content},
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": instructions}],
                },
            ],
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            **({"text": {"format": {"type": "json_object"}}} if self.json_mode else {}),
            **({"reasoning": reasoning} if reasoning else {}),
            **({"stream": True} if self.stream else {}),
        }

    def _diagnostic_instructions(self) -> str:
        return DIAGNOSTIC_INSTRUCTIONS if self.diagnostic_mode else ""

    def _context_instructions(self, waist: bool = True) -> str:
        """Instructions for optional per-request context: waist control and motion options."""
        return (
            (WAIST_INSTRUCTIONS if waist and self._waist_control_active else "")
            + (FINAL_HOLD_INSTRUCTIONS if self._final_hold_active else "")
            + (
                ranked_motion_instructions(self._ranked_motion_candidate_count)
                if self._ranked_motion_candidates
                else ""
            )
            + self._confirmation_wait_instructions
        )

    def _response_text(self, response: dict[str, Any]) -> str:
        if response.get("status") not in (None, "completed") or response.get("error"):
            raise ContractError("LLM returned an incomplete or failed response")
        return "".join(
            part.get("text", "")
            for item in response.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )

    def close(self) -> None:
        self._closed.set()
        self.reset()
