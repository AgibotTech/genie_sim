# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""LLM decisions with bounded read-only sensor tools, then finite motion."""

import json
import logging
import time
from copy import deepcopy
from typing import Any

from agent2act.adapters.telemetry import record_artifact
from agent2act.domain.errors import TransientBackendError
from agent2act.domain.models import ContractError, Observation, vector

from .diagnostics import response_summary
from .geometry_tools import (
    GRASP_POSE_TOOL,
    SURFACE_TOOL,
    compose_grasp_pose,
    measure_surface_points,
    rotate_vector,
)
from .raw_depth import RAW_DEPTH_INSTRUCTIONS, RAW_DEPTH_TOOL, read_depth_pixels
from .responses import InvalidEncryptedContentError, ResponsesPlanner

logger = logging.getLogger(__name__)

GEOMETRY_INSTRUCTIONS = (
    "\nUse only the read-only tools advertised in this request. Select pixels yourself from the "
    "CURRENT RGB image using normalized 0..1000 coordinates. Measurements describe visible surfaces, "
    "not object centers, semantic identities, contact states or collision clearance. A query uses "
    "the current observation; it neither moves the robot nor obtains a newer frame. At most 2 tool "
    "rounds are available per decision. Use results only within their declared frame and calibration "
    "limits. Do not treat robot target arrival or a geometric calculation as task completion."
)

RGB_GEOMETRY_INSTRUCTIONS = (
    "\nDepth access is disabled. Estimate the next EEF target from RGB and measured robot state; "
    "visual estimates are not measured object coordinates. Use available views and action feedback "
    "to correct the estimate after moving. Take useful approach steps in clear space and smaller "
    "corrections near contact. A grasp attempt can test an estimate; check the object's response "
    "in the next observation."
)


class EmbodiedPlanner(ResponsesPlanner):
    def __init__(
        self,
        wrist_to_center_m: dict[str, list[float]] | None = None,
        enable_depth_measurement: bool = False,
        enable_raw_depth_query: bool = False,
        geometry_history_queries: int = 8,
        invalid_encrypted_content_replans: int = 0,
        gripper_open_closed: dict[str, list[float]] | None = None,
        **options: Any,
    ) -> None:
        if (
            type(invalid_encrypted_content_replans) is not int
            or not 0 <= invalid_encrypted_content_replans <= 99
        ):
            raise ContractError("invalid_encrypted_content_replans must be an integer between 0 and 99")
        self.invalid_encrypted_content_replans = invalid_encrypted_content_replans
        if type(enable_depth_measurement) is not bool:
            raise ContractError("enable_depth_measurement must be boolean")
        if type(enable_raw_depth_query) is not bool:
            raise ContractError("enable_raw_depth_query must be boolean")
        if enable_raw_depth_query and enable_depth_measurement:
            raise ContractError("choose raw depth queries or projected surface measurements, not both")
        self.enable_raw_depth_query = enable_raw_depth_query
        self.enable_depth_measurement = enable_depth_measurement
        if type(geometry_history_queries) is not int or not 0 <= geometry_history_queries <= 8:
            raise ContractError("geometry_history_queries must be an integer between 0 and 8")
        self.geometry_history_queries = geometry_history_queries
        if options.get("model_identity_policy", "warn") != "warn":
            raise ContractError(
                "strict model identity retries require a planner without geometry tool rounds"
            )
        if options.get("use_gripper_center") is False:
            wrist_to_center_m = None
        if (
            wrist_to_center_m is not None
            and options.get("gripper_center_offset_m") is not None
            and options.get("use_gripper_center") is not False
        ):
            raise ContractError("choose gripper_center_offset_m or wrist_to_center_m, not both")
        self.wrist_to_center_m = {}
        if wrist_to_center_m is not None:
            if not isinstance(wrist_to_center_m, dict) or set(wrist_to_center_m) != {"left", "right"}:
                raise ContractError("tool offsets require left and right")
            self.wrist_to_center_m = {
                arm: vector(value, 3, "tool offset") for arm, value in wrist_to_center_m.items()
            }
        super().__init__(gripper_open_closed=gripper_open_closed, **options)

    def reset(self) -> None:
        super().reset()
        self._geometry_memory: list[dict[str, Any]] = []

    def _request_body(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        body = super()._request_body(content)
        specs = self._geometry_tools()
        if specs:
            body["tools"] = specs
            body["parallel_tool_calls"] = False
            body["include"] = ["reasoning.encrypted_content"]
            # The shared motion contract describes final decisions, not the sensor-query phase.
            for message in body["input"]:
                if message.get("role") == "developer":
                    for part in message["content"]:
                        if part.get("type") == "input_text":
                            part["text"] = part["text"].replace(
                                "Do not return markdown fences, commentary, or a tool call.",
                                "Do not return markdown fences or commentary in the final decision.",
                            )
                            part["text"] += (
                                "\nThe JSON output contract applies to your FINAL decision. Before it, "
                                "you may call the advertised read-only sensor tools."
                                + self._geometry_instructions()
                            )
        if not specs:
            body["instructions"] += self._geometry_instructions()
        return body

    def _geometry_tools(self) -> list[dict[str, Any]]:
        return [
            *([RAW_DEPTH_TOOL] if self.use_depth and self.enable_raw_depth_query else []),
            *([SURFACE_TOOL] if self.use_depth and self.enable_depth_measurement else []),
            *([GRASP_POSE_TOOL] if self.wrist_to_center_m else []),
        ]

    def _geometry_instructions(self) -> str:
        instructions = (
            RAW_DEPTH_INSTRUCTIONS
            if self.use_depth and self.enable_raw_depth_query
            else GEOMETRY_INSTRUCTIONS
            if self.use_depth and self.enable_depth_measurement
            else (
                "\nUse RGB, the supplied full-resolution uint16 depth PNGs, camera calibration and measured robot state. "
                "Pixel-depth measurement tools are not available. Provider image processing may alter numeric precision; "
                "do not claim exact depth readings from display brightness alone."
                if self.use_depth
                else RGB_GEOMETRY_INSTRUCTIONS
            )
        )
        if not self.wrist_to_center_m and not self._tool_center:
            instructions += (
                " EEF poses describe the robot's reported control point, not the point between the "
                "fingers. No wrist-to-grasp-center calibration or conversion tool is provided; "
                "estimate corrections from visible finger motion without assuming a fixed offset."
            )
        if not self._geometry_tools():
            instructions += " No tools are available. Return your decision directly."
        return instructions

    def _with_geometry_context(self, body: dict[str, Any], observation: Observation) -> dict[str, Any]:
        if self.gripper_open_closed:
            reference = {
                arm: {"fully_open_command": pair[0], "fully_closed_command": pair[1]}
                for arm, pair in self.gripper_open_closed.items()
            }
            body = {
                **body,
                "input": [
                    *body["input"],
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": json.dumps(
                                    {
                                        "robot_gripper_command_reference": reference,
                                        "meaning": "Configured actuator endpoint semantics, not an instruction to close. You choose closure magnitude and timing from observations; intermediate commands are allowed. Commanded closure does not establish contact or grasp.",
                                    },
                                    allow_nan=False,
                                ),
                            }
                        ],
                    },
                ],
            }
        if self.wrist_to_center_m:
            geometry = {}
            for arm, offset in self.wrist_to_center_m.items():
                wrist = observation.eef(arm, "base_link")
                rotated = rotate_vector(wrist[3:], offset)
                geometry[arm] = {
                    "wrist_pose": wrist,
                    "center_m": [a + b for a, b in zip(wrist[:3], rotated)],
                    "wrist_to_center_local_m": offset,
                    "frame_axes": "local +z wrist toward fingers; local x closing axis; center is a robot frame, not object center",
                }
            body = {
                **body,
                "input": [
                    *body["input"],
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": json.dumps({"robot_tool_geometry": geometry}, allow_nan=False),
                            }
                        ],
                    },
                ],
            }
        if not hasattr(self, "_geometry_memory"):
            self._geometry_memory = []
        if (
            self._geometry_memory
            and self.use_depth
            and (self.enable_depth_measurement or self.enable_raw_depth_query)
        ):
            memory = {
                "prior_geometry_queries": deepcopy(self._geometry_memory),
                "current_sequence": self._call_context.get("sequence"),
                "meaning": "Historical measurements, NOT current object positions. Labels were chosen by you, not verified. Re-measure moved objects and fingers; never reuse old image pixels after camera motion.",
            }
            body = {
                **body,
                "input": [
                    *body["input"],
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": json.dumps(memory, allow_nan=False)}],
                    },
                ],
            }
        return body

    def _decision_response(self, body: dict[str, Any], observation: Observation) -> dict[str, Any]:
        original_memory = deepcopy(getattr(self, "_geometry_memory", []))
        initial_body = self._with_geometry_context(deepcopy(body), observation)
        previous_deadline = self._recovery_deadline
        previous_attempt = self._call_context.get("encrypted_content_recovery_attempt")
        if self.retry_budget_s is not None and (
            self.invalid_encrypted_content_replans or self.invalid_encrypted_content_retries
        ):
            deadline = time.monotonic() + self.retry_budget_s
            self._recovery_deadline = (
                deadline if previous_deadline is None else min(deadline, previous_deadline)
            )
        try:
            for attempt in range(self.invalid_encrypted_content_replans + 1):
                self._call_context["encrypted_content_recovery_attempt"] = attempt
                if self._recovery_deadline is not None and time.monotonic() >= self._recovery_deadline:
                    raise TransientBackendError("LLM planning recovery budget exhausted")
                try:
                    return self._tool_decision_response(deepcopy(initial_body), observation)
                except InvalidEncryptedContentError:
                    exhausted = attempt == self.invalid_encrypted_content_replans
                    event = {
                        **self._call_context,
                        "error_code": "invalid_encrypted_content",
                        "recovery_attempt": attempt + 1,
                        "max_restarts": self.invalid_encrypted_content_replans,
                        "stage": "exhausted" if exhausted else "restart_current_planning",
                        "action_issued": False,
                        "observation_reused": True,
                        "prior_action_history_preserved": True,
                    }
                    record_artifact("llm-planning-recovery", event)
                    logger.warning("llm_planning_recovery", extra={"event": event})
                    # Queries from an abandoned tool exchange must not leak into the retry.
                    self._geometry_memory = deepcopy(original_memory)
                    if exhausted:
                        raise
            raise RuntimeError("LLM planning recovery exhausted")
        finally:
            self._recovery_deadline = previous_deadline
            if previous_attempt is None:
                self._call_context.pop("encrypted_content_recovery_attempt", None)
            else:
                self._call_context["encrypted_content_recovery_attempt"] = previous_attempt

    def _tool_decision_response(self, body: dict[str, Any], observation: Observation) -> dict[str, Any]:
        for turn in range(3):
            if self._closed.is_set():
                raise RuntimeError("LLM planner is closed")
            round_context = {
                "round": turn,
                "recovery_attempt": self._call_context.get("encrypted_content_recovery_attempt", 0),
            }
            record_artifact("llm-tool-round-request", {**round_context, "body": body})
            response = self.post(body)
            record_artifact("llm-tool-round-response", {**round_context, "body": response})
            if response.get("status") not in (None, "completed") or response.get("error"):
                raise ContractError("incomplete tool response")
            outputs = response.get("output", [])
            if not isinstance(outputs, list):
                raise ContractError("invalid tool response output")
            self._on_tool_round_response(body, response)
            if self.reasoning_summary is not None:
                record_artifact(
                    "reasoning-summary",
                    {
                        **self._call_context,
                        "source": "responses_tool_round",
                        "round": turn,
                        **response_summary(response),
                    },
                )
            calls = [item for item in outputs if item.get("type") == "function_call"]
            if not calls:
                return (
                    {**response, "_summary_recorded": True}
                    if self.reasoning_summary is not None
                    else response
                )
            if turn == 2 or len(calls) > 1:
                raise ContractError("geometry tool budget exceeded")
            call = calls[0]
            if call.get("name") not in {s["name"] for s in self._geometry_tools()} or not isinstance(
                call.get("call_id"), str
            ):
                raise ContractError("unsupported geometry tool call")
            result = self._execute_geometry_call(call, observation)
            # Preserve reasoning items together with function calls, as required
            # for stateless Responses continuation (store remains false).
            body = {
                **body,
                "input": [
                    *body["input"],
                    *outputs,
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": json.dumps(result, allow_nan=False),
                    },
                ],
            }
            if turn == 1:
                body["tool_choice"] = "none"
        raise ContractError("no final decision after geometry tools")

    def _on_tool_round_response(self, body: dict[str, Any], response: dict[str, Any]) -> None:
        """Optional adapter state update, after the raw response has been traced."""

    def _execute_geometry_call(self, call: dict[str, Any], observation: Observation) -> dict[str, Any]:
        if call.get("name") not in {s["name"] for s in self._geometry_tools()}:
            raise ContractError("geometry tool unavailable with current depth/robot configuration")
        try:
            arguments = json.loads(call.get("arguments", ""))
            result = (
                read_depth_pixels(observation, arguments)
                if call["name"] == "read_depth_pixels"
                else measure_surface_points(observation, arguments)
                if call["name"] == "measure_surface_points"
                else compose_grasp_pose(arguments, self.wrist_to_center_m)
            )
        except (json.JSONDecodeError, ContractError) as exc:
            result = {"error": str(exc), "motion_executed": False}
        record_artifact(
            "geometry-tool-result",
            {
                "name": call["name"],
                "call_id": call["call_id"],
                "arguments": call.get("arguments"),
                "result": result,
            },
        )
        self._geometry_memory.append(
            {
                "sequence": self._call_context.get("sequence"),
                "name": call["name"],
                "arguments": call.get("arguments"),
                "result": deepcopy(result),
            }
        )
        if self.geometry_history_queries:
            del self._geometry_memory[: -self.geometry_history_queries]
        else:
            self._geometry_memory.clear()
        return result
