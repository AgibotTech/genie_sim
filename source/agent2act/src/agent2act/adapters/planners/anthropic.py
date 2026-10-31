# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Anthropic Messages wire protocol with the existing read-only geometry tools."""

import json
from copy import deepcopy
from typing import Any

from agent2act.adapters.telemetry import record_artifact
from agent2act.domain.errors import ContractError
from agent2act.domain.models import Observation

from .diagnostics import DIAGNOSTIC_SCHEMA
from .embodied import EmbodiedPlanner


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("ambiguous duplicate JSON key")
        result[key] = value
    return result


def _decision_json_text(text: str) -> str:
    """Strip an unambiguous text envelope around the decision JSON; never repair JSON.

    Valid JSON passes through. A fallback must be exactly one complete object: other values, code
    blocks, malformed objects and duplicate keys are rejected."""
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    parts = text.split("```")
    if len(parts) == 3:
        prefix, block, suffix = parts
        language, newline, candidate = block.partition("\n")
        if not newline or language.strip().lower() not in ("", "json"):
            return text
    elif len(parts) == 1:
        first, last = text.find("{"), text.rfind("}")
        if first < 0 or last < first:
            return text
        prefix, candidate, suffix = text[:first], text[first : last + 1], text[last + 1 :]
    else:
        return text
    for outside in (prefix, suffix):
        if any(char in outside for char in "{}[]"):
            return text
        if outside.strip():
            try:
                json.JSONDecoder().raw_decode(outside.lstrip())
                return text  # Another JSON value, including an unfinished string wrapper.
            except json.JSONDecodeError:
                if outside.lstrip().startswith('"'):
                    return text
    try:
        payload = json.loads(candidate, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError):
        return text
    return candidate if isinstance(payload, dict) else text


class AnthropicPlanner(EmbodiedPlanner):
    """Reuse observations, motion prompts, depth tools, HTTP retries and lifecycle."""

    def __init__(
        self,
        model: str,
        endpoint: str = "https://api.anthropic.com/v1/messages",
        api_key_env: str | None = "ANTHROPIC_API_KEY",
        **options: Any,
    ) -> None:
        if options.get("json_mode", False):
            raise ContractError("anthropic does not support json_mode; use false (prompt JSON contract)")
        if options.get("reasoning_summary") is not None:
            raise ContractError("anthropic does not support reasoning_summary; use null or omit it")
        if options.get("stream", False):
            raise ContractError("anthropic does not support streaming in this adapter")
        if options.get("reasoning_effort") not in (None, "low", "medium", "high", "max"):
            raise ContractError("anthropic reasoning_effort must be low, medium, high, max or null")
        if options.get("invalid_encrypted_content_retries", 0) or options.get(
            "invalid_encrypted_content_replans", 0
        ):
            raise ContractError("anthropic does not support Responses encrypted-content recovery")
        super().__init__(model=model, endpoint=endpoint, api_key_env=api_key_env, **options)

    def _http_headers(self, key: str) -> dict[str, str]:
        headers = super()._http_headers(key)
        headers["anthropic-version"] = "2023-06-01"
        if key:
            headers["x-api-key"] = key
        return headers

    def _diagnostic_instructions(self) -> str:
        instructions = super()._diagnostic_instructions()
        if self.diagnostic_mode:
            instructions += (
                "\nThe diagnostics object must match this exact JSON Schema. Include every required "
                "field, with no extra fields; use null or empty arrays for unknown evidence. "
                "Keep each string under 2000 characters. This is a public evidence report, "
                "not internal reasoning. Diagnostics schema: "
                + json.dumps(DIAGNOSTIC_SCHEMA, separators=(",", ":"))
            )
        return instructions

    @staticmethod
    def _parts(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parts = []
        for part in content:
            if part["type"] == "input_text":
                parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "input_image":
                url = part["image_url"]
                if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
                    raise ContractError("anthropic requires base64 image data URLs")
                header, data = url.split(";base64,", 1)
                parts.append(
                    {"type": "image", "source": {"type": "base64", "media_type": header[5:], "data": data}}
                )
            else:
                raise ContractError("unsupported Anthropic input type")
        return parts

    def _request_body(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        shared = super()._request_body(content)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": "\n".join(
                part["text"]
                for message in shared["input"]
                if message["role"] == "developer"
                for part in message["content"]
            ),
            "messages": [{"role": "user", "content": self._parts(content)}],
        }
        if not self._geometry_tools():
            body["system"] += self._geometry_instructions()
        if self.reasoning_effort is not None:
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": self.reasoning_effort}
        if shared.get("tools"):
            body["tools"] = [
                {"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]}
                for tool in shared["tools"]
            ]
            body["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        return body

    def _with_geometry_context(self, body: dict[str, Any], observation: Observation) -> dict[str, Any]:
        shared = super()._with_geometry_context({"input": []}, observation)
        body["messages"].extend(
            {"role": "user", "content": self._parts(message["content"])} for message in shared["input"]
        )
        return body

    @staticmethod
    def _blocks(response: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(response, dict) or response.get("error") or response.get("type") != "message":
            raise ContractError("LLM returned a failed Anthropic response")
        content = response.get("content")
        if not isinstance(content, list) or any(not isinstance(part, dict) for part in content):
            raise ContractError("invalid Anthropic response content")
        if any(
            part.get("type") not in ("text", "thinking", "redacted_thinking", "tool_use") for part in content
        ):
            raise ContractError("unsupported Anthropic response content")
        return content

    def _tool_decision_response(self, body: dict[str, Any], observation: Observation) -> dict[str, Any]:
        tool_calls = 0
        for turn in range(3):
            if self._closed.is_set():
                raise RuntimeError("LLM planner is closed")
            record_artifact("llm-tool-round-request", {"round": turn, "body": body})
            response = self.post(deepcopy(body))
            record_artifact("llm-tool-round-response", {"round": turn, "body": response})
            blocks = self._blocks(response)
            self._check_output_budget(response)
            calls = [part for part in blocks if part.get("type") == "tool_use"]
            if not calls:
                self._response_text(response)
                return response
            if response.get("stop_reason") != "tool_use":
                raise ContractError("incomplete Anthropic tool response")
            if turn == 2 or tool_calls + len(calls) > 2:
                raise ContractError("geometry tool budget exceeded")
            ids = [call.get("id") for call in calls]
            if any(
                not isinstance(call.get("id"), str)
                or not call["id"]
                or not isinstance(call.get("input"), dict)
                or call.get("name") not in {tool["name"] for tool in self._geometry_tools()}
                for call in calls
            ):
                raise ContractError("invalid Anthropic tool call")
            if len(set(ids)) != len(ids):
                raise ContractError("duplicate Anthropic tool call id")
            results = []
            # Some gateways ignore disable_parallel_tool_use. Honor both read-only
            # queries serially, within the existing total budget of two calls.
            for call in calls:
                result = self._execute_geometry_call(
                    {"name": call["name"], "call_id": call["id"], "arguments": json.dumps(call["input"])},
                    observation,
                )
                results.append(
                    {"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(result)}
                )
            tool_calls += len(calls)
            # Signed thinking blocks must be returned completely unchanged with tool results.
            body["messages"].extend(
                [
                    {"role": "assistant", "content": deepcopy(blocks)},
                    {
                        "role": "user",
                        "content": results,
                    },
                ]
            )
            if tool_calls == 2:
                body["tool_choice"] = {"type": "none"}
        raise ContractError("no final decision after geometry tools")

    def _check_output_budget(self, response: dict[str, Any]) -> None:
        if response.get("stop_reason") != "max_tokens":
            return
        usage = response.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("output_tokens_details")
        details = details if isinstance(details, dict) else {}
        counters = {"max_output_tokens": self.max_output_tokens}
        for name, value in (
            ("output_tokens", usage.get("output_tokens")),
            ("thinking_tokens", details.get("thinking_tokens")),
        ):
            if type(value) is int and value >= 0:
                counters[name] = value
        counts = ", ".join(f"{name}={value}" for name, value in counters.items())
        raise ContractError(
            f"Anthropic output budget exhausted (stop_reason=max_tokens, {counts}); "
            "no complete decision; increase max_output_tokens or explicitly lower reasoning_effort"
        )

    def _response_text(self, response: dict[str, Any]) -> str:
        blocks = self._blocks(response)
        self._check_output_budget(response)
        if response.get("stop_reason") != "end_turn" or any(
            part.get("type") == "tool_use" for part in blocks
        ):
            raise ContractError("LLM returned an incomplete or unsupported Anthropic response")
        texts: list[str] = []
        for part in blocks:
            if part.get("type") == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ContractError("LLM returned no text decision")
                texts.append(text)
        if not texts or not "".join(texts).strip():
            raise ContractError("LLM returned no text decision")
        return _decision_json_text("".join(texts))
