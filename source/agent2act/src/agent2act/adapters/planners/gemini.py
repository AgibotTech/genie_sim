# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Gemini GenerateContent REST protocol with bounded read-only geometry tools."""

import json
from copy import deepcopy
from typing import Any
from urllib.parse import quote

from agent2act.adapters.telemetry import record_artifact
from agent2act.domain.errors import ContractError
from agent2act.domain.models import Observation

from .diagnostics import DIAGNOSTIC_SCHEMA
from .embodied import EmbodiedPlanner


class GeminiPlanner(EmbodiedPlanner):
    """Reuse observation preparation, geometry queries, retries and reset/close."""

    def __init__(
        self,
        model: str,
        endpoint: str | None = None,
        api_key_env: str | None = "GEMINI_API_KEY",
        **options: Any,
    ) -> None:
        if options.get("stream", False):
            raise ContractError("gemini does not support streaming in this adapter")
        if options.get("reasoning_effort") not in (None, "low", "medium", "high"):
            raise ContractError("gemini reasoning_effort must be low, medium, high or null")
        if options.get("reasoning_summary") not in (None, "auto"):
            raise ContractError("gemini reasoning_summary must be auto or null")
        if options.get("invalid_encrypted_content_retries", 0) or options.get(
            "invalid_encrypted_content_replans", 0
        ):
            raise ContractError("gemini does not support Responses encrypted-content recovery")
        if endpoint is None:
            endpoint = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                + quote(model, safe="")
                + ":generateContent"
            )
        super().__init__(model=model, endpoint=endpoint, api_key_env=api_key_env, **options)

    def _http_headers(self, key: str) -> dict[str, str]:
        headers = super()._http_headers("")
        if key:
            headers["x-goog-api-key"] = key
        return headers

    def _diagnostic_instructions(self) -> str:
        instructions = super()._diagnostic_instructions()
        if self.diagnostic_mode:
            instructions += (
                "\nDiagnostics must match this exact JSON Schema, including all required fields "
                "and enum values. Use null or empty arrays for unknown evidence; no extra fields. "
                "This is a public evidence report, not internal reasoning. Diagnostics schema: "
                + json.dumps(DIAGNOSTIC_SCHEMA, separators=(",", ":"))
            )
        return instructions

    @staticmethod
    def _parts(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parts = []
        for part in content:
            if part["type"] == "input_text":
                parts.append({"text": part["text"]})
            elif part["type"] == "input_image":
                url = part["image_url"]
                if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
                    raise ContractError("gemini requires base64 image data URLs")
                header, data = url.split(";base64,", 1)
                parts.append({"inlineData": {"mimeType": header[5:], "data": data}})
            else:
                raise ContractError("unsupported Gemini input type")
        return parts

    def _request_body(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        shared = super()._request_body(content)
        instructions = "\n".join(
            part["text"]
            for message in shared["input"]
            if message["role"] == "developer"
            for part in message["content"]
        )
        if not self._geometry_tools():
            instructions += self._geometry_instructions()
        generation: dict[str, Any] = {"maxOutputTokens": self.max_output_tokens, "candidateCount": 1}
        if self.json_mode:
            generation["responseMimeType"] = "application/json"
        thinking: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            thinking["thinkingLevel"] = self.reasoning_effort.upper()
        if self.reasoning_summary is not None:
            thinking["includeThoughts"] = True
        if thinking:
            generation["thinkingConfig"] = thinking
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": instructions}]},
            "contents": [{"role": "user", "parts": self._parts(content)}],
            "generationConfig": generation,
        }
        if shared.get("tools"):
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parametersJsonSchema": tool["parameters"],
                        }
                        for tool in shared["tools"]
                    ]
                }
            ]
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
        return body

    def _with_geometry_context(self, body: dict[str, Any], observation: Observation) -> dict[str, Any]:
        shared = super()._with_geometry_context({"input": []}, observation)
        # Keep one user turn; merge reference text without changing the image order.
        for message in shared["input"]:
            body["contents"][0]["parts"].extend(self._parts(message["content"]))
        return body

    def _candidate_content(self, response: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(response, dict) or response.get("error"):
            raise ContractError("LLM returned a failed Gemini response")
        feedback = response.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            raise ContractError("Gemini blocked the prompt")
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
            raise ContractError("Gemini must return exactly one candidate")
        candidate = candidates[0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise ContractError(
                "Gemini output budget exhausted (finishReason=MAX_TOKENS, "
                f"max_output_tokens={self.max_output_tokens}); no complete decision"
            )
        if candidate.get("finishReason") != "STOP":
            raise ContractError("LLM returned an incomplete or unsupported Gemini response")
        content = candidate.get("content")
        if not isinstance(content, dict) or content.get("role") != "model":
            raise ContractError("invalid Gemini candidate content")
        parts = content.get("parts")
        if not isinstance(parts, list) or not parts or any(not isinstance(p, dict) for p in parts):
            raise ContractError("invalid Gemini response parts")
        for part in parts:
            if set(part) - {"text", "thought", "thoughtSignature", "functionCall"}:
                raise ContractError("unsupported Gemini response part")
            if "thought" in part and type(part["thought"]) is not bool:
                raise ContractError("invalid Gemini thought marker")
            if "thoughtSignature" in part and not isinstance(part["thoughtSignature"], str):
                raise ContractError("invalid Gemini thought signature")
            if "text" in part and not isinstance(part["text"], str):
                raise ContractError("invalid Gemini text")
            if "functionCall" in part and ("text" in part or part.get("thought")):
                raise ContractError("invalid Gemini function call part")
        return content

    def _tool_decision_response(self, body: dict[str, Any], observation: Observation) -> dict[str, Any]:
        tool_calls = 0
        for turn in range(3):
            if self._closed.is_set():
                raise RuntimeError("LLM planner is closed")
            record_artifact("llm-tool-round-request", {"round": turn, "body": body})
            response = self.post(deepcopy(body))
            record_artifact("llm-tool-round-response", {"round": turn, "body": response})
            content = self._candidate_content(response)
            if self.reasoning_summary is not None:
                summaries = [p["text"] for p in content["parts"] if p.get("thought") and p.get("text")]
                record_artifact(
                    "reasoning-summary",
                    {
                        **self._call_context,
                        "source": "gemini_tool_round",
                        "round": turn,
                        "status": "available" if summaries else "not_provided",
                        "summary": summaries,
                        "response_status": "STOP",
                        "meaning": "Provider-returned thought summaries, not a full internal reasoning trace.",
                    },
                )
            calls = [p["functionCall"] for p in content["parts"] if "functionCall" in p]
            if not calls:
                self._response_text(response)
                return {**response, "_summary_recorded": True}
            if tool_calls + len(calls) > 2:
                raise ContractError("geometry tool budget exceeded")
            names = {tool["name"] for tool in self._geometry_tools()}
            if any(
                not isinstance(call, dict)
                or not isinstance(call.get("name"), str)
                or call["name"] not in names
                or not isinstance(call.get("args"), dict)
                or ("id" in call and (not isinstance(call["id"], str) or not call["id"]))
                for call in calls
            ):
                raise ContractError("invalid Gemini geometry tool call")
            ids = [call["id"] for call in calls if "id" in call]
            if len(set(ids)) != len(ids):
                raise ContractError("duplicate Gemini tool call id")
            results = []
            for index, call in enumerate(calls):
                result = self._execute_geometry_call(
                    {
                        "name": call["name"],
                        "call_id": call.get("id", f"gemini-{turn}-{index}"),
                        "arguments": json.dumps(call["args"], allow_nan=False),
                    },
                    observation,
                )
                results.append(
                    {
                        "functionResponse": {
                            "name": call["name"],
                            **({"id": call["id"]} if "id" in call else {}),
                            "response": result,
                        }
                    }
                )
            tool_calls += len(calls)
            # Preserve every model part in order, including opaque thought signatures.
            body["contents"].extend([deepcopy(content), {"role": "user", "parts": results}])
            if tool_calls == 2:
                body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
        raise ContractError("no final decision after geometry tools")

    def _response_text(self, response: dict[str, Any]) -> str:
        content = self._candidate_content(response)
        if any("functionCall" in p for p in content["parts"]):
            raise ContractError("Gemini must return a final text decision")
        text = "".join(p.get("text", "") for p in content["parts"] if not p.get("thought"))
        if not text.strip():
            raise ContractError("Gemini returned no text decision")
        return text

    def _record_model_identity(self, response: dict[str, Any]) -> dict[str, Any]:
        return super()._record_model_identity({"model": response.get("modelVersion")})
