# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Multimodal Chat Completions protocol, including DashScope-compatible services."""

from typing import Any

from agent2act.domain.errors import ContractError

from .prompts import planner_instructions
from .responses import ResponsesPlanner


class ChatCompletionsPlanner(ResponsesPlanner):
    """Reuse HTTP retries, observations, traces, reset and close with a distinct wire format."""

    def __init__(
        self,
        model: str,
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        enable_thinking: bool | None = None,
        json_mode: bool = False,
        **options: Any,
    ) -> None:
        if enable_thinking is not None and type(enable_thinking) is not bool:
            raise ContractError("enable_thinking must be boolean or null")
        if options.get("reasoning_effort") is not None:
            raise ContractError("chat_completions does not support reasoning_effort; use null or omit it")
        if options.get("reasoning_summary") is not None:
            raise ContractError("chat_completions does not support reasoning_summary; use null or omit it")
        super().__init__(model=model, endpoint=endpoint, json_mode=json_mode, **options)
        self.enable_thinking = enable_thinking

    def _request_body(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        parts = []
        for part in content:
            if part["type"] == "input_text":
                parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "input_image":
                parts.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
            else:
                raise ContractError("unsupported Chat Completions input type")
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self._with_gripper_example(
                        planner_instructions(
                            self._prompt_profile, grasp_geometry=self._grasp_geometry is not None
                        )
                    )
                    + self._diagnostic_instructions()
                    + self._context_instructions(),
                },
                {"role": "user", "content": parts},
            ],
            "max_tokens": self.max_output_tokens,
            "stream": False,
            **({"response_format": {"type": "json_object"}} if self.json_mode else {}),
            **({"enable_thinking": self.enable_thinking} if self.enable_thinking is not None else {}),
        }

    def _response_text(self, response: dict[str, Any]) -> str:
        if not isinstance(response, dict) or response.get("error"):
            raise ContractError("LLM returned a failed Chat Completions response")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ContractError("LLM must return exactly one Chat Completions choice")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            raise ContractError("LLM returned an incomplete or unsupported Chat Completions response")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("refusal") or message.get("tool_calls"):
            raise ContractError("LLM must return a text decision, not a refusal or tool call")
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise ContractError("LLM returned no text decision")
        return text
