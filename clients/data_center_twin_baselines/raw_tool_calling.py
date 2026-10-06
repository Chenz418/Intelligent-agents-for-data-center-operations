"""Provider-native raw tool-calling Data Center Twin LLM baseline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
import json
import math

from .openai_compatible import OpenAICompatibleAgent, OpenAICompatibleSettings
from .json_utils import strict_json_loads
from .gemini_native import (
    NATIVE_CONTENT_KEY,
    candidate_content,
    native_text,
    native_tool_calls,
)
from .prompts import (
    ALLOWED_AGENT_ACTION_TYPES,
    build_raw_tool_calling_prompt,
    fence_api_call,
)


class RawToolCallingAgent(OpenAICompatibleAgent):
    """Provider-native tool-calling baseline for OpenAI and Gemini protocols.

    This agent receives only the benchmark-provided task text, task APIs,
    sanitized action space, sanitized observations, and turn history. It does
    not build a handcrafted compact state or domain summary. The provider must
    return exactly one structured tool call, which is converted to the existing
    markdown-fenced benchmark API-call format consumed by ResponseParser.
    """

    agent_type = "tool-calling"

    def __init__(self, settings: OpenAICompatibleSettings) -> None:
        super().__init__(settings)
        self.emitted_calls: list[str] = []
        self.last_normalization_status: dict[str, Any] | None = None
        self._pending_tool_call_id: str | None = None

    def init_context(
        self,
        *,
        agent_task_id: str,
        task_description: str,
        instructions: str,
        actions: dict[str, Any],
        action_space_payload: dict[str, Any] | None,
        initial_observation: dict[str, Any] | None,
    ) -> str:
        content = build_raw_tool_calling_prompt(
            agent_task_id=agent_task_id,
            task_description=task_description,
            instructions=instructions,
            actions=actions,
            action_space_payload=action_space_payload,
            initial_observation=initial_observation,
        )
        self._initialize_messages(content)
        self.emitted_calls = []
        self.last_normalization_status = None
        self._pending_tool_call_id = None
        return content

    async def get_action(self, input_text: str) -> str:
        if self._pending_tool_call_id is None:
            self.messages.append({"role": "user", "content": input_text})
        else:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": self._pending_tool_call_id,
                    "content": input_text,
                }
            )
            self._pending_tool_call_id = None
        self._capture_turn_decision_checkpoint()
        response = await self._call_tool_llm(self.messages)
        normalized, call, error, original = self._normalize_tool_response(response)
        self.last_normalization_status = {
            "normalized_ok": error is None,
            "normalization_error": error,
            "original_model_output": original,
        }
        if call:
            self.emitted_calls.append(call)
        if error is None:
            assistant_message, tool_call_id = provider_tool_history_message(response)
            self.messages.append(assistant_message)
            self._pending_tool_call_id = tool_call_id
        else:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": provider_text_content(response) or normalized,
                }
            )
        return normalized

    def _capture_turn_decision_checkpoint(self) -> None:
        super()._capture_turn_decision_checkpoint()
        if isinstance(self._turn_decision_checkpoint, dict):
            self._turn_decision_checkpoint["pending_tool_call_id"] = (
                self._pending_tool_call_id
            )

    def discard_late_provider_response(self) -> None:
        super().discard_late_provider_response()
        checkpoint = self._turn_decision_checkpoint
        if isinstance(checkpoint, dict):
            pending = checkpoint.get("pending_tool_call_id")
            self._pending_tool_call_id = pending if isinstance(pending, str) else None

    async def _call_tool_llm(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return await self._call_tool_completion(messages)

    async def _call_tool_completion(
        self, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        payload = self._chat_completion_payload(messages)
        payload.update(
            {
                "tools": data_center_twin_tool_schemas(),
                "tool_choice": self.settings.tool_choice,
            }
        )
        return await self._request_chat_completion_async(payload)

    def _chat_completion_with_tools(
        self, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        payload = self._chat_completion_payload(messages)
        payload.update(
            {
                "tools": data_center_twin_tool_schemas(),
                "tool_choice": self.settings.tool_choice,
            }
        )
        return self._request_chat_completion_json(payload)

    def _normalize_tool_response(
        self,
        response: dict[str, Any],
    ) -> tuple[str, str | None, str | None, str]:
        original = json.dumps(response, sort_keys=True, default=str)
        try:
            tool_calls = extract_tool_calls(response)
            if len(tool_calls) != 1:
                raise ValueError(
                    f"expected exactly one provider tool call, got {len(tool_calls)}"
                )
            tool_call_id = tool_calls[0].get("id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError("provider tool call missing id")
            call = tool_call_to_api_call(tool_calls[0])
        except ValueError as error:
            message = f"provider tool-call normalization failed: {error}"
            return message, None, message, original
        return fence_api_call(call), call, None, original


def extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    if "candidates" in response:
        return native_tool_calls(response)
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices")
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response missing message")
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [item for item in tool_calls if isinstance(item, dict)]


def provider_text_content(response: dict[str, Any]) -> str | None:
    if "candidates" in response:
        return native_text(response)
    choices = response.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    first_choice = first_choice if isinstance(first_choice, dict) else {}
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) and content else None


def provider_tool_history_message(
    response: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Retain native assistant/tool history for the next provider turn."""

    if "candidates" in response:
        tool_calls = native_tool_calls(response)
        if len(tool_calls) != 1:
            raise ValueError("expected exactly one Gemini function call")
        return (
            {
                "role": "assistant",
                "content": native_text(response),
                "tool_calls": deepcopy(tool_calls),
                NATIVE_CONTENT_KEY: deepcopy(candidate_content(response)),
            },
            tool_calls[0]["id"],
        )

    choices = response.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    first_choice = first_choice if isinstance(first_choice, dict) else {}
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response missing message")
    tool_calls = extract_tool_calls(response)
    if len(tool_calls) != 1:
        raise ValueError(
            f"expected exactly one provider tool call, got {len(tool_calls)}"
        )
    tool_call_id = tool_calls[0].get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError("provider tool call missing id")
    return (
        {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": deepcopy(tool_calls),
        },
        tool_call_id,
    )


def tool_call_to_api_call(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise ValueError("tool call missing function object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("tool call missing function name")
    arguments = parse_tool_arguments(function.get("arguments"))
    if name == "dc_twin_action_space":
        if arguments:
            raise ValueError("dc_twin_action_space does not accept arguments")
        return "dc_twin_action_space()"
    if name == "dc_twin_observe":
        kwargs: dict[str, Any] = {}
        if "log_limit" in arguments:
            if (
                not isinstance(arguments["log_limit"], int)
                or isinstance(arguments["log_limit"], bool)
                or arguments["log_limit"] < 0
            ):
                raise ValueError(
                    "dc_twin_observe.log_limit must be a non-negative integer"
                )
            kwargs["log_limit"] = arguments["log_limit"]
        if "include_config" in arguments:
            if not isinstance(arguments["include_config"], bool):
                raise ValueError("dc_twin_observe.include_config must be a boolean")
            kwargs["include_config"] = arguments["include_config"]
        if "lookback_seconds" in arguments:
            lookback = arguments["lookback_seconds"]
            if (
                isinstance(lookback, bool)
                or not isinstance(lookback, (int, float))
                or not math.isfinite(float(lookback))
                or lookback < 0
            ):
                raise ValueError(
                    "dc_twin_observe.lookback_seconds must be non-negative"
                )
            kwargs["lookback_seconds"] = lookback
        if "channels" in arguments:
            channels = arguments["channels"]
            supported_channels = {"log", "metric", "alert", "trace", "config"}
            if (
                not isinstance(channels, list)
                or not all(isinstance(channel, str) for channel in channels)
                or len(set(channels)) != len(channels)
                or not set(channels) <= supported_channels
            ):
                raise ValueError(
                    "dc_twin_observe.channels must be a unique list of supported channels"
                )
            kwargs["channels"] = channels
        if "detail" in arguments:
            detail = arguments["detail"]
            if detail not in {"overview", "raw"}:
                raise ValueError("dc_twin_observe.detail must be 'overview' or 'raw'")
            kwargs["detail"] = detail
        for field_name in (
            "metric_names",
            "entity_ids",
            "subsystem_ids",
            "alert_names",
        ):
            if field_name not in arguments:
                continue
            values = arguments[field_name]
            if not isinstance(values, list) or not all(
                isinstance(item, str) and item.strip() for item in values
            ):
                raise ValueError(
                    f"dc_twin_observe.{field_name} must be a list of non-empty strings"
                )
            kwargs[field_name] = values
        return format_api_call("dc_twin_observe", [], kwargs)
    if name == "dc_twin_action":
        action_type = arguments.get("action_type")
        if not isinstance(action_type, str) or not action_type:
            raise ValueError("dc_twin_action requires string action_type")
        if action_type not in ALLOWED_AGENT_ACTION_TYPES:
            raise ValueError(f"unsupported dc_twin_action action_type: {action_type}")
        kwargs = {
            key: value for key, value in arguments.items() if key not in {"action_type"}
        }
        return format_api_call("dc_twin_action", [action_type], kwargs)
    if name == "submit":
        if not arguments:
            return "submit()"
        # A mixed envelope must retain all fields, including contradictions.
        if set(arguments) == {"payload"}:
            return format_api_call("submit", [arguments["payload"]], {})
        return format_api_call("submit", [arguments], {})
    raise ValueError(f"unsupported provider tool name: {name}")


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments in (None, ""):
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str):
        raise ValueError("tool arguments must be a JSON object")
    try:
        parsed = strict_json_loads(raw_arguments)
    except ValueError as error:
        raise ValueError(f"tool arguments are not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must decode to a JSON object")
    return parsed


def format_api_call(api_name: str, args: list[Any], kwargs: dict[str, Any]) -> str:
    rendered_args = [repr(arg) for arg in args]
    rendered_args.extend(f"{key}={value!r}" for key, value in kwargs.items())
    return f"{api_name}({', '.join(rendered_args)})"


def data_center_twin_tool_schemas() -> list[dict[str, Any]]:
    action_type_enum = sorted(ALLOWED_AGENT_ACTION_TYPES)
    return [
        {
            "type": "function",
            "function": {
                "name": "dc_twin_action_space",
                "description": "Return the visible Data Center Twin agent action schema.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "dc_twin_observe",
                "description": "Inspect current visible telemetry without mutating simulator state.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "log_limit": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Maximum recent log/event count to include.",
                            "default": 20,
                        },
                        "include_config": {
                            "type": "boolean",
                            "description": "Whether to include visible configuration facts.",
                            "default": True,
                        },
                        "lookback_seconds": {
                            "type": "number",
                            "minimum": 0,
                            "description": "Visible lookback window in simulated seconds.",
                            "default": 300,
                        },
                        "channels": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["log", "metric", "alert", "trace", "config"],
                            },
                            "uniqueItems": True,
                            "description": "Telemetry channels to include.",
                        },
                        "detail": {
                            "type": "string",
                            "enum": ["overview", "raw"],
                            "description": (
                                "Use overview first; raw returns complete series/details "
                                "for the requested narrow view."
                            ),
                            "default": "overview",
                        },
                        "metric_names": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "description": "Optional metric-series names for drill-down.",
                        },
                        "entity_ids": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "description": "Optional visible entity IDs for drill-down.",
                        },
                        "subsystem_ids": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "description": "Optional visible subsystem IDs for drill-down.",
                        },
                        "alert_names": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "description": "Optional visible alert names for drill-down.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "dc_twin_action",
                "description": (
                    "Execute one visible Data Center Twin agent action. Use only action types "
                    "advertised by the visible action space."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "enum": action_type_enum,
                        },
                        "parameters": {
                            "description": "Action-specific parameter object when required.",
                            "anyOf": [
                                {"type": "object"},
                                {"type": "null"},
                            ],
                        },
                    },
                    "required": ["action_type"],
                    "additionalProperties": True,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Submit the final task answer. Omit payload for mitigation submit().",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payload": {
                            "description": "Final answer payload for detection, localization, or analysis.",
                            "anyOf": [
                                {"type": "object"},
                                {"type": "array"},
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                                {"type": "null"},
                            ],
                        },
                    },
                    "additionalProperties": True,
                },
            },
        },
    ]
