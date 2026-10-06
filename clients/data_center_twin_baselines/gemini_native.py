"""Lossless Gemini generateContent transport for the existing tool agent.

The benchmark still validates exactly one function call per decision. Native
candidate content is retained intact so subsequent turns preserve signatures.
Protocol and usage reference: https://ai.google.dev/api/generate-content
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from urllib.parse import quote


NATIVE_CONTENT_KEY = "_gemini_native_content"


def generate_content_url(base_url: str, model: str) -> str:
    base = base_url.rstrip("/")
    model_path = quote(model.removeprefix("models/"), safe="-._")
    if base.endswith(":generateContent"):
        if not base.endswith(f"/models/{model_path}:generateContent"):
            raise ValueError("Gemini endpoint model does not match configured model")
        return base
    if not base.endswith(("/v1beta", "/v1")):
        base += "/v1beta"
    return f"{base}/models/{model_path}:generateContent"


def generate_content_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate the existing history/schema without changing visible content."""
    if "contents" in payload:
        return deepcopy(payload)
    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, Any]] = []
    pending_functions: dict[str, dict[str, Any]] = {}
    for message in payload.get("messages", []):
        role = message.get("role")
        if role == "system":
            system_parts.append({"text": message["content"]})
        elif role == "assistant" and NATIVE_CONTENT_KEY in message:
            contents.append(deepcopy(message[NATIVE_CONTENT_KEY]))
            for tool_call in message.get("tool_calls", []):
                pending_functions[tool_call["id"]] = tool_call["function"]
        elif role == "tool":
            call_id = message["tool_call_id"]
            function = pending_functions.pop(call_id, None)
            if function is None:
                raise ValueError("Gemini tool response has no preceding function call")
            response = {
                "name": function["name"],
                # Keep runner-delivered text intact, including any StateBundle.
                "response": {"output": message["content"]},
            }
            if function.get("_native_id"):
                response["id"] = function["_native_id"]
            contents.append({"role": "user", "parts": [{"functionResponse": response}]})
        elif role in {"user", "assistant"}:
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": message.get("content") or ""}],
            })
        else:
            raise ValueError(f"unsupported Gemini history role: {role!r}")
    request: dict[str, Any] = {"contents": contents}
    if system_parts:
        request["systemInstruction"] = {"parts": system_parts}
    generation: dict[str, Any] = {"candidateCount": 1}
    if "temperature" in payload:
        generation["temperature"] = payload["temperature"]
    for limit in ("max_tokens", "max_completion_tokens"):
        if limit in payload:
            generation["maxOutputTokens"] = payload[limit]
    if "reasoning_effort" in payload:
        raise ValueError("Gemini native reasoning_effort is unsupported; use thinking_mode")
    thinking = payload.get("thinking", {}).get("type")
    if thinking is not None:
        generation["thinkingConfig"] = {
            "thinkingBudget": 0 if thinking == "disabled" else -1
        }
    request["generationConfig"] = generation
    if "tools" in payload:
        declarations = []
        for tool in payload["tools"]:
            function = tool["function"]
            declarations.append({
                "name": function["name"],
                "description": function.get("description", ""),
                # Gemini's JSON Schema field preserves free-form answer values
                # and action parameters without lossy OpenAPI conversions.
                "parametersJsonSchema": deepcopy(function["parameters"]),
            })
        request["tools"] = [{"functionDeclarations": declarations}]
        mode = {"required": "ANY", "auto": "AUTO", "none": "NONE"}.get(
            payload.get("tool_choice", "required")
        )
        if mode is None:
            raise ValueError("unsupported Gemini tool_choice")
        request["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
    return request


def candidate_content(response: dict[str, Any]) -> dict[str, Any]:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("Gemini response must contain exactly one candidate")
    candidate = candidates[0]
    content = candidate.get("content") if isinstance(candidate, dict) else None
    if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
        raise ValueError("Gemini response missing candidate content parts")
    return content


def native_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for index, part in enumerate(candidate_content(response)["parts"]):
        if not isinstance(part, dict) or "functionCall" not in part:
            continue
        function = part["functionCall"]
        if not isinstance(function, dict):
            raise ValueError("Gemini functionCall must be an object")
        # Older Gemini versions have no call ID. This deterministic local ID
        # only pairs history; it is never presented as a provider-issued ID.
        call_id = function.get("id") or f"gemini-local-call-{index}"
        calls.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": function.get("name"),
                "arguments": deepcopy(function.get("args", {})),
                "_native_id": function.get("id"),
            },
        })
    return calls


def native_text(response: dict[str, Any]) -> str | None:
    try:
        content = candidate_content(response)
    except ValueError:
        return None
    texts = [
        part["text"] for part in content["parts"]
        if isinstance(part, dict) and isinstance(part.get("text"), str)
        and not part.get("thought")
    ]
    return "".join(texts) or None
