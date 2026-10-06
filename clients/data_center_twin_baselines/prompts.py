"""Prompt utilities for raw Data Center Twin LLM baselines."""

from __future__ import annotations

from typing import Any
import json

from aiopslab.orchestrator.problems.data_center_twin.scenarios import (
    AGENT_ACTIONS as SCENARIO_AGENT_ACTIONS,
)
from aiopslab.orchestrator.problems.data_center_twin.visibility import (
    sanitize_agent_payload,
    sanitize_agent_action_space,
)


# Tool schemas use the benchmark's public action set.
ALLOWED_AGENT_ACTION_TYPES = frozenset(SCENARIO_AGENT_ACTIONS)


def build_raw_tool_calling_prompt(
    *,
    agent_task_id: str,
    task_description: str,
    instructions: str,
    actions: dict[str, Any],
    action_space_payload: dict[str, Any] | None,
    initial_observation: dict[str, Any] | None,
) -> str:
    """Build static raw-tool instructions without embedding telemetry."""
    context = raw_context_payload(
        agent_task_id=agent_task_id,
        task_description=task_description,
        instructions=instructions,
        actions=actions,
        action_space_payload=action_space_payload,
        initial_observation=initial_observation,
    )
    return (
        "You are a controlled raw tool-calling baseline for one Data Center Twin "
        "benchmark problem.\n"
        "Use only the visible benchmark task description, instructions, available "
        "APIs, sanitized action space, and separately delivered observations.\n"
        "Do not use learned or hidden state summaries, hidden scenario fields, "
        "evaluator endpoints, or non-listed APIs.\n"
        "On every turn, invoke exactly one of the provider tools supplied with the "
        "request. Put the arguments in the provider-native structured tool call. "
        "Do not write an API call, JSON arguments, markdown code block, "
        "chain-of-thought, or explanatory text in assistant message content. "
        "Results from earlier calls "
        "are delivered in provider-native tool-result messages.\n"
        "The initial observation arrives separately as the first user message and "
        "is already evidence-eligible. Do not request the same snapshot again; "
        "observe only for a newer causal cut or deliberate drill-down. For final answers, "
        "use the submit(...) format required by the task instructions.\n\n"
        + json.dumps(context, indent=2, sort_keys=True, default=str)
    )


def raw_context_payload(
    *,
    agent_task_id: str,
    task_description: str,
    instructions: str,
    actions: dict[str, Any],
    action_space_payload: dict[str, Any] | None,
    initial_observation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return static benchmark-visible context with no telemetry payload."""
    # Preserve the public builder signature while keeping observations out of
    # system prompts. Agent implementations stage the value as a user message.
    del initial_observation
    return {
        "agent_task_id": agent_task_id,
        "task_description": task_description,
        "instructions": instructions,
        "available_task_apis": actions,
        "agent_action_space": sanitize_action_space(action_space_payload),
    }


def sanitize_action_space(
    action_space_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if isinstance(action_space_payload, dict):
        return sanitize_agent_action_space(action_space_payload)
    return action_space_payload


def sanitize_observation(observation: dict[str, Any] | None) -> dict[str, Any] | None:
    if isinstance(observation, dict):
        sanitized = sanitize_agent_payload(observation)
        return sanitized if isinstance(sanitized, dict) else observation
    return observation


def fence_api_call(call: str) -> str:
    return f"```\n{call.strip()}\n```"
