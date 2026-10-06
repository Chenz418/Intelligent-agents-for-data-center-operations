"""Shared protocol and helpers for Data Center Twin baseline agents."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
import json

from aiopslab.orchestrator.problems.data_center_twin.visibility import (
    sanitize_agent_payload,
    sanitize_agent_action_space,
)


TOKEN_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


class TerminationReason(StrEnum):
    """Controlled reasons why a benchmark episode stopped.

    These values describe control flow, independently of evaluator success.
    Existing ``status`` and ``final_state`` fields remain available for
    backward-compatible, more detailed reporting.
    """

    FINAL_SUBMISSION = "final_submission"
    EVALUATOR_COMPLETE = "evaluator_complete"
    TURN_LIMIT = "turn_limit"
    WALL_CLOCK_TIMEOUT = "wall_clock_timeout"
    AGENT_ERROR = "agent_error"
    BENCHMARK_ERROR = "benchmark_error"
    INVALID_SUBMISSION = "invalid_submission"
    CANCELLED = "cancelled"
    AGENT_EXIT_WITHOUT_SUBMISSION = "agent_exit_without_submission"


def zero_token_usage() -> dict[str, int]:
    return {key: 0 for key in TOKEN_USAGE_KEYS}


def unknown_token_usage() -> dict[str, None]:
    return {key: None for key in TOKEN_USAGE_KEYS}


def build_initial_observation_message(
    initial_observation: dict[str, Any] | None,
) -> str | None:
    """Serialize one sanitized initial observation as a user message.

    ``sanitize_agent_payload`` understands canonical and StateBundle payloads
    while preserving the separate ``agent.telemetry.compact.v1`` rendering.
    Using the legacy observation allowlist here would erase compact tables.
    """
    if initial_observation is None:
        return None
    sanitized = (
        sanitize_agent_payload(initial_observation)
        if isinstance(initial_observation, dict)
        else initial_observation
    )
    return (
        "Initial agent-visible observation (current causal cut; do not request "
        "the same snapshot again):\n"
        + json.dumps(
            sanitized,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def compact_action_space(
    action_space_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep action artifacts concise while preserving executable action names."""
    if not isinstance(action_space_payload, dict):
        return action_space_payload
    action_space_payload = sanitize_agent_action_space(action_space_payload)
    return {
        "agent_actions": action_space_payload.get("agent_actions"),
        "read_actions": sorted((action_space_payload.get("read_actions") or {}).keys()),
        "time_actions": sorted((action_space_payload.get("time_actions") or {}).keys()),
        "control_actions": sorted(
            (action_space_payload.get("control_actions") or {}).keys()
        ),
        "task_action_scope": action_space_payload.get("task_action_scope"),
    }
