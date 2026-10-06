"""Contract tests for centralized Data Center Twin invalid-action accounting."""

from __future__ import annotations

import json
from typing import Any

import pytest

from clients.data_center_twin_baselines.metrics import (
    InvalidActionCategory,
    classify_action_attempt,
    compute_episode_process_metrics,
)


def action_record(
    *,
    step: int = 1,
    api_name: str | None = "dc_twin_action",
    env_response: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    """Return the action-record shape emitted by the controlled harnesses."""
    record = {
        "step": step,
        "raw": '```\ndc_twin_action("observe")\n```',
        "api_name": api_name,
        "args": ["observe"] if api_name == "dc_twin_action" else [],
        "kwargs": {},
        "env_response": env_response,
    }
    record.update(extra)
    return record


def structured_error(status: int, detail: str) -> str:
    """Render the same structured response representation stored in trajectories."""
    return json.dumps(
        {
            "http_status": status,
            "error": {
                "type": "SimulationError",
                "detail": detail,
            },
        },
        sort_keys=True,
    )


def assert_invalid(
    record: dict[str, Any],
    category: InvalidActionCategory,
) -> None:
    classification = classify_action_attempt(record)
    assert classification.attempted is True
    assert classification.invalid is True
    assert classification.category is category
    assert classification.reason


def test_parser_failure_is_one_invalid_attempt():
    record = action_record(
        api_name=None,
        env_response="Error parsing response: missing fenced API call",
        raw="not an API call",
        normalized_ok=False,
        normalization_error="no API call found",
        parse_error={
            "phase": "parser",
            "type": "ResponseParsingError",
            "message": "missing fenced API call",
        },
    )

    assert_invalid(record, InvalidActionCategory.PARSER_FAILURE)
    metrics = compute_episode_process_metrics(
        [record],
        [
            {
                "phase": "parser",
                "type": "ResponseParsingError",
                "message": "missing fenced API call",
            }
        ],
    )

    assert metrics["attempted_actions"] == 1
    assert metrics["invalid_actions"] == 1
    assert metrics["invalid_action_count"] == 1
    assert metrics["invalid_action_rate"] == 1.0


@pytest.mark.parametrize(
    ("record", "expected_category"),
    [
        (
            action_record(
                api_name="invalid_tool_command",
                raw=json.dumps(
                    {
                        "command": "not-a-tool",
                        "arguments": {},
                    }
                ),
                env_response="invalid tool command: not-a-tool",
                invalid_action=True,
            ),
            InvalidActionCategory.UNSUPPORTED_ACTION,
        ),
        (
            action_record(
                api_name="invalid_tool_command",
                raw=json.dumps(
                    {
                        "command": "action",
                        "arguments": {
                            "payload": {
                                "action_type": "set_cooling",
                                "parameters": {"target": "cooling-unit-1"},
                            }
                        },
                    }
                ),
                env_response="unsupported action_type for this task: set_cooling",
                invalid_action=True,
            ),
            InvalidActionCategory.DISALLOWED_ACTION,
        ),
    ],
)
def test_unsupported_and_task_disallowed_actions_are_distinguished(
    record: dict[str, Any],
    expected_category: InvalidActionCategory,
):
    assert_invalid(record, expected_category)


def test_structured_http_400_invalid_parameters_is_counted():
    record = action_record(
        args=["set_cooling"],
        kwargs={
            "parameters": {
                "target": "cooling-unit-1",
                "unknown_parameter": 1,
            }
        },
        env_response=structured_error(
            400,
            "set_cooling does not accept nested parameter(s): unknown_parameter",
        ),
    )

    assert_invalid(record, InvalidActionCategory.INVALID_PARAMETERS)


def test_structured_http_400_invalid_target_is_counted():
    record = action_record(
        args=["set_cooling"],
        kwargs={
            "parameters": {
                "target": "cooling-unit-does-not-exist",
                "fan_speed_percent": 100,
            }
        },
        env_response=structured_error(
            400,
            "cooling unit not found: cooling-unit-does-not-exist",
        ),
    )

    classification = classify_action_attempt(record)

    assert classification.attempted is True
    assert classification.invalid is True
    assert classification.category is InvalidActionCategory.INVALID_TARGET
    assert "cooling unit not found" in classification.reason


def test_unclassified_structured_http_400_is_a_rejected_action():
    record = action_record(
        env_response=structured_error(400, "request rejected by simulator"),
    )

    assert_invalid(record, InvalidActionCategory.REJECTED_ACTION)


def test_rejected_terminal_submission_is_invalid():
    record = action_record(
        api_name="submit",
        raw="```\nsubmit()\n```",
        env_response="INVALID_SUBMISSION",
    )

    assert_invalid(record, InvalidActionCategory.REJECTED_SUBMISSION)


def test_valid_but_ineffective_action_is_not_invalid():
    record = action_record(
        args=["set_cooling"],
        kwargs={
            "parameters": {
                "target": "cooling-unit-1",
                "fan_speed_percent": 70,
            }
        },
        env_response=json.dumps(
            {
                "accepted": True,
                "action_type": "set_cooling",
                "action_result": {
                    "status": "no_change",
                    "success": False,
                },
            },
            sort_keys=True,
        ),
    )

    classification = classify_action_attempt(record)

    assert classification.attempted is True
    assert classification.invalid is False
    assert classification.category is None


def test_successful_telemetry_words_do_not_trigger_legacy_invalid_markers():
    record = action_record(
        api_name="dc_twin_observe",
        env_response=json.dumps(
            {
                "summary": {"sla_status": "violated"},
                "configuration": {
                    "forbidden_rack_ids": [],
                    "optional_target": "not available",
                },
            },
            sort_keys=True,
        ),
    )

    classification = classify_action_attempt(record)

    assert classification.attempted is True
    assert classification.invalid is False


def test_one_attempt_with_multiple_failure_signals_is_counted_once():
    record = action_record(
        api_name=None,
        raw="provider emitted two tool calls",
        normalized_ok=False,
        normalization_error="expected exactly one provider tool call, got 2",
        parse_error={
            "phase": "parser",
            "type": "ResponseParsingError",
            "message": "no API call found",
        },
        env_response="Error parsing response: no API call found",
    )
    errors = [
        {
            "phase": "parser",
            "type": "ResponseParsingError",
            "message": "no API call found",
        }
    ]

    metrics = compute_episode_process_metrics([record], errors)

    assert metrics["attempted_actions"] == 1
    assert metrics["invalid_actions"] == 1
    assert metrics["invalid_action_count"] == 1
    assert metrics["invalid_actions_by_category"] == {
        InvalidActionCategory.PARSER_FAILURE.value: 1
    }
    assert len(metrics["invalid_action_details"]) == 1


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (500, "internal simulator failure"),
        (503, "simulator temporarily unavailable"),
        (429, "environment request rate limited"),
    ],
)
def test_server_and_environment_failures_are_not_agent_invalid(
    status: int,
    detail: str,
):
    classification = classify_action_attempt(
        action_record(env_response=structured_error(status, detail))
    )

    assert classification.attempted is True
    assert classification.invalid is False
    assert classification.category is None


def test_episode_denominator_rate_categories_and_details():
    records = [
        action_record(
            step=1,
            api_name=None,
            raw="malformed response",
            env_response="Error parsing response: no API call found",
            parse_error={"phase": "parser", "message": "no API call found"},
        ),
        action_record(
            step=2,
            api_name="invalid_tool_command",
            env_response="unsupported action_type for this task: set_cooling",
            invalid_action=True,
        ),
        action_record(
            step=3,
            args=["set_cooling"],
            env_response=structured_error(
                400,
                "cooling unit not found: cooling-unit-missing",
            ),
        ),
        action_record(
            step=4,
            api_name="dc_twin_observe",
            raw="```\ndc_twin_observe()\n```",
            env_response=json.dumps({"summary": {"sla_status": "violated"}}),
        ),
    ]

    metrics = compute_episode_process_metrics(records, [])

    assert metrics["attempted_actions"] == 4
    assert metrics["invalid_actions"] == 3
    assert metrics["invalid_action_count"] == 3
    assert metrics["valid_action_count"] == 1
    assert metrics["invalid_action_rate"] == pytest.approx(0.75)
    assert metrics["invalid_actions_by_category"] == {
        InvalidActionCategory.DISALLOWED_ACTION.value: 1,
        InvalidActionCategory.INVALID_TARGET.value: 1,
        InvalidActionCategory.PARSER_FAILURE.value: 1,
    }
    assert len(metrics["invalid_action_details"]) == 3
    assert {item["step"] for item in metrics["invalid_action_details"]} == {1, 2, 3}
    assert all(item["reason"] for item in metrics["invalid_action_details"])


def test_zero_attempt_convention_is_zero_rate():
    metrics = compute_episode_process_metrics([], [])

    assert metrics["attempted_actions"] == 0
    assert metrics["invalid_actions"] == 0
    assert metrics["invalid_action_count"] == 0
    assert metrics["invalid_action_rate"] == 0.0
    assert metrics["invalid_actions_by_category"] == {}
    assert metrics["invalid_action_details"] == []
