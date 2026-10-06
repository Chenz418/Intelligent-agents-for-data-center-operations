"""Process-quality metrics for Data Center Twin baseline episodes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
import json
import re
from typing import Any, Callable

from aiopslab.orchestrator.problems.data_center_twin.scenarios import (
    READ_TIME_ACTIONS as SCENARIO_READ_TIME_ACTIONS,
    WRITE_ACTIONS as SCENARIO_WRITE_ACTIONS,
)

from .base import TOKEN_USAGE_KEYS, unknown_token_usage, zero_token_usage
from aiopslab.orchestrator.problems.data_center_twin.semantic_evaluation import success_statistics


READ_ACTION_TYPES = frozenset(SCENARIO_READ_TIME_ACTIONS)
WRITE_ACTION_TYPES = frozenset(SCENARIO_WRITE_ACTIONS)
TASK_TYPES = ("detection", "localization", "analysis", "mitigation")


class InvalidActionCategory(StrEnum):
    """Stable audit categories for requests attributable to the agent."""

    PARSER_FAILURE = "parser_failure"
    UNSUPPORTED_ACTION = "unsupported_action"
    DISALLOWED_ACTION = "disallowed_action"
    INVALID_PARAMETERS = "invalid_parameters"
    INVALID_TARGET = "invalid_target"
    REJECTED_SUBMISSION = "rejected_submission"
    REJECTED_ACTION = "rejected_action"


@dataclass(frozen=True)
class ActionAttemptClassification:
    """Validity classification for one trajectory action record."""

    attempted: bool
    invalid: bool
    category: InvalidActionCategory | None = None
    reason: str | None = None
    source: str | None = None


def compute_episode_process_metrics(
    action_sequence: list[dict[str, Any]] | None,
    errors: list[dict[str, Any]] | None = None,
    *,
    runtime_seconds: Any = None,
    token_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute process metrics from one sanitized Data Center Twin trajectory."""
    actions = [item for item in (action_sequence or []) if isinstance(item, dict)]
    parse_error_records = sum(1 for item in actions if is_parse_error_record(item))
    parser_error_count = sum(
        1
        for error in (errors or [])
        if isinstance(error, dict) and str(error.get("phase", "")).lower() == "parser"
    )
    parse_error_count = max(parse_error_records, parser_error_count)

    classifications = [classify_action_attempt(item) for item in actions]
    invalid_action_details = [
        invalid_action_detail(item, classification, index)
        for index, (item, classification) in enumerate(zip(actions, classifications))
        if classification.attempted and classification.invalid
    ]
    orphan_parse_error_count = max(0, parse_error_count - parse_error_records)
    parser_errors = [
        error
        for error in (errors or [])
        if isinstance(error, dict) and str(error.get("phase", "")).lower() == "parser"
    ]
    for orphan_index in range(orphan_parse_error_count):
        error = parser_errors[parse_error_records + orphan_index]
        reason = normalize_scalar(error.get("message")) or "parser failure"
        invalid_action_details.append(
            {
                "step": None,
                "api_name": None,
                "category": InvalidActionCategory.PARSER_FAILURE.value,
                "reason": reason,
                "source": "parser_error",
            }
        )

    attempted_actions = sum(1 for item in classifications if item.attempted) + orphan_parse_error_count
    invalid_actions = sum(
        1 for item in classifications if item.attempted and item.invalid
    ) + orphan_parse_error_count
    valid_action_count = attempted_actions - invalid_actions
    invalid_actions_by_category = Counter(
        item["category"] for item in invalid_action_details
    )

    tool_records = [item for item in actions if is_tool_record(item)]
    write_records = [item for item in actions if is_write_record(item)]
    observe_count = sum(1 for item in actions if is_observe_record(item))
    submit_steps = [step_number(item, index) for index, item in enumerate(actions) if item.get("api_name") == "submit"]
    first_write_steps = [step_number(item, index) for index, item in enumerate(actions) if is_write_record(item)]

    histogram = Counter(action_histogram_label(item) for item in actions)
    redundant_count = count_redundant_actions(tool_records)
    tool_call_count = len(tool_records)

    return {
        "tool_call_count": tool_call_count,
        "valid_action_count": valid_action_count,
        # Keep the historical singular field while exposing manuscript names.
        "invalid_action_count": invalid_actions,
        "attempted_actions": attempted_actions,
        "invalid_actions": invalid_actions,
        "invalid_action_rate": invalid_actions / attempted_actions if attempted_actions else 0.0,
        "invalid_actions_by_category": dict(sorted(invalid_actions_by_category.items())),
        "invalid_action_details": invalid_action_details,
        "parse_error_count": parse_error_count,
        "redundant_action_count": redundant_count,
        "redundant_action_rate": redundant_count / tool_call_count if tool_call_count else 0.0,
        "zero_tool_diagnosis": zero_tool_diagnosis(actions),
        "first_write_step": min(first_write_steps) if first_write_steps else None,
        "steps_to_submit": min(submit_steps) if submit_steps else None,
        "observe_count": observe_count,
        "write_action_count": len(write_records),
        "action_type_histogram": dict(sorted(histogram.items())),
        "runtime_seconds": float(runtime_seconds)
        if isinstance(runtime_seconds, int | float) and not isinstance(runtime_seconds, bool)
        else None,
        "token_usage": normalized_token_usage(token_usage),
    }


def aggregate_process_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate process metrics across a result slice."""
    metrics = [ensure_process_metrics(item) for item in results]
    count = len(metrics)
    histogram: Counter[str] = Counter()
    token_usage: dict[str, Any] = zero_token_usage()
    all_known_input_tokens = 0
    all_known_output_tokens = 0
    all_known_total_tokens = 0
    known_input_usage_count = 0
    known_output_usage_count = 0
    known_total_usage_count = 0
    known_token_usage_count = 0
    unknown_token_usage_count = 0
    for item in metrics:
        histogram.update(item.get("action_type_histogram") or {})
        usage = normalized_token_usage(item.get("token_usage"))
        input_value = usage.get("prompt_tokens")
        output_value = usage.get("completion_tokens")
        if is_token_count(input_value):
            all_known_input_tokens += int(input_value)
            known_input_usage_count += 1
        if is_token_count(output_value):
            all_known_output_tokens += int(output_value)
            known_output_usage_count += 1
        total_value = usage.get("total_tokens")
        if is_token_count(total_value):
            all_known_total_tokens += int(total_value)
            known_total_usage_count += 1
        if is_explicit_unknown_token_usage(usage):
            unknown_token_usage_count += 1
            continue
        known_token_usage_count += 1
        for key in TOKEN_USAGE_KEYS:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                token_usage[key] += value
    known_partial_usage = {
        "prompt_tokens": all_known_input_tokens,
        "completion_tokens": all_known_output_tokens,
        "total_tokens": all_known_total_tokens,
        "input_tokens": all_known_input_tokens,
        "output_tokens": all_known_output_tokens,
    }
    if unknown_token_usage_count:
        token_usage.update(unknown_token_usage())
        token_usage["known_partial_token_usage"] = known_partial_usage
    token_usage["known_episode_count"] = known_token_usage_count
    token_usage["unknown_episode_count"] = unknown_token_usage_count
    token_usage["token_usage_available"] = unknown_token_usage_count == 0
    input_complete = known_input_usage_count == count
    output_complete = known_output_usage_count == count
    if input_complete:
        token_usage["prompt_tokens"] = all_known_input_tokens
    if output_complete:
        token_usage["completion_tokens"] = all_known_output_tokens
    token_usage["input_tokens"] = token_usage.get("prompt_tokens")
    token_usage["output_tokens"] = token_usage.get("completion_tokens")
    token_usage["input_token_usage_available"] = input_complete
    token_usage["output_token_usage_available"] = output_complete
    token_usage["known_input_tokens"] = all_known_input_tokens
    token_usage["known_output_tokens"] = all_known_output_tokens
    token_usage["known_total_tokens"] = all_known_total_tokens
    token_usage["known_input_episode_count"] = known_input_usage_count
    token_usage["known_output_episode_count"] = known_output_usage_count
    token_usage["known_total_episode_count"] = known_total_usage_count

    def total(name: str) -> int:
        return sum(int(item.get(name) or 0) for item in metrics)

    def average(name: str) -> float | None:
        values = [
            float(item[name])
            for item in metrics
            if isinstance(item.get(name), int | float) and not isinstance(item.get(name), bool)
        ]
        return sum(values) / len(values) if values else None

    total_tool_calls = total("tool_call_count")
    total_redundant = total("redundant_action_count")
    total_attempted_actions = sum(episode_attempted_actions(item) for item in metrics)
    total_invalid_actions = sum(episode_invalid_actions(item) for item in metrics)
    invalid_actions_by_category: Counter[str] = Counter()
    for item in metrics:
        categories = item.get("invalid_actions_by_category")
        if not isinstance(categories, dict):
            continue
        for category, value in categories.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                invalid_actions_by_category[str(category)] += value
    zero_tool_count = sum(1 for item in metrics if item.get("zero_tool_diagnosis"))
    return {
        "episode_count": count,
        "tool_call_count": total_tool_calls,
        "valid_action_count": total("valid_action_count"),
        "invalid_action_count": total_invalid_actions,
        "attempted_actions": total_attempted_actions,
        "invalid_actions": total_invalid_actions,
        "invalid_action_rate": (
            total_invalid_actions / total_attempted_actions
            if total_attempted_actions
            else 0.0
        ),
        "average_invalid_action_rate": average("invalid_action_rate"),
        "invalid_actions_by_category": dict(sorted(invalid_actions_by_category.items())),
        "parse_error_count": total("parse_error_count"),
        "redundant_action_count": total_redundant,
        "redundant_action_rate": total_redundant / total_tool_calls if total_tool_calls else 0.0,
        "average_redundant_action_rate": average("redundant_action_rate"),
        "zero_tool_diagnosis_count": zero_tool_count,
        "zero_tool_diagnosis_rate": zero_tool_count / count if count else 0.0,
        "average_first_write_step": average("first_write_step"),
        "average_steps_to_submit": average("steps_to_submit"),
        "observe_count": total("observe_count"),
        "write_action_count": total("write_action_count"),
        "average_runtime_seconds": average("runtime_seconds"),
        "action_type_histogram": dict(sorted(histogram.items())),
        "total_token_usage": token_usage,
    }


def aggregate_process_metrics_by(
    results: list[dict[str, Any]],
    key_func: Callable[[dict[str, Any]], str | None],
) -> dict[str, dict[str, Any]]:
    """Aggregate process metrics for non-empty groups selected by key_func."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        key = key_func(item)
        if key:
            grouped[key].append(item)
    return {
        key: {
            **aggregate_process_metrics(items),
            **success_statistics(items),
            "average_score": average_score(items),
        }
        for key, items in sorted(grouped.items())
    }


def episode_attempted_actions(metrics: dict[str, Any]) -> int:
    """Return the additive attempted-action total with legacy fallback."""
    value = metrics.get("attempted_actions")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    valid = metrics.get("valid_action_count")
    invalid = metrics.get("invalid_action_count")
    return sum(
        value
        for value in (valid, invalid)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    )


def episode_invalid_actions(metrics: dict[str, Any]) -> int:
    """Return the additive invalid-action total with legacy fallback."""
    for key in ("invalid_actions", "invalid_action_count"):
        value = metrics.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def ensure_process_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Return existing process metrics or compute them from a result record."""
    metrics = result.get("process_metrics")
    if isinstance(metrics, dict):
        return metrics
    return compute_episode_process_metrics(
        result.get("action_sequence") if isinstance(result.get("action_sequence"), list) else [],
        result.get("errors") if isinstance(result.get("errors"), list) else [],
        runtime_seconds=result.get("runtime_seconds"),
        token_usage=result.get("token_usage") if isinstance(result.get("token_usage"), dict) else None,
    )


def process_metric_group_key(result: dict[str, Any], field: str) -> str | None:
    """Return an explicit result field or derive task/fault from the problem ID."""
    value = result.get(field)
    if value:
        return str(value)
    problem_id = str(result.get("problem_id") or "")
    derived_fault, derived_task = derive_fault_task_from_problem_id(problem_id)
    if field == "task_type":
        return derived_task
    if field == "fault_type":
        return derived_fault
    return None


def derive_fault_task_from_problem_id(problem_id: str) -> tuple[str | None, str | None]:
    """Derive fault and task type from canonical Data Center Twin problem IDs."""
    pattern = re.compile(
        r"^data_center_twin-(?P<fault>.+)-(?P<task>detection|localization|analysis|mitigation)-\d+$"
    )
    match = pattern.match(problem_id)
    if not match:
        return None, None
    return match.group("fault"), match.group("task")


def normalized_token_usage(token_usage: dict[str, Any] | None) -> dict[str, Any]:
    """Return normalized usage, preserving explicit unknown LLM accounting."""
    explicit_unknown = is_explicit_unknown_token_usage(token_usage)
    normalized: dict[str, Any] = unknown_token_usage() if explicit_unknown else zero_token_usage()
    if explicit_unknown:
        normalized["token_usage_available"] = False
    if isinstance(token_usage, dict):
        for key in TOKEN_USAGE_KEYS:
            value = token_usage.get(key)
            if is_token_count(value):
                normalized[key] = value
            elif explicit_unknown:
                normalized[key] = None
    normalized["input_tokens"] = normalized.get("prompt_tokens")
    normalized["output_tokens"] = normalized.get("completion_tokens")
    normalized["input_token_usage_available"] = is_token_count(
        normalized.get("prompt_tokens")
    )
    normalized["output_token_usage_available"] = is_token_count(
        normalized.get("completion_tokens")
    )
    return normalized


def is_explicit_unknown_token_usage(token_usage: dict[str, Any] | None) -> bool:
    if not isinstance(token_usage, dict):
        return True
    if token_usage.get("token_usage_available") is False:
        return True
    return any(
        not is_token_count(token_usage.get(key))
        for key in TOKEN_USAGE_KEYS
    )


def is_token_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_parse_error_record(record: dict[str, Any]) -> bool:
    return bool(record.get("parse_error")) or record.get("api_name") is None


def classify_action_attempt(record: dict[str, Any]) -> ActionAttemptClassification:
    """Classify one attempted action once, using structured evidence first.

    Provider and environment failures are deliberately not agent-invalid.  In
    particular, rate limiting, request timeouts, and server-side HTTP failures
    remain attempted actions but do not enter the invalid-action numerator.
    """
    if not isinstance(record, dict):
        return ActionAttemptClassification(attempted=False, invalid=False)

    explicit = explicit_action_classification(record)
    if explicit is not None:
        return explicit

    response = record.get("env_response")
    response_payload = parse_jsonish(response)
    response_reason = action_response_reason(response_payload)
    response_text = response_reason or normalize_scalar(response)

    # A terminal rejected submission has a precise category independent of
    # the rendering used for the SubmissionStatus enum.
    if (
        record.get("api_name") == "submit"
        and normalize_scalar(response).strip().lower() == "invalid_submission"
    ):
        return invalid_classification(
            InvalidActionCategory.REJECTED_SUBMISSION,
            response_text or "invalid submission",
            "submission_status",
        )

    normalization_error = normalize_scalar(record.get("normalization_error")).strip()
    if record.get("normalized_ok") is False or normalization_error:
        category = category_from_invalid_text(normalization_error)
        # Unsupported provider tool names and malformed tool arguments retain
        # their more precise category; all other normalization failures are
        # parser/structured-response failures.
        if category is None or category not in {
            InvalidActionCategory.UNSUPPORTED_ACTION,
            InvalidActionCategory.DISALLOWED_ACTION,
            InvalidActionCategory.INVALID_PARAMETERS,
            InvalidActionCategory.INVALID_TARGET,
        }:
            category = InvalidActionCategory.PARSER_FAILURE
        return invalid_classification(
            category,
            normalization_error or response_text or "model action normalization failed",
            "provider_normalization",
        )

    if is_parse_error_record(record):
        parse_error = record.get("parse_error")
        reason = (
            normalize_scalar(parse_error.get("message"))
            if isinstance(parse_error, dict)
            else ""
        )
        return invalid_classification(
            InvalidActionCategory.PARSER_FAILURE,
            reason or response_text or "response parser failure",
            "parser",
        )

    if record.get("api_name") == "invalid_tool_command" or record.get("invalid_action") is True:
        category = category_from_invalid_text(response_text)
        if category is None:
            category = InvalidActionCategory.UNSUPPORTED_ACTION
        return invalid_classification(
            category,
            response_text or "invalid tool command",
            "tool_wrapper",
        )

    http_status = structured_http_status(response_payload)
    if http_status is not None:
        if http_status in {408, 429} or http_status >= 500:
            return ActionAttemptClassification(attempted=True, invalid=False)
        if 400 <= http_status < 500:
            category = category_from_invalid_text(response_text)
            if category is None:
                category = InvalidActionCategory.REJECTED_ACTION
            if record.get("api_name") == "submit":
                category = InvalidActionCategory.REJECTED_SUBMISSION
            return invalid_classification(
                category,
                response_text or f"HTTP {http_status} request rejection",
                "structured_response",
            )
        # An explicit non-error status is authoritative; do not subsequently
        # classify incidental words in a successful response as invalid.
        return ActionAttemptClassification(attempted=True, invalid=False)

    if isinstance(response_payload, dict) and response_payload.get("accepted") is False:
        category = category_from_invalid_text(response_text)
        return invalid_classification(
            category or InvalidActionCategory.REJECTED_ACTION,
            response_text or "action rejected by environment",
            "structured_response",
        )

    if isinstance(response_payload, dict):
        # Do not search arbitrary successful telemetry/action payloads for
        # words such as "forbidden" or "not available". Those commonly occur
        # as configuration field names and are not rejection signals.
        if "error" in response_payload:
            category = category_from_invalid_text(response_reason)
            if category is not None:
                return invalid_classification(
                    category,
                    response_reason or "structured request rejection",
                    "structured_response",
                )
        return ActionAttemptClassification(attempted=True, invalid=False)
    if isinstance(response_payload, list):
        return ActionAttemptClassification(attempted=True, invalid=False)

    legacy_category = category_from_legacy_invalid_response(response_text)
    if legacy_category is not None:
        return invalid_classification(
            legacy_category,
            response_text or "invalid action",
            "legacy_response",
        )

    return ActionAttemptClassification(attempted=True, invalid=False)


def explicit_action_classification(
    record: dict[str, Any],
) -> ActionAttemptClassification | None:
    """Read an optional harness annotation without requiring it for legacy traces."""
    accounting = record.get("action_accounting")
    if not isinstance(accounting, dict):
        return None
    attempted = accounting.get("attempted", True) is not False
    invalid = accounting.get("invalid") is True
    category = invalid_action_category(accounting.get("category")) if invalid else None
    if invalid and category is None:
        category = InvalidActionCategory.REJECTED_ACTION
    reason = normalize_scalar(accounting.get("reason")).strip() or None
    source = normalize_scalar(accounting.get("source")).strip() or "harness"
    return ActionAttemptClassification(
        attempted=attempted,
        invalid=invalid,
        category=category,
        reason=reason,
        source=source,
    )


def invalid_classification(
    category: InvalidActionCategory,
    reason: str,
    source: str,
) -> ActionAttemptClassification:
    return ActionAttemptClassification(
        attempted=True,
        invalid=True,
        category=category,
        reason=reason,
        source=source,
    )


def invalid_action_category(value: Any) -> InvalidActionCategory | None:
    if isinstance(value, InvalidActionCategory):
        return value
    if isinstance(value, str):
        try:
            return InvalidActionCategory(value)
        except ValueError:
            return None
    return None


def structured_http_status(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("http_status")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def action_response_reason(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        for key in ("detail", "message", "error"):
            value = error.get(key)
            if value is not None:
                return normalize_reason(value)
        return normalize_reason(error)
    if error is not None:
        return normalize_reason(error)
    detail = payload.get("detail")
    return normalize_reason(detail) if detail is not None else ""


def normalize_reason(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return stable_json(value)


def category_from_invalid_text(text: str) -> InvalidActionCategory | None:
    lowered = normalize_scalar(text).strip().lower()
    if not lowered:
        return None
    if "submission" in lowered and any(
        marker in lowered for marker in ("invalid", "reject", "not allowed")
    ):
        return InvalidActionCategory.REJECTED_SUBMISSION
    if any(
        marker in lowered
        for marker in (
            "for this task",
            "not allowed",
            "not available",
            "forbidden",
            "episode already submitted",
            "max tool steps exceeded",
        )
    ):
        return InvalidActionCategory.DISALLOWED_ACTION
    if any(
        marker in lowered
        for marker in (
            "not found",
            "does not exist",
            "invalid target",
            "unknown target",
            "target resource",
        )
    ):
        return InvalidActionCategory.INVALID_TARGET
    if any(
        marker in lowered
        for marker in (
            "invalid tool command",
            "unsupported command",
            "unsupported tool",
            "unsupported provider tool name",
            "unsupported action",
            "unsupported dc_twin_action action_type",
            "unsupported agent action_type",
        )
    ):
        return InvalidActionCategory.UNSUPPORTED_ACTION
    if any(
        marker in lowered
        for marker in (
            "parameter",
            "argument",
            "requires ",
            "required ",
            "must be",
            "out of range",
            "conflicting value",
            "invalid control action",
            "invalid json",
            "valid json",
            "json object",
        )
    ):
        return InvalidActionCategory.INVALID_PARAMETERS
    return None


def category_from_legacy_invalid_response(text: str) -> InvalidActionCategory | None:
    lowered = normalize_scalar(text).strip().lower()
    if lowered == "invalid_submission":
        return InvalidActionCategory.REJECTED_SUBMISSION
    category = category_from_invalid_text(lowered)
    if category is not None:
        return category
    if "invalid action" in lowered or "invalidactionerror" in lowered:
        return InvalidActionCategory.UNSUPPORTED_ACTION
    return None


def invalid_action_detail(
    record: dict[str, Any],
    classification: ActionAttemptClassification,
    index: int,
) -> dict[str, Any]:
    return {
        "step": step_number(record, index),
        "api_name": record.get("api_name"),
        "category": (
            classification.category.value
            if classification.category is not None
            else InvalidActionCategory.REJECTED_ACTION.value
        ),
        "reason": classification.reason or "invalid action",
        "source": classification.source,
    }


def is_invalid_record(record: dict[str, Any]) -> bool:
    """Backward-compatible Boolean facade over centralized classification."""
    return classify_action_attempt(record).invalid


def is_valid_record(record: dict[str, Any]) -> bool:
    classification = classify_action_attempt(record)
    return classification.attempted and not classification.invalid


def is_tool_record(record: dict[str, Any]) -> bool:
    api_name = record.get("api_name")
    return bool(api_name) and api_name != "submit"


def is_observe_record(record: dict[str, Any]) -> bool:
    if record.get("api_name") == "dc_twin_observe":
        return True
    return record.get("api_name") == "dc_twin_action" and dc_twin_action_type(record) == "observe"


def is_write_record(record: dict[str, Any]) -> bool:
    return record.get("api_name") == "dc_twin_action" and dc_twin_action_type(record) in WRITE_ACTION_TYPES


def dc_twin_action_type(record: dict[str, Any]) -> str | None:
    args = record.get("args")
    if isinstance(args, list) and args:
        return str(args[0])
    kwargs = record.get("kwargs")
    if isinstance(kwargs, dict) and kwargs.get("action_type"):
        return str(kwargs["action_type"])
    return None


def action_signature(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("api_name")),
        stable_json(record.get("args", [])),
        stable_json(record.get("kwargs", {})),
    )


def action_histogram_label(record: dict[str, Any]) -> str:
    if is_parse_error_record(record):
        return "parse_error"
    api_name = str(record.get("api_name"))
    if api_name == "dc_twin_action":
        action_type = dc_twin_action_type(record)
        if action_type:
            return f"dc_twin_action:{action_type}"
    return api_name


def count_redundant_actions(records: list[dict[str, Any]]) -> int:
    """Count conservative repeated actions with unchanged responses."""
    redundant = 0
    previous_by_signature: dict[tuple[str, str, str], str] = {}
    for record in records:
        if not is_valid_record(record):
            continue
        signature = action_signature(record)
        response_signature = material_response_signature(record.get("env_response"))
        if previous_by_signature.get(signature) == response_signature and not response_has_state_change(
            record.get("env_response")
        ):
            redundant += 1
        else:
            previous_by_signature[signature] = response_signature
    return redundant


def response_has_state_change(response: Any) -> bool:
    payload = parse_jsonish(response)
    if isinstance(payload, dict):
        action_result = payload.get("action_result")
        if isinstance(action_result, dict):
            status = normalize_scalar(action_result.get("status")).lower()
            if status in {"ok", "success", "succeeded", "applied"}:
                return True
            if action_result.get("success") is True:
                return True
        status = normalize_scalar(payload.get("status")).lower()
        if status in {"ok", "success", "succeeded", "applied"}:
            return True
        if payload.get("success") is True:
            return True
    return False


def material_response_signature(response: Any) -> str:
    return stable_json(parse_jsonish(response))


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return json.dumps(str(value))


def normalize_scalar(value: Any) -> str:
    return "" if value is None else str(value)


def step_number(record: dict[str, Any], index: int) -> int:
    step = record.get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        return step
    return index + 1


def zero_tool_diagnosis(actions: list[dict[str, Any]]) -> bool:
    if not actions:
        return False
    first = actions[0]
    if first.get("api_name") != "submit" or is_invalid_record(first):
        return False
    return not any(is_tool_record(item) for item in actions[:1])


def average_score(results: list[dict[str, Any]]) -> float | None:
    scores = [
        float(item["score"])
        for item in results
        if isinstance(item.get("score"), int | float) and not isinstance(item.get("score"), bool)
    ]
    return sum(scores) / len(scores) if scores else None
