"""Focused contract tests for controlled-agent token accounting.

These tests intentionally exercise provider responses directly.  No network
request or paid model call is made.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest


def token_usage_ledger_class():
    """Import lazily so the missing production module is reported as test RED."""
    from clients.data_center_twin_baselines.token_accounting import TokenUsageLedger

    return TokenUsageLedger


def chat_request_payload() -> dict[str, Any]:
    return {
        "model": "gpt-5.6-luna",
        "messages": [
            {"role": "system", "content": "Use one benchmark tool per turn."},
            {"role": "user", "content": "Inspect the current telemetry."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "dc_twin_observe",
                    "description": "Inspect telemetry.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "required",
    }


def text_response(*, usage: dict[str, Any] | None) -> dict[str, Any]:
    response: dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "```\ndc_twin_observe()\n```",
                }
            }
        ]
    }
    if usage is not None:
        response["usage"] = usage
    return response


def tool_response(*, usage: dict[str, Any] | None) -> dict[str, Any]:
    response: dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "dc_twin_observe",
                                "arguments": '{"include_config":true,"log_limit":20}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    if usage is not None:
        response["usage"] = usage
    return response


def assert_legacy_and_canonical_totals(
    usage: dict[str, Any],
    *,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """The additive schema must retain the benchmark's legacy aliases."""
    assert usage["input_tokens"] == input_tokens
    assert usage["output_tokens"] == output_tokens
    assert usage["prompt_tokens"] == input_tokens
    assert usage["completion_tokens"] == output_tokens
    assert usage["total_tokens"] == input_tokens + output_tokens
    assert usage["token_usage_available"] is True


def test_pre_registered_call_transitions_in_place_without_persisting_payloads():
    TokenUsageLedger = token_usage_ledger_class()
    ledger = TokenUsageLedger(provider="openai", model="gpt-5.6-luna")
    request_marker = "unique-request-body-must-not-be-persisted"
    response_marker = "unique-late-response-must-not-be-persisted"
    request = chat_request_payload()
    request["messages"][-1]["content"] = request_marker
    response = tool_response(
        usage={"prompt_tokens": 43, "completion_tokens": 9, "total_tokens": 52}
    )
    response["choices"][0]["message"]["audit_marker"] = response_marker

    call_index = ledger.begin_call(request_payload=request)
    pending_record = ledger.call_records[0]

    assert call_index == 1
    assert pending_record["call_status"] == "pending"
    assert pending_record["token_count_source"] == "unavailable"
    assert pending_record["input_token_count_source"] == "estimated"
    assert isinstance(pending_record["input_tokens"], int)
    assert pending_record["input_tokens"] > 0
    assert pending_record["output_tokens"] is None
    assert pending_record["request_payload_sha256"]
    assert ledger.usage["pending_call_count"] == 1
    assert ledger.usage["input_token_usage_available"] is True
    assert ledger.usage["output_token_usage_available"] is False

    assert (
        ledger.mark_in_flight_calls_accounting_only(
            termination_reason="wall_clock_timeout",
            termination_timestamp="2026-08-23T12:00:00+00:00",
        )
        == 1
    )
    completed_record = ledger.complete_call(
        call_index,
        request_payload=request,
        response_payload=response,
    )

    assert completed_record is pending_record
    assert len(ledger.call_records) == 1
    assert completed_record["call_index"] == call_index
    assert completed_record["call_status"] == "completed"
    assert completed_record["token_count_source"] == "provider_native"
    assert completed_record["input_tokens"] == 43
    assert completed_record["output_tokens"] == 9
    assert completed_record["total_tokens"] == 52
    assert completed_record["completed_after_episode_termination"] is True
    assert (
        completed_record["trajectory_disposition"]
        == "accounting_only_after_episode_termination"
    )
    assert completed_record["response_payload_sha256"]
    assert completed_record["accounting_completed_at"]
    assert_legacy_and_canonical_totals(ledger.usage, input_tokens=43, output_tokens=9)
    assert ledger.usage["pending_call_count"] == 0
    assert ledger.usage["late_completed_call_count"] == 1

    serialized_record = json.dumps(completed_record, sort_keys=True)
    assert request_marker not in serialized_record
    assert response_marker not in serialized_record
    assert "messages" not in completed_record
    assert "choices" not in completed_record


def test_native_provider_usage_is_retained_per_call_and_aggregated():
    TokenUsageLedger = token_usage_ledger_class()
    ledger = TokenUsageLedger(provider="openai", model="gpt-5.6-luna")
    provider_usage = {
        "prompt_tokens": 31,
        "completion_tokens": 7,
        "total_tokens": 38,
        "prompt_tokens_details": {"cached_tokens": 3},
    }

    ledger.record_call(
        request_payload=chat_request_payload(),
        response_payload=text_response(usage=provider_usage),
    )

    assert_legacy_and_canonical_totals(ledger.usage, input_tokens=31, output_tokens=7)
    assert len(ledger.call_records) == 1
    record = ledger.call_records[0]
    assert record["input_tokens"] == 31
    assert record["output_tokens"] == 7
    assert record["total_tokens"] == 38
    assert record["token_count_source"] == "provider_native"
    assert record["estimator"] is None
    assert record["provider_usage"] == provider_usage


def test_native_input_output_aliases_are_supported():
    TokenUsageLedger = token_usage_ledger_class()
    ledger = TokenUsageLedger(provider="openai_compatible", model="provider-model")

    ledger.record_call(
        request_payload=chat_request_payload(),
        response_payload=text_response(
            usage={"input_tokens": 13, "output_tokens": 5, "total_tokens": 18}
        ),
    )

    assert_legacy_and_canonical_totals(ledger.usage, input_tokens=13, output_tokens=5)
    assert ledger.call_records[0]["token_count_source"] == "provider_native"


def test_authoritative_native_zero_is_not_reclassified_as_missing():
    TokenUsageLedger = token_usage_ledger_class()
    ledger = TokenUsageLedger(provider="openai", model="gpt-5.6-luna")

    ledger.record_call(
        request_payload=chat_request_payload(),
        response_payload=text_response(
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ),
    )

    assert_legacy_and_canonical_totals(ledger.usage, input_tokens=0, output_tokens=0)
    assert ledger.call_records[0]["token_count_source"] == "provider_native"
    assert ledger.call_records[0]["estimator"] is None


def test_missing_usage_uses_a_nonzero_deterministic_estimate():
    TokenUsageLedger = token_usage_ledger_class()
    request = chat_request_payload()
    response = text_response(usage=None)
    first = TokenUsageLedger(provider="openai", model="gpt-5.6-luna")
    second = TokenUsageLedger(provider="openai", model="gpt-5.6-luna")

    first.record_call(request_payload=request, response_payload=response)
    second.record_call(request_payload=request, response_payload=response)

    first_record = first.call_records[0]
    second_record = second.call_records[0]
    assert first_record["token_count_source"] == "estimated"
    assert isinstance(first_record["estimator"], str) and first_record["estimator"]
    assert "tiktoken:o200k_base" in first_record["estimator"]
    assert first_record["input_tokens"] > 0
    assert first_record["output_tokens"] > 0
    assert {
        key: first_record[key]
        for key in ("input_tokens", "output_tokens", "total_tokens", "estimator")
    } == {
        key: second_record[key]
        for key in ("input_tokens", "output_tokens", "total_tokens", "estimator")
    }
    assert_legacy_and_canonical_totals(
        first.usage,
        input_tokens=first_record["input_tokens"],
        output_tokens=first_record["output_tokens"],
    )


def test_multiple_native_and_estimated_calls_sum_from_call_records():
    TokenUsageLedger = token_usage_ledger_class()
    ledger = TokenUsageLedger(provider="openai", model="gpt-5.6-luna")

    ledger.record_call(
        request_payload=chat_request_payload(),
        response_payload=text_response(
            usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
        ),
    )
    second_request = chat_request_payload()
    second_request["messages"] = [
        *second_request["messages"],
        {"role": "assistant", "content": "```\ndc_twin_observe()\n```"},
        {"role": "user", "content": "Now submit the final result."},
    ]
    ledger.record_call(
        request_payload=second_request,
        response_payload=tool_response(usage=None),
    )

    assert [record["token_count_source"] for record in ledger.call_records] == [
        "provider_native",
        "estimated",
    ]
    expected_input = sum(record["input_tokens"] for record in ledger.call_records)
    expected_output = sum(record["output_tokens"] for record in ledger.call_records)
    assert_legacy_and_canonical_totals(
        ledger.usage,
        input_tokens=expected_input,
        output_tokens=expected_output,
    )


def test_genuinely_impossible_estimation_is_unavailable_not_zero(monkeypatch):
    TokenUsageLedger = token_usage_ledger_class()
    from clients.data_center_twin_baselines import token_accounting

    def fail_estimation(**_kwargs):
        raise RuntimeError("no deterministic tokenizer could encode this payload")

    monkeypatch.setattr(token_accounting, "_estimate_call_tokens", fail_estimation)
    ledger = TokenUsageLedger(provider="unknown", model="unknown")

    ledger.record_call(
        request_payload=chat_request_payload(),
        response_payload=text_response(usage=None),
    )

    assert ledger.call_records[0]["token_count_source"] == "unavailable"
    assert ledger.call_records[0]["input_tokens"] is None
    assert ledger.call_records[0]["output_tokens"] is None
    assert ledger.usage["prompt_tokens"] is None
    assert ledger.usage["completion_tokens"] is None
    assert ledger.usage["total_tokens"] is None
    assert ledger.usage["token_usage_available"] is False
    assert ledger.usage["warnings"]


def test_terminal_provider_failure_is_unavailable_not_known_zero():
    from clients.data_center_twin_baselines.openai_compatible import (
        LLMProviderRequestError,
        OpenAICompatibleSettings,
    )

    from clients.data_center_twin_baselines.raw_tool_calling import RawToolCallingAgent

    class FailedProviderAgent(RawToolCallingAgent):
        def _request_json(self, _url: str, _payload: dict[str, Any]) -> dict[str, Any]:
            raise LLMProviderRequestError(
                status_code=500,
                reason="Internal Server Error",
                body='{"error":{"message":"provider failed"}}',
                message="LLM provider request failed (500 Internal Server Error)",
            )

    agent = FailedProviderAgent(
        OpenAICompatibleSettings(
            api_key="test-key",
            base_url="https://llm.example/v1",
            provider="openai",
            model="gpt-5.6-luna",
        )
    )

    with pytest.raises(LLMProviderRequestError):
        agent._chat_completion_with_tools(
            [{"role": "user", "content": "Inspect telemetry."}]
        )

    assert len(agent.model_call_token_usage) == 1
    call = agent.model_call_token_usage[0]
    assert call["token_count_source"] == "unavailable"
    assert isinstance(call["input_tokens"], int) and call["input_tokens"] > 0
    assert call["output_tokens"] is None
    assert call["total_tokens"] is None
    assert call["warnings"]
    assert agent.token_usage["prompt_tokens"] == call["input_tokens"]
    assert agent.token_usage["input_tokens"] == call["input_tokens"]
    assert agent.token_usage["input_token_usage_available"] is True
    assert agent.token_usage["completion_tokens"] is None
    assert agent.token_usage["output_tokens"] is None
    assert agent.token_usage["output_token_usage_available"] is False
    assert agent.token_usage["total_tokens"] is None
    assert agent.token_usage["token_usage_available"] is False


@pytest.mark.asyncio
async def test_raw_tool_completion_records_each_provider_response_exactly_once():
    TokenUsageLedger = token_usage_ledger_class()
    from clients.data_center_twin_baselines.openai_compatible import (
        OpenAICompatibleSettings,
    )
    from clients.data_center_twin_baselines.raw_tool_calling import (
        RawToolCallingAgent,
        data_center_twin_tool_schemas,
    )

    response = tool_response(usage=None)

    class MissingUsageRawToolAgent(RawToolCallingAgent):
        def _request_json(self, _url: str, _payload: dict[str, Any]) -> dict[str, Any]:
            return response

    settings = OpenAICompatibleSettings(
        api_key="test-key",
        base_url="https://llm.example/v1",
        provider="openai",
        model="gpt-5.6-luna",
    )
    messages = [{"role": "user", "content": "Inspect telemetry with a tool."}]
    expected_payload = MissingUsageRawToolAgent(settings)._chat_completion_payload(
        messages
    )
    expected_payload.update(
        {"tools": data_center_twin_tool_schemas(), "tool_choice": "required"}
    )
    expected = TokenUsageLedger(provider=settings.provider, model=settings.model)
    expected.record_call(request_payload=expected_payload, response_payload=response)
    agent = MissingUsageRawToolAgent(settings)

    assert await agent._call_tool_llm(messages) == response
    assert_legacy_and_canonical_totals(
        agent.token_usage,
        input_tokens=expected.usage["input_tokens"],
        output_tokens=expected.usage["output_tokens"],
    )
    # Matching one-call totals catches both a missed record and duplicate
    # recording between the shared transport and raw-tool caller.
    assert agent.token_usage["total_tokens"] == expected.call_records[0]["total_tokens"]


@pytest.mark.asyncio
async def test_cancelled_raw_tool_turn_captures_late_native_usage_without_action_mutation():
    from clients.data_center_twin_baselines.openai_compatible import (
        OpenAICompatibleSettings,
    )
    from clients.data_center_twin_baselines.raw_tool_calling import RawToolCallingAgent

    request_started = threading.Event()
    release_response = threading.Event()
    request_marker = "cancelled-turn-request-body-marker"
    response_marker = "late-submit-response-body-marker"
    response = tool_response(
        usage={"prompt_tokens": 127, "completion_tokens": 11, "total_tokens": 138}
    )
    late_tool = response["choices"][0]["message"]["tool_calls"][0]["function"]
    late_tool["name"] = "submit"
    late_tool["arguments"] = json.dumps({"payload": response_marker})

    class GatedLateResponseAgent(RawToolCallingAgent):
        def _request_json(
            self,
            _url: str,
            _payload: dict[str, Any],
        ) -> dict[str, Any]:
            request_started.set()
            if not release_response.wait(timeout=5.0):
                raise AssertionError("test did not release the gated provider response")
            return response

    agent = GatedLateResponseAgent(
        OpenAICompatibleSettings(
            api_key="test-key",
            base_url="https://llm.example/v1",
            provider="openai",
            model="gpt-5.6-luna",
        )
    )
    agent._initialize_messages("static-system-instructions")
    action_task = asyncio.create_task(agent.get_action(request_marker))

    try:
        assert await asyncio.to_thread(request_started.wait, 2.0)
        assert len(agent.model_call_token_usage) == 1
        pending_record = agent.model_call_token_usage[0]
        assert pending_record["call_status"] == "pending"
        assert pending_record["token_count_source"] == "unavailable"
        assert pending_record["request_payload_sha256"]
        assert isinstance(pending_record["input_tokens"], int)
        assert pending_record["input_tokens"] > 0

        action_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await action_task
        assert (
            agent.mark_episode_terminated_for_accounting(
                "wall_clock_timeout",
                termination_timestamp="2026-08-23T12:00:00+00:00",
            )
            == 1
        )

        # Start settlement while the worker is still blocked, then let only its
        # accounting result cross the closed episode boundary.
        settlement_task = asyncio.create_task(agent.settle_provider_accounting())
        await asyncio.sleep(0)
        release_response.set()
        settlement = await settlement_task

        assert settlement["status"] == "settled"
        assert settlement["tracked_call_count"] == 1
        assert settlement["unfinished_call_count_at_settlement_start"] == 1
        assert settlement["pending_call_count_at_settlement_end"] == 0
        assert settlement["late_completed_call_count"] == 1
        assert settlement["late_failed_call_count"] == 0
        assert settlement["episode_state_frozen_before_settlement"] is True

        assert agent.model_call_token_usage == [pending_record]
        assert pending_record["call_status"] == "completed"
        assert pending_record["token_count_source"] == "provider_native"
        assert pending_record["input_tokens"] == 127
        assert pending_record["output_tokens"] == 11
        assert pending_record["total_tokens"] == 138
        assert pending_record["completed_after_episode_termination"] is True
        assert (
            pending_record["trajectory_disposition"]
            == "accounting_only_after_episode_termination"
        )
        assert_legacy_and_canonical_totals(
            agent.token_usage,
            input_tokens=127,
            output_tokens=11,
        )

        # The provider response may update audit accounting only. In particular,
        # its late submit must never be normalized into agent decision history.
        assert [message["role"] for message in agent.messages] == ["system", "user"]
        assert agent.messages[-1]["content"] == request_marker
        assert agent.emitted_calls == []
        assert agent.last_normalization_status is None
        serialized_accounting = json.dumps(
            agent.model_call_token_usage,
            sort_keys=True,
        )
        assert request_marker not in serialized_accounting
        assert response_marker not in serialized_accounting
        assert "choices" not in pending_record
        assert "messages" not in pending_record
    finally:
        release_response.set()
        if not action_task.done():
            action_task.cancel()
        await asyncio.gather(action_task, return_exceptions=True)
        await agent.settle_provider_accounting()


@pytest.mark.asyncio
async def test_cancelled_provider_failure_retains_issued_input_estimate():
    from clients.data_center_twin_baselines.openai_compatible import (
        LLMProviderRequestError,
        OpenAICompatibleSettings,
    )
    from clients.data_center_twin_baselines.raw_tool_calling import RawToolCallingAgent

    request_started = threading.Event()
    release_failure = threading.Event()

    class GatedLateFailureAgent(RawToolCallingAgent):
        def _request_json(
            self,
            _url: str,
            _payload: dict[str, Any],
        ) -> dict[str, Any]:
            request_started.set()
            if not release_failure.wait(timeout=5.0):
                raise AssertionError("test did not release the gated provider failure")
            raise LLMProviderRequestError(
                status_code=500,
                reason="Internal Server Error",
                body='{"error":{"message":"late provider failure"}}',
                message="LLM provider request failed (500 Internal Server Error)",
            )

    agent = GatedLateFailureAgent(
        OpenAICompatibleSettings(
            api_key="test-key",
            base_url="https://llm.example/v1",
            provider="openai",
            model="gpt-5.6-luna",
        )
    )
    agent._initialize_messages("static-system-instructions")
    action_task = asyncio.create_task(agent.get_action("request-that-will-fail-late"))

    try:
        assert await asyncio.to_thread(request_started.wait, 2.0)
        pending_record = agent.model_call_token_usage[0]
        issued_input_estimate = pending_record["input_tokens"]
        assert isinstance(issued_input_estimate, int) and issued_input_estimate > 0

        action_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await action_task
        agent.mark_episode_terminated_for_accounting(
            "wall_clock_timeout",
            termination_timestamp="2026-08-23T12:00:00+00:00",
        )
        settlement_task = asyncio.create_task(agent.settle_provider_accounting())
        await asyncio.sleep(0)
        release_failure.set()
        settlement = await settlement_task

        assert settlement["status"] == "settled"
        assert settlement["late_completed_call_count"] == 0
        assert settlement["late_failed_call_count"] == 1
        assert len(agent.model_call_token_usage) == 1
        failed_record = agent.model_call_token_usage[0]
        assert failed_record is pending_record
        assert failed_record["call_status"] == "failed"
        assert failed_record["input_tokens"] == issued_input_estimate
        assert failed_record["input_token_count_source"] == "estimated"
        assert failed_record["output_tokens"] is None
        assert failed_record["output_token_count_source"] == "unavailable"
        assert failed_record["total_tokens"] is None
        assert failed_record["token_count_source"] == "unavailable"
        assert failed_record["request_payload_sha256"]
        assert failed_record["response_payload_sha256"] is None
        assert failed_record["completed_after_episode_termination"] is True
        assert failed_record["warnings"]
        assert agent.token_usage["input_tokens"] == issued_input_estimate
        assert agent.token_usage["input_token_usage_available"] is True
        assert agent.token_usage["output_tokens"] is None
        assert agent.token_usage["output_token_usage_available"] is False
        assert agent.token_usage["token_usage_available"] is False
    finally:
        release_failure.set()
        if not action_task.done():
            action_task.cancel()
        await asyncio.gather(action_task, return_exceptions=True)
        await agent.settle_provider_accounting()


@pytest.mark.asyncio
async def test_completed_raw_tool_response_can_be_rolled_back_to_accounting_only():
    from clients.data_center_twin_baselines.openai_compatible import (
        OpenAICompatibleSettings,
    )
    from clients.data_center_twin_baselines.raw_tool_calling import RawToolCallingAgent

    response = tool_response(
        usage={"prompt_tokens": 23, "completion_tokens": 5, "total_tokens": 28}
    )
    function = response["choices"][0]["message"]["tool_calls"][0]["function"]
    function["name"] = "submit"
    function["arguments"] = '{"payload":{"incident_detected":true}}'

    class ImmediateResponseAgent(RawToolCallingAgent):
        def _request_json(self, _url: str, _payload: dict[str, Any]) -> dict[str, Any]:
            return response

    agent = ImmediateResponseAgent(
        OpenAICompatibleSettings(
            api_key="test-key",
            base_url="https://llm.example/v1",
            provider="openai",
            model="gpt-5.6-luna",
        )
    )
    agent._initialize_messages("static-system-instructions")

    action = await agent.get_action("boundary-turn")
    assert "submit" in action
    assert [message["role"] for message in agent.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert len(agent.emitted_calls) == 1
    assert agent.last_normalization_status is not None

    # This is the runner's absolute-deadline race path: the provider finished,
    # but the decision boundary closed before the action could be dispatched.
    agent.discard_late_provider_response()
    agent.mark_episode_terminated_for_accounting(
        "wall_clock_timeout",
        termination_timestamp="2026-08-23T12:00:00+00:00",
    )

    assert [message["role"] for message in agent.messages] == ["system", "user"]
    assert agent.emitted_calls == []
    assert agent.last_normalization_status is None
    assert agent.token_usage["input_tokens"] == 23
    assert agent.token_usage["output_tokens"] == 5
    record = agent.model_call_token_usage[0]
    assert record["call_status"] == "completed"
    assert record["completed_after_episode_termination"] is True
    assert record["trajectory_disposition"] == (
        "accounting_only_after_episode_termination"
    )
