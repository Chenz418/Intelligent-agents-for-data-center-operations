"""Native Gemini tool/history, HTTP, and usage contracts; no live API calls."""

import asyncio
from copy import deepcopy
import json

import pytest

from clients.data_center_twin_baselines.gemini_native import generate_content_url
from clients.data_center_twin_baselines.openai_compatible import (
    LLMProviderModelMismatchError,
    LLMProviderTransportError,
    OpenAICompatibleSettings,
)
from clients.data_center_twin_baselines.raw_tool_calling import (
    RawToolCallingAgent,
    data_center_twin_tool_schemas,
)
from clients.data_center_twin_baselines.token_accounting import (
    TokenUsageLedger,
    gemini_usage_counts,
)


def settings(**kwargs):
    return OpenAICompatibleSettings(
        provider="gemini_native", model="gemini-2.5-flash-lite",
        base_url="https://gemini.example/v1beta", api_key="unit-test-key", **kwargs,
    )


def native_response(*, args=None, name="dc_twin_observe", native_id=None):
    call = {"name": name, "args": {} if args is None else args}
    if native_id:
        call["id"] = native_id
    return {
        "candidates": [{"content": {"role": "model", "parts": [
            {"text": "native thought", "thought": True, "thoughtSignature": "sig-one"},
            {"functionCall": call, "thoughtSignature": "sig-two"},
        ]}, "finishReason": "STOP"}],
        "modelVersion": "gemini-2.5-flash-lite", "responseId": "native-response-id",
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10,
                          "thoughtsTokenCount": 5, "totalTokenCount": 115,
                          "cachedContentTokenCount": 40},
    }


@pytest.mark.parametrize("native_id", [None, "provider-call-123"])
def test_two_turns_preserve_native_signatures_and_tool_result(native_id):
    response = native_response(native_id=native_id)
    issued = []

    class Agent(RawToolCallingAgent):
        def _request_json(self, url, payload):
            issued.append((url, deepcopy(payload)))
            return deepcopy(response)

    agent = Agent(settings())
    agent._initialize_messages("Use exactly one function per turn.")
    first = asyncio.run(agent.get_action("Visible observation"))
    second = asyncio.run(agent.get_action("STATEBUNDLE unchanged bytes\n{}"))
    assert "dc_twin_observe()" in first and "dc_twin_observe()" in second
    assert len(issued) == 2
    url, payload = issued[-1]
    assert url == "https://gemini.example/v1beta/models/gemini-2.5-flash-lite:generateContent"
    assert "model" not in payload and "messages" not in payload
    assert payload["contents"][1] == response["candidates"][0]["content"]
    result = payload["contents"][2]["parts"][0]["functionResponse"]
    assert result["name"] == "dc_twin_observe"
    assert result["response"] == {"output": "STATEBUNDLE unchanged bytes\n{}"}
    assert result.get("id") == native_id
    assert payload["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert payload["generationConfig"] == {"candidateCount": 1, "temperature": 0.0, "maxOutputTokens": 1024}
    assert agent.token_usage["input_tokens"] == 200
    assert agent.token_usage["output_tokens"] == 30
    assert agent.model_call_token_usage[0]["provider_usage"] == response["usageMetadata"]
    assert json.loads(agent.last_normalization_status["original_model_output"]) == response
    assert agent.provider_transport_attempts[0]["server_reported_model"] == response["modelVersion"]
    assert agent.provider_transport_attempts[0]["native_response"] == response


def test_native_json_schema_keeps_free_form_parameters_and_payload():
    agent = RawToolCallingAgent(settings(thinking_mode="disabled"))
    payload = agent._chat_completion_payload([{"role": "user", "content": "visible"}])
    payload["tools"] = data_center_twin_tool_schemas()
    native = agent._provider_request_payload(payload)
    for expected, actual in zip(payload["tools"], native["tools"][0]["functionDeclarations"]):
        assert actual["parametersJsonSchema"] == expected["function"]["parameters"]
    assert native["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


@pytest.mark.parametrize("args", [
    {"payload": {"diagnosis": "example", "incident_detected": True}},
    {"payload": ["server_a"]},
    {"payload": '{"diagnosis":"example"}'},
    {},
])
def test_native_submit_preserves_supported_answer_values(args):
    agent = RawToolCallingAgent(settings())
    normalized, call, error, original = agent._normalize_tool_response(native_response(name="submit", args=args))
    assert error is None
    assert call.startswith("submit(")
    assert json.loads(original)["candidates"][0]["content"]["parts"][1]["functionCall"]["args"] == args


def test_rejects_parallel_function_calls_and_multiple_candidates():
    agent = RawToolCallingAgent(settings())
    response = native_response()
    response["candidates"][0]["content"]["parts"].append({"functionCall": {"name": "submit", "args": {}}})
    assert "exactly one" in agent._normalize_tool_response(response)[2]
    response = native_response()
    response["candidates"].append(deepcopy(response["candidates"][0]))
    assert "exactly one candidate" in agent._normalize_tool_response(response)[2]


def test_native_http_auth_and_duplicate_argument_rejection(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"candidates": [{"content": {"parts": [{"functionCall": {"args": {"payload": 1, "payload": 2}}}]}}]}'

    def open_request(request, **kwargs):
        requests.append(request)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    agent = RawToolCallingAgent(settings())
    with pytest.raises(LLMProviderTransportError):
        agent._request_json(agent._chat_completion_url(), {"contents": []})
    assert requests[0].get_header("X-goog-api-key") == "unit-test-key"
    assert requests[0].get_header("Authorization") is None
    assert "unit-test-key" not in requests[0].full_url


def test_wrong_response_model_rejected_after_exact_usage_settlement():
    class Agent(RawToolCallingAgent):
        def _request_json(self, *args):
            response = native_response()
            response["modelVersion"] = "other-model"
            return response

    agent = Agent(settings())
    with pytest.raises(LLMProviderModelMismatchError):
        asyncio.run(agent.get_action("visible observation"))
    assert agent.emitted_calls == []
    assert agent.token_usage["total_tokens"] == 115
    assert agent.model_call_token_usage[0]["call_status"] == "completed"
    assert len(agent.provider_transport_attempts) == 1


@pytest.mark.parametrize("usage,expected", [
    ({"promptTokenCount": 100, "candidatesTokenCount": 10, "thoughtsTokenCount": 5, "totalTokenCount": 115}, (100, 15, 5)),
    ({"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 115}, (100, 15, 5)),
    ({"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 110}, (100, 10, 0)),
    ({"promptTokenCount": 100, "totalTokenCount": 115}, (100, 15, None)),
    ({"promptTokenCount": 100, "candidatesTokenCount": 10}, (100, None, None)),
    ({"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 99}, (100, None, None)),
    ({"promptTokenCount": 100, "candidatesTokenCount": 10, "thoughtsTokenCount": True, "totalTokenCount": 111}, (100, None, None)),
])
def test_native_usage_separate_thoughts_and_safe_missing_fields(usage, expected):
    counts = gemini_usage_counts(usage)
    assert (counts["input_tokens"], counts["output_tokens"], counts["reasoning_tokens"]) == expected
    assert counts["cached_input_tokens"] is None
    ledger = TokenUsageLedger(provider="gemini_native", model="gemini-2.5-flash-lite")
    record = ledger.record_call(request_payload={"contents": []}, response_payload={"usageMetadata": usage})
    assert record["provider_usage"] == usage
    assert record["token_count_source"] == ("provider_native" if expected[1] is not None else "estimated")


def test_native_url_refuses_model_substitution():
    with pytest.raises(ValueError, match="model does not match"):
        generate_content_url("https://gemini.example/v1beta/models/other:generateContent", "gemini-2.5-flash-lite")
