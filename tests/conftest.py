"""Local protocol fixtures; live provider calls belong to explicit smoke runs."""

import json
import pytest
from aiopslab.orchestrator.problems.data_center_twin import (
    semantic_evaluation as semantic,
)


@pytest.fixture
def semantic_transport(monkeypatch):
    """Return schema-valid judgments for harness tests, without testing semantics."""
    requests = []

    def transport(config, body):
        evaluation_input = json.loads(body["messages"][-1]["content"])
        requests.append(evaluation_input)
        task = evaluation_input["task_type"]
        result = {key: True for key in semantic.TASK_GATES[task]}
        result.update(success=True, reason="Test transport: schema and routing only.")
        if "evidence_supported" in result:
            result["supporting_evidence"] = []
        if "evidence_supported" in result:
            history = evaluation_input["agent_visible_observation_history"]
            if history:
                result["supporting_evidence"] = [
                    {
                        "answer_excerpt": "test answer",
                        "observation_index": 0,
                        "observation_excerpt": "test observation",
                        "explanation": "test transport",
                    }
                ]
            else:
                result["evidence_supported"] = False
                result["success"] = False
        return {
            "model": config.model,
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(result)}}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

    monkeypatch.setattr(semantic, "_request", transport)
    return requests


@pytest.fixture
def native_submit_agent(monkeypatch):
    """Provider fixture that sends a native submission through the real agent."""
    from scripts import evaluate_data_center_twin as runner
    from clients.data_center_twin_baselines.raw_tool_calling import RawToolCallingAgent
    from clients.data_center_twin_baselines.openai_compatible import (
        OpenAICompatibleSettings,
    )

    agent = RawToolCallingAgent(OpenAICompatibleSettings(api_key="test", model="test"))

    async def response(messages):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "test-call",
                                "type": "function",
                                "function": {
                                    "name": "submit",
                                    "arguments": json.dumps(
                                        {
                                            "payload": {
                                                "incident_detected": True,
                                                "diagnosis": "cooling failure",
                                            }
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr(agent, "_call_tool_llm", response)
    monkeypatch.setattr(runner, "create_agent", lambda args: agent)
    return agent
