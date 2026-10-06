"""Exercise native tool calls through both standard observation conditions."""

import argparse
import asyncio
import json

import pytest

from clients.data_center_twin_baselines.openai_compatible import (
    OpenAICompatibleSettings,
)
from clients.data_center_twin_baselines.raw_tool_calling import RawToolCallingAgent
from scripts import evaluate_data_center_twin as evaluator


@pytest.mark.parametrize("condition", ["full-canonical", "statebundle"])
def test_native_tool_episode_with_standard_observations(
    condition, monkeypatch, tmp_path
):
    agent = RawToolCallingAgent(
        OpenAICompatibleSettings(
            api_key="test",
            base_url="http://unused.invalid",
            model="test-provider",
        )
    )
    messages_seen = []

    async def provider(messages):
        messages_seen.append(json.loads(json.dumps(messages)))
        index = len(messages_seen)
        name = "dc_twin_observe" if index == 1 else "submit"
        arguments = (
            {"channels": ["metric", "alert"], "lookback_seconds": 300}
            if index == 1
            else {
                "payload": {"incident_detected": True, "fault": "cooling degradation"}
            }
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call-{index}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr(agent, "_call_tool_llm", provider)
    monkeypatch.setattr(evaluator, "create_agent", lambda args: agent)
    args = argparse.Namespace(
        agent="tool-calling",
        telemetry_view="canonical",
        observation_condition=condition,
        observation_token_budget=4096,
        statebundle_config=evaluator.REPO_ROOT / "configs" / "statebundle.dc_twin.yaml",
        statebundle_checkpoint=(
            evaluator.REPO_ROOT
            / "checkpoints"
            / "statebundle-stage3"
            / "stage3-epoch-9.pt"
        ),
    )
    evaluator.prepare_observation_condition(args)
    result = asyncio.run(
        evaluator.run_problem(
            "data_center_twin-cooling_degradation-detection-1",
            args=args,
            output_dir=tmp_path,
            max_steps=3,
            timeout_seconds=120,
            seed=42,
            agent_name="tool-calling",
            deterministic=True,
            agent_task_nonce="standard-smoke",
            debug_logs=True,
            verbose=False,
        )
    )
    assert result["errors"] == []
    assert result["observation_condition"] == condition
    assert len(messages_seen) == 2
    assert messages_seen[1][-1]["role"] == "tool"
    assert messages_seen[1][-1]["tool_call_id"] == "call-1"
    delivered = result["initial_agent_visible_observation"]
    assert delivered["schema_version"] == "agent.telemetry.compact.v1"
    assert delivered["tables"]
    assert "agent.telemetry." in messages_seen[1][-1]["content"]
    if condition == "statebundle":
        assert delivered["source_schema_version"] == "statebundle.output.v1"
        assert delivered["budget"]["actual_serialized_tokens"] <= 4096
    else:
        assert delivered["source_schema_version"] == "statebundle.canonical.v1"
    evaluator.write_results(tmp_path, [result])
    artifact = json.loads((tmp_path / "results.json").read_text())
    assert artifact["results"][0]["errors"] == []


pytestmark = pytest.mark.usefixtures("semantic_transport")
