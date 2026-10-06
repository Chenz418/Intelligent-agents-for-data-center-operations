"""The supported DC-Bench CLI, observation paths, and result contract."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import pytest

from scripts import evaluate_data_center_twin as runner
from aiopslab.orchestrator.problems.data_center_twin.visibility import (
    assert_no_agent_leakage,
)
from clients.data_center_twin_baselines.openai_compatible import (
    OpenAICompatibleSettings,
)
from clients.data_center_twin_baselines.raw_tool_calling import RawToolCallingAgent


def test_canonical_72_task_manifest():
    registry = runner.ProblemRegistry()
    ids = registry.get_problem_ids()
    assert len(ids) == len(set(ids)) == 72
    tasks = [registry.get_problem_instance(name) for name in ids]
    assert {task.scenario.task_type for task in tasks} == {
        "detection",
        "localization",
        "analysis",
        "mitigation",
    }
    assert len({task.scenario.fault_type for task in tasks}) == 18
    for task in tasks:
        assert set(task.get_available_actions()) == runner.RUNNER_ALLOWED_TASK_ACTIONS
        assert task.telemetry_view == "canonical"


@pytest.mark.parametrize("agent", ["tool-calling", "codex"])
@pytest.mark.parametrize("condition", ["full-canonical", "statebundle"])
def test_only_supported_cli_combinations(agent, condition):
    args = runner.parse_args(["--agent-type", agent, "--observation", condition])
    assert args.agent == agent and args.observation_condition == condition
    assert not hasattr(args, "diagnostic_evaluator")
    assert not hasattr(args, "telemetry_view")


@pytest.mark.parametrize(
    "args",
    [
        ["--agent-type", "random_controller"],
        ["--observation", "trunc-b"],
        ["--diagnostic-evaluator", "legacy"],
        ["--observation", "full-compact"],
    ],
)
def test_removed_cli_modes_are_rejected(args):
    with pytest.raises(SystemExit):
        runner.parse_args(args)


def test_semantic_input_cannot_fall_back_to_hidden_canonical_history():
    task = runner.ProblemRegistry().get_problem_instance(
        "data_center_twin-cooling_degradation-detection-1"
    )
    task.agent_action_history = [
        {"action_name": "dc_twin_observe", "response": {"hidden": "not delivered"}}
    ]
    assert task._agent_visible_observations() == []
    compact = {"schema_version": "agent.telemetry.compact.v1", "tables": {}}
    task._record_agent_rendered_observation_for_evaluation(compact)
    assert task._agent_visible_observations() == [compact]
    compact["oracle"] = "later mutation"
    assert "oracle" not in task._agent_visible_observations()[0]
    with pytest.raises(ValueError, match="rendered compact"):
        task._record_agent_rendered_observation_for_evaluation(
            {"schema_version": "statebundle.canonical.v1"}
        )


@pytest.mark.parametrize("condition", ["full-canonical", "statebundle"])
@pytest.mark.parametrize("task_type", ["detection", "mitigation"])
def test_native_tool_loop_both_observations_and_evaluators(
    condition, task_type, tmp_path, monkeypatch, semantic_transport
):
    args = runner.parse_args(
        ["--observation", condition, "--model", "test", "--output-dir", str(tmp_path)]
    )
    runner.prepare_observation_condition(args)
    agent = RawToolCallingAgent(
        OpenAICompatibleSettings(
            api_key="test", model="test", reasoning_effort="medium"
        )
    )
    messages_seen = []
    # The fixture provides known effective simulator controls solely to test the
    # native tool protocol and mitigation verification, not agent intelligence.
    calls = [("dc_twin_observe", {"channels": ["metric", "alert"]})]
    if task_type == "mitigation":
        scenario = (
            runner.ProblemRegistry()
            .get_problem_instance("data_center_twin-cooling_degradation-mitigation-1")
            .scenario
        )
        action = deepcopy(scenario.success_criteria["mitigation_action"])
        calls.append(("dc_twin_action", action))
        calls.append(("dc_twin_action", {"action_type": "step", "ticks": 10}))
        calls.append(("submit", {}))
    else:
        calls.append(
            (
                "submit",
                {
                    "payload": {
                        "incident_detected": True,
                        "diagnosis": "Cooling capacity reduced",
                        "evidence": ["Cooling capacity dropped"],
                    }
                },
            )
        )

    async def provider(messages):
        messages_seen.append(deepcopy(messages))
        name, arguments = calls[len(messages_seen) - 1]
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call-{len(messages_seen)}",
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
    monkeypatch.setattr(runner, "create_agent", lambda _args: agent)
    result = asyncio.run(
        runner.run_problem(
            f"data_center_twin-cooling_degradation-{task_type}-1",
            args=args,
            output_dir=tmp_path,
            max_steps=8,
            timeout_seconds=120,
            seed=None,
            agent_name="tool-calling",
            debug_logs=False,
            verbose=False,
        )
    )
    assert result["errors"] == []
    assert result["termination_reason"] == "final_submission"
    assert result["success"] is True
    initial = result["initial_agent_visible_observation"]
    assert initial["schema_version"] == "agent.telemetry.compact.v1"
    assert initial["condition"] == condition
    assert_no_agent_leakage(initial)
    if condition == "statebundle":
        assert initial["source_schema_version"] == "statebundle.output.v1"
        assert initial["budget"]["actual_serialized_tokens"] <= 4096
        assert Path(result["statebundle"]["checkpoint_path"]).is_relative_to(
            runner.REPO_ROOT
        )
    else:
        assert initial["source_schema_version"] == "statebundle.canonical.v1"
    if task_type == "detection":
        assert len(semantic_transport) == 1
        assert semantic_transport[0]["agent_visible_observation_history"][0] == initial
        assert result["evaluator_results"]["diagnostic_evaluator"] == "semantic"
    else:
        assert semantic_transport == []
        assert result["evaluator_results"]["stable_recovery_succeeded"] is True
    assert messages_seen[1][-1]["role"] == "tool"
    assert messages_seen[1][-1]["tool_call_id"] == "call-1"
    runner.write_results(
        tmp_path,
        [result],
        run_config=runner.evaluation_run_config(args, [result["problem_id"]]),
    )
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["results"][0]["success"] is True
    assert (tmp_path / "results.csv").exists() and (tmp_path / "results.md").exists()
    assert "api_key" not in payload["run_config"]


def test_checkpoint_loader_rejects_incompatible_tensors(tmp_path):
    import torch
    from aiopslab.statebundle.runtime import StateBundleObservationProcessor

    args = runner.parse_args(["--observation", "statebundle"])
    payload = torch.load(
        args.statebundle_checkpoint, map_location="cpu", weights_only=True
    )
    key = next(iter(payload["model_state"]))
    payload["model_state"][key] = torch.zeros(1)
    checkpoint = tmp_path / "bad.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="tensor mismatch"):
        StateBundleObservationProcessor(
            config_path=args.statebundle_config, checkpoint_path=checkpoint
        )


def test_reasoning_setting_passes_through_to_provider():
    agent = RawToolCallingAgent(
        OpenAICompatibleSettings(
            api_key="test", model="gpt-5.6-sol", reasoning_effort="medium"
        )
    )
    payload = agent._chat_completion_payload([])
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["reasoning_effort"] == "medium"
