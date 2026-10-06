from copy import deepcopy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest


READ_TIME_ACTIONS = ["observe", "noop", "step"]
WRITE_ACTIONS = [
    "calibrate_sensor",
    "set_cooling",
    "migrate_workload",
    "throttle_workload",
    "update_autoscaler_policy",
    "repair_monitoring_pipeline",
    "update_placement_policy",
    "update_load_balancer_config",
    "set_server_maintenance",
    "clear_server_maintenance",
]
AGENT_ACTIONS = READ_TIME_ACTIONS + WRITE_ACTIONS
FORBIDDEN_AGENT_KEYS = {
    "active_faults",
    "active_faults_after",
    "active_faults_before",
    "benchmark_action_coverage",
    "fault_id",
    "fault_target",
    "fault_type",
    "ground_truth",
    "incident_domains",
    "initial_seed",
    "oracle",
    "reset_config",
    "scenario_setup",
    "score_hints",
    "seed",
    "supported_faults",
}
FORBIDDEN_AGENT_SUBSTRINGS = {
    "active fault",
    "active_fault",
    "application_error",
    "autoscaler_misconfiguration",
    "control_plane_degradation",
    "cooling_degradation",
    "fault_injected",
    "injected",
    "intermittent_server_failure",
    "load_balancer_misconfiguration",
    "network_congestion_burst",
    "network_partition",
    "monitoring_pipeline_failure",
    "placement_policy_misconfiguration",
    "power_overload",
    "power_budget_violation",
    "rack_hotspot",
    "server_failure",
    "storage_io_saturation",
    "thermal_sensor_miscalibration",
    "thermal_throttling",
    "tor_packet_loss",
}


def assert_no_forbidden_agent_leakage(payload):
    leaks = []

    def walk(value, path="$"):
        if isinstance(value, dict):
            forbidden = set(value) & FORBIDDEN_AGENT_KEYS
            for key in sorted(forbidden):
                leaks.append(f"forbidden key {path}.{key}")
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(payload)
    text = json.dumps(payload, sort_keys=True, default=str).lower()
    for substring in sorted(FORBIDDEN_AGENT_SUBSTRINGS):
        if substring in text:
            leaks.append(f"forbidden substring {substring!r}")
    assert not leaks, "\n".join(leaks)


class FakeKubeCtl:
    pass


def load_scenarios_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "aiopslab"
        / "orchestrator"
        / "problems"
        / "data_center_twin"
        / "scenarios.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_data_center_twin_scenarios", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["test_data_center_twin_scenarios"] = module
    spec.loader.exec_module(module)
    return module


def load_evaluate_data_center_twin_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "evaluate_data_center_twin.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_evaluate_data_center_twin", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeLLMToolResponse:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

    def to_dict(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": self.name,
                                    "arguments": json.dumps(self.arguments),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


def raw_baseline_kwargs(task_type="detection"):
    submit_instructions = {
        "detection": (
            "This task is a Data Center Twin detection task. "
            'Final answer format: submit({"incident_detected": True, '
            '"suspected_category": "unknown", "target": None, "evidence": []})'
        ),
        "localization": (
            "This task is a Data Center Twin localization task. "
            'Final answer format: submit(["component-id-from-telemetry"])'
        ),
        "analysis": (
            "This task is a Data Center Twin analysis task. "
            'Final answer format: submit({"root_cause_category": "unknown", '
            '"target": None, "domain": "unknown", "evidence": []})'
        ),
        "mitigation": (
            "This task is a Data Center Twin mitigation task. "
            "Use dc_twin_action(...) for controls, verify, then final answer format: submit()"
        ),
    }
    return {
        "agent_task_id": "dcopslab-episode-test",
        "task_description": f"You are operating a Data Center Twin {task_type} task.",
        "instructions": submit_instructions[task_type],
        "actions": {
            "dc_twin_action_space": "Return the action schema.",
            "dc_twin_observe": "Return current telemetry.",
            "dc_twin_action": "Apply an allowed agent action.",
            "submit": "Submit the task.",
        },
        "action_space_payload": {
            "agent_actions": AGENT_ACTIONS + ["inject_fault"],
            "read_actions": {"observe": {"parameters": {"log_limit": "integer"}}},
            "time_actions": {"noop": {}, "step": {}},
            "control_actions": {"set_cooling": {"parameters": {"target": "string"}}},
            "domain_contract": {
                "incident_domains": {
                    "cooling_degradation": {"answer": "cooling-unit-1"}
                }
            },
            "benchmark_action_coverage": {"oracle": ["set_cooling"]},
        },
        "initial_observation": {
            "episode_id": "episode-visible",
            "summary": {
                "sla_status": "violated",
                "thermal_critical": 1,
                "active_faults": [
                    {"fault_type": "cooling_degradation", "target": "cooling-unit-1"}
                ],
            },
            "alerts": [
                {
                    "alert_type": "active_fault",
                    "message": "Active fault cooling_degradation",
                },
                {
                    "alert_type": "RackTemperatureHigh",
                    "message": "Rack temperature is high",
                },
            ],
            "recent_events": [
                {
                    "event_type": "fault_injected",
                    "message": "cooling_degradation injected",
                },
                {"event_type": "workload_started", "message": "Workload started"},
            ],
        },
    }


def write_fake_blackbox_agent(tmp_path, source):
    script = tmp_path / "fake_agent.py"
    script.write_text(source, encoding="utf-8")
    return script


def blackbox_config(
    tmp_path, command_template, timeout_seconds=30.0, keep_workspace=False
):
    from clients.data_center_twin_baselines.blackbox_harness import (
        BlackboxEpisodeConfig,
    )

    return BlackboxEpisodeConfig(
        agent_name="codex",
        command_template=command_template,
        output_dir=tmp_path,
        max_steps=20,
        timeout_seconds=timeout_seconds,
        keep_workspace=keep_workspace,
        sandbox_mode="none",
    )


def cleanup_retained_blackbox_workspace(result):
    workspace = result.get("blackbox", {}).get("workspace")
    if workspace:
        shutil.rmtree(Path(workspace).parent, ignore_errors=True)


class FakeRawLLMAgentMixin:
    def __init__(self, responses):
        from clients.data_center_twin_baselines.openai_compatible import (
            OpenAICompatibleSettings,
        )

        super().__init__(
            OpenAICompatibleSettings(
                api_key="test-key",
                base_url="https://llm.example/v1",
                model="fake-model",
            )
        )
        self.responses = list(responses)
        self.requests = []

    def _request_json(self, _url, payload):
        self.requests.append(payload)
        assert self.responses
        return self.responses.pop(0).to_dict()


def fake_raw_tool_agent_class():
    from clients.data_center_twin_baselines.raw_tool_calling import RawToolCallingAgent

    class FakeRawToolCallingAgent(FakeRawLLMAgentMixin, RawToolCallingAgent):
        pass

    return FakeRawToolCallingAgent


@pytest.mark.asyncio
async def test_data_center_twin_raw_tool_calling_uses_provider_tools():
    from aiopslab.orchestrator.parser import ResponseParser

    agent = fake_raw_tool_agent_class()(
        [
            FakeLLMToolResponse(
                "dc_twin_observe", {"log_limit": 20, "include_config": True}
            )
        ]
    )
    agent.init_context(**raw_baseline_kwargs())

    response = await agent.get_action("Please take the next action")
    parsed = ResponseParser().parse(response)

    assert parsed["api_name"] == "dc_twin_observe"
    assert parsed["kwargs"] == {"log_limit": 20, "include_config": True}
    assert agent.requests[0]["tool_choice"] == "required"
    assert {tool["function"]["name"] for tool in agent.requests[0]["tools"]} == {
        "dc_twin_action_space",
        "dc_twin_observe",
        "dc_twin_action",
        "submit",
    }
    action_tool = next(
        tool
        for tool in agent.requests[0]["tools"]
        if tool["function"]["name"] == "dc_twin_action"
    )
    assert set(
        action_tool["function"]["parameters"]["properties"]["action_type"]["enum"]
    ) == set(AGENT_ACTIONS)
    assert agent.last_normalization_status["normalized_ok"] is True


@pytest.mark.asyncio
async def test_data_center_twin_raw_tool_calling_retains_native_tool_history():
    agent = fake_raw_tool_agent_class()(
        [
            FakeLLMToolResponse("dc_twin_observe", {"log_limit": 20}),
            FakeLLMToolResponse("submit", {"payload": ["cooling-unit-1"]}),
        ]
    )
    agent.init_context(**raw_baseline_kwargs(task_type="localization"))

    await agent.get_action("initial observation")
    await agent.get_action("observation result")

    second_messages = agent.requests[1]["messages"]
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_messages[2]["content"] is None
    assert second_messages[2]["tool_calls"][0]["id"] == "call_1"
    assert second_messages[3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "observation result",
    }


@pytest.mark.asyncio
async def test_data_center_twin_raw_tool_calling_rejects_missing_tool_call_id():
    agent = fake_raw_tool_agent_class()(
        [FakeLLMToolResponse("dc_twin_observe", {"log_limit": 20})]
    )
    agent.responses[0].to_dict = lambda: {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "dc_twin_observe",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    agent.init_context(**raw_baseline_kwargs())

    response = await agent.get_action("initial observation")

    assert response == (
        "provider tool-call normalization failed: provider tool call missing id"
    )
    assert agent.last_normalization_status["normalized_ok"] is False


def test_data_center_twin_raw_tool_calling_prompt_requires_native_tools():
    from clients.data_center_twin_baselines.prompts import build_raw_tool_calling_prompt

    prompt = build_raw_tool_calling_prompt(**raw_baseline_kwargs())

    assert "provider-native structured tool call" in prompt
    assert "Do not write an API call" in prompt
    assert "Return exactly one markdown code block" not in prompt


@pytest.mark.asyncio
async def test_data_center_twin_raw_tool_calling_exposes_safe_progressive_drilldown():
    from aiopslab.orchestrator.parser import ResponseParser

    arguments = {
        "log_limit": 0,
        "include_config": False,
        "lookback_seconds": 30,
        "channels": ["metric"],
        "detail": "raw",
        "metric_names": ["rack.inlet_temperature"],
        "entity_ids": ["rack-r1-row1-1"],
    }
    agent = fake_raw_tool_agent_class()(
        [FakeLLMToolResponse("dc_twin_observe", arguments)]
    )
    agent.init_context(**raw_baseline_kwargs())

    parsed = ResponseParser().parse(await agent.get_action("Drill down."))

    assert parsed["api_name"] == "dc_twin_observe"
    assert parsed["kwargs"] == arguments
    observe_tool = next(
        tool
        for tool in agent.requests[0]["tools"]
        if tool["function"]["name"] == "dc_twin_observe"
    )
    assert {
        "lookback_seconds",
        "channels",
        "detail",
        "metric_names",
        "entity_ids",
    } <= set(observe_tool["function"]["parameters"]["properties"])


@pytest.mark.asyncio
async def test_data_center_twin_raw_tool_calling_retries_max_completion_tokens():
    from aiopslab.orchestrator.parser import ResponseParser
    from clients.data_center_twin_baselines.openai_compatible import (
        LLMProviderRequestError,
        OpenAICompatibleSettings,
    )
    from clients.data_center_twin_baselines.raw_tool_calling import RawToolCallingAgent

    class MaxCompletionRetryAgent(RawToolCallingAgent):
        def __init__(self):
            super().__init__(
                OpenAICompatibleSettings(
                    api_key="test-key",
                    base_url="https://llm.example/v1",
                    model="gpt-5.4",
                    max_tokens=123,
                )
            )
            self.requests = []

        def _request_json(self, _url, payload):
            self.requests.append(dict(payload))
            if len(self.requests) == 1:
                body = json.dumps(
                    {
                        "error": {
                            "message": (
                                "Unsupported parameter: 'max_tokens' is not supported with this model. "
                                "Use 'max_completion_tokens' instead."
                            ),
                            "type": "invalid_request_error",
                            "param": "max_tokens",
                            "code": "unsupported_parameter",
                        }
                    }
                )
                raise LLMProviderRequestError(
                    status_code=400,
                    reason="Bad Request",
                    body=body,
                    message=f"LLM provider request failed (400 Bad Request): {body}",
                )
            return FakeLLMToolResponse(
                "dc_twin_observe", {"log_limit": 20, "include_config": True}
            ).to_dict()

    agent = MaxCompletionRetryAgent()
    agent.init_context(**raw_baseline_kwargs())

    response = await agent.get_action("Please take the next action")
    parsed = ResponseParser().parse(response)

    assert parsed["api_name"] == "dc_twin_observe"
    assert agent.requests[0]["max_tokens"] == 123
    assert "max_completion_tokens" not in agent.requests[0]
    assert "max_tokens" not in agent.requests[1]
    assert agent.requests[1]["max_completion_tokens"] == 123
    assert agent.requests[1]["tool_choice"] == "required"


def test_deepseek_non_thinking_payload_uses_provider_native_toggle():
    from clients.data_center_twin_baselines.openai_compatible import (
        OpenAICompatibleAgent,
        OpenAICompatibleSettings,
    )

    agent = OpenAICompatibleAgent(
        OpenAICompatibleSettings(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            max_tokens=321,
            thinking_mode="disabled",
        )
    )

    payload = agent._chat_completion_payload([{"role": "user", "content": "x"}])

    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 321
    assert "reasoning_effort" not in payload
    assert "max_completion_tokens" not in payload


@pytest.mark.asyncio
async def test_data_center_twin_raw_tool_calling_retries_rate_limits_with_backoff(
    monkeypatch,
):
    from aiopslab.orchestrator.parser import ResponseParser
    from clients.data_center_twin_baselines import openai_compatible
    from clients.data_center_twin_baselines.openai_compatible import (
        LLMProviderRequestError,
        OpenAICompatibleSettings,
    )
    from clients.data_center_twin_baselines.raw_tool_calling import RawToolCallingAgent

    class RateLimitRetryAgent(RawToolCallingAgent):
        def __init__(self):
            super().__init__(
                OpenAICompatibleSettings(
                    api_key="test-key",
                    base_url="https://llm.example/v1",
                    model="gpt-5.6-luna",
                    reasoning_effort="none",
                    rate_limit_max_retries=2,
                    rate_limit_initial_delay_seconds=1.0,
                    rate_limit_max_delay_seconds=10.0,
                )
            )
            self.requests = []

        def _request_json(self, _url, payload):
            self.requests.append(dict(payload))
            if len(self.requests) <= 2:
                retry_hint = "250ms" if len(self.requests) == 1 else "1.5s"
                body = json.dumps(
                    {
                        "error": {
                            "message": (
                                "Rate limit reached on tokens per min. "
                                f"Please try again in {retry_hint}."
                            ),
                            "type": "tokens",
                            "param": None,
                            "code": "rate_limit_exceeded",
                        }
                    }
                )
                raise LLMProviderRequestError(
                    status_code=429,
                    reason="Too Many Requests",
                    body=body,
                    message=f"LLM provider request failed (429 Too Many Requests): {body}",
                )
            return FakeLLMToolResponse(
                "dc_twin_observe",
                {"log_limit": 20, "include_config": True},
            ).to_dict()

    sleeps = []
    monkeypatch.setattr(openai_compatible.random, "uniform", lambda _low, _high: 0.0)
    agent = RateLimitRetryAgent()
    monkeypatch.setattr(
        agent,
        "_wait_for_retry_delay",
        lambda delay: sleeps.append(delay) or False,
    )
    agent.init_context(**raw_baseline_kwargs())

    response = await agent.get_action("Please take the next action")
    parsed = ResponseParser().parse(response)

    assert parsed["api_name"] == "dc_twin_observe"
    assert len(agent.requests) == 3
    assert sleeps == [1.0, 2.0]
    assert all(request["reasoning_effort"] == "none" for request in agent.requests)


def test_data_center_twin_blackbox_fake_agent_observes_submits_and_evaluates(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import run_blackbox_problem

    fake_agent = write_fake_blackbox_agent(
        tmp_path,
        """
from pathlib import Path
import json
import subprocess
import sys

task_file = Path(sys.argv[1])
tool = task_file.with_name("dc_twin_tool.py")
subprocess.run([sys.executable, str(tool), "observe", "--log-limit", "20", "--include-config"], cwd=task_file.parent, check=True)
submission = {"incident_detected": True, "suspected_category": "unknown", "target": None, "evidence": ["RackTemperatureHigh"]}
subprocess.run([sys.executable, str(tool), "submit", "--json", json.dumps(submission)], cwd=task_file.parent, check=True)
""",
    )

    result = run_blackbox_problem(
        "data_center_twin-cooling_degradation-detection-1",
        blackbox_config(tmp_path, f"{sys.executable} {fake_agent} {{task_file}}"),
    )

    assert result["agent_type"] == "codex"
    assert result["blackbox"]["exit_code"] == 0
    assert result["action_sequence"][0]["api_name"] == "dc_twin_observe"
    assert result["action_sequence"][-1]["api_name"] == "submit"
    assert "Detection Accuracy" in result["evaluator_results"]
    assert result["token_usage"]["total_tokens"] is None
    assert result["token_usage_available"] is False
    assert result["termination_reason"] == "final_submission"
    assert result["agent_turns"] is None
    assert result["episode_start_timestamp"] <= result["episode_end_timestamp"]
    assert result["elapsed_wall_clock_seconds"] >= 0.0
    assert result["blackbox"]["sandbox_mode"] == "none"
    assert "not an OS sandbox" in result["blackbox"]["sandbox_isolation_guarantee"]


def test_data_center_twin_blackbox_canonical_boundary_writes_compact_initial_and_delta(
    tmp_path,
):
    from clients.data_center_twin_baselines.blackbox_harness import (
        run_blackbox_problem,
    )

    fake_agent = write_fake_blackbox_agent(
        tmp_path,
        """
from pathlib import Path
import json
import subprocess
import sys

task_file = Path(sys.argv[1])
workspace = task_file.parent
tool = workspace / "dc_twin_tool.py"
initial = json.loads((workspace / "INITIAL_OBSERVATION.json").read_text())
assert initial["schema_version"] == "agent.telemetry.compact.v1"
assert initial["mode"] == "snapshot"
assert "observations" not in initial
assert "query_watermark_sequence" not in initial
observed = subprocess.run(
    [sys.executable, str(tool), "observe", "--log-limit", "20", "--include-config"],
    cwd=workspace,
    check=True,
    capture_output=True,
    text=True,
)
response = json.loads(observed.stdout)
assert response["response"]["schema_version"] == "agent.telemetry.delta.v1"
submission = {
    "incident_detected": True,
    "suspected_category": "thermal",
    "target": None,
    "evidence": ["RackTemperatureHigh"],
}
subprocess.run(
    [sys.executable, str(tool), "submit", "--json", json.dumps(submission)],
    cwd=workspace,
    check=True,
)
""",
    )
    config = blackbox_config(
        tmp_path,
        f"{sys.executable} {fake_agent} {{task_file}}",
        keep_workspace=True,
    )
    config.telemetry_view = "canonical"
    config.observation_condition = "full-canonical"

    result = run_blackbox_problem(
        "data_center_twin-cooling_degradation-detection-1",
        config,
    )

    try:
        initial = result["initial_agent_visible_observation"]
        assert initial["schema_version"] == "agent.telemetry.compact.v1"
        assert initial["condition"] == "full-canonical"
        assert result["agent_visible_observations"][0]["schema_version"] == (
            "agent.telemetry.delta.v1"
        )
        assert (
            json.loads(result["action_sequence"][0]["env_response"])["schema_version"]
            == "agent.telemetry.delta.v1"
        )
        workspace = Path(result["blackbox"]["workspace"])
        written = json.loads(
            (workspace / "INITIAL_OBSERVATION.json").read_text(encoding="utf-8")
        )
        assert written == initial
        serialized = json.dumps(written, sort_keys=True)
        assert '"source_refs"' not in serialized
        assert '"provenance"' not in serialized
        rendering = dict(result["agent_observation_rendering"])
        audit = rendering.pop("audit")
        assert rendering == {
            "schema_versions": [
                "agent.telemetry.compact.v1",
                "agent.telemetry.delta.v1",
            ],
            "condition": "full-canonical",
            "token_budget": None,
            "rendered_observation_count": 2,
            "canonical_schema_unchanged": True,
        }
        assert len(audit) == 2
        assert all(record["condition"] == "full-canonical" for record in audit)
    finally:
        cleanup_retained_blackbox_workspace(result)


def test_blackbox_observe_bridge_accepts_progressive_drilldown_controls():
    state = make_bridge_state()

    api_name, args, kwargs, solution = state._tool_payload_to_api(
        "observe",
        {
            "log_limit": 7,
            "include_config": False,
            "channels": ["metric", "log"],
            "lookback_seconds": 45.0,
            "detail": "raw",
            "metric_names": ["rack.inlet_temperature"],
            "entity_ids": ["rack-r1-row1-01"],
        },
    )

    assert api_name == "dc_twin_observe"
    assert args == []
    assert solution is None
    assert kwargs == {
        "log_limit": 7,
        "include_config": False,
        "channels": ["metric", "log"],
        "lookback_seconds": 45.0,
        "detail": "raw",
        "metric_names": ["rack.inlet_temperature"],
        "entity_ids": ["rack-r1-row1-01"],
    }


def test_data_center_twin_blackbox_artifact_redacts_credentials(monkeypatch, tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import run_blackbox_problem

    environment_secret = "sk-blackbox-environment-secret"
    inline_secret = "sk-blackbox-inline-secret"
    monkeypatch.setenv("OPENAI_API_KEY", environment_secret)
    fake_agent = write_fake_blackbox_agent(
        tmp_path,
        """
import os
print(f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY')}")
""",
    )
    config = blackbox_config(
        tmp_path,
        f"{sys.executable} {fake_agent} {{task_file}} --api-key={inline_secret}",
    )
    config.allowed_env_vars = ("OPENAI_API_KEY",)

    result = run_blackbox_problem(
        "data_center_twin-cooling_degradation-detection-1",
        config,
    )
    serialized = json.dumps(result, sort_keys=True)

    assert environment_secret not in serialized
    assert inline_secret not in serialized
    assert "<redacted>" in serialized


def test_data_center_twin_blackbox_workspace_excludes_hidden_repo_files(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import run_blackbox_problem

    fake_agent = write_fake_blackbox_agent(
        tmp_path,
        """
from pathlib import Path
import json
import subprocess
import sys

task_file = Path(sys.argv[1])
tool = task_file.with_name("dc_twin_tool.py")
subprocess.run([sys.executable, str(tool), "read-file", "scenarios.json"], cwd=task_file.parent, check=False)
subprocess.run([sys.executable, str(tool), "observe", "--log-limit", "20", "--include-config"], cwd=task_file.parent, check=True)
submission = {"incident_detected": True, "suspected_category": "unknown", "target": None, "evidence": ["RackTemperatureHigh"]}
subprocess.run([sys.executable, str(tool), "submit", "--json", json.dumps(submission)], cwd=task_file.parent, check=True)
""",
    )

    result = run_blackbox_problem(
        "data_center_twin-cooling_degradation-detection-1",
        blackbox_config(
            tmp_path,
            f"{sys.executable} {fake_agent} {{task_file}}",
            keep_workspace=True,
        ),
    )
    try:
        workspace = Path(result["blackbox"]["workspace"])
        workspace_files = set(result["blackbox"]["workspace_initial_files"])

        assert workspace.exists()
        assert workspace.resolve() != Path.cwd().resolve()
        assert workspace_files == {
            "TASK.md",
            "INITIAL_OBSERVATION.json",
            "README_TOOL.md",
            "dc_twin_tool.py",
        }
        assert "scenarios.json" not in result["blackbox"]["workspace_final_files"]
        assert "registry.py" not in result["blackbox"]["workspace_final_files"]
        task_text = (workspace / "TASK.md").read_text(encoding="utf-8")
        assert_no_forbidden_agent_leakage({"task": task_text})
        assert (
            "Respond with exactly one API call in one markdown code block"
            not in task_text
        )
        assert "python3 dc_twin_tool.py submit --json" in task_text
        assert "The evaluator only records actions executed via" in task_text
        assert any(item.get("ok") is False for item in result["tool_call_log"])
    finally:
        cleanup_retained_blackbox_workspace(result)


def test_data_center_twin_blackbox_fake_agent_post_submit_action_is_rejected(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import run_blackbox_problem

    fake_agent = write_fake_blackbox_agent(
        tmp_path,
        """
from pathlib import Path
import json
import subprocess
import sys

task_file = Path(sys.argv[1])
tool = task_file.with_name("dc_twin_tool.py")
submission = {"incident_detected": True, "suspected_category": "unknown", "target": None, "evidence": []}
subprocess.run([sys.executable, str(tool), "submit", "--json", json.dumps(submission)], cwd=task_file.parent, check=False)
action = {"action_type": "observe"}
subprocess.run([sys.executable, str(tool), "action", "--json", json.dumps(action)], cwd=task_file.parent, check=False)
""",
    )

    result = run_blackbox_problem(
        "data_center_twin-cooling_degradation-detection-1",
        blackbox_config(tmp_path, f"{sys.executable} {fake_agent} {{task_file}}"),
    )

    assert result["final_state"] == "submitted"
    assert result["tool_call_log"][1]["ok"] is False
    assert result["tool_call_log"][1]["error"] == "episode already submitted"
    assert result["action_sequence"][1]["api_name"] == "invalid_tool_command"
    assert result["action_sequence"][1]["env_response"] == "episode already submitted"
    assert not any(
        record.get("api_name") == "dc_twin_action"
        for record in result["action_sequence"][1:]
    )


def test_data_center_twin_blackbox_timeout_is_recorded(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import run_blackbox_problem

    fake_agent = write_fake_blackbox_agent(
        tmp_path,
        """
import time
time.sleep(5)
""",
    )

    result = run_blackbox_problem(
        "data_center_twin-cooling_degradation-detection-1",
        blackbox_config(
            tmp_path,
            f"{sys.executable} {fake_agent} {{task_file}}",
            timeout_seconds=0.2,
        ),
    )

    assert result["status"] == "timeout"
    assert result["termination_reason"] == "wall_clock_timeout"
    assert result["episode_start_timestamp"] <= result["episode_end_timestamp"]
    assert result["elapsed_wall_clock_seconds"] >= 0.0
    assert result["blackbox"]["timed_out"] is True
    assert result["blackbox"]["exit_code"] in (None, -9)


def test_data_center_twin_blackbox_invalid_tool_command_is_recorded(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import run_blackbox_problem

    fake_agent = write_fake_blackbox_agent(
        tmp_path,
        """
from pathlib import Path
import json
import subprocess
import sys

task_file = Path(sys.argv[1])
tool = task_file.with_name("dc_twin_tool.py")
subprocess.run([sys.executable, str(tool), "unsupported-command"], cwd=task_file.parent, check=False)
submission = {"incident_detected": False, "evidence": []}
subprocess.run([sys.executable, str(tool), "submit", "--json", json.dumps(submission)], cwd=task_file.parent, check=True)
""",
    )

    result = run_blackbox_problem(
        "data_center_twin-cooling_degradation-detection-1",
        blackbox_config(tmp_path, f"{sys.executable} {fake_agent} {{task_file}}"),
    )

    assert any(
        record.get("api_name") == "invalid_tool_command"
        for record in result["action_sequence"]
    )
    assert any(error["phase"] == "tool" for error in result["errors"])
    assert "Detection Accuracy" in result["evaluator_results"]


class FakeBridgeProblem:
    def __init__(self, submission_status=None):
        from aiopslab.utils.status import SubmissionStatus

        self.submission_status = submission_status or SubmissionStatus.VALID_SUBMISSION
        self.mutations = []

    def perform_action(self, api_name, *args, **kwargs):
        if api_name == "submit":
            return self.submission_status
        if api_name == "dc_twin_action":
            self.mutations.append((args, kwargs))
            return {"action_result": {"status": "ok"}, "observation": {"summary": {}}}
        if api_name in {"dc_twin_observe", "dc_twin_action_space"}:
            return {"summary": {}}
        raise ValueError(f"unsupported action: {api_name}")


def make_bridge_state(problem=None):
    from aiopslab.session import SessionItem
    from clients.data_center_twin_baselines.blackbox_harness import BridgeState

    return BridgeState(
        problem=problem or FakeBridgeProblem(),
        history=[SessionItem(role="system", content="task")],
        action_sequence=[],
        errors=[],
        max_steps=20,
        action_space_payload={"agent_actions": ["observe", "set_cooling"]},
    )


def test_data_center_twin_blackbox_rejects_tool_calls_after_submission():
    state = make_bridge_state()

    first_status, first_response = state.handle_tool_call(
        {
            "command": "submit",
            "arguments": {"payload": {"incident_detected": True, "evidence": []}},
        }
    )
    second_status, second_response = state.handle_tool_call(
        {
            "command": "action",
            "arguments": {
                "payload": {
                    "action_type": "set_cooling",
                    "parameters": {"target": "visible-target"},
                }
            },
        }
    )

    assert first_status == 200
    assert first_response["ok"] is True
    assert second_status == 400
    assert second_response == {"ok": False, "error": "episode already submitted"}
    assert state.problem.mutations == []
    assert state.final_state == "submitted"
    assert state.action_sequence[-1]["api_name"] == "invalid_tool_command"
    assert not any(
        record.get("api_name") == "dc_twin_action"
        and "ok" in str(record.get("env_response", "")).lower()
        for record in state.action_sequence[1:]
    )


def test_data_center_twin_blackbox_rejects_payload_submission_for_mitigation():
    problem = FakeBridgeProblem()
    problem.scenario = types.SimpleNamespace(task_type="mitigation")
    state = make_bridge_state(problem)

    invalid_status, invalid_response = state.handle_tool_call(
        {"command": "submit", "arguments": {"payload": {}}}
    )
    valid_status, valid_response = state.handle_tool_call(
        {"command": "submit_empty", "arguments": {}}
    )

    assert invalid_status == 400
    assert invalid_response == {
        "ok": False,
        "error": "mitigation tasks require submit_empty without a payload",
    }
    assert state.errors == [
        {
            "phase": "tool",
            "type": "InvalidToolCommand",
            "message": "mitigation tasks require submit_empty without a payload",
        }
    ]
    assert state.action_sequence[0]["api_name"] == "invalid_tool_command"
    assert valid_status == 200
    assert valid_response["ok"] is True
    assert state.final_state == "submitted"


def test_data_center_twin_blackbox_rejects_empty_submission_for_diagnosis():
    problem = FakeBridgeProblem()
    problem.scenario = types.SimpleNamespace(task_type="analysis")
    state = make_bridge_state(problem)

    status, response = state.handle_tool_call(
        {"command": "submit_empty", "arguments": {}}
    )

    assert status == 400
    assert response == {
        "ok": False,
        "error": "analysis tasks require submit with a payload",
    }
    assert state.errors[0]["phase"] == "tool"
    assert state.final_state == "not_started"


def test_data_center_twin_blackbox_rejects_tool_calls_after_absolute_deadline():
    state = make_bridge_state()
    state.deadline_monotonic = time.monotonic() - 0.001

    status, response = state.handle_tool_call(
        {
            "command": "action",
            "arguments": {
                "payload": {
                    "action_type": "set_cooling",
                    "parameters": {"target": "visible-target"},
                }
            },
        }
    )

    assert status == 408
    assert response == {
        "ok": False,
        "error": "benchmark episode wall-clock deadline exceeded",
    }
    assert state.deadline_exceeded is True
    assert state.late_tool_call_count == 1
    assert state.final_state == "timeout"
    assert state.problem.mutations == []
    assert state.action_sequence == []


def test_data_center_twin_blackbox_forbidden_filter_allows_fault_diagnosis_rejects_hidden_paths():
    state = make_bridge_state()

    diagnosis_status, diagnosis_response = state.handle_tool_call(
        {
            "command": "submit",
            "arguments": {
                "payload": {
                    "root_cause_category": "hardware fault",
                    "target": "visible-target",
                    "evidence": ["observed-alert"],
                }
            },
        }
    )
    assert diagnosis_status == 200
    assert diagnosis_response["ok"] is True

    hidden_file_state = make_bridge_state()
    hidden_status, hidden_response = hidden_file_state.handle_tool_call(
        {
            "command": "invalid",
            "arguments": {"raw_args": ["read-file", "scenarios.json"]},
        }
    )
    assert hidden_status == 400
    assert hidden_response["ok"] is False
    assert "hidden or forbidden" in hidden_response["error"]

    debug_state = make_bridge_state()
    debug_status, debug_response = debug_state.handle_tool_call(
        {
            "command": "invalid",
            "arguments": {"raw_args": ["curl", "http://localhost/evaluator/debug"]},
        }
    )
    assert debug_status == 400
    assert debug_response["ok"] is False
    assert "hidden or forbidden" in debug_response["error"]


def test_data_center_twin_blackbox_task_template_has_no_concrete_scenario_examples():
    template = (
        Path(__file__).resolve().parents[2]
        / "clients"
        / "data_center_twin_baselines"
        / "blackbox_task_template.md"
    ).read_text(encoding="utf-8")

    forbidden_examples = {
        "cooling-unit-1",
        "thermal",
        "RackTemperatureHigh",
        "cooling_degradation",
        "power_overload",
        "server_failure",
        "network_partition",
        "storage_io_saturation",
        "control_plane_degradation",
        "application_error",
        "rack_hotspot",
        "rack-row",
    }

    leaks = sorted(term for term in forbidden_examples if term in template)
    assert leaks == []
    assert "markdown API-call" in template
    assert '"diagnosis"' in template
    assert '"suspected_category"' not in template


def test_detection_and_rca_hide_the_internal_fault_taxonomy():
    from aiopslab.orchestrator.problems.data_center_twin.scenarios import (
        BENCHMARK_FAULT_TAXONOMY,
        SUPPORTED_BENCHMARK_FAULTS,
        fault_diagnosis_contract_lines,
    )
    from clients.data_center_twin_baselines.blackbox_harness import (
        blackbox_task_instructions,
    )

    assert len(BENCHMARK_FAULT_TAXONOMY) == 18
    assert BENCHMARK_FAULT_TAXONOMY == tuple(sorted(SUPPORTED_BENCHMARK_FAULTS))

    contract = list(fault_diagnosis_contract_lines())
    detection = blackbox_task_instructions("detection")
    rca = blackbox_task_instructions("analysis")
    assert all(line in detection for line in contract)
    assert all(line in rca for line in contract)

    detection_text = "\n".join(detection)
    rca_text = "\n".join(rca)
    for fault_type in BENCHMARK_FAULT_TAXONOMY:
        assert fault_type not in detection_text
        assert fault_type not in rca_text
        assert fault_type.replace("_", " ") not in detection_text
        assert fault_type.replace("_", " ") not in rca_text
    assert "underlying fault mechanism" in detection_text
    assert "underlying fault mechanism" in rca_text
    assert "in your own words" in detection_text
    assert "in your own words" in rca_text
    assert "predefined" not in detection_text
    assert "predefined" not in rca_text
    assert "root_cause_category" not in rca_text
    assert '"diagnosis"' in detection_text
    assert '"root_cause"' in rca_text
    assert '"suspected_category"' not in detection_text
    assert '"suspected_category"' not in rca_text

    # The shared contract is episode-independent and contains neither the
    # internal class catalog nor scenario-specific answer material.
    contract_text = "\n".join(contract).lower()
    assert not any(
        fault_type in contract_text for fault_type in BENCHMARK_FAULT_TAXONOMY
    )
    for forbidden in (
        "cooling-unit-1",
        "rack-r1-row1-01",
        "seed",
        "expected evidence",
        "correct category",
        "ground truth",
    ):
        assert forbidden not in contract_text


def test_data_center_twin_blackbox_render_task_uses_tool_instructions():
    from clients.data_center_twin_baselines.blackbox_harness import render_task_file

    task_text = render_task_file(
        agent_task_id="dcopslab-test",
        task_description="Detection task",
        instructions="Respond with exactly one API call in one markdown code block.\n```\nsubmit({})\n```",
        actions={"submit": "Submit the task."},
        action_space_payload={
            "agent_actions": ["observe"],
            "read_actions": {"observe": {}},
        },
        initial_observation={"summary": {"sla_status": "violated"}},
        task_type="detection",
    )

    assert (
        "Respond with exactly one API call in one markdown code block" not in task_text
    )
    assert "submit({})" not in task_text
    assert "INITIAL_OBSERVATION.json" in task_text
    assert '"sla_status": "violated"' not in task_text
    assert "Run at least one observation" not in task_text
    assert (
        "python3 dc_twin_tool.py observe --log-limit 20 --include-config" in task_text
    )
    assert "python3 dc_twin_tool.py submit --json" in task_text

    mitigation_text = render_task_file(
        agent_task_id="dcopslab-test",
        task_description="Mitigation task",
        instructions='```\ndc_twin_action("set_cooling")\n```',
        actions={"submit": "Submit the task."},
        action_space_payload={"agent_actions": ["observe", "set_cooling"]},
        initial_observation={
            "configuration": {"cooling_units": [{"status": "degraded"}]}
        },
        task_type="mitigation",
    )

    assert "offending component" in mitigation_text
    assert "python3 dc_twin_tool.py action --json" in mitigation_text


def test_data_center_twin_blackbox_environment_does_not_leak_secrets_by_default(
    tmp_path,
):
    from clients.data_center_twin_baselines.blackbox_harness import blackbox_environment

    base_env = {
        "PATH": "/usr/bin",
        "GITHUB_TOKEN": "gh-secret",
        "OPENAI_API_KEY": "openai-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "ALLOW_ME": "allowed",
        "PREFIX_SAFE": "safe",
    }

    env = blackbox_environment(
        base_env,
        workspace_dir=tmp_path,
        bridge_url="http://127.0.0.1:1/tool",
        token="token",
    )
    assert env == {
        "PATH": "/usr/bin",
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "PWD": str(tmp_path),
        "PYTHONNOUSERSITE": "1",
        "DC_TWIN_TOOL_URL": "http://127.0.0.1:1/tool",
        "DC_TWIN_TOOL_TOKEN": "token",
        "DC_TWIN_WORKSPACE": str(tmp_path),
    }

    allowed = blackbox_environment(
        base_env,
        workspace_dir=tmp_path,
        bridge_url="http://127.0.0.1:1/tool",
        token="token",
        allowed_env_vars=("OPENAI_API_KEY",),
        allowed_env_prefixes=("PREFIX_",),
    )
    assert allowed["OPENAI_API_KEY"] == "openai-secret"
    assert allowed["PREFIX_SAFE"] == "safe"
    assert "GITHUB_TOKEN" not in allowed
    assert "AWS_SECRET_ACCESS_KEY" not in allowed


def test_data_center_twin_blackbox_token_usage_unknown_is_not_zero(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import (
        BlackboxEpisodeConfig,
        blackbox_token_usage,
    )
    from clients.data_center_twin_baselines.metrics import (
        aggregate_process_metrics,
        compute_episode_process_metrics,
    )
    from scripts import evaluate_data_center_twin as runner

    config = BlackboxEpisodeConfig(
        agent_name="codex",
        command_template="external-agent {task_file}",
        output_dir=tmp_path,
    )
    usage = blackbox_token_usage(config, {})
    metrics = compute_episode_process_metrics([], [], token_usage=usage)
    aggregate = runner.aggregate_results(
        [
            {
                "success": False,
                "score": None,
                "runtime_seconds": 0.1,
                "token_usage": usage,
                "token_usage_available": False,
                "process_metrics": metrics,
            }
        ]
    )
    process_aggregate = aggregate_process_metrics([{"process_metrics": metrics}])

    assert usage["prompt_tokens"] is None
    assert usage["completion_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["token_usage_available"] is False
    assert metrics["token_usage"]["total_tokens"] is None
    assert aggregate["total_token_usage"]["total_tokens"] is None
    assert process_aggregate["total_token_usage"]["total_tokens"] is None


def test_data_center_twin_blackbox_token_log_parser_reports_actual_usage(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import (
        BlackboxEpisodeConfig,
        blackbox_token_usage,
    )

    token_log = tmp_path / "usage.jsonl"
    token_log.write_text(
        '{"usage": {"input_tokens": 3, "output_tokens": 4}}\n', encoding="utf-8"
    )
    config = BlackboxEpisodeConfig(
        agent_name="codex",
        command_template="external-agent {task_file}",
        output_dir=tmp_path,
        token_log=token_log,
        token_parser="auto",
    )

    usage = blackbox_token_usage(config, {})

    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 7
    assert usage["token_usage_available"] is True


def test_data_center_twin_blackbox_token_log_sums_multiple_model_calls(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import (
        BlackboxEpisodeConfig,
        blackbox_token_usage,
    )

    token_log = tmp_path / "usage.jsonl"
    token_log.write_text(
        "\n".join(
            [
                '{"usage": {"input_tokens": 3, "output_tokens": 4}}',
                '{"usage": {"prompt_tokens": 5, "completion_tokens": 2}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = BlackboxEpisodeConfig(
        agent_name="codex",
        command_template="external-agent {task_file}",
        output_dir=tmp_path,
        token_log=token_log,
        token_parser="auto",
    )

    usage = blackbox_token_usage(config, {})

    assert usage["prompt_tokens"] == 8
    assert usage["completion_tokens"] == 6
    assert usage["total_tokens"] == 14
    assert usage["model_call_count"] == 2
    assert len(usage["model_call_token_usage"]) == 2
    assert usage["token_count_source"] == "provider_native"


def test_data_center_twin_blackbox_plain_text_log_sums_multiple_model_calls(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import (
        BlackboxEpisodeConfig,
        blackbox_token_usage,
    )

    token_log = tmp_path / "usage.log"
    token_log.write_text(
        "\n".join(
            [
                "prompt_tokens: 3 completion_tokens: 4 total_tokens: 7",
                "prompt_tokens: 5 completion_tokens: 2 total_tokens: 7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = BlackboxEpisodeConfig(
        agent_name="codex",
        command_template="external-agent {task_file}",
        output_dir=tmp_path,
        token_log=token_log,
        token_parser="auto",
    )

    usage = blackbox_token_usage(config, {})

    assert usage["prompt_tokens"] == 8
    assert usage["completion_tokens"] == 6
    assert usage["total_tokens"] == 14
    assert usage["model_call_count"] == 2
    assert len(usage["model_call_token_usage"]) == 2
    assert usage["token_count_source"] == "provider_native"


def test_data_center_twin_blackbox_json_preserves_raw_provider_usage_details(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import (
        BlackboxEpisodeConfig,
        blackbox_token_usage,
    )

    token_log = tmp_path / "usage.jsonl"
    token_log.write_text(
        json.dumps(
            {
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 2},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = BlackboxEpisodeConfig(
        agent_name="codex",
        command_template="external-agent {task_file}",
        output_dir=tmp_path,
        token_log=token_log,
        token_parser="auto",
    )

    usage = blackbox_token_usage(config, {})

    record = usage["model_call_token_usage"][0]
    assert record["provider_usage"]["prompt_tokens_details"] == {"cached_tokens": 2}


def test_data_center_twin_blackbox_wrapped_usage_preserves_per_call_records(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import (
        BlackboxEpisodeConfig,
        blackbox_token_usage,
    )

    call_records = [
        {
            "call_index": 1,
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
            "token_count_source": "provider_native",
            "estimator": None,
            "provider_usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 1},
            },
            "warnings": [],
        },
        {
            "call_index": 2,
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
            "token_count_source": "estimated",
            "estimator": "utf8_bytes_div4_ceil:canonical_chat_json_v1",
            "provider_usage": None,
            "warnings": [],
        },
    ]
    token_log = tmp_path / "usage.json"
    token_log.write_text(
        json.dumps(
            {
                "prompt_tokens": 8,
                "completion_tokens": 6,
                "total_tokens": 14,
                "model_call_count": 2,
                "model_call_token_usage": call_records,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = BlackboxEpisodeConfig(
        agent_name="codex",
        command_template="external-agent {task_file}",
        output_dir=tmp_path,
        token_log=token_log,
        token_parser="auto",
    )

    usage = blackbox_token_usage(config, {})

    assert usage["prompt_tokens"] == 8
    assert usage["completion_tokens"] == 6
    assert usage["total_tokens"] == 14
    assert usage["model_call_count"] == 2
    assert usage["model_call_token_usage"] == call_records
    assert usage["token_count_source"] == "mixed"


def test_data_center_twin_blackbox_wrapped_usage_retains_unavailable_call(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import (
        BlackboxEpisodeConfig,
        blackbox_token_usage,
    )

    call_records = [
        {
            "call_index": 1,
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
            "token_count_source": "provider_native",
            "provider_usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
            "warnings": [],
        },
        {
            "call_index": 2,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "token_count_source": "unavailable",
            "provider_usage": None,
            "warnings": ["provider call usage unavailable"],
        },
    ]
    token_log = tmp_path / "usage.json"
    token_log.write_text(
        json.dumps({"model_call_token_usage": call_records}) + "\n",
        encoding="utf-8",
    )
    config = BlackboxEpisodeConfig(
        agent_name="codex",
        command_template="external-agent {task_file}",
        output_dir=tmp_path,
        token_log=token_log,
        token_parser="auto",
    )

    usage = blackbox_token_usage(config, {})

    assert usage["prompt_tokens"] is None
    assert usage["completion_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["token_usage_available"] is False
    assert usage["model_call_count"] == 2
    assert len(usage["model_call_token_usage"]) == 2
    assert usage["model_call_token_usage"][1]["token_count_source"] == "unavailable"
    assert usage["token_count_sources"] == {
        "provider_native": 1,
        "estimated": 0,
        "unavailable": 1,
    }


def test_data_center_twin_blackbox_incomplete_usage_is_unavailable_not_zero(tmp_path):
    from clients.data_center_twin_baselines.blackbox_harness import (
        BlackboxEpisodeConfig,
        blackbox_token_usage,
    )

    token_log = tmp_path / "usage.jsonl"
    token_log.write_text(
        '{"usage": {"total_tokens": 9}}\n',
        encoding="utf-8",
    )
    config = BlackboxEpisodeConfig(
        agent_name="codex",
        command_template="external-agent {task_file}",
        output_dir=tmp_path,
        token_log=token_log,
        token_parser="auto",
    )

    usage = blackbox_token_usage(config, {})

    assert usage["prompt_tokens"] is None
    assert usage["completion_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["token_usage_available"] is False
    assert usage["token_count_source"] == "unavailable"
    assert usage["warnings"]


def test_data_center_twin_process_metrics_repeated_observe_counts_redundancy():
    from clients.data_center_twin_baselines.metrics import (
        compute_episode_process_metrics,
    )

    env_response = json.dumps({"summary": {"thermal_critical": 1}}, sort_keys=True)
    metrics = compute_episode_process_metrics(
        [
            {
                "step": 1,
                "api_name": "dc_twin_observe",
                "args": [],
                "kwargs": {"log_limit": 20, "include_config": True},
                "env_response": env_response,
            },
            {
                "step": 2,
                "api_name": "dc_twin_observe",
                "args": [],
                "kwargs": {"log_limit": 20, "include_config": True},
                "env_response": env_response,
            },
        ],
        [],
        runtime_seconds=1.5,
    )

    assert metrics["tool_call_count"] == 2
    assert metrics["observe_count"] == 2
    assert metrics["redundant_action_count"] == 1
    assert metrics["redundant_action_rate"] == 0.5


def test_data_center_twin_process_metrics_parse_errors_are_invalid_actions():
    from clients.data_center_twin_baselines.metrics import (
        compute_episode_process_metrics,
    )

    metrics = compute_episode_process_metrics(
        [
            {
                "step": 1,
                "api_name": None,
                "args": [],
                "kwargs": {},
                "env_response": "Could not parse response.",
                "parse_error": {"phase": "parser", "message": "missing fenced code"},
            }
        ],
        [
            {
                "phase": "parser",
                "type": "ResponseParsingError",
                "message": "missing fenced code",
            }
        ],
    )

    assert metrics["parse_error_count"] == 1
    assert metrics["invalid_action_count"] == 1
    assert metrics["valid_action_count"] == 0
    assert metrics["action_type_histogram"] == {"parse_error": 1}


def test_data_center_twin_process_metrics_direct_submit_sets_zero_tool_diagnosis():
    from clients.data_center_twin_baselines.metrics import (
        compute_episode_process_metrics,
    )

    metrics = compute_episode_process_metrics(
        [
            {
                "step": 1,
                "api_name": "submit",
                "args": [{"incident_detected": True, "evidence": []}],
                "kwargs": {},
                "env_response": "VALID_SUBMISSION",
            }
        ],
        [],
    )

    assert metrics["zero_tool_diagnosis"] is True
    assert metrics["steps_to_submit"] == 1
    assert metrics["tool_call_count"] == 0


def test_data_center_twin_process_metrics_missing_token_usage_is_unknown():
    from clients.data_center_twin_baselines.metrics import (
        compute_episode_process_metrics,
    )

    metrics = compute_episode_process_metrics(
        [], [], runtime_seconds=0.25, token_usage=None
    )

    assert metrics["token_usage"]["prompt_tokens"] is None
    assert metrics["token_usage"]["completion_tokens"] is None
    assert metrics["token_usage"]["total_tokens"] is None
    assert metrics["token_usage"]["token_usage_available"] is False
    assert metrics["runtime_seconds"] == 0.25


class FakeDataCenterTwin:
    def __init__(self):
        self.namespace = "dc-twin"
        self.helm_configs = {}
        self.calls = []
        self.workload = {}
        self.network_fault_active = False
        self.network_fault_target = "rack-r1-row1-01"
        self.network_mitigated = False
        self.active_fault = None
        self.summary = self._normal_summary(sim_time_seconds=0)

    def get_app_summary(self):
        return "Data Center Twin fake summary"

    def request_in_pod(self, method, path, payload=None):
        self.calls.append(("legacy", method, path, payload))
        if method == "POST" and path == "/faults":
            self.active_fault = {
                "fault_type": payload["fault_type"],
                "target": payload["target"],
                "severity": payload["severity"],
                "duration_seconds": payload["duration_seconds"],
                "started_at_sim_time_seconds": self.summary["sim_time_seconds"],
                "status": "active",
            }
            if payload["fault_type"] == "network_partition":
                self.network_fault_active = True
                self.network_mitigated = False
                self.network_fault_target = payload["target"]
                self._set_network_degraded()
            else:
                self.summary["active_faults"] = [dict(self.active_fault)]
                self._set_fault_specific_telemetry(payload)
            return json.dumps(
                {"started_at_sim_time_seconds": self.summary["sim_time_seconds"]}
            )
        return json.dumps({"status": "ok"})

    def agent_action_space(self):
        self.calls.append(("agent_action_space",))
        return self._action_space_response()

    def _action_space_response(self):
        return {
            "agent_actions": list(AGENT_ACTIONS),
            "read_actions": {"observe": {}},
            "time_actions": {"noop": {}, "step": {}},
            "control_actions": {action: {} for action in WRITE_ACTIONS},
            "benchmark_action_coverage": {
                "scored_actions": [
                    "calibrate_sensor",
                    "set_cooling",
                    "migrate_workload",
                    "throttle_workload",
                    "update_autoscaler_policy",
                    "repair_monitoring_pipeline",
                    "update_placement_policy",
                    "update_load_balancer_config",
                    "set_server_maintenance",
                ],
                "supported_non_scored_actions": {
                    "clear_server_maintenance": "supported, not scored",
                },
                "all_write_actions_have_benchmark_status": True,
            },
            "domain_contract": {
                "incident_domains": {
                    "cooling_degradation": {
                        "agent_response_actions": [
                            "set_cooling",
                            "migrate_workload",
                            "observe",
                        ],
                    },
                    "network_partition": {
                        "agent_response_actions": [
                            "migrate_workload",
                            "throttle_workload",
                            "observe",
                        ],
                    },
                },
                "action_domains": {action: {"domains": []} for action in AGENT_ACTIONS},
                "coverage_invariant": {
                    "all_response_actions_are_agent_actions": True,
                    "all_write_actions_have_benchmark_status": True,
                },
            },
        }

    def agent_reset(
        self,
        seed=None,
        config_override=None,
        workload=None,
        stabilization_ticks=0,
        log_limit=20,
        include_config=True,
    ):
        self.calls.append(
            (
                "agent_reset",
                seed,
                config_override,
                workload,
                stabilization_ticks,
                log_limit,
                include_config,
            )
        )
        self.workload = workload or {}
        self.network_fault_active = False
        self.network_mitigated = False
        self.active_fault = None
        target_rack = self.workload.get("target_rack_id", "rack-r1-row1-01")
        self.summary = self._normal_summary(
            sim_time_seconds=stabilization_ticks,
            allocated_server_ids=[self._server_id_for_rack(target_rack)],
        )
        return {
            "observation": {
                "summary": dict(self.summary),
                "available_actions": self._action_space_response(),
            },
            "available_actions": self._action_space_response(),
        }

    def agent_observation(self, log_limit=20, include_config=True):
        self.calls.append(("agent_observation", log_limit, include_config))
        return {
            "summary": dict(self.summary),
            "available_actions": self._action_space_response(),
        }

    def agent_telemetry(self, **kwargs):
        from scripts.evaluate_data_center_twin import InProcessDataCenterTwinApp

        if not hasattr(self, "_canonical_app"):
            self._canonical_app = InProcessDataCenterTwinApp()
        simulator = self._canonical_app.simulator
        current = simulator.state_summary()["sim_time_seconds"]
        target = self.summary["sim_time_seconds"]
        if target > current:
            simulator.step(int(target - current))
        self.calls.append(("agent_telemetry",))
        return self._canonical_app.agent_telemetry(**kwargs)

    def agent_action(self, action_type, parameters=None, **kwargs):
        self.calls.append(("agent_action", action_type, parameters, kwargs))
        if action_type == "bad_action":
            return {
                "http_status": 400,
                "error": {
                    "detail": {"error": "unsupported agent action_type: bad_action"}
                },
            }
        before_summary = self._summary_copy()
        if action_type == "set_cooling":
            self.summary.update(
                {
                    "thermal_critical": 0,
                    "sla_status": "normal",
                }
            )
        elif action_type == "migrate_workload":
            self.network_mitigated = True
            target_rack = (
                (parameters or {}).get("target_rack_id")
                or kwargs.get("target_rack_id")
                or "rack-r1-row1-02"
            )
            self._set_network_recovered(target_rack)
        elif action_type == "throttle_workload":
            request_rate = (parameters or {}).get(
                "request_rate_per_second",
                kwargs.get("request_rate_per_second", 0),
            )
            self.summary.update(
                {
                    "sla_status": "normal",
                    "workload_average_latency_ms": 8.0,
                    "workload_current_demand_per_second": request_rate,
                    "workload_queue_length": 0,
                }
            )
            if (
                self.active_fault
                and self.active_fault["fault_type"] == "storage_io_saturation"
            ):
                self.summary.update(
                    {
                        "workload_storage_latency_penalty_ms": 0.0,
                        "workload_storage_utilization_ratio": 0.2,
                    }
                )
            if (
                self.active_fault
                and self.active_fault["fault_type"] == "application_error"
            ):
                self.summary.update(
                    {
                        "workload_application_error_rate_percent": 0.0,
                        "workload_dropped_requests_per_second": 0.0,
                    }
                )
        self.summary["sim_time_seconds"] += kwargs.get("advance_ticks", 0)
        self._expire_active_fault_if_needed()
        if self.network_fault_active and not self.network_mitigated:
            self._set_network_degraded()
        after_summary = self._summary_copy()
        return {
            "action_type": action_type,
            "sim_time_seconds_before": before_summary["sim_time_seconds"],
            "sim_time_seconds_after": after_summary["sim_time_seconds"],
            "active_faults_before": before_summary["active_faults"],
            "active_faults_after": after_summary["active_faults"],
            "observation": {
                "summary": after_summary,
                "available_actions": self._action_space_response(),
            },
            "action_result": {"status": action_type},
        }

    def evaluator_state(self, log_limit=50, include_config=True):
        self.calls.append(("evaluator_state", log_limit, include_config))
        return {"summary": self._summary_copy()}

    def _normal_summary(self, sim_time_seconds=0, allocated_server_ids=None):
        return {
            "sim_time_seconds": sim_time_seconds,
            "sla_status": "normal",
            "thermal_critical": 0,
            "workload_queue_length": 0,
            "workload_network_congestion_ratio": 0.2,
            "workload_average_latency_ms": 8.0,
            "workload_p95_latency_ms": 12.0,
            "workload_network_packet_loss_percent": 0.0,
            "workload_network_retransmit_rate": 0.0,
            "workload_network_error_rate": 0.0,
            "workload_network_affected_rack_id": None,
            "network_packet_loss_racks": 0,
            "max_network_packet_loss_percent": 0.0,
            "network_retransmit_rate": 0.0,
            "network_error_rate": 0.0,
            "affected_rack_id": None,
            "autoscaler_enabled": True,
            "autoscaler_min_capacity": 1,
            "autoscaler_max_capacity": 80,
            "autoscaler_target_utilization_percent": 65.0,
            "autoscaler_cooldown_seconds": 60,
            "autoscaler_current_capacity_units": 80,
            "autoscaler_effective_server_limit": 80,
            "autoscaler_last_scale_action_time": None,
            "autoscaler_status": "normal",
            "metrics_last_updated_sim_time_seconds": sim_time_seconds,
            "logs_last_updated_sim_time_seconds": sim_time_seconds,
            "telemetry_lag_seconds": 0,
            "metrics_missing_ratio": 0.0,
            "logs_missing_ratio": 0.0,
            "telemetry_pipeline_status": "normal",
            "placement_policy_status": "normal",
            "placement_policy_target_rack_id": None,
            "placement_policy_violating_racks": 0,
            "workload_placement_imbalance_ratio": 0.2,
            "load_balancer_enabled": True,
            "load_balancer_backend_server_ids": ["server-r1-row1-rack01-01"],
            "load_balancer_backend_weights": {"server-r1-row1-rack01-01": 1.0},
            "load_balancer_unhealthy_backend_ids": [],
            "load_balancer_routing_policy": "round_robin",
            "load_balancer_backend_skew_ratio": 0.0,
            "load_balancer_unhealthy_routing_fraction": 0.0,
            "load_balancer_error_rate_percent": 0.0,
            "workload_max_server_count": None,
            "workload_forbidden_rack_ids": [],
            "workload_storage_latency_penalty_ms": 0.0,
            "workload_storage_utilization_ratio": 0.2,
            "workload_service_capacity_requests_per_second": 2000,
            "workload_application_error_rate_percent": 0.0,
            "workload_dropped_requests_per_second": 0.0,
            "failed_servers": 0,
            "power_overloaded_racks": 0,
            "power_budget_violating_racks": 0,
            "max_power_budget_utilization_ratio": 0.5,
            "temperature_sensor_unhealthy_count": 0,
            "max_temperature_sensor_disagreement_c": 0.0,
            "host_health_flapping_count": 0,
            "host_health_status_change_count": 0,
            "thermal_throttled_servers": 0,
            "min_thermal_throttle_factor": 1.0,
            "workload_allocated_server_ids": allocated_server_ids
            or ["server-r1-row1-rack01-01"],
            "active_faults": [],
        }

    def _set_fault_specific_telemetry(self, payload):
        if payload["fault_type"] in {
            "cooling_degradation",
            "rack_hotspot",
        }:
            self.summary.update(
                {
                    "thermal_critical": 1,
                    "sla_status": "violated",
                }
            )
        elif payload["fault_type"] == "storage_io_saturation":
            self.summary.update(
                {
                    "workload_storage_latency_penalty_ms": 150.0,
                    "workload_storage_utilization_ratio": 1.0,
                }
            )
        elif payload["fault_type"] == "control_plane_degradation":
            self.summary.update(
                {
                    "workload_service_capacity_requests_per_second": 250,
                    "workload_queue_length": 1250,
                    "sla_status": "violated",
                }
            )
        elif payload["fault_type"] == "application_error":
            self.summary.update(
                {
                    "workload_application_error_rate_percent": 35.0,
                    "workload_dropped_requests_per_second": 350.0,
                    "sla_status": "violated",
                }
            )
        elif payload["fault_type"] == "server_failure":
            self.summary["failed_servers"] = 1
        elif payload["fault_type"] == "power_overload":
            self.summary.update({"power_overloaded_racks": 1, "sla_status": "violated"})
        elif payload["fault_type"] == "thermal_sensor_miscalibration":
            self.summary.update(
                {
                    "temperature_sensor_unhealthy_count": 1,
                    "max_temperature_sensor_disagreement_c": 12.6,
                    "thermal_critical": 1,
                    "sla_status": "violated",
                }
            )
        elif payload["fault_type"] == "power_budget_violation":
            self.summary.update(
                {
                    "power_budget_violating_racks": 1,
                    "max_power_budget_utilization_ratio": 1.4,
                    "sla_status": "violated",
                }
            )
        elif payload["fault_type"] == "intermittent_server_failure":
            self.summary.update(
                {
                    "host_health_flapping_count": 1,
                    "host_health_status_change_count": 1,
                    "failed_servers": 1,
                    "sla_status": "violated",
                }
            )
        elif payload["fault_type"] == "thermal_throttling":
            self.summary.update(
                {
                    "thermal_throttled_servers": 10,
                    "min_thermal_throttle_factor": 0.325,
                    "workload_queue_length": 1200,
                    "sla_status": "violated",
                }
            )
        elif payload["fault_type"] == "network_congestion_burst":
            self.summary.update(
                {
                    "workload_network_congestion_ratio": 1.5,
                    "workload_average_latency_ms": 750.0,
                    "workload_p95_latency_ms": 1400.0,
                    "workload_queue_length": 1800,
                    "sla_status": "violated",
                }
            )
        elif payload["fault_type"] == "tor_packet_loss":
            self.summary.update(
                {
                    "workload_network_packet_loss_percent": 10.2,
                    "workload_network_retransmit_rate": 23.8,
                    "workload_network_error_rate": 5.1,
                    "workload_network_affected_rack_id": payload["target"],
                    "network_packet_loss_racks": 1,
                    "max_network_packet_loss_percent": 10.2,
                    "network_retransmit_rate": 23.8,
                    "network_error_rate": 5.1,
                    "affected_rack_id": payload["target"],
                    "workload_average_latency_ms": 350.0,
                    "workload_p95_latency_ms": 650.0,
                    "sla_status": "violated",
                }
            )
        elif payload["fault_type"] == "autoscaler_misconfiguration":
            self.summary.update(
                {
                    "autoscaler_max_capacity": 2,
                    "autoscaler_target_utilization_percent": 98.0,
                    "autoscaler_cooldown_seconds": 900,
                    "autoscaler_effective_server_limit": 2,
                    "autoscaler_current_capacity_units": 2,
                    "autoscaler_status": "misconfigured",
                    "workload_service_capacity_requests_per_second": 2000,
                    "workload_queue_length": 1800,
                    "sla_status": "violated",
                }
            )
        elif payload["fault_type"] == "monitoring_pipeline_failure":
            self.summary.update(
                {
                    "metrics_last_updated_sim_time_seconds": max(
                        0, self.summary["sim_time_seconds"] - 100
                    ),
                    "logs_last_updated_sim_time_seconds": max(
                        0, self.summary["sim_time_seconds"] - 100
                    ),
                    "telemetry_lag_seconds": 100,
                    "metrics_missing_ratio": 0.6,
                    "logs_missing_ratio": 0.5,
                    "telemetry_pipeline_status": "stale",
                }
            )
        elif payload["fault_type"] == "placement_policy_misconfiguration":
            self.summary.update(
                {
                    "placement_policy_status": "misconfigured",
                    "placement_policy_target_rack_id": payload["target"],
                    "placement_policy_violating_racks": 1,
                    "workload_placement_imbalance_ratio": 1.0,
                    "workload_queue_length": 1600,
                    "workload_allocated_server_ids": [
                        self._server_id_for_rack(payload["target"])
                    ],
                    "sla_status": "violated",
                }
            )
        elif payload["fault_type"] == "load_balancer_misconfiguration":
            self.summary.update(
                {
                    "load_balancer_routing_policy": "sticky",
                    "load_balancer_backend_skew_ratio": 0.82,
                    "load_balancer_unhealthy_routing_fraction": 0.82,
                    "load_balancer_error_rate_percent": 35.0,
                    "load_balancer_unhealthy_backend_ids": ["server-r1-row1-rack02-03"],
                    "workload_application_error_rate_percent": 35.0,
                    "workload_dropped_requests_per_second": 1575.0,
                    "workload_queue_length": 1800,
                    "sla_status": "violated",
                }
            )

    def _set_network_degraded(self):
        self.summary.update(
            {
                "sla_status": "violated",
                "workload_queue_length": max(
                    self.summary.get("workload_queue_length", 0), 2000
                ),
                "workload_network_congestion_ratio": 0.9,
                "workload_average_latency_ms": 2000.0,
                "workload_allocated_server_ids": [],
                "active_faults": [dict(self.active_fault)] if self.active_fault else [],
            }
        )

    def _set_network_recovered(self, target_rack):
        self.summary.update(
            {
                "sla_status": "normal",
                "workload_queue_length": 0,
                "workload_network_congestion_ratio": 0.3,
                "workload_average_latency_ms": 8.0,
                "workload_allocated_server_ids": [
                    self._server_id_for_rack(target_rack)
                ],
                "active_faults": [dict(self.active_fault)] if self.active_fault else [],
            }
        )

    def _server_id_for_rack(self, rack_id):
        rack_number = rack_id.rsplit("-", 1)[-1]
        return f"server-r1-row1-rack{rack_number}-01"

    def _expire_active_fault_if_needed(self):
        if not self.active_fault:
            return
        started_at = self.active_fault["started_at_sim_time_seconds"]
        duration = self.active_fault["duration_seconds"]
        if self.summary["sim_time_seconds"] - started_at < duration:
            return
        self.active_fault = None
        self.network_fault_active = False
        self.summary["active_faults"] = []

    def _summary_copy(self):
        return deepcopy(self.summary)


class ResetFailureFakeDataCenterTwin(FakeDataCenterTwin):
    def agent_reset(self, *args, **kwargs):
        self.calls.append(("agent_reset_failed", args, kwargs))
        return {"http_status": 400, "error": {"detail": "invalid reset workload"}}


class FaultFailureFakeDataCenterTwin(FakeDataCenterTwin):
    def request_in_pod(self, method, path, payload=None):
        self.calls.append(("legacy", method, path, payload))
        if method == "POST" and path == "/faults":
            return json.dumps(
                {"http_status": 400, "error": {"detail": "invalid fault target"}}
            )
        return json.dumps({"status": "ok"})


def test_canonical_action_response_has_no_second_legacy_projection(monkeypatch):
    from dc_twin.simulator import DataCenterSimulator

    cooling_failure = load_cooling_failure_module(monkeypatch)
    canonical_snapshot = DataCenterSimulator(
        telemetry_opaque_key=b"canonical-action-test-private-key",
    ).canonical_snapshot()

    class CanonicalFakeDataCenterTwin(FakeDataCenterTwin):
        def agent_telemetry(
            self,
            lookback_seconds=300,
            channels=None,
            log_limit=20,
            include_config=True,
        ):
            self.calls.append(
                (
                    "agent_telemetry",
                    lookback_seconds,
                    channels,
                    log_limit,
                    include_config,
                )
            )
            return deepcopy(canonical_snapshot)

        def agent_action(self, action_type, parameters=None, **kwargs):
            response = super().agent_action(
                action_type,
                parameters=parameters,
                **kwargs,
            )
            response["step_summary"] = {
                "sim_time_seconds": self.summary["sim_time_seconds"],
                "sla_status": self.summary["sla_status"],
            }
            return response

    monkeypatch.setattr(
        cooling_failure,
        "DataCenterTwin",
        CanonicalFakeDataCenterTwin,
    )

    task = cooling_failure.DataCenterTwinCoolingDegradationDetection()
    task.configure_telemetry_view("canonical")
    task.start_workload()
    task.inject_fault()
    response = task.perform_action(
        "dc_twin_action",
        "noop",
        advance_ticks=1,
    )

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert response["observation"]["schema_version"] == ("statebundle.canonical.v1")
    assert "step_summary" not in response
    assert "summary" not in set(keys(response))


def test_canonical_observe_propagates_agent_query_controls(monkeypatch):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    from dc_twin.simulator import DataCenterSimulator

    class CanonicalQueryFakeDataCenterTwin(FakeDataCenterTwin):
        def __init__(self):
            super().__init__()
            self.simulator = DataCenterSimulator(
                telemetry_opaque_key=b"canonical-query-controls-private-key",
            )
            self.simulator.start_workload(
                {"request_rate_per_second": 400, "noise_enabled": False}
            )
            self.simulator.step(3)

        def agent_telemetry(self, **kwargs):
            self.calls.append(("agent_telemetry", kwargs))
            return self.simulator.canonical_snapshot(**kwargs)

    monkeypatch.setattr(
        cooling_failure,
        "DataCenterTwin",
        CanonicalQueryFakeDataCenterTwin,
    )

    task = cooling_failure.DataCenterTwinCoolingDegradationDetection()
    task.configure_telemetry_view("canonical")
    observation = task.dc_twin_observe(
        log_limit=0,
        include_config=False,
        lookback_seconds=2,
        channels=["metric", "log", "config"],
        detail="raw",
        metric_names=["rack.inlet_temperature"],
        entity_ids=["rack-r1-row1-1"],
    )

    assert task.app.calls[-1] == (
        "agent_telemetry",
        {
            "lookback_seconds": 2,
            "channels": ["log", "metric", "config"],
            "include_config": False,
            "log_limit": 0,
        },
    )
    assert observation["channels"] == ["log", "metric"]
    assert observation["channel_counts"]["log"] == 0
    assert all(item["channel"] == "metric" for item in observation["observations"])
    assert task.agent_action_history[-1]["request"] == {
        "log_limit": 0,
        "include_config": False,
        "lookback_seconds": 2,
        "channels": ["metric", "log", "config"],
        "detail": "raw",
        "metric_names": ["rack.inlet_temperature"],
        "entity_ids": ["rack-r1-row1-1"],
        "telemetry_view": "canonical",
    }


def test_statebundle_observe_filters_only_after_complete_snapshot_selection(
    monkeypatch,
):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    from dc_twin.simulator import DataCenterSimulator

    class CanonicalQueryFakeDataCenterTwin(FakeDataCenterTwin):
        def __init__(self):
            super().__init__()
            self.simulator = DataCenterSimulator(
                telemetry_opaque_key=b"statebundle-query-controls-private-key",
            )
            self.simulator.start_workload(
                {"request_rate_per_second": 400, "noise_enabled": False}
            )
            self.simulator.step(3)

        def agent_telemetry(self, **kwargs):
            self.calls.append(("agent_telemetry", kwargs))
            return self.simulator.canonical_snapshot(**kwargs)

    monkeypatch.setattr(
        cooling_failure,
        "DataCenterTwin",
        CanonicalQueryFakeDataCenterTwin,
    )

    transform_inputs = []

    def public_observation(item):
        item = deepcopy(item)
        metadata = item["metadata"]
        metadata.pop("primary_subsystem_provenance", None)
        metadata.pop("correlation_ids", None)
        metadata.pop("source_references", None)
        for entity in metadata.get("entities", []):
            entity.pop("provenance", None)
        return item

    def transform(snapshot, *, request, action):
        transform_inputs.append(deepcopy(snapshot))
        selected = {
            channel: public_observation(
                next(
                    item
                    for item in snapshot["observations"]
                    if item["channel"] == channel
                )
            )
            for channel in ("config", "log", "metric")
        }
        return {
            "schema_version": "statebundle.output.v1",
            "incident_id": snapshot["episode_id"],
            "snapshot_id": snapshot["snapshot_id"],
            "query_time_seconds": snapshot["query_time_seconds"],
            "token_budget": 100,
            "used_tokens": 3,
            "candidate_count": len(snapshot["observations"]),
            "evidence_groups": [
                {
                    "anchor": selected["config"],
                    "corroborating_observations": [
                        selected["log"],
                        selected["metric"],
                    ],
                }
            ],
        }

    task = cooling_failure.DataCenterTwinCoolingDegradationDetection()
    task.configure_telemetry_view("canonical")
    task.configure_observation_transform(transform, condition="statebundle")
    observation = task.dc_twin_observe(
        log_limit=0,
        include_config=False,
        lookback_seconds=2,
        channels=["metric", "log", "config"],
    )

    assert task.app.calls[-2] == (
        "agent_telemetry",
        {"lookback_seconds": task.CANONICAL_LOOKBACK_SECONDS},
    )
    requested_call = task.app.calls[-1]
    assert requested_call[0] == "agent_telemetry"
    assert requested_call[1] == {
        "query_time_seconds": transform_inputs[0]["query_time_seconds"],
        "query_watermark_sequence": transform_inputs[0]["query_watermark_sequence"],
        "lookback_seconds": 2,
        "channels": ["log", "metric", "config"],
        "include_config": False,
        "log_limit": 0,
    }
    assert len(transform_inputs) == 1
    assert transform_inputs[0]["channels"] == list(task.CANONICAL_CHANNELS)
    assert transform_inputs[0]["channel_counts"]["config"] > 0
    assert transform_inputs[0]["channel_counts"]["log"] > 0
    assert task._latest_complete_canonical_snapshot == transform_inputs[0]
    assert "_latest_complete_canonical_snapshot" not in json.dumps(
        task.agent_action_history,
        sort_keys=True,
    )
    selected_channels = [
        item["channel"]
        for group in observation["evidence_groups"]
        for item in [group["anchor"], *group["corroborating_observations"]]
    ]
    assert selected_channels == ["metric"]


def test_statebundle_rematerialization_never_substitutes_config_occurrences(
    monkeypatch,
):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    monkeypatch.setattr(cooling_failure, "DataCenterTwin", FakeDataCenterTwin)
    task = cooling_failure.DataCenterTwinCoolingDegradationDetection()

    def config_observation(identifier, change_time, operation):
        return {
            "observation_id": identifier,
            "channel": "config",
            "window": {
                "start_time_seconds": change_time,
                "end_time_seconds": change_time,
            },
            "payload": {
                "scope": "rack/rack-a",
                "path": "cooling.target_temperature_celsius",
                "change_time_seconds": change_time,
                "operation": operation,
                "value": 22.0,
            },
            "metadata": {"event_end_time_seconds": change_time},
        }

    selected_old = config_observation("selected-old", 100.0, "update")
    requested_new = config_observation("requested-new", 200.0, "state")
    output = {
        "schema_version": "statebundle.output.v1",
        "evidence_groups": [{"anchor": selected_old, "corroborating_observations": []}],
    }
    requested_snapshot = {
        "schema_version": "statebundle.canonical.v1",
        "observations": [requested_new],
    }

    filtered = task._filter_selected_statebundle_observations(
        output,
        requested_snapshot=requested_snapshot,
    )

    assert filtered["evidence_groups"] == []

    same_occurrence = config_observation("same-cut-rematerialized", 100.0, "state")
    requested_snapshot["observations"] = [same_occurrence]
    filtered = task._filter_selected_statebundle_observations(
        output,
        requested_snapshot=requested_snapshot,
    )
    assert filtered["evidence_groups"][0]["anchor"] == same_occurrence


def test_statebundle_rematerialization_preserves_pre_action_comparator(monkeypatch):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    monkeypatch.setattr(cooling_failure, "DataCenterTwin", FakeDataCenterTwin)
    task = cooling_failure.DataCenterTwinCoolingDegradationMitigation()

    def metric(identifier, value):
        return {
            "observation_id": identifier,
            "channel": "metric",
            "window": {"start_time_seconds": 90.0, "end_time_seconds": 100.0},
            "payload": {
                "metric_name": "workload.error_rate",
                "values": [value],
                "resource": {"entity_id": "workload"},
            },
            "metadata": {
                "event_end_time_seconds": 100.0,
                "primary_subsystem": "application",
                "entities": [],
            },
        }

    retained = metric(
        "retained:pre-action-1:metric:workload.error_rate:workload:metric-pre",
        35.0,
    )
    current = metric("metric-post", 20.0)
    output = {
        "schema_version": "statebundle.output.v1",
        "evidence_groups": [
            {"anchor": current, "corroborating_observations": [retained]}
        ],
    }
    requested_snapshot = {
        "schema_version": "statebundle.canonical.v1",
        "observations": [current],
    }

    filtered = task._filter_selected_statebundle_observations(
        output,
        requested_snapshot=requested_snapshot,
    )

    group = filtered["evidence_groups"][0]
    assert group["anchor"] == current
    assert group["corroborating_observations"] == [retained]


def test_data_center_twin_registered_mitigation_uses_agent_lifecycle(monkeypatch):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    created_apps = []

    def app_factory():
        app = FakeDataCenterTwin()
        created_apps.append(app)
        return app

    monkeypatch.setattr(cooling_failure, "DataCenterTwin", app_factory)

    task = cooling_failure.DataCenterTwinCoolingDegradationMitigation()
    app = created_apps[0]

    assert task.START_WORKLOAD_BEFORE_FAULT is True

    task.start_workload()
    task.inject_fault()
    action_result = task.perform_action(
        "dc_twin_action",
        "set_cooling",
        parameters={
            "target": "cooling-unit-1",
            "fan_speed_percent": 100,
            "supply_air_temperature_c": 16,
        },
        advance_ticks=1,
        include_config=True,
    )
    results = task.eval(None, [], 1.0)

    call_names = [call[0] for call in app.calls]
    assert "agent_reset" in call_names
    assert "agent_telemetry" in call_names
    assert (
        "legacy",
        "POST",
        "/faults",
        {
            "fault_type": task.scenario.fault_type,
            "target": task.scenario.fault_target,
            "severity": task.scenario.fault_severity,
            "duration_seconds": task.scenario.fault_duration_seconds,
        },
    ) in app.calls
    assert (
        len(
            [
                call
                for call in app.calls
                if call[0] == "agent_action" and call[1] == "noop"
            ]
        )
        == 10
    )
    assert task.initial_agent_observation["query_time_seconds"] == 5
    assert action_result["action_result"]["status"] == "set_cooling"
    assert task.latest_agent_observation["query_time_seconds"] == 16
    assert results["mitigation_success"] is True
    assert results["agent_action_space"]["agent_actions"] == AGENT_ACTIONS
    assert results["fault_injection_time"] == 5.0
    assert results["episode_horizon"] == 1805.0
    assert results["stability_window_length"] == 10.0
    assert results["stable_recovery_succeeded"] is True
    assert results["recovery_start_time"] == 6.0
    assert results["time_to_stable_recovery"] == 1.0
    assert results["penalized_time_to_stable_recovery"] == 1.0
    assert results["slo_violation_area"] >= 0.0
    assert results["raw_slo_violation_duration"] >= 0.0

    trajectory = results["health_trajectory"]
    assert [sample["simulator_time"] for sample in trajectory] == sorted(
        sample["simulator_time"] for sample in trajectory
    )
    assert trajectory[0]["source"] == "fault_injection"
    assert trajectory[-1]["stable_recovery_achieved"] is True
    assert all(sample["after_fault_injection"] for sample in trajectory)
    for sample in trajectory:
        assert {
            "simulator_time",
            "after_fault_injection",
            "constraints",
            "health_condition",
            "evaluator_state",
            "stable_recovery_achieved",
        } <= set(sample)
        for constraint in sample["constraints"]:
            assert {
                "identifier",
                "name",
                "observed_value",
                "threshold",
                "direction",
                "normalized_signed_violation",
                "healthy",
            } <= set(constraint)
            assert constraint["direction"] in {"upper_bound", "lower_bound"}

    constraint_ids = {
        constraint["identifier"] for constraint in trajectory[0]["constraints"]
    }
    assert constraint_ids == {"sla_status", "thermal_critical_max"}
    assert "health_trajectory" not in action_result
    assert "evaluator_state" not in action_result

    from aiopslab.orchestrator.problems.data_center_twin.mitigation_metrics import (
        compute_slo_metrics,
        compute_stable_recovery,
    )

    independently_computed_slo = compute_slo_metrics(
        trajectory,
        fault_injection_time=results["fault_injection_time"],
        episode_horizon=results["episode_horizon"],
        epsilon=results["mitigation_metric_config"]["normalization_epsilon"],
    )
    independently_computed_recovery = compute_stable_recovery(
        trajectory,
        fault_injection_time=results["fault_injection_time"],
        episode_horizon=results["episode_horizon"],
        stability_window=results["stability_window_length"],
    )
    assert (
        independently_computed_slo["slo_violation_area"]
        == results["slo_violation_area"]
    )
    assert (
        independently_computed_slo["raw_slo_violation_duration"]
        == results["raw_slo_violation_duration"]
    )
    assert (
        independently_computed_recovery["time_to_stable_recovery"]
        == results["time_to_stable_recovery"]
    )


def test_stability_qualified_success_uses_exact_recovery_window(monkeypatch):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    monkeypatch.setattr(cooling_failure, "DataCenterTwin", FakeDataCenterTwin)
    task = cooling_failure.DataCenterTwinCoolingDegradationMitigation()

    task._record_mitigation_evaluation_results(
        True,
        {},
        {},
        {
            "stable_recovery_succeeded": False,
            "time_to_stable_recovery": None,
            "penalized_time_to_stable_recovery": 1800.0,
        },
    )

    # Existing benchmark success semantics remain untouched, while the new
    # additive field reflects the manuscript's full elapsed stability window.
    assert task.results["success"] is True
    assert task.results["mitigation_success"] is True
    assert task.results["stability_qualified_success"] is False


def test_data_center_twin_mitigation_accepts_alternative_effective_parameters(
    monkeypatch,
):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    monkeypatch.setattr(cooling_failure, "DataCenterTwin", FakeDataCenterTwin)

    task = cooling_failure.DataCenterTwinCoolingDegradationMitigation()
    task.start_workload()
    task.inject_fault()
    task.perform_action(
        "dc_twin_action",
        "set_cooling",
        parameters={
            "target": "cooling-unit-1",
            "fan_speed_percent": 90,
            "supply_air_temperature_c": 17,
        },
        advance_ticks=1,
    )

    results = task.eval(None, [], 1.0)

    assert results["mitigation_success"] is True


def test_data_center_twin_mitigation_scores_privileged_raw_summary(monkeypatch):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    created_apps = []

    def app_factory():
        app = FakeDataCenterTwin()
        created_apps.append(app)
        return app

    monkeypatch.setattr(cooling_failure, "DataCenterTwin", app_factory)

    task = cooling_failure.DataCenterTwinStorageIoSaturationMitigation()
    task.start_workload()
    task.inject_fault()
    task.perform_action(
        "dc_twin_action",
        "throttle_workload",
        parameters={"request_rate_per_second": 100},
        advance_ticks=1,
    )

    results = task.eval(None, [], 1.0)

    assert results["mitigation_success"] is True
    # This criterion is evaluator-only and intentionally absent from the
    # agent-visible artifact used as final_summary.
    assert any(call[0] == "evaluator_state" for call in created_apps[0].calls)


def test_data_center_twin_mitigation_rejects_control_after_fault_expiry(monkeypatch):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    created_apps = []

    def app_factory():
        app = FakeDataCenterTwin()
        created_apps.append(app)
        return app

    monkeypatch.setattr(cooling_failure, "DataCenterTwin", app_factory)

    task = cooling_failure.DataCenterTwinCoolingDegradationMitigation()
    task.start_workload()
    task.inject_fault()
    expiry_response = task.perform_action(
        "dc_twin_action",
        "noop",
        advance_ticks=task.fault_duration_seconds,
        include_config=True,
    )
    action_result = task.perform_action(
        "dc_twin_action",
        "set_cooling",
        parameters={
            "target": "cooling-unit-1",
            "fan_speed_percent": 100,
            "supply_air_temperature_c": 16,
        },
        advance_ticks=1,
        include_config=True,
    )
    results = task.eval(None, [], 1.0)

    assert_no_forbidden_agent_leakage(expiry_response)
    assert_no_forbidden_agent_leakage(action_result)
    assert "active_faults_after" not in expiry_response
    assert "active_faults_before" not in action_result
    assert action_result["action_result"]["status"] == "set_cooling"
    assert results["mitigation_success"] is False


def test_data_center_twin_mitigation_rejects_wrong_same_type_control_before_expiry(
    monkeypatch,
):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    created_apps = []

    def app_factory():
        app = FakeDataCenterTwin()
        created_apps.append(app)
        return app

    monkeypatch.setattr(cooling_failure, "DataCenterTwin", app_factory)

    task = cooling_failure.DataCenterTwinCoolingDegradationMitigation()
    task.start_workload()
    task.inject_fault()
    wrong_action = task.perform_action(
        "dc_twin_action",
        "set_cooling",
        parameters={
            "target": "cooling-unit-1",
            "fan_speed_percent": 0,
            "supply_air_temperature_c": 40,
        },
        advance_ticks=1,
        include_config=True,
    )
    expiry_response = task.perform_action(
        "dc_twin_action",
        "noop",
        advance_ticks=task.fault_duration_seconds,
        include_config=True,
    )
    results = task.eval(None, [], 1.0)

    assert_no_forbidden_agent_leakage(wrong_action)
    assert_no_forbidden_agent_leakage(expiry_response)
    assert "active_faults_before" not in wrong_action
    assert (
        wrong_action["sim_time_seconds_before"]
        < task.fault_injection_time + task.fault_duration_seconds
    )
    assert "active_faults_after" not in expiry_response
    assert results["mitigation_success"] is False


def test_data_center_twin_network_partition_mitigation_uses_agent_lifecycle(
    monkeypatch,
):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    created_apps = []

    def app_factory():
        app = FakeDataCenterTwin()
        created_apps.append(app)
        return app

    monkeypatch.setattr(cooling_failure, "DataCenterTwin", app_factory)

    task = cooling_failure.DataCenterTwinNetworkPartitionMitigation()
    app = created_apps[0]

    task.start_workload()
    task.inject_fault()
    degraded_observation = task.latest_agent_observation
    noop_results = task.eval(None, [], 1.0)

    call_names = [call[0] for call in app.calls]
    assert "agent_reset" in call_names
    assert "agent_telemetry" in call_names
    assert (
        "legacy",
        "POST",
        "/faults",
        {
            "fault_type": "network_partition",
            "target": "rack-r1-row1-01",
            "severity": 1.0,
            "duration_seconds": 1800,
        },
    ) in app.calls
    assert (
        len(
            [
                call
                for call in app.calls
                if call[0] == "agent_action" and call[1] == "noop"
            ]
        )
        == 10
    )
    assert task.initial_agent_observation["query_time_seconds"] == 5
    assert task.latest_agent_observation["query_time_seconds"] == 15
    assert (
        "agent_reset",
        42,
        {"simulation": {"auto_advance": False}},
        {
            "request_rate_per_second": 2000,
            "workload_class": "web_service",
            "placement_strategy": "spread",
            "noise_enabled": False,
        },
        5,
        20,
        True,
    ) in app.calls
    assert noop_results["mitigation_success"] is False
    assert noop_results["network_partition_recovered"] is False

    created_apps.clear()
    task = cooling_failure.DataCenterTwinNetworkPartitionMitigation()
    app = created_apps[0]

    task.start_workload()
    task.inject_fault()
    action_result = task.perform_action(
        "dc_twin_action",
        "migrate_workload",
        parameters={
            "source_rack_id": "rack-r1-row1-01",
            "target_rack_id": "rack-r1-row1-02",
            "workload_fraction": 1.0,
        },
        advance_ticks=1,
        include_config=True,
    )
    recovered_results = task.eval(None, [], 1.0)

    assert action_result["action_result"]["status"] == "migrate_workload"
    assert recovered_results["mitigation_success"] is True
    assert recovered_results["network_partition_recovered"] is True
    assert recovered_results["final_agent_action"]["action_type"] == "noop"
    assert recovered_results["agent_action_space"]["agent_actions"] == AGENT_ACTIONS


def test_data_center_twin_network_partition_mitigation_is_registered():
    from aiopslab.orchestrator.problems.registry import ProblemRegistry

    registry = ProblemRegistry()

    assert (
        registry.get_problem("data_center_twin-network_partition-mitigation-1").__name__
        == "DataCenterTwinNetworkPartitionMitigation"
    )


def test_data_center_twin_scenario_manifest_covers_supported_faults_and_registry():
    scenarios = load_scenarios_module()
    app_path = (
        Path(__file__).resolve().parents[2]
        / "aiopslab-applications"
        / "dataCenterTwin"
        / "app"
    )
    sys.path.insert(0, str(app_path))
    try:
        from dc_twin.faults import SUPPORTED_FAULTS
    finally:
        try:
            sys.path.remove(str(app_path))
        except ValueError:
            pass

    from aiopslab.orchestrator.problems.registry import ProblemRegistry

    scenario_map = scenarios.validate_scenario_manifest(
        supported_faults=SUPPORTED_FAULTS
    )
    registry = ProblemRegistry()
    registered_ids = {
        problem_id
        for problem_id in registry.get_problem_ids()
        if problem_id.startswith("data_center_twin-")
    }
    registered_class_names = {
        problem_id: registry.get_problem(problem_id).__name__
        for problem_id in registered_ids
    }
    core_task_types = {"detection", "localization", "analysis", "mitigation"}

    assert registered_ids == set(scenario_map)
    assert len(SUPPORTED_FAULTS) == 18
    assert len(registered_ids) == 72
    assert len(scenario_map) == len(SUPPORTED_FAULTS) * len(core_task_types)
    assert scenarios.SUPPORTED_TASK_TYPES == core_task_types
    assert registered_class_names == {
        problem_id: scenario.class_name for problem_id, scenario in scenario_map.items()
    }
    assert {
        scenario.fault_type for scenario in scenario_map.values()
    } == SUPPORTED_FAULTS
    task_types_by_fault = {
        fault_type: {
            scenario.task_type
            for scenario in scenario_map.values()
            if scenario.fault_type == fault_type
        }
        for fault_type in SUPPORTED_FAULTS
    }
    assert all(
        task_types == core_task_types for task_types in task_types_by_fault.values()
    )
    assert all(
        set(scenario.allowed_agent_actions) <= set(AGENT_ACTIONS)
        for scenario in scenario_map.values()
    )
    assert all(
        scenario.config_override["simulation"]["auto_advance"] is False
        for scenario in scenario_map.values()
    )


def test_data_center_twin_registry_entrypoint_has_no_import_time_external_dependencies(
    tmp_path,
):
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["TIKTOKEN_CACHE_DIR"] = str(tmp_path / "tiktoken-cache")
    env["KUBECONFIG"] = str(tmp_path / "missing-kubeconfig")
    env.pop("DC_TWIN_CONFIG", None)

    command = [
        sys.executable,
        "-c",
        "\n".join(
            [
                "from aiopslab.orchestrator.problems.registry import ProblemRegistry",
                "registry = ProblemRegistry()",
                "problem = registry.get_problem_instance('data_center_twin-storage_io_saturation-detection-1')",
                "print(type(problem).__name__)",
                "print(problem.scenario.problem_id)",
            ]
        ),
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DataCenterTwinStorageIoSaturationDetection" in result.stdout
    assert "data_center_twin-storage_io_saturation-detection-1" in result.stdout


def test_data_center_twin_scenario_validation_rejects_invalid_fixture(tmp_path):
    scenarios = load_scenarios_module()
    raw_manifest = json.loads(scenarios.SCENARIOS_PATH.read_text())
    invalid_record = dict(raw_manifest["scenarios"][0])
    invalid_record["problem_id"] = "data_center_twin-network_partition-invalid-1"
    invalid_record["fault"] = {
        "type": "network_partition",
        "target": "storage",
        "severity": 0.5,
        "duration_seconds": 300,
    }
    invalid_record["expected"] = dict(invalid_record["expected"])
    invalid_record["expected"]["evidence_terms"] = []
    invalid_path = tmp_path / "invalid_scenarios.json"
    invalid_path.write_text(
        json.dumps({"scenarios": [invalid_record]}), encoding="utf-8"
    )

    with pytest.raises(
        scenarios.ScenarioValidationError, match="target must be a rack|evidence_terms"
    ):
        scenarios.load_scenarios(invalid_path)


@pytest.mark.parametrize(
    ("task_class_name", "expected_terms", "rejected_terms"),
    [
        (
            "DataCenterTwinStorageIoSaturationDetection",
            [
                "incident_detected",
                "diagnosis",
                "free-form",
                "underlying fault mechanism",
                "evidence",
                "dc_twin_observe",
            ],
            [
                'submit("yes")',
                "submit('yes')",
                "active_faults",
                "storage_io_saturation",
            ],
        ),
        (
            "DataCenterTwinServerFailureLocalization",
            [
                "datacenter target",
                "component-id-from-telemetry",
                "submit([",
                "dc_twin_observe",
            ],
            ["service names"],
        ),
        (
            "DataCenterTwinCoolingDegradationAnalysis",
            [
                "root-cause",
                "root_cause",
                "free-form",
                "underlying fault mechanism",
                "target",
                "domain",
                "evidence",
            ],
            [
                "operating system",
                "code defect",
                "active_faults",
                "cooling_degradation",
            ],
        ),
        (
            "DataCenterTwinCoolingDegradationMitigation",
            [
                "dc_twin_action",
                "set_cooling",
                "cooling-unit-id-from-telemetry",
                "submit()",
            ],
            ["cooling-unit-1", "active_faults"],
        ),
    ],
)
def test_data_center_twin_task_prompts_match_evaluation_contract(
    monkeypatch,
    task_class_name,
    expected_terms,
    rejected_terms,
):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    created_apps = []

    def app_factory():
        app = FakeDataCenterTwin()
        created_apps.append(app)
        return app

    monkeypatch.setattr(cooling_failure, "DataCenterTwin", app_factory)

    task = getattr(cooling_failure, task_class_name)()
    prompt_text = f"{task.get_task_description()}\n{task.get_instructions()}".lower()

    assert "data center twin" in prompt_text
    assert "dc_twin_action_space" in prompt_text
    assert "dc_twin_observe" in prompt_text
    assert "submit" in prompt_text
    assert_no_forbidden_agent_leakage({"prompt": prompt_text})
    for term in expected_terms:
        assert term.lower() in prompt_text
    for term in rejected_terms:
        assert term.lower() not in prompt_text


def test_data_center_twin_evaluator_context_and_rendered_responses_are_sanitized():
    from clients.data_center_twin_baselines.raw_tool_calling import RawToolCallingAgent
    from clients.data_center_twin_baselines.openai_compatible import (
        OpenAICompatibleSettings,
    )

    evaluate = load_evaluate_data_center_twin_module()
    semantic_problem_id = "data_center_twin-cooling_degradation-detection-1"
    action_space = {
        "agent_actions": AGENT_ACTIONS,
        "read_actions": {"observe": {}},
        "time_actions": {"noop": {}, "step": {}},
        "control_actions": {action: {} for action in WRITE_ACTIONS},
        "benchmark_action_coverage": {"scored_actions": ["set_cooling"]},
        "domain_contract": {
            "incident_domains": {
                "cooling_degradation": {"agent_response_actions": ["set_cooling"]}
            },
        },
    }
    observation = {
        "episode_id": "episode-1",
        "sim_time_seconds": 5,
        "sla_status": "violated",
        "summary": {
            "sla_status": "violated",
            "sim_time_seconds": 5,
            "active_faults": [
                {
                    "fault_type": "cooling_degradation",
                    "target": "cooling-unit-1",
                    "duration_seconds": 300,
                }
            ],
            "thermal_critical": 1,
            "workload_average_latency_ms": 250.0,
            "workload_storage_latency_penalty_ms": 100.0,
        },
        "alerts": [
            {
                "alert_type": "active_fault",
                "message": "Active cooling_degradation fault on cooling-unit-1",
            },
            {
                "alert_type": "RackTemperatureHigh",
                "message": "Rack temperature is above threshold",
            },
        ],
        "recent_events": [
            {
                "event_type": "fault_injected",
                "message": "cooling_degradation injected",
                "details": {"fault_type": "cooling_degradation"},
            },
            {
                "event_type": "workload_started",
                "message": "Workload started",
                "details": {"noise_enabled": False, "workload_class": "web_service"},
            },
        ],
        "available_actions": action_space,
    }

    agent_task_id = evaluate.opaque_agent_task_id(semantic_problem_id)
    context = RawToolCallingAgent(
        OpenAICompatibleSettings(api_key="test")
    ).init_context(
        agent_task_id=agent_task_id,
        task_description="Operate the benchmark.",
        instructions="Use telemetry and submit one API call.",
        actions={"submit": "Submit the task."},
        action_space_payload=action_space,
        initial_observation=observation,
    )
    rendered_response = evaluate.render_env_response(
        {
            "action_type": "observe",
            "active_faults_before": observation["summary"]["active_faults"],
            "active_faults_after": observation["summary"]["active_faults"],
            "score_hints": {"answer": "cooling_degradation"},
            "observation": observation,
            "available_actions": action_space,
            "action_result": {"status": "observed"},
        }
    )
    from aiopslab.orchestrator.problems.data_center_twin.visibility import (
        sanitize_agent_payload,
    )

    sanitized_observation = sanitize_agent_payload(observation)

    assert agent_task_id in context
    assert semantic_problem_id not in context
    assert_no_forbidden_agent_leakage({"context": context})
    assert_no_forbidden_agent_leakage(json.loads(rendered_response))
    assert_no_forbidden_agent_leakage(sanitized_observation)
    assert (
        json.loads(rendered_response)["observation"]["alerts"][0]["alert_type"]
        == "RackTemperatureHigh"
    )
    assert sanitized_observation["summary"]["thermal_critical"] == 1


def test_data_center_twin_setup_rejects_failed_reset(monkeypatch):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    created_apps = []

    def app_factory():
        app = ResetFailureFakeDataCenterTwin()
        created_apps.append(app)
        return app

    monkeypatch.setattr(cooling_failure, "DataCenterTwin", app_factory)

    task = cooling_failure.DataCenterTwinCoolingDegradationMitigation()

    with pytest.raises(
        cooling_failure.DataCenterTwinSetupError, match="agent reset failed"
    ):
        task.start_workload()

    assert task.initial_agent_observation is None
    assert task.latest_agent_observation is None


def test_data_center_twin_setup_rejects_failed_fault_injection(monkeypatch):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    created_apps = []

    def app_factory():
        app = FaultFailureFakeDataCenterTwin()
        created_apps.append(app)
        return app

    monkeypatch.setattr(cooling_failure, "DataCenterTwin", app_factory)

    task = cooling_failure.DataCenterTwinCoolingDegradationMitigation()
    task.start_workload()
    pre_fault_observation = task.latest_agent_observation

    with pytest.raises(
        cooling_failure.DataCenterTwinSetupError, match="fault injection failed"
    ):
        task.inject_fault()

    assert task.fault_injection_time is None
    assert task.latest_agent_observation == pre_fault_observation


def test_data_center_twin_task_gates_write_capable_agent_actions(monkeypatch):
    cooling_failure = load_cooling_failure_module(monkeypatch)
    created_apps = []

    def app_factory():
        app = FakeDataCenterTwin()
        created_apps.append(app)
        return app

    monkeypatch.setattr(cooling_failure, "DataCenterTwin", app_factory)

    read_task_classes = [
        cooling_failure.DataCenterTwinCoolingDegradationDetection,
        cooling_failure.DataCenterTwinCoolingDegradationLocalization,
        cooling_failure.DataCenterTwinCoolingDegradationAnalysis,
    ]
    for task_class in read_task_classes:
        task = task_class()
        actions = task.get_available_actions()
        assert set(actions) == {
            "submit",
            "dc_twin_action_space",
            "dc_twin_observe",
            "dc_twin_action",
        }
        assert "exec_shell" not in actions
        assert "get_metrics" not in actions
        assert "get_logs" not in actions

        read_action_space = task.perform_action("dc_twin_action_space")
        read_observation = task.perform_action("dc_twin_observe")
        assert_no_forbidden_agent_leakage(read_action_space)
        assert_no_forbidden_agent_leakage(read_observation)
        assert set(read_action_space["agent_actions"]) == set(READ_TIME_ACTIONS)
        assert all(
            action not in read_action_space["agent_actions"] for action in WRITE_ACTIONS
        )
        assert read_action_space["control_actions"] == {}
        assert set(
            read_action_space["task_action_scope"]["disabled_control_actions"]
        ) == set(WRITE_ACTIONS)
        assert set(
            read_action_space["task_action_scope"]["executable_agent_actions"]
        ) == set(READ_TIME_ACTIONS)
        assert set(
            read_action_space["task_action_scope"]["executable_task_actions"]
        ) == {
            "dc_twin_action_space",
            "dc_twin_observe",
            "dc_twin_action",
        }
        assert all(
            action not in task.dc_twin_action_space()["agent_actions"]
            for action in WRITE_ACTIONS
        )
        assert "domain_contract" not in read_action_space
        assert "benchmark_action_coverage" not in read_action_space

        observe_result = task.perform_action(
            "dc_twin_action", "observe", include_config=False
        )
        noop_result = task.perform_action(
            "dc_twin_action", "noop", advance_ticks=0, include_config=False
        )
        step_result = task.perform_action(
            "dc_twin_action",
            "step",
            parameters={"ticks": 1},
            include_config=False,
        )
        assert_no_forbidden_agent_leakage(observe_result)
        assert_no_forbidden_agent_leakage(noop_result)
        assert_no_forbidden_agent_leakage(step_result)
        assert observe_result["action_result"]["status"] == "observe"
        assert noop_result["action_result"]["status"] == "noop"
        assert step_result["action_result"]["status"] == "step"
        assert all(
            action not in task.dc_twin_action_space()["agent_actions"]
            for action in WRITE_ACTIONS
        )

        with pytest.raises(cooling_failure.InvalidActionError):
            task.perform_action(
                "dc_twin_action",
                "set_cooling",
                parameters={"target": "cooling-unit-1", "fan_speed_percent": 90},
            )
        with pytest.raises(cooling_failure.InvalidActionError):
            task.dc_twin_action(
                "set_cooling",
                parameters={"target": "cooling-unit-1", "fan_speed_percent": 90},
            )
        with pytest.raises(cooling_failure.InvalidActionError):
            task.perform_action("exec_shell", "cat results.json")
        with pytest.raises(cooling_failure.InvalidActionError):
            task.perform_action(
                "dc_twin_action",
                "observe",
                host_visibility="host",
            )
        assert not any(
            call[0] == "agent_action" and call[1] in WRITE_ACTIONS
            for call in task.app.calls
        )

    task = cooling_failure.DataCenterTwinCoolingDegradationMitigation()
    app = created_apps[-1]

    actions = task.get_available_actions()
    assert set(actions) == {
        "submit",
        "dc_twin_action_space",
        "dc_twin_observe",
        "dc_twin_action",
    }

    action_space = task.perform_action("dc_twin_action_space")
    observation = task.perform_action(
        "dc_twin_observe", log_limit=7, include_config=False
    )
    action_result = task.perform_action(
        "dc_twin_action",
        "set_cooling",
        parameters={"target": "cooling-unit-1", "fan_speed_percent": 90},
        advance_ticks=1,
        include_config=True,
    )
    error_result = task.perform_action("dc_twin_action", "bad_action")

    assert_no_forbidden_agent_leakage(action_space)
    assert_no_forbidden_agent_leakage(observation)
    assert_no_forbidden_agent_leakage(action_result)
    assert_no_forbidden_agent_leakage(error_result)
    assert "domain_contract" not in action_space
    assert "benchmark_action_coverage" not in action_space
    assert action_space["agent_actions"] == AGENT_ACTIONS
    assert set(action_space["control_actions"]) == set(WRITE_ACTIONS)
    assert action_result["action_result"]["status"] == "set_cooling"
    assert task.latest_agent_observation["query_time_seconds"] == 1
    assert error_result["http_status"] == 400
    assert "raw" not in error_result
    assert task.agent_action_history[-1]["response"] == error_result
    assert (
        "agent_action",
        "set_cooling",
        {"target": "cooling-unit-1", "fan_speed_percent": 90},
        {
            "host_visibility": "rack",
            "advance_ticks": 1,
            "include_config": True,
        },
    ) in app.calls


def load_cooling_failure_module(monkeypatch):
    from aiopslab.orchestrator.problems.data_center_twin import cooling_failure

    monkeypatch.setattr(cooling_failure, "DataCenterTwin", FakeDataCenterTwin)
    return cooling_failure


pytestmark = pytest.mark.usefixtures("semantic_transport")
