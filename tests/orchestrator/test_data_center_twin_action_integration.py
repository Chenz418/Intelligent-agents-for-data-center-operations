"""Integration regressions for controlled Data Center Twin action accounting."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from aiopslab.orchestrator.actions.analysis import AnalysisActions
from aiopslab.orchestrator.actions.detection import DetectionActions
from aiopslab.orchestrator.actions.localization import LocalizationActions
from aiopslab.orchestrator.actions.mitigation import MitigationActions
from aiopslab.orchestrator.problems.data_center_twin.cooling_failure import (
    DataCenterTwinBaseTask,
)
from aiopslab.session import SessionItem
from clients.data_center_twin_baselines import blackbox_harness
from clients.data_center_twin_baselines.blackbox_harness import (
    BlackboxEpisodeConfig,
    BridgeServer,
    BridgeState,
)
from clients.data_center_twin_baselines.metrics import (
    InvalidActionCategory,
    classify_action_attempt,
    compute_episode_process_metrics,
)
from scripts import evaluate_data_center_twin as runner


def test_task_observation_preserves_safe_structured_client_error() -> None:
    probe = SimpleNamespace(
        allowed_dc_twin_agent_actions=None,
        _task_action_space=lambda payload: payload,
    )
    raw = {
        "http_status": 400,
        "error": {
            "type": "SimulationError",
            "detail": "log_limit must be a non-negative integer",
        },
        # Privileged/unrelated fields must not ride along with the error.
        "active_faults": [{"fault_type": "cooling_degradation"}],
        "ground_truth": {"answer": "hidden"},
    }

    response = DataCenterTwinBaseTask._task_observation(probe, raw)

    assert response == {
        "http_status": 400,
        "error": {
            "type": "SimulationError",
            "detail": "log_limit must be a non-negative integer",
        },
    }
    classification = classify_action_attempt(
        {
            "step": 1,
            "api_name": "dc_twin_observe",
            "args": [],
            "kwargs": {"log_limit": -1},
            "env_response": json.dumps(response, sort_keys=True),
        }
    )
    assert classification.invalid is True
    assert classification.category is InvalidActionCategory.INVALID_PARAMETERS


@pytest.mark.parametrize(
    ("actions", "args", "kwargs"),
    [
        (DetectionActions(), [], {}),
        (LocalizationActions(), [], {}),
        (AnalysisActions(), [], {}),
        (MitigationActions(), ["unexpected"], {}),
    ],
)
def test_submit_signature_rejects_invalid_arity(actions, args, kwargs) -> None:
    problem = SimpleNamespace(actions=actions)

    with pytest.raises(runner.ActionParameterValidationError):
        runner.validate_task_action_arguments(problem, "submit", args, kwargs)


@pytest.mark.parametrize(
    ("actions", "args"),
    [
        (DetectionActions(), ["Yes"]),
        (LocalizationActions(), [["rack-1"]]),
        (AnalysisActions(), [{"system_level": "rack", "fault_type": "cooling"}]),
        (MitigationActions(), []),
    ],
)
def test_submit_signature_accepts_valid_arity(actions, args) -> None:
    problem = SimpleNamespace(actions=actions)

    runner.validate_task_action_arguments(problem, "submit", args, {})


class _BridgeProblem:
    def __init__(self, *, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def perform_action(self, api_name, *_args, **_kwargs):
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
        if api_name == "dc_twin_action_space":
            return {"agent_actions": ["observe"]}
        return {"summary": {}}


def _bridge_state(*, response=None, error: Exception | None = None) -> BridgeState:
    return BridgeState(
        problem=_BridgeProblem(response=response, error=error),
        history=[SessionItem(role="system", content="task")],
        action_sequence=[],
        errors=[],
        max_steps=20,
        action_space_payload={"agent_actions": ["observe"]},
    )


def _assert_one_invalid_parameter_attempt(state: BridgeState) -> None:
    metrics = compute_episode_process_metrics(state.action_sequence, state.errors)
    assert metrics["attempted_actions"] == 1
    assert metrics["invalid_actions"] == 1
    assert metrics["invalid_actions_by_category"] == {
        InvalidActionCategory.INVALID_PARAMETERS.value: 1
    }


def test_blackbox_rejects_non_object_arguments() -> None:
    state = _bridge_state()

    status, response = state.handle_tool_call(
        {"command": "action_space", "arguments": "not-an-object"}
    )

    assert status == 400
    assert response["ok"] is False
    _assert_one_invalid_parameter_attempt(state)


@pytest.mark.parametrize(
    "arguments",
    [
        {"log_limit": -1},
        {"log_limit": True},
        {"log_limit": 1.5},
        {"log_limit": "20"},
        {"include_config": "false"},
        {"include_config": 1},
        {"unexpected": "value"},
    ],
)
def test_blackbox_observe_rejects_invalid_parameter_types_and_ranges(arguments) -> None:
    state = _bridge_state()

    status, response = state.handle_tool_call(
        {"command": "observe", "arguments": arguments}
    )

    assert status == 400
    assert response["ok"] is False
    _assert_one_invalid_parameter_attempt(state)


def test_blackbox_observe_preserves_valid_false_boolean() -> None:
    state = _bridge_state()

    status, response = state.handle_tool_call(
        {
            "command": "observe",
            "arguments": {"log_limit": 0, "include_config": False},
        }
    )

    assert status == 200
    assert response["ok"] is True
    assert state.action_sequence[0]["kwargs"] == {
        "log_limit": 0,
        "include_config": False,
    }


def test_blackbox_bridge_propagates_structured_client_rejection() -> None:
    inner = {
        "http_status": 400,
        "error": {
            "type": "SimulationError",
            "detail": "server_id does not exist",
        },
    }
    state = _bridge_state(response=inner)

    status, response = state.handle_tool_call({"command": "observe", "arguments": {}})

    assert status == 400
    assert response == {"ok": False, "response": inner}
    assert state.tool_call_log[0]["ok"] is False
    metrics = compute_episode_process_metrics(state.action_sequence, state.errors)
    assert metrics["attempted_actions"] == 1
    assert metrics["invalid_actions"] == 1
    assert metrics["invalid_actions_by_category"] == {
        InvalidActionCategory.INVALID_TARGET.value: 1
    }


def test_blackbox_bridge_records_server_exception_without_blame() -> None:
    state = _bridge_state(error=RuntimeError("simulator crashed"))

    status, response = state.handle_tool_call({"command": "observe", "arguments": {}})

    assert status == 500
    assert response["ok"] is False
    assert any(
        error.get("phase") == "environment" and error.get("type") == "RuntimeError"
        for error in state.errors
    )
    metrics = compute_episode_process_metrics(state.action_sequence, state.errors)
    assert metrics["attempted_actions"] == 1
    assert metrics["invalid_actions"] == 0


def test_authenticated_unsupported_bridge_path_is_one_attempt() -> None:
    state = _bridge_state()
    server = BridgeServer(state, "test-token")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/unsupported",
        data=b"{}",
        method="POST",
        headers={
            "Authorization": "Bearer test-token",
            "Content-Type": "application/json",
        },
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as captured:
            urllib.request.urlopen(request, timeout=2)
        assert captured.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    metrics = compute_episode_process_metrics(state.action_sequence, state.errors)
    assert metrics["attempted_actions"] == 1
    assert metrics["invalid_actions"] == 1
    assert metrics["invalid_actions_by_category"] == {
        InvalidActionCategory.UNSUPPORTED_ACTION.value: 1
    }


@pytest.mark.parametrize(
    "tool_args",
    [
        ["observe", "--log-limit", "not-an-integer"],
        ["action"],
        ["submit", "--unexpected"],
    ],
)
def test_blackbox_cli_argument_failures_reach_central_accounting(
    tmp_path,
    tool_args,
) -> None:
    state = _bridge_state()
    server = BridgeServer(state, "test-token")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    tool_path = tmp_path / "dc_twin_tool.py"
    tool_path.write_text(blackbox_harness.DC_TWIN_TOOL_SOURCE, encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        {
            "DC_TWIN_WORKSPACE": str(tmp_path),
            "DC_TWIN_TOOL_URL": f"http://127.0.0.1:{server.server_port}/tool",
            "DC_TWIN_TOOL_TOKEN": "test-token",
        }
    )
    try:
        completed = subprocess.run(
            [sys.executable, str(tool_path), *tool_args],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert completed.returncode == 2
    metrics = compute_episode_process_metrics(state.action_sequence, state.errors)
    assert metrics["attempted_actions"] == 1
    assert metrics["invalid_actions"] == 1
    assert metrics["invalid_actions_by_category"] == {
        InvalidActionCategory.INVALID_PARAMETERS.value: 1
    }


class _FakeApp:
    namespace = "fake"
    helm_configs = {}

    def cleanup(self) -> None:
        return None

    def delete(self) -> None:
        return None


class _RunnerProblem:
    START_WORKLOAD_BEFORE_FAULT = True

    def __init__(self) -> None:
        self.app = _FakeApp()
        self.namespace = "fake"
        self.seed = 7
        self.config_override = {}
        self.telemetry_view = "canonical"
        self.scenario = argparse.Namespace(
            problem_id="data_center_twin-probe-detection-1",
            task_type="detection",
        )
        self.fault_type = "probe"
        self.faulty_component = "probe-target"

    def start_workload(self) -> None:
        return None

    def inject_fault(self) -> None:
        return None

    def recover_fault(self) -> None:
        return None

    def get_task_description(self) -> str:
        return "Probe task"

    def get_instructions(self) -> str:
        return "Use the APIs."

    def get_available_actions(self):
        return {
            "dc_twin_action_space": "Inspect actions.",
            "dc_twin_observe": "Observe.",
            "submit": "Submit.",
        }

    def dc_twin_action_space(self):
        return {"agent_actions": ["observe"]}

    def dc_twin_observe(self, **_kwargs):
        return {"summary": {}}

    def perform_action(self, api_name, *_args, **_kwargs):
        if api_name == "dc_twin_observe":
            raise RuntimeError("unexpected dispatch failure")
        raise AssertionError(api_name)

    def eval(self, _solution, _history, _duration):
        return {"success": False}


class _OneActionAgent:
    token_usage = runner.zero_token_usage()

    def init_context(self, **_kwargs) -> str:
        return "context"

    async def get_action(self, _input_text: str) -> str:
        return "```\ndc_twin_observe()\n```"


def test_normal_runner_records_unexpected_dispatch_exception(
    monkeypatch, tmp_path
) -> None:
    problem = _RunnerProblem()

    class Registry:
        def get_problem_instance(self, _problem_id):
            return problem

    monkeypatch.setattr(runner, "ProblemRegistry", Registry)
    monkeypatch.setattr(runner, "install_inprocess_app", lambda _problem: None)
    monkeypatch.setattr(runner, "create_agent", lambda _args: _OneActionAgent())

    result = asyncio.run(
        runner.run_problem(
            problem.scenario.problem_id,
            args=argparse.Namespace(agent="tool-calling", telemetry_view="canonical"),
            output_dir=tmp_path,
            max_steps=1,
            timeout_seconds=2.0,
            seed=None,
            agent_name="tool-calling",
            deterministic=True,
            fail_fast=False,
            debug_logs=False,
            verbose=False,
        )
    )

    assert any(
        error.get("phase") == "environment" and error.get("type") == "RuntimeError"
        for error in result["errors"]
    )
    assert result["attempted_actions"] == 1
    assert result["invalid_actions"] == 0


def test_blackbox_post_setup_exception_is_benchmark_error(
    monkeypatch, tmp_path
) -> None:
    problem = _RunnerProblem()

    class Registry:
        def get_problem_instance(self, _problem_id):
            return problem

    monkeypatch.setattr(runner, "ProblemRegistry", Registry)
    monkeypatch.setattr(runner, "install_inprocess_app", lambda _problem: None)

    def fail_after_setup(*_args, **_kwargs):
        raise RuntimeError("post-setup harness failure")

    monkeypatch.setattr(blackbox_harness, "run_external_command", fail_after_setup)
    result = blackbox_harness.run_blackbox_problem(
        problem.scenario.problem_id,
        BlackboxEpisodeConfig(
            agent_name="codex",
            command_template="python missing.py",
            output_dir=Path(tmp_path),
            max_steps=2,
            timeout_seconds=2.0,
            seed=7,
            deterministic=True,
            keep_workspace=False,
            verbose=False,
            sandbox_mode="none",
            telemetry_view="canonical",
        ),
    )

    assert result["status"] == "error"
    assert result["final_state"] == "runner_error"
    assert result["termination_reason"] == "benchmark_error"
    assert any(
        error.get("phase") == "blackbox" and error.get("type") == "RuntimeError"
        for error in result["errors"]
    )
