import argparse
import asyncio
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_data_center_twin.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "test_data_center_twin_episode_metadata_runner",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeApp:
    helm_configs = {}
    namespace = "fake"

    def cleanup(self):
        pass

    def delete(self):
        pass

    def get_app_summary(self):
        return "fake app"


class FakeProblem:
    START_WORKLOAD_BEFORE_FAULT = True

    def __init__(self, runner, *, invalid_submission=False, setup_error=False):
        self.runner = runner
        self.invalid_submission = invalid_submission
        self.setup_error = setup_error
        self.app = FakeApp()
        self.namespace = "fake"
        self.scenario = argparse.Namespace(
            problem_id="data_center_twin-fake-detection-1",
            task_type="detection",
        )
        self.seed = 7
        self.config_override = {}
        self.fault_type = "fake"
        self.faulty_component = "fake-target"

    def start_workload(self):
        if self.setup_error:
            raise RuntimeError("setup failed")

    def inject_fault(self):
        pass

    def recover_fault(self):
        pass

    def get_task_description(self):
        return "Detection task"

    def get_instructions(self):
        return "Use one API call."

    def get_available_actions(self):
        return {"submit": "Submit."}

    def dc_twin_action_space(self):
        return {"agent_actions": []}

    def dc_twin_observe(self, **_kwargs):
        return {"summary": {}}

    def perform_action(self, action_name, *_args, **_kwargs):
        assert action_name == "submit"
        if self.invalid_submission:
            return self.runner.SubmissionStatus.INVALID_SUBMISSION
        return self.runner.SubmissionStatus.VALID_SUBMISSION

    def eval(self, _solution, _history, _duration):
        return {"success": True, "Detection Accuracy": "Correct"}


class SequenceAgent:
    def __init__(self, runner, responses):
        self.token_usage = runner.zero_token_usage()
        self.responses = iter(responses)
        self.invocations = 0

    def init_context(self, **_kwargs):
        return "context"

    async def get_action(self, _input_text):
        self.invocations += 1
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return await response()
        return response


async def never_finishes():
    await asyncio.sleep(1.0)
    return "``\nsubmit()\n```"


def run_episode(
    monkeypatch,
    tmp_path,
    *,
    problem,
    agent,
    max_steps=1,
    timeout=1.0,
    debug_logs=False,
):
    runner = problem.runner

    class FakeRegistry:
        def get_problem_instance(self, _problem_id):
            return problem

    monkeypatch.setattr(runner, "ProblemRegistry", FakeRegistry)
    monkeypatch.setattr(runner, "install_inprocess_app", lambda _problem: None)
    monkeypatch.setattr(runner, "create_agent", lambda _args: agent)
    return asyncio.run(
        runner.run_problem(
            problem.scenario.problem_id,
            args=argparse.Namespace(agent="tool-calling"),
            output_dir=tmp_path,
            max_steps=max_steps,
            timeout_seconds=timeout,
            seed=None,
            agent_name="tool-calling",
            deterministic=True,
            fail_fast=False,
            debug_logs=debug_logs,
            verbose=False,
        )
    )


def assert_ordered_utc_timestamps(result):
    started = datetime.fromisoformat(result["episode_start_timestamp"])
    ended = datetime.fromisoformat(result["episode_end_timestamp"])
    assert started.tzinfo is not None
    assert ended.tzinfo is not None
    assert started.utcoffset() == timezone.utc.utcoffset(started)
    assert ended.utcoffset() == timezone.utc.utcoffset(ended)
    assert started <= ended
    assert result["elapsed_wall_clock_seconds"] >= 0.0
    assert result["agent_turns"] == len(result["agent_turn_trajectory"])
    assert all(
        record["start_timestamp"] <= record["end_timestamp"]
        for record in result["agent_turn_trajectory"]
    )


def test_final_submission_records_metadata(monkeypatch, tmp_path):
    runner = load_runner_module()
    problem = FakeProblem(runner)
    agent = SequenceAgent(runner, ["```\nsubmit({'incident_detected': True})\n```"])

    result = run_episode(monkeypatch, tmp_path, problem=problem, agent=agent)

    assert result["termination_reason"] == "final_submission"
    assert result["agent_turns"] == 1
    assert result["max_agent_turns"] == 1
    assert result["agent_configuration"]["max_agent_turns"] == 1
    assert result["agent_configuration"]["wall_clock_timeout_seconds"] == 1.0
    assert_ordered_utc_timestamps(result)


def test_turn_limit_records_metadata(monkeypatch, tmp_path):
    runner = load_runner_module()
    problem = FakeProblem(runner)
    agent = SequenceAgent(runner, ["not an API call"])

    result = run_episode(monkeypatch, tmp_path, problem=problem, agent=agent)

    assert result["termination_reason"] == "turn_limit"
    assert result["agent_turns"] == 1
    assert_ordered_utc_timestamps(result)


def test_wall_clock_timeout_records_attempted_turn(monkeypatch, tmp_path):
    runner = load_runner_module()
    problem = FakeProblem(runner)
    agent = SequenceAgent(runner, [never_finishes])
    agent.settings = argparse.Namespace(provider="openai", model="gpt-5.6-luna")
    agent.model_call_token_usage = []

    result = run_episode(
        monkeypatch,
        tmp_path,
        problem=problem,
        agent=agent,
        timeout=0.001,
    )

    assert result["termination_reason"] == "wall_clock_timeout"
    assert result["agent_turns"] == 1
    assert result["input_tokens"] is None
    assert result["output_tokens"] is None
    assert result["token_usage_available"] is False
    assert result["model_call_token_usage"][0]["token_count_source"] == "unavailable"
    assert_ordered_utc_timestamps(result)


def test_agent_error_records_metadata(monkeypatch, tmp_path):
    runner = load_runner_module()
    problem = FakeProblem(runner)
    agent = SequenceAgent(runner, [RuntimeError("provider failed")])

    result = run_episode(monkeypatch, tmp_path, problem=problem, agent=agent)

    assert result["termination_reason"] == "agent_error"
    assert result["agent_turns"] == 1
    assert_ordered_utc_timestamps(result)


def test_mitigation_agent_error_has_explicit_negative_stability_fields(
    monkeypatch,
    tmp_path,
):
    runner = load_runner_module()
    problem = FakeProblem(runner)
    problem.scenario.task_type = "mitigation"
    problem.scenario.problem_id = "data_center_twin-fake-mitigation-1"
    agent = SequenceAgent(runner, [RuntimeError("provider failed")])

    result = run_episode(monkeypatch, tmp_path, problem=problem, agent=agent)

    assert result["success"] is False
    assert result["mitigation_success"] is False
    assert result["stability_qualified_success"] is False


def test_episode_artifact_redacts_configured_credentials(monkeypatch, tmp_path):
    runner = load_runner_module()
    problem = FakeProblem(runner)
    secret = "sk-artifact-secret-12345"
    query_secret = "query-secret-67890"
    agent = SequenceAgent(
        runner,
        [RuntimeError(f"provider failed while using {secret}")],
    )
    agent.settings = argparse.Namespace(
        api_key=secret,
        api_key_env="TEST_PROVIDER_API_KEY",
        base_url=(
            "https://user:password@example.test/v1"
            f"?api_key={query_secret}&api-version=2026-01-01"
        ),
        provider="test-provider",
        model="test-model",
    )

    result = run_episode(
        monkeypatch,
        tmp_path,
        problem=problem,
        agent=agent,
        debug_logs=True,
    )
    serialized = json.dumps(result, sort_keys=True)
    debug_text = next((tmp_path / "debug").iterdir()).read_text(encoding="utf-8")

    assert secret not in serialized
    assert query_secret not in serialized
    assert "user:password@" not in serialized
    assert "<redacted>" in serialized
    assert secret not in debug_text
    assert query_secret not in debug_text
    assert "user:password@" not in debug_text


def test_benchmark_setup_error_records_metadata(monkeypatch, tmp_path):
    runner = load_runner_module()
    problem = FakeProblem(runner, setup_error=True)
    agent = SequenceAgent(runner, [])

    result = run_episode(monkeypatch, tmp_path, problem=problem, agent=agent)

    assert result["termination_reason"] == "benchmark_error"
    assert result["agent_turns"] == 0
    assert_ordered_utc_timestamps(result)


def test_invalid_terminal_submission_has_distinct_reason(monkeypatch, tmp_path):
    runner = load_runner_module()
    problem = FakeProblem(runner, invalid_submission=True)
    agent = SequenceAgent(runner, ["```\nsubmit()\n```"])

    result = run_episode(monkeypatch, tmp_path, problem=problem, agent=agent)

    assert result["termination_reason"] == "invalid_submission"
    assert result["success"] is False
    assert result["agent_turns"] == 1
    assert_ordered_utc_timestamps(result)


@pytest.mark.parametrize(
    ("final_state", "submitted", "timed_out", "runtime_error", "expected"),
    [
        ("submitted", True, False, False, "final_submission"),
        ("max_steps", False, False, False, "turn_limit"),
        ("timeout", False, True, False, "wall_clock_timeout"),
        ("agent_error", False, False, True, "agent_error"),
        ("setup_error", False, False, True, "benchmark_error"),
        ("invalid_submission", False, False, False, "invalid_submission"),
        ("cancelled", False, False, False, "cancelled"),
    ],
)
def test_termination_reason_mapping(
    final_state,
    submitted,
    timed_out,
    runtime_error,
    expected,
):
    runner = load_runner_module()
    assert (
        runner.termination_reason_from_state(
            final_state=final_state,
            submitted=submitted,
            timed_out=timed_out,
            runtime_error=runtime_error,
            evaluation_results={},
        )
        == expected
    )
