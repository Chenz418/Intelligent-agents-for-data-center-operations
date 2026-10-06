#!/usr/bin/env python3
"""Batch evaluation runner for registered Data Center Twin benchmark problems.

The runner discovers Data Center Twin problems through the existing
ProblemRegistry, runs each problem against a fresh in-process simulator, drives
an agent through the benchmark action interface, and writes concise artifacts.

The LLM agent uses an OpenAI-compatible chat-completions API. Provider
settings are intentionally blank in committed configuration; provide them at
runtime with environment variables or CLI arguments.

Execution model:
1. Discover registered problem IDs through the normal AIOpsLab ProblemRegistry.
2. For each problem ID, construct a new problem with a fresh in-process
   simulator adapter.
3. Let the problem class own reset, workload start, fault injection, action
   gating, and evaluation. This runner only sequences the lifecycle.
4. Give the agent only sanitized Data Center Twin action/telemetry payloads.
5. Parse one markdown-fenced API call per turn with the existing ResponseParser.
6. Always attempt fault recovery and simulator cleanup, then write concise
   machine-readable and human-readable artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import secrets
import statistics
import sys
import time
import traceback
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
DC_TWIN_APP_DIR = REPO_ROOT / "aiopslab-applications" / "dataCenterTwin" / "app"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "data_center_twin_eval"
BENCHMARK_HOST_VISIBILITY = "rack"
RUNNER_ALLOWED_TASK_ACTIONS = frozenset(
    {
        "submit",
        "dc_twin_action_space",
        "dc_twin_observe",
        "dc_twin_action",
    }
)


class ActionParameterValidationError(ValueError):
    """Raised when parsed arguments cannot bind to a task API signature."""


# Support direct execution without installing the repository as a package.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DC_TWIN_APP_DIR) not in sys.path:
    sys.path.insert(0, str(DC_TWIN_APP_DIR))

from aiopslab.orchestrator.parser import ResponseParser  # noqa: E402
from aiopslab.orchestrator.problems.data_center_twin.semantic_evaluation import (  # noqa: E402
    add_semantic_evaluator_arguments,
    configure_problem_evaluator,
    diagnostic_metadata,
    evaluator_infrastructure_error,
    success_statistics,
    evaluator_options,
)
from aiopslab.agent_telemetry import (  # noqa: E402
    AGENT_DELTA_SCHEMA_VERSION,
    AGENT_SCHEMA_VERSION,
    AgentObservationRequest,
    AgentTelemetryRenderer,
)
from aiopslab.orchestrator.problems.data_center_twin.visibility import (  # noqa: E402
    sanitize_agent_payload,
)
from aiopslab.orchestrator.problems.data_center_twin.scenarios import (  # noqa: E402
    SCENARIOS_PATH,
)
from aiopslab.orchestrator.problems.registry import ProblemRegistry  # noqa: E402
from aiopslab.session import SessionItem  # noqa: E402
from aiopslab.utils.status import (  # noqa: E402
    InvalidActionError,
    ResponseParsingError,
    SubmissionStatus,
)
from clients.data_center_twin_baselines.base import (  # noqa: E402
    TOKEN_USAGE_KEYS,
    TerminationReason,
    compact_action_space,
    unknown_token_usage,
    zero_token_usage,
)
from clients.data_center_twin_baselines.metrics import (  # noqa: E402
    InvalidActionCategory,
    aggregate_process_metrics,
    aggregate_process_metrics_by,
    classify_action_attempt,
    compute_episode_process_metrics,
    ensure_process_metrics,
    is_explicit_unknown_token_usage,
    process_metric_group_key,
)
from clients.data_center_twin_baselines.registry import (  # noqa: E402
    create_agent as create_baseline_agent,
    list_agent_names,
)


from aiopslab.service.apps.data_center_twin import (
    DataCenterTwin as InProcessDataCenterTwinApp,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DC-Bench: 72 tasks, tool-calling or Codex, canonical observations."
    )
    add_semantic_evaluator_arguments(parser)
    parser.add_argument(
        "--agent-type",
        dest="agent",
        choices=("tool-calling", "codex"),
        default="tool-calling",
    )
    parser.add_argument(
        "--observation",
        dest="observation_condition",
        choices=("full-canonical", "statebundle"),
        default="full-canonical",
    )
    parser.add_argument("--model", default=os.getenv("DC_TWIN_LLM_MODEL", ""))
    parser.add_argument(
        "--base-url",
        default=os.getenv("DC_TWIN_LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--api-key-env",
        default=os.getenv("DC_TWIN_LLM_API_KEY_ENV", "DC_TWIN_LLM_API_KEY"),
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "openai_compatible", "gemini_native", "dashscope"),
        default=os.getenv("DC_TWIN_LLM_PROVIDER", "openai"),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=os.getenv("DC_TWIN_LLM_REASONING_EFFORT"),
    )
    parser.add_argument(
        "--thinking-mode", choices=("enabled", "disabled"), default=None
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--use-max-completion-tokens", action="store_true")
    parser.add_argument(
        "--tool-choice", choices=("required", "auto"), default="required"
    )
    parser.add_argument("--rate-limit-max-retries", type=int, default=6)
    parser.add_argument("--rate-limit-initial-delay-seconds", type=float, default=1.0)
    parser.add_argument("--rate-limit-max-delay-seconds", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--problem-filter", help="Regex subset of the fixed 72-task manifest."
    )
    parser.add_argument("--expected-problem-count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trial-index", type=int, default=1)
    parser.add_argument("--task-id-nonce", default=None)
    parser.add_argument("--observation-token-budget", type=int, default=4096)
    parser.add_argument(
        "--statebundle-config",
        type=Path,
        default=REPO_ROOT / "configs/statebundle.dc_twin.yaml",
    )
    parser.add_argument(
        "--statebundle-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/statebundle-stage3/stage3-epoch-9.pt",
    )
    parser.add_argument(
        "--codex-command",
        default=os.getenv("DC_BENCH_CODEX_COMMAND"),
        help="Codex command template; {task_file} and {workspace} are available.",
    )
    parser.add_argument(
        "--codex-env",
        action="append",
        default=[],
        help="Explicit credential/config environment variable passed to Codex.",
    )
    parser.add_argument("--codex-sandbox", choices=("docker", "none"), default="docker")
    parser.add_argument("--codex-docker-image", default="dcbench-codex:local")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug-logs", action="store_true")
    parser.add_argument("--list-problems", action="store_true")
    parser.add_argument("--list-agents", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def create_agent(args: argparse.Namespace):
    """Construct the configured tool-calling agent."""
    return create_baseline_agent(args.agent, args)


def discover_problem_ids(problem_filter: str | None = None) -> list[str]:
    """Return registered Data Center Twin problem IDs, optionally regex-filtered."""
    registry = ProblemRegistry()
    problem_ids = sorted(
        problem_id
        for problem_id in registry.get_problem_ids()
        if problem_id.startswith("data_center_twin-")
    )
    if problem_filter:
        try:
            pattern = re.compile(problem_filter)
        except re.error as error:
            raise SystemExit(f"invalid --problem-filter regex: {error}") from error
        problem_ids = [
            problem_id for problem_id in problem_ids if pattern.search(problem_id)
        ]
    return problem_ids


async def run_problem(
    problem_id: str,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    max_steps: int,
    timeout_seconds: float,
    seed: int | None,
    agent_name: str = "tool-calling",
    deterministic: bool = True,
    agent_task_nonce: str | bytes | None = None,
    fail_fast: bool = False,
    debug_logs: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Run one registered benchmark episode and return its result record."""
    episode_started_at = datetime.now(timezone.utc)
    monotonic_started_at = time.monotonic()
    deadline = monotonic_started_at + timeout_seconds
    debug: list[str] = []
    errors: list[dict[str, Any]] = []
    action_sequence: list[dict[str, Any]] = []
    agent_turn_trajectory: list[dict[str, Any]] = []
    history: list[SessionItem] = []
    parser = ResponseParser()
    registry: Any = None
    agent: Any = None

    problem = None
    solution: Any = None
    final_diagnosis: Any = None
    final_diagnosis_raw: Any = None
    final_state = "not_started"
    evaluation_results: dict[str, Any] = {}
    action_space_payload: dict[str, Any] | None = None
    initial_observation: dict[str, Any] | None = None
    initial_agent_rendering: dict[str, Any] | None = None
    agent_rendered_observations: list[dict[str, Any]] = []
    pending_evaluator_observation: dict[str, Any] | None = None
    observation_renderer: AgentTelemetryRenderer | None = None
    setup_complete = False
    app_installed = False
    submitted = False
    timed_out = False
    runtime_error = False
    agent_turns = 0
    observation_condition = (
        getattr(args, "observation_condition", "full-canonical")
        if args is not None
        else "full-canonical"
    )
    statebundle_processor: Any = None
    observation_audit_start = 0
    statebundle_episode_started = False

    try:
        registry = ProblemRegistry()
        agent = create_agent(args)
        debug.append(f"constructing problem {problem_id}")
        problem = registry.get_problem_instance(problem_id)
        configure_problem_evaluator(problem, args)
        install_inprocess_app(problem)
        app_installed = True
        telemetry_view = (
            getattr(args, "telemetry_view", "canonical")
            if args is not None
            else "canonical"
        )
        if hasattr(problem, "configure_telemetry_view"):
            problem.configure_telemetry_view(telemetry_view)
        if observation_condition in {"statebundle"}:
            statebundle_processor = statebundle_processor_from_args(args)
            observation_audit_start = len(statebundle_processor.audit_records)
            statebundle_processor.begin_episode()
            statebundle_episode_started = True
            if not hasattr(problem, "configure_observation_transform"):
                raise RuntimeError(
                    "selected benchmark problem does not support observation transforms"
                )
            problem.configure_observation_transform(
                statebundle_processor.transform,
                condition="statebundle",
            )
        if telemetry_view == "canonical":
            observation_renderer = AgentTelemetryRenderer(
                condition=normalized_observation_condition(observation_condition),
            )
            if hasattr(problem, "_activate_agent_rendering_boundary"):
                problem._activate_agent_rendering_boundary()
        if deterministic:
            force_deterministic_simulation(problem, seed)
        elif seed is not None:
            problem.seed = seed
        # Preserve the lifecycle order declared by the problem.
        debug.append("starting workload and injecting fault through problem lifecycle")
        if getattr(problem, "START_WORKLOAD_BEFORE_FAULT", False):
            problem.start_workload()
            problem.inject_fault()
        else:
            problem.inject_fault()
            problem.start_workload()
        setup_complete = True

        task_description = problem.get_task_description()
        instructions = problem.get_instructions()
        actions = {
            name: description
            for name, description in problem.get_available_actions().items()
            if name in RUNNER_ALLOWED_TASK_ACTIONS
        }

        debug.append("collecting agent-visible action space and observation")
        action_space_payload = problem.dc_twin_action_space()
        if statebundle_episode_started:
            statebundle_processor.activate_episode()
        initial_observation = problem.dc_twin_observe(log_limit=20, include_config=True)
        agent_task_id = opaque_agent_task_id(problem_id, nonce=agent_task_nonce)

        initial_agent_rendering = render_agent_observation(
            initial_observation,
            renderer=observation_renderer,
            request=AgentObservationRequest(),
            canonical_context=latest_complete_canonical_snapshot(problem),
            initial=True,
        )
        if isinstance(initial_agent_rendering, dict):
            agent_rendered_observations.append(initial_agent_rendering)
            record_agent_rendered_observation_for_evaluation(
                problem,
                initial_agent_rendering,
                renderer=observation_renderer,
            )

        # Keep the trace shape consumed by task evaluators.
        initial_context = agent.init_context(
            agent_task_id=agent_task_id,
            task_description=task_description,
            instructions=instructions,
            actions=actions,
            action_space_payload=action_space_payload,
            initial_observation=initial_agent_rendering,
        )
        history.append(SessionItem(role="system", content=initial_context))
        initial_message = render_initial_observation_message(initial_agent_rendering)
        history.append(SessionItem(role="env", content=initial_message))
        next_input = initial_message + "\nPlease take the next action"

        for step_index in range(1, max_steps + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                final_state = "timeout"
                debug.append("timeout reached before next agent step")
                break

            # An environment response becomes agent-visible only when it is
            # supplied to the next model turn.  Keeping this boundary here
            # prevents a final, unconsumed observation from influencing the
            # evaluator when the episode ends at max_steps or a deadline.
            if pending_evaluator_observation is not None:
                record_agent_rendered_observation_for_evaluation(
                    problem,
                    pending_evaluator_observation,
                    renderer=observation_renderer,
                )
                pending_evaluator_observation = None

            agent_turns += 1
            turn_record = {
                "turn": agent_turns,
                "state_machine_step": step_index,
                "start_timestamp": datetime.now(timezone.utc).isoformat(),
                "end_timestamp": None,
                "status": "in_progress",
            }
            agent_turn_trajectory.append(turn_record)
            try:
                raw_action = await asyncio.wait_for(
                    agent.get_action(next_input), timeout=remaining
                )
            except TimeoutError as error:
                cutoff_timestamp = datetime.now(timezone.utc).isoformat()
                turn_record["end_timestamp"] = cutoff_timestamp
                turn_record["status"] = "wall_clock_timeout"
                timed_out = True
                final_state = "timeout"
                quarantine_provider_accounting(
                    agent,
                    termination_reason=TerminationReason.WALL_CLOCK_TIMEOUT.value,
                    termination_timestamp=cutoff_timestamp,
                    errors=errors,
                )
                errors.append(error_record("agent", error))
                debug.append(f"agent timeout at step {step_index}")
                break
            except Exception as error:
                turn_record["end_timestamp"] = datetime.now(timezone.utc).isoformat()
                turn_record["status"] = "agent_error"
                turn_record["error_type"] = type(error).__name__
                turn_record["error_message"] = str(error)
                runtime_error = True
                final_state = "agent_error"
                errors.append(error_record("agent", error))
                debug.append(f"agent error at step {step_index}: {error}")
                break

            # ``asyncio.wait_for`` can return on the scheduler boundary even
            # though the absolute episode deadline has just elapsed.  Reject
            # that response before it can enter history, normalization,
            # actions, or evaluator state; its usage remains accounting-only.
            if time.monotonic() >= deadline:
                cutoff_timestamp = datetime.now(timezone.utc).isoformat()
                turn_record["end_timestamp"] = cutoff_timestamp
                turn_record["status"] = "wall_clock_timeout"
                timed_out = True
                final_state = "timeout"
                quarantine_provider_accounting(
                    agent,
                    termination_reason=TerminationReason.WALL_CLOCK_TIMEOUT.value,
                    termination_timestamp=cutoff_timestamp,
                    errors=errors,
                )
                errors.append(
                    {
                        "phase": "agent",
                        "type": "TimeoutError",
                        "message": (
                            "provider response completed after the absolute "
                            "episode deadline and was excluded from the trajectory"
                        ),
                        "traceback": "",
                    }
                )
                debug.append(
                    f"agent response excluded after deadline at step {step_index}"
                )
                break

            turn_record["end_timestamp"] = datetime.now(timezone.utc).isoformat()
            turn_record["status"] = "completed"
            turn_record["action_record_step"] = step_index

            history.append(SessionItem(role="assistant", content=raw_action))
            action_record = {
                "step": step_index,
                "raw": raw_action,
                "api_name": None,
                "args": [],
                "kwargs": {},
                "env_response": None,
            }
            normalization_status = getattr(agent, "last_normalization_status", None)
            if isinstance(normalization_status, dict):
                action_record.update(
                    {
                        "normalized_ok": bool(
                            normalization_status.get("normalized_ok")
                        ),
                        "normalization_error": normalization_status.get(
                            "normalization_error"
                        ),
                        "original_model_output": normalization_status.get(
                            "original_model_output"
                        ),
                    }
                )

            # Feed parse errors back unless fail-fast mode is enabled.
            try:
                parsed = parser.parse(raw_action)
                api_name = parsed["api_name"]
                parsed_args = parsed["args"]
                kwargs = parsed["kwargs"]
                action_record.update(
                    {"api_name": api_name, "args": parsed_args, "kwargs": kwargs}
                )
            except ResponseParsingError as error:
                env_response = str(error)
                action_record["env_response"] = env_response
                action_record["parse_error"] = error_record("parser", error)
                history.append(SessionItem(role="env", content=env_response))
                action_sequence.append(annotate_action_accounting(action_record))
                errors.append(error_record("parser", error))
                next_input = env_response + "\nPlease take the next action"
                debug.append(f"parse error at step {step_index}: {error}")
                if fail_fast:
                    final_state = "parse_error"
                    runtime_error = True
                    break
                continue

            if api_name == "submit":
                # Keep the original submission and normalize only its format.
                # This pure transform cannot see expected answers or telemetry.
                solution = solution_from_submit(parsed_args, kwargs)
                final_diagnosis_raw = solution
                final_diagnosis = solution

            observation_request = AgentObservationRequest()
            try:
                # Centralize action gating in the problem implementation.
                if (
                    api_name not in RUNNER_ALLOWED_TASK_ACTIONS
                    or api_name not in actions
                ):
                    raise InvalidActionError(api_name)
                validate_task_action_arguments(problem, api_name, parsed_args, kwargs)
                try:
                    observation_request = observation_request_from_action(
                        api_name, parsed_args, kwargs
                    )
                except (TypeError, ValueError) as error:
                    raise ActionParameterValidationError(str(error)) from error
                env_response_obj = problem.perform_action(
                    api_name, *parsed_args, **kwargs
                )
            except ActionParameterValidationError as error:
                env_response_obj = str(error)
                action_record["action_accounting"] = {
                    "attempted": True,
                    "invalid": True,
                    "category": InvalidActionCategory.INVALID_PARAMETERS.value,
                    "reason": str(error),
                    "source": "task_signature",
                }
            except InvalidActionError as error:
                env_response_obj = str(error)
                category = (
                    InvalidActionCategory.UNSUPPORTED_ACTION
                    if api_name not in RUNNER_ALLOWED_TASK_ACTIONS
                    else InvalidActionCategory.DISALLOWED_ACTION
                )
                action_record["action_accounting"] = {
                    "attempted": True,
                    "invalid": True,
                    "category": category.value,
                    "reason": str(error),
                    "source": "scenario_action_scope",
                }
            except Exception as error:
                env_response_obj = str(error)
                action_record["action_accounting"] = {
                    "attempted": True,
                    "invalid": False,
                    "category": None,
                    "reason": str(error),
                    "source": "benchmark_environment_error",
                }
                errors.append(error_record("environment", error))
                debug.append(f"environment action error at step {step_index}: {error}")

            rendered_env_response, rendered_observation = (
                render_env_response_with_observation(
                    env_response_obj,
                    renderer=observation_renderer,
                    request=observation_request,
                    canonical_context=latest_complete_canonical_snapshot(problem),
                )
            )
            if isinstance(rendered_observation, dict):
                agent_rendered_observations.append(rendered_observation)
                pending_evaluator_observation = rendered_observation
            action_record["env_response"] = rendered_env_response
            if env_response_obj == SubmissionStatus.INVALID_SUBMISSION:
                action_record["action_accounting"] = {
                    "attempted": True,
                    "invalid": True,
                    "category": InvalidActionCategory.REJECTED_SUBMISSION.value,
                    "reason": "Agent made an invalid submission.",
                    "source": "submission_status",
                }
            action_sequence.append(annotate_action_accounting(action_record))
            history.append(SessionItem(role="env", content=rendered_env_response))

            if env_response_obj == SubmissionStatus.VALID_SUBMISSION:
                submitted = True
                final_state = "submitted"
                debug.append(f"valid submission at step {step_index}")
                break
            if env_response_obj == SubmissionStatus.INVALID_SUBMISSION:
                final_state = "invalid_submission"
                errors.append(
                    {
                        "phase": "environment",
                        "type": "InvalidSubmission",
                        "message": "Agent made an invalid submission.",
                    }
                )
                break

            next_input = rendered_env_response + "\nPlease take the next action"

        if final_state == "not_started":
            final_state = "max_steps"

        if should_evaluate(
            final_state=final_state, setup_complete=setup_complete, submitted=submitted
        ):
            evaluation_results = evaluate_problem(
                problem, solution, history, monotonic_started_at, errors
            )

    except Exception as error:
        runtime_error = True
        final_state = "setup_error" if not setup_complete else "runner_error"
        errors.append(error_record("setup" if not setup_complete else "runner", error))
        debug.append(traceback.format_exc())
        if (
            setup_complete
            and problem is not None
            and should_evaluate(
                final_state=final_state,
                setup_complete=setup_complete,
                submitted=submitted,
            )
        ):
            evaluation_results = evaluate_problem(
                problem, solution, history, monotonic_started_at, errors
            )
    finally:
        if problem is not None and app_installed:
            if setup_complete and hasattr(problem, "finalize_mitigation_metrics"):
                try:
                    mitigation_metrics = problem.finalize_mitigation_metrics()
                    if isinstance(mitigation_metrics, dict):
                        for key, value in mitigation_metrics.items():
                            evaluation_results.setdefault(key, value)
                except Exception as error:
                    errors.append(error_record("metric_instrumentation", error))
            # Record cleanup failures without masking the episode outcome.
            try:
                problem.recover_fault()
            except Exception as error:
                errors.append(error_record("cleanup", error))
            try:
                problem.app.cleanup()
            except Exception as error:
                errors.append(error_record("cleanup", error))
            try:
                problem.app.delete()
            except Exception as error:
                errors.append(error_record("cleanup", error))

    episode_ended_at = datetime.now(timezone.utc)
    runtime_seconds = time.monotonic() - monotonic_started_at
    provider_accounting = await settle_provider_accounting(
        agent,
        errors=errors,
    )
    artifact_finalized_at = datetime.now(timezone.utc)
    evaluator_error = evaluator_infrastructure_error(evaluation_results)
    success = None if evaluator_error else bool(evaluation_results.get("success"))
    status = status_from_state(
        final_state=final_state,
        success=success,
        submitted=submitted,
        timed_out=timed_out,
        runtime_error=runtime_error,
    )
    if evaluator_error:
        status = "evaluator_infrastructure_error"
    score, accuracy = extract_score_accuracy(evaluation_results, success)
    termination_reason = termination_reason_from_state(
        final_state=final_state,
        submitted=submitted,
        timed_out=timed_out,
        runtime_error=runtime_error,
        evaluation_results=evaluation_results,
    )

    agent_visible_observations = (
        problem._agent_visible_observations()
        if problem is not None and hasattr(problem, "_agent_visible_observations")
        else []
    )
    agent_token_usage = getattr(agent, "token_usage", zero_token_usage())
    model_call_token_usage = getattr(agent, "model_call_token_usage", None)
    agent_token_usage, model_call_token_usage = account_unfinished_model_turn_usage(
        agent=agent,
        agent_usage=agent_token_usage,
        model_call_token_usage=model_call_token_usage,
        agent_turns=agent_turns,
        final_state=final_state,
    )
    token_usage = result_token_usage(
        agent_token_usage,
        evaluation_results,
        model_call_token_usage=model_call_token_usage,
    )
    token_usage_available = is_token_usage_available(token_usage)
    process_metrics = compute_episode_process_metrics(
        action_sequence,
        errors,
        runtime_seconds=runtime_seconds,
        token_usage=token_usage,
    )

    # Include scoring fields and the sanitized trace needed for diagnosis.
    agent_configuration = public_agent_configuration(agent)
    agent_configuration.update(
        {
            "max_agent_turns": max_steps,
            "wall_clock_timeout_seconds": timeout_seconds,
            "deterministic_simulation": deterministic,
            "telemetry_view": (
                getattr(problem, "telemetry_view", "canonical")
                if problem is not None
                else "canonical"
            ),
            "observation_condition": observation_condition,
            "observation_token_budget": getattr(
                args,
                "observation_token_budget",
                None,
            )
            if args is not None
            else None,
        }
    )
    pairing = episode_pairing_metadata(
        problem,
        trial_index=getattr(args, "trial_index", 1) if args is not None else 1,
    )
    observation_pipeline_audit = (
        [
            dict(record)
            for record in statebundle_processor.audit_records[observation_audit_start:]
        ]
        if statebundle_processor is not None
        else []
    )
    statebundle_preprocessing_latency_seconds = sum(
        float(record.get("preprocessing_latency_seconds", 0.0))
        for record in observation_pipeline_audit
        if isinstance(record.get("preprocessing_latency_seconds"), int | float)
        and not isinstance(record.get("preprocessing_latency_seconds"), bool)
    )
    result = {
        "problem_id": problem_id,
        "scenario_id": getattr(
            getattr(problem, "scenario", None), "problem_id", problem_id
        ),
        **pairing,
        "task_type": getattr(getattr(problem, "scenario", None), "task_type", None),
        "fault_type": getattr(problem, "fault_type", None),
        "fault_target": getattr(problem, "faulty_component", None),
        "status": status,
        "success": success,
        "score": score,
        "accuracy": accuracy,
        "steps": count_agent_steps(history),
        "agent_turns": agent_turns,
        "max_agent_turns": max_steps,
        "agent_turn_trajectory": agent_turn_trajectory,
        "episode_start_timestamp": episode_started_at.isoformat(),
        "episode_end_timestamp": episode_ended_at.isoformat(),
        "artifact_finalized_timestamp": artifact_finalized_at.isoformat(),
        "elapsed_wall_clock_seconds": runtime_seconds,
        "termination_reason": termination_reason,
        "termination_detail": final_state,
        "runtime_seconds": runtime_seconds,
        "timeout_seconds": timeout_seconds,
        "seed": getattr(problem, "seed", seed),
        "agent": agent_name,
        "agent_type": getattr(agent, "agent_type", "unknown"),
        "provider": getattr(getattr(agent, "settings", None), "provider", None),
        "model": getattr(getattr(agent, "settings", None), "model", None),
        "base_url": sanitize_artifact_url(
            getattr(getattr(agent, "settings", None), "base_url", None)
        ),
        "api_key_env": getattr(
            getattr(agent, "settings", None),
            "api_key_env",
            None,
        ),
        "agent_configuration": agent_configuration,
        "telemetry_view": (
            getattr(problem, "telemetry_view", "canonical")
            if problem is not None
            else "canonical"
        ),
        "observation_condition": observation_condition,
        "statebundle": (
            statebundle_processor.public_metadata()
            if statebundle_processor is not None
            else None
        ),
        "observation_pipeline_audit": observation_pipeline_audit,
        "statebundle_preprocessing_latency_seconds": (
            statebundle_preprocessing_latency_seconds
            if statebundle_processor is not None
            else None
        ),
        "statebundle_preprocessing_call_count": (
            len(observation_pipeline_audit)
            if statebundle_processor is not None
            else None
        ),
        "token_usage": token_usage,
        "token_usage_available": token_usage_available,
        "input_tokens": token_usage.get(
            "input_tokens", token_usage.get("prompt_tokens")
        ),
        "output_tokens": token_usage.get(
            "output_tokens", token_usage.get("completion_tokens")
        ),
        "total_tokens": token_usage.get("total_tokens"),
        "model_call_token_usage": token_usage.get("model_call_token_usage", []),
        "provider_transport_attempts": [
            dict(record)
            for record in getattr(agent, "provider_transport_attempts", [])
            if isinstance(record, dict)
        ],
        "token_count_source": token_usage.get("token_count_source"),
        "token_accounting_warnings": token_usage.get("warnings", []),
        "provider_accounting": provider_accounting,
        "process_metrics": process_metrics,
        "attempted_actions": process_metrics["attempted_actions"],
        "invalid_actions": process_metrics["invalid_actions"],
        "invalid_action_count": process_metrics["invalid_action_count"],
        "invalid_action_rate": process_metrics["invalid_action_rate"],
        "invalid_actions_by_category": process_metrics["invalid_actions_by_category"],
        "invalid_action_details": process_metrics["invalid_action_details"],
        "action_sequence": action_sequence,
        "final_diagnosis": final_diagnosis,
        "final_diagnosis_raw": final_diagnosis_raw,
        "final_state": final_state,
        "action_space": compact_action_space(action_space_payload),
        "initial_agent_visible_observation": initial_agent_rendering,
        "agent_visible_observations": agent_visible_observations[1:],
        "agent_observation_rendering": {
            "schema_versions": [AGENT_SCHEMA_VERSION, AGENT_DELTA_SCHEMA_VERSION],
            "condition": (
                normalized_observation_condition(observation_condition)
                if observation_renderer is not None
                else "canonical"
            ),
            "token_budget": (
                statebundle_processor.config.inference.total_token_budget
                if statebundle_processor is not None
                else None
            ),
            "rendered_observation_count": len(agent_rendered_observations),
            "delivered_observation_count": len(agent_visible_observations),
            "canonical_schema_unchanged": True,
            "audit": (
                [dict(record) for record in observation_renderer.audit_records]
                if observation_renderer is not None
                else []
            ),
        },
        "evaluator_results": evaluation_results,
        "errors": errors,
        "warnings": list(token_usage.get("warnings", [])),
    }
    result.update(mitigation_result_fields(problem, evaluation_results))
    artifact_credentials = configured_credential_values(agent=agent, args=args)
    result = redact_artifact_credentials(result, artifact_credentials)

    # The episode artifact owns a copy of all StateBundle audit/public metadata
    # before inference clears its per-episode memory.
    if statebundle_episode_started:
        statebundle_processor.end_episode()
        statebundle_episode_started = False

    if debug_logs:
        safe_debug = redact_artifact_credentials(debug, artifact_credentials)
        write_debug_log(
            output_dir,
            problem_id,
            safe_debug if isinstance(safe_debug, list) else [],
            result,
        )
    if verbose:
        print(
            f"{problem_id}: status={status} success={success} "
            f"steps={result['steps']} runtime={runtime_seconds:.2f}s"
        )

    return result


def install_inprocess_app(problem: Any) -> None:
    """Install a fresh local simulator adapter for the episode."""
    app = InProcessDataCenterTwinApp()
    problem.app = app
    problem.namespace = app.namespace
    if hasattr(problem, "app_summary"):
        problem.app_summary = app.get_app_summary()


def public_agent_configuration(agent: Any) -> dict[str, Any]:
    """Return auditable non-secret agent settings for episode artifacts."""
    settings = getattr(agent, "settings", None)
    public_settings: dict[str, Any] = {}
    if settings is not None:
        for key, value in vars(settings).items():
            if key != "api_key_env" and is_sensitive_setting_name(key):
                continue
            public_settings[key] = (
                sanitize_artifact_url(value) if key == "base_url" else value
            )
    return {
        "agent_type": getattr(
            agent, "agent_type", type(agent).__name__ if agent else None
        ),
        "settings": public_settings,
    }


SENSITIVE_SETTING_MARKERS = (
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "authorization",
    "credential",
    "password",
    "secret",
)


def is_sensitive_setting_name(name: Any) -> bool:
    normalized = str(name).strip().lower().replace("-", "_")
    return (
        normalized == "token"
        or normalized.endswith("_token")
        or any(marker in normalized for marker in SENSITIVE_SETTING_MARKERS)
    )


def configured_credential_values(*, agent: Any, args: Any) -> tuple[str, ...]:
    """Collect configured secret values solely for artifact redaction."""
    values: set[str] = set()
    settings = getattr(agent, "settings", None)
    if settings is not None:
        for key, value in vars(settings).items():
            if (
                key != "api_key_env"
                and is_sensitive_setting_name(key)
                and isinstance(value, str)
                and value
            ):
                values.add(value)
            if key == "base_url":
                values.update(artifact_url_credential_values(value))
    explicit_key = getattr(args, "api_key", None) if args is not None else None
    if isinstance(explicit_key, str) and explicit_key:
        values.add(explicit_key)
    if args is not None:
        values.update(artifact_url_credential_values(getattr(args, "base_url", None)))
    key_env = getattr(args, "api_key_env", None) if args is not None else None
    if isinstance(key_env, str) and key_env:
        environment_key = os.getenv(key_env)
        if environment_key:
            values.add(environment_key)
    return tuple(sorted(values, key=len, reverse=True))


def artifact_url_credential_values(value: Any) -> set[str]:
    """Extract only credential-bearing URL components for later redaction."""
    if not isinstance(value, str) or not value:
        return set()
    try:
        parsed = urlsplit(value)
    except ValueError:
        return set()
    values = {
        item
        for item in (parsed.username, parsed.password)
        if isinstance(item, str) and item
    }
    values.update(
        item
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if item and is_sensitive_setting_name(key)
    )
    return values


def sanitize_artifact_url(value: Any) -> Any:
    """Remove URL userinfo and secret-valued query parameters for artifacts."""
    if not isinstance(value, str) or not value:
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or not parsed.netloc:
        return value
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        parsed_port = parsed.port
    except ValueError:
        parsed_port = None
    if parsed_port is not None:
        hostname = f"{hostname}:{parsed_port}"
    netloc = (
        f"<redacted>@{hostname}"
        if parsed.username is not None or parsed.password is not None
        else parsed.netloc
    )
    query = urlencode(
        [
            (
                key,
                "<redacted>" if is_sensitive_setting_name(key) else item,
            )
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        ],
        doseq=True,
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))


def redact_artifact_credentials(value: Any, secrets_to_redact: tuple[str, ...]) -> Any:
    """Recursively remove configured credential values from saved metadata."""
    if isinstance(value, dict):
        return {
            key: redact_artifact_credentials(item, secrets_to_redact)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_artifact_credentials(item, secrets_to_redact) for item in value]
    if isinstance(value, tuple):
        return tuple(
            redact_artifact_credentials(item, secrets_to_redact) for item in value
        )
    if not isinstance(value, str):
        return value
    redacted = value
    for secret in secrets_to_redact:
        redacted = redacted.replace(secret, "<redacted>")
    return sanitize_artifact_url(redacted)


def force_deterministic_simulation(problem: Any, seed: int | None) -> None:
    """Force reproducible simulator timing and optionally override scenario seed."""
    if seed is not None:
        problem.seed = seed
    config_override = dict(getattr(problem, "config_override", {}) or {})
    simulation = dict(config_override.get("simulation") or {})
    simulation["auto_advance"] = False
    config_override["simulation"] = simulation
    problem.config_override = config_override


def solution_from_submit(args: list[Any], kwargs: dict[str, Any]) -> Any:
    """Normalize submit(...) arguments into the solution passed to problem.eval."""
    if len(args) == 1 and not kwargs:
        return args[0]
    if args:
        return args
    if kwargs:
        return kwargs
    return None


def evaluate_problem(
    problem: Any,
    solution: Any,
    history: list[SessionItem],
    monotonic_started_at: float,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call the problem evaluator and convert evaluator exceptions to results."""
    duration = time.monotonic() - monotonic_started_at
    try:
        return dict(problem.eval(solution, history, duration))
    except Exception as error:
        errors.append(error_record("evaluation", error))
        semantic_error = (
            getattr(problem, "diagnostic_evaluator", "canonical") == "semantic"
            and getattr(getattr(problem, "scenario", None), "task_type", None)
            != "mitigation"
        )
        return {
            "success": None if semantic_error else False,
            **(
                {"evaluation_status": "evaluator_infrastructure_error"}
                if semantic_error
                else {}
            ),
            "evaluation_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }


def should_evaluate(*, final_state: str, setup_complete: bool, submitted: bool) -> bool:
    """Return whether the episode reached a scoreable benchmark attempt."""
    if not setup_complete:
        return False
    if final_state in {"agent_error", "setup_error", "runner_error", "parse_error"}:
        return False
    if final_state == "invalid_submission":
        return False
    return submitted or final_state in {"max_steps", "timeout"}


def result_token_usage(
    agent_usage: dict[str, Any],
    evaluation_results: dict[str, Any],
    *,
    model_call_token_usage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge agent-reported provider usage with evaluator token estimates."""
    explicit_unknown = is_explicit_unknown_token_usage(agent_usage)
    usage: dict[str, Any] = dict(agent_usage) if isinstance(agent_usage, dict) else {}
    usage.update(
        unknown_token_usage()
        if explicit_unknown
        else {key: usage.get(key, 0) for key in TOKEN_USAGE_KEYS}
    )
    for key in TOKEN_USAGE_KEYS:
        value = agent_usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            usage[key] = value
        elif explicit_unknown:
            usage[key] = None
    usage["input_tokens"] = usage.get("prompt_tokens")
    usage["output_tokens"] = usage.get("completion_tokens")
    usage["input_token_usage_available"] = _is_nonnegative_token_count(
        usage.get("prompt_tokens")
    )
    usage["output_token_usage_available"] = _is_nonnegative_token_count(
        usage.get("completion_tokens")
    )
    if model_call_token_usage is not None:
        usage["model_call_token_usage"] = [
            dict(record)
            for record in model_call_token_usage
            if isinstance(record, dict)
        ]
        usage["model_call_count"] = len(usage["model_call_token_usage"])
    else:
        usage.setdefault("model_call_token_usage", [])
        usage.setdefault("model_call_count", 0)
    usage["evaluator_in_tokens"] = evaluation_results.get("in_tokens")
    usage["evaluator_out_tokens"] = evaluation_results.get("out_tokens")
    usage["token_usage_available"] = is_token_usage_available(usage)
    return usage


def account_unfinished_model_turn_usage(
    *,
    agent: Any,
    agent_usage: Any,
    model_call_token_usage: Any,
    agent_turns: int,
    final_state: str,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    """Mark interrupted controlled-provider turns as unavailable, never zero.

    This is artifact accounting only. It does not alter provider threading,
    cancellation, retries, or timeout behavior.
    """
    usage = dict(agent_usage) if isinstance(agent_usage, dict) else {}
    records = (
        [dict(record) for record in model_call_token_usage if isinstance(record, dict)]
        if isinstance(model_call_token_usage, list)
        else None
    )
    settings = getattr(agent, "settings", None)
    if (
        final_state not in {"timeout", "agent_error"}
        or settings is None
        or records is None
    ):
        return usage, records

    missing_calls = max(0, agent_turns - len(records))
    if missing_calls == 0:
        return usage, records

    provider = getattr(settings, "provider", None)
    model = getattr(settings, "model", None)
    warning = (
        "token usage is unavailable for a model turn that ended before the "
        "provider response could be accounted"
    )
    for _ in range(missing_calls):
        records.append(
            {
                "call_index": len(records) + 1,
                "provider": provider,
                "model": model,
                "call_status": "failed",
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "token_count_source": "unavailable",
                "input_token_count_source": "unavailable",
                "output_token_count_source": "unavailable",
                "estimator": None,
                "provider_usage": None,
                "request_payload_sha256": None,
                "response_payload_sha256": None,
                "request_started_at": None,
                "accounting_completed_at": datetime.now(timezone.utc).isoformat(),
                "completed_after_episode_termination": True,
                "episode_termination_reason": termination_reason_from_state(
                    final_state=final_state,
                    submitted=False,
                    timed_out=final_state == "timeout",
                    runtime_error=final_state == "agent_error",
                    evaluation_results=None,
                ),
                "episode_termination_timestamp": None,
                "trajectory_disposition": ("accounting_only_after_episode_termination"),
                "warnings": [warning],
            }
        )

    raw_source_counts = usage.get("token_count_sources")
    source_counts = (
        dict(raw_source_counts) if isinstance(raw_source_counts, dict) else {}
    )
    source_counts["unavailable"] = (
        int(source_counts.get("unavailable") or 0) + missing_calls
    )
    source_counts.setdefault("provider_native", 0)
    source_counts.setdefault("estimated", 0)
    active_sources = [source for source, count in source_counts.items() if count]
    raw_warnings = usage.get("warnings")
    warnings = (
        [str(item) for item in raw_warnings if isinstance(item, str)]
        if isinstance(raw_warnings, list)
        else []
    )
    warnings.extend([warning] * missing_calls)
    usage.update(
        {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "input_tokens": None,
            "output_tokens": None,
            "token_usage_available": False,
            "token_count_sources": source_counts,
            "token_count_source": (
                active_sources[0]
                if len(active_sources) == 1
                else ("mixed" if active_sources else "unavailable")
            ),
            "model_call_count": len(records),
            "warnings": warnings,
        }
    )
    return usage, records


def quarantine_provider_accounting(
    agent: Any,
    *,
    termination_reason: str,
    termination_timestamp: str,
    errors: list[dict[str, Any]],
) -> None:
    """Close the decision boundary while leaving only accounting work alive."""

    discard = getattr(agent, "discard_late_provider_response", None)
    if callable(discard):
        try:
            discard()
        except Exception as error:  # pragma: no cover - defensive integration guard
            errors.append(error_record("token_accounting", error))
    marker = getattr(agent, "mark_episode_terminated_for_accounting", None)
    if not callable(marker):
        return
    try:
        marker(
            termination_reason,
            termination_timestamp=termination_timestamp,
        )
    except Exception as error:  # pragma: no cover - defensive integration guard
        errors.append(error_record("token_accounting", error))


async def settle_provider_accounting(
    agent: Any,
    *,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Settle late provider usage after evaluator and episode state are frozen."""

    settlement = getattr(agent, "settle_provider_accounting", None)
    if not callable(settlement):
        now = datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": "controlled_provider.accounting_settlement.v1",
            "supported": False,
            "status": "not_supported",
            "tracked_call_count": 0,
            "unfinished_call_count_at_settlement_start": 0,
            "pending_call_count_at_settlement_end": 0,
            "late_completed_call_count": 0,
            "late_failed_call_count": 0,
            "settlement_started_at": now,
            "settlement_completed_at": now,
            "settlement_wait_seconds": 0.0,
            "episode_state_frozen_before_settlement": True,
        }
    try:
        result = await settlement()
        if isinstance(result, dict):
            return dict(result)
        raise TypeError("provider accounting settlement must return an object")
    except Exception as error:  # pragma: no cover - defensive integration guard
        errors.append(error_record("token_accounting", error))
        now = datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": "controlled_provider.accounting_settlement.v1",
            "supported": True,
            "status": "settlement_error",
            "tracked_call_count": None,
            "unfinished_call_count_at_settlement_start": None,
            "pending_call_count_at_settlement_end": None,
            "late_completed_call_count": None,
            "late_failed_call_count": None,
            "settlement_started_at": now,
            "settlement_completed_at": now,
            "settlement_wait_seconds": None,
            "episode_state_frozen_before_settlement": True,
        }


def is_token_usage_available(token_usage: dict[str, Any] | None) -> bool:
    if not isinstance(token_usage, dict):
        return False
    if token_usage.get("token_usage_available") is False:
        return False
    return all(
        _is_nonnegative_token_count(token_usage.get(key)) for key in TOKEN_USAGE_KEYS
    )


def _is_nonnegative_token_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def render_env_response(env_response: Any) -> str:
    """Render one environment response for the next agent turn and trace log."""
    rendered, _ = render_env_response_with_observation(env_response)
    return rendered


def normalized_observation_condition(condition: str) -> str:
    if condition not in {"full-canonical", "statebundle"}:
        raise ValueError(f"unsupported observation condition: {condition}")
    return condition


def latest_complete_canonical_snapshot(problem: Any) -> dict[str, Any] | None:
    snapshot = getattr(problem, "_latest_complete_canonical_snapshot", None)
    return snapshot if isinstance(snapshot, dict) else None


def record_agent_rendered_observation_for_evaluation(
    problem: Any,
    observation: dict[str, Any],
    *,
    renderer: AgentTelemetryRenderer | None,
) -> None:
    """Give task evaluators the same compact evidence delivered to the agent."""

    if renderer is None:
        return
    recorder = getattr(
        problem,
        "_record_agent_rendered_observation_for_evaluation",
        None,
    )
    if callable(recorder):
        recorder(observation)


def render_agent_observation(
    observation: Any,
    *,
    renderer: AgentTelemetryRenderer | None,
    request: AgentObservationRequest,
    canonical_context: dict[str, Any] | None,
    initial: bool,
) -> Any:
    """Sanitize, then compact one canonical/StateBundle agent observation."""

    safe = sanitize_agent_payload(observation)
    if not isinstance(safe, dict) or renderer is None:
        return safe
    if safe.get("schema_version") not in {
        "statebundle.canonical.v1",
        "statebundle.output.v1",
    }:
        return safe
    return renderer.render(
        safe,
        request=request,
        canonical_context=canonical_context,
        initial=initial,
    )


def render_initial_observation_message(observation: Any) -> str:
    """Build the sole initial environment message, separate from the system prompt."""

    return json.dumps(
        {"initial_observation": observation},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def render_env_response_with_observation(
    env_response: Any,
    *,
    renderer: AgentTelemetryRenderer | None = None,
    request: AgentObservationRequest | None = None,
    canonical_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Render an environment response and return any compact telemetry payload."""

    if isinstance(env_response, SubmissionStatus):
        return env_response.name, None
    if isinstance(env_response, str):
        return env_response, None
    if isinstance(env_response, (dict, list, tuple)):
        safe = sanitize_agent_payload(env_response)
        rendered_observation: dict[str, Any] | None = None
        effective_request = request or AgentObservationRequest()
        if isinstance(safe, dict):
            if safe.get("schema_version") in {
                "statebundle.canonical.v1",
                "statebundle.output.v1",
            }:
                projected = render_agent_observation(
                    safe,
                    renderer=renderer,
                    request=effective_request,
                    canonical_context=canonical_context,
                    initial=False,
                )
                safe = projected
                rendered_observation = (
                    projected if isinstance(projected, dict) else None
                )
            elif isinstance(safe.get("observation"), dict) and safe["observation"].get(
                "schema_version"
            ) in {
                "statebundle.canonical.v1",
                "statebundle.output.v1",
            }:
                projected = render_agent_observation(
                    safe["observation"],
                    renderer=renderer,
                    request=effective_request,
                    canonical_context=canonical_context,
                    initial=False,
                )
                safe = dict(safe)
                safe["observation"] = projected
                rendered_observation = (
                    projected if isinstance(projected, dict) else None
                )
        return (
            json.dumps(
                safe,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ),
            rendered_observation,
        )
    return str(env_response), None


def observation_request_from_action(
    api_name: str,
    args: list[Any],
    kwargs: dict[str, Any],
) -> AgentObservationRequest:
    """Extract agent-view controls without changing the canonical request store."""

    if api_name not in {"dc_twin_observe", "dc_twin_action"}:
        return AgentObservationRequest()
    values = dict(kwargs)
    if api_name == "dc_twin_observe":
        if args:
            values.setdefault("log_limit", args[0])
        if len(args) > 1:
            values.setdefault("include_config", args[1])
    channels = values.get("channels")
    if isinstance(channels, str):
        channels = tuple(
            channel.strip() for channel in channels.split(",") if channel.strip()
        )
    lookback = values.get("lookback_seconds")
    # Action responses use the same optional observation controls when the
    # action accepts them; write-action parameters are ignored here.
    return make_agent_observation_request(
        log_limit=values.get("log_limit", 20),
        include_config=values.get("include_config", True),
        channels=channels,
        lookback_seconds=300 if lookback is None else lookback,
        detail=values.get("detail") or "overview",
        metric_names=values.get("metric_names") or (),
        entity_ids=values.get("entity_ids") or (),
        subsystem_ids=values.get("subsystem_ids") or (),
        alert_names=values.get("alert_names") or (),
    )


def make_agent_observation_request(**values: Any) -> AgentObservationRequest:
    """Validate an observation request against the current tool schema."""
    return AgentObservationRequest(**values)


def validate_task_action_arguments(
    problem: Any,
    api_name: str,
    args: list[Any],
    kwargs: dict[str, Any],
) -> None:
    """Classify signature-binding failures before task dispatch.

    Arbitrary ``TypeError`` exceptions raised inside the environment are not
    treated as agent-invalid; only this explicit bind step creates an invalid
    parameter classification.
    """
    if api_name == "submit":
        # ``problem.perform_action`` intentionally has a variadic dispatcher
        # signature. Bind against the concrete task submission action so that
        # missing/extra arguments are attributed to the attempted request.
        method = getattr(getattr(problem, "actions", None), "submit", None)
    else:
        method = getattr(problem, api_name, None)
    if not callable(method):
        return
    try:
        inspect.signature(method).bind(*args, **kwargs)
    except TypeError as error:
        raise ActionParameterValidationError(str(error)) from error


def annotate_action_accounting(record: dict[str, Any]) -> dict[str, Any]:
    """Persist the centralized invalid-attempt classification in the trace."""
    classification = classify_action_attempt(record)
    record["action_accounting"] = {
        "attempted": classification.attempted,
        "invalid": classification.invalid,
        "category": (
            classification.category.value
            if classification.category is not None
            else None
        ),
        "reason": classification.reason,
        "source": classification.source,
    }
    return record


def opaque_agent_task_id(
    problem_id: str,
    *,
    nonce: str | bytes | None = None,
) -> str:
    """Hide semantic problem IDs behind a nonce-bound per-run identifier."""
    nonce_bytes = (
        secrets.token_bytes(32)
        if nonce is None
        else (nonce.encode("utf-8") if isinstance(nonce, str) else nonce)
    )
    digest = hashlib.sha256(
        nonce_bytes + b"\0" + problem_id.encode("utf-8")
    ).hexdigest()[:12]
    return f"dcopslab-episode-{digest}"


def episode_pairing_metadata(problem: Any, *, trial_index: int) -> dict[str, Any]:
    """Return a condition-independent identifier for one hidden initial state."""

    scenario = getattr(problem, "scenario", None)
    specification = {
        "problem_id": getattr(scenario, "problem_id", None),
        "trial_index": trial_index,
        "simulator_seed": getattr(problem, "seed", None),
        "config_override": getattr(problem, "config_override", None),
        "workload": getattr(problem, "workload_config", None),
        "stabilization_ticks": getattr(problem, "stabilization_ticks", None),
        "post_injection_ticks": getattr(problem, "post_injection_ticks", None),
        "fault": {
            "mechanism": getattr(problem, "fault_type", None),
            "target": getattr(problem, "faulty_component", None),
            "severity": getattr(problem, "fault_severity", None),
            "duration_seconds": getattr(problem, "fault_duration_seconds", None),
            "parameters": getattr(problem, "fault_parameters", None),
        },
    }
    encoded = json.dumps(
        specification,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return {
        "pairing_id": f"dc-twin-pair-{digest[:20]}",
        "pairing_specification_sha256": digest,
        "trial_index": trial_index,
    }


def error_record(phase: str, error: Exception) -> dict[str, Any]:
    """Create a structured error payload for result/debug artifacts."""
    return {
        "phase": phase,
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


def status_from_state(
    *,
    final_state: str,
    success: bool,
    submitted: bool,
    timed_out: bool,
    runtime_error: bool,
) -> str:
    """Map internal loop state flags to the externally reported status string."""
    if timed_out:
        return "timeout"
    if final_state == "max_steps":
        return "max_steps"
    if final_state in {"invalid_submission", "parse_error"}:
        return final_state
    if runtime_error:
        return "error"
    if submitted:
        return "success" if success else "failure"
    return final_state


def termination_reason_from_state(
    *,
    final_state: str,
    submitted: bool,
    timed_out: bool,
    runtime_error: bool,
    evaluation_results: dict[str, Any] | None,
) -> str:
    """Map loop state to a stable stop-reason category.

    This deliberately does not encode task correctness.  For example, a
    submitted but incorrect solution still stopped because of a final
    submission.
    """
    if evaluator_infrastructure_error(evaluation_results or {}):
        return TerminationReason.BENCHMARK_ERROR.value
    if final_state == "cancelled":
        return TerminationReason.CANCELLED.value
    if timed_out or final_state == "timeout":
        return TerminationReason.WALL_CLOCK_TIMEOUT.value
    if final_state == "invalid_submission":
        return TerminationReason.INVALID_SUBMISSION.value
    if final_state in {"agent_error", "parse_error"}:
        return TerminationReason.AGENT_ERROR.value
    if final_state in {"setup_error", "runner_error", "environment_error"}:
        return TerminationReason.BENCHMARK_ERROR.value
    if submitted or final_state == "submitted":
        return TerminationReason.FINAL_SUBMISSION.value
    if final_state == "max_steps":
        return TerminationReason.TURN_LIMIT.value
    if final_state == "evaluator_complete":
        return TerminationReason.EVALUATOR_COMPLETE.value
    if isinstance(evaluation_results, dict) and evaluation_results.get(
        "evaluation_error"
    ):
        return TerminationReason.BENCHMARK_ERROR.value
    if runtime_error:
        return TerminationReason.BENCHMARK_ERROR.value
    if isinstance(evaluation_results, dict) and evaluation_results.get("success"):
        return TerminationReason.EVALUATOR_COMPLETE.value
    return TerminationReason.BENCHMARK_ERROR.value


MITIGATION_RESULT_FIELD_NAMES = (
    "mitigation_success",
    "stability_qualified_success",
    "stable_recovery_succeeded",
    "health_trajectory",
    "slo_violation_area",
    "raw_slo_violation_duration",
    "time_to_stable_recovery",
    "penalized_time_to_stable_recovery",
    "recovery_start_time",
    "recovery_verification_time",
    "fault_injection_time",
    "episode_horizon",
    "stability_window_length",
    "slo_constraint_count",
    "slo_integrated_post_fault_time",
    "mitigation_metric_config",
)


def mitigation_result_fields(
    problem: Any,
    evaluation_results: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return additive mitigation fields, or explicit not-applicable values."""
    task_type = getattr(getattr(problem, "scenario", None), "task_type", None)
    if task_type != "mitigation":
        return {field: None for field in MITIGATION_RESULT_FIELD_NAMES}
    results = evaluation_results if isinstance(evaluation_results, dict) else {}
    fields = {field: results.get(field) for field in MITIGATION_RESULT_FIELD_NAMES}
    # Episodes stopped before the task evaluator runs still have a definite
    # negative mitigation/stability outcome, even though trajectory-derived
    # quantities may remain unavailable or be finalized separately.
    fields["mitigation_success"] = results.get("mitigation_success") is True
    fields["stability_qualified_success"] = (
        results.get("stability_qualified_success") is True
    )
    return fields


def count_agent_steps(history: list[SessionItem]) -> int:
    """Count assistant turns in the evaluator trace."""
    return sum(1 for item in history if item.role == "assistant")


def extract_score_accuracy(
    results: dict[str, Any], success: bool | None
) -> tuple[float | None, Any]:
    """Normalize heterogeneous task evaluator fields to score/accuracy columns."""
    if not results or evaluator_infrastructure_error(results):
        return None, None
    if results.get("diagnostic_evaluator") == "semantic":
        return (1.0 if results["success"] else 0.0), results["success"]
    for key in ("Localization Accuracy", "RCA Score", "reasoning_score"):
        value = results.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            normalized = float(value)
            if key == "Localization Accuracy" and normalized > 1.0:
                normalized = normalized / 100.0
            return normalized, value

    for key in ("Detection Accuracy", "Analysis Accuracy", "Mitigation Success"):
        value = results.get(key)
        if isinstance(value, bool):
            return (1.0 if value else 0.0), value
        if isinstance(value, str):
            correct = value.strip().lower() in {"correct", "success", "true"}
            return (1.0 if correct else 0.0), value

    return (1.0 if success else 0.0), success


def write_results(
    output_dir: Path,
    results: list[dict[str, Any]],
    run_config: dict[str, Any] | None = None,
    *,
    runner_name: str = "normal",
) -> None:
    """Write both complete JSON artifacts and a concise Markdown summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        if not isinstance(result.get("process_metrics"), dict):
            result["process_metrics"] = ensure_process_metrics(result)
    aggregate = aggregate_results(results)
    payload = {
        "benchmark": "data_center_twin",
        "runner": runner_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_at_unix": time.time(),
        "run_config": run_config or {},
        "problem_count": len(results),
        "success_count": aggregate["success_count"],
        "aggregate_success_rate": aggregate["success_rate"],
        "aggregate_average_score": aggregate["average_score"],
        "aggregate_average_runtime_seconds": aggregate["average_runtime_seconds"],
        "aggregate_total_token_usage": aggregate["total_token_usage"],
        "aggregate_mitigation_metrics": aggregate["mitigation_metrics"],
        "aggregate_process_metrics": aggregate["process_metrics"],
        "aggregate_process_metrics_by_agent": aggregate["process_metrics_by_agent"],
        "aggregate_process_metrics_by_task_type": aggregate[
            "process_metrics_by_task_type"
        ],
        "aggregate_process_metrics_by_fault_type": aggregate[
            "process_metrics_by_fault_type"
        ],
        "results": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "results.md").write_text(
        render_markdown_summary(results, aggregate, run_config=run_config),
        encoding="utf-8",
    )
    write_results_csv(output_dir / "results.csv", results)


CSV_SCALAR_FIELDS = (
    "problem_id",
    "scenario_id",
    "trial_index",
    "pairing_id",
    "pairing_specification_sha256",
    "task_type",
    "fault_type",
    "fault_target",
    "agent",
    "agent_type",
    "provider",
    "model",
    "telemetry_view",
    "observation_condition",
    "statebundle_preprocessing_latency_seconds",
    "statebundle_preprocessing_call_count",
    "status",
    "success",
    "score",
    "agent_turns",
    "max_agent_turns",
    "steps",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "token_count_source",
    "attempted_actions",
    "invalid_actions",
    "invalid_action_rate",
    "stability_qualified_success",
    "stable_recovery_succeeded",
    "slo_violation_area",
    "raw_slo_violation_duration",
    "time_to_stable_recovery",
    "penalized_time_to_stable_recovery",
    "recovery_start_time",
    "recovery_verification_time",
    "fault_injection_time",
    "episode_horizon",
    "stability_window_length",
    "slo_constraint_count",
    "slo_integrated_post_fault_time",
    "episode_start_timestamp",
    "episode_end_timestamp",
    "artifact_finalized_timestamp",
    "elapsed_wall_clock_seconds",
    "termination_reason",
    "termination_detail",
)
CSV_JSON_FIELDS = (
    "agent_configuration",
    "statebundle",
    "observation_pipeline_audit",
    "invalid_actions_by_category",
    "invalid_action_details",
    "agent_turn_trajectory",
    "model_call_token_usage",
    "token_usage",
    "token_accounting_warnings",
    "provider_accounting",
    "mitigation_metric_config",
    "health_trajectory",
    "action_sequence",
    "tool_call_log",
    "action_space",
    "initial_agent_visible_observation",
    "agent_visible_observations",
    "final_diagnosis",
    "evaluator_results",
    "errors",
    "warnings",
    "blackbox",
)


def write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    """Write an audit-preserving CSV companion to the JSON artifact."""
    fieldnames = [*CSV_SCALAR_FIELDS, *CSV_JSON_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field) for field in CSV_SCALAR_FIELDS}
            row.update(
                {
                    field: json.dumps(
                        result.get(field),
                        sort_keys=True,
                        separators=(",", ":"),
                        default=json_default,
                    )
                    for field in CSV_JSON_FIELDS
                }
            )
            writer.writerow(row)


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute batch-level summary metrics from per-problem records."""
    scores = [
        float(item["score"])
        for item in results
        if isinstance(item.get("score"), int | float)
        and not isinstance(item.get("score"), bool)
    ]
    runtimes = [
        float(item["runtime_seconds"])
        for item in results
        if isinstance(item.get("runtime_seconds"), int | float)
        and not isinstance(item.get("runtime_seconds"), bool)
    ]
    total_token_usage: dict[str, Any] = zero_token_usage()
    all_known_input_tokens = 0
    all_known_output_tokens = 0
    all_known_total_tokens = 0
    known_input_usage_count = 0
    known_output_usage_count = 0
    known_total_usage_count = 0
    known_token_usage_count = 0
    unknown_token_usage_count = 0
    token_source_counts: dict[str, int] = {}
    token_estimators: set[str] = set()
    for item in results:
        raw_usage = item.get("token_usage")
        if raw_usage is not None and not isinstance(raw_usage, dict):
            unknown_token_usage_count += 1
            continue
        usage = raw_usage or {}
        input_value = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_value = usage.get("output_tokens", usage.get("completion_tokens"))
        if _is_nonnegative_token_count(input_value):
            all_known_input_tokens += int(input_value)
            known_input_usage_count += 1
        if _is_nonnegative_token_count(output_value):
            all_known_output_tokens += int(output_value)
            known_output_usage_count += 1
        total_value = usage.get("total_tokens")
        if _is_nonnegative_token_count(total_value):
            all_known_total_tokens += int(total_value)
            known_total_usage_count += 1
        source_counts = usage.get("token_count_sources")
        if isinstance(source_counts, dict):
            for source, count in source_counts.items():
                if (
                    isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= 0
                ):
                    token_source_counts[str(source)] = (
                        token_source_counts.get(str(source), 0) + count
                    )
        elif isinstance(usage.get("token_count_source"), str):
            source = str(usage["token_count_source"])
            token_source_counts[source] = token_source_counts.get(source, 0) + int(
                usage.get("model_call_count") or 0
            )
        estimators = usage.get("estimators")
        if isinstance(estimators, list):
            token_estimators.update(str(estimator) for estimator in estimators)
        if (
            is_explicit_unknown_token_usage(usage)
            or item.get("token_usage_available") is False
        ):
            unknown_token_usage_count += 1
            continue
        known_token_usage_count += 1
        for key in TOKEN_USAGE_KEYS:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                total_token_usage[key] += value
    known_partial_usage = {
        "prompt_tokens": all_known_input_tokens,
        "completion_tokens": all_known_output_tokens,
        "total_tokens": all_known_total_tokens,
        "input_tokens": all_known_input_tokens,
        "output_tokens": all_known_output_tokens,
    }
    if unknown_token_usage_count:
        total_token_usage.update(unknown_token_usage())
        total_token_usage["known_partial_token_usage"] = known_partial_usage
    total_token_usage["known_episode_count"] = known_token_usage_count
    total_token_usage["unknown_episode_count"] = unknown_token_usage_count
    total_token_usage["token_usage_available"] = unknown_token_usage_count == 0
    input_complete = known_input_usage_count == len(results)
    output_complete = known_output_usage_count == len(results)
    if input_complete:
        total_token_usage["prompt_tokens"] = all_known_input_tokens
    if output_complete:
        total_token_usage["completion_tokens"] = all_known_output_tokens
    total_token_usage["input_tokens"] = total_token_usage.get("prompt_tokens")
    total_token_usage["output_tokens"] = total_token_usage.get("completion_tokens")
    total_token_usage["input_token_usage_available"] = input_complete
    total_token_usage["output_token_usage_available"] = output_complete
    total_token_usage["known_input_tokens"] = all_known_input_tokens
    total_token_usage["known_output_tokens"] = all_known_output_tokens
    total_token_usage["known_total_tokens"] = all_known_total_tokens
    total_token_usage["known_input_episode_count"] = known_input_usage_count
    total_token_usage["known_output_episode_count"] = known_output_usage_count
    total_token_usage["known_total_episode_count"] = known_total_usage_count
    total_token_usage["token_count_sources"] = dict(sorted(token_source_counts.items()))
    active_sources = [source for source, count in token_source_counts.items() if count]
    total_token_usage["token_count_source"] = (
        active_sources[0]
        if len(active_sources) == 1
        else ("mixed" if active_sources else None)
    )
    total_token_usage["estimators"] = sorted(token_estimators)
    return {
        **success_statistics(results),
        "average_score": sum(scores) / len(scores) if scores else None,
        "average_runtime_seconds": sum(runtimes) / len(runtimes) if runtimes else None,
        "total_token_usage": total_token_usage,
        "mitigation_metrics": aggregate_mitigation_metrics(results),
        "process_metrics": aggregate_process_metrics(results),
        "process_metrics_by_agent": aggregate_process_metrics_by(
            results,
            lambda item: str(item.get("agent")) if item.get("agent") else None,
        ),
        "process_metrics_by_task_type": aggregate_process_metrics_by(
            results,
            lambda item: process_metric_group_key(item, "task_type"),
        ),
        "process_metrics_by_fault_type": aggregate_process_metrics_by(
            results,
            lambda item: process_metric_group_key(item, "fault_type"),
        ),
    }


def aggregate_mitigation_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate manuscript mitigation metrics without hiding failed recovery."""
    mitigation_results = [
        item
        for item in results
        if process_metric_group_key(item, "task_type") == "mitigation"
    ]
    penalized = [
        float(item["penalized_time_to_stable_recovery"])
        for item in mitigation_results
        if isinstance(item.get("penalized_time_to_stable_recovery"), (int, float))
        and not isinstance(item.get("penalized_time_to_stable_recovery"), bool)
    ]
    successful_recovery_times = [
        float(item["time_to_stable_recovery"])
        for item in mitigation_results
        if item.get("stable_recovery_succeeded") is True
        and isinstance(item.get("time_to_stable_recovery"), (int, float))
        and not isinstance(item.get("time_to_stable_recovery"), bool)
    ]
    slo_areas = [
        float(item["slo_violation_area"])
        for item in mitigation_results
        if isinstance(item.get("slo_violation_area"), (int, float))
        and not isinstance(item.get("slo_violation_area"), bool)
    ]
    violation_durations = [
        float(item["raw_slo_violation_duration"])
        for item in mitigation_results
        if isinstance(item.get("raw_slo_violation_duration"), (int, float))
        and not isinstance(item.get("raw_slo_violation_duration"), bool)
    ]
    complete_penalized_accounting = len(penalized) == len(mitigation_results)
    return {
        "mitigation_episode_count": len(mitigation_results),
        "penalized_recovery_episode_count": len(penalized),
        "penalized_mean_time_to_stable_recovery": (
            sum(penalized) / len(penalized)
            if penalized and complete_penalized_accounting
            else None
        ),
        "successful_recovery_count": len(successful_recovery_times),
        "successful_recovery_median_time": (
            statistics.median(successful_recovery_times)
            if successful_recovery_times
            else None
        ),
        "mean_slo_violation_area": (
            sum(slo_areas) / len(slo_areas)
            if slo_areas and len(slo_areas) == len(mitigation_results)
            else None
        ),
        "mean_raw_slo_violation_duration": (
            sum(violation_durations) / len(violation_durations)
            if violation_durations
            and len(violation_durations) == len(mitigation_results)
            else None
        ),
        "metrics_complete": bool(
            len(penalized) == len(mitigation_results)
            and len(slo_areas) == len(mitigation_results)
            and len(violation_durations) == len(mitigation_results)
        ),
    }


def render_markdown_summary(
    results: list[dict[str, Any]],
    aggregate: dict[str, Any] | None = None,
    *,
    run_config: dict[str, Any] | None = None,
) -> str:
    """Render the human-readable table stored as results.md."""
    aggregate = aggregate or aggregate_results(results)
    lines = [
        "# Data Center Twin Evaluation Results",
        "",
        "## Summary",
        "",
        f"- Problems: {len(results)}",
        f"- Agent: {md_cell((run_config or {}).get('agent', 'unknown'))}",
        f"- Provider: {md_cell((run_config or {}).get('provider', 'unknown'))}",
        f"- Model: {md_cell((run_config or {}).get('model', 'unknown'))}",
        f"- Successes: {aggregate['success_count']}",
        f"- Success rate: {format_optional_float(aggregate['success_rate'])}",
        f"- Average score: {format_optional_float(aggregate['average_score'])}",
        f"- Average runtime: {format_optional_float(aggregate['average_runtime_seconds'])} s",
        f"- Total tokens: {format_token_total(aggregate['total_token_usage'].get('total_tokens'))}",
        f"- Tool calls: {aggregate['process_metrics']['tool_call_count']}",
        f"- Invalid actions: {aggregate['process_metrics']['invalid_action_count']}",
        f"- Invalid action rate: {format_optional_float(aggregate['process_metrics']['invalid_action_rate'])}",
        "- Penalized mean recovery time: "
        f"{format_optional_float(aggregate['mitigation_metrics']['penalized_mean_time_to_stable_recovery'])} s",
        "- Successful-only median recovery time: "
        f"{format_optional_float(aggregate['mitigation_metrics']['successful_recovery_median_time'])} s",
        f"- Redundant action rate: {format_optional_float(aggregate['process_metrics']['redundant_action_rate'])}",
        "",
        "## Episodes",
        "",
        "| Problem | Task | Fault | Status | Termination | Success | Score | Turns | Input | Output | Tools | Invalid | Invalid Rate | SLO Area | Recovery (penalized) | Runtime (s) | Error |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        first_error = item.get("errors", [{}])[0] if item.get("errors") else {}
        raw_token_usage = item.get("token_usage")
        token_usage: dict[str, Any] = (
            raw_token_usage if isinstance(raw_token_usage, dict) else {}
        )
        process_metrics = ensure_process_metrics(item)
        lines.append(
            "| {problem} | {task} | {fault} | {status} | {termination} | {success} | {score} | "
            "{turns} | {input_tokens} | {output_tokens} | {tools} | {invalid} | {invalid_rate} | "
            "{slo_area} | {recovery} | {runtime:.2f} | {error} |".format(
                problem=md_cell(item.get("problem_id")),
                task=md_cell(item.get("task_type")),
                fault=md_cell(item.get("fault_type")),
                status=md_cell(item.get("status")),
                termination=md_cell(item.get("termination_reason")),
                success="yes" if item.get("success") else "no",
                score=format_optional_float(item.get("score")),
                turns=item.get("agent_turns", item.get("steps", 0)),
                input_tokens=format_token_total(
                    token_usage.get("input_tokens", token_usage.get("prompt_tokens"))
                ),
                output_tokens=format_token_total(
                    token_usage.get(
                        "output_tokens", token_usage.get("completion_tokens")
                    )
                ),
                tools=process_metrics.get("tool_call_count", 0),
                invalid=process_metrics.get("invalid_action_count", 0),
                invalid_rate=format_optional_float(
                    process_metrics.get("invalid_action_rate")
                ),
                slo_area=format_optional_float(item.get("slo_violation_area")),
                recovery=format_optional_float(
                    item.get("penalized_time_to_stable_recovery")
                ),
                runtime=float(item.get("runtime_seconds") or 0.0),
                error=md_cell(first_error.get("message", "")),
            )
        )
    lines.extend([""])
    append_process_group_table(
        lines, "Process Metrics by Agent", aggregate["process_metrics_by_agent"]
    )
    append_process_group_table(
        lines, "Process Metrics by Task Type", aggregate["process_metrics_by_task_type"]
    )
    append_process_group_table(
        lines,
        "Process Metrics by Fault Type",
        aggregate["process_metrics_by_fault_type"],
    )
    return "\n".join(lines)


def append_process_group_table(
    lines: list[str], title: str, grouped: dict[str, dict[str, Any]]
) -> None:
    """Append one aggregate process-metrics table to the Markdown report."""
    lines.extend(
        [
            f"## {title}",
            "",
            "| Group | Episodes | Success Rate | Tools | Invalid | Parse Errors | Redundant Rate | Zero-Tool Rate | Avg Submit Step | Observes | Writes | Runtime (s) | Tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if not grouped:
        lines.append(
            "| n/a | 0 | 0.000 | 0 | 0 | 0 | 0.000 | 0.000 | n/a | 0 | 0 | n/a | 0 |"
        )
    for group, metrics in sorted(grouped.items()):
        raw_token_usage = metrics.get("total_token_usage")
        token_usage: dict[str, Any] = (
            raw_token_usage if isinstance(raw_token_usage, dict) else {}
        )
        lines.append(
            "| {group} | {episodes} | {success_rate} | {tools} | {invalid} | {parse_errors} | "
            "{redundant_rate} | {zero_tool_rate} | {submit_step} | {observes} | {writes} | "
            "{runtime} | {tokens} |".format(
                group=md_cell(group),
                episodes=metrics.get("episode_count", 0),
                success_rate=format_optional_float(metrics.get("success_rate")),
                tools=metrics.get("tool_call_count", 0),
                invalid=metrics.get("invalid_action_count", 0),
                parse_errors=metrics.get("parse_error_count", 0),
                redundant_rate=format_optional_float(
                    metrics.get("redundant_action_rate")
                ),
                zero_tool_rate=format_optional_float(
                    metrics.get("zero_tool_diagnosis_rate")
                ),
                submit_step=format_optional_float(
                    metrics.get("average_steps_to_submit")
                ),
                observes=metrics.get("observe_count", 0),
                writes=metrics.get("write_action_count", 0),
                runtime=format_optional_float(metrics.get("average_runtime_seconds")),
                tokens=format_token_total(token_usage.get("total_tokens", 0)),
            )
        )
    lines.append("")


def write_debug_log(
    output_dir: Path,
    problem_id: str,
    debug: list[str],
    result: dict[str, Any],
) -> None:
    """Write optional per-problem debug data outside the concise result files."""
    log_dir = output_dir / "debug"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", problem_id)
    log_payload = {
        "problem_id": problem_id,
        "episode_start_timestamp": result.get("episode_start_timestamp"),
        "episode_end_timestamp": result.get("episode_end_timestamp"),
        "artifact_finalized_timestamp": result.get("artifact_finalized_timestamp"),
        "elapsed_wall_clock_seconds": result.get("elapsed_wall_clock_seconds"),
        "termination_reason": result.get("termination_reason"),
        "termination_detail": result.get("termination_detail"),
        "agent_configuration": result.get("agent_configuration"),
        "observation_condition": result.get("observation_condition"),
        "statebundle": result.get("statebundle"),
        "observation_pipeline_audit": result.get("observation_pipeline_audit", []),
        "debug": debug,
        "warnings": result.get("warnings", []),
        "errors": result.get("errors", []),
        "token_usage": result.get("token_usage", {}),
        "provider_accounting": result.get("provider_accounting", {}),
        "process_metrics": result.get("process_metrics", {}),
        "agent_turn_trajectory": result.get("agent_turn_trajectory", []),
        "action_sequence": result.get("action_sequence", []),
        "evaluator_results": result.get("evaluator_results", {}),
        "health_trajectory": result.get("health_trajectory"),
    }
    (log_dir / f"{safe_name}.log").write_text(
        json.dumps(log_payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def md_cell(value: Any) -> str:
    """Escape and truncate one Markdown table cell."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")[:160]


def format_optional_float(value: Any) -> str:
    """Format numeric fields while rendering missing values as n/a."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.3f}"
    return "n/a"


def format_token_total(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return "n/a"


def json_default(value: Any) -> Any:
    """JSON fallback for pydantic models and session-like objects."""
    if isinstance(value, SessionItem):
        return value.model_dump()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


async def async_main(args: argparse.Namespace) -> int:
    if args.list_agents:
        print("\n".join(list_agent_names()))
        return 0
    if args.trial_index < 1 or args.max_steps < 1 or args.timeout_seconds <= 0:
        raise ValueError("trial index, max steps and timeout must be positive")
    problem_ids = discover_problem_ids(args.problem_filter)
    if args.list_problems:
        print("\n".join(problem_ids))
        return 0
    if not problem_ids:
        raise ValueError("problem filter selected no benchmark tasks")
    if (
        args.expected_problem_count is not None
        and len(problem_ids) != args.expected_problem_count
    ):
        raise ValueError(
            f"selected {len(problem_ids)} tasks; expected {args.expected_problem_count}"
        )
    prepare_observation_condition(args)
    if args.agent == "tool-calling":
        if not args.model:
            raise ValueError("--model or DC_TWIN_LLM_MODEL is required")
        create_agent(args)  # Validate credentials before creating an evaluation.
    elif not args.codex_command:
        raise ValueError("Codex requires --codex-command or DC_BENCH_CODEX_COMMAND")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    run_config = evaluation_run_config(args, problem_ids)
    for problem_id in problem_ids:
        if args.agent == "codex":
            from clients.data_center_twin_baselines.blackbox_harness import (
                BlackboxEpisodeConfig,
                run_blackbox_problem,
            )

            result = await asyncio.to_thread(
                run_blackbox_problem,
                problem_id,
                BlackboxEpisodeConfig(
                    agent_name="codex",
                    command_template=args.codex_command,
                    output_dir=args.output_dir,
                    max_steps=args.max_steps,
                    timeout_seconds=args.timeout_seconds,
                    seed=args.seed,
                    allowed_env_vars=tuple(args.codex_env),
                    sandbox_mode=args.codex_sandbox,
                    docker_image=args.codex_docker_image,
                    observation_condition=args.observation_condition,
                    observation_token_budget=args.observation_token_budget,
                    statebundle_processor=getattr(args, "_statebundle_processor", None),
                    verbose=args.verbose,
                    **evaluator_options(args),
                ),
            )
        else:
            result = await run_problem(
                problem_id,
                args=args,
                output_dir=args.output_dir,
                max_steps=args.max_steps,
                timeout_seconds=args.timeout_seconds,
                seed=args.seed,
                agent_name="tool-calling",
                agent_task_nonce=args.task_id_nonce,
                fail_fast=args.fail_fast,
                debug_logs=args.debug_logs,
                verbose=args.verbose,
            )
        results.append(result)
        write_results(args.output_dir, results, run_config=run_config)
        if args.fail_fast and (
            result.get("errors") or evaluator_infrastructure_error(result)
        ):
            return 1
    print(f"Wrote {len(results)} DC-Bench result(s) to {args.output_dir}")
    return int(
        any(
            item.get("errors") or evaluator_infrastructure_error(item)
            for item in results
        )
    )


def evaluation_run_config(
    args: argparse.Namespace,
    problem_ids: list[str],
) -> dict[str, Any]:
    """Return reproducibility metadata without credential values."""
    manifest_path = Path(SCENARIOS_PATH).expanduser().resolve(strict=True)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    nonce = getattr(args, "task_id_nonce", None)
    return {
        "diagnostic_evaluation": diagnostic_metadata(args),
        "agent": args.agent,
        "provider": getattr(args, "provider", ""),
        "model": getattr(args, "model", ""),
        "base_url": sanitize_artifact_url(getattr(args, "base_url", "")),
        "api_key_env": getattr(args, "api_key_env", ""),
        "telemetry_view": getattr(args, "telemetry_view", "canonical"),
        "observation_condition": getattr(
            args, "observation_condition", "full-canonical"
        ),
        "observation_token_budget": getattr(args, "observation_token_budget", None),
        "statebundle": (
            getattr(args, "_statebundle_processor").public_metadata()
            if getattr(args, "_statebundle_processor", None) is not None
            else None
        ),
        "max_steps": args.max_steps,
        "timeout_seconds": args.timeout_seconds,
        "temperature": getattr(args, "temperature", None),
        "tool_choice": getattr(args, "tool_choice", "required"),
        "max_tokens": getattr(args, "max_tokens", None),
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "thinking_mode": getattr(args, "thinking_mode", None),
        "rate_limit_max_retries": getattr(args, "rate_limit_max_retries", 6),
        "rate_limit_initial_delay_seconds": getattr(
            args,
            "rate_limit_initial_delay_seconds",
            1.0,
        ),
        "rate_limit_max_delay_seconds": getattr(
            args,
            "rate_limit_max_delay_seconds",
            60.0,
        ),
        "seed_override": args.seed,
        "auto_advance": False,
        "expected_problem_count": getattr(
            args,
            "expected_problem_count",
            None,
        ),
        "selected_problem_count": len(problem_ids),
        "selected_problem_ids": list(problem_ids),
        "trial_index": getattr(args, "trial_index", 1),
        "scenario_manifest_path": str(manifest_path),
        "scenario_manifest_schema_version": manifest_payload.get("schema_version"),
        "evaluation_suite_id": manifest_payload.get("suite_id"),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "task_id_nonce_sha256": (
            hashlib.sha256(str(nonce).encode("utf-8")).hexdigest()
            if nonce is not None
            else None
        ),
    }


def prepare_observation_condition(args: argparse.Namespace) -> None:
    condition = normalized_observation_condition(args.observation_condition)
    if condition == "full-canonical":
        args._statebundle_processor = None
        return
    from aiopslab.statebundle.runtime import StateBundleObservationProcessor

    args._statebundle_processor = StateBundleObservationProcessor(
        config_path=args.statebundle_config,
        checkpoint_path=args.statebundle_checkpoint,
        token_budget=args.observation_token_budget,
    )


def statebundle_processor_from_args(args: argparse.Namespace | None):
    if args is None:
        raise ValueError("StateBundle observation condition requires runner arguments")
    processor = getattr(args, "_statebundle_processor", None)
    if processor is None:
        prepare_observation_condition(args)
        processor = getattr(args, "_statebundle_processor", None)
    if processor is None:
        raise RuntimeError("StateBundle observation processor was not initialized")
    return processor


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
