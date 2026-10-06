"""Black-box coding-agent harness for Data Center Twin baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
import inspect
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socketserver
import subprocess
import tempfile
import textwrap
import time
import traceback
from urllib.parse import parse_qsl, urlsplit

from aiopslab.agent_telemetry import (
    AGENT_DELTA_SCHEMA_VERSION,
    AGENT_SCHEMA_VERSION,
    AgentObservationRequest,
    AgentTelemetryRenderer,
)
from aiopslab.orchestrator.problems.data_center_twin.scenarios import (
    fault_diagnosis_contract_lines,
)
from aiopslab.orchestrator.problems.data_center_twin.semantic_evaluation import (
    configure_problem_evaluator,
    diagnostic_metadata,
    evaluator_infrastructure_error,
)
from aiopslab.orchestrator.problems.data_center_twin.visibility import (
    sanitize_agent_payload,
)
from aiopslab.session import SessionItem
from aiopslab.utils.status import InvalidActionError, SubmissionStatus

from .base import compact_action_space, unknown_token_usage, zero_token_usage
from .metrics import (
    InvalidActionCategory,
    classify_action_attempt,
    compute_episode_process_metrics,
)
from .prompts import sanitize_action_space, sanitize_observation


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_TEMPLATE_PATH = Path(__file__).with_name("blackbox_task_template.md")
FORBIDDEN_WORKSPACE_NAMES = {
    "scenarios.json",
    "registry.py",
}
FORBIDDEN_TOOL_TERMS = {
    "scenarios.json",
    "registry.py",
    "aiopslab/orchestrator/problems",
    "orchestrator/problems/data_center_twin",
    "evaluator/debug",
    "evaluator debug",
    "/debug",
    "debug endpoint",
    "success_criteria",
    "ground_truth",
    "hidden benchmark state",
    "hidden state",
}
ALLOWED_INITIAL_WORKSPACE_FILES = {
    "TASK.md",
    "INITIAL_OBSERVATION.json",
    "README_TOOL.md",
    "dc_twin_tool.py",
}
PROCESS_TERMINATION_GRACE_SECONDS = 2.0


class ExternalProcessTimeout(subprocess.TimeoutExpired):
    """Timeout raised only after the episode process group has been cleaned up."""

    process_group_terminated: bool
    process_group_reaped: bool


def _annotate_action_record(record: dict[str, Any]) -> None:
    """Persist the shared invalid-attempt classification in a black-box trace."""
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


def require_only_tool_arguments(
    command: str,
    arguments: dict[str, Any],
    allowed: frozenset[str],
) -> None:
    """Reject wrapper parameters outside the published command schema."""
    unexpected = sorted(str(key) for key in set(arguments) - allowed)
    if unexpected:
        raise ValueError(
            f"{command} received unsupported parameter(s): {', '.join(unexpected)}"
        )


def bridge_response_status(
    response: Any,
    *,
    agent_invalid: bool,
    status_override: int | None = None,
) -> int:
    """Return a bridge HTTP status consistent with the inner task response."""
    if status_override is not None:
        return status_override
    if isinstance(response, dict):
        http_status = response.get("http_status")
        if (
            isinstance(http_status, int)
            and not isinstance(http_status, bool)
            and 400 <= http_status <= 599
        ):
            return http_status
        if response.get("accepted") is False or "error" in response:
            return 400 if agent_invalid else 500
    if agent_invalid:
        return 400
    return 200


@dataclass
class BlackboxEpisodeConfig:
    agent_name: str
    command_template: str
    output_dir: Path
    max_steps: int = 20
    timeout_seconds: float = 120.0
    seed: int | None = None
    deterministic: bool = True
    keep_workspace: bool = False
    verbose: bool = False
    allowed_env_vars: tuple[str, ...] = ()
    allowed_env_prefixes: tuple[str, ...] = ()
    sandbox_mode: str = "docker"
    docker_image: str = "python:3.11-slim"
    telemetry_view: str = "canonical"
    observation_condition: str = "full-canonical"
    observation_token_budget: int | None = 4096
    statebundle_processor: Any | None = None
    token_log: Path | None = None
    token_parser: str = "none"
    bridge_transport: str = "auto"
    task_prompt_via_stdin: bool = False
    environment_overrides: dict[str, str] = field(default_factory=dict)
    protocol_timeout_seconds: float | None = None
    timeout_exit_codes: tuple[int, ...] = ()
    semantic_evaluator_model: str = "gpt-5.6-luna"
    semantic_evaluator_base_url: str = "https://api.openai.com/v1"
    semantic_evaluator_api_key_env: str = "OPENAI_API_KEY"
    semantic_evaluator_reasoning_effort: str = "none"
    semantic_evaluator_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if self.agent_name != "codex":
            raise ValueError("only the Codex agent is supported by this bridge")
        if self.observation_condition not in {"full-canonical", "statebundle"}:
            raise ValueError("unsupported observation condition")


@dataclass
class BridgeState:
    problem: Any
    history: list[SessionItem]
    action_sequence: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    max_steps: int
    action_space_payload: dict[str, Any] | None
    observation_renderer: AgentTelemetryRenderer | None = None
    agent_rendered_observations: list[dict[str, Any]] = field(default_factory=list)
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)
    step_count: int = 0
    submitted: bool = False
    final_state: str = "not_started"
    solution: Any = None
    final_diagnosis: Any = None
    deadline_monotonic: float | None = None
    deadline_exceeded: bool = False
    late_tool_call_count: int = 0
    submitted_at_monotonic: float | None = None
    final_diagnosis_raw: Any = None

    def handle_tool_call(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with self.lock:
            dispatch_monotonic = time.monotonic()
            if (
                self.deadline_monotonic is not None
                and dispatch_monotonic >= self.deadline_monotonic
            ):
                self.deadline_exceeded = True
                self.late_tool_call_count += 1
                if not self.submitted:
                    self.final_state = "timeout"
                return 408, {
                    "ok": False,
                    "error": "benchmark episode wall-clock deadline exceeded",
                }
            if self.submitted or self.final_state in {
                "submitted",
                "invalid_submission",
            }:
                self.step_count += 1
                return self._record_invalid(payload, "episode already submitted")
            if self.step_count >= self.max_steps:
                return self._record_invalid(payload, "max tool steps exceeded")
            self.step_count += 1
            started = time.time()
            command = payload.get("command")
            raw_arguments = payload.get("arguments", {})
            if not isinstance(raw_arguments, dict):
                return self._record_invalid(
                    payload, "tool arguments must be a JSON object"
                )
            arguments = raw_arguments
            if contains_forbidden_tool_term(payload):
                return self._record_invalid(
                    payload,
                    "tool request references hidden or forbidden benchmark paths",
                )
            if command == "invalid":
                message = arguments.get("message")
                return self._record_invalid(
                    payload,
                    str(message) if message else "invalid local tool request",
                )
            if command not in {
                "action_space",
                "observe",
                "action",
                "submit",
                "submit_empty",
            }:
                return self._record_invalid(payload, f"invalid tool command: {command}")

            try:
                api_name, args, kwargs, solution = self._tool_payload_to_api(
                    command, arguments
                )
            except ValueError as error:
                return self._record_invalid(payload, str(error))

            raw_api_call = format_api_call(api_name, args, kwargs)
            action_record = {
                "step": self.step_count,
                "raw": fence_api_call(raw_api_call),
                "api_name": api_name,
                "args": args,
                "kwargs": kwargs,
                "env_response": None,
                "tool_command": command,
                "tool_arguments": arguments,
            }
            self.history.append(
                SessionItem(role="assistant", content=action_record["raw"])
            )
            if api_name == "submit":
                self.final_diagnosis_raw = solution
                self.solution = solution
                self.final_diagnosis = solution

            dispatch_status_override: int | None = None
            try:
                env_response_obj = self.problem.perform_action(
                    api_name, *args, **kwargs
                )
            except InvalidActionError as error:
                env_response_obj = str(error)
                action_record["action_accounting"] = {
                    "attempted": True,
                    "invalid": True,
                    "category": InvalidActionCategory.DISALLOWED_ACTION.value,
                    "reason": str(error),
                    "source": "scenario_action_scope",
                }
            except Exception as error:
                env_response_obj = str(error)
                dispatch_status_override = 500
                action_record["action_accounting"] = {
                    "attempted": True,
                    "invalid": False,
                    "category": None,
                    "reason": str(error),
                    "source": "benchmark_environment_error",
                }
                self.errors.append(
                    {
                        "phase": "environment",
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )

            observation_request = observation_request_from_tool_call(
                api_name,
                kwargs,
            )
            rendered, rendered_observation = render_env_response_with_observation(
                env_response_obj,
                renderer=self.observation_renderer,
                request=observation_request,
                canonical_context=latest_complete_canonical_snapshot(self.problem),
            )
            if isinstance(rendered_observation, dict):
                self.agent_rendered_observations.append(rendered_observation)
                record_agent_rendered_observation_for_evaluation(
                    self.problem,
                    rendered_observation,
                    renderer=self.observation_renderer,
                )
            action_record["env_response"] = rendered
            if env_response_obj == SubmissionStatus.INVALID_SUBMISSION:
                action_record["action_accounting"] = {
                    "attempted": True,
                    "invalid": True,
                    "category": InvalidActionCategory.REJECTED_SUBMISSION.value,
                    "reason": "Black-box agent made an invalid submission.",
                    "source": "submission_status",
                }
            _annotate_action_record(action_record)
            self.action_sequence.append(action_record)
            self.history.append(SessionItem(role="env", content=rendered))

            rendered_payload = (
                json.loads(rendered) if is_json_text(rendered) else rendered
            )
            response_status = bridge_response_status(
                rendered_payload,
                agent_invalid=action_record["action_accounting"]["invalid"],
                status_override=dispatch_status_override,
            )

            if env_response_obj == SubmissionStatus.VALID_SUBMISSION:
                self.submitted = True
                self.final_state = "submitted"
                self.submitted_at_monotonic = dispatch_monotonic
            elif env_response_obj == SubmissionStatus.INVALID_SUBMISSION:
                self.final_state = "invalid_submission"
                self.errors.append(
                    {
                        "phase": "environment",
                        "type": "InvalidSubmission",
                        "message": "Black-box agent made an invalid submission.",
                    }
                )

            tool_record = {
                "step": self.step_count,
                "command": command,
                "arguments": arguments,
                "api_name": api_name,
                "ok": response_status < 400,
                "agent_invalid": action_record["action_accounting"]["invalid"],
                "invalid_category": action_record["action_accounting"]["category"],
                "runtime_seconds": time.time() - started,
            }
            self.tool_call_log.append(tool_record)
            return response_status, {
                "ok": response_status < 400,
                "response": rendered_payload,
            }

    def _tool_payload_to_api(
        self,
        command: str,
        arguments: dict[str, Any],
    ) -> tuple[str, list[Any], dict[str, Any], Any]:
        if command == "action_space":
            require_only_tool_arguments(command, arguments, frozenset())
            return "dc_twin_action_space", [], {}, None
        if command == "observe":
            require_only_tool_arguments(
                command,
                arguments,
                frozenset(
                    {
                        "log_limit",
                        "include_config",
                        "channels",
                        "lookback_seconds",
                        "detail",
                        "metric_names",
                        "entity_ids",
                    }
                    | (
                        {"subsystem_ids"}
                        if "subsystem_ids"
                        in agent_observation_request_parameter_names()
                        else set()
                    )
                    | (
                        {"alert_names"}
                        if "alert_names" in agent_observation_request_parameter_names()
                        else set()
                    )
                ),
            )
            log_limit = arguments.get("log_limit", 20)
            include_config = arguments.get("include_config", True)
            if isinstance(log_limit, bool) or not isinstance(log_limit, int):
                raise ValueError("observe log_limit must be an integer")
            if log_limit < 0:
                raise ValueError("observe log_limit must be non-negative")
            if not isinstance(include_config, bool):
                raise ValueError("observe include_config must be a boolean")
            lookback_seconds = arguments.get("lookback_seconds")
            if lookback_seconds is not None and (
                isinstance(lookback_seconds, bool)
                or not isinstance(lookback_seconds, (int, float))
                or lookback_seconds < 0
            ):
                raise ValueError(
                    "observe lookback_seconds must be a non-negative number"
                )
            channels = validate_observation_string_list(
                arguments.get("channels"),
                field_name="channels",
                allowed={"log", "metric", "alert", "trace", "config"},
            )
            detail = arguments.get("detail", "overview")
            if detail not in {"overview", "raw"}:
                raise ValueError("observe detail must be 'overview' or 'raw'")
            metric_names = validate_observation_string_list(
                arguments.get("metric_names"),
                field_name="metric_names",
            )
            entity_ids = validate_observation_string_list(
                arguments.get("entity_ids"),
                field_name="entity_ids",
            )
            subsystem_ids = validate_observation_string_list(
                arguments.get("subsystem_ids"),
                field_name="subsystem_ids",
            )
            alert_names = validate_observation_string_list(
                arguments.get("alert_names"),
                field_name="alert_names",
            )
            kwargs = {
                "log_limit": log_limit,
                "include_config": include_config,
            }
            optional_values = {
                "lookback_seconds": lookback_seconds,
                "channels": channels,
                "detail": detail,
                "metric_names": metric_names,
                "entity_ids": entity_ids,
                "subsystem_ids": subsystem_ids,
                "alert_names": alert_names,
            }
            for key, value in optional_values.items():
                if key in arguments:
                    kwargs[key] = value
            return "dc_twin_observe", [], kwargs, None
        if command == "action":
            require_only_tool_arguments(command, arguments, frozenset({"payload"}))
            action_payload = arguments.get("payload")
            if not isinstance(action_payload, dict):
                raise ValueError("action requires a JSON object payload")
            action_type = action_payload.get("action_type")
            if not isinstance(action_type, str) or not action_type:
                raise ValueError("action payload requires action_type")
            if action_type not in visible_agent_actions(self.action_space_payload):
                raise ValueError(
                    f"unsupported action_type for this task: {action_type}"
                )
            parameters = action_payload.get("parameters")
            kwargs = {
                key: value
                for key, value in action_payload.items()
                if key not in {"action_type", "parameters"}
            }
            if parameters is not None:
                kwargs["parameters"] = parameters
            return "dc_twin_action", [action_type], kwargs, None
        if command == "submit":
            require_only_tool_arguments(command, arguments, frozenset({"payload"}))
            task_type = getattr(
                getattr(self.problem, "scenario", None), "task_type", None
            )
            if task_type == "mitigation":
                raise ValueError(
                    "mitigation tasks require submit_empty without a payload"
                )
            submission = arguments.get("payload")
            return "submit", [submission], {}, submission
        if command == "submit_empty":
            require_only_tool_arguments(command, arguments, frozenset())
            task_type = getattr(
                getattr(self.problem, "scenario", None), "task_type", None
            )
            if task_type in {"detection", "localization", "analysis"}:
                raise ValueError(f"{task_type} tasks require submit with a payload")
            return "submit", [], {}, None
        raise ValueError(f"invalid tool command: {command}")

    def _record_invalid(
        self, payload: dict[str, Any], message: str
    ) -> tuple[int, dict[str, Any]]:
        record = {
            "step": self.step_count,
            "command": payload.get("command"),
            "arguments": payload.get("arguments"),
            "api_name": None,
            "ok": False,
            "error": message,
        }
        self.tool_call_log.append(record)
        action_record = {
            "step": self.step_count,
            "raw": json.dumps(payload, sort_keys=True, default=str),
            "api_name": "invalid_tool_command",
            "args": [],
            "kwargs": {},
            "env_response": message,
            "invalid_action": True,
        }
        _annotate_action_record(action_record)
        self.action_sequence.append(action_record)
        self.errors.append(
            {
                "phase": "tool",
                "type": "InvalidToolCommand",
                "message": message,
            }
        )
        return 400, {"ok": False, "error": message}

    def record_malformed_request(self, raw_payload: str, message: str) -> None:
        """Retain an authenticated malformed tool submission as one attempt."""
        self.record_invalid_request(
            {"command": None, "raw_payload": raw_payload},
            f"invalid JSON tool request: {message}",
        )

    def record_unsupported_endpoint(self) -> None:
        """Retain an authenticated request to an unsupported bridge endpoint."""
        self.record_invalid_request(
            {"command": None, "endpoint": "unsupported"},
            "unsupported bridge endpoint",
        )

    def record_invalid_request(self, payload: dict[str, Any], message: str) -> None:
        """Record one request rejected before normal tool dispatch."""
        with self.lock:
            self.step_count += 1
            self._record_invalid(payload, message)


class BridgeRequestHandler(BaseHTTPRequestHandler):
    server: Any

    def do_POST(self) -> None:
        if self.path != "/tool":
            if self.headers.get("Authorization") == f"Bearer {self.server.token}":
                self.server.state.record_unsupported_endpoint()
            self._write_json(404, {"ok": False, "error": "unsupported endpoint"})
            return
        if self.headers.get("Authorization") != f"Bearer {self.server.token}":
            self._write_json(403, {"ok": False, "error": "forbidden"})
            return
        raw_payload = ""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_payload = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw_payload)
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
        except Exception as error:
            self.server.state.record_malformed_request(raw_payload, str(error))
            self._write_json(400, {"ok": False, "error": str(error)})
            return
        status, response = self.server.state.handle_tool_call(payload)
        self._write_json(status, response)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class BridgeServer(ThreadingHTTPServer):
    def __init__(self, state: BridgeState, token: str) -> None:
        super().__init__(("127.0.0.1", 0), BridgeRequestHandler)
        self.state = state
        self.token = token


class UnixBridgeServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """HTTP-over-Unix-socket bridge used by containerized black-box agents."""

    daemon_threads = True

    def __init__(self, socket_path: str, state: BridgeState, token: str) -> None:
        super().__init__(socket_path, BridgeRequestHandler)
        self.state = state
        self.token = token


def run_blackbox_problem(
    problem_id: str, config: BlackboxEpisodeConfig
) -> dict[str, Any]:
    from scripts import evaluate_data_center_twin as runner

    episode_started_at = datetime.now(timezone.utc)
    monotonic_started_at = time.monotonic()
    errors: list[dict[str, Any]] = []
    action_sequence: list[dict[str, Any]] = []
    history: list[SessionItem] = []
    registry = None
    problem = None
    setup_complete = False
    app_installed = False
    workspace_dir: Path | None = None
    workspace_parent: Path | None = None
    initial_files: list[str] = []
    process_result: subprocess.CompletedProcess[str] | None = None
    timed_out = False
    runtime_error = False
    process_group_terminated = False
    process_group_reaped = False
    bridge: BridgeServer | UnixBridgeServer | None = None
    bridge_thread: Thread | None = None
    bridge_token: str | None = None
    forwarded_environment: dict[str, str] = {}
    action_space_payload: dict[str, Any] | None = None
    initial_observation: dict[str, Any] | None = None
    initial_agent_rendering: dict[str, Any] | None = None
    agent_rendered_observations: list[dict[str, Any]] = []
    observation_renderer: AgentTelemetryRenderer | None = None
    statebundle_processor = config.statebundle_processor
    observation_audit_start = 0
    statebundle_episode_started = False

    try:
        registry = runner.ProblemRegistry()
        problem = registry.get_problem_instance(problem_id)
        configure_problem_evaluator(problem, config)
        runner.install_inprocess_app(problem)
        app_installed = True
        if config.telemetry_view != "canonical":
            raise ValueError(f"unsupported telemetry_view: {config.telemetry_view}")
        if hasattr(problem, "configure_telemetry_view"):
            problem.configure_telemetry_view(config.telemetry_view)
        if (
            statebundle_processor is not None
            and normalized_blackbox_observation_condition(config.observation_condition)
            == "statebundle"
        ):
            observation_audit_start = len(statebundle_processor.audit_records)
            statebundle_processor.begin_episode()
            statebundle_episode_started = True
        observation_renderer = configure_blackbox_observation_pipeline(
            problem,
            config,
        )
        if config.deterministic:
            runner.force_deterministic_simulation(problem, config.seed)
        elif config.seed is not None:
            problem.seed = config.seed

        if getattr(problem, "START_WORKLOAD_BEFORE_FAULT", False):
            problem.start_workload()
            problem.inject_fault()
        else:
            problem.inject_fault()
            problem.start_workload()
        setup_complete = True

        action_space_payload = problem.dc_twin_action_space()
        if statebundle_episode_started:
            assert statebundle_processor is not None
            statebundle_processor.activate_episode()
        initial_observation = problem.dc_twin_observe(log_limit=20, include_config=True)
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
        agent_task_id = runner.opaque_agent_task_id(problem_id)
        workspace_parent = Path(tempfile.mkdtemp(prefix="dc-twin-blackbox-"))
        workspace_dir = workspace_parent / agent_task_id
        workspace_dir.mkdir()
        assert_workspace_isolated(workspace_dir)

        task_text = render_task_file(
            agent_task_id=agent_task_id,
            task_description=sanitize_task_text(problem.get_task_description()),
            instructions=sanitize_task_text(problem.get_instructions()),
            actions=problem.get_available_actions(),
            action_space_payload=action_space_payload,
            initial_observation=initial_agent_rendering,
            task_type=getattr(getattr(problem, "scenario", None), "task_type", None),
        )
        write_workspace_files(
            workspace_dir,
            task_text,
            initial_observation=initial_agent_rendering,
        )
        initial_files = list_workspace_files(workspace_dir)
        history.append(SessionItem(role="system", content=task_text))
        history.append(
            SessionItem(
                role="env",
                content=json.dumps(
                    sanitize_agent_payload(initial_agent_rendering),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    default=str,
                ),
            )
        )

        state = BridgeState(
            problem=problem,
            history=history,
            action_sequence=action_sequence,
            errors=errors,
            max_steps=config.max_steps,
            action_space_payload=sanitize_action_space(action_space_payload),
            observation_renderer=observation_renderer,
            agent_rendered_observations=agent_rendered_observations,
            deadline_monotonic=(
                monotonic_started_at
                + (
                    config.protocol_timeout_seconds
                    if config.protocol_timeout_seconds is not None
                    else config.timeout_seconds
                )
            ),
        )
        bridge_token = secrets.token_urlsafe(24)
        if config.bridge_transport not in {"auto", "http", "unix"}:
            raise ValueError(
                f"unsupported black-box bridge transport: {config.bridge_transport}"
            )
        use_unix_bridge = config.bridge_transport == "unix" or (
            config.bridge_transport == "auto" and config.sandbox_mode == "docker"
        )
        if use_unix_bridge:
            bridge_socket = workspace_dir / ".dc_twin_bridge.sock"
            bridge = UnixBridgeServer(str(bridge_socket), state, bridge_token)
            bridge_url = f"unix://{bridge_socket}"
        else:
            bridge = BridgeServer(state, bridge_token)
            bridge_url = f"http://127.0.0.1:{bridge.server_port}/tool"
        bridge_thread = Thread(target=bridge.serve_forever, daemon=True)
        bridge_thread.start()

        elapsed_before_agent = time.monotonic() - monotonic_started_at
        process_timeout_seconds = config.timeout_seconds - elapsed_before_agent
        protocol_timeout_seconds = (
            config.protocol_timeout_seconds - elapsed_before_agent
            if config.protocol_timeout_seconds is not None
            else process_timeout_seconds
        )
        if process_timeout_seconds <= 0 or protocol_timeout_seconds <= 0:
            timed_out = True
            state.final_state = "timeout"
            errors.append(
                {
                    "phase": "blackbox",
                    "type": "ControlledTimeout",
                    "message": "Benchmark setup exhausted the protocol wall-clock limit.",
                }
            )
        else:
            command_argv = command_argv_from_template(
                config.command_template,
                task_file=workspace_dir / "TASK.md",
                workspace=workspace_dir,
                tool_file=workspace_dir / "dc_twin_tool.py",
                agent_name=config.agent_name,
                protocol_timeout_seconds=protocol_timeout_seconds,
            )
            forwarded_environment = blackbox_environment(
                os.environ,
                workspace_dir=workspace_dir,
                bridge_url=bridge_url,
                token=bridge_token,
                allowed_env_vars=config.allowed_env_vars,
                allowed_env_prefixes=config.allowed_env_prefixes,
                environment_overrides=config.environment_overrides,
            )
            process_result = run_external_command(
                command_argv,
                cwd=workspace_dir,
                env=forwarded_environment,
                timeout_seconds=process_timeout_seconds,
                sandbox_mode=config.sandbox_mode,
                docker_image=config.docker_image,
                stdin_text=task_text if config.task_prompt_via_stdin else None,
            )
            if state.deadline_exceeded and not state.submitted:
                timed_out = True
                state.final_state = "timeout"
            if (
                process_result.returncode in config.timeout_exit_codes
                and not state.submitted
            ):
                timed_out = True
                state.final_state = "timeout"
                errors.append(
                    {
                        "phase": "blackbox",
                        "type": "ControlledTimeout",
                        "message": (
                            "Black-box agent reached the protocol wall-clock limit "
                            f"and exited with code {process_result.returncode}."
                        ),
                    }
                )
            elif (
                process_result.returncode != 0 and not state.submitted and not timed_out
            ):
                runtime_error = True
                state.final_state = "agent_error"
                errors.append(
                    {
                        "phase": "blackbox",
                        "type": "AgentProcessError",
                        "message": (
                            "Black-box agent process exited with code "
                            f"{process_result.returncode}."
                        ),
                    }
                )
    except subprocess.TimeoutExpired as error:
        timed_out = True
        process_group_terminated = bool(
            getattr(error, "process_group_terminated", False)
        )
        process_group_reaped = bool(getattr(error, "process_group_reaped", False))
        stdout = (
            error.stdout
            if isinstance(error.stdout, str)
            else (error.stdout or b"").decode("utf-8", "replace")
        )
        stderr = (
            error.stderr
            if isinstance(error.stderr, str)
            else (error.stderr or b"").decode("utf-8", "replace")
        )
        process_result = subprocess.CompletedProcess(
            error.cmd, returncode=-9, stdout=stdout, stderr=stderr
        )
        errors.append(
            {"phase": "blackbox", "type": "TimeoutExpired", "message": str(error)}
        )
    except Exception as error:
        runtime_error = True
        errors.append(
            runner.error_record("setup" if not setup_complete else "blackbox", error)
        )
    finally:
        if bridge is not None:
            bridge.shutdown()
            bridge.server_close()
        if bridge_thread is not None:
            bridge_thread.join(timeout=5)

    state = bridge.state if bridge is not None else None
    submitted = bool(state.submitted) if state is not None else False
    final_state = (
        "timeout"
        if timed_out
        else (
            state.final_state
            if state is not None
            else ("runner_error" if setup_complete and runtime_error else "setup_error")
        )
    )
    if final_state == "not_started":
        if submitted:
            final_state = "submitted"
        elif runtime_error:
            final_state = "runner_error"
        elif state is not None and state.step_count >= config.max_steps:
            final_state = "max_steps"
        else:
            final_state = "agent_exit_without_submission"
    solution = state.solution if state is not None else None
    final_diagnosis = state.final_diagnosis if state is not None else None
    evaluation_results: dict[str, Any] = {}
    if problem is not None and runner.should_evaluate(
        final_state=final_state,
        setup_complete=setup_complete,
        submitted=submitted,
    ):
        evaluation_results = runner.evaluate_problem(
            problem, solution, history, monotonic_started_at, errors
        )

    if (
        problem is not None
        and setup_complete
        and hasattr(problem, "finalize_mitigation_metrics")
    ):
        try:
            mitigation_metrics = problem.finalize_mitigation_metrics()
            if isinstance(mitigation_metrics, dict):
                for key, value in mitigation_metrics.items():
                    evaluation_results.setdefault(key, value)
        except Exception as error:
            errors.append(runner.error_record("metric_instrumentation", error))

    if problem is not None and app_installed:
        try:
            problem.recover_fault()
        except Exception as error:
            errors.append(runner.error_record("cleanup", error))
        try:
            problem.app.cleanup()
        except Exception as error:
            errors.append(runner.error_record("cleanup", error))
        try:
            problem.app.delete()
        except Exception as error:
            errors.append(runner.error_record("cleanup", error))

    episode_ended_at = datetime.now(timezone.utc)
    runtime_seconds = time.monotonic() - monotonic_started_at
    evaluation_success_before_timeout_override = bool(evaluation_results.get("success"))
    # A deadline is an episode-level unsuccessful outcome even if partial
    # evaluator state happened to satisfy a criterion immediately beforehand.
    success = (
        False
        if final_state == "timeout"
        else evaluation_success_before_timeout_override
    )
    evaluator_error = evaluator_infrastructure_error(evaluation_results)
    if evaluator_error:
        success = None
    status = runner.status_from_state(
        final_state=final_state,
        success=success,
        submitted=submitted,
        timed_out=timed_out,
        runtime_error=runtime_error,
    )
    if evaluator_error:
        status = "evaluator_infrastructure_error"
    score, accuracy = runner.extract_score_accuracy(evaluation_results, success)
    if final_state == "agent_exit_without_submission":
        termination_reason = "agent_exit_without_submission"
    else:
        termination_reason = runner.termination_reason_from_state(
            final_state=final_state,
            submitted=submitted,
            timed_out=timed_out,
            runtime_error=runtime_error,
            evaluation_results=evaluation_results,
        )
    workspace_final_files = (
        list_workspace_files(workspace_dir)
        if workspace_dir and workspace_dir.exists()
        else []
    )
    workspace_leaks = forbidden_workspace_files(workspace_final_files)
    if workspace_leaks:
        errors.append(
            {
                "phase": "workspace",
                "type": "ForbiddenWorkspaceFile",
                "message": f"black-box workspace contains forbidden file(s): {workspace_leaks}",
            }
        )
    token_usage = blackbox_token_usage(config, evaluation_results)
    token_usage_available = runner.is_token_usage_available(token_usage)
    process_metrics = compute_episode_process_metrics(
        action_sequence,
        errors,
        runtime_seconds=runtime_seconds,
        token_usage=token_usage,
    )
    result = {
        "problem_id": problem_id,
        "scenario_id": getattr(
            getattr(problem, "scenario", None), "problem_id", problem_id
        ),
        "task_type": getattr(getattr(problem, "scenario", None), "task_type", None),
        "fault_type": getattr(problem, "fault_type", None),
        "fault_target": getattr(problem, "faulty_component", None),
        "status": status,
        "success": success,
        "evaluation_success_before_timeout_override": (
            evaluation_success_before_timeout_override
            if final_state == "timeout"
            else None
        ),
        "score": score,
        "accuracy": accuracy,
        "steps": len(action_sequence),
        "agent_turns": blackbox_agent_turn_count(
            config.agent_name,
            action_sequence,
            token_usage,
        ),
        "max_agent_turns": config.max_steps,
        "agent_turn_trajectory": None,
        "episode_start_timestamp": episode_started_at.isoformat(),
        "episode_end_timestamp": episode_ended_at.isoformat(),
        "elapsed_wall_clock_seconds": runtime_seconds,
        "protocol_deadline_enforced_at_bridge": True,
        "late_tool_call_count": (
            state.late_tool_call_count if state is not None else 0
        ),
        "submission_elapsed_seconds": (
            state.submitted_at_monotonic - monotonic_started_at
            if state is not None and state.submitted_at_monotonic is not None
            else None
        ),
        "termination_reason": termination_reason,
        "termination_detail": final_state,
        "runtime_seconds": runtime_seconds,
        "timeout_seconds": (
            config.protocol_timeout_seconds
            if config.protocol_timeout_seconds is not None
            else config.timeout_seconds
        ),
        "seed": getattr(problem, "seed", config.seed),
        "agent": config.agent_name,
        "agent_type": "codex",
        "provider": None,
        "model": None,
        "agent_configuration": {
            "agent_type": "codex",
            "max_steps": config.max_steps,
            "timeout_seconds": (
                config.protocol_timeout_seconds
                if config.protocol_timeout_seconds is not None
                else config.timeout_seconds
            ),
            "deterministic": config.deterministic,
            "telemetry_view": config.telemetry_view,
            "observation_condition": config.observation_condition,
            "observation_token_budget": config.observation_token_budget,
            "sandbox_mode": config.sandbox_mode,
            "token_parser": config.token_parser,
            "bridge_transport": config.bridge_transport,
            "task_prompt_via_stdin": config.task_prompt_via_stdin,
            "process_timeout_seconds": config.timeout_seconds,
            "timeout_exit_codes": list(config.timeout_exit_codes),
            "diagnostic_evaluation": diagnostic_metadata(config),
        },
        "telemetry_view": config.telemetry_view,
        "observation_condition": config.observation_condition,
        "statebundle": (
            statebundle_processor.public_metadata()
            if statebundle_processor is not None
            else None
        ),
        "observation_pipeline_audit": (
            [
                dict(record)
                for record in statebundle_processor.audit_records[
                    observation_audit_start:
                ]
            ]
            if statebundle_processor is not None
            else []
        ),
        "token_usage": token_usage,
        "token_usage_available": token_usage_available,
        "input_tokens": token_usage.get(
            "input_tokens", token_usage.get("prompt_tokens")
        ),
        "output_tokens": token_usage.get(
            "output_tokens", token_usage.get("completion_tokens")
        ),
        "model_call_token_usage": token_usage.get("model_call_token_usage", []),
        "token_count_source": token_usage.get("token_count_source"),
        "token_accounting_warnings": token_usage.get("warnings", []),
        "process_metrics": process_metrics,
        "attempted_actions": process_metrics["attempted_actions"],
        "invalid_actions": process_metrics["invalid_actions"],
        "invalid_action_count": process_metrics["invalid_action_count"],
        "invalid_action_rate": process_metrics["invalid_action_rate"],
        "invalid_actions_by_category": process_metrics["invalid_actions_by_category"],
        "invalid_action_details": process_metrics["invalid_action_details"],
        "action_sequence": action_sequence,
        "tool_call_log": state.tool_call_log if state is not None else [],
        "final_diagnosis": final_diagnosis,
        "final_diagnosis_raw": state.final_diagnosis_raw if state is not None else None,
        "final_state": final_state,
        "action_space": compact_action_space(
            state.action_space_payload if state is not None else None
        ),
        "initial_agent_visible_observation": initial_agent_rendering,
        "agent_visible_observations": agent_rendered_observations[1:],
        "agent_observation_rendering": {
            "schema_versions": [AGENT_SCHEMA_VERSION, AGENT_DELTA_SCHEMA_VERSION],
            "condition": (
                normalized_blackbox_observation_condition(config.observation_condition)
                if observation_renderer is not None
                else "canonical"
            ),
            "token_budget": (
                statebundle_processor.config.inference.total_token_budget
                if statebundle_processor is not None
                else None
            ),
            "rendered_observation_count": len(agent_rendered_observations),
            "canonical_schema_unchanged": True,
            "audit": (
                [dict(record) for record in observation_renderer.audit_records]
                if observation_renderer is not None
                else []
            ),
        },
        "evaluator_results": evaluation_results,
        "errors": errors,
        "warnings": [
            *token_usage.get("warnings", []),
            *(
                [sandbox_warning(config.sandbox_mode)]
                if sandbox_warning(config.sandbox_mode)
                else []
            ),
        ],
        "blackbox": {
            "agent_name": config.agent_name,
            "command_template": config.command_template,
            "command_argv": list(process_result.args)
            if process_result is not None
            else None,
            "exit_code": process_result.returncode
            if process_result is not None
            else None,
            "timed_out": timed_out,
            "process_group_terminated": process_group_terminated,
            "process_group_reaped": process_group_reaped,
            "stdout": process_result.stdout if process_result is not None else "",
            "stderr": process_result.stderr if process_result is not None else "",
            "stdout_bytes": len((process_result.stdout or "").encode("utf-8"))
            if process_result
            else 0,
            "stderr_bytes": len((process_result.stderr or "").encode("utf-8"))
            if process_result
            else 0,
            "workspace": str(workspace_dir) if workspace_dir is not None else None,
            "workspace_initial_files": initial_files,
            "workspace_final_files": workspace_final_files,
            "workspace_retained": config.keep_workspace,
            "sandbox_mode": config.sandbox_mode,
            "sandbox_isolation_guarantee": sandbox_isolation_guarantee(
                config.sandbox_mode
            ),
            "sandbox_warning": sandbox_warning(config.sandbox_mode),
            "token_usage_available": token_usage_available,
            "token_parser": config.token_parser,
            "token_log": str(config.token_log)
            if config.token_log is not None
            else None,
        },
    }
    result.update(runner.mitigation_result_fields(problem, evaluation_results))
    if result["agent_turns"] is None:
        result["warnings"].append(
            "agent turn count is unavailable because the external black-box "
            "process did not provide per-model-call records"
        )
    result = runner.redact_artifact_credentials(
        result,
        blackbox_artifact_credential_values(
            config,
            forwarded_environment,
            bridge_token,
        ),
    )
    # Keep the copied audit/public metadata in the result, then clear only the
    # inference backend's per-episode memory before the processor is reused.
    if statebundle_episode_started:
        assert statebundle_processor is not None
        statebundle_processor.end_episode()
        statebundle_episode_started = False
    if workspace_parent is not None and not config.keep_workspace:
        shutil.rmtree(workspace_parent, ignore_errors=True)
    if config.verbose:
        print(
            f"{problem_id}: status={status} success={success} "
            f"steps={result['steps']} runtime={runtime_seconds:.2f}s"
        )
    return result


def normalized_blackbox_observation_condition(condition: str) -> str:
    if condition not in {"full-canonical", "statebundle"}:
        raise ValueError(f"unsupported observation condition: {condition}")
    return condition


def configure_blackbox_observation_pipeline(
    problem: Any,
    config: BlackboxEpisodeConfig,
) -> AgentTelemetryRenderer | None:
    """Configure only the agent boundary; canonical telemetry stays untouched."""

    condition = normalized_blackbox_observation_condition(config.observation_condition)
    if condition == "statebundle":
        processor = config.statebundle_processor
        if processor is None:
            raise ValueError("StateBundle condition requires a trained processor")
        if not hasattr(problem, "configure_observation_transform"):
            raise RuntimeError(
                "selected benchmark problem does not support observation transforms"
            )
        problem.configure_observation_transform(
            processor.transform,
            condition="statebundle",
        )
    elif config.statebundle_processor is not None:
        raise ValueError(
            "StateBundle processor may only be used with the StateBundle condition"
        )

    renderer = AgentTelemetryRenderer(condition=condition)
    if hasattr(problem, "_activate_agent_rendering_boundary"):
        problem._activate_agent_rendering_boundary()
    return renderer


def latest_complete_canonical_snapshot(problem: Any) -> dict[str, Any] | None:
    """Return the trusted full snapshot retained by the task for StateBundle."""

    snapshot = getattr(problem, "_latest_complete_canonical_snapshot", None)
    return snapshot if isinstance(snapshot, dict) else None


def record_agent_rendered_observation_for_evaluation(
    problem: Any,
    observation: dict[str, Any],
    *,
    renderer: AgentTelemetryRenderer | None,
) -> None:
    """Expose exactly the rendered evidence to task evaluators."""

    if renderer is None:
        return
    recorder = getattr(
        problem,
        "_record_agent_rendered_observation_for_evaluation",
        None,
    )
    if callable(recorder):
        recorder(observation)


def validate_observation_string_list(
    value: Any,
    *,
    field_name: str,
    allowed: set[str] | None = None,
) -> list[str] | None:
    """Validate bridge list controls before invoking the benchmark task."""

    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple, set)) or not all(
        isinstance(item, str) and item.strip() for item in values
    ):
        raise ValueError(f"observe {field_name} must contain non-empty strings")
    normalized = [item.strip() for item in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"observe {field_name} must not contain duplicates")
    if allowed is not None:
        unsupported = sorted(set(normalized) - allowed)
        if unsupported:
            raise ValueError(
                f"observe {field_name} contains unsupported values: {unsupported}"
            )
    return normalized


def observation_request_from_tool_call(
    api_name: str,
    kwargs: dict[str, Any],
) -> AgentObservationRequest:
    """Extract renderer-only view controls from an agent observation call."""

    if api_name != "dc_twin_observe":
        return AgentObservationRequest()
    channels = kwargs.get("channels")
    if isinstance(channels, str):
        channels = (channels,)
    lookback = kwargs.get("lookback_seconds")
    return make_agent_observation_request(
        log_limit=kwargs.get("log_limit", 20),
        include_config=kwargs.get("include_config", True),
        channels=channels,
        lookback_seconds=300 if lookback is None else lookback,
        detail=kwargs.get("detail", "overview"),
        metric_names=kwargs.get("metric_names") or (),
        entity_ids=kwargs.get("entity_ids") or (),
        subsystem_ids=kwargs.get("subsystem_ids") or (),
        alert_names=kwargs.get("alert_names") or (),
    )


def agent_observation_request_parameter_names() -> set[str]:
    """Return public request controls supported by the installed renderer."""

    try:
        return set(inspect.signature(AgentObservationRequest).parameters)
    except (TypeError, ValueError):
        return set()


def make_agent_observation_request(**values: Any) -> AgentObservationRequest:
    """Validate an observation request against the current tool schema."""
    return AgentObservationRequest(**values)


def render_agent_observation(
    observation: Any,
    *,
    renderer: AgentTelemetryRenderer | None,
    request: AgentObservationRequest,
    canonical_context: dict[str, Any] | None,
    initial: bool,
) -> Any:
    """Sanitize then project canonical telemetry at the black-box boundary."""

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


def render_env_response_with_observation(
    env_response: Any,
    *,
    renderer: AgentTelemetryRenderer | None = None,
    request: AgentObservationRequest | None = None,
    canonical_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Render one bridge response and return its projected telemetry, if any."""

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
            elif isinstance(safe.get("observation"), dict):
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


def blackbox_agent_turn_count(
    agent_name: str,
    action_sequence: list[dict[str, Any]],
    token_usage: dict[str, Any],
) -> int | None:
    """Return an audited turn count, or ``None`` when the bridge cannot know it."""
    model_call_count = token_usage.get("model_call_count")
    if (
        isinstance(model_call_count, int)
        and not isinstance(model_call_count, bool)
        and model_call_count >= 0
    ):
        return model_call_count
    return None


def blackbox_artifact_credential_values(
    config: BlackboxEpisodeConfig,
    forwarded_environment: dict[str, str],
    bridge_token: str | None,
) -> tuple[str, ...]:
    """Collect black-box credentials solely to redact persisted artifacts."""
    from scripts import evaluate_data_center_twin as runner

    values: set[str] = set()
    if bridge_token:
        values.add(bridge_token)
    for name, value in forwarded_environment.items():
        if value and (
            name == "DC_TWIN_TOOL_TOKEN" or runner.is_sensitive_setting_name(name)
        ):
            values.add(value)

    try:
        tokens = shlex.split(config.command_template)
    except ValueError:
        tokens = config.command_template.split()
    previous_was_sensitive_flag = False
    for token in tokens:
        if previous_was_sensitive_flag and token:
            values.add(token)
            previous_was_sensitive_flag = False
            continue
        flag, separator, inline_value = token.partition("=")
        normalized_flag = flag.lstrip("-").replace("-", "_")
        if runner.is_sensitive_setting_name(normalized_flag):
            if separator and inline_value:
                values.add(inline_value)
            elif not separator:
                previous_was_sensitive_flag = True
        try:
            parsed = urlsplit(token)
        except ValueError:
            continue
        if not parsed.scheme or not parsed.netloc:
            continue
        if parsed.username:
            values.add(parsed.username)
        if parsed.password:
            values.add(parsed.password)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True):
            if item and runner.is_sensitive_setting_name(key):
                values.add(item)
    return tuple(sorted(values, key=len, reverse=True))


def blackbox_token_usage(
    config: BlackboxEpisodeConfig,
    evaluation_results: dict[str, Any],
) -> dict[str, Any]:
    from scripts import evaluate_data_center_twin as runner

    parsed_usage = parse_token_usage_log(config.token_log, config.token_parser)
    if parsed_usage is not None:
        usage = runner.result_token_usage(parsed_usage, evaluation_results)
        usage.setdefault("token_count_source", "provider_native")
        usage.setdefault("warnings", [])
        return usage
    usage: dict[str, Any] = unknown_token_usage()
    usage["token_usage_available"] = False
    usage["input_tokens"] = None
    usage["output_tokens"] = None
    usage["token_count_source"] = "unavailable"
    usage["model_call_token_usage"] = []
    usage["model_call_count"] = None
    usage["warnings"] = [
        "black-box provider usage was unavailable and cannot be estimated "
        "without the external model request/response payloads"
    ]
    usage["evaluator_in_tokens"] = evaluation_results.get("in_tokens")
    usage["evaluator_out_tokens"] = evaluation_results.get("out_tokens")
    return usage


def parse_token_usage_log(path: Path | None, parser_name: str) -> dict[str, Any] | None:
    if parser_name == "none" or path is None:
        return None
    if parser_name != "auto":
        raise ValueError(f"unsupported black-box token parser: {parser_name}")
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_token_usage_json_text(text) or parse_token_usage_regex(text)
    return parsed if parsed is not None and any(parsed.values()) else parsed


def parse_token_usage_json_text(text: str) -> dict[str, Any] | None:
    candidates: list[Any]
    try:
        candidates = [json.loads(text)]
    except json.JSONDecodeError:
        candidates = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    call_records: list[dict[str, Any]] = []
    for candidate in candidates:
        call_records.extend(find_all_token_usage_records(candidate))
    if not call_records:
        return None
    return summarize_token_call_records(call_records)


def find_all_token_usage_records(value: Any) -> list[dict[str, Any]]:
    """Find non-overlapping call records while preferring explicit call logs."""
    if isinstance(value, dict):
        wrapped_records = value.get("model_call_token_usage")
        if isinstance(wrapped_records, list):
            records = [
                record
                for item in wrapped_records
                if isinstance(item, dict)
                for record in [token_call_record_from_mapping(item)]
                if record is not None
            ]
            if records:
                return records

        direct = token_call_record_from_mapping(value)
        if direct is not None:
            return [direct]
        records: list[dict[str, Any]] = []
        for item in value.values():
            records.extend(find_all_token_usage_records(item))
        return records
    if isinstance(value, list):
        records = []
        for item in value:
            records.extend(find_all_token_usage_records(item))
        return records
    return []


def token_call_record_from_mapping(value: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one call while retaining its raw provider usage metadata."""
    is_explicit_call_record = any(
        key in value
        for key in (
            "call_index",
            "token_count_source",
            "estimator",
            "provider_usage",
        )
    )
    normalized = token_usage_from_mapping(value)
    provider_usage = value.get("provider_usage")
    if normalized is None and isinstance(provider_usage, dict):
        normalized = token_usage_from_mapping(provider_usage)
    if normalized is None:
        if not is_explicit_call_record:
            return None
        source = value.get("token_count_source")
        if source != "unavailable":
            return None
        record = dict(value)
        record["input_tokens"] = None
        record["output_tokens"] = None
        record["total_tokens"] = None
        record["token_count_source"] = "unavailable"
        record.setdefault("estimator", None)
        record.setdefault("provider_usage", None)
        record.setdefault("warnings", ["provider call token usage unavailable"])
        return record

    record = dict(value) if is_explicit_call_record else {}
    record["input_tokens"] = normalized["prompt_tokens"]
    record["output_tokens"] = normalized["completion_tokens"]
    record["total_tokens"] = normalized["total_tokens"]
    record.setdefault("token_count_source", "provider_native")
    record.setdefault("estimator", None)
    if is_explicit_call_record and "provider_usage" in value:
        record["provider_usage"] = (
            dict(provider_usage) if isinstance(provider_usage, dict) else provider_usage
        )
    else:
        record["provider_usage"] = dict(value)
    record.setdefault("warnings", [])
    return record


def summarize_token_call_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate normalized call records without losing their audit payloads."""
    normalized_records: list[dict[str, Any]] = []
    totals = zero_token_usage()
    source_counts = {
        "provider_native": 0,
        "estimated": 0,
        "unavailable": 0,
    }
    estimators: set[str] = set()
    warnings: list[str] = []
    unavailable = False

    for index, raw_record in enumerate(records, start=1):
        record = dict(raw_record)
        record.setdefault("call_index", index)
        source = str(record.get("token_count_source") or "provider_native")
        if source not in source_counts:
            source = "unavailable"
            record["token_count_source"] = source
        source_counts[source] += 1
        input_tokens = record.get("input_tokens")
        output_tokens = record.get("output_tokens")
        if not valid_token_count(input_tokens) or not valid_token_count(output_tokens):
            unavailable = True
            if source != "unavailable":
                source_counts[source] -= 1
                source_counts["unavailable"] += 1
                record["token_count_source"] = "unavailable"
        else:
            assert isinstance(input_tokens, int) and not isinstance(input_tokens, bool)
            assert isinstance(output_tokens, int) and not isinstance(
                output_tokens, bool
            )
            input_count = input_tokens
            output_count = output_tokens
            totals["prompt_tokens"] += input_count
            totals["completion_tokens"] += output_count
            totals["total_tokens"] += input_count + output_count
            record["total_tokens"] = input_count + output_count
        estimator = record.get("estimator")
        if isinstance(estimator, str) and estimator:
            estimators.add(estimator)
        record_warnings = record.get("warnings")
        if isinstance(record_warnings, list):
            warnings.extend(str(warning) for warning in record_warnings)
        normalized_records.append(record)

    if unavailable:
        totals = unknown_token_usage()
    active_sources = [source for source, count in source_counts.items() if count]
    return {
        **totals,
        "input_tokens": totals["prompt_tokens"],
        "output_tokens": totals["completion_tokens"],
        "token_usage_available": not unavailable,
        "token_count_source": (
            active_sources[0]
            if len(active_sources) == 1
            else ("mixed" if active_sources else None)
        ),
        "token_count_sources": source_counts,
        "estimators": sorted(estimators),
        "model_call_count": len(normalized_records),
        "model_call_token_usage": normalized_records,
        "warnings": warnings,
    }


def find_all_token_usage(value: Any) -> list[dict[str, int]]:
    """Find non-overlapping provider usage records in a JSON value."""
    return [
        {
            "prompt_tokens": record["input_tokens"],
            "completion_tokens": record["output_tokens"],
            "total_tokens": record["total_tokens"],
        }
        for record in find_all_token_usage_records(value)
        if valid_token_count(record.get("input_tokens"))
        and valid_token_count(record.get("output_tokens"))
    ]


def find_token_usage(value: Any) -> dict[str, int] | None:
    if isinstance(value, dict):
        direct = token_usage_from_mapping(value)
        if direct is not None:
            return direct
        for item in value.values():
            found = find_token_usage(item)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = find_token_usage(item)
            if found is not None:
                return found
    return None


def token_usage_from_mapping(value: dict[str, Any]) -> dict[str, int] | None:
    prompt = value.get("prompt_tokens", value.get("input_tokens"))
    completion = value.get("completion_tokens", value.get("output_tokens"))
    total = value.get("total_tokens")
    prompt = prompt if valid_token_count(prompt) else None
    completion = completion if valid_token_count(completion) else None
    total = total if valid_token_count(total) else None
    if prompt is None and completion is None:
        return None
    if (
        prompt is None
        and total is not None
        and completion is not None
        and total >= completion
    ):
        prompt = total - completion
    if (
        completion is None
        and total is not None
        and prompt is not None
        and total >= prompt
    ):
        completion = total - prompt
    if prompt is None or completion is None:
        return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def valid_token_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def parse_token_usage_regex(text: str) -> dict[str, Any] | None:

    aliases = {
        "prompt_tokens": "prompt_tokens",
        "input_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "output_tokens": "completion_tokens",
        "total_tokens": "total_tokens",
    }
    pattern = re.compile(
        r"\b(prompt_tokens|input_tokens|completion_tokens|output_tokens|total_tokens)"
        r"\b[^0-9\r\n]*?(\d+)",
        flags=re.IGNORECASE,
    )
    call_fields: list[dict[str, int]] = []
    current: dict[str, int] = {}
    for match in pattern.finditer(text):
        canonical = aliases[match.group(1).lower()]
        if canonical in current:
            call_fields.append(current)
            current = {}
        current[canonical] = int(match.group(2))
    if current:
        call_fields.append(current)

    records: list[dict[str, Any]] = []
    for fields in call_fields:
        usage = token_usage_from_mapping(fields)
        if usage is None:
            continue
        records.append(
            {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "token_count_source": "provider_native",
                "estimator": None,
                "provider_usage": dict(fields),
                "warnings": [],
            }
        )
    return summarize_token_call_records(records) if records else None


def re_search_int_field(text: str, name: str) -> int | None:

    match = re.search(rf"{re.escape(name)}\D+(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def render_task_file(
    *,
    agent_task_id: str,
    task_description: str,
    instructions: str,
    actions: dict[str, Any],
    action_space_payload: dict[str, Any] | None,
    initial_observation: dict[str, Any] | None,
    task_type: str | None = None,
) -> str:
    # Keep TASK.md static. The sanitized observation is written separately to
    # INITIAL_OBSERVATION.json by ``write_workspace_files``.
    del initial_observation
    context = {
        "agent_task_id": agent_task_id,
        "task_description": task_description,
        "instructions": blackbox_task_instructions(task_type),
        "in_process_task_api_reference": summarize_task_api_reference(actions),
        "agent_action_space": sanitize_action_space(action_space_payload),
    }
    template = TASK_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("{{AGENT_TASK_ID}}", agent_task_id).replace(
        "{{CONTEXT_JSON}}", json.dumps(context, indent=2, sort_keys=True, default=str)
    )


def summarize_task_api_reference(actions: dict[str, Any]) -> dict[str, str]:
    """Expose API names as reference only, without normal-runner response rules."""
    allowed = {"dc_twin_action_space", "dc_twin_observe", "dc_twin_action", "submit"}
    return {
        name: str(description).splitlines()[0]
        for name, description in sorted(actions.items())
        if name in allowed
    }


def blackbox_task_instructions(task_type: str | None) -> list[str]:
    """Return black-box-specific instructions that cannot be confused with parser output."""
    common = [
        "Use shell commands to call python3 dc_twin_tool.py; do not answer with markdown API-call blocks.",
        "The evaluator only records actions sent through dc_twin_tool.py.",
        "Read INITIAL_OBSERVATION.json first; it contains the current agent-visible causal snapshot and is already eligible evidence.",
        "Do not call observe merely to repeat the initial snapshot; observe after state changes or when a newer causal cut or deliberate drill-down is needed.",
        "Use only telemetry from INITIAL_OBSERVATION.json or dc_twin_tool.py responses as evidence.",
    ]
    fault_contract = list(fault_diagnosis_contract_lines())
    if task_type == "detection":
        return (
            common
            + fault_contract
            + [
                "For final detection, run: python3 dc_twin_tool.py submit --json '<json object>'.",
                'Submit JSON shape: {"incident_detected": true, "diagnosis": "<concise free-form underlying mechanism>", "target": "<visible target or null>", "evidence": ["<agent-visible supporting evidence>"]}.',
                'If no incident is present, submit: {"incident_detected": false, "evidence": []}.',
            ]
        )
    if task_type == "localization":
        return common + [
            "For final localization, run: python3 dc_twin_tool.py submit --json '<json array>'.",
            "Submit a JSON array of concrete visible datacenter target IDs or target scopes where the root cause lies.",
            "If no fault is localized, submit an empty JSON array: [].",
        ]
    if task_type == "analysis":
        return (
            common
            + fault_contract
            + [
                "For final root-cause analysis, run: python3 dc_twin_tool.py submit --json '<json object>'.",
                'Submit JSON shape: {"root_cause": "<concise free-form underlying mechanism>", "target": "<visible target>", "domain": "<operational domain>", "evidence": ["<agent-visible supporting evidence>"]}.',
                'If no incident is present, submit: {"root_cause": "none", "evidence": []}.',
            ]
        )
    if task_type == "mitigation":
        return common + [
            "For mitigation, inspect telemetry, apply visible controls through python3 dc_twin_tool.py action --json '<json object>', verify the observation is stable, then run python3 dc_twin_tool.py submit-empty.",
            "If telemetry shows a degraded, failed, overloaded, partitioned, or otherwise offending component that is also a visible action target, apply the relevant control to that observed component before relying on compensating controls elsewhere.",
            'Example cooling control JSON: {"action_type": "set_cooling", "parameters": {"target": "<cooling-unit-id-from-telemetry>", "fan_speed_percent": 100, "supply_air_temperature_c": 16}, "advance_ticks": 1}.',
        ]
    return common + [
        "When ready to submit a final answer, use python3 dc_twin_tool.py submit --json '<json value>' or python3 dc_twin_tool.py submit-empty as appropriate.",
    ]


def write_workspace_files(
    workspace: Path,
    task_text: str,
    initial_observation: dict[str, Any] | None = None,
) -> None:
    (workspace / "TASK.md").write_text(task_text, encoding="utf-8")
    (workspace / "INITIAL_OBSERVATION.json").write_text(
        json.dumps(
            sanitize_observation(initial_observation),
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "README_TOOL.md").write_text(tool_readme_text(), encoding="utf-8")
    (workspace / "dc_twin_tool.py").write_text(DC_TWIN_TOOL_SOURCE, encoding="utf-8")
    (workspace / "dc_twin_tool.py").chmod(0o700)


def tool_readme_text() -> str:
    return textwrap.dedent(
        """\
        # Data Center Twin Tool

        Read `INITIAL_OBSERVATION.json` before using the tool. It contains the
        current observation, so do not immediately call `observe` to repeat it.

        Use only this tool to interact with the benchmark:

        - `python3 dc_twin_tool.py action-space`
        - `python3 dc_twin_tool.py observe --log-limit 20 --include-config`
        - `python3 dc_twin_tool.py observe --channel metric --detail raw --metric-name rack.inlet_temperature --entity-id rack-r1-row1-01`
        - `python3 dc_twin_tool.py action --json '<json object>'`
        - `python3 dc_twin_tool.py submit --json '<json value>'`
        - `python3 dc_twin_tool.py submit-empty`
        """
    )


def command_argv_from_template(
    template: str,
    *,
    task_file: Path,
    workspace: Path,
    tool_file: Path,
    agent_name: str,
    protocol_timeout_seconds: float | None = None,
) -> list[str]:
    rendered = template
    replacements = {
        "{task_file}": str(task_file),
        "{workspace}": str(workspace),
        "{tool_file}": str(tool_file),
        "{agent_name}": agent_name,
    }
    if "{protocol_timeout_seconds}" in rendered:
        if protocol_timeout_seconds is None or protocol_timeout_seconds <= 0:
            raise ValueError("command template requires a positive protocol timeout")
        replacements["{protocol_timeout_seconds}"] = (
            f"{protocol_timeout_seconds:.6f}".rstrip("0").rstrip(".")
        )
    for key, value in replacements.items():
        rendered = rendered.replace(key, shlex.quote(value))
    argv = shlex.split(rendered)
    if not argv:
        raise ValueError("blackbox command template rendered to an empty command")
    return argv


def blackbox_environment(
    base_env: Mapping[str, str],
    *,
    workspace_dir: Path,
    bridge_url: str,
    token: str,
    allowed_env_vars: tuple[str, ...] = (),
    allowed_env_prefixes: tuple[str, ...] = (),
    environment_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    env["PATH"] = base_env.get("PATH", os.defpath)
    for name in allowed_env_vars:
        if name in base_env:
            env[name] = base_env[name]
    for prefix in allowed_env_prefixes:
        for name, value in base_env.items():
            if name.startswith(prefix):
                env[name] = value
    for name, value in (environment_overrides or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "black-box environment override names must be non-empty strings"
            )
        if not isinstance(value, str):
            raise ValueError(
                f"black-box environment override {name!r} must be a string"
            )
        env[name] = value
    env["HOME"] = str(workspace_dir)
    env["TMPDIR"] = str(workspace_dir)
    env["PWD"] = str(workspace_dir)
    env["PYTHONNOUSERSITE"] = "1"
    env["DC_TWIN_TOOL_URL"] = bridge_url
    env["DC_TWIN_TOOL_TOKEN"] = token
    env["DC_TWIN_WORKSPACE"] = str(workspace_dir)
    return env


def run_external_command(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    sandbox_mode: str = "docker",
    docker_image: str = "python:3.11-slim",
    stdin_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if cwd.resolve() == REPO_ROOT.resolve():
        raise ValueError("refusing to run black-box command in repository root")
    if sandbox_mode not in {"none", "docker"}:
        raise ValueError(f"unsupported black-box sandbox mode: {sandbox_mode}")
    run_argv = argv
    run_cwd = cwd
    run_env = env
    if sandbox_mode == "docker":
        run_argv = docker_command_argv(argv, cwd, env, docker_image)
        run_cwd = cwd
        run_env = {"PATH": env.get("PATH", os.defpath)}
    process = subprocess.Popen(
        run_argv,
        cwd=run_cwd,
        env=run_env,
        text=True,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(
            input=stdin_text,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        process_group_terminated, process_group_reaped, stdout, stderr = (
            _terminate_and_reap_process_group(process)
        )
        error = ExternalProcessTimeout(
            run_argv,
            timeout_seconds,
            output=stdout,
            stderr=stderr,
        )
        error.process_group_terminated = process_group_terminated
        error.process_group_reaped = process_group_reaped
        raise error
    except BaseException:
        # A runner restart (SIGINT/KeyboardInterrupt, cancellation, or another
        # non-Exception control flow) must not orphan the isolated episode.
        _terminate_and_reap_process_group(process)
        raise
    return subprocess.CompletedProcess(
        run_argv,
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _terminate_and_reap_process_group(
    process: subprocess.Popen[str],
) -> tuple[bool, bool, str, str]:
    """Terminate an episode's process group and wait for its direct child."""

    terminated = False
    try:
        os.killpg(process.pid, signal.SIGTERM)
        terminated = True
    except ProcessLookupError:
        pass
    try:
        stdout, stderr = process.communicate(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            terminated = True
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        pass
    else:
        # The group leader may have exited while a detached descendant still
        # holds the original process group.  Kill that remainder as well.
        try:
            os.killpg(process.pid, signal.SIGKILL)
            terminated = True
        except ProcessLookupError:
            pass
    group_deadline = time.monotonic() + PROCESS_TERMINATION_GRACE_SECONDS
    reaped = False
    while time.monotonic() < group_deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            reaped = process.poll() is not None
            break
        time.sleep(0.01)
    return terminated, reaped, stdout or "", stderr or ""


def docker_command_argv(
    argv: list[str], workspace: Path, env: dict[str, str], image: str
) -> list[str]:
    workspace = workspace.resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "--network=bridge",
        "--mount",
        f"type=bind,src={workspace},dst={workspace},rw",
        "--workdir",
        str(workspace),
    ]
    for name, value in sorted(env.items()):
        command.extend(["--env", f"{name}={value}"])
    command.append(image)
    command.extend(argv)
    return command


def sandbox_isolation_guarantee(mode: str) -> str:
    if mode == "docker":
        return (
            "Docker container with only the episode workspace bind-mounted; "
            "the benchmark tool uses a Unix socket and host networking is disabled."
        )
    return "Workspace file isolation only; not an OS sandbox or security boundary."


def sandbox_warning(mode: str) -> str | None:
    if mode == "none":
        return "sandbox=none does not prevent arbitrary host filesystem reads by the external process."
    return None


def assert_workspace_isolated(workspace: Path) -> None:
    workspace = workspace.resolve()
    repo = REPO_ROOT.resolve()
    if workspace == repo:
        raise ValueError("black-box workspace cannot be repository root")
    try:
        workspace.relative_to(repo)
    except ValueError:
        return
    raise ValueError("black-box workspace must be outside the repository tree")


def list_workspace_files(workspace: Path) -> list[str]:
    if not workspace.exists():
        return []
    return sorted(
        str(path.relative_to(workspace))
        for path in workspace.rglob("*")
        if path.is_file()
    )


def forbidden_workspace_files(files: list[str]) -> list[str]:
    leaks = []
    for file_name in files:
        name = Path(file_name).name
        if name in FORBIDDEN_WORKSPACE_NAMES:
            leaks.append(file_name)
    return leaks


def visible_agent_actions(action_space_payload: dict[str, Any] | None) -> set[str]:
    action_space = sanitize_action_space(action_space_payload) or {}
    actions = action_space.get("agent_actions")
    if isinstance(actions, list):
        return {action for action in actions if isinstance(action, str)}
    return set()


def contains_forbidden_tool_term(payload: Any) -> bool:
    text = json.dumps(payload, sort_keys=True, default=str).lower()
    return any(term in text for term in FORBIDDEN_TOOL_TERMS)


def sanitize_task_text(text: str) -> str:
    text = text.replace(str(REPO_ROOT), "<repo-redacted>")
    text = text.replace("scenarios.json", "<redacted>")
    text = text.replace("registry.py", "<redacted>")
    text = text.replace("success_criteria", "<redacted>")
    text = text.replace("expected", "required")
    text = text.replace("ground_truth", "<redacted>")
    text = text.replace("fault injection", "incident setup")
    text = text.replace("Fault Injection", "Incident Setup")
    return text


def render_env_response(env_response: Any) -> str:
    rendered, _ = render_env_response_with_observation(env_response)
    return rendered


def format_api_call(api_name: str, args: list[Any], kwargs: dict[str, Any]) -> str:
    rendered_args = [repr(arg) for arg in args]
    rendered_args.extend(f"{key}={value!r}" for key, value in kwargs.items())
    return f"{api_name}({', '.join(rendered_args)})"


def fence_api_call(call: str) -> str:
    return f"```\n{call}\n```"


def is_json_text(value: str) -> bool:
    try:
        json.loads(value)
        return True
    except json.JSONDecodeError:
        return False


def json_default(value: Any) -> Any:
    if isinstance(value, SessionItem):
        return value.model_dump()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def error_record(phase: str, error: Exception) -> dict[str, Any]:
    return {
        "phase": phase,
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


DC_TWIN_TOOL_SOURCE = r"""#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import http.client
import json
import os
import socket
import sys
import urllib.error
import urllib.request


FORBIDDEN_TERMS = {
    "scenarios.json",
    "registry.py",
    "aiopslab/orchestrator/problems",
    "orchestrator/problems/data_center_twin",
    "evaluator/debug",
    "evaluator debug",
    "/debug",
    "debug endpoint",
    "success_criteria",
    "ground_truth",
    "hidden benchmark state",
    "hidden state",
}


class ToolArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main() -> int:
    workspace = Path(os.environ.get("DC_TWIN_WORKSPACE", "")).resolve()
    if not workspace or Path.cwd().resolve() != workspace:
        print(json.dumps({"ok": False, "error": "tool must run from isolated workspace"}))
        return 2
    if len(sys.argv) < 2:
        return invalid("missing command", sys.argv[1:])
    command = sys.argv[1]
    if command == "action-space":
        if sys.argv[2:]:
            return invalid("action-space received unexpected arguments", sys.argv[2:])
        return send({"command": "action_space", "arguments": {}})
    if command == "observe":
        parser = ToolArgumentParser(prog="dc_twin_tool.py observe")
        parser.add_argument("--log-limit", type=int, default=20)
        parser.add_argument("--include-config", action="store_true")
        parser.add_argument(
            "--channel",
            dest="channels",
            action="append",
            choices=["log", "metric", "alert", "trace", "config"],
        )
        parser.add_argument("--lookback-seconds", type=float)
        parser.add_argument("--detail", choices=["overview", "raw"], default="overview")
        parser.add_argument("--metric-name", dest="metric_names", action="append")
        parser.add_argument("--entity-id", dest="entity_ids", action="append")
        parser.add_argument("--subsystem-id", dest="subsystem_ids", action="append")
        parser.add_argument("--alert-name", dest="alert_names", action="append")
        try:
            args = parser.parse_args(sys.argv[2:])
        except ValueError as error:
            return invalid(f"invalid observe parameters: {error}", sys.argv[2:])
        arguments = {
            "log_limit": args.log_limit,
            "include_config": args.include_config,
        }
        if args.channels is not None:
            arguments["channels"] = args.channels
        if args.lookback_seconds is not None:
            arguments["lookback_seconds"] = args.lookback_seconds
        if args.detail != "overview":
            arguments["detail"] = args.detail
        if args.metric_names is not None:
            arguments["metric_names"] = args.metric_names
        if args.entity_ids is not None:
            arguments["entity_ids"] = args.entity_ids
        if args.subsystem_ids is not None:
            arguments["subsystem_ids"] = args.subsystem_ids
        if args.alert_names is not None:
            arguments["alert_names"] = args.alert_names
        return send({"command": "observe", "arguments": arguments})
    if command == "action":
        payload = json_arg("dc_twin_tool.py action", sys.argv[2:])
        if payload is None:
            return 2
        return send({"command": "action", "arguments": {"payload": payload}})
    if command == "submit":
        payload = json_arg("dc_twin_tool.py submit", sys.argv[2:])
        if payload is None:
            return 2
        return send({"command": "submit", "arguments": {"payload": payload}})
    if command == "submit-empty":
        if sys.argv[2:]:
            return invalid("submit-empty received unexpected arguments", sys.argv[2:])
        return send({"command": "submit_empty", "arguments": {}})
    return invalid(f"invalid command: {command}", sys.argv[1:])


def json_arg(prog: str, args: list[str]):
    parser = ToolArgumentParser(prog=prog)
    parser.add_argument("--json", required=True)
    try:
        parsed = parser.parse_args(args)
    except ValueError as error:
        invalid(f"invalid command parameters: {error}", args)
        return None
    if contains_forbidden(parsed.json):
        invalid("request references hidden or forbidden benchmark paths", args)
        return None
    try:
        return json.loads(parsed.json)
    except json.JSONDecodeError as error:
        invalid(f"invalid JSON: {error}", args)
        return None


def invalid(message: str, raw_args: list[str]) -> int:
    response = rpc({"command": "invalid", "arguments": {"raw_args": raw_args, "message": message}})
    print(json.dumps(response, sort_keys=True))
    return 2


def send(payload: dict) -> int:
    response = rpc(payload)
    print(json.dumps(response, sort_keys=True))
    return 0 if response.get("ok") else 2



def rpc(payload: dict) -> dict:
    url = os.environ.get("DC_TWIN_TOOL_URL")
    token = os.environ.get("DC_TWIN_TOOL_TOKEN")
    if not url or not token:
        return {"ok": False, "error": "tool bridge is not configured"}
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if url.startswith("unix://"):
        socket_path = url.removeprefix("unix://")

        class UnixHTTPConnection(http.client.HTTPConnection):
            def connect(self):
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect(socket_path)

        try:
            connection = UnixHTTPConnection("localhost", timeout=30)
            connection.request("POST", "/tool", body=body, headers=headers)
            response = connection.getresponse()
            response_body = response.read().decode("utf-8", "replace")
            connection.close()
            try:
                return json.loads(response_body)
            except json.JSONDecodeError:
                return {"ok": False, "error": response_body}
        except Exception as error:
            return {"ok": False, "error": str(error)}
    request = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"ok": False, "error": body}
    except Exception as error:
        return {"ok": False, "error": str(error)}


def contains_forbidden(value: str) -> bool:
    lowered = value.lower()
    return any(term in lowered for term in FORBIDDEN_TERMS)


if __name__ == "__main__":
    raise SystemExit(main())
"""
