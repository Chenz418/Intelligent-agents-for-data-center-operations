"""In-process DC-Bench simulator adapter."""

from copy import deepcopy
import json
from typing import Any
from pathlib import Path
import sys

APP_DIR = (
    Path(__file__).resolve().parents[3] / "aiopslab-applications/dataCenterTwin/app"
)
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
BENCHMARK_HOST_VISIBILITY = "rack"


class DataCenterTwin:
    """Local simulator adapter implementing the DataCenterTwin app surface."""

    namespace = "data-center-twin-inprocess"

    def __init__(self) -> None:
        # Keep listing and validation commands independent of simulator imports.
        from dc_twin.config import default_config
        from dc_twin.simulator import DataCenterSimulator

        class EvaluatorInstrumentedSimulator(DataCenterSimulator):
            """Retain privileged state after each logical simulator tick.

            Agent actions still execute once through ``apply_agent_action`` and
            return the same endpoint response. Splitting only the simulator's
            internal tick loop lets the evaluator audit a multi-tick action
            without exposing intermediate state to the agent.
            """

            evaluator_tick_sink: Any = None

            def apply_control(self, request: Any) -> dict[str, Any]:
                result = super().apply_control(request)
                if self.evaluator_tick_sink is not None:
                    # ``apply_control`` recalculates health at the current
                    # simulator timestamp before ``apply_agent_action``
                    # optionally advances time. Retain that ordered,
                    # same-timestamp transition so recovery can begin at the
                    # true earliest simulator time.
                    self.evaluator_tick_sink(self.state_summary())
                return result

            def step(self, ticks: int = 1) -> dict[str, Any]:
                if self.evaluator_tick_sink is None:
                    return super().step(ticks)
                if ticks < 1:
                    return super().step(ticks)
                final_summary: dict[str, Any] = {}
                for _ in range(ticks):
                    final_summary = super().step(1)
                    self.evaluator_tick_sink(final_summary)
                return final_summary

        self._pending_evaluator_tick_summaries: list[dict[str, Any]] = []
        self.simulator = EvaluatorInstrumentedSimulator(default_config())
        self.simulator.evaluator_tick_sink = self._record_evaluator_tick_summary

    def _record_evaluator_tick_summary(self, summary: dict[str, Any]) -> None:
        self._pending_evaluator_tick_summaries.append(deepcopy(summary))

    def consume_evaluator_tick_summaries(self) -> list[dict[str, Any]]:
        """Drain evaluation-only transition/per-tick summaries from the last action."""
        summaries = self._pending_evaluator_tick_summaries
        self._pending_evaluator_tick_summaries = []
        return summaries

    def delete(self) -> None:
        return None

    def cleanup(self) -> None:
        self.clear_faults()

    def get_app_summary(self) -> str:
        return (
            "In-process Data Center Twin simulator "
            f"(episode={self.simulator.episode_id})"
        )

    def request_in_pod(
        self, method: str, path: str, payload: dict | None = None
    ) -> str:
        """Translate problem fault API calls into simulator operations."""
        from dc_twin.faults import FaultRequest

        try:
            method = method.upper()
            if path == "/faults" and method == "POST":
                return json.dumps(
                    self.simulator.inject_fault(FaultRequest(**(payload or {})))
                )
            if path == "/faults" and method == "GET":
                return json.dumps(self.simulator.list_faults())
            if path == "/faults" and method == "DELETE":
                return json.dumps(self.clear_faults())
            return json.dumps(
                {
                    "http_status": 404,
                    "error": {"detail": f"unsupported path: {method} {path}"},
                }
            )
        except Exception as error:
            return json.dumps(self._error_response(error))

    def agent_action_space(self) -> dict[str, Any]:
        """Return the simulator's agent-visible action schema."""
        from dc_twin.agent_interface import action_space

        return action_space()

    def agent_telemetry(
        self,
        query_time_seconds: int | float | None = None,
        query_watermark_sequence: str | None = None,
        lookback_seconds: int | float = 300,
        channels: list[str] | tuple[str, ...] | None = None,
        include_config: bool = True,
        log_limit: int | None = None,
    ) -> dict[str, Any]:
        """Return the same canonical causal snapshot used by StateBundle."""
        from dc_twin.agent_interface import canonical_agent_telemetry

        try:
            return canonical_agent_telemetry(
                self.simulator,
                query_time_seconds=query_time_seconds,
                query_watermark_sequence=query_watermark_sequence,
                lookback_seconds=lookback_seconds,
                channels=channels,
                include_config=include_config,
                log_limit=log_limit,
            )
        except Exception as error:
            return self._error_response(error)

    def agent_reset(
        self,
        seed: int | None = None,
        config_override: dict[str, Any] | None = None,
        workload: dict[str, Any] | None = None,
        stabilization_ticks: int = 0,
        log_limit: int = 20,
        include_config: bool = True,
        include_action_schema: bool = False,
    ) -> dict[str, Any]:
        """Reset simulator state with scenario config/workload from the problem."""
        from dc_twin.agent_interface import AgentResetRequest, reset_agent_environment

        try:
            self._pending_evaluator_tick_summaries = []
            request = AgentResetRequest(
                seed=seed,
                config_override=config_override or {},
                workload=workload,
                stabilization_ticks=stabilization_ticks,
                log_limit=log_limit,
                include_config=include_config,
                include_action_schema=include_action_schema,
                host_visibility=BENCHMARK_HOST_VISIBILITY,
            )
            return reset_agent_environment(self.simulator, request)
        except Exception as error:
            return self._error_response(error)

    def agent_action(
        self,
        action_type: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Apply one agent action through the simulator's structured interface."""
        from dc_twin.agent_interface import AgentActionRequest, apply_agent_action

        try:
            self._pending_evaluator_tick_summaries = []
            # Host granularity is fixed by the benchmark.
            kwargs.pop("host_visibility", None)
            payload = {
                "action_type": action_type,
                "host_visibility": BENCHMARK_HOST_VISIBILITY,
                **kwargs,
            }
            if parameters is not None:
                payload["parameters"] = parameters
            return apply_agent_action(self.simulator, AgentActionRequest(**payload))
        except Exception as error:
            return self._error_response(error)

    def evaluator_state(
        self,
        log_limit: int = 50,
        include_config: bool = True,
    ) -> dict[str, Any]:
        """Return raw simulator state exclusively to the task evaluator."""
        from dc_twin.agent_interface import get_evaluator_state

        try:
            return get_evaluator_state(
                self.simulator,
                log_limit=log_limit,
                include_config=include_config,
            )
        except Exception as error:
            return self._error_response(error)

    def clear_faults(self) -> list[dict[str, Any]]:
        """Best-effort local equivalent of deleting all active scenario faults."""
        from dc_twin.simulator import SimulationError

        try:
            return self.simulator.clear_faults()
        except SimulationError:
            return []

    def _error_response(self, error: Exception) -> dict[str, Any]:
        from dc_twin.simulator import SimulationError

        try:
            from pydantic import ValidationError
        except ImportError:  # pragma: no cover - pydantic is a runtime dependency
            ValidationError = ()  # type: ignore[assignment,misc]
        client_error_types = (SimulationError,)
        if isinstance(ValidationError, type):
            client_error_types = (*client_error_types, ValidationError)
        return {
            # Only known request-validation failures are attributable to the
            # agent. Unexpected implementation failures are server errors and
            # must not inflate invalid-action metrics.
            "http_status": 400 if isinstance(error, client_error_types) else 500,
            "error": {
                "type": type(error).__name__,
                "detail": str(error),
            },
        }
