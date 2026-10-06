"""Data Center Twin cooling degradation benchmark problems."""

from copy import deepcopy
import json
import math
import textwrap
from typing import Any

from aiopslab.agent_telemetry import AgentObservationRequest, logical_observation_key
from aiopslab.orchestrator.tasks import (
    DetectionTask,
    LocalizationTask,
    AnalysisTask,
    MitigationTask,
)
from aiopslab.service.apps.data_center_twin import DataCenterTwin
from aiopslab.session import SessionItem
from aiopslab.utils.status import InvalidActionError
from .semantic_evaluation import (
    make_input,
    scoring_ground_truth,
    SemanticEvaluatorError,
)
from .mitigation_metrics import (
    SLO_NORMALIZATION_EPSILON,
    annotate_stable_recovery,
    compute_slo_metrics,
    compute_stable_recovery,
    make_constraint,
)
from .scenarios import (
    fault_diagnosis_contract_lines,
    get_scenario,
)
from .visibility import (
    sanitize_agent_action_response,
    sanitize_agent_action_space,
    sanitize_agent_observation,
)


class DataCenterTwinSetupError(RuntimeError):
    """Raised when evaluator-owned data-center simulator setup fails."""


class DataCenterTwinBaseTask:
    SCENARIO_ID = "data_center_twin-cooling_degradation-detection-1"
    START_WORKLOAD_BEFORE_FAULT = True
    BENCHMARK_HOST_VISIBILITY = "rack"
    TELEMETRY_VIEWS = frozenset({"canonical"})
    DEFAULT_TELEMETRY_VIEW = "canonical"
    CANONICAL_LOOKBACK_SECONDS = 300
    CANONICAL_CHANNELS = ("log", "metric", "alert", "trace", "config")
    OBSERVATION_QUERY_FIELDS = frozenset(
        {
            "log_limit",
            "include_config",
            "channels",
            "lookback_seconds",
            "detail",
            "metric_names",
            "entity_ids",
            "subsystem_ids",
            "alert_names",
        }
    )
    ACTION_CONTEXT_EXCLUDED_FIELDS = OBSERVATION_QUERY_FIELDS | frozenset(
        {
            "host_visibility",
            "include_action_schema",
            "accepted",
            "action_result",
            "active_faults",
            "active_faults_after",
            "active_faults_before",
            "available_actions",
            "error",
            "evaluator_state",
            "fault_id",
            "fault_target",
            "fault_type",
            "ground_truth",
            "http_status",
            "observation",
            "oracle",
            "score_hints",
            "sim_time_seconds_after",
            "sim_time_seconds_before",
            "step_summary",
            "success_criteria",
        }
    )
    READ_TIME_AGENT_ACTIONS = frozenset({"observe", "noop", "step"})
    WRITE_AGENT_ACTIONS = frozenset(
        {
            "calibrate_sensor",
            "set_cooling",
            "migrate_workload",
            "throttle_workload",
            "set_server_maintenance",
            "clear_server_maintenance",
            "update_autoscaler_policy",
            "repair_monitoring_pipeline",
            "update_placement_policy",
            "update_load_balancer_config",
        }
    )
    DEFAULT_WORKLOAD = {
        "request_rate_per_second": 300,
        "placement_strategy": "spread",
        "noise_enabled": False,
    }
    DATA_CENTER_TWIN_READ_ACTION_DOCS = {
        "dc_twin_action_space": "Return the data-center-twin agent action schema advertised by /agent/action-space.",
        "dc_twin_observe": (
            "Return canonical telemetry from the shared causal snapshot. "
            "Args: log_limit: int = 20, include_config: bool = True; canonical "
            "requests also accept lookback_seconds, channels, detail='overview' "
            "or 'raw', metric_names, entity_ids, subsystem_ids, and alert_names. "
            "Start with the compact "
            "overview, then request raw detail only for suspicious channels or "
            "metric series."
        ),
    }
    DATA_CENTER_TWIN_WRITE_ACTION_DOCS = {
        "dc_twin_action": (
            "Apply a structured data-center-twin agent action through /agent/actions. "
            "Args: action_type: str, parameters: dict | None = None, plus action-specific keyword fields."
        ),
    }
    DATA_CENTER_TWIN_ACTION_DOCS = DATA_CENTER_TWIN_READ_ACTION_DOCS
    EXPOSE_DC_TWIN_ACTION = True
    FILTER_DC_TWIN_ACTION_SPACE = True
    ALLOWED_DC_TWIN_AGENT_ACTIONS: frozenset[str] | None = READ_TIME_AGENT_ACTIONS

    def __init__(self):
        self.scenario = get_scenario(self.SCENARIO_ID)
        self.app = DataCenterTwin()
        self.namespace = self.app.namespace
        self.seed = self.scenario.seed
        self.config_override = deepcopy(self.scenario.config_override)
        self.fault_type = self.scenario.fault_type
        self.faulty_component = self.scenario.fault_target
        self.fault_severity = self.scenario.fault_severity
        self.fault_duration_seconds = self.scenario.fault_duration_seconds
        self.fault_parameters = deepcopy(self.scenario.fault_parameters)
        self.workload_config = deepcopy(self.scenario.workload)
        self.stabilization_ticks = self.scenario.stabilization_ticks
        self.post_injection_ticks = self.scenario.post_injection_ticks
        self.expected = deepcopy(self.scenario.expected)
        self.success_criteria = deepcopy(self.scenario.success_criteria)
        self.allowed_dc_twin_agent_actions = (
            None
            if self.ALLOWED_DC_TWIN_AGENT_ACTIONS is None
            else frozenset(self.scenario.allowed_agent_actions)
        )
        self.stability_window_seconds = int(
            self.success_criteria.get("stability_window_seconds", 10)
        )
        self.failover_rack = self.success_criteria.get("failover_rack")
        self.network_queue_success_threshold = self.success_criteria.get(
            "workload_queue_length_max", 0
        )
        self.network_congestion_success_threshold = self.success_criteria.get(
            "workload_network_congestion_ratio_max",
            0.7,
        )
        self.network_latency_success_threshold_ms = self.success_criteria.get(
            "workload_average_latency_ms_max",
            250.0,
        )
        self.fault_injection_time = None
        self.episode_horizon_time = None
        self.evaluator_health_trajectory: list[dict[str, Any]] = []
        self._mitigation_metrics_cache: dict[str, Any] | None = None
        self.agent_action_space = None
        self.initial_agent_observation = None
        self.latest_agent_observation = None
        self.agent_action_history = []
        self.telemetry_view = self.DEFAULT_TELEMETRY_VIEW
        self.observation_condition = "full-canonical"
        self._observation_transform = None
        self._latest_complete_canonical_snapshot: dict[str, Any] | None = None
        # Evaluators must score only evidence that crossed the agent boundary.
        # Compact rendering happens in the benchmark runner after the
        # task records its canonical response, so keep the rendered projection
        # separately.  This history contains no evaluator-only or hidden
        # canonical fields and is never used by StateBundle.
        self._agent_rendered_observation_history: list[dict[str, Any]] = []
        self._agent_rendering_boundary_active = False

    def _api(
        self, method: str, path: str, payload: dict | None = None
    ) -> dict[str, Any]:
        output = self.app.request_in_pod(method, path, payload)
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"raw": output}

    def start_workload(self):
        print("== Start Workload ==")
        action_space = self._require_backend_success(
            self.app.agent_action_space(),
            "agent action-space",
            required_keys=("agent_actions",),
        )
        self.agent_action_space = self._task_action_space(action_space)
        reset_response = self._require_backend_success(
            self.app.agent_reset(
                seed=self.seed,
                config_override=self.config_override,
                workload=self.workload_config,
                stabilization_ticks=self.stabilization_ticks,
                log_limit=20,
            ),
            "agent reset",
            required_keys=("observation",),
        )
        reset_response = self._task_reset_response(reset_response)
        self.initial_agent_observation = reset_response.get("observation")
        self.latest_agent_observation = self._task_observation(
            self._require_backend_success(
                self._backend_agent_observation(log_limit=20, include_config=True),
                "post-reset observation",
                required_keys=self._observation_required_keys(),
            )
        )
        print("Data Center Twin workload started.")

    def inject_fault(self):
        print("== Fault Injection ==")
        response = self._require_backend_success(
            self._api(
                "POST",
                "/faults",
                {
                    "fault_type": self.fault_type,
                    "target": self.faulty_component,
                    "severity": self.fault_severity,
                    "duration_seconds": self.fault_duration_seconds,
                    **self.fault_parameters,
                },
            ),
            "fault injection",
            required_keys=("started_at_sim_time_seconds",),
        )
        self.fault_injection_time = response.get("started_at_sim_time_seconds")
        if isinstance(self.fault_injection_time, (int, float)) and not isinstance(
            self.fault_injection_time, bool
        ):
            self.episode_horizon_time = float(self.fault_injection_time) + float(
                self.fault_duration_seconds
            )
        self._capture_evaluator_health_sample("fault_injection")
        if self.post_injection_ticks:
            self._require_backend_success(
                self.app.agent_action(
                    "noop",
                    advance_ticks=self.post_injection_ticks,
                    log_limit=20,
                    include_config=True,
                    host_visibility=self.BENCHMARK_HOST_VISIBILITY,
                ),
                "post-injection settling",
                required_keys=("observation",),
            )
            captured = self._capture_pending_evaluator_tick_samples(
                "post_injection_settling"
            )
            if not captured:
                self._capture_evaluator_health_sample("post_injection_settling")
        self.latest_agent_observation = self._task_observation(
            self._require_backend_success(
                self._backend_agent_observation(log_limit=20, include_config=True),
                "post-fault observation",
                required_keys=self._observation_required_keys(),
            )
        )
        # Do not print evaluator-only mechanism or target data into a benchmark
        # process whose output may be captured alongside an agent run.
        print(f"Incident injection completed in namespace: {self.namespace}\n")

    def recover_fault(self):
        print("== Fault Recovery ==")
        self._require_backend_success(self._api("DELETE", "/faults"), "fault recovery")

    def configure_telemetry_view(self, telemetry_view: str) -> None:
        """Pin one representation for the whole episode and every method."""
        if telemetry_view not in self.TELEMETRY_VIEWS:
            raise ValueError(f"unsupported telemetry_view: {telemetry_view}")
        if self.agent_action_history:
            raise RuntimeError("telemetry_view cannot change after agent interaction")
        self.telemetry_view = telemetry_view

    def configure_observation_transform(self, transform, *, condition: str) -> None:
        """Install a trusted evaluator-side observation projection before interaction."""
        if condition != "statebundle":
            raise ValueError(f"unsupported observation condition: {condition}")
        if self.agent_action_history:
            raise RuntimeError(
                "observation transform cannot change after agent interaction"
            )
        if not callable(transform):
            raise TypeError("observation transform must be callable")
        self.observation_condition = condition
        self._observation_transform = transform

    def _observation_required_keys(self) -> tuple[str, ...]:
        if self.observation_condition == "statebundle":
            return ("evidence_groups",)
        return ("observations",)

    def _backend_agent_observation(
        self,
        *,
        log_limit: int | None = 20,
        include_config: bool = True,
        lookback_seconds: int | float | None = None,
        channels: list[str] | tuple[str, ...] | set[str] | str | None = None,
        detail: str = "overview",
        metric_names: list[str] | tuple[str, ...] | None = None,
        entity_ids: list[str] | tuple[str, ...] | None = None,
        subsystem_ids: list[str] | tuple[str, ...] | None = None,
        alert_names: list[str] | tuple[str, ...] | None = None,
        action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = self._canonical_observation_request(
            log_limit=log_limit,
            include_config=include_config,
            lookback_seconds=lookback_seconds,
            channels=channels,
        )
        selection_request = self._agent_observation_request(
            include_config=include_config,
            channels=request.get("channels"),
            lookback_seconds=float(request["lookback_seconds"]),
            log_limit=log_limit,
            detail=detail,
            metric_names=metric_names,
            entity_ids=entity_ids,
            subsystem_ids=subsystem_ids,
            alert_names=alert_names,
        )
        if self._observation_transform is None:
            observation = self.app.agent_telemetry(**request)
            if self._is_complete_canonical_request(request):
                self._latest_complete_canonical_snapshot = deepcopy(observation)
            return observation

        # StateBundle selection is defined over the complete benchmark
        # causal snapshot.  Safe agent-requested view controls accompany
        # that complete input as query context; storage rematerialization
        # below still cannot introduce an unselected observation.
        complete_snapshot = self.app.agent_telemetry(
            lookback_seconds=self.CANONICAL_LOOKBACK_SECONDS,
        )
        self._latest_complete_canonical_snapshot = deepcopy(complete_snapshot)
        observation = self._invoke_observation_transform(
            complete_snapshot,
            request=selection_request,
            action=action,
        )
        if not isinstance(observation, dict):
            raise DataCenterTwinSetupError(
                "observation transform must return an object"
            )
        # Aggregate payloads (metrics, logs, and traces) must be rebuilt by
        # the canonical recorder for the requested window.  Filtering a
        # selected 300-second aggregate by event end would leave its count,
        # statistics, and normalization tied to the wrong window.  Bind the
        # second, read-only query to the exact same causal cut, then expose
        # only logical observations selected by StateBundle.
        requested_snapshot = self.app.agent_telemetry(
            query_time_seconds=complete_snapshot.get("query_time_seconds"),
            query_watermark_sequence=complete_snapshot.get("query_watermark_sequence"),
            **request,
        )
        return self._filter_selected_statebundle_observations(
            observation,
            requested_snapshot=requested_snapshot,
        )

    def _agent_observation_request(
        self,
        *,
        include_config: bool,
        channels: list[str] | tuple[str, ...] | None,
        lookback_seconds: float,
        log_limit: int | None,
        detail: str,
        metric_names: list[str] | tuple[str, ...] | None,
        entity_ids: list[str] | tuple[str, ...] | None,
        subsystem_ids: list[str] | tuple[str, ...] | None,
        alert_names: list[str] | tuple[str, ...] | None,
    ) -> AgentObservationRequest:
        values: dict[str, Any] = {
            "include_config": include_config,
            "channels": channels,
            "lookback_seconds": lookback_seconds,
            "log_limit": log_limit,
            "detail": detail,
            "metric_names": metric_names or (),
            "entity_ids": entity_ids or (),
            "subsystem_ids": subsystem_ids or (),
            "alert_names": alert_names or (),
        }
        return AgentObservationRequest(**values)

    def _invoke_observation_transform(self, snapshot, *, request, action):
        if self._observation_transform is None:
            raise RuntimeError("observation transform is not configured")
        return self._observation_transform(snapshot, request=request, action=action)

    def _is_complete_canonical_request(self, request: dict[str, Any]) -> bool:
        channels = request.get("channels")
        return (
            float(request.get("lookback_seconds", -1.0))
            == float(self.CANONICAL_LOOKBACK_SECONDS)
            and (channels is None or tuple(channels) == self.CANONICAL_CHANNELS)
            and request.get("include_config") is True
            and request.get("log_limit") is None
        )

    def _canonical_observation_request(
        self,
        *,
        log_limit: int | None,
        include_config: bool,
        lookback_seconds: int | float | None,
        channels: list[str] | tuple[str, ...] | set[str] | str | None,
    ) -> dict[str, Any]:
        if log_limit is not None and (
            isinstance(log_limit, bool)
            or not isinstance(log_limit, int)
            or log_limit < 0
        ):
            raise ValueError("log_limit must be a non-negative integer")
        if not isinstance(include_config, bool):
            raise ValueError("include_config must be a boolean")
        lookback = (
            self.CANONICAL_LOOKBACK_SECONDS
            if lookback_seconds is None
            else lookback_seconds
        )
        if (
            isinstance(lookback, bool)
            or not isinstance(lookback, (int, float))
            or not math.isfinite(float(lookback))
            or float(lookback) < 0.0
        ):
            raise ValueError("lookback_seconds must be a finite non-negative number")
        if isinstance(channels, str):
            raw_channels: list[str] | None = [
                channel.strip() for channel in channels.split(",") if channel.strip()
            ]
        elif channels is None:
            raw_channels = None
        elif isinstance(channels, (list, tuple, set)) and all(
            isinstance(channel, str) for channel in channels
        ):
            raw_channels = list(channels)
        else:
            raise ValueError("channels must be a collection of channel names")
        if raw_channels is not None:
            requested = set(raw_channels)
            unsupported = requested - set(self.CANONICAL_CHANNELS)
            if unsupported:
                raise ValueError(
                    f"unsupported canonical telemetry channel(s): {sorted(unsupported)}"
                )
            normalized_channels: list[str] | None = [
                channel for channel in self.CANONICAL_CHANNELS if channel in requested
            ]
        else:
            normalized_channels = None
        return {
            "lookback_seconds": lookback,
            "channels": normalized_channels,
            "include_config": include_config,
            "log_limit": log_limit,
        }

    def _filter_selected_statebundle_observations(
        self,
        observation: dict[str, Any],
        *,
        requested_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Rematerialize selected evidence from an exact-cut canonical view.

        This never adds a logical candidate or changes model ordering.  It only
        replaces a selected full-window aggregate with the same logical
        series/entity aggregate produced for the explicit request.  If a
        selected anchor is absent from that view while a support remains, the
        first remaining selected observation becomes the public group anchor.
        """
        if (
            observation.get("schema_version") != "statebundle.output.v1"
            or requested_snapshot.get("schema_version") != "statebundle.canonical.v1"
        ):
            return observation
        groups = observation.get("evidence_groups")
        if not isinstance(groups, list):
            return observation

        def event_end(item: dict[str, Any]) -> float:
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                value = metadata.get("event_end_time_seconds")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
            window = item.get("window")
            if isinstance(window, dict):
                value = window.get("end_time_seconds")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
            return float("-inf")

        def rematerialization_key(item: dict[str, Any]) -> tuple[Any, ...]:
            payload = item.get("payload")
            payload = payload if isinstance(payload, dict) else {}
            channel = item.get("channel")
            if channel == "config":
                return (
                    "config-occurrence",
                    payload.get("scope"),
                    payload.get("path"),
                    payload.get("change_time_seconds"),
                )
            if channel == "alert":
                window = item.get("window")
                window = window if isinstance(window, dict) else {}
                return (
                    "alert-episode",
                    payload.get("alert_fingerprint"),
                    payload.get("target"),
                    window.get("start_time_seconds"),
                )
            return ("aggregate-series", logical_observation_key(item))

        available_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        requested_observations = requested_snapshot.get("observations")
        if isinstance(requested_observations, list):
            for item in requested_observations:
                if not isinstance(item, dict):
                    continue
                available_by_key.setdefault(rematerialization_key(item), []).append(
                    item
                )
        for candidates in available_by_key.values():
            candidates.sort(
                key=lambda item: (
                    event_end(item),
                    str(item.get("observation_id", "")),
                )
            )

        replacement_by_selected_id: dict[str, dict[str, Any]] = {}
        used_replacement_ids: set[str] = set()

        def rematerialize(selected: dict[str, Any]) -> dict[str, Any] | None:
            selected_id = str(selected.get("observation_id", ""))
            if selected_id in replacement_by_selected_id:
                return replacement_by_selected_id[selected_id]
            if selected_id.startswith("retained:pre-action-"):
                # This is a bounded, selector-owned copy from the preceding
                # causal cut.  Replacing it with the current aggregate would
                # collapse the before/after comparison; absence from the
                # current snapshot is therefore intentional, not a reason to
                # discard it.
                replacement_by_selected_id[selected_id] = selected
                return selected
            candidates = available_by_key.get(rematerialization_key(selected), [])
            if not candidates:
                return None
            for candidate in candidates:
                candidate_id = str(candidate.get("observation_id", ""))
                if (
                    candidate_id == selected_id
                    and candidate_id not in used_replacement_ids
                ):
                    replacement_by_selected_id[selected_id] = candidate
                    used_replacement_ids.add(candidate_id)
                    return candidate
            selected_end = event_end(selected)
            unused = [
                candidate
                for candidate in candidates
                if str(candidate.get("observation_id", "")) not in used_replacement_ids
            ]
            if not unused:
                return None
            replacement = min(
                unused,
                key=lambda item: (
                    abs(event_end(item) - selected_end),
                    str(item.get("observation_id", "")),
                ),
            )
            replacement_by_selected_id[selected_id] = replacement
            used_replacement_ids.add(str(replacement.get("observation_id", "")))
            return replacement

        filtered_groups: list[dict[str, Any]] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            candidates: list[dict[str, Any]] = []
            anchor = group.get("anchor")
            if isinstance(anchor, dict):
                candidates.append(anchor)
            supports = group.get("corroborating_observations")
            if isinstance(supports, list):
                candidates.extend(item for item in supports if isinstance(item, dict))
            retained = [
                rendered
                for item in candidates
                if (rendered := rematerialize(item)) is not None
            ]
            if retained:
                filtered_groups.append(
                    {
                        "anchor": retained[0],
                        "corroborating_observations": retained[1:],
                    }
                )

        filtered = deepcopy(observation)
        filtered["evidence_groups"] = filtered_groups
        if "target_scope_ambiguity" in observation and isinstance(
            observation.get("target_scope_candidates"), list
        ):
            retained_ids = {
                str(item.get("observation_id"))
                for group in filtered_groups
                for item in (
                    group["anchor"],
                    *group["corroborating_observations"],
                )
                if isinstance(item.get("observation_id"), str)
            }
            rematerialized_scope_candidates: list[dict[str, Any]] = []
            for candidate in observation["target_scope_candidates"]:
                if not isinstance(candidate, dict):
                    continue
                rematerialized_ids: list[str] = []
                support_ids = candidate.get("supporting_observation_ids")
                if not isinstance(support_ids, list):
                    continue
                for selected_id in support_ids:
                    if not isinstance(selected_id, str):
                        continue
                    replacement = replacement_by_selected_id.get(selected_id)
                    replacement_id = (
                        str(replacement.get("observation_id"))
                        if isinstance(replacement, dict)
                        and isinstance(replacement.get("observation_id"), str)
                        else selected_id
                    )
                    if (
                        replacement_id in retained_ids
                        and replacement_id not in rematerialized_ids
                    ):
                        rematerialized_ids.append(replacement_id)
                if not rematerialized_ids:
                    continue
                rematerialized_scope_candidates.append(
                    {
                        "scope": candidate.get("scope"),
                        "estimated_role": candidate.get("estimated_role"),
                        "supporting_observation_ids": rematerialized_ids,
                    }
                )
            filtered["target_scope_candidates"] = rematerialized_scope_candidates
            filtered["target_scope_ambiguity"] = bool(
                observation.get("target_scope_ambiguity")
                and len(rematerialized_scope_candidates) >= 2
            )
        return filtered

    def _require_backend_success(
        self,
        response: Any,
        operation: str,
        required_keys: tuple[str, ...] = (),
    ) -> Any:
        if isinstance(response, dict):
            http_status = response.get("http_status")
            failed_status = isinstance(http_status, int) and http_status >= 400
            if failed_status or "error" in response or "raw" in response:
                raise DataCenterTwinSetupError(
                    f"Data Center Twin {operation} failed: {self._format_backend_response(response)}"
                )
            missing_keys = [
                key
                for key in required_keys
                if key not in response or response[key] is None
            ]
            if missing_keys:
                raise DataCenterTwinSetupError(
                    f"Data Center Twin {operation} response missing keys {missing_keys}: "
                    f"{self._format_backend_response(response)}"
                )
            return response
        if required_keys:
            raise DataCenterTwinSetupError(
                f"Data Center Twin {operation} returned non-object response: "
                f"{self._format_backend_response(response)}"
            )
        return response

    def _format_backend_response(self, response: Any) -> str:
        try:
            return json.dumps(response, sort_keys=True, default=str)
        except TypeError:
            return str(response)

    def get_task_description(self):
        return textwrap.dedent(
            f"""\
            You are operating the Data Center Twin benchmark environment.

            Service Details:
            {self.app.get_app_summary()}

            Incident scope:
            - This task is a Data Center Twin {self.scenario.task_type} task.
            - The environment can expose facility, thermal, power, hardware, network,
              storage, software-control-plane, and application incidents.
            - Use the Data Center Twin action APIs to inspect observations and, for
              mitigation tasks, apply controls through the agent action interface.
            """
        )

    def get_instructions(self):
        task_type = self.scenario.task_type
        fault_contract = "\n".join(
            f"                {line}" for line in fault_diagnosis_contract_lines()
        )
        common = """\
            Use these task APIs:
            - dc_twin_action_space(): inspect the agent action schema and parameter requirements.
            - dc_twin_observe(log_limit=20, include_config=True, lookback_seconds=None, channels=None, detail="overview", metric_names=None, entity_ids=None, subsystem_ids=None, alert_names=None): inspect current telemetry without changing state. Canonical views honor the optional lookback/channel controls. Use detail="raw" with narrow channel/metric/entity/subsystem/alert selectors only after the compact overview identifies a suspicious area.
            - dc_twin_action(action_type, parameters=None, **kwargs): execute observe/noop/step and, when enabled for the task, control actions.

            Evidence is only credited when it appears in an observation returned by dc_twin_observe() or dc_twin_action("observe").
            Respond with exactly one API call in one markdown code block on each turn.
            """
        if task_type == "detection":
            task_specific = f"""\
                For final detection, submit a dictionary with incident status, one concise free-form diagnosis of the underlying fault mechanism, optional target, and supporting evidence fields.
{fault_contract}
                Example final answer:
                ```
                submit({{"incident_detected": True, "diagnosis": "<concise free-form underlying mechanism>", "target": None, "evidence": ["<agent-visible supporting evidence>"]}})
                ```
                If no incident is present:
                ```
                submit({{"incident_detected": False, "evidence": []}})
                ```
                """
        elif task_type == "localization":
            task_specific = """\
                For final localization, submit the concrete datacenter target IDs or target scopes where the root cause lies.
                The benchmark visibility policy is rack-granular: localize host failures to their visible rack, not to a hidden server ID.
                Valid target examples include cooling-unit IDs, rack IDs, storage, control-plane, workload, or application.
                Example final answer:
                ```
                submit(["component-id-from-telemetry"])
                ```
                If no fault is localized:
                ```
                submit([])
                ```
                """
        elif task_type == "analysis":
            task_specific = f"""\
                For final root-cause analysis, submit a dictionary containing one concise free-form diagnosis of the underlying fault mechanism, target, operational domain, and evidence fields observed from telemetry.
{fault_contract}
                Example final answer:
                ```
                submit({{"root_cause": "<concise free-form underlying mechanism>", "target": "<visible target>", "domain": "<operational domain>", "evidence": ["<agent-visible supporting evidence>"]}})
                ```
                If no incident is present:
                ```
                submit({{"root_cause": "none", "evidence": []}})
                ```
                """
        else:
            task_specific = """\
                For mitigation, inspect the environment, apply appropriate controls through dc_twin_action(...), verify the final observation is stable, then call submit().
                Example control and final submission:
                ```
                dc_twin_action("set_cooling", parameters={"target": "cooling-unit-id-from-telemetry", "fan_speed_percent": 100, "supply_air_temperature_c": 16}, advance_ticks=1)
                ```
                ```
                submit()
                ```
                """
        return textwrap.dedent(common + "\n" + task_specific)

    def get_available_actions(self):
        # Publish only task submission and the simulator's controlled APIs.
        parent_actions: dict[str, str] = {}
        parent_get_available_actions = getattr(super(), "get_available_actions", None)
        if parent_get_available_actions is not None:
            parent_actions = parent_get_available_actions()
        data_center_actions = {
            "submit": parent_actions.get("submit", "Submit the final task answer."),
            **self.DATA_CENTER_TWIN_READ_ACTION_DOCS,
        }
        if self.EXPOSE_DC_TWIN_ACTION:
            data_center_actions.update(self.DATA_CENTER_TWIN_WRITE_ACTION_DOCS)
        return data_center_actions

    def perform_action(self, action_name, *args, **kwargs):
        if action_name == "submit":
            parent_perform_action = getattr(super(), "perform_action", None)
            if parent_perform_action is None:
                raise InvalidActionError(action_name)
            return parent_perform_action(action_name, *args, **kwargs)
        if action_name == "dc_twin_action_space":
            return self.dc_twin_action_space()
        if action_name == "dc_twin_observe":
            return self.dc_twin_observe(*args, **kwargs)
        if action_name == "dc_twin_action":
            if not self.EXPOSE_DC_TWIN_ACTION:
                raise InvalidActionError(action_name)
            return self.dc_twin_action(*args, **kwargs)
        raise InvalidActionError(action_name)

    def dc_twin_action_space(self):
        response = self._task_action_space(self.app.agent_action_space())
        self.agent_action_space = response
        self._record_agent_interface_call("dc_twin_action_space", {}, response)
        return response

    def dc_twin_observe(
        self,
        log_limit: int = 20,
        include_config: bool = True,
        lookback_seconds: int | float | None = None,
        channels: list[str] | tuple[str, ...] | set[str] | str | None = None,
        detail: str = "overview",
        metric_names: list[str] | tuple[str, ...] | set[str] | str | None = None,
        entity_ids: list[str] | tuple[str, ...] | set[str] | str | None = None,
        subsystem_ids: list[str] | tuple[str, ...] | set[str] | str | None = None,
        alert_names: list[str] | tuple[str, ...] | set[str] | str | None = None,
    ):
        if detail not in {"overview", "raw"}:
            raise ValueError("detail must be 'overview' or 'raw'")
        normalized_metric_names = self._normalize_agent_drilldown_values(
            metric_names,
            field_name="metric_names",
        )
        normalized_entity_ids = self._normalize_agent_drilldown_values(
            entity_ids,
            field_name="entity_ids",
        )
        normalized_subsystem_ids = self._normalize_agent_drilldown_values(
            subsystem_ids,
            field_name="subsystem_ids",
        )
        normalized_alert_names = self._normalize_agent_drilldown_values(
            alert_names,
            field_name="alert_names",
        )
        response = self._task_observation(
            self._backend_agent_observation(
                log_limit=log_limit,
                include_config=include_config,
                lookback_seconds=lookback_seconds,
                channels=channels,
                detail=detail,
                metric_names=normalized_metric_names,
                entity_ids=normalized_entity_ids,
                subsystem_ids=normalized_subsystem_ids,
                alert_names=normalized_alert_names,
            )
        )
        self.latest_agent_observation = response
        recorded_request = {
            "log_limit": log_limit,
            "include_config": include_config,
            "lookback_seconds": lookback_seconds,
            "channels": channels,
            "detail": detail,
            "metric_names": normalized_metric_names,
            "entity_ids": normalized_entity_ids,
            "telemetry_view": self.telemetry_view,
        }
        # Preserve the existing trace shape until these optional selectors are
        # actually issued by the agent (and supported by the request schema).
        if normalized_subsystem_ids is not None:
            recorded_request["subsystem_ids"] = normalized_subsystem_ids
        if normalized_alert_names is not None:
            recorded_request["alert_names"] = normalized_alert_names
        self._record_agent_interface_call(
            "dc_twin_observe",
            recorded_request,
            response,
        )
        return response

    @staticmethod
    def _normalize_agent_drilldown_values(
        value: list[str] | tuple[str, ...] | set[str] | str | None,
        *,
        field_name: str,
    ) -> list[str] | None:
        """Validate inference/rendering selectors without touching storage."""

        if value is None:
            return None
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, (list, tuple, set)) or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            raise ValueError(f"{field_name} must contain non-empty strings")
        return sorted({item.strip() for item in values})

    def dc_twin_action(
        self, action_type: str, parameters: dict | None = None, **kwargs
    ):
        if not self.EXPOSE_DC_TWIN_ACTION:
            raise InvalidActionError("dc_twin_action")
        if (
            self.allowed_dc_twin_agent_actions is not None
            and action_type not in self.allowed_dc_twin_agent_actions
        ):
            raise InvalidActionError(action_type)
        if "host_visibility" in kwargs:
            raise InvalidActionError("host_visibility")
        response = self.app.agent_action(
            action_type,
            parameters=parameters,
            host_visibility=self.BENCHMARK_HOST_VISIBILITY,
            **kwargs,
        )
        action_context = self._agent_action_context(
            action_type,
            parameters=parameters,
            keyword_parameters=kwargs,
        )
        post_action_query = {
            name: kwargs[name]
            for name in self.OBSERVATION_QUERY_FIELDS
            if name in kwargs
        }
        response = self._task_action_response(
            response,
            action=action_context,
            post_action_query=post_action_query,
        )
        observation = (
            response.get("observation") if isinstance(response, dict) else None
        )
        if observation is not None:
            self.latest_agent_observation = self._task_observation(observation)
        self._record_agent_interface_call(
            "dc_twin_action",
            {"action_type": action_type, "parameters": parameters, **kwargs},
            response,
        )
        captured = self._capture_pending_evaluator_tick_samples(
            f"agent_action:{action_type}"
        )
        if not captured:
            self._capture_evaluator_health_sample(f"agent_action:{action_type}")
        return response

    def _agent_action_context(
        self,
        action_type: str,
        *,
        parameters: dict[str, Any] | None,
        keyword_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Retain only control inputs issued by the agent, never its response."""

        merged = {
            key: deepcopy(value)
            for key, value in keyword_parameters.items()
            if key not in self.ACTION_CONTEXT_EXCLUDED_FIELDS
        }
        if isinstance(parameters, dict):
            # The agent interface gives nested parameters precedence after
            # rejecting conflicts, so mirror its normalized control payload.
            merged.update(
                {
                    key: deepcopy(value)
                    for key, value in parameters.items()
                    if key not in self.ACTION_CONTEXT_EXCLUDED_FIELDS
                }
            )
        return {"action_type": action_type, "parameters": merged}

    def _task_action_response(
        self,
        response: dict[str, Any],
        *,
        action: dict[str, Any] | None = None,
        post_action_query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(response, dict):
            return response
        filtered = deepcopy(response)
        filtered.pop("step_summary", None)
        observation = filtered.get("observation")
        if isinstance(observation, dict):
            observation = self._require_backend_success(
                self._backend_agent_observation(
                    **(post_action_query or {}),
                    action=action,
                ),
                "canonical post-action observation",
                required_keys=self._observation_required_keys(),
            )
            filtered["observation"] = self._task_observation(observation)
        available_actions = filtered.get("available_actions")
        if isinstance(available_actions, dict):
            filtered["available_actions"] = self._task_action_space(available_actions)
        return sanitize_agent_action_response(
            filtered,
            allowed_actions=self.allowed_dc_twin_agent_actions,
        )

    def _task_reset_response(self, response: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(response, dict):
            return response
        filtered = deepcopy(response)
        filtered.pop("step_summary", None)
        observation = filtered.get("observation")
        if observation is not None:
            observation = self._require_backend_success(
                self._backend_agent_observation(),
                "canonical reset observation",
                required_keys=self._observation_required_keys(),
            )
            filtered["observation"] = self._task_observation(observation)
        available_actions = filtered.get("available_actions")
        if isinstance(available_actions, dict):
            filtered["available_actions"] = self._task_action_space(available_actions)
        return sanitize_agent_action_response(
            filtered,
            allowed_actions=self.allowed_dc_twin_agent_actions,
        )

    def _task_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(observation, dict):
            return observation
        filtered = deepcopy(observation)
        http_status = filtered.get("http_status")
        if (
            isinstance(http_status, int)
            and not isinstance(http_status, bool)
            and http_status >= 400
        ) or "error" in filtered:
            # Observation failures use the action-response allowlist because it
            # explicitly retains the safe ``http_status``/``error`` envelope.
            # This lets the evaluator classify rejected requests without
            # exposing any evaluator-only fields included by a backend error.
            return sanitize_agent_action_response(
                filtered,
                allowed_actions=self.allowed_dc_twin_agent_actions,
            )
        available_actions = filtered.get("available_actions")
        if isinstance(available_actions, dict):
            filtered["available_actions"] = self._task_action_space(available_actions)
        return sanitize_agent_observation(filtered)

    def _task_action_space(self, action_space: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(action_space, dict):
            return action_space

        filtered = deepcopy(action_space)
        executable_agent_actions = self.allowed_dc_twin_agent_actions
        original_agent_actions = filtered.get("agent_actions")
        if executable_agent_actions is not None and isinstance(
            original_agent_actions, list
        ):
            filtered["agent_actions"] = [
                action
                for action in original_agent_actions
                if action in executable_agent_actions
            ]

        if executable_agent_actions is not None:
            self._filter_action_group(
                filtered, "read_actions", executable_agent_actions
            )
            self._filter_action_group(
                filtered, "time_actions", executable_agent_actions
            )
            self._filter_action_group(
                filtered, "control_actions", executable_agent_actions
            )

        control_actions = filtered.get("control_actions")
        disabled_control_actions: list[str] = []
        if executable_agent_actions is not None:
            disabled_control_actions = sorted(
                self.WRITE_AGENT_ACTIONS - set(executable_agent_actions)
            )
        if executable_agent_actions is not None and isinstance(control_actions, dict):
            disabled_control_actions = sorted(
                set(control_actions)
                | (set(original_agent_actions or []) & self.WRITE_AGENT_ACTIONS)
            )

        executable_task_actions = list(self.DATA_CENTER_TWIN_READ_ACTION_DOCS)
        if self.EXPOSE_DC_TWIN_ACTION:
            executable_task_actions.extend(self.DATA_CENTER_TWIN_WRITE_ACTION_DOCS)
        executable_global_agent_actions = set(filtered.get("agent_actions", []))
        filtered["task_action_scope"] = {
            "write_actions_enabled": bool(
                executable_global_agent_actions & self.WRITE_AGENT_ACTIONS
            ),
            "agent_action_invocation": "dc_twin_action",
            "executable_agent_actions": list(filtered.get("agent_actions", [])),
            "executable_task_actions": executable_task_actions,
            "disabled_control_actions": disabled_control_actions,
            "non_executable_global_agent_actions": [
                action
                for action in original_agent_actions or []
                if action not in set(filtered.get("agent_actions", []))
            ],
        }
        return sanitize_agent_action_space(
            filtered,
            allowed_actions=self.allowed_dc_twin_agent_actions,
        )

    def _filter_action_group(
        self,
        action_space: dict[str, Any],
        group_name: str,
        executable_agent_actions: frozenset[str],
    ) -> None:
        actions = action_space.get(group_name)
        if isinstance(actions, dict):
            action_space[group_name] = {
                action: details
                for action, details in actions.items()
                if action in executable_agent_actions
            }

    def _filter_incident_domain_contract(
        self,
        contract: dict[str, Any],
        filtered_agent_actions: set[str],
    ) -> dict[str, Any]:
        if not isinstance(contract, dict):
            return contract
        filtered_contract = deepcopy(contract)
        response_actions = filtered_contract.get("agent_response_actions")
        if isinstance(response_actions, list):
            filtered_contract["agent_response_actions"] = [
                action
                for action in response_actions
                if action in filtered_agent_actions
            ]
        return filtered_contract

    def _all_response_actions_are_agent_actions(
        self,
        incident_domains: dict[str, Any],
        filtered_agent_actions: set[str],
    ) -> bool:
        if not isinstance(incident_domains, dict):
            return True
        return all(
            action in filtered_agent_actions
            for details in incident_domains.values()
            if isinstance(details, dict)
            for action in details.get("agent_response_actions", [])
        )

    def _record_agent_interface_call(
        self, action_name: str, request: dict[str, Any], response: dict[str, Any]
    ):
        self.agent_action_history.append(
            {
                "action_name": action_name,
                "request": request,
                "response": response,
            }
        )

    def _record_agent_rendered_observation_for_evaluation(
        self,
        observation: dict[str, Any],
    ) -> None:
        """Record exactly one compact payload delivered to the agent.

        The runner invokes this only after sanitization and deterministic
        rendering.  Rejecting other schemas prevents a caller from placing a
        complete canonical snapshot in this evaluator-only visibility path.
        """

        if not isinstance(observation, dict) or observation.get(
            "schema_version"
        ) not in {
            "agent.telemetry.compact.v1",
            "agent.telemetry.delta.v1",
        }:
            raise ValueError(
                "evaluator visibility history accepts only rendered compact telemetry"
            )
        self._agent_rendering_boundary_active = True
        self._agent_rendered_observation_history.append(deepcopy(observation))

    def _activate_agent_rendering_boundary(self) -> None:
        """Prevent fallback to hidden canonical history for compact episodes."""

        self._agent_rendering_boundary_active = True

    def _evaluate_semantic_diagnosis(self, soln: Any, duration: float) -> None:
        """Score only submitted diagnostics against the agent's visibility boundary.

        This method runs after the agent has stopped. Its privileged inputs and
        output are retained solely in self.results, never in a tool response.
        Every diagnostic task uses semantic adjudication.
        """
        task_type = self.scenario.task_type
        evaluation_input = make_input(
            task_type=task_type,
            final_answer=soln,
            observation_history=self._agent_visible_observations(),
            ground_truth=scoring_ground_truth(self.scenario),
        )
        try:
            from .semantic_evaluation import SemanticEvaluator, SemanticEvaluatorConfig

            judge = getattr(self, "semantic_evaluator", None) or SemanticEvaluator(
                SemanticEvaluatorConfig()
            )
            audit = judge.adjudicate(evaluation_input)
        except SemanticEvaluatorError as error:
            self.results.update(
                success=None,
                evaluation_status="evaluator_infrastructure_error",
                diagnostic_evaluator="semantic",
                semantic_adjudication=error.audit,
                evaluation_error=error.audit["error"],
                scenario_id=self.scenario.problem_id,
            )
            return
        output = audit["output"]
        correct = output["success"]
        accuracy_key = {
            "detection": "Detection Accuracy",
            "localization": "Localization Accuracy",
            "analysis": "Analysis Accuracy",
        }[task_type]
        self.results.update(
            success=correct,
            evaluation_status="scored",
            diagnostic_evaluator="semantic",
            semantic_adjudication=audit,
            scenario_id=self.scenario.problem_id,
            **{
                key: value
                for key, value in output.items()
                if key not in {"supporting_evidence", "success"}
            },
        )
        self.results[f"{task_type}_correct"] = correct
        self.results[accuracy_key] = (
            (100.0 if correct else 0.0)
            if task_type == "localization"
            else ("Correct" if correct else "Incorrect")
        )
        if task_type == "detection":
            self.results["time_to_detection_seconds"] = duration

    def _scenario_fault_from_active_faults(
        self, active_faults: Any
    ) -> dict[str, Any] | None:
        if not isinstance(active_faults, list):
            return None
        for fault in active_faults:
            if (
                isinstance(fault, dict)
                and fault.get("fault_type") == self.fault_type
                and fault.get("target") == self.faulty_component
            ):
                return fault
        return None

    def _agent_visible_observations(self) -> list[dict[str, Any]]:
        return deepcopy(self._agent_rendered_observation_history)

    def _required_control_actions_satisfied(self) -> bool:
        required_actions = set(
            self.success_criteria.get("required_control_actions") or []
        )
        if not required_actions:
            return True

        applied_actions = set()
        for call in self.agent_action_history:
            if (
                not isinstance(call, dict)
                or call.get("action_name") != "dc_twin_action"
            ):
                continue
            request = call.get("request") or {}
            response = call.get("response") or {}
            action_type = request.get("action_type")
            if action_type not in self.WRITE_AGENT_ACTIONS:
                continue
            if not self._request_matches_declared_mitigation_action(request):
                continue
            if not self._successful_agent_action_response(response):
                continue
            if not self._control_action_started_during_scenario_fault(response):
                continue
            applied_actions.add(action_type)
        return required_actions <= applied_actions

    def _request_matches_declared_mitigation_action(self, request: Any) -> bool:
        """Accept any simulator-validated control of a declared response type.

        Success is determined independently from evaluator-only postconditions
        while the scenario fault is still active. Requiring the exact hidden
        parameter tuple would unfairly reject alternative effective controls.
        """
        if not isinstance(request, dict):
            return False

        request_action_type = request.get("action_type")
        for mitigation_action in self._declared_mitigation_actions():
            if mitigation_action.get("action_type") == request_action_type:
                return True
        return False

    def _declared_mitigation_actions(self) -> list[dict[str, Any]]:
        mitigation_actions = self.success_criteria.get("mitigation_actions")
        if mitigation_actions is None:
            mitigation_action = self.success_criteria.get("mitigation_action")
            mitigation_actions = (
                [mitigation_action] if mitigation_action is not None else []
            )
        if not isinstance(mitigation_actions, list):
            return []
        return [action for action in mitigation_actions if isinstance(action, dict)]

    def _successful_agent_action_response(self, response: Any) -> bool:
        if not isinstance(response, dict):
            return False
        http_status = response.get("http_status")
        if isinstance(http_status, int) and http_status >= 400:
            return False
        return "error" not in response and "raw" not in response

    def _control_action_started_during_scenario_fault(
        self, response: dict[str, Any]
    ) -> bool:
        action_start = response.get("sim_time_seconds_before")
        if not isinstance(action_start, (int, float)) or isinstance(action_start, bool):
            return False

        scenario_fault = self._scenario_fault_from_active_faults(
            response.get("active_faults_before")
        )
        if scenario_fault is not None:
            fault_start = scenario_fault.get(
                "started_at_sim_time_seconds", self.fault_injection_time
            )
            fault_duration = scenario_fault.get(
                "duration_seconds", self.fault_duration_seconds
            )
        else:
            fault_start = self.fault_injection_time
            fault_duration = self.fault_duration_seconds
        if not isinstance(fault_start, (int, float)) or isinstance(fault_start, bool):
            return False
        if not isinstance(fault_duration, (int, float)) or isinstance(
            fault_duration, bool
        ):
            return False

        return action_start < fault_start + fault_duration

    def _is_mitigation_episode(self) -> bool:
        return getattr(self.scenario, "task_type", None) == "mitigation"

    def _read_evaluator_summary(self) -> dict[str, Any]:
        """Read the privileged evaluator projection without exposing it."""
        evaluator_state = self._require_backend_success(
            self.app.evaluator_state(log_limit=0, include_config=False),
            "evaluator state",
            required_keys=("summary",),
        )
        evaluator_summary = evaluator_state["summary"]
        if not isinstance(evaluator_summary, dict):
            raise DataCenterTwinSetupError(
                "Data Center Twin evaluator state summary must be an object."
            )
        return evaluator_summary

    def _capture_pending_evaluator_tick_samples(
        self,
        source: str,
    ) -> list[dict[str, Any]]:
        """Consume privileged per-tick summaries when the backend provides them."""
        consumer = getattr(self.app, "consume_evaluator_tick_summaries", None)
        if not callable(consumer):
            return []
        summaries = consumer()
        if not isinstance(summaries, list):
            raise DataCenterTwinSetupError(
                "Data Center Twin evaluator tick history must be a list."
            )
        captured: list[dict[str, Any]] = []
        for tick_index, summary in enumerate(summaries, start=1):
            if not isinstance(summary, dict):
                raise DataCenterTwinSetupError(
                    "Data Center Twin evaluator tick summary must be an object."
                )
            self._capture_evaluator_health_sample(
                f"{source}:tick:{tick_index}",
                summary,
            )
            captured.append(summary)
        return captured

    def _capture_evaluator_health_sample(
        self,
        source: str,
        evaluator_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Append one evaluation-only health sample for mitigation metrics.

        This method is never called by an agent-visible API response builder;
        the returned sample remains solely in the evaluator result artifact.
        """
        if not self._is_mitigation_episode():
            return None
        summary = (
            evaluator_summary
            if evaluator_summary is not None
            else self._read_evaluator_summary()
        )
        sim_time = summary.get("sim_time_seconds")
        if isinstance(sim_time, bool) or not isinstance(sim_time, (int, float)):
            raise DataCenterTwinSetupError(
                "Data Center Twin evaluator summary requires numeric sim_time_seconds."
            )

        constraints = self._health_constraints(summary)
        health_condition = all(constraint["healthy"] for constraint in constraints)
        required_actions_satisfied = self._required_control_actions_satisfied()
        within_fault_horizon = self._within_scenario_fault_horizon(summary)
        fault_time = self.fault_injection_time
        after_fault = (
            isinstance(fault_time, (int, float))
            and not isinstance(fault_time, bool)
            and float(sim_time) >= float(fault_time)
        )
        sample = {
            "sequence": len(self.evaluator_health_trajectory),
            "source": source,
            "simulator_time": float(sim_time),
            "evaluator_time": float(sim_time),
            "after_fault_injection": bool(after_fault),
            "within_episode_horizon": bool(
                self.episode_horizon_time is not None
                and float(sim_time) <= float(self.episode_horizon_time)
            ),
            "constraints": constraints,
            "health_condition": health_condition,
            "all_health_constraints_satisfied": health_condition,
            "evaluator_state": {
                "summary": deepcopy(summary),
                "required_control_actions": list(
                    self.success_criteria.get("required_control_actions") or []
                ),
                "required_control_actions_satisfied": required_actions_satisfied,
                "within_fault_horizon": within_fault_horizon,
                "task_success_condition": bool(
                    required_actions_satisfied
                    and within_fault_horizon
                    and health_condition
                ),
            },
            "stable_recovery_achieved": False,
        }
        self.evaluator_health_trajectory.append(sample)
        self._mitigation_metrics_cache = None
        return sample

    @staticmethod
    def _constraint_with_name(constraint: dict[str, Any]) -> dict[str, Any]:
        constraint["name"] = constraint["identifier"]
        constraint["bound"] = constraint["direction"]
        return constraint

    def _numeric_health_constraint(
        self,
        summary: dict[str, Any],
        criterion_name: str,
        summary_field: str,
        direction: str,
    ) -> dict[str, Any]:
        return self._constraint_with_name(
            make_constraint(
                criterion_name,
                summary.get(summary_field),
                self.success_criteria[criterion_name],
                direction,
                source_field=summary_field,
                raw_observed_value=summary.get(summary_field),
            )
        )

    def _predicate_health_constraint(
        self,
        identifier: str,
        healthy: bool,
        *,
        source_field: str,
        raw_observed_value: Any,
    ) -> dict[str, Any]:
        return self._constraint_with_name(
            make_constraint(
                identifier,
                1.0 if healthy else 0.0,
                1.0,
                "lower_bound",
                source_field=source_field,
                raw_observed_value=deepcopy(raw_observed_value),
            )
        )

    def _health_constraints(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        """Materialize the existing scenario health predicates for auditing."""
        constraints: list[dict[str, Any]] = []
        expected_sla = self.success_criteria.get("sla_status")
        if expected_sla is not None:
            actual_sla = summary.get("sla_status")
            constraints.append(
                self._predicate_health_constraint(
                    "sla_status",
                    actual_sla == expected_sla,
                    source_field="sla_status",
                    raw_observed_value={
                        "observed": actual_sla,
                        "expected": expected_sla,
                    },
                )
            )

        for criterion_name, summary_field in self.SUMMARY_MAX_CRITERIA.items():
            if criterion_name in self.success_criteria:
                constraints.append(
                    self._numeric_health_constraint(
                        summary,
                        criterion_name,
                        summary_field,
                        "upper_bound",
                    )
                )
        for criterion_name, summary_field in self.SUMMARY_MIN_CRITERIA.items():
            if criterion_name in self.success_criteria:
                constraints.append(
                    self._numeric_health_constraint(
                        summary,
                        criterion_name,
                        summary_field,
                        "lower_bound",
                    )
                )

        allocated_server_ids = list(summary.get("workload_allocated_server_ids") or [])
        current_demand = summary.get("workload_current_demand_per_second", 0.0)
        has_required_allocation = bool(allocated_server_ids) or current_demand <= 0.0

        avoided_racks = self.success_criteria.get("workload_allocated_away_from_rack")
        if avoided_racks is not None:
            avoided_rack_ids = (
                [avoided_racks]
                if isinstance(avoided_racks, str)
                else list(avoided_racks)
            )
            healthy = has_required_allocation and not any(
                self._server_ids_include_rack(allocated_server_ids, rack_id)
                for rack_id in avoided_rack_ids
            )
            constraints.append(
                self._predicate_health_constraint(
                    "workload_allocated_away_from_rack",
                    healthy,
                    source_field="workload_allocated_server_ids",
                    raw_observed_value={
                        "allocated_server_ids": allocated_server_ids,
                        "avoided_rack_ids": avoided_rack_ids,
                        "current_demand_per_second": current_demand,
                    },
                )
            )

        avoided_servers = self.success_criteria.get(
            "workload_allocated_away_from_server"
        )
        if avoided_servers is not None:
            avoided_server_ids = (
                [avoided_servers]
                if isinstance(avoided_servers, str)
                else list(avoided_servers)
            )
            healthy = has_required_allocation and not (
                set(allocated_server_ids) & set(avoided_server_ids)
            )
            constraints.append(
                self._predicate_health_constraint(
                    "workload_allocated_away_from_server",
                    healthy,
                    source_field="workload_allocated_server_ids",
                    raw_observed_value={
                        "allocated_server_ids": allocated_server_ids,
                        "avoided_server_ids": avoided_server_ids,
                        "current_demand_per_second": current_demand,
                    },
                )
            )

        required_rack = self.success_criteria.get("workload_allocated_to_rack")
        if required_rack is not None:
            constraints.append(
                self._predicate_health_constraint(
                    "workload_allocated_to_rack",
                    self._server_ids_include_rack(allocated_server_ids, required_rack),
                    source_field="workload_allocated_server_ids",
                    raw_observed_value={
                        "allocated_server_ids": allocated_server_ids,
                        "required_rack_id": required_rack,
                    },
                )
            )
        return constraints

    def _health_condition(self, summary: dict[str, Any]) -> bool:
        return all(
            constraint["healthy"] for constraint in self._health_constraints(summary)
        )

    def finalize_mitigation_metrics(self) -> dict[str, Any]:
        """Finalize reproducible mitigation metrics without advancing time."""
        if not self._is_mitigation_episode():
            return {}
        if self._mitigation_metrics_cache is not None:
            return deepcopy(self._mitigation_metrics_cache)
        if self.fault_injection_time is None or self.episode_horizon_time is None:
            raise DataCenterTwinSetupError(
                "Mitigation metrics require fault-injection time and episode horizon."
            )

        recovery = compute_stable_recovery(
            self.evaluator_health_trajectory,
            fault_injection_time=self.fault_injection_time,
            episode_horizon=self.episode_horizon_time,
            stability_window=self.stability_window_seconds,
        )
        self.evaluator_health_trajectory = annotate_stable_recovery(
            self.evaluator_health_trajectory,
            recovery,
            self.stability_window_seconds,
        )
        slo = compute_slo_metrics(
            self.evaluator_health_trajectory,
            fault_injection_time=self.fault_injection_time,
            episode_horizon=self.episode_horizon_time,
            epsilon=SLO_NORMALIZATION_EPSILON,
        )
        metric_config = {
            "normalization_epsilon": SLO_NORMALIZATION_EPSILON,
            "integration_rule": slo["integration_rule"],
            "sample_intervals": "actual_simulator_time_deltas",
            "interval_clipping": "post_fault_to_episode_horizon",
            "last_sample_extrapolation": False,
            "raw_violation_duration": "union_of_intervals_with_any_violated_constraint",
            "constraint_normalization": "mean_over_scenario_health_constraints",
            "sampling": "privileged_simulator_ticks_and_evaluator_health_checks",
            "multi_tick_action_sampling": (
                "per_logical_simulator_tick"
                if callable(getattr(self.app, "consume_evaluator_tick_summaries", None))
                else "endpoint_only_backend_fallback"
            ),
            "episode_horizon_definition": (
                "fault_injection_time_plus_scenario_fault_duration_seconds"
            ),
            "stable_recovery_definition": (
                "earliest_post_fault_healthy_start_with_full_continuously_healthy_window"
            ),
            "failed_recovery_penalty": "episode_horizon_minus_fault_injection_time",
        }
        finalized = {
            "health_trajectory": deepcopy(self.evaluator_health_trajectory),
            "fault_injection_time": float(self.fault_injection_time),
            "episode_horizon": float(self.episode_horizon_time),
            "stability_window_length": float(self.stability_window_seconds),
            "slo_violation_area": slo["slo_violation_area"],
            "raw_slo_violation_duration": slo["raw_slo_violation_duration"],
            "slo_constraint_count": slo["constraint_count"],
            "slo_integrated_post_fault_time": slo["integrated_post_fault_time"],
            "mitigation_metric_config": metric_config,
            **recovery,
        }
        self._mitigation_metrics_cache = finalized
        return deepcopy(finalized)

    SUMMARY_MAX_CRITERIA = {
        "thermal_critical_max": "thermal_critical",
        "power_overloaded_racks_max": "power_overloaded_racks",
        "failed_servers_max": "failed_servers",
        "workload_queue_length_max": "workload_queue_length",
        "workload_network_congestion_ratio_max": "workload_network_congestion_ratio",
        "workload_average_latency_ms_max": "workload_average_latency_ms",
        "workload_p95_latency_ms_max": "workload_p95_latency_ms",
        "workload_network_packet_loss_percent_max": "workload_network_packet_loss_percent",
        "workload_network_retransmit_rate_max": "workload_network_retransmit_rate",
        "workload_network_error_rate_max": "workload_network_error_rate",
        "max_network_packet_loss_percent_max": "max_network_packet_loss_percent",
        "network_retransmit_rate_max": "network_retransmit_rate",
        "network_error_rate_max": "network_error_rate",
        "telemetry_lag_seconds_max": "telemetry_lag_seconds",
        "metrics_missing_ratio_max": "metrics_missing_ratio",
        "logs_missing_ratio_max": "logs_missing_ratio",
        "placement_policy_violating_racks_max": "placement_policy_violating_racks",
        "workload_placement_imbalance_ratio_max": "workload_placement_imbalance_ratio",
        "load_balancer_backend_skew_ratio_max": "load_balancer_backend_skew_ratio",
        "load_balancer_unhealthy_routing_fraction_max": "load_balancer_unhealthy_routing_fraction",
        "load_balancer_error_rate_percent_max": "load_balancer_error_rate_percent",
        "workload_storage_utilization_ratio_max": "workload_storage_utilization_ratio",
        "workload_storage_latency_penalty_ms_max": "workload_storage_latency_penalty_ms",
        "workload_application_error_rate_percent_max": "workload_application_error_rate_percent",
        "workload_dropped_requests_per_second_max": "workload_dropped_requests_per_second",
        "workload_current_demand_per_second_max": "workload_current_demand_per_second",
        "power_budget_violating_racks_max": "power_budget_violating_racks",
        "max_power_budget_utilization_ratio_max": "max_power_budget_utilization_ratio",
        "temperature_sensor_unhealthy_count_max": "temperature_sensor_unhealthy_count",
        "max_temperature_sensor_disagreement_c_max": "max_temperature_sensor_disagreement_c",
        "host_health_flapping_count_max": "host_health_flapping_count",
        "thermal_throttled_servers_max": "thermal_throttled_servers",
    }
    SUMMARY_MIN_CRITERIA = {
        "min_thermal_throttle_factor_min": "min_thermal_throttle_factor",
        "workload_service_capacity_requests_per_second_min": "workload_service_capacity_requests_per_second",
        "autoscaler_effective_server_limit_min": "autoscaler_effective_server_limit",
    }

    def _summary_satisfies_success_criteria(self, summary: dict[str, Any]) -> bool:
        if not self._required_control_actions_satisfied():
            return False
        if not self._within_scenario_fault_horizon(summary):
            return False
        return self._health_condition(summary)

    def _within_scenario_fault_horizon(self, summary: dict[str, Any]) -> bool:
        """Prevent natural fault expiry from being scored as agent mitigation."""
        if not self.success_criteria.get("required_control_actions"):
            return True
        fault_start = self.fault_injection_time
        fault_duration = self.fault_duration_seconds
        sim_time = summary.get("sim_time_seconds")
        numeric_values = (fault_start, fault_duration, sim_time)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in numeric_values
        ):
            return False
        return sim_time < fault_start + fault_duration

    def _advance_evaluator_tick(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Advance through the agent API and read raw state only for scoring.

        The returned action and visible summary are safe for benchmark
        artifacts. The evaluator summary is privileged and must remain local to
        the scoring call.
        """
        action_response = self._require_backend_success(
            self.app.agent_action(
                "noop",
                advance_ticks=1,
                log_limit=20,
                include_config=True,
                host_visibility=self.BENCHMARK_HOST_VISIBILITY,
            ),
            "evaluator stability action",
            required_keys=("observation",),
        )
        visible_action = self._task_action_response(action_response)
        visible_observation = visible_action.get("observation") or {}
        self.latest_agent_observation = self._task_observation(visible_observation)
        visible_summary = (self.latest_agent_observation or {}).get("summary", {})

        captured = self._capture_pending_evaluator_tick_samples(
            "evaluation_stability_tick"
        )
        if captured:
            evaluator_summary = captured[-1]
        else:
            evaluator_summary = self._read_evaluator_summary()
            self._capture_evaluator_health_sample(
                "evaluation_stability_tick",
                evaluator_summary,
            )
        return visible_action, visible_summary, evaluator_summary

    def _evaluate_mitigation_stability(
        self,
    ) -> tuple[bool, dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Run the existing evaluator stability loop and finalize metrics."""
        self._capture_evaluator_health_sample("evaluation_start")
        stable = True
        final_summary: dict[str, Any] = {}
        final_agent_action: dict[str, Any] = {}
        for _ in range(self.stability_window_seconds):
            final_agent_action, final_summary, evaluator_summary = (
                self._advance_evaluator_tick()
            )
            if not self._summary_satisfies_success_criteria(evaluator_summary):
                stable = False
        metrics = self.finalize_mitigation_metrics()
        return stable, final_summary, final_agent_action, metrics

    def _record_mitigation_evaluation_results(
        self,
        stable: bool,
        final_summary: dict[str, Any],
        final_agent_action: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        self.add_result("Mitigation Success", stable)
        self.add_result("mitigation_success", stable)
        self.add_result(
            "stability_qualified_success",
            bool(stable and metrics.get("stable_recovery_succeeded") is True),
        )
        self.add_result("scenario_id", self.scenario.problem_id)
        self.add_result("final_summary", final_summary)
        self.add_result("final_agent_action", final_agent_action)
        self.add_result("agent_action_space", self.agent_action_space)
        for key, value in metrics.items():
            self.add_result(key, value)
        self.results["success"] = stable

    def _server_ids_include_rack(self, server_ids: list[Any], rack_id: str) -> bool:
        server_prefix = self._server_id_prefix_for_rack(rack_id)
        if server_prefix is None:
            return False
        return any(
            isinstance(server_id, str) and server_id.startswith(server_prefix)
            for server_id in server_ids
        )

    def _server_id_prefix_for_rack(self, rack_id: str) -> str | None:
        parts = rack_id.split("-")
        if len(parts) != 4 or parts[0] != "rack":
            return None
        return f"server-{parts[1]}-{parts[2]}-rack{parts[3]}-"


class DataCenterTwinCoolingDegradationDetection(DataCenterTwinBaseTask, DetectionTask):
    SCENARIO_ID = "data_center_twin-cooling_degradation-detection-1"

    def __init__(self):
        DataCenterTwinBaseTask.__init__(self)
        DetectionTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        self._evaluate_semantic_diagnosis(soln, duration)
        return super().eval(soln, trace, duration)


class DataCenterTwinCoolingDegradationLocalization(
    DataCenterTwinBaseTask, LocalizationTask
):
    SCENARIO_ID = "data_center_twin-cooling_degradation-localization-1"

    def __init__(self):
        DataCenterTwinBaseTask.__init__(self)
        LocalizationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        self._evaluate_semantic_diagnosis(soln, duration)
        return super().eval(soln, trace, duration)


class DataCenterTwinCoolingDegradationAnalysis(DataCenterTwinBaseTask, AnalysisTask):
    SCENARIO_ID = "data_center_twin-cooling_degradation-analysis-1"

    def __init__(self):
        DataCenterTwinBaseTask.__init__(self)
        AnalysisTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        self._evaluate_semantic_diagnosis(soln, duration)
        return super().eval(soln, trace, duration)


class DataCenterTwinCoolingDegradationMitigation(
    DataCenterTwinBaseTask, MitigationTask
):
    SCENARIO_ID = "data_center_twin-cooling_degradation-mitigation-1"
    EXPOSE_DC_TWIN_ACTION = True
    ALLOWED_DC_TWIN_AGENT_ACTIONS = None

    def __init__(self):
        DataCenterTwinBaseTask.__init__(self)
        MitigationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        stable, final_summary, final_agent_action, metrics = (
            self._evaluate_mitigation_stability()
        )
        self._record_mitigation_evaluation_results(
            stable,
            final_summary,
            final_agent_action,
            metrics,
        )
        return super().eval(soln, trace, duration)


class DataCenterTwinGenericMitigation(DataCenterTwinBaseTask, MitigationTask):
    EXPOSE_DC_TWIN_ACTION = True
    ALLOWED_DC_TWIN_AGENT_ACTIONS = None

    def __init__(self):
        DataCenterTwinBaseTask.__init__(self)
        MitigationTask.__init__(self, self.app)

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        stable, final_summary, final_agent_action, metrics = (
            self._evaluate_mitigation_stability()
        )
        self._record_mitigation_evaluation_results(
            stable,
            final_summary,
            final_agent_action,
            metrics,
        )
        self.add_result("fault_type", self.fault_type)
        self.add_result("fault_target", self.faulty_component)
        self.add_result("success_criteria", self.success_criteria)
        return MitigationTask.eval(self, soln, trace, duration)


class DataCenterTwinRackHotspotDetection(DataCenterTwinCoolingDegradationDetection):
    SCENARIO_ID = "data_center_twin-rack_hotspot-detection-1"


class DataCenterTwinRackHotspotLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-rack_hotspot-localization-1"


class DataCenterTwinRackHotspotAnalysis(DataCenterTwinCoolingDegradationAnalysis):
    SCENARIO_ID = "data_center_twin-rack_hotspot-analysis-1"


class DataCenterTwinRackHotspotMitigation(DataCenterTwinGenericMitigation):
    SCENARIO_ID = "data_center_twin-rack_hotspot-mitigation-1"


class DataCenterTwinPowerOverloadDetection(DataCenterTwinCoolingDegradationDetection):
    SCENARIO_ID = "data_center_twin-power_overload-detection-1"


class DataCenterTwinPowerOverloadLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-power_overload-localization-1"


class DataCenterTwinPowerOverloadAnalysis(DataCenterTwinCoolingDegradationAnalysis):
    SCENARIO_ID = "data_center_twin-power_overload-analysis-1"


class DataCenterTwinPowerOverloadMitigation(DataCenterTwinGenericMitigation):
    SCENARIO_ID = "data_center_twin-power_overload-mitigation-1"


class DataCenterTwinServerFailureDetection(DataCenterTwinCoolingDegradationDetection):
    SCENARIO_ID = "data_center_twin-server_failure-detection-1"


class DataCenterTwinServerFailureLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-server_failure-localization-1"


class DataCenterTwinServerFailureAnalysis(DataCenterTwinCoolingDegradationAnalysis):
    SCENARIO_ID = "data_center_twin-server_failure-analysis-1"


class DataCenterTwinServerFailureMitigation(DataCenterTwinGenericMitigation):
    SCENARIO_ID = "data_center_twin-server_failure-mitigation-1"


class DataCenterTwinStorageIoSaturationDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-storage_io_saturation-detection-1"


class DataCenterTwinStorageIoSaturationLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-storage_io_saturation-localization-1"


class DataCenterTwinStorageIoSaturationAnalysis(
    DataCenterTwinCoolingDegradationAnalysis
):
    SCENARIO_ID = "data_center_twin-storage_io_saturation-analysis-1"


class DataCenterTwinStorageIoSaturationMitigation(DataCenterTwinGenericMitigation):
    SCENARIO_ID = "data_center_twin-storage_io_saturation-mitigation-1"


class DataCenterTwinControlPlaneDegradationDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-control_plane_degradation-detection-1"


class DataCenterTwinControlPlaneDegradationLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-control_plane_degradation-localization-1"


class DataCenterTwinControlPlaneDegradationAnalysis(
    DataCenterTwinCoolingDegradationAnalysis
):
    SCENARIO_ID = "data_center_twin-control_plane_degradation-analysis-1"


class DataCenterTwinControlPlaneDegradationMitigation(DataCenterTwinGenericMitigation):
    SCENARIO_ID = "data_center_twin-control_plane_degradation-mitigation-1"


class DataCenterTwinAutoscalerMisconfigurationDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-autoscaler_misconfiguration-detection-1"


class DataCenterTwinAutoscalerMisconfigurationLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-autoscaler_misconfiguration-localization-1"


class DataCenterTwinAutoscalerMisconfigurationAnalysis(
    DataCenterTwinCoolingDegradationAnalysis
):
    SCENARIO_ID = "data_center_twin-autoscaler_misconfiguration-analysis-1"


class DataCenterTwinAutoscalerMisconfigurationMitigation(
    DataCenterTwinGenericMitigation
):
    SCENARIO_ID = "data_center_twin-autoscaler_misconfiguration-mitigation-1"


class DataCenterTwinMonitoringPipelineFailureDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-monitoring_pipeline_failure-detection-1"


class DataCenterTwinMonitoringPipelineFailureLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-monitoring_pipeline_failure-localization-1"


class DataCenterTwinMonitoringPipelineFailureAnalysis(
    DataCenterTwinCoolingDegradationAnalysis
):
    SCENARIO_ID = "data_center_twin-monitoring_pipeline_failure-analysis-1"


class DataCenterTwinMonitoringPipelineFailureMitigation(
    DataCenterTwinGenericMitigation
):
    SCENARIO_ID = "data_center_twin-monitoring_pipeline_failure-mitigation-1"


class DataCenterTwinPlacementPolicyMisconfigurationDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-placement_policy_misconfiguration-detection-1"


class DataCenterTwinPlacementPolicyMisconfigurationLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-placement_policy_misconfiguration-localization-1"


class DataCenterTwinPlacementPolicyMisconfigurationAnalysis(
    DataCenterTwinCoolingDegradationAnalysis
):
    SCENARIO_ID = "data_center_twin-placement_policy_misconfiguration-analysis-1"


class DataCenterTwinPlacementPolicyMisconfigurationMitigation(
    DataCenterTwinGenericMitigation
):
    SCENARIO_ID = "data_center_twin-placement_policy_misconfiguration-mitigation-1"


class DataCenterTwinLoadBalancerMisconfigurationDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-load_balancer_misconfiguration-detection-1"


class DataCenterTwinLoadBalancerMisconfigurationLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-load_balancer_misconfiguration-localization-1"


class DataCenterTwinLoadBalancerMisconfigurationAnalysis(
    DataCenterTwinCoolingDegradationAnalysis
):
    SCENARIO_ID = "data_center_twin-load_balancer_misconfiguration-analysis-1"


class DataCenterTwinLoadBalancerMisconfigurationMitigation(
    DataCenterTwinGenericMitigation
):
    SCENARIO_ID = "data_center_twin-load_balancer_misconfiguration-mitigation-1"


class DataCenterTwinApplicationErrorDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-application_error-detection-1"


class DataCenterTwinApplicationErrorLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-application_error-localization-1"


class DataCenterTwinApplicationErrorAnalysis(DataCenterTwinCoolingDegradationAnalysis):
    SCENARIO_ID = "data_center_twin-application_error-analysis-1"


class DataCenterTwinApplicationErrorMitigation(DataCenterTwinGenericMitigation):
    SCENARIO_ID = "data_center_twin-application_error-mitigation-1"


class DataCenterTwinNetworkPartitionDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-network_partition-detection-1"


class DataCenterTwinNetworkPartitionLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-network_partition-localization-1"


class DataCenterTwinNetworkPartitionAnalysis(DataCenterTwinCoolingDegradationAnalysis):
    SCENARIO_ID = "data_center_twin-network_partition-analysis-1"


class DataCenterTwinNetworkPartitionMitigation(
    DataCenterTwinCoolingDegradationMitigation
):
    SCENARIO_ID = "data_center_twin-network_partition-mitigation-1"

    def eval(self, soln: Any, trace: list[SessionItem], duration: float):
        print("== Evaluation ==")
        recovered, final_summary, final_agent_action, metrics = (
            self._evaluate_mitigation_stability()
        )
        self._record_mitigation_evaluation_results(
            recovered,
            final_summary,
            final_agent_action,
            metrics,
        )
        self.add_result("network_partition_recovered", recovered)
        self.add_result("network_partition_target_rack", self.faulty_component)
        self.add_result("network_partition_failover_rack", self.failover_rack)
        return MitigationTask.eval(self, soln, trace, duration)

    def _health_constraints(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        constraints = super()._health_constraints(summary)
        active_faults = summary.get("active_faults") or []
        partition_active = any(
            fault.get("fault_type") == "network_partition"
            and fault.get("target") == self.faulty_component
            for fault in active_faults
            if isinstance(fault, dict)
        )
        allocated_server_ids = summary.get("workload_allocated_server_ids") or []
        allocated_to_partition = self._server_ids_include_rack(
            allocated_server_ids,
            self.faulty_component,
        )
        has_serving_path = bool(allocated_server_ids) or (
            summary.get("workload_current_demand_per_second") == 0
        )
        constraints.extend(
            [
                self._predicate_health_constraint(
                    "network_serving_path_available",
                    has_serving_path,
                    source_field="workload_allocated_server_ids",
                    raw_observed_value={
                        "allocated_server_ids": allocated_server_ids,
                        "current_demand_per_second": summary.get(
                            "workload_current_demand_per_second"
                        ),
                    },
                ),
                self._predicate_health_constraint(
                    "network_partition_not_serving_workload",
                    not (partition_active and allocated_to_partition),
                    source_field="active_faults_and_workload_allocated_server_ids",
                    raw_observed_value={
                        "partition_active": partition_active,
                        "allocated_to_partition": allocated_to_partition,
                        "partition_target": self.faulty_component,
                    },
                ),
            ]
        )
        return constraints

    def _network_partition_recovered(self, summary: dict[str, Any]) -> bool:
        active_faults = summary.get("active_faults") or []
        partition_active = any(
            fault.get("fault_type") == "network_partition"
            and fault.get("target") == self.faulty_component
            for fault in active_faults
            if isinstance(fault, dict)
        )
        allocated_server_ids = summary.get("workload_allocated_server_ids") or []
        allocated_to_partition = self._server_ids_include_rack(
            allocated_server_ids, self.faulty_component
        )
        has_serving_path = (
            bool(allocated_server_ids)
            or summary.get("workload_current_demand_per_second") == 0
        )
        return (
            summary.get("sla_status") == "normal"
            and summary.get("workload_queue_length", 0)
            <= self.network_queue_success_threshold
            and summary.get("workload_network_congestion_ratio", 0.0)
            <= self.network_congestion_success_threshold
            and summary.get("workload_average_latency_ms", 0.0)
            <= self.network_latency_success_threshold_ms
            and has_serving_path
            and not (partition_active and allocated_to_partition)
        )

    def _server_ids_include_rack(self, server_ids: list[Any], rack_id: str) -> bool:
        server_prefix = self._server_id_prefix_for_rack(rack_id)
        if server_prefix is None:
            return False
        return any(
            isinstance(server_id, str) and server_id.startswith(server_prefix)
            for server_id in server_ids
        )

    def _server_id_prefix_for_rack(self, rack_id: str) -> str | None:
        parts = rack_id.split("-")
        if len(parts) != 4 or parts[0] != "rack":
            return None
        return f"server-{parts[1]}-{parts[2]}-rack{parts[3]}-"


class DataCenterTwinNetworkCongestionBurstDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-network_congestion_burst-detection-1"


class DataCenterTwinNetworkCongestionBurstLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-network_congestion_burst-localization-1"


class DataCenterTwinNetworkCongestionBurstAnalysis(
    DataCenterTwinCoolingDegradationAnalysis
):
    SCENARIO_ID = "data_center_twin-network_congestion_burst-analysis-1"


class DataCenterTwinNetworkCongestionBurstMitigation(DataCenterTwinGenericMitigation):
    SCENARIO_ID = "data_center_twin-network_congestion_burst-mitigation-1"


class DataCenterTwinTorPacketLossDetection(DataCenterTwinCoolingDegradationDetection):
    SCENARIO_ID = "data_center_twin-tor_packet_loss-detection-1"


class DataCenterTwinTorPacketLossLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-tor_packet_loss-localization-1"


class DataCenterTwinTorPacketLossAnalysis(DataCenterTwinCoolingDegradationAnalysis):
    SCENARIO_ID = "data_center_twin-tor_packet_loss-analysis-1"


class DataCenterTwinTorPacketLossMitigation(DataCenterTwinGenericMitigation):
    SCENARIO_ID = "data_center_twin-tor_packet_loss-mitigation-1"


class DataCenterTwinThermalSensorMiscalibrationDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-thermal_sensor_miscalibration-detection-1"


class DataCenterTwinThermalSensorMiscalibrationLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-thermal_sensor_miscalibration-localization-1"


class DataCenterTwinThermalSensorMiscalibrationAnalysis(
    DataCenterTwinCoolingDegradationAnalysis
):
    SCENARIO_ID = "data_center_twin-thermal_sensor_miscalibration-analysis-1"


class DataCenterTwinThermalSensorMiscalibrationMitigation(
    DataCenterTwinGenericMitigation
):
    SCENARIO_ID = "data_center_twin-thermal_sensor_miscalibration-mitigation-1"


class DataCenterTwinPowerBudgetViolationDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-power_budget_violation-detection-1"


class DataCenterTwinPowerBudgetViolationLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-power_budget_violation-localization-1"


class DataCenterTwinPowerBudgetViolationAnalysis(
    DataCenterTwinCoolingDegradationAnalysis
):
    SCENARIO_ID = "data_center_twin-power_budget_violation-analysis-1"


class DataCenterTwinPowerBudgetViolationMitigation(DataCenterTwinGenericMitigation):
    SCENARIO_ID = "data_center_twin-power_budget_violation-mitigation-1"


class DataCenterTwinIntermittentServerFailureDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-intermittent_server_failure-detection-1"


class DataCenterTwinIntermittentServerFailureLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-intermittent_server_failure-localization-1"


class DataCenterTwinIntermittentServerFailureAnalysis(
    DataCenterTwinCoolingDegradationAnalysis
):
    SCENARIO_ID = "data_center_twin-intermittent_server_failure-analysis-1"


class DataCenterTwinIntermittentServerFailureMitigation(
    DataCenterTwinGenericMitigation
):
    SCENARIO_ID = "data_center_twin-intermittent_server_failure-mitigation-1"


class DataCenterTwinThermalThrottlingDetection(
    DataCenterTwinCoolingDegradationDetection
):
    SCENARIO_ID = "data_center_twin-thermal_throttling-detection-1"


class DataCenterTwinThermalThrottlingLocalization(
    DataCenterTwinCoolingDegradationLocalization
):
    SCENARIO_ID = "data_center_twin-thermal_throttling-localization-1"


class DataCenterTwinThermalThrottlingAnalysis(DataCenterTwinCoolingDegradationAnalysis):
    SCENARIO_ID = "data_center_twin-thermal_throttling-analysis-1"


class DataCenterTwinThermalThrottlingMitigation(DataCenterTwinGenericMitigation):
    SCENARIO_ID = "data_center_twin-thermal_throttling-mitigation-1"
