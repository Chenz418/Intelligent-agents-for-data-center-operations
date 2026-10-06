"""Agent-facing environment interface for the data center twin."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dc_twin.canonical_telemetry import CHANNELS as CANONICAL_TELEMETRY_CHANNELS
from dc_twin.controls import ControlRequest, SUPPORTED_ACTIONS
from dc_twin.faults import SUPPORTED_FAULTS
from dc_twin.metric_history import MetricPoint, MetricSeries
from dc_twin.simulator import DataCenterSimulator, SimulationError

AGENT_READ_ACTIONS = {"observe"}
AGENT_TIME_ACTIONS = {"noop", "step"}
AGENT_ACTIONS = AGENT_READ_ACTIONS | AGENT_TIME_ACTIONS | SUPPORTED_ACTIONS
VISIBILITY_AGENT = "agent"
VISIBILITY_EVALUATOR = "evaluator"
VISIBILITY_DEBUG = "debug"
VISIBILITIES = {VISIBILITY_AGENT, VISIBILITY_EVALUATOR, VISIBILITY_DEBUG}
HOST_VISIBILITY_HOST = "host"
HOST_VISIBILITY_RACK = "rack"
HOST_VISIBILITY_AGGREGATE = "aggregate"
HOST_VISIBILITIES = {HOST_VISIBILITY_HOST, HOST_VISIBILITY_RACK, HOST_VISIBILITY_AGGREGATE}
CONTROL_PARAMETER_KEYS = {
    "calibration_offset_c",
    "mark_untrusted",
    "target",
    "fan_speed_percent",
    "supply_air_temperature_c",
    "source_rack_id",
    "target_rack_id",
    "tenant_id",
    "workload_fraction",
    "request_rate_per_second",
    "server_id",
    "rack_id",
    "min_capacity",
    "max_capacity",
    "target_utilization_percent",
    "cooldown_seconds",
    "placement_strategy",
    "forbidden_rack_ids",
    "max_server_count",
    "backend_weights",
    "remove_backend_ids",
    "add_backend_ids",
    "routing_policy",
    "reset_to_equal_weights",
}
ACTION_PARAMETER_KEYS = {
    "observe": set(),
    "noop": set(),
    "step": {"ticks"},
    "calibrate_sensor": {"target", "calibration_offset_c", "mark_untrusted"},
    "set_cooling": {"target", "fan_speed_percent", "supply_air_temperature_c"},
    "migrate_workload": {"source_rack_id", "target_rack_id", "tenant_id", "workload_fraction"},
    "throttle_workload": {"tenant_id", "request_rate_per_second"},
    "update_autoscaler_policy": {
        "min_capacity",
        "max_capacity",
        "target_utilization_percent",
        "cooldown_seconds",
    },
    "repair_monitoring_pipeline": {"target"},
    "update_placement_policy": {
        "placement_strategy",
        "target_rack_id",
        "forbidden_rack_ids",
        "max_server_count",
    },
    "update_load_balancer_config": {
        "backend_weights",
        "remove_backend_ids",
        "add_backend_ids",
        "routing_policy",
        "reset_to_equal_weights",
    },
    "set_server_maintenance": {"server_id", "rack_id"},
    "clear_server_maintenance": {"server_id", "rack_id"},
}
TOP_LEVEL_PARAMETER_KEYS = CONTROL_PARAMETER_KEYS | {"ticks", "advance_ticks"}
ADVANCING_ACTIONS = SUPPORTED_ACTIONS | {"noop"}
SCORED_BENCHMARK_ACTIONS = {
    "calibrate_sensor",
    "set_cooling",
    "migrate_workload",
    "throttle_workload",
    "update_autoscaler_policy",
    "repair_monitoring_pipeline",
    "update_placement_policy",
    "update_load_balancer_config",
    "set_server_maintenance",
}
NON_SCORED_BENCHMARK_ACTIONS = {
    "clear_server_maintenance": (
        "Supported recovery action for operator-set maintenance state only; not scored as an incident mitigation "
        "and does not repair failed hardware or remove active faults."
    ),
}
INCIDENT_DOMAIN_CONTRACT = {
    "cooling_degradation": {
        "domain": "facility_thermal",
        "target_type": "cooling_unit",
        "observable_fields": [
            "cooling_units",
            "thermal",
            "thermal_throttled_servers",
            "min_thermal_throttle_factor",
            "workload",
            "workload_service_capacity_requests_per_second",
            "facility",
            "alerts",
        ],
        "agent_response_actions": ["set_cooling", "migrate_workload", "observe"],
    },
    "rack_hotspot": {
        "domain": "thermal",
        "target_type": "rack",
        "observable_fields": ["racks", "thermal", "workload", "alerts"],
        "agent_response_actions": ["migrate_workload", "set_cooling", "observe"],
    },
    "power_overload": {
        "domain": "power",
        "target_type": "rack",
        "observable_fields": ["racks", "power", "workload", "alerts"],
        "agent_response_actions": ["migrate_workload", "throttle_workload", "observe"],
    },
    "server_failure": {
        "domain": "hardware",
        "target_type": "server",
        "observable_fields": ["servers", "power", "workload", "alerts"],
        "agent_response_actions": ["migrate_workload", "set_server_maintenance", "observe"],
    },
    "network_partition": {
        "domain": "network",
        "target_type": "rack",
        "observable_fields": ["workload", "racks", "tenant_summaries", "alerts"],
        "agent_response_actions": ["migrate_workload", "throttle_workload", "observe"],
    },
    "network_congestion_burst": {
        "domain": "network",
        "target_type": "workload_tenant_or_rack_path",
        "observable_fields": [
            "workload_network_congestion_ratio",
            "workload_average_latency_ms",
            "workload_p95_latency_ms",
            "workload_queue_length",
            "alerts",
        ],
        "agent_response_actions": ["throttle_workload", "migrate_workload", "observe"],
    },
    "tor_packet_loss": {
        "domain": "network",
        "target_type": "rack",
        "observable_fields": [
            "network_packet_loss_percent",
            "network_retransmit_rate",
            "network_error_rate",
            "affected_rack_id",
            "alerts",
        ],
        "agent_response_actions": ["migrate_workload", "throttle_workload", "observe"],
    },
    "storage_io_saturation": {
        "domain": "storage",
        "target_type": "storage_subsystem",
        "observable_fields": ["workload_storage_utilization_ratio", "workload_storage_latency_penalty_ms", "alerts"],
        "agent_response_actions": ["throttle_workload", "observe"],
    },
    "control_plane_degradation": {
        "domain": "software_control_plane",
        "target_type": "control_plane",
        "observable_fields": [
            "control_plane_scheduler_api_latency_ms",
            "control_plane_scheduler_pending_operations",
            "workload_service_capacity_requests_per_second",
            "workload_queue_length",
            "alerts",
        ],
        "agent_response_actions": ["throttle_workload", "observe"],
    },
    "autoscaler_misconfiguration": {
        "domain": "software_control_plane",
        "target_type": "autoscaler",
        "observable_fields": [
            "autoscaler_effective_server_limit",
            "autoscaler_target_utilization_percent",
            "workload_service_capacity_requests_per_second",
            "workload_queue_length",
            "alerts",
        ],
        "agent_response_actions": ["update_autoscaler_policy", "observe"],
    },
    "monitoring_pipeline_failure": {
        "domain": "observability",
        "target_type": "monitoring_pipeline",
        "observable_fields": [
            "metrics_last_updated_sim_time_seconds",
            "logs_last_updated_sim_time_seconds",
            "telemetry_lag_seconds",
            "metrics_missing_ratio",
            "logs_missing_ratio",
            "alerts",
        ],
        "agent_response_actions": ["repair_monitoring_pipeline", "observe"],
    },
    "placement_policy_misconfiguration": {
        "domain": "scheduling",
        "target_type": "scheduler_or_rack",
        "observable_fields": [
            "placement_policy_violating_racks",
            "workload_placement_imbalance_ratio",
            "workload_allocated_server_ids",
            "alerts",
        ],
        "agent_response_actions": ["update_placement_policy", "migrate_workload", "observe"],
    },
    "load_balancer_misconfiguration": {
        "domain": "application",
        "target_type": "application_load_balancer",
        "observable_fields": [
            "load_balancer_backend_skew_ratio",
            "load_balancer_unhealthy_routing_fraction",
            "load_balancer_error_rate_percent",
            "workload_application_error_rate_percent",
            "workload_dropped_requests_per_second",
            "workload_queue_length",
            "alerts",
        ],
        "agent_response_actions": ["update_load_balancer_config", "observe"],
    },
    "application_error": {
        "domain": "application",
        "target_type": "application_or_tenant",
        "observable_fields": ["workload_application_error_rate_percent", "workload_dropped_requests_per_second", "alerts"],
        "agent_response_actions": ["throttle_workload", "observe"],
    },
    "thermal_sensor_miscalibration": {
        "domain": "facility_thermal",
        "target_type": "rack_or_server_sensor",
        "observable_fields": [
            "max_temperature_sensor_disagreement_c",
            "temperature_sensor_unhealthy_count",
            "alerts",
        ],
        "agent_response_actions": ["calibrate_sensor", "observe"],
    },
    "power_budget_violation": {
        "domain": "power",
        "target_type": "rack",
        "observable_fields": ["power_budget_violating_racks", "max_power_budget_utilization_ratio", "alerts"],
        "agent_response_actions": ["migrate_workload", "throttle_workload", "observe"],
    },
    "intermittent_server_failure": {
        "domain": "hardware",
        "target_type": "server",
        "observable_fields": ["host_health_flapping_count", "host_health_status_change_count", "alerts", "recent_events"],
        "agent_response_actions": ["set_server_maintenance", "migrate_workload", "observe"],
    },
    "thermal_throttling": {
        "domain": "thermal",
        "target_type": "rack_or_server",
        "observable_fields": ["thermal_throttled_servers", "min_thermal_throttle_factor", "workload_queue_length", "alerts"],
        "agent_response_actions": ["set_cooling", "migrate_workload", "throttle_workload", "observe"],
    },
}
ACTION_DOMAIN_CONTRACT = {
    "observe": {
        "intent": "diagnose",
        "domains": sorted({item["domain"] for item in INCIDENT_DOMAIN_CONTRACT.values()}),
        "target_type": "datacenter",
    },
    "noop": {"intent": "advance_time", "domains": ["simulation"], "target_type": "episode"},
    "step": {"intent": "advance_time", "domains": ["simulation"], "target_type": "episode"},
    "set_cooling": {
        "intent": "mitigate",
        "domains": ["facility_thermal", "thermal"],
        "target_type": "cooling_unit",
    },
    "calibrate_sensor": {
        "intent": "mitigate",
        "domains": ["facility_thermal"],
        "target_type": "rack_or_server_sensor",
    },
    "migrate_workload": {
        "intent": "mitigate",
        "domains": ["application", "hardware", "network", "power", "scheduling", "thermal"],
        "target_type": "rack",
    },
    "throttle_workload": {
        "intent": "mitigate",
        "domains": ["application", "network", "power", "software_control_plane", "storage"],
        "target_type": "workload_or_tenant",
    },
    "update_autoscaler_policy": {
        "intent": "mitigate",
        "domains": ["software_control_plane"],
        "target_type": "autoscaler",
    },
    "repair_monitoring_pipeline": {
        "intent": "mitigate",
        "domains": ["observability"],
        "target_type": "monitoring_pipeline",
    },
    "update_placement_policy": {
        "intent": "mitigate",
        "domains": ["scheduling"],
        "target_type": "scheduler_policy",
    },
    "update_load_balancer_config": {
        "intent": "mitigate",
        "domains": ["application"],
        "target_type": "application_load_balancer",
    },
    "set_server_maintenance": {
        "intent": "mitigate",
        "domains": ["hardware"],
        "target_type": "server",
    },
    "clear_server_maintenance": {
        "intent": "recover",
        "domains": ["hardware"],
        "target_type": "server",
    },
}
HIDDEN_AGENT_KEYS = {
    "active_faults",
    "active_faults_after",
    "active_faults_before",
    "benchmark_action_coverage",
    "expected_diagnosis",
    "expected_mitigation",
    "fault_id",
    "fault_summary",
    "fault_target",
    "fault_type",
    "ground_truth",
    "incident_domains",
    "injected_faults",
    "initial_seed",
    "oracle",
    "remaining_duration_seconds",
    "reset_config",
    "scenario",
    "scenario_setup",
    "score_hints",
    "seed",
    "solution",
    "started_at_sim_time_seconds",
    "supported_faults",
    "desired_allocated_server_ids",
    "desired_allocation_weights",
    "workload_desired_allocated_server_ids",
    "workload_desired_allocation_weights",
}
HIDDEN_EVENT_TYPES = {"fault_injected", "fault_expired", "fault_removed"}
HIDDEN_ALERT_TYPES = {"active_fault"}
HIDDEN_AGENT_SUBSTRINGS = {
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
INTERNAL_AGENT_KEYS = {
    "base_latency_ms",
    "cpu_cost_per_request",
    "fault_latency_penalty_ms",
    "latency_fault_penalty_ms",
    "memory_cost_per_request_mb",
    "network_base_latency_ms",
    "network_capacity_mbps",
    "network_congestion_penalty_ms",
    "noise_enabled",
    "noise_stddev",
    "service_latency_ms",
    "storage_base_latency_ms",
    "storage_capacity_iops",
    "storage_congestion_penalty_ms",
    "baseline_capacity_kw",
}
AGENT_SUMMARY_HIDDEN_KEYS = {
    "active_faults",
    "workload_desired_allocated_server_ids",
    "workload_desired_allocation_weights",
}
AGENT_SIMULATION_CONFIG_KEYS = {"tick_seconds", "auto_advance"}
AGENT_WORKLOAD_CONFIG_KEYS = {
    "running",
    "tenant_id",
    "autoscaler",
    "request_rate_per_second",
    "configured_request_rate_per_second",
    "workload_class",
    "active_workload_class",
    "workload_profile_type",
    "current_profile_type",
    "workload_profile_parameters",
    "placement_strategy",
    "forbidden_rack_ids",
    "max_server_count",
    "placement_policy_status",
    "target_rack_id",
    "telemetry_freshness",
    "load_balancer",
    "trace_replay_enabled",
    "trace_replay_progress",
    "job_duration_seconds",
    "job_started_at_sim_time_seconds",
    "job_completed",
    "throttle_rate_per_second",
}


class AgentResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    config_override: dict[str, Any] = Field(default_factory=dict)
    workload: dict[str, Any] | None = None
    stabilization_ticks: int = 0
    log_limit: int = 20
    include_config: bool = True
    include_action_schema: bool = False
    host_visibility: str = HOST_VISIBILITY_RACK


class AgentActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    target: str | None = None
    fan_speed_percent: float | None = None
    supply_air_temperature_c: float | None = None
    source_rack_id: str | None = None
    target_rack_id: str | None = None
    tenant_id: str | None = None
    workload_fraction: float | None = None
    request_rate_per_second: float | None = None
    server_id: str | None = None
    rack_id: str | None = None
    calibration_offset_c: float | None = None
    mark_untrusted: bool | None = None
    min_capacity: int | None = None
    max_capacity: int | None = None
    target_utilization_percent: float | None = None
    cooldown_seconds: int | None = None
    placement_strategy: str | None = None
    forbidden_rack_ids: list[str] | None = None
    max_server_count: int | None = None
    backend_weights: dict[str, float] | None = None
    remove_backend_ids: list[str] | None = None
    add_backend_ids: list[str] | None = None
    routing_policy: str | None = None
    reset_to_equal_weights: bool | None = None
    ticks: int | None = None
    advance_ticks: int = 1
    log_limit: int = 20
    include_config: bool = True
    include_action_schema: bool = False
    host_visibility: str = HOST_VISIBILITY_RACK


def action_space(visibility: str = VISIBILITY_AGENT) -> dict[str, Any]:
    """Return the deterministic action contract for the selected visibility."""
    _validate_visibility(visibility)
    contract = {
        "agent_actions": sorted(AGENT_ACTIONS),
        "read_actions": {
            "observe": {
                "description": "Return the current observation without changing simulator state.",
                "parameters": {},
            }
        },
        "time_actions": {
            "noop": {
                "description": "Advance simulation time without applying a control.",
                "parameters": {"advance_ticks": {"type": "integer", "default": 1, "minimum": 0}},
            },
            "step": {
                "description": "Advance simulation time by a deterministic number of ticks.",
                "parameters": {"ticks": {"type": "integer", "default": 1, "minimum": 1}},
            },
        },
        "control_actions": {
            "calibrate_sensor": {
                "description": "Calibrate or mark a rack/server temperature sensor as untrusted.",
                "required": ["target"],
                "required_any": ["calibration_offset_c", "mark_untrusted"],
                "parameters": {
                    "target": {"type": "string"},
                    "calibration_offset_c": {"type": "number"},
                    "mark_untrusted": {"type": "boolean"},
                },
            },
            "set_cooling": {
                "description": "Adjust cooling-unit fan speed and/or supply air temperature.",
                "required_any": ["fan_speed_percent", "supply_air_temperature_c"],
                "parameters": {
                    "target": {"type": "string", "default": "cooling-unit-1"},
                    "fan_speed_percent": {"type": "number", "minimum": 0, "maximum": 100},
                    "supply_air_temperature_c": {"type": "number"},
                },
            },
            "migrate_workload": {
                "description": "Move workload allocation between racks.",
                "required": ["source_rack_id", "target_rack_id", "workload_fraction"],
                "parameters": {
                    "source_rack_id": {"type": "string"},
                    "target_rack_id": {"type": "string"},
                    "tenant_id": {"type": "string", "required_in_multi_tenant": True},
                    "workload_fraction": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            "throttle_workload": {
                "description": "Cap workload demand rate for the workload or selected tenant.",
                "required": ["request_rate_per_second"],
                "parameters": {
                    "tenant_id": {"type": "string", "required_in_multi_tenant": True},
                    "request_rate_per_second": {"type": "number", "minimum": 0},
                },
            },
            "update_autoscaler_policy": {
                "description": "Update autoscaler policy bounds and reaction timing.",
                "required_any": [
                    "min_capacity",
                    "max_capacity",
                    "target_utilization_percent",
                    "cooldown_seconds",
                ],
                "parameters": {
                    "min_capacity": {"type": "integer", "minimum": 0},
                    "max_capacity": {"type": "integer", "minimum": 1},
                    "target_utilization_percent": {"type": "number", "minimum": 1, "maximum": 100},
                    "cooldown_seconds": {"type": "integer", "minimum": 0},
                },
            },
            "repair_monitoring_pipeline": {
                "description": "Restore the monitoring and telemetry pipeline.",
                "parameters": {
                    "target": {"type": "string", "default": "monitoring-pipeline"},
                },
            },
            "update_placement_policy": {
                "description": "Update workload placement strategy, target rack, and rack/server constraints.",
                "required_any": ["placement_strategy", "target_rack_id", "forbidden_rack_ids", "max_server_count"],
                "parameters": {
                    "placement_strategy": {"type": "string", "enum": ["spread", "rack_hotspot", "random"]},
                    "target_rack_id": {"type": "string"},
                    "forbidden_rack_ids": {"type": "array", "items": {"type": "string"}},
                    "max_server_count": {"type": "integer", "minimum": 1},
                },
            },
            "update_load_balancer_config": {
                "description": "Update application load-balancer backends, routing policy, and backend weights.",
                "required_any": [
                    "backend_weights",
                    "remove_backend_ids",
                    "add_backend_ids",
                    "routing_policy",
                    "reset_to_equal_weights",
                ],
                "parameters": {
                    "backend_weights": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0}},
                    "remove_backend_ids": {"type": "array", "items": {"type": "string"}},
                    "add_backend_ids": {"type": "array", "items": {"type": "string"}},
                    "routing_policy": {"type": "string", "enum": ["round_robin", "weighted", "sticky"]},
                    "reset_to_equal_weights": {"type": "boolean"},
                },
            },
            "set_server_maintenance": {
                "description": (
                    "Mark one visible server, or every server in a visible rack, "
                    "as intentionally out of service for maintenance; this is not "
                    "a diagnostic oracle."
                ),
                "required_any": ["server_id", "rack_id"],
                "parameters": {
                    "server_id": {"type": "string"},
                    "rack_id": {"type": "string"},
                },
            },
            "clear_server_maintenance": {
                "description": (
                    "Clear operator-set maintenance flags for a server or rack if they exist; "
                    "this does not repair failed hardware or remove benchmark incidents."
                ),
                "required_any": ["server_id", "rack_id"],
                "parameters": {
                    "server_id": {"type": "string"},
                    "rack_id": {"type": "string"},
                },
            },
        },
        "observation": {
            "endpoint": "/agent/observation",
            "fields": [
                "episode_id",
                "sim_time_seconds",
                "sla_status",
                "summary",
                "alerts",
                "recent_events",
                "configuration",
                "action_schema_ref",
            ],
            "optional_fields": {
                "available_actions": "Included only when include_action_schema=true; otherwise use /agent/action-space."
            },
        },
        "invalid_action_behavior": {
            "http_status": 400,
            "state_mutation": "none",
            "control_recorded": False,
            "unknown_parameters": "rejected",
            "conflicting_parameters": "rejected",
        },
    }
    if visibility in {VISIBILITY_EVALUATOR, VISIBILITY_DEBUG}:
        contract["benchmark_action_coverage"] = {
            "scored_actions": sorted(SCORED_BENCHMARK_ACTIONS),
            "supported_non_scored_actions": dict(sorted(NON_SCORED_BENCHMARK_ACTIONS.items())),
            "all_write_actions_have_benchmark_status": SUPPORTED_ACTIONS
            <= SCORED_BENCHMARK_ACTIONS | set(NON_SCORED_BENCHMARK_ACTIONS),
        }
        contract["domain_contract"] = {
            "version": "2026-05-27",
            "incident_domains": INCIDENT_DOMAIN_CONTRACT,
            "action_domains": ACTION_DOMAIN_CONTRACT,
            "coverage_invariant": {
                "supported_faults": sorted(SUPPORTED_FAULTS),
                "all_supported_faults_have_incident_domain": set(INCIDENT_DOMAIN_CONTRACT) == SUPPORTED_FAULTS,
                "all_response_actions_are_agent_actions": all(
                    action in AGENT_ACTIONS
                    for item in INCIDENT_DOMAIN_CONTRACT.values()
                    for action in item["agent_response_actions"]
                ),
                "all_write_actions_have_benchmark_status": SUPPORTED_ACTIONS
                <= SCORED_BENCHMARK_ACTIONS | set(NON_SCORED_BENCHMARK_ACTIONS),
            },
        }
    if visibility == VISIBILITY_AGENT:
        return sanitize_agent_visible_response(contract)
    return contract


def agent_observation(
    simulator: DataCenterSimulator,
    log_limit: int = 20,
    include_config: bool = True,
    visibility: str = VISIBILITY_AGENT,
    include_action_schema: bool = False,
    host_visibility: str = HOST_VISIBILITY_RACK,
) -> dict[str, Any]:
    """Return a compact, visibility-aware observation for an agent or evaluator.

    The full action schema is intentionally opt-in for agent visibility. The
    canonical action contract is /agent/action-space, and compact observations
    carry only an action_schema_ref by default.
    """
    _validate_visibility(visibility)
    _validate_host_visibility(host_visibility)
    _validate_non_negative_int("log_limit", log_limit)
    with simulator.lock:
        observation = build_agent_visible_observation(
            simulator,
            log_limit=log_limit,
            include_config=include_config,
            visibility=visibility,
            include_action_schema=include_action_schema,
        )
        if visibility == VISIBILITY_AGENT:
            return sanitize_agent_visible_response(observation, host_visibility=host_visibility)
        return observation


def canonical_agent_telemetry(
    simulator: DataCenterSimulator,
    *,
    query_time_seconds: int | float | None = None,
    query_watermark_sequence: str | None = None,
    lookback_seconds: int | float = 300,
    channels: list[str] | tuple[str, ...] | set[str] | None = None,
    include_config: bool = True,
    log_limit: int | None = None,
) -> dict[str, Any]:
    """Return the shared StateBundle/baseline causal telemetry snapshot.

    Visibility is fixed at rack scope by the recorder.  The function has no
    evaluator/debug mode and cannot receive a label sidecar.
    """
    if isinstance(lookback_seconds, bool) or not isinstance(
        lookback_seconds, (int, float)
    ):
        raise SimulationError("lookback_seconds must be numeric")
    if lookback_seconds < 0:
        raise SimulationError("lookback_seconds must be non-negative")
    if not isinstance(include_config, bool):
        raise SimulationError("include_config must be a boolean")
    if log_limit is not None:
        _validate_non_negative_int("log_limit", log_limit)
    if channels is not None:
        unsupported = set(channels) - set(CANONICAL_TELEMETRY_CHANNELS)
        if unsupported:
            raise SimulationError(
                f"unsupported canonical telemetry channel(s): {sorted(unsupported)}"
            )
    return simulator.canonical_snapshot(
        query_time_seconds=query_time_seconds,
        query_watermark_sequence=query_watermark_sequence,
        lookback_seconds=lookback_seconds,
        channels=channels,
        include_config=include_config,
        log_limit=log_limit,
    )


def agent_metric_history(
    simulator: DataCenterSimulator,
    start_time_seconds: int | float,
    end_time_seconds: int | float,
    entity_id: str | None = None,
    entity_type: str | None = None,
    metric_name: str | None = None,
) -> dict[str, Any]:
    """Return legacy-shaped metric series from the canonical agent-visible cut.

    The simulator's raw metric-history store contains host-level samples and
    precedes the monitoring delivery model.  It must therefore never back an
    agent endpoint directly.  This compatibility view keeps the historical
    response envelope and common legacy metric names, while sourcing every
    point from the same rack-scoped, causal canonical snapshot used by the
    benchmark observation pipeline.
    """
    start_time = _coerce_metric_history_time(
        "start_time_seconds",
        start_time_seconds,
    )
    end_time = _coerce_metric_history_time(
        "end_time_seconds",
        end_time_seconds,
    )
    if start_time > end_time:
        raise SimulationError(
            "start_time_seconds must be <= end_time_seconds"
        )
    if start_time == end_time:
        return {"series": []}

    with simulator.lock:
        query_time = float(simulator.sim_time_seconds)
        snapshot = canonical_agent_telemetry(
            simulator,
            query_time_seconds=query_time,
            lookback_seconds=max(0.0, query_time - start_time),
            channels=("metric",),
            include_config=False,
        )
    series = _legacy_metric_series_from_canonical(
        snapshot,
        start_time_seconds=start_time,
        end_time_seconds=end_time,
        entity_id=entity_id,
        entity_type=entity_type,
        metric_name=metric_name,
    )
    return {"series": [item.model_dump(mode="json") for item in series]}


# Names retained by the pre-canonical metric-history API.  Canonical-only
# metrics receive a deterministic dotted-name fallback below; no simulator
# state or incident label is consulted to construct this projection.
_LEGACY_METRIC_NAME_BY_CANONICAL = {
    "facility.total_it_power": "total_it_power_kw",
    "facility.total_cooling_power": "total_cooling_power_kw",
    "facility.total_power": "facility_power_kw",
    "facility.pue": "pue",
    "rack.inlet_temperature": "rack_inlet_temperature_c",
    "rack.outlet_temperature": "rack_outlet_temperature_c",
    "rack.power": "rack_power_kw",
    "rack.cpu_utilization": "rack_average_cpu_utilization_percent",
    "rack.reported_inlet_temperature": "rack_reported_inlet_temperature_c",
    "rack.temperature_sensor_disagreement": (
        "rack_temperature_sensor_disagreement_c"
    ),
    "rack.thermal_throttle_factor": "rack_thermal_throttle_factor",
    "rack.network_packet_loss": "rack_network_packet_loss_percent",
    "rack.network_retransmit_rate": "rack_network_retransmit_rate",
    "rack.network_error_rate": "rack_network_error_rate",
    "cooling.capacity": "cooling_capacity_kw",
    "cooling.supply_air_temperature": "cooling_supply_air_temperature_c",
    "cooling.fan_speed": "cooling_fan_speed_percent",
    "control_plane.scheduler_api_latency": (
        "control_plane_scheduler_api_latency_ms"
    ),
    "control_plane.scheduler_pending_operations": (
        "control_plane_scheduler_pending_operations"
    ),
    "workload.current_demand": "workload_current_demand_per_second",
    "workload.configured_demand": (
        "workload_configured_request_rate_per_second"
    ),
    "workload.queue_length": "workload_queue_length",
    "workload.service_capacity": (
        "workload_service_capacity_requests_per_second"
    ),
    "workload.average_latency": "workload_average_latency_ms",
    "workload.p95_latency": "workload_p95_latency_ms",
    "workload.network_demand": "workload_network_demand_mbps",
    "workload.network_congestion": "workload_network_congestion_ratio",
    "workload.network_packet_loss": "workload_network_packet_loss_percent",
    "workload.network_retransmit_rate": "workload_network_retransmit_rate",
    "workload.network_error_rate": "workload_network_error_rate",
    "workload.storage_demand": "workload_storage_demand_iops",
    "workload.storage_utilization": "workload_storage_utilization_ratio",
    "workload.error_rate": "workload_error_rate_percent",
    "workload.dropped_requests": "workload_dropped_requests_per_second",
    "workload.gpu_utilization": "workload_gpu_utilization_percent",
    "control_plane.autoscaler_effective_server_limit": (
        "autoscaler_effective_server_limit"
    ),
    "control_plane.autoscaler_target_utilization": (
        "autoscaler_target_utilization_percent"
    ),
    "control_plane.autoscaler_max_capacity": "autoscaler_max_capacity",
    "control_plane.autoscaler_cooldown": "autoscaler_cooldown_seconds",
    "control_plane.placement_policy_violations": (
        "placement_policy_violating_racks"
    ),
    "workload.placement_imbalance": "workload_placement_imbalance_ratio",
    "observability.metrics_last_updated": (
        "metrics_last_updated_sim_time_seconds"
    ),
    "observability.logs_last_updated": (
        "logs_last_updated_sim_time_seconds"
    ),
    "observability.telemetry_lag": "telemetry_lag_seconds",
    "observability.metrics_missing_ratio": "metrics_missing_ratio",
    "observability.logs_missing_ratio": "logs_missing_ratio",
    "application.load_balancer_backend_skew": (
        "load_balancer_backend_skew_ratio"
    ),
    "application.load_balancer_unhealthy_routing": (
        "load_balancer_unhealthy_routing_fraction"
    ),
    "application.load_balancer_error_rate": (
        "load_balancer_error_rate_percent"
    ),
}

_LEGACY_METRIC_UNIT_BY_CANONICAL = {
    "kW": "kilowatt",
    "degC": "celsius",
    "requests/second": "requests_per_second",
    "megabits/second": "megabits_per_second",
    "operations/second": "iops",
    "events/second": "events_per_second",
}


def _legacy_metric_series_from_canonical(
    snapshot: dict[str, Any],
    *,
    start_time_seconds: float,
    end_time_seconds: float,
    entity_id: str | None,
    entity_type: str | None,
    metric_name: str | None,
) -> list[MetricSeries]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for observation in snapshot.get("observations", []):
        if observation.get("channel") != "metric":
            continue
        payload = observation.get("payload")
        if not isinstance(payload, dict):
            continue
        canonical_name = payload.get("metric_name")
        resource = payload.get("resource")
        if not isinstance(canonical_name, str) or not isinstance(resource, dict):
            continue
        resource_id = resource.get("entity_id")
        if not isinstance(resource_id, str):
            continue
        legacy_name = _LEGACY_METRIC_NAME_BY_CANONICAL.get(
            canonical_name,
            canonical_name.replace(".", "_"),
        )
        legacy_entity_type = _legacy_metric_entity_type(
            canonical_name,
            resource_id,
        )
        if entity_id is not None and resource_id != entity_id:
            continue
        if entity_type is not None and legacy_entity_type != entity_type:
            continue
        if metric_name is not None and metric_name not in {
            legacy_name,
            canonical_name,
        }:
            continue

        timestamps = payload.get("timestamps_seconds")
        values = payload.get("values")
        missingness = payload.get("missingness_mask")
        if not isinstance(timestamps, list) or not isinstance(values, list):
            continue
        if not isinstance(missingness, list):
            missingness = [False] * len(values)
        key = (resource_id, legacy_name)
        record = grouped.setdefault(
            key,
            {
                "entity_type": legacy_entity_type,
                "unit": _LEGACY_METRIC_UNIT_BY_CANONICAL.get(
                    str(payload.get("unit", "")),
                    str(payload.get("unit", "")),
                ),
                "points": {},
            },
        )
        for timestamp, value, missing in zip(
            timestamps,
            values,
            missingness,
            strict=True,
        ):
            if missing or value is None:
                continue
            numeric_time = float(timestamp)
            if (
                numeric_time < start_time_seconds
                or numeric_time >= end_time_seconds
            ):
                continue
            if not numeric_time.is_integer():
                # The legacy response model uses integer simulation ticks.
                # Canonical cadence is integer today, so silently excluding a
                # future fractional sample is safer than rounding its event time.
                continue
            record["points"][int(numeric_time)] = float(value)

    result: list[MetricSeries] = []
    for (resource_id, legacy_name), record in sorted(grouped.items()):
        points = [
            MetricPoint(sim_time_seconds=timestamp, value=value)
            for timestamp, value in sorted(record["points"].items())
        ]
        if not points:
            continue
        result.append(
            MetricSeries(
                metric_name=legacy_name,
                entity_id=resource_id,
                entity_type=record["entity_type"],
                unit=record["unit"],
                metric_kind="gauge",
                points=points,
            )
        )
    return result


def _legacy_metric_entity_type(
    canonical_name: str,
    resource_id: str,
) -> str:
    namespace = canonical_name.split(".", 1)[0]
    if namespace == "facility":
        return "datacenter"
    if namespace == "rack":
        return "rack"
    if namespace == "cooling":
        return "cooling_unit"
    if namespace == "observability":
        return "monitoring-pipeline"
    if namespace == "control_plane":
        return "control-plane"
    if namespace == "application":
        return "application"
    if namespace == "workload":
        return "workload"
    return resource_id


def _coerce_metric_history_time(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulationError(f"{name} must be numeric")
    numeric = float(value)
    if numeric < 0:
        raise SimulationError(f"{name} must be >= 0")
    return numeric


def build_agent_visible_observation(
    simulator: DataCenterSimulator,
    log_limit: int = 20,
    include_config: bool = True,
    visibility: str = VISIBILITY_AGENT,
    include_action_schema: bool = False,
) -> dict[str, Any]:
    """Build the stable observation envelope before visibility sanitization."""
    _validate_visibility(visibility)
    _validate_non_negative_int("log_limit", log_limit)
    observation = simulator.observation(log_limit=log_limit, include_config=include_config)
    observation["episode_id"] = simulator.episode_id
    if visibility == VISIBILITY_AGENT:
        observation = _apply_monitoring_pipeline_agent_effect(observation)
    if include_action_schema:
        observation["available_actions"] = action_space(visibility=visibility)
    else:
        observation["action_schema_ref"] = "/agent/action-space"
    return observation


def _apply_monitoring_pipeline_agent_effect(observation: dict[str, Any]) -> dict[str, Any]:
    summary = observation.get("summary")
    if not isinstance(summary, dict):
        return observation
    lag_seconds = summary.get("telemetry_lag_seconds", 0)
    if not isinstance(lag_seconds, (int, float)) or lag_seconds <= 0:
        return observation

    stale_observation = deepcopy(observation)
    # The monitoring pipeline health signal is supplied by an independent
    # health channel.  Operational values from the failed pipeline must not
    # remain current merely because the simulator can still calculate them.
    health_keys = {
        "sim_time_seconds",
        "metrics_last_updated_sim_time_seconds",
        "logs_last_updated_sim_time_seconds",
        "telemetry_lag_seconds",
        "metrics_missing_ratio",
        "logs_missing_ratio",
        "telemetry_pipeline_status",
    }
    stale_observation["summary"] = {
        key: deepcopy(value) if key in health_keys else None
        for key, value in summary.items()
    }
    # Preserve the stable observation contract without reporting the
    # simulator's current SLA calculation through the failed pipeline.
    stale_observation["sla_status"] = "unknown"

    alerts = stale_observation.get("alerts")
    if isinstance(alerts, list):
        stale_observation["alerts"] = [
            alert
            for alert in alerts
            if isinstance(alert, dict)
            and alert.get("alert_type") == "TelemetryStale"
        ]

    logs_last_updated = summary.get("logs_last_updated_sim_time_seconds", 0)
    recent_events = stale_observation.get("recent_events")
    if isinstance(recent_events, list):
        stale_events = [
            event
            for event in recent_events
            if isinstance(event, dict)
            and event.get("sim_time_seconds", 0) <= logs_last_updated
        ]
        logs_missing_ratio = float(summary.get("logs_missing_ratio", 0.0) or 0.0)
        if stale_events and logs_missing_ratio > 0.0:
            keep_count = max(1, int(round(len(stale_events) * max(0.0, 1.0 - logs_missing_ratio))))
            stale_events = stale_events[:keep_count]
        stale_observation["recent_events"] = stale_events
    return stale_observation


def reset_agent_environment(
    simulator: DataCenterSimulator,
    request: AgentResetRequest | None = None,
    visibility: str = VISIBILITY_AGENT,
) -> dict[str, Any]:
    _validate_visibility(visibility)
    request = request or AgentResetRequest()
    _validate_non_negative_int("stabilization_ticks", request.stabilization_ticks)
    _validate_non_negative_int("log_limit", request.log_limit)
    with simulator.lock:
        snapshot = simulator.snapshot_state()
        try:
            reset_summary = simulator.reset(seed=request.seed, config_override=request.config_override)
            workload = None
            if request.workload is not None:
                workload = simulator.start_workload(request.workload)
            if request.stabilization_ticks:
                simulator.step(request.stabilization_ticks)
            response = {
                "episode_id": simulator.episode_id,
                "status": "reset",
                "reset_summary": reset_summary,
                "workload": workload,
                "observation": agent_observation(
                    simulator,
                    log_limit=request.log_limit,
                    include_config=request.include_config,
                    visibility=visibility,
                    include_action_schema=request.include_action_schema,
                    host_visibility=request.host_visibility,
                ),
                "action_schema_ref": "/agent/action-space",
            }
            if request.include_action_schema or visibility != VISIBILITY_AGENT:
                response["available_actions"] = action_space(visibility=visibility)
            if visibility == VISIBILITY_AGENT:
                return sanitize_agent_visible_response(response, host_visibility=request.host_visibility)
            return response
        except ValueError as error:
            simulator.restore_state(snapshot)
            if isinstance(error, SimulationError):
                raise
            raise SimulationError(str(error)) from error


def apply_agent_action(
    simulator: DataCenterSimulator,
    request: AgentActionRequest,
    visibility: str = VISIBILITY_AGENT,
) -> dict[str, Any]:
    _validate_visibility(visibility)
    _validate_non_negative_int("log_limit", request.log_limit)
    action_type = request.action_type
    if action_type not in AGENT_ACTIONS:
        raise SimulationError(f"unsupported agent action_type: {action_type}")
    _validate_agent_action_request(request)

    with simulator.lock:
        before = simulator.state_summary()
        action_result: dict[str, Any]
        step_summary: dict[str, Any] | None = None

        if action_type == "observe":
            action_result = {"status": "observed"}
        elif action_type == "noop":
            action_result = {"status": "noop", "advance_ticks": request.advance_ticks}
            if request.advance_ticks:
                step_summary = simulator.step(request.advance_ticks)
        elif action_type == "step":
            ticks = request.ticks if request.ticks is not None else request.parameters.get("ticks", 1)
            step_summary = simulator.step(int(ticks))
            action_result = {"status": "stepped", "ticks": int(ticks)}
        else:
            control = _control_request_from_agent(request)
            action_result = simulator.apply_control(control)
            if request.advance_ticks:
                step_summary = simulator.step(request.advance_ticks)

        after = simulator.state_summary()
        response = {
            "episode_id": simulator.episode_id,
            "accepted": True,
            "action_type": action_type,
            "sim_time_seconds_before": before["sim_time_seconds"],
            "sim_time_seconds_after": after["sim_time_seconds"],
            "active_faults_before": before.get("active_faults", []),
            "active_faults_after": after.get("active_faults", []),
            "action_result": action_result,
            "step_summary": step_summary,
            "observation": agent_observation(
                simulator,
                log_limit=request.log_limit,
                include_config=request.include_config,
                visibility=visibility,
                include_action_schema=request.include_action_schema,
                host_visibility=request.host_visibility,
            ),
            "score_hints": _score_hints(before, after),
            "action_schema_ref": "/agent/action-space",
        }
        if request.include_action_schema or visibility != VISIBILITY_AGENT:
            response["available_actions"] = action_space(visibility=visibility)
        if visibility == VISIBILITY_AGENT:
            return sanitize_agent_visible_response(response, host_visibility=request.host_visibility)
        return response


def get_evaluator_state(
    simulator: DataCenterSimulator,
    log_limit: int = 50,
    include_config: bool = True,
) -> dict[str, Any]:
    """Return evaluator/debug state that is never used by the agent API."""
    _validate_non_negative_int("log_limit", log_limit)
    with simulator.lock:
        return {
            "episode_id": simulator.episode_id,
            "summary": simulator.state_summary(),
            "observation": agent_observation(
                simulator,
                log_limit=log_limit,
                include_config=include_config,
                visibility=VISIBILITY_EVALUATOR,
            ),
            "telemetry": simulator.telemetry(log_limit=log_limit, include_config=include_config),
            "active_faults": simulator.list_faults(),
            "available_actions": action_space(visibility=VISIBILITY_EVALUATOR),
        }


def sanitize_agent_visible_response(
    response: dict[str, Any],
    host_visibility: str = HOST_VISIBILITY_RACK,
) -> dict[str, Any]:
    """Remove evaluator/debug-only fields from an agent-facing response."""
    _validate_host_visibility(host_visibility)
    return _sanitize_value(response, host_visibility=host_visibility)


def _sanitize_value(value: Any, host_visibility: str = HOST_VISIBILITY_RACK) -> Any:
    if isinstance(value, dict):
        value = _coarsen_direct_host_allocations(value, host_visibility)
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if (
                key in HIDDEN_AGENT_KEYS
                or key in INTERNAL_AGENT_KEYS
                or _is_internal_agent_metric_key(key)
                or _is_agent_hidden_key_name(key)
            ):
                continue
            output_key = _agent_visible_key(key)
            if key == "summary" and isinstance(item, dict):
                sanitized[output_key] = _sanitize_summary(item, host_visibility=host_visibility)
                continue
            if key == "alerts" and isinstance(item, list):
                sanitized[output_key] = _sanitize_alerts(item, host_visibility=host_visibility)
                continue
            if key in {"recent_events", "logs"} and isinstance(item, list):
                sanitized[output_key] = _sanitize_events(item, host_visibility=host_visibility)
                continue
            if key == "configuration" and isinstance(item, dict):
                sanitized[output_key] = _agent_configuration(item, host_visibility=host_visibility)
                continue
            sanitized_item = _sanitize_value(item, host_visibility=host_visibility)
            if isinstance(sanitized_item, str) and _contains_hidden_agent_substring(sanitized_item):
                continue
            sanitized[output_key] = sanitized_item
        return sanitized
    if isinstance(value, list):
        sanitized_items = []
        for item in value:
            sanitized_item = _sanitize_value(item, host_visibility=host_visibility)
            if isinstance(sanitized_item, str) and _contains_hidden_agent_substring(sanitized_item):
                continue
            sanitized_items.append(sanitized_item)
        return sanitized_items
    return value


def _sanitize_summary(summary: dict[str, Any], host_visibility: str = HOST_VISIBILITY_RACK) -> dict[str, Any]:
    summary = _coarsen_direct_host_allocations(summary, host_visibility)
    return {
        _agent_visible_key(key): _sanitize_value(value, host_visibility=host_visibility)
        for key, value in summary.items()
        if key not in AGENT_SUMMARY_HIDDEN_KEYS
        and key not in HIDDEN_AGENT_KEYS
        and key not in INTERNAL_AGENT_KEYS
        and not _is_internal_agent_metric_key(key)
    }


def _coarsen_direct_host_allocations(
    value: dict[str, Any],
    host_visibility: str,
) -> dict[str, Any]:
    """Replace host allocation identities with policy-authorized aggregates."""
    if host_visibility == HOST_VISIBILITY_HOST:
        return value
    coarsened = dict(value)
    id_fields = {
        "allocated_server_ids": (
            "allocated_rack_ids",
            "allocated_host_counts_by_rack",
            "allocated_host_count",
        ),
        "workload_allocated_server_ids": (
            "workload_allocated_rack_ids",
            "workload_allocated_host_counts_by_rack",
            "workload_allocated_host_count",
        ),
    }
    for source_key, (rack_key, counts_key, aggregate_key) in id_fields.items():
        identifiers = coarsened.pop(source_key, None)
        if not isinstance(identifiers, list):
            continue
        rack_counts: dict[str, int] = {}
        for identifier in identifiers:
            if not isinstance(identifier, str):
                continue
            rack_id = _rack_id_from_server_id(identifier)
            rack_counts[rack_id] = rack_counts.get(rack_id, 0) + 1
        if host_visibility == HOST_VISIBILITY_AGGREGATE:
            coarsened[aggregate_key] = sum(rack_counts.values())
        else:
            coarsened[rack_key] = sorted(rack_counts)
            coarsened[counts_key] = dict(sorted(rack_counts.items()))

    weight_fields = {
        "allocation_weights": (
            "allocation_weights_by_rack",
            "allocation_weight_total",
        ),
        "workload_allocation_weights": (
            "workload_allocation_weights_by_rack",
            "workload_allocation_weight_total",
        ),
        "load_balancer_backend_weights": (
            "load_balancer_backend_weights_by_rack",
            "load_balancer_backend_weight_total",
        ),
        "backend_weights": (
            "backend_weights_by_rack",
            "backend_weight_total",
        ),
    }
    for source_key, (rack_key, aggregate_key) in weight_fields.items():
        weights = coarsened.pop(source_key, None)
        if not isinstance(weights, dict):
            continue
        rack_weights: dict[str, float] = {}
        for identifier, weight in weights.items():
            if (
                not isinstance(identifier, str)
                or isinstance(weight, bool)
                or not isinstance(weight, (int, float))
            ):
                continue
            rack_id = _rack_id_from_server_id(identifier)
            rack_weights[rack_id] = rack_weights.get(rack_id, 0.0) + float(weight)
        if host_visibility == HOST_VISIBILITY_AGGREGATE:
            coarsened[aggregate_key] = round(sum(rack_weights.values()), 6)
        else:
            coarsened[rack_key] = {
                rack_id: round(weight, 6)
                for rack_id, weight in sorted(rack_weights.items())
            }

    for source_key, identifiers in list(coarsened.items()):
        if "desired" in source_key.lower():
            continue
        if (
            (source_key == "server_ids" or source_key.endswith("_server_ids"))
            and isinstance(identifiers, list)
        ):
            coarsened.pop(source_key, None)
            rack_counts: dict[str, int] = {}
            for identifier in identifiers:
                if isinstance(identifier, str):
                    rack_id = _rack_id_from_server_id(identifier)
                    rack_counts[rack_id] = rack_counts.get(rack_id, 0) + 1
            prefix = source_key[: -len("server_ids")]
            if host_visibility == HOST_VISIBILITY_AGGREGATE:
                coarsened[f"{prefix}host_count"] = sum(rack_counts.values())
            else:
                coarsened[f"{prefix}rack_ids"] = sorted(rack_counts)
                coarsened[f"{prefix}host_counts_by_rack"] = dict(
                    sorted(rack_counts.items())
                )
        elif (
            (
                source_key == "backend_ids"
                or source_key.endswith("_backend_ids")
            )
            and isinstance(identifiers, list)
        ):
            coarsened.pop(source_key, None)
            rack_counts: dict[str, int] = {}
            for identifier in identifiers:
                if isinstance(identifier, str):
                    rack_id = _rack_id_from_server_id(identifier)
                    rack_counts[rack_id] = rack_counts.get(rack_id, 0) + 1
            prefix = source_key[: -len("backend_ids")]
            if host_visibility == HOST_VISIBILITY_AGGREGATE:
                coarsened[f"{prefix}backend_host_count"] = sum(
                    rack_counts.values()
                )
            else:
                coarsened[f"{prefix}backend_rack_ids"] = sorted(rack_counts)
                coarsened[f"{prefix}backend_host_counts_by_rack"] = dict(
                    sorted(rack_counts.items())
                )
        elif (
            (source_key == "server_id" or source_key.endswith("_server_id"))
            and isinstance(identifiers, str)
        ):
            coarsened.pop(source_key, None)
            prefix = source_key[: -len("server_id")]
            coarsened[f"{prefix}rack_id"] = (
                "datacenter"
                if host_visibility == HOST_VISIBILITY_AGGREGATE
                else _rack_id_from_server_id(identifiers)
            )
    return coarsened


def _rack_id_from_server_id(server_id: str) -> str:
    match = re.match(r"^server-(r\d+)-(row\d+)-rack(\d+)-", server_id)
    if match is None:
        return "rack"
    return f"rack-{match.group(1)}-{match.group(2)}-{match.group(3)}"


def _sanitize_alerts(alerts: list[Any], host_visibility: str = HOST_VISIBILITY_RACK) -> list[dict[str, Any]]:
    sanitized_alerts: list[dict[str, Any]] = []
    failed_host_counts_by_rack: dict[str, int] = {}
    host_alert_types = {
        "server_failed": ("HostHealthCheckFailed", "Host health check failures detected"),
        "host_health_flapping": ("HostHealthFlapping", "Host health checks are flapping"),
        "host_temperature_sensor_health": ("HostTemperatureSensorHealth", "Host temperature sensor readings are inconsistent"),
        "host_capacity_throttled": ("HostCapacityThrottled", "Host service capacity is reduced"),
    }
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        if alert.get("alert_type") in HIDDEN_ALERT_TYPES:
            continue
        sanitized = _sanitize_value(alert, host_visibility=host_visibility)
        if sanitized.get("alert_type") in host_alert_types:
            visible_type, message = host_alert_types[sanitized["alert_type"]]
            sanitized["alert_type"] = visible_type
            sanitized = _redact_host_alert(
                sanitized,
                host_visibility,
                failed_host_counts_by_rack,
                message,
            )
            if not sanitized:
                continue
        elif sanitized.get("alert_type") == "application_errors":
            sanitized["alert_type"] = "ApplicationErrorRateElevated"
            sanitized["message"] = "Application-level error rate is elevated"
        if _contains_hidden_agent_substring(sanitized):
            continue
        sanitized_alerts.append(sanitized)
    return sanitized_alerts


def _sanitize_events(events: list[Any], host_visibility: str = HOST_VISIBILITY_RACK) -> list[dict[str, Any]]:
    sanitized_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("event_type") in HIDDEN_EVENT_TYPES:
            continue
        sanitized = _sanitize_value(event, host_visibility=host_visibility)
        if _contains_hidden_agent_substring(sanitized):
            continue
        sanitized_events.append(sanitized)
    return sanitized_events


def _agent_configuration(
    configuration: dict[str, Any],
    host_visibility: str = HOST_VISIBILITY_RACK,
) -> dict[str, Any]:
    current = configuration.get("current") if isinstance(configuration.get("current"), dict) else configuration
    simulation = current.get("simulation", {}) if isinstance(current, dict) else {}
    workload = current.get("workload", {}) if isinstance(current, dict) else {}
    return _sanitize_value(
        {
            "episode_id": current.get("episode_id"),
            "simulation": {
                key: simulation[key]
                for key in AGENT_SIMULATION_CONFIG_KEYS
                if isinstance(simulation, dict) and key in simulation
            },
            "topology": deepcopy(current.get("topology", configuration.get("topology", {}))),
            "thresholds": deepcopy(current.get("thresholds", configuration.get("thresholds", {}))),
            "workload": {
                key: workload[key]
                for key in AGENT_WORKLOAD_CONFIG_KEYS
                if isinstance(workload, dict) and key in workload
            },
            "tenants": deepcopy(current.get("tenants", configuration.get("tenants", {}))),
            "cooling_units": deepcopy(current.get("cooling_units", configuration.get("cooling_units", []))),
            "active_controls": deepcopy(current.get("active_controls", configuration.get("active_controls", []))),
            "supported_actions": sorted(SUPPORTED_ACTIONS),
        },
        host_visibility=host_visibility,
    )


def _redact_host_alert(
    alert: dict[str, Any],
    host_visibility: str,
    failed_host_counts_by_rack: dict[str, int],
    message: str = "Host health check failures detected",
) -> dict[str, Any]:
    if host_visibility == HOST_VISIBILITY_HOST:
        target = alert.get("target", "host")
        alert["message"] = f"{message} for {target}"
        return alert

    details = alert.get("details") if isinstance(alert.get("details"), dict) else {}
    rack_id = details.get("rack_id")
    if not isinstance(rack_id, str) or not rack_id:
        rack_id = "rack"

    if host_visibility == HOST_VISIBILITY_AGGREGATE:
        alert["target"] = "datacenter"
        alert["message"] = message
        alert["details"] = {"affected_host_count": 1}
        return alert

    failed_host_counts_by_rack[rack_id] = failed_host_counts_by_rack.get(rack_id, 0) + 1
    alert["target"] = rack_id
    alert["message"] = f"{message} in {rack_id}"
    alert["details"] = {
        "rack_id": rack_id,
        "affected_host_count": failed_host_counts_by_rack[rack_id],
    }
    return alert


def _agent_visible_key(key: Any) -> Any:
    if not isinstance(key, str):
        return key
    return key.replace("application_error", "error").replace("power_overloaded", "power_limit_exceeded")


def _is_internal_agent_metric_key(key: Any) -> bool:
    return isinstance(key, str) and key.endswith("_penalty_ms")


def _is_agent_hidden_key_name(key: Any) -> bool:
    return isinstance(key, str) and "desired" in key


def _contains_hidden_agent_substring(value: Any) -> bool:
    text = _flatten_text(value)
    return any(substring in text for substring in HIDDEN_AGENT_SUBSTRINGS)


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            parts.append(str(key).lower())
            parts.append(_flatten_text(item))
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return ""


def _control_request_from_agent(request: AgentActionRequest) -> ControlRequest:
    payload = {
        "action_type": request.action_type,
        "target": request.target,
        "fan_speed_percent": request.fan_speed_percent,
        "supply_air_temperature_c": request.supply_air_temperature_c,
        "source_rack_id": request.source_rack_id,
        "target_rack_id": request.target_rack_id,
        "tenant_id": request.tenant_id,
        "workload_fraction": request.workload_fraction,
        "request_rate_per_second": request.request_rate_per_second,
        "server_id": request.server_id,
        "rack_id": request.rack_id,
        "calibration_offset_c": request.calibration_offset_c,
        "mark_untrusted": request.mark_untrusted,
        "min_capacity": request.min_capacity,
        "max_capacity": request.max_capacity,
        "target_utilization_percent": request.target_utilization_percent,
        "cooldown_seconds": request.cooldown_seconds,
        "placement_strategy": request.placement_strategy,
        "forbidden_rack_ids": request.forbidden_rack_ids,
        "max_server_count": request.max_server_count,
        "backend_weights": request.backend_weights,
        "remove_backend_ids": request.remove_backend_ids,
        "add_backend_ids": request.add_backend_ids,
        "routing_policy": request.routing_policy,
        "reset_to_equal_weights": request.reset_to_equal_weights,
    }
    payload.update(request.parameters)
    payload = {key: value for key, value in payload.items() if value is not None}
    try:
        return ControlRequest(**payload)
    except ValidationError as error:
        raise SimulationError(f"invalid control action: {error}") from error


def _validate_agent_action_request(request: AgentActionRequest) -> None:
    action_type = request.action_type
    allowed_parameters = ACTION_PARAMETER_KEYS[action_type]
    allowed_top_level = set(allowed_parameters)
    if action_type in ADVANCING_ACTIONS:
        allowed_top_level.add("advance_ticks")

    provided_top_level = request.model_fields_set & TOP_LEVEL_PARAMETER_KEYS
    rejected_top_level = provided_top_level - allowed_top_level
    if rejected_top_level:
        raise SimulationError(
            f"{action_type} does not accept top-level parameter(s): {_format_keys(rejected_top_level)}"
        )

    nested_parameters = set(request.parameters)
    rejected_nested = nested_parameters - allowed_parameters
    if rejected_nested:
        raise SimulationError(
            f"{action_type} does not accept nested parameter(s): {_format_keys(rejected_nested)}"
        )

    for key in sorted(nested_parameters & provided_top_level):
        top_level_value = getattr(request, key)
        nested_value = request.parameters[key]
        if top_level_value != nested_value:
            raise SimulationError(f"conflicting values for parameter: {key}")

    if action_type in ADVANCING_ACTIONS:
        _validate_non_negative_int("advance_ticks", request.advance_ticks)
    if action_type == "step":
        ticks = request.ticks if request.ticks is not None else request.parameters.get("ticks", 1)
        _validate_positive_int("ticks", ticks)


def _score_hints(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "sla_was_violated": before["sla_status"] == "violated",
        "sla_is_normal": after["sla_status"] == "normal",
        "thermal_critical_delta": after["thermal_critical"] - before["thermal_critical"],
        "power_overloaded_racks_delta": after["power_overloaded_racks"] - before["power_overloaded_racks"],
        "failed_servers_delta": after["failed_servers"] - before["failed_servers"],
        "workload_queue_delta": after["workload_queue_length"] - before["workload_queue_length"],
    }


def _validate_non_negative_int(name: str, value: Any) -> None:
    if type(value) is not int or value < 0:
        raise SimulationError(f"{name} must be a non-negative integer")


def _validate_positive_int(name: str, value: Any) -> None:
    if type(value) is not int or value < 1:
        raise SimulationError(f"{name} must be a positive integer")


def _validate_visibility(visibility: str) -> None:
    if visibility not in VISIBILITIES:
        raise SimulationError(f"unsupported visibility: {visibility}")


def _validate_host_visibility(host_visibility: str) -> None:
    if host_visibility not in HOST_VISIBILITIES:
        raise SimulationError(f"unsupported host_visibility: {host_visibility}")


def _format_keys(keys: set[str]) -> str:
    return ", ".join(sorted(keys))
