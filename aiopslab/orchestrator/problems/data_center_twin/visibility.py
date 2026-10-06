"""Agent-visibility controls for Data Center Twin benchmark payloads."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any


READ_ACTIONS = {"observe"}
TIME_ACTIONS = {"noop", "step"}
CONTROL_ACTIONS = {
    "calibrate_sensor",
    "clear_server_maintenance",
    "migrate_workload",
    "repair_monitoring_pipeline",
    "set_cooling",
    "set_server_maintenance",
    "throttle_workload",
    "update_autoscaler_policy",
    "update_load_balancer_config",
    "update_placement_policy",
}
AGENT_ACTIONS = READ_ACTIONS | TIME_ACTIONS | CONTROL_ACTIONS
HOST_VISIBILITY_HOST = "host"
HOST_VISIBILITY_RACK = "rack"
HOST_VISIBILITY_AGGREGATE = "aggregate"
HOST_VISIBILITIES = {HOST_VISIBILITY_HOST, HOST_VISIBILITY_RACK, HOST_VISIBILITY_AGGREGATE}
CANONICAL_TELEMETRY_SCHEMA_VERSION = "statebundle.canonical.v1"
STATEBUNDLE_OUTPUT_SCHEMA_VERSION = "statebundle.output.v1"
CANONICAL_CHANNELS = {"log", "metric", "alert", "trace", "config"}
CANONICAL_CHANNEL_ORDER = ("log", "metric", "alert", "trace", "config")
CANONICAL_REQUIRED_PAYLOAD_FIELDS = {
    "log": {
        "unit_type", "template_id", "template", "event_type", "count",
        "severity", "severity_histogram", "rarity", "burst_rate_per_minute",
        "variable_summaries", "time_features",
    },
    "metric": {
        "unit_type", "metric_name", "unit", "scale", "sample_period_seconds",
        "timestamps_seconds", "values", "missingness_mask", "statistics",
        "normalization_reference", "resource",
    },
    "alert": {
        "unit_type", "alert_fingerprint", "alert_type", "message", "target",
        "status", "severity", "threshold", "duration_seconds", "details",
    },
    "trace": {
        "unit_type", "operation", "source", "destination", "count",
        "status_counts", "retry_count", "latency_ms", "critical_path",
    },
    "config": {
        "unit_type", "path", "value", "value_type", "previous_value", "scope",
        "operation", "change_time_seconds",
    },
}
CANONICAL_SNAPSHOT_FIELDS = {
    "schema_version",
    "episode_id",
    "snapshot_id",
    "query_time_seconds",
    "query_watermark_sequence",
    "window",
    "channels",
    "channel_counts",
    "policy",
    "observations",
}
CANONICAL_WINDOW_FIELDS = {
    "start_time_seconds",
    "end_time_seconds",
    "start_inclusive",
    "end_inclusive",
}
CANONICAL_OBSERVATION_FIELDS = {
    "observation_id",
    "channel",
    "window",
    "payload",
    "metadata",
}
CANONICAL_METADATA_FIELDS = {
    "event_start_time_seconds",
    "event_end_time_seconds",
    "ingest_time_seconds",
    "available_at_time_seconds",
    "available_at_sequence",
    "entities",
    "primary_subsystem",
    "primary_subsystem_provenance",
    "correlation_ids",
    "source_references",
    "data_quality",
}
CANONICAL_QUALITY_FIELDS = {
    "parse_confidence",
    "missingness_fraction",
    "delay_seconds",
    "availability_mask",
    "validation_flags",
}
CANONICAL_POLICY_FIELDS = {
    "availability_predicate",
    "host_visibility",
    "timestamp_clock",
    "metric_cadence_seconds",
    "adapter_pipeline",
    "label_plane",
}
CANONICAL_ENTITY_ROLES = {"producer", "target", "source", "destination", "scope"}
CANONICAL_FORBIDDEN_KEYS = {
    "active_fault",
    "active_faults",
    "annotation",
    "annotation_basis",
    "annotations",
    "associated_fault_ids",
    "associated_fault_effect_ids",
    "background",
    "background_flag",
    "causal_role",
    "collection_phase",
    "effect_id",
    "fault_id",
    "fault_label_id",
    "fault_mechanism",
    "fault_target",
    "fault_type",
    "ground_truth",
    "incident_label",
    "injected_faults",
    "label",
    "label_provenance",
    "labels",
    "local_effect_id",
    "local_effect_or_anomaly_id",
    "observable_evidence",
    "oracle",
    "pair_label",
    "phase",
    "propagation_depth",
    "propagation_membership",
    "propagation_path",
    "propagation_path_ordered",
    "prototype_id",
    "root_cause",
    "root_cause_catalog",
    "scenario",
    "seed",
    "selector_gate",
    "split",
    "split_assignments",
    "subsystem_supervision",
    "symptom_family",
    "timing_regime",
    "training_labels",
}

FORBIDDEN_AGENT_KEYS = {
    "active_faults",
    "active_faults_after",
    "active_faults_before",
    "benchmark_action_coverage",
    "domain_contract",
    "duration_seconds",
    "expected",
    "expected_diagnosis",
    "expected_mitigation",
    "fault",
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
EXACT_SERVER_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])server-r\d+-row\d+-rack\d+-\d+"
    r"(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)
HIDDEN_EVENT_TYPES = {"fault_injected", "fault_expired", "fault_removed"}
HIDDEN_ALERT_TYPES = {"active_fault"}
INTERNAL_MODEL_KEYS = {
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

SUMMARY_ALLOWED_KEYS = {
    "control_plane_scheduler_api_latency_ms",
    "control_plane_scheduler_pending_operations",
    "average_cpu_utilization_percent",
    "average_rack_inlet_temperature_c",
    "average_rack_outlet_temperature_c",
    "average_reported_rack_inlet_temperature_c",
    "affected_rack_id",
    "autoscaler_cooldown_seconds",
    "autoscaler_current_capacity_units",
    "autoscaler_effective_server_limit",
    "autoscaler_enabled",
    "autoscaler_last_scale_action_time",
    "autoscaler_max_capacity",
    "autoscaler_min_capacity",
    "autoscaler_status",
    "autoscaler_target_utilization_percent",
    "facility_power_kw",
    "failed_servers",
    "host_health_flapping_count",
    "host_health_status_change_count",
    "max_rack_inlet_temperature_c",
    "max_rack_outlet_temperature_c",
    "max_power_budget_utilization_ratio",
    "max_reported_rack_inlet_temperature_c",
    "max_temperature_sensor_disagreement_c",
    "metrics_last_updated_sim_time_seconds",
    "metrics_missing_ratio",
    "max_network_packet_loss_percent",
    "min_thermal_throttle_factor",
    "network_error_rate",
    "network_packet_loss_racks",
    "network_retransmit_rate",
    "logs_last_updated_sim_time_seconds",
    "logs_missing_ratio",
    "load_balancer_backend_server_ids",
    "load_balancer_backend_skew_ratio",
    "load_balancer_backend_weights",
    "load_balancer_enabled",
    "load_balancer_error_rate_percent",
    "load_balancer_routing_policy",
    "load_balancer_unhealthy_backend_ids",
    "load_balancer_unhealthy_routing_fraction",
    "placement_policy_status",
    "placement_policy_target_rack_id",
    "placement_policy_violating_racks",
    "power_budget_violating_racks",
    "min_rack_inlet_temperature_c",
    "power_limit_exceeded_racks",
    "pue",
    "sim_time_seconds",
    "sla_status",
    "tenant_summaries",
    "thermal_critical",
    "temperature_sensor_unhealthy_count",
    "temperature_sensor_untrusted_count",
    "telemetry_lag_seconds",
    "telemetry_pipeline_status",
    "thermal_throttled_servers",
    "thermal_warnings",
    "total_cooling_power_kw",
    "total_it_power_kw",
    "workload_active_tenant_count",
    "workload_allocated_host_count",
    "workload_allocated_host_counts_by_rack",
    "workload_allocated_rack_ids",
    "workload_allocated_server_ids",
    "workload_allocation_weight_total",
    "workload_allocation_weights_by_rack",
    "workload_allocation_weights",
    "workload_average_latency_ms",
    "workload_class",
    "workload_class_resource_demand",
    "workload_configured_request_rate_per_second",
    "workload_cpu_demand",
    "workload_current_demand_per_second",
    "workload_dropped_requests_per_second",
    "workload_error_rate_percent",
    "workload_gpu_demand",
    "workload_gpu_utilization_percent",
    "workload_job_completed",
    "workload_job_duration_seconds",
    "workload_job_started_at_sim_time_seconds",
    "workload_maintenance_window_active",
    "workload_memory_demand",
    "workload_forbidden_rack_ids",
    "workload_max_server_count",
    "workload_network_congestion_ratio",
    "workload_network_affected_rack_id",
    "workload_network_demand",
    "workload_network_demand_mbps",
    "workload_network_error_rate",
    "workload_network_packet_loss_percent",
    "workload_network_retransmit_rate",
    "workload_p95_latency_ms",
    "workload_placement_imbalance_ratio",
    "workload_queue_length",
    "workload_queueing_latency_ms",
    "workload_request_rate_per_second",
    "workload_running",
    "workload_service_capacity_requests_per_second",
    "workload_service_time_latency_ms",
    "workload_storage_demand",
    "workload_storage_demand_iops",
    "workload_storage_utilization_ratio",
    "workload_trace_replay_enabled",
    "workload_trace_replay_progress",
    "workload_type",
    "workload_uncapped_demand_per_second",
}
OBSERVATION_ALLOWED_KEYS = {
    "action_schema_ref",
    "alerts",
    "available_actions",
    "configuration",
    "episode_id",
    "recent_events",
    "sim_time_seconds",
    "sla_status",
    "summary",
}
ACTION_RESPONSE_ALLOWED_KEYS = {
    "accepted",
    "action_result",
    "action_type",
    "action_schema_ref",
    "available_actions",
    "episode_id",
    "error",
    "http_status",
    "observation",
    "sim_time_seconds_after",
    "sim_time_seconds_before",
    "step_summary",
}
ACTION_SPACE_ALLOWED_KEYS = {
    "agent_actions",
    "control_actions",
    "invalid_action_behavior",
    "observation",
    "read_actions",
    "task_action_scope",
    "time_actions",
}
CONFIG_ALLOWED_KEYS = {
    "active_controls",
    "cooling_units",
    "episode_id",
    "simulation",
    "supported_actions",
    "tenants",
    "thresholds",
    "topology",
    "workload",
}
SIMULATION_CONFIG_ALLOWED_KEYS = {"auto_advance", "tick_seconds"}
WORKLOAD_CONFIG_ALLOWED_KEYS = {
    "active_workload_class",
    "autoscaler",
    "configured_request_rate_per_second",
    "current_profile_type",
    "job_completed",
    "job_duration_seconds",
    "job_started_at_sim_time_seconds",
    "forbidden_rack_ids",
    "max_server_count",
    "placement_strategy",
    "placement_policy_status",
    "request_rate_per_second",
    "running",
    "target_rack_id",
    "tenant_id",
    "telemetry_freshness",
    "load_balancer",
    "throttle_rate_per_second",
    "trace_replay_enabled",
    "trace_replay_progress",
    "workload_class",
    "workload_profile_parameters",
    "workload_profile_type",
}
EVENT_ALLOWED_KEYS = {"details", "episode_id", "event_type", "message", "sequence_id", "sim_time_seconds"}
EVENT_DETAIL_ALLOWED_KEYS = {
    "action_type",
    "advance_ticks",
    "add_backend_ids",
    "backend_weights",
    "calibration_offset_c",
    "cooldown_seconds",
    "control_id",
    "event_type",
    "fan_speed_percent",
    "message",
    "placement_strategy",
    "request_rate_per_second",
    "remove_backend_ids",
    "reset_to_equal_weights",
    "routing_policy",
    "mark_untrusted",
    "max_capacity",
    "max_server_count",
    "min_capacity",
    "forbidden_rack_ids",
    "placement_strategy",
    "server_id",
    "source_rack_id",
    "status",
    "supply_air_temperature_c",
    "target",
    "target_rack_id",
    "target_utilization_percent",
    "tenant_id",
    "ticks",
    "workload_class",
    "workload_fraction",
}
ACTION_RESULT_ALLOWED_KEYS = EVENT_DETAIL_ALLOWED_KEYS | {
    "accepted",
    "reason",
}


class AgentVisibilityLeakageError(AssertionError):
    """Raised when a supposedly agent-visible payload contains hidden state."""


def _canonical_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AgentVisibilityLeakageError(
            f"canonical {field} must be a finite number"
        )
    return float(value)


def _canonical_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise AgentVisibilityLeakageError(
            f"canonical {field} must be non-empty text"
        )
    return value


def _validate_canonical_json(value: Any, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AgentVisibilityLeakageError(
                f"canonical {field} contains a non-finite number"
            )
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise AgentVisibilityLeakageError(
                f"canonical {field} contains a non-string key"
            )
        for key, child in value.items():
            _validate_canonical_json(child, f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_canonical_json(child, f"{field}[{index}]")
        return
    raise AgentVisibilityLeakageError(
        f"canonical {field} contains a non-JSON value"
    )


def _canonical_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _validate_canonical_payload(
    channel: str,
    payload: dict[str, Any],
    *,
    window_start: float,
    window_end: float,
) -> None:
    required = CANONICAL_REQUIRED_PAYLOAD_FIELDS[channel]
    if set(payload) != required:
        raise AgentVisibilityLeakageError(
            f"canonical {channel} payload fields are invalid"
        )
    _canonical_text(payload.get("unit_type"), f"{channel}.unit_type")

    if channel == "log":
        for field in ("template_id", "template", "event_type", "severity"):
            _canonical_text(payload.get(field), f"log.{field}")
        count = payload.get("count")
        histogram = payload.get("severity_histogram")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or not isinstance(histogram, dict)
            or not histogram
            or any(
                severity not in {"debug", "info", "warning", "critical"}
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for severity, value in histogram.items()
            )
            or sum(histogram.values()) != count
        ):
            raise AgentVisibilityLeakageError(
                "canonical log count or severity histogram is invalid"
            )
        rarity = _canonical_number(payload.get("rarity"), "log.rarity")
        burst = _canonical_number(
            payload.get("burst_rate_per_minute"),
            "log.burst_rate_per_minute",
        )
        summaries = payload.get("variable_summaries")
        time_features = payload.get("time_features")
        if (
            not 0.0 <= rarity <= 1.0
            or burst < 0.0
            or not isinstance(summaries, dict)
            or not isinstance(time_features, dict)
            or set(time_features)
            != {"first_offset_seconds", "last_offset_seconds"}
        ):
            raise AgentVisibilityLeakageError(
                "canonical log summaries are invalid"
            )
        first_offset = _canonical_number(
            time_features.get("first_offset_seconds"),
            "log.time_features.first_offset_seconds",
        )
        last_offset = _canonical_number(
            time_features.get("last_offset_seconds"),
            "log.time_features.last_offset_seconds",
        )
        if first_offset < 0.0 or first_offset > last_offset:
            raise AgentVisibilityLeakageError(
                "canonical log time features are invalid"
            )
        _validate_canonical_json(summaries, "log.variable_summaries")
        return

    if channel == "metric":
        for field in ("metric_name", "unit", "scale"):
            _canonical_text(payload.get(field), f"metric.{field}")
        if (
            _canonical_number(
                payload.get("sample_period_seconds"),
                "metric.sample_period_seconds",
            )
            <= 0.0
        ):
            raise AgentVisibilityLeakageError(
                "canonical metric cadence must be positive"
            )
        timestamps = payload.get("timestamps_seconds")
        values = payload.get("values")
        missingness_mask = payload.get("missingness_mask")
        if (
            not isinstance(timestamps, list)
            or not isinstance(values, list)
            or not isinstance(missingness_mask, list)
            or not timestamps
            or not len(timestamps) == len(values) == len(missingness_mask)
            or not all(isinstance(missing, bool) for missing in missingness_mask)
        ):
            raise AgentVisibilityLeakageError(
                "canonical metric series arrays are invalid"
            )
        numeric_timestamps = [
            _canonical_number(value, "metric.timestamps_seconds")
            for value in timestamps
        ]
        if (
            any(
                later <= earlier
                for earlier, later in zip(
                    numeric_timestamps,
                    numeric_timestamps[1:],
                )
            )
            or numeric_timestamps[0] != window_start
            or numeric_timestamps[-1] != window_end
        ):
            raise AgentVisibilityLeakageError(
                "canonical metric timestamps are invalid"
            )
        for index, (value, missing) in enumerate(
            zip(values, missingness_mask, strict=True)
        ):
            if missing:
                if value is not None:
                    raise AgentVisibilityLeakageError(
                        "canonical metric mask disagrees with its value"
                    )
            else:
                _canonical_number(value, f"metric.values[{index}]")
        statistics = payload.get("statistics")
        if (
            not isinstance(statistics, dict)
            or set(statistics)
            != {
                "count",
                "min",
                "max",
                "mean",
                "median",
                "stddev",
                "p95",
                "last",
                "slope_per_second",
            }
        ):
            raise AgentVisibilityLeakageError(
                "canonical metric statistics are invalid"
            )
        for field, value in statistics.items():
            _canonical_number(value, f"metric.statistics.{field}")
        normalization = payload.get("normalization_reference")
        if (
            not isinstance(normalization, dict)
            or set(normalization)
            != {
                "method",
                "median",
                "iqr",
                "last",
                "sample_count",
                "reference_end_time_seconds",
            }
            or normalization.get("method")
            not in {"causal_pre_window_robust", "causal_window_fallback"}
        ):
            raise AgentVisibilityLeakageError(
                "canonical metric normalization reference is invalid"
            )
        _canonical_number(
            normalization.get("median"),
            "metric.normalization_reference.median",
        )
        if (
            _canonical_number(
                normalization.get("iqr"),
                "metric.normalization_reference.iqr",
            )
            < 0.0
        ):
            raise AgentVisibilityLeakageError(
                "canonical metric normalization IQR is invalid"
            )
        _canonical_number(
            normalization.get("last"),
            "metric.normalization_reference.last",
        )
        sample_count = normalization.get("sample_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 1
        ):
            raise AgentVisibilityLeakageError(
                "canonical metric normalization sample count is invalid"
            )
        reference_end = normalization.get("reference_end_time_seconds")
        if normalization["method"] == "causal_pre_window_robust":
            if (
                _canonical_number(
                    reference_end,
                    "metric.normalization_reference.reference_end_time_seconds",
                )
                >= window_start
            ):
                raise AgentVisibilityLeakageError(
                    "canonical metric normalization is not pre-window"
                )
        elif reference_end is not None:
            raise AgentVisibilityLeakageError(
                "canonical metric fallback has a reference end time"
            )
        resource = payload.get("resource")
        if (
            not isinstance(resource, dict)
            or set(resource) != {"entity_id"}
        ):
            raise AgentVisibilityLeakageError(
                "canonical metric resource is invalid"
            )
        _canonical_text(
            resource.get("entity_id"),
            "metric.resource.entity_id",
        )
        return

    if channel == "alert":
        for field in (
            "alert_fingerprint",
            "alert_type",
            "message",
            "target",
            "status",
            "severity",
        ):
            _canonical_text(payload.get(field), f"alert.{field}")
        threshold = payload.get("threshold")
        details = payload.get("details")
        duration = _canonical_number(
            payload.get("duration_seconds"),
            "alert.duration_seconds",
        )
        if (
            payload["status"] not in {"firing", "resolved"}
            or payload["severity"]
            not in {"debug", "info", "warning", "critical"}
            or (threshold is not None and not isinstance(threshold, dict))
            or not isinstance(details, dict)
            or duration < 0.0
            or not math.isclose(
                duration,
                max(0.0, window_end - window_start),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise AgentVisibilityLeakageError(
                "canonical alert payload is invalid"
            )
        _validate_canonical_json(threshold, "alert.threshold")
        _validate_canonical_json(details, "alert.details")
        return

    if channel == "trace":
        for field in ("operation", "source", "destination"):
            _canonical_text(payload.get(field), f"trace.{field}")
        count = _canonical_number(payload.get("count"), "trace.count")
        retry_count = _canonical_number(
            payload.get("retry_count"),
            "trace.retry_count",
        )
        status_counts = payload.get("status_counts")
        latency = payload.get("latency_ms")
        if (
            count < 0.0
            or retry_count < 0.0
            or not isinstance(status_counts, dict)
            or set(status_counts) != {"ok", "error"}
            or not isinstance(latency, dict)
            or set(latency) != {"mean", "p95", "min", "max"}
            or not isinstance(payload.get("critical_path"), bool)
        ):
            raise AgentVisibilityLeakageError(
                "canonical trace payload is invalid"
            )
        ok_count = _canonical_number(
            status_counts.get("ok"),
            "trace.status_counts.ok",
        )
        error_count = _canonical_number(
            status_counts.get("error"),
            "trace.status_counts.error",
        )
        if (
            ok_count < 0.0
            or error_count < 0.0
            or not math.isclose(
                ok_count + error_count,
                count,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
        ):
            raise AgentVisibilityLeakageError(
                "canonical trace status counts are invalid"
            )
        for field, value in latency.items():
            if value is not None and _canonical_number(
                value,
                f"trace.latency_ms.{field}",
            ) < 0.0:
                raise AgentVisibilityLeakageError(
                    "canonical trace latency is invalid"
                )
        return

    if channel == "config":
        for field in ("path", "value_type", "scope", "operation"):
            _canonical_text(payload.get(field), f"config.{field}")
        if (
            payload["operation"] not in {"state", "set", "update", "remove"}
            or not payload["path"].startswith(
                (
                    "simulation.",
                    "topology.",
                    "thresholds.",
                    "workload.",
                    "cooling_units.",
                    "controls.",
                )
            )
            or _canonical_number(
                payload.get("change_time_seconds"),
                "config.change_time_seconds",
            )
            != window_start
            or payload["value_type"]
            != _canonical_value_type(payload.get("value"))
        ):
            raise AgentVisibilityLeakageError(
                "canonical config payload is invalid"
            )
        _validate_canonical_json(payload.get("value"), "config.value")
        _validate_canonical_json(
            payload.get("previous_value"),
            "config.previous_value",
        )
        return

    raise AgentVisibilityLeakageError(
        f"unsupported canonical channel {channel!r}"
    )


def _strict_validate_canonical_snapshot(snapshot: dict[str, Any]) -> None:
    if set(snapshot) != CANONICAL_SNAPSHOT_FIELDS:
        raise AgentVisibilityLeakageError(
            "canonical snapshot fields are invalid"
        )
    _validate_canonical_json(snapshot, "snapshot")
    _canonical_text(snapshot.get("episode_id"), "episode_id")
    query_time = _canonical_number(
        snapshot.get("query_time_seconds"),
        "query_time_seconds",
    )
    if query_time < 0.0:
        raise AgentVisibilityLeakageError(
            "canonical query_time_seconds must be non-negative"
        )
    query_watermark = snapshot.get("query_watermark_sequence")
    if (
        not isinstance(query_watermark, str)
        or re.fullmatch(r"cut-[0-9a-f]{32}", query_watermark) is None
    ):
        raise AgentVisibilityLeakageError(
            "canonical query watermark must be an opaque cut token"
        )
    window = snapshot.get("window")
    if (
        not isinstance(window, dict)
        or set(window) != CANONICAL_WINDOW_FIELDS
    ):
        raise AgentVisibilityLeakageError(
            "canonical snapshot window is invalid"
        )
    window_start = _canonical_number(
        window.get("start_time_seconds"),
        "window.start_time_seconds",
    )
    window_end = _canonical_number(
        window.get("end_time_seconds"),
        "window.end_time_seconds",
    )
    if (
        window_start < 0.0
        or window_start > window_end
        or window_end != query_time
        or window.get("start_inclusive") is not True
        or window.get("end_inclusive") is not True
    ):
        raise AgentVisibilityLeakageError(
            "canonical snapshot window values are invalid"
        )
    policy = snapshot.get("policy")
    if (
        not isinstance(policy, dict)
        or set(policy) != CANONICAL_POLICY_FIELDS
        or policy.get("availability_predicate")
        != (
            "available_at_time_seconds <= query_time_seconds and "
            "available_at_sequence == query_watermark_sequence"
        )
        or policy.get("host_visibility") != HOST_VISIBILITY_RACK
        or policy.get("timestamp_clock") != "simulation_seconds"
        or policy.get("label_plane") != "separate_training_sidecar"
        or policy.get("adapter_pipeline")
        != ["decode", "consolidate", "normalize", "window", "validate"]
        or _canonical_number(
            policy.get("metric_cadence_seconds"),
            "policy.metric_cadence_seconds",
        )
        <= 0.0
    ):
        raise AgentVisibilityLeakageError(
            "canonical snapshot policy is invalid"
        )
    observations = snapshot.get("observations")
    channels = snapshot.get("channels")
    if (
        not isinstance(observations, list)
        or not isinstance(channels, list)
        or not all(isinstance(channel, str) for channel in channels)
        or channels
        != [
            channel
            for channel in CANONICAL_CHANNEL_ORDER
            if channel in channels
        ]
        or len(set(channels)) != len(channels)
    ):
        raise AgentVisibilityLeakageError(
            "canonical channels or observations are invalid"
        )
    channel_counts = snapshot.get("channel_counts")
    expected_counts = {
        channel: sum(
            isinstance(observation, dict)
            and observation.get("channel") == channel
            for observation in observations
        )
        for channel in channels
    }
    if (
        not isinstance(channel_counts, dict)
        or set(channel_counts) != set(channels)
        or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for count in channel_counts.values()
        )
        or channel_counts != expected_counts
    ):
        raise AgentVisibilityLeakageError(
            "canonical channel counts are invalid"
        )
    seen_ids: set[str] = set()
    for index, observation in enumerate(observations):
        if (
            not isinstance(observation, dict)
            or set(observation) != CANONICAL_OBSERVATION_FIELDS
        ):
            raise AgentVisibilityLeakageError(
                f"canonical observation {index} fields are invalid"
            )
        observation_id = _canonical_text(
            observation.get("observation_id"),
            f"observations[{index}].observation_id",
        )
        if observation_id in seen_ids:
            raise AgentVisibilityLeakageError(
                f"duplicate canonical observation_id {observation_id!r}"
            )
        seen_ids.add(observation_id)
        channel = observation.get("channel")
        if channel not in channels:
            raise AgentVisibilityLeakageError(
                f"canonical observation {index} has an invalid channel"
            )
        observation_window = observation.get("window")
        if (
            not isinstance(observation_window, dict)
            or set(observation_window) != CANONICAL_WINDOW_FIELDS
        ):
            raise AgentVisibilityLeakageError(
                f"canonical observation {index} window is invalid"
            )
        observation_start = _canonical_number(
            observation_window.get("start_time_seconds"),
            f"observations[{index}].window.start_time_seconds",
        )
        observation_end = _canonical_number(
            observation_window.get("end_time_seconds"),
            f"observations[{index}].window.end_time_seconds",
        )
        if (
            observation_start < 0.0
            or observation_start > observation_end
            or observation_end > query_time
            or observation_window.get("start_inclusive") is not True
            or observation_window.get("end_inclusive") is not True
        ):
            raise AgentVisibilityLeakageError(
                f"canonical observation {index} window values are invalid"
            )
        payload = observation.get("payload")
        if not isinstance(payload, dict):
            raise AgentVisibilityLeakageError(
                f"canonical observation {index} payload is invalid"
            )
        _validate_canonical_payload(
            channel,
            payload,
            window_start=observation_start,
            window_end=observation_end,
        )
        metadata = observation.get("metadata")
        if (
            not isinstance(metadata, dict)
            or set(metadata) != CANONICAL_METADATA_FIELDS
        ):
            raise AgentVisibilityLeakageError(
                f"canonical observation {index} metadata fields are invalid"
            )
        event_start = _canonical_number(
            metadata.get("event_start_time_seconds"),
            f"observations[{index}].metadata.event_start_time_seconds",
        )
        event_end = _canonical_number(
            metadata.get("event_end_time_seconds"),
            f"observations[{index}].metadata.event_end_time_seconds",
        )
        ingest_time = _canonical_number(
            metadata.get("ingest_time_seconds"),
            f"observations[{index}].metadata.ingest_time_seconds",
        )
        available_at = _canonical_number(
            metadata.get("available_at_time_seconds"),
            f"observations[{index}].metadata.available_at_time_seconds",
        )
        if (
            event_start != observation_start
            or event_end != observation_end
            or event_start > event_end
            or event_end > query_time
            or available_at > query_time
            or ingest_time != available_at
            or metadata.get("available_at_sequence") != query_watermark
        ):
            raise AgentVisibilityLeakageError(
                f"canonical observation {index} violates causal availability"
            )
        entities = metadata.get("entities")
        if not isinstance(entities, list):
            raise AgentVisibilityLeakageError(
                f"canonical observation {index} entities are invalid"
            )
        for entity in entities:
            confidence = entity.get("confidence") if isinstance(entity, dict) else None
            if (
                not isinstance(entity, dict)
                or set(entity)
                != {"entity_id", "role", "confidence", "provenance"}
                or not isinstance(entity.get("entity_id"), str)
                or not entity.get("entity_id")
                or entity.get("role") not in CANONICAL_ENTITY_ROLES
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
                or not isinstance(entity.get("provenance"), str)
                or not entity.get("provenance")
            ):
                raise AgentVisibilityLeakageError(
                    f"canonical observation {index} entity metadata is invalid"
                )
        correlations = metadata.get("correlation_ids")
        source_references = metadata.get("source_references")
        if (
            not isinstance(metadata.get("primary_subsystem"), str)
            or not metadata.get("primary_subsystem")
            or not isinstance(
                metadata.get("primary_subsystem_provenance"),
                str,
            )
            or not metadata.get("primary_subsystem_provenance")
            or not isinstance(correlations, dict)
            or not all(
                isinstance(key, str)
                and key
                and isinstance(value, str)
                and value
                for key, value in correlations.items()
            )
            or not isinstance(source_references, list)
            or not source_references
            or not all(
                isinstance(reference, str)
                and re.fullmatch(r"source-[0-9a-f]{20}", reference)
                is not None
                for reference in source_references
            )
        ):
            raise AgentVisibilityLeakageError(
                f"canonical observation {index} metadata is incomplete"
            )
        quality = metadata.get("data_quality")
        if (
            not isinstance(quality, dict)
            or set(quality) != CANONICAL_QUALITY_FIELDS
            or not isinstance(quality.get("availability_mask"), dict)
        ):
            raise AgentVisibilityLeakageError(
                f"canonical observation {index} data quality fields are invalid"
            )
        parse_confidence = _canonical_number(
            quality.get("parse_confidence"),
            f"observations[{index}].data_quality.parse_confidence",
        )
        missingness = _canonical_number(
            quality.get("missingness_fraction"),
            f"observations[{index}].data_quality.missingness_fraction",
        )
        delay = _canonical_number(
            quality.get("delay_seconds"),
            f"observations[{index}].data_quality.delay_seconds",
        )
        if (
            not 0.0 <= parse_confidence <= 1.0
            or not 0.0 <= missingness <= 1.0
            or delay < 0.0
            or not all(
                isinstance(key, str) and isinstance(value, bool)
                for key, value in quality["availability_mask"].items()
            )
            or not isinstance(quality.get("validation_flags"), list)
            or not all(
                isinstance(flag, str)
                for flag in quality["validation_flags"]
            )
            or not math.isclose(
                delay,
                max(0.0, available_at - event_end),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise AgentVisibilityLeakageError(
                f"canonical observation {index} data quality is invalid"
            )


def sanitize_agent_observation(
    observation: dict[str, Any],
    host_visibility: str = HOST_VISIBILITY_RACK,
) -> dict[str, Any]:
    """Return an allowlisted observation suitable for agent prompts/history."""
    _validate_host_visibility(host_visibility)
    if not isinstance(observation, dict):
        return observation
    if observation.get("schema_version") == CANONICAL_TELEMETRY_SCHEMA_VERSION:
        return sanitize_canonical_telemetry(observation)
    if observation.get("schema_version") == STATEBUNDLE_OUTPUT_SCHEMA_VERSION:
        return sanitize_statebundle_output(observation, host_visibility=host_visibility)
    sanitized: dict[str, Any] = {}
    for key in OBSERVATION_ALLOWED_KEYS:
        if key not in observation:
            continue
        value = observation[key]
        if key == "summary" and isinstance(value, dict):
            sanitized[key] = sanitize_agent_summary(
                value,
                host_visibility=host_visibility,
            )
        elif key == "alerts" and isinstance(value, list):
            sanitized[key] = sanitize_agent_alerts(value, host_visibility=host_visibility)
        elif key == "recent_events" and isinstance(value, list):
            sanitized[key] = [
                event
                for event in (
                    sanitize_agent_event(item, host_visibility=host_visibility)
                    for item in value
                )
                if event
            ]
        elif key == "configuration" and isinstance(value, dict):
            sanitized[key] = sanitize_agent_configuration(
                value,
                host_visibility=host_visibility,
            )
        elif key == "available_actions" and isinstance(value, dict):
            sanitized[key] = sanitize_agent_action_space(value)
        else:
            sanitized[key] = _sanitize_generic(
                value,
                host_visibility=host_visibility,
            )
    return _drop_hidden_strings(sanitized)


def sanitize_agent_action_response(
    response: dict[str, Any],
    allowed_actions: set[str] | frozenset[str] | None = None,
    host_visibility: str = HOST_VISIBILITY_RACK,
) -> dict[str, Any]:
    """Return an allowlisted action response suitable for the agent."""
    _validate_host_visibility(host_visibility)
    if not isinstance(response, dict):
        return response
    sanitized: dict[str, Any] = {}
    for key in ACTION_RESPONSE_ALLOWED_KEYS:
        if key not in response:
            continue
        value = response[key]
        if key == "observation" and isinstance(value, dict):
            observation = sanitize_agent_observation(value, host_visibility=host_visibility)
            available_actions = observation.get("available_actions")
            if isinstance(available_actions, dict):
                observation["available_actions"] = sanitize_agent_action_space(
                    available_actions,
                    allowed_actions=allowed_actions,
                )
            sanitized[key] = observation
        elif key == "available_actions" and isinstance(value, dict):
            sanitized[key] = sanitize_agent_action_space(value, allowed_actions=allowed_actions)
        elif key == "step_summary" and isinstance(value, dict):
            sanitized[key] = sanitize_agent_summary(
                value,
                host_visibility=host_visibility,
            )
        elif key == "action_result" and isinstance(value, dict):
            sanitized[key] = _sanitize_allowed_mapping(value, ACTION_RESULT_ALLOWED_KEYS)
        else:
            sanitized[key] = _sanitize_generic(value)
    return _drop_hidden_strings(sanitized)


def sanitize_agent_action_space(
    action_space: dict[str, Any],
    allowed_actions: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Return safe action names and schemas, without incident answer mappings."""
    if not isinstance(action_space, dict):
        return action_space
    allowed = set(allowed_actions) if allowed_actions is not None else None
    sanitized: dict[str, Any] = {}
    for key in ACTION_SPACE_ALLOWED_KEYS:
        if key not in action_space:
            continue
        value = action_space[key]
        if key == "agent_actions" and isinstance(value, list):
            actions = [action for action in value if isinstance(action, str) and action in AGENT_ACTIONS]
            sanitized[key] = [action for action in actions if allowed is None or action in allowed]
        elif key in {"read_actions", "time_actions", "control_actions"} and isinstance(value, dict):
            sanitized[key] = _sanitize_action_group(value, allowed)
        elif key == "task_action_scope" and isinstance(value, dict):
            sanitized[key] = _sanitize_task_action_scope(value)
        else:
            sanitized[key] = _sanitize_generic(value)
    return _drop_hidden_strings(sanitized)


def sanitize_agent_event(
    event: dict[str, Any],
    host_visibility: str = HOST_VISIBILITY_RACK,
) -> dict[str, Any]:
    """Return one safe operational event or an empty dict if it is hidden."""
    if not isinstance(event, dict):
        return {}
    if event.get("event_type") in HIDDEN_EVENT_TYPES:
        return {}
    sanitized = _sanitize_allowed_mapping(
        event,
        EVENT_ALLOWED_KEYS,
        host_visibility=host_visibility,
    )
    details = sanitized.get("details")
    if isinstance(details, dict):
        sanitized["details"] = _sanitize_allowed_mapping(
            details,
            EVENT_DETAIL_ALLOWED_KEYS,
            host_visibility=host_visibility,
        )
    if _contains_hidden_agent_substring(sanitized):
        return {}
    return sanitized


def sanitize_agent_payload(payload: Any) -> Any:
    """Best-effort sanitizer for rendered environment responses."""
    if isinstance(payload, dict):
        if payload.get("schema_version") == CANONICAL_TELEMETRY_SCHEMA_VERSION:
            return sanitize_canonical_telemetry(payload)
        if payload.get("schema_version") == STATEBUNDLE_OUTPUT_SCHEMA_VERSION:
            return sanitize_statebundle_output(payload)
        if "summary" in payload or "recent_events" in payload or "alerts" in payload:
            return sanitize_agent_observation(payload)
        if "observation" in payload or "action_result" in payload or "step_summary" in payload:
            return sanitize_agent_action_response(payload)
        if "agent_actions" in payload or "control_actions" in payload:
            return sanitize_agent_action_space(payload)
    return _sanitize_generic(payload)


def sanitize_canonical_telemetry(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Validate and preserve the versioned canonical inference projection."""
    if not isinstance(snapshot, dict):
        return snapshot
    if snapshot.get("schema_version") != CANONICAL_TELEMETRY_SCHEMA_VERSION:
        raise AgentVisibilityLeakageError("unsupported canonical telemetry schema")
    _strict_validate_canonical_snapshot(snapshot)
    query_time = snapshot.get("query_time_seconds")
    if isinstance(query_time, bool) or not isinstance(query_time, (int, float)):
        raise AgentVisibilityLeakageError("canonical query_time_seconds must be numeric")
    query_watermark = snapshot.get("query_watermark_sequence")
    if (
        not isinstance(query_watermark, str)
        or re.fullmatch(r"cut-[0-9a-f]{32}", query_watermark) is None
    ):
        raise AgentVisibilityLeakageError(
            "canonical query_watermark_sequence must be an opaque cut token"
        )
    observations = snapshot.get("observations")
    if not isinstance(observations, list):
        raise AgentVisibilityLeakageError("canonical observations must be a list")
    channels = snapshot.get("channels")
    if (
        not isinstance(channels, list)
        or not all(isinstance(channel, str) for channel in channels)
        or channels
        != [channel for channel in CANONICAL_CHANNEL_ORDER if channel in channels]
        or len(set(channels)) != len(channels)
    ):
        raise AgentVisibilityLeakageError("canonical channels are invalid")
    channel_counts = snapshot.get("channel_counts")
    expected_counts = {
        channel: sum(
            isinstance(observation, dict)
            and observation.get("channel") == channel
            for observation in observations
        )
        for channel in channels
    }
    if (
        not isinstance(channel_counts, dict)
        or set(channel_counts) != set(channels)
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in channel_counts.values()
        )
        or channel_counts != expected_counts
    ):
        raise AgentVisibilityLeakageError("canonical channel_counts are invalid")
    snapshot_without_id = deepcopy(snapshot)
    snapshot_without_id.pop("snapshot_id", None)
    serialized_snapshot = json.dumps(
        snapshot_without_id,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    expected_snapshot_id = (
        "snapshot-" + hashlib.sha256(serialized_snapshot).hexdigest()[:24]
    )
    if snapshot.get("snapshot_id") != expected_snapshot_id:
        raise AgentVisibilityLeakageError(
            "canonical snapshot_id does not match snapshot content"
        )

    leaks: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key)
                if key_text.lower() in CANONICAL_FORBIDDEN_KEYS:
                    leaks.append(f"forbidden canonical key {path}.{key_text}")
                walk(item, f"{path}.{key_text}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif (
            isinstance(value, str)
            and EXACT_SERVER_IDENTIFIER_PATTERN.search(value)
        ):
            leaks.append(f"exact host identifier in canonical value {path}")

    walk(snapshot, "$")
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            leaks.append(f"canonical observation {index} is not an object")
            continue
        channel = observation.get("channel")
        if channel not in channels:
            leaks.append(f"canonical observation {index} has an invalid channel")
        payload = observation.get("payload")
        if not isinstance(payload, dict):
            leaks.append(f"canonical observation {index} has no payload")
        elif channel in CANONICAL_REQUIRED_PAYLOAD_FIELDS:
            missing_fields = (
                CANONICAL_REQUIRED_PAYLOAD_FIELDS[channel] - set(payload)
            )
            if missing_fields:
                leaks.append(
                    f"canonical observation {index} payload misses "
                    f"{sorted(missing_fields)}"
                )
        window = observation.get("window")
        if not isinstance(window, dict):
            leaks.append(f"canonical observation {index} has no window")
            continue
        window_start = window.get("start_time_seconds")
        window_end = window.get("end_time_seconds")
        if (
            isinstance(window_start, bool)
            or not isinstance(window_start, (int, float))
            or isinstance(window_end, bool)
            or not isinstance(window_end, (int, float))
            or window_start > window_end
            or window_end > query_time
            or window.get("start_inclusive") is not True
            or window.get("end_inclusive") is not True
        ):
            leaks.append(f"canonical observation {index} has an invalid window")
        metadata = observation.get("metadata")
        if not isinstance(metadata, dict):
            leaks.append(f"canonical observation {index} has no metadata")
            continue
        event_start = metadata.get("event_start_time_seconds")
        event_end = metadata.get("event_end_time_seconds")
        available_at = metadata.get("available_at_time_seconds")
        if (
            isinstance(event_start, bool)
            or not isinstance(event_start, (int, float))
            or isinstance(event_end, bool)
            or not isinstance(event_end, (int, float))
            or event_start > event_end
            or event_end > query_time
            or event_start != window_start
            or event_end != window_end
            or isinstance(available_at, bool)
            or not isinstance(available_at, (int, float))
            or available_at > query_time
        ):
            leaks.append(f"canonical observation {index} violates causal availability")
        available_sequence = metadata.get("available_at_sequence")
        if available_sequence != query_watermark:
            leaks.append(f"canonical observation {index} violates sequence availability")
        entities = metadata.get("entities")
        if not isinstance(entities, list):
            leaks.append(f"canonical observation {index} has no entity list")
            entities = []
        for entity in entities:
            confidence = entity.get("confidence") if isinstance(entity, dict) else None
            if (
                not isinstance(entity, dict)
                or not isinstance(entity.get("entity_id"), str)
                or not entity.get("entity_id")
                or entity.get("role") not in CANONICAL_ENTITY_ROLES
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
                or not isinstance(entity.get("provenance"), str)
            ):
                leaks.append(f"canonical observation {index} has invalid entity metadata")
        quality = metadata.get("data_quality")
        if (
            not isinstance(quality, dict)
            or not isinstance(quality.get("availability_mask"), dict)
            or not all(
                isinstance(key, str) and isinstance(value, bool)
                for key, value in quality.get("availability_mask", {}).items()
            )
            or not isinstance(quality.get("validation_flags"), list)
            or not all(
                isinstance(flag, str)
                for flag in quality.get("validation_flags", [])
            )
        ):
            leaks.append(f"canonical observation {index} has invalid data quality")
    text = json.dumps(snapshot, sort_keys=True, default=str).lower()
    for substring in sorted(FORBIDDEN_AGENT_SUBSTRINGS):
        if substring in text:
            leaks.append(f"forbidden canonical substring {substring!r}")
    if leaks:
        raise AgentVisibilityLeakageError("\n".join(leaks))
    return deepcopy(snapshot)


def assert_no_agent_leakage(payload: Any) -> None:
    """Raise if hidden benchmark/evaluator state appears in an agent payload."""
    if (
        isinstance(payload, dict)
        and payload.get("schema_version") == CANONICAL_TELEMETRY_SCHEMA_VERSION
    ):
        sanitize_canonical_telemetry(payload)
        return
    if (
        isinstance(payload, dict)
        and payload.get("schema_version") == STATEBUNDLE_OUTPUT_SCHEMA_VERSION
    ):
        sanitize_statebundle_output(payload)
        return
    leaks: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and key in FORBIDDEN_AGENT_KEYS:
                    leaks.append(f"forbidden key {path}.{key}")
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(payload, "$")
    text = json.dumps(payload, sort_keys=True, default=str).lower()
    for substring in sorted(FORBIDDEN_AGENT_SUBSTRINGS):
        if substring in text:
            leaks.append(f"forbidden substring {substring!r}")
    if leaks:
        raise AgentVisibilityLeakageError("\n".join(leaks))


def sanitize_statebundle_output(
    output: dict[str, Any],
    host_visibility: str = HOST_VISIBILITY_RACK,
) -> dict[str, Any]:
    """Fail closed while preserving the trained selector's public output."""

    _validate_host_visibility(host_visibility)
    allowed_top_level = {
        "schema_version",
        "incident_id",
        "snapshot_id",
        "query_time_seconds",
        "token_budget",
        "used_tokens",
        "candidate_count",
        "evidence_groups",
        "request_status",
        "target_scope_ambiguity",
        "target_scope_candidates",
    }
    if set(output) - allowed_top_level:
        raise AgentVisibilityLeakageError(
            "StateBundle output contains unsupported fields: "
            f"{sorted(set(output) - allowed_top_level)}"
        )
    optional = {
        "snapshot_id",
        "request_status",
        # Optional for compatibility with saved statebundle.output.v1
        # artifacts produced before selector-estimated target roles existed.
        "target_scope_ambiguity",
        "target_scope_candidates",
    }
    required = allowed_top_level - optional
    if required - set(output):
        raise AgentVisibilityLeakageError(
            f"StateBundle output is missing fields: {sorted(required - set(output))}"
        )
    if output.get("schema_version") != STATEBUNDLE_OUTPUT_SCHEMA_VERSION:
        raise AgentVisibilityLeakageError("unsupported StateBundle output schema")
    if not isinstance(output.get("incident_id"), str) or not output["incident_id"]:
        raise AgentVisibilityLeakageError("StateBundle incident_id must be non-empty")
    query_time = output.get("query_time_seconds")
    if (
        isinstance(query_time, bool)
        or not isinstance(query_time, (int, float))
        or not math.isfinite(float(query_time))
        or float(query_time) < 0.0
    ):
        raise AgentVisibilityLeakageError("StateBundle query time is invalid")
    for name in ("token_budget", "used_tokens", "candidate_count"):
        value = output.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AgentVisibilityLeakageError(f"StateBundle {name} is invalid")
    if output["token_budget"] < 1 or output["used_tokens"] > output["token_budget"]:
        raise AgentVisibilityLeakageError("StateBundle token budget accounting is invalid")
    request_status = output.get("request_status")
    if request_status is not None:
        if not isinstance(request_status, dict):
            raise AgentVisibilityLeakageError(
                "StateBundle request_status must be an object"
            )
        allowed_request_status = {
            "status",
            "message",
            "matched_observation_count",
            "unique_matched_fact_count",
            "duplicate_matched_observation_count",
            "selected_match_count",
            "requested_metric_names",
            "requested_entity_ids",
            "requested_subsystem_ids",
            "requested_alert_names",
            "lookback_seconds",
        }
        if set(request_status) - allowed_request_status:
            raise AgentVisibilityLeakageError(
                "StateBundle request_status contains unsupported fields"
            )
        if request_status.get("status") not in {
            "satisfied",
            "narrow_request",
            "no_matching_canonical_observation",
        }:
            raise AgentVisibilityLeakageError(
                "StateBundle request_status has an invalid status"
            )
        if not isinstance(request_status.get("message"), str):
            raise AgentVisibilityLeakageError(
                "StateBundle request_status message is invalid"
            )
        for count_name in (
            "matched_observation_count",
            "unique_matched_fact_count",
            "duplicate_matched_observation_count",
            "selected_match_count",
        ):
            count = request_status.get(count_name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise AgentVisibilityLeakageError(
                    f"StateBundle request_status {count_name} is invalid"
                )
        if request_status["selected_match_count"] > request_status[
            "unique_matched_fact_count"
        ]:
            raise AgentVisibilityLeakageError(
                "StateBundle request_status counts are inconsistent"
            )
        if (
            request_status["unique_matched_fact_count"]
            + request_status["duplicate_matched_observation_count"]
            != request_status["matched_observation_count"]
        ):
            raise AgentVisibilityLeakageError(
                "StateBundle request_status duplicate counts are inconsistent"
            )
        for request_name in (
            "requested_metric_names",
            "requested_entity_ids",
            "requested_subsystem_ids",
            "requested_alert_names",
        ):
            values = request_status.get(request_name)
            if not isinstance(values, list) or not all(
                isinstance(item, str) for item in values
            ):
                raise AgentVisibilityLeakageError(
                    f"StateBundle request_status {request_name} is invalid"
                )
        lookback = request_status.get("lookback_seconds")
        if (
            isinstance(lookback, bool)
            or not isinstance(lookback, (int, float))
            or not math.isfinite(float(lookback))
            or float(lookback) < 0.0
        ):
            raise AgentVisibilityLeakageError(
                "StateBundle request_status lookback_seconds is invalid"
            )
    groups = output.get("evidence_groups")
    if not isinstance(groups, list):
        raise AgentVisibilityLeakageError("StateBundle evidence_groups must be a list")
    selected_observation_ids: set[str] = set()
    for index, group in enumerate(groups):
        if not isinstance(group, dict) or set(group) != {
            "anchor",
            "corroborating_observations",
        }:
            raise AgentVisibilityLeakageError(
                f"StateBundle evidence group {index} has an invalid shape"
            )
        if not isinstance(group.get("anchor"), dict) or not isinstance(
            group.get("corroborating_observations"), list
        ):
            raise AgentVisibilityLeakageError(
                f"StateBundle evidence group {index} is malformed"
            )
        for observation in [
            group["anchor"],
            *group["corroborating_observations"],
        ]:
            if not isinstance(observation, dict):
                raise AgentVisibilityLeakageError(
                    f"StateBundle evidence group {index} contains a non-object"
                )
            if set(observation) - {
                "observation_id",
                "channel",
                "window",
                "payload",
                "metadata",
            }:
                raise AgentVisibilityLeakageError(
                    f"StateBundle evidence group {index} contains internal fields"
                )
            if not {"observation_id", "channel", "window", "metadata"} <= set(
                observation
            ):
                raise AgentVisibilityLeakageError(
                    f"StateBundle evidence group {index} is missing observation fields"
                )
            if observation.get("channel") not in CANONICAL_CHANNELS:
                raise AgentVisibilityLeakageError(
                    f"StateBundle evidence group {index} has an invalid channel"
                )
            observation_id = observation.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                raise AgentVisibilityLeakageError(
                    f"StateBundle evidence group {index} has an invalid observation ID"
                )
            selected_observation_ids.add(observation_id)
    has_ambiguity = "target_scope_ambiguity" in output
    has_candidates = "target_scope_candidates" in output
    if has_ambiguity != has_candidates:
        raise AgentVisibilityLeakageError(
            "StateBundle target-scope fields must be present together"
        )
    if has_ambiguity:
        ambiguity = output.get("target_scope_ambiguity")
        candidates = output.get("target_scope_candidates")
        if not isinstance(ambiguity, bool):
            raise AgentVisibilityLeakageError(
                "StateBundle target_scope_ambiguity must be a boolean"
            )
        if not isinstance(candidates, list):
            raise AgentVisibilityLeakageError(
                "StateBundle target_scope_candidates must be a list"
            )
        allowed_roles = {
            "direct_target_candidate",
            "downstream_affected_scope",
            "contextual_peer",
        }
        seen_scopes: set[str] = set()
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or set(candidate) != {
                "scope",
                "estimated_role",
                "supporting_observation_ids",
            }:
                raise AgentVisibilityLeakageError(
                    f"StateBundle target scope {index} has an invalid shape"
                )
            scope = candidate.get("scope")
            role = candidate.get("estimated_role")
            support_ids = candidate.get("supporting_observation_ids")
            if not isinstance(scope, str) or not scope or scope in seen_scopes:
                raise AgentVisibilityLeakageError(
                    f"StateBundle target scope {index} is invalid or duplicated"
                )
            seen_scopes.add(scope)
            if role not in allowed_roles:
                raise AgentVisibilityLeakageError(
                    f"StateBundle target scope {index} has an invalid estimated role"
                )
            if (
                not isinstance(support_ids, list)
                or not support_ids
                or not all(
                    isinstance(item, str) and item for item in support_ids
                )
                or len(set(support_ids)) != len(support_ids)
                or not set(support_ids) <= selected_observation_ids
            ):
                raise AgentVisibilityLeakageError(
                    f"StateBundle target scope {index} has invalid supporting evidence"
                )
        if ambiguity and len(candidates) < 2:
            raise AgentVisibilityLeakageError(
                "StateBundle target ambiguity requires at least two scopes"
            )
    if _contains_hidden_agent_substring(output):
        raise AgentVisibilityLeakageError("StateBundle output contains hidden incident text")
    if EXACT_SERVER_IDENTIFIER_PATTERN.search(json.dumps(output, default=str)):
        raise AgentVisibilityLeakageError("StateBundle output contains a hidden server identifier")
    sanitized = _sanitize_generic(output, host_visibility=host_visibility)
    if sanitized.get("schema_version") != STATEBUNDLE_OUTPUT_SCHEMA_VERSION:
        raise AgentVisibilityLeakageError("StateBundle output was not safely preserved")
    return sanitized


def sanitize_agent_summary(
    summary: dict[str, Any],
    host_visibility: str = HOST_VISIBILITY_RACK,
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return summary
    summary = _coarsen_direct_host_allocations(summary, host_visibility)
    sanitized: dict[str, Any] = {}
    for key, value in summary.items():
        visible_key = _agent_visible_key(key)
        if not _is_allowed_field(visible_key, SUMMARY_ALLOWED_KEYS):
            continue
        sanitized[visible_key] = _sanitize_generic(
            value,
            host_visibility=host_visibility,
        )
    return _drop_hidden_strings(sanitized)


def sanitize_agent_alerts(
    alerts: list[Any],
    host_visibility: str = HOST_VISIBILITY_RACK,
) -> list[dict[str, Any]]:
    _validate_host_visibility(host_visibility)
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
        sanitized = _sanitize_generic(
            alert,
            host_visibility=host_visibility,
        )
        if isinstance(sanitized, dict) and sanitized.get("alert_type") in host_alert_types:
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
        if _contains_hidden_agent_substring(sanitized):
            continue
        sanitized_alerts.append(sanitized)
    return sanitized_alerts


def sanitize_agent_configuration(
    configuration: dict[str, Any],
    host_visibility: str = HOST_VISIBILITY_RACK,
) -> dict[str, Any]:
    if not isinstance(configuration, dict):
        return configuration
    current = configuration.get("current") if isinstance(configuration.get("current"), dict) else configuration
    sanitized: dict[str, Any] = {}
    for key in CONFIG_ALLOWED_KEYS:
        if key not in current:
            continue
        value = current[key]
        if key == "simulation" and isinstance(value, dict):
            sanitized[key] = _sanitize_allowed_mapping(value, SIMULATION_CONFIG_ALLOWED_KEYS)
        elif key == "workload" and isinstance(value, dict):
            sanitized[key] = _sanitize_allowed_mapping(value, WORKLOAD_CONFIG_ALLOWED_KEYS)
        else:
            sanitized[key] = _sanitize_generic(
                value,
                host_visibility=host_visibility,
            )
    return _drop_hidden_strings(sanitized)


def _sanitize_action_group(
    group: dict[str, Any],
    allowed_actions: set[str] | None,
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for action, schema in group.items():
        if not isinstance(action, str) or action not in AGENT_ACTIONS:
            continue
        if allowed_actions is not None and action not in allowed_actions:
            continue
        sanitized[action] = _sanitize_generic(schema)
    return sanitized


def _sanitize_task_action_scope(scope: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "agent_action_invocation",
        "disabled_control_actions",
        "executable_agent_actions",
        "executable_task_actions",
        "non_executable_global_agent_actions",
        "write_actions_enabled",
    }
    return _sanitize_allowed_mapping(scope, allowed_keys)


def _sanitize_allowed_mapping(
    mapping: dict[str, Any],
    allowed_keys: set[str],
    host_visibility: str = HOST_VISIBILITY_RACK,
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in mapping.items():
        visible_key = _agent_visible_key(key)
        if not _is_allowed_field(visible_key, allowed_keys):
            continue
        sanitized[visible_key] = _sanitize_generic(
            value,
            host_visibility=host_visibility,
        )
    return _drop_hidden_strings(sanitized)


def _coarsen_direct_host_allocations(
    value: dict[str, Any],
    host_visibility: str,
) -> dict[str, Any]:
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


def _sanitize_generic(
    value: Any,
    host_visibility: str = HOST_VISIBILITY_RACK,
) -> Any:
    if isinstance(value, dict):
        value = _coarsen_direct_host_allocations(value, host_visibility)
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            visible_key = _agent_visible_key(key)
            if not _is_allowed_generic_key(visible_key):
                continue
            sanitized_item = _sanitize_generic(
                item,
                host_visibility=host_visibility,
            )
            if _contains_hidden_agent_substring(sanitized_item):
                continue
            sanitized[visible_key] = sanitized_item
        return sanitized
    if isinstance(value, list):
        sanitized_items = []
        for item in value:
            sanitized_item = _sanitize_generic(
                item,
                host_visibility=host_visibility,
            )
            if _contains_hidden_agent_substring(sanitized_item):
                continue
            sanitized_items.append(sanitized_item)
        return sanitized_items
    return deepcopy(value)


def _drop_hidden_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: item
            for key, item in value.items()
            if not _contains_hidden_agent_substring(key) and not _contains_hidden_agent_substring(item)
        }
    if isinstance(value, list):
        return [item for item in value if not _contains_hidden_agent_substring(item)]
    return value


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


def _is_allowed_field(key: str, allowed_keys: set[str]) -> bool:
    return _is_allowed_generic_key(key) and key in allowed_keys


def _is_allowed_generic_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    if key in FORBIDDEN_AGENT_KEYS or key in INTERNAL_MODEL_KEYS:
        return False
    if "desired" in key:
        return False
    if key.endswith("_penalty_ms"):
        return False
    return not _contains_hidden_agent_substring(key)


def _agent_visible_key(key: Any) -> Any:
    if not isinstance(key, str):
        return key
    return key.replace("application_error", "error").replace("power_overloaded", "power_limit_exceeded")


def _contains_hidden_agent_substring(value: Any) -> bool:
    text = _flatten_text(value)
    return any(substring in text for substring in FORBIDDEN_AGENT_SUBSTRINGS)


def _validate_host_visibility(host_visibility: str) -> None:
    if host_visibility not in HOST_VISIBILITIES:
        raise ValueError(f"unsupported host_visibility: {host_visibility}")


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
